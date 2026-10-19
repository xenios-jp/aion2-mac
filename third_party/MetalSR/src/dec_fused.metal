// Fused decoder passes P8-P11 (sr.dec_level4, sr.dec_two_conv):
// bilinear 2x upsample + skip add staged into threadgroup memory, two chained 1x1 convolutions on the matrix units
// (the intermediate never leaves the threadgroup), epilogues in registers, coalesced 8-channel stores.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "sr_math.h"
using namespace metal;
using namespace mpp::tensor_ops;

struct DecParams {
    uint w, h;       // level size (fine)
    uint wc, hc;     // coarse size
    uint npix;
    uint variant;    // 0 = N, 1 = W
};

struct H8 { half4 lo, hi; };

// Epilogue of a 1x1 convolution: one rounding of acc + bias (decoders), or the head's two roundings (sr.numerics, Epilogues),
// where the accumulator is rounded to fp16 first and the bias added in fp16. relu_last: the second layer has a rectifier.
struct EpiOneRound { static constexpr constant bool relu_last = true; static inline half run(float acc, half b) { return (half)(acc + (float)b); } };
struct EpiTwoRound {
    static constexpr constant bool relu_last = false;
    static inline half run(float acc, half b) { return sr_two_round_add(acc, b); }
};

static inline half4 lerp4(uint variant, half4 n, half4 f) {
    if (variant == 0) return fma(half4(0.75h), n - f, f);
    return fma(half4(0.75h), n, f * half4(0.25h));
}
static inline H8 lerp8(uint variant, H8 n, H8 f) { H8 r; r.lo = lerp4(variant, n.lo, f.lo); r.hi = lerp4(variant, n.hi, f.hi); return r; }

template <bool M8>
static inline H8 load8(device const void *p, uint idx) {
    H8 r;
    if (M8) {
        ulong raw = *(device const ulong *)((device const uchar *)p + idx);
        for (int c = 0; c < 4; c++) {
            r.lo[c] = as_type<half>((ushort)(((raw >> (8 * c)) & 0xFF) << 7));
            r.hi[c] = as_type<half>((ushort)(((raw >> (8 * (c + 4))) & 0xFF) << 7));
        }
    } else {
        r.lo = *(device const half4 *)((device const half *)p + idx);
        r.hi = *(device const half4 *)((device const half *)p + idx + 4);
    }
    return r;
}
static inline uchar enc_m8(half x) {
    ushort b = as_type<ushort>(x) & 0x7FFF;
    if (b > 0x7C00) return 0xFF;
    if (b < 0x400) return (uchar)(b >> 7);
    return (uchar)((b + 0x40) >> 7);
}

template <int TP, int C0, int C1, int C2, int C2P, int SGM, int SGN, bool CM8, bool SM8, bool OM8, class EPI>
static inline void dec_impl(device const void *coarse, device const void *skip, device const half *wa, device const half *ba,
                            device const half *wb, device const half *bb, device void *out, constant DecParams &p,
                            uint tgx, uint tid, uint sg, threadgroup half *smem) {
    constexpr int NSG = SGM * SGN, NT = NSG * 32;
    constexpr int XS = C0 + 8, HS = C1 + 8, OS = C2P + 8;
    constexpr int XO = (TP * XS > TP * OS) ? TP * XS : TP * OS;       // x tile, reused as the output tile
    threadgroup half *xt = smem;
    threadgroup half *ht = smem + XO;
    const uint base = tgx * TP;

    // 1. stage x = rn16(up2(coarse) + skip) for TP pixels, 8 channels per item
    for (int it = (int)tid; it < TP * (C0 / 8); it += NT) {
        int pl = it / (C0 / 8), g = it % (C0 / 8);
        uint gp = base + (uint)pl;
        H8 xv; xv.lo = half4(0); xv.hi = half4(0);
        if (gp < p.npix) {
            int X = (int)(gp % p.w), Y = (int)(gp / p.w);
            int nx = X >> 1, ny = Y >> 1;
            int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
            fx = clamp(fx, 0, (int)p.wc - 1); fy = clamp(fy, 0, (int)p.hc - 1);
            uint ch = (uint)(8 * g);
            H8 a = load8<CM8>(coarse, ((uint)ny * p.wc + (uint)nx) * (CM8 ? C0 : C0) + ch);
            H8 b = load8<CM8>(coarse, ((uint)ny * p.wc + (uint)fx) * C0 + ch);
            H8 d = load8<CM8>(coarse, ((uint)fy * p.wc + (uint)nx) * C0 + ch);
            H8 e = load8<CM8>(coarse, ((uint)fy * p.wc + (uint)fx) * C0 + ch);
            H8 hn = lerp8(p.variant, a, b), hf = lerp8(p.variant, d, e);
            H8 up = lerp8(p.variant, hn, hf);
            H8 s = load8<SM8>(skip, gp * C0 + ch);
            xv.lo = up.lo + s.lo; xv.hi = up.hi + s.hi;
        }
        *(threadgroup half4 *)(xt + pl * XS + 8 * g) = xv.lo;
        *(threadgroup half4 *)(xt + pl * XS + 8 * g + 4) = xv.hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int mt = (int)sg % SGM, nt = (int)sg / SGM;
    using AT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using BT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    // 2. first convolution
    {
        constexpr int N1S = C1 / SGN;
        constexpr auto d1 = matmul2d_descriptor(16, N1S, C0, false, true, false);
        matmul2d<d1, execution_simdgroup> op;
        AT A(xt + mt * 16 * XS, dextents<int32_t, 2>(C0, 16), array<int32_t, 2>{1, XS});
        BT B((device half *)wa + (size_t)nt * N1S * C0, dextents<int32_t, 2>(C0, N1S));
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        op.run(A, B, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            auto idx = cT.get_multidimensional_index(i);
            int n = nt * N1S + (int)idx[0], m = mt * 16 + (int)idx[1];
            half v = EPI::run(cT[i], ba[n]);
            ht[m * HS + n] = v > (half)0 ? v : (half)0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. second convolution -> output tile (aliases the x tile)
    {
        constexpr int N2S = C2P / SGN;
        constexpr auto d2 = matmul2d_descriptor(16, N2S, C1, false, true, false);
        matmul2d<d2, execution_simdgroup> op;
        AT A(ht + mt * 16 * HS, dextents<int32_t, 2>(C1, 16), array<int32_t, 2>{1, HS});
        BT B((device half *)wb + (size_t)nt * N2S * C1, dextents<int32_t, 2>(C1, N2S));
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        op.run(A, B, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            auto idx = cT.get_multidimensional_index(i);
            int n = nt * N2S + (int)idx[0], m = mt * 16 + (int)idx[1];
            if (n >= C2) continue;
            half v = EPI::run(cT[i], bb[n]);
            if (EPI::relu_last) v = v > (half)0 ? v : (half)0;
            xt[m * OS + n] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4. coalesced stores, 8 channels per item
    for (int it = (int)tid; it < TP * (C2 / 8); it += NT) {
        int pl = it / (C2 / 8), g = it % (C2 / 8);
        uint gp = base + (uint)pl;
        if (gp >= p.npix) continue;
        half4 lo = *(threadgroup half4 *)(xt + pl * OS + 8 * g), hi = *(threadgroup half4 *)(xt + pl * OS + 8 * g + 4);
        if (OM8) {
            ulong raw = 0;
            for (int c = 0; c < 4; c++) raw |= ((ulong)enc_m8(lo[c]) << (8 * c)) | ((ulong)enc_m8(hi[c]) << (8 * (c + 4)));
            *(device ulong *)((device uchar *)out + gp * C2 + 8 * g) = raw;
        } else {
            *(device half4 *)((device half *)out + gp * C2 + 8 * g) = lo;
            *(device half4 *)((device half *)out + gp * C2 + 8 * g + 4) = hi;
        }
    }
}

#define DEFINE_DEC(NAME, TP, C0, C1, C2, C2P, SGM, SGN, CM8, SM8, OM8, EPI)                                                    \
    kernel void NAME(device const void *coarse [[buffer(0)]], device const void *skip [[buffer(1)]],                                 \
                     device const half *wa [[buffer(2)]], device const half *ba [[buffer(3)]],                                        \
                     device const half *wb [[buffer(4)]], device const half *bb [[buffer(5)]],                                        \
                     device void *out [[buffer(6)]], constant DecParams &p [[buffer(7)]],                                             \
                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                             \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                                    \
        threadgroup half smem[((TP * (C0 + 8)) > (TP * (C2P + 8)) ? (TP * (C0 + 8)) : (TP * (C2P + 8))) + TP * (C1 + 8)];             \
        dec_impl<TP, C0, C1, C2, C2P, SGM, SGN, CM8, SM8, OM8, EPI>(coarse, skip, wa, ba, wb, bb, out, p, tg.x, tid, sg, smem);    \
    }

DEFINE_DEC(dec_p8, 16, 256, 256, 128, 128, 1, 4, false, false, false, EpiOneRound)
DEFINE_DEC(dec_p9, 16, 128, 128, 128, 128, 1, 4, false, false, false, EpiOneRound)
DEFINE_DEC(dec_p10, 16, 128, 128, 64, 64, 1, 4, false, false, true, EpiOneRound)
DEFINE_DEC(dec_p11, 32, 64, 64, 32, 32, 2, 2, true, true, true, EpiOneRound)
// head network of P12 (sr.head_filter): the tile of 64 L0 pixels, 40 outputs padded to 48 columns, no rectifier on the last layer
DEFINE_DEC(dec_head, 64, 32, 32, 40, 48, 4, 1, true, true, false, EpiTwoRound)
