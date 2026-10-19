// 3x3 convolution + ReLU + 2x2 max-pool on the matrix units (sr.enc_conv3x3_pool, sr.metal_techniques).
// Threadgroup = TH simdgroups, one pixel row of TW pixels each, TN output channels. The zero-padded halo of
// (TH + 2) x (TW + 2) x KC activations is staged once per K chunk; each tap is one matmul whose left operand is a
// strided view of the halo shifted by the tap, weights are read in place from device memory as a transposed view.
// Epilogue in registers (bias, rn16, ReLU) -> threadgroup tile -> coalesced 8-channel stores of the skip and the pooled maxima.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct TileParams {
    uint w, h;       // level size
    uint cin, cout;
};

struct h8 { half4 lo, hi; };
static inline h8 ld8(device const half *p) { h8 r; r.lo = *(device const half4 *)p; r.hi = *(device const half4 *)(p + 4); return r; }
static inline h8 ld8t(threadgroup const half *p) { h8 r; r.lo = *(threadgroup const half4 *)p; r.hi = *(threadgroup const half4 *)(p + 4); return r; }
static inline void st8(device half *p, h8 v) { *(device half4 *)p = v.lo; *(device half4 *)(p + 4) = v.hi; }
static inline void st8t(threadgroup half *p, h8 v) { *(threadgroup half4 *)p = v.lo; *(threadgroup half4 *)(p + 4) = v.hi; }
static inline h8 hmax8(h8 a, h8 b) { h8 r; r.lo = max(a.lo, b.lo); r.hi = max(a.hi, b.hi); return r; }
static inline half m8_dec_t(uchar c) { return as_type<half>((ushort)((ushort)c << 7)); }
static inline uchar m8_enc_t(half x) {
    ushort b = as_type<ushort>(x) & 0x7FFF;
    if (b > 0x7C00) return 0xFF;
    if (b < 0x400) return (uchar)(b >> 7);
    return (uchar)((b + 0x40) >> 7);
}

// M8IN: input tensor holds m8 codes. M8OUT: skip is stored as m8 (pooled stays fp16).
// SPLIT: the epilogue tile is written and pooled in SPLIT phases of TH / SPLIT rows each (less threadgroup memory, the same values)
template <int TW, int TH, int TN, int KC, bool M8IN, bool M8OUT, int SPLIT>
static inline void conv_tile_impl(device const void *x, device const half *wt, device const half *bias, device void *skip, device half *pooled,
                                  constant TileParams &p, uint3 tg, uint tid, uint sg, threadgroup half *smem) {
    constexpr int NT = TH * 32;
    constexpr int HW = TW + 2;
    constexpr auto desc = matmul2d_descriptor(TW, TN, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    using AT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using BT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    const int x0 = (int)tg.x * TW, y0 = (int)tg.y * TH, n0 = (int)tg.z * TN;
    for (int k0 = 0; k0 < (int)p.cin; k0 += KC) {
        // stage the halo: items of 8 channels
        constexpr int G = KC / 8;
        for (int it = (int)tid; it < (TH + 2) * HW * G; it += NT) {
            int g = it % G, col = (it / G) % HW, row = it / (G * HW);
            int gx = x0 + col - 1, gy = y0 + row - 1;
            h8 v; v.lo = half4(0); v.hi = half4(0);
            if (gx >= 0 && gx < (int)p.w && gy >= 0 && gy < (int)p.h) {
                uint idx = ((uint)gy * p.w + (uint)gx) * p.cin + (uint)(k0 + 8 * g);
                if (M8IN) {
                    ulong raw = *(device const ulong *)((device const uchar *)x + idx);
                    for (int c = 0; c < 4; c++) {
                        v.lo[c] = as_type<half>((ushort)(((raw >> (8 * c)) & 0xFF) << 7));
                        v.hi[c] = as_type<half>((ushort)(((raw >> (8 * (c + 4))) & 0xFF) << 7));
                    }
                } else v = ld8((device const half *)x + idx);
            }
            st8t(smem + ((row * HW + col) * KC + 8 * g), v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int tap = 0; tap < 9; tap++) {
            int ky = tap / 3, kx = tap % 3;
            AT A(smem + (((int)sg + ky) * HW + kx) * KC, dextents<int32_t, 2>(KC, TW), array<int32_t, 2>{1, KC});
            BT B((device half *)wt + ((size_t)(n0 * 9 + tap)) * p.cin + k0, dextents<int32_t, 2>(KC, TN), array<int32_t, 2>{1, (int)(9 * p.cin)});
            op.run(A, B, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue into a [TH / SPLIT][TW][TN] tile (aliases the halo), then pooling and stores, once per phase
    constexpr int TR = TH / SPLIT, GN = TN / 8;
    const int qw = (int)p.w / 2;
    for (int ph = 0; ph < SPLIT; ph++) {
        if ((int)sg / TR == ph) {
            // in the matrix unit's layout: bias as a cooperative tensor (broadcast view), one cooperative store of the rounded values
            auto bT = op.template get_destination_cooperative_tensor<AT, BT, half>();
            auto hT = op.template get_destination_cooperative_tensor<AT, BT, half>();
            bT.load(BT((device half *)bias + n0, dextents<int32_t, 2>(TN, TW), array<int32_t, 2>{1, 0}));
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                const half v = (half)(cT[i] + (float)bT[i]);
                hT[i] = v > (half)0 ? v : (half)0;
            }
            hT.store(AT(smem + ((int)sg % TR) * TW * TN, dextents<int32_t, 2>(TN, TW), array<int32_t, 2>{1, TN}));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int it = (int)tid; it < (TR / 2) * (TW / 2) * GN; it += NT) {
            int g = it % GN, pp = it / GN;
            int pxl = pp % (TW / 2), pyl_l = pp / (TW / 2), pyl = ph * (TR / 2) + pyl_l;
            int gxb = x0 + 2 * pxl, gyb = y0 + 2 * pyl;
            if (gxb >= (int)p.w || gyb >= (int)p.h) continue;
            h8 mx; mx.lo = half4(-INFINITY); mx.hi = half4(-INFINITY);
            for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                h8 v = ld8t(smem + (((2 * pyl_l + dy) * TW) + (2 * pxl + dx)) * TN + 8 * g);
                mx = hmax8(mx, v);
                uint o = ((uint)(gyb + dy) * p.w + (uint)(gxb + dx)) * p.cout + (uint)(n0 + 8 * g);
                if (M8OUT) {
                    ulong raw = 0;
                    for (int c = 0; c < 4; c++) raw |= ((ulong)m8_enc_t(v.lo[c]) << (8 * c)) | ((ulong)m8_enc_t(v.hi[c]) << (8 * (c + 4)));
                    *(device ulong *)((device uchar *)skip + o) = raw;
                } else st8((device half *)skip + o, v);
            }
            st8(pooled + ((uint)(gyb / 2) * (uint)qw + (uint)(gxb / 2)) * p.cout + (uint)(n0 + 8 * g), mx);
        }
        if (ph + 1 < SPLIT) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define DEFINE_CONV_TILE(NAME, TW, TH, TN, KC, M8IN, M8OUT, SPLIT)                                                                \
    kernel void NAME(device const void *x [[buffer(0)]], device const half *wt [[buffer(1)]], device const half *bias [[buffer(2)]], \
                     device void *skip [[buffer(3)]], device half *pooled [[buffer(4)]], constant TileParams &p [[buffer(5)]],       \
                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                             \
                     uint sg [[simdgroup_index_in_threadgroup]]) {                                                                    \
        threadgroup half smem[((TH + 2) * (TW + 2) * KC) > (TH / SPLIT * TW * TN) ? ((TH + 2) * (TW + 2) * KC) : (TH / SPLIT * TW * TN)];           \
        conv_tile_impl<TW, TH, TN, KC, M8IN, M8OUT, SPLIT>(x, wt, bias, skip, pooled, p, tg, tid, sg, smem);                               \
    }

DEFINE_CONV_TILE(conv_tile_p2, 16, 8, 32, 16, false, true, 1)        // 16 -> 32, skip as m8
DEFINE_CONV_TILE(conv_tile_p3, 32, 4, 64, 16, true, false, 2)        // 64 -> 128, m8 input
DEFINE_CONV_TILE(conv_tile_p4, 32, 4, 32, 16, false, false, 1)       // 128 -> 128
DEFINE_CONV_TILE(conv_tile_p5, 16, 4, 32, 16, false, false, 1)       // 128 -> 256
// path W: K chunks of 32 (sr.enc_conv3x3_pool); the order of the fp32 sums follows the chunking
DEFINE_CONV_TILE(conv_tile_w3, 16, 4, 64, 32, true, false, 1)
DEFINE_CONV_TILE(conv_tile_w4, 16, 4, 64, 32, false, false, 1)
DEFINE_CONV_TILE(conv_tile_w5, 16, 2, 32, 32, false, false, 1)
