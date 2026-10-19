// P2 in one kernel (sr.enc1_conv_pool): 3x3 conv 16 -> 32 + ReLU, pool, 1x1 conv 32 -> 64 + ReLU, pool.
// A tile of 32 x 4 L0 pixels per threadgroup (4 simdgroups, one row each); the intermediate tensors A, P(A), B stay in
// threadgroup memory. Stores: s0 = E(A), s1 = E(B), q2 = E(P(B)) as m8.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "sr_math.h"
using namespace mpp::tensor_ops;

struct P2fParams { uint w, h; };

static inline uchar m8e(half x) {
    ushort b = as_type<ushort>(x) & 0x7FFF;
    if (b > 0x7C00) return 0xFF;
    if (b < 0x400) return (uchar)(b >> 7);
    return (uchar)((b + 0x40) >> 7);
}

// Tile of TW x TH L0 pixels, one simdgroup per row (M = TW): 16 x 8 (256 threads) or 32 x 4 (128 threads). Both have 32 pooled pixels.
template <int TW, int TH>
static inline void p2_impl(device const half *feat, device const half *w0, device const half *b0, device const half *w1, device const half *b1,
                           device uchar *s0, device uchar *s1, device uchar *q2, constant P2fParams &p, uint3 tg, uint tid, uint sg,
                           threadgroup half *atile, threadgroup half *pat_) {
    constexpr int KC = 16, HW = TW + 2, NT = TH * 32;
    constexpr int PW = TW / 2, QW = TW / 4, QN = (TW / 4) * (TH / 4);   // pooled row width, L2 row width, L2 pixels
    constexpr int N2 = 128 / TH;                                          // stage 2: two pixel slices of 16, 64 / (TH / 2) channels per simdgroup
    static_assert((TW / 2) * (TH / 2) == 32, "32 pooled pixels per tile");
    // atile: A [row][col][32]; the halo of stage 1 lives in the same memory
    threadgroup half *halo = atile;                     // the halo is dead when A is written (a barrier separates the last tap from the epilogue)
    threadgroup half *pat = pat_;
    threadgroup half *bt = atile;                       // B: [L1 px][64 + 8 pad] (A is dead after pooling)
    using AT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using BT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    const int x0 = (int)tg.x * TW, y0 = (int)tg.y * TH;

    // ---- stage 1: 3x3 conv, 9 taps x one K = 16 chunk, M = 16 pixels per simdgroup, N = 32 ----
    {
        for (int it = (int)tid; it < (TH + 2) * HW * 2; it += NT) {
            int g = it & 1, col = (it >> 1) % HW, row = (it >> 1) / HW;
            int gx = x0 + col - 1, gy = y0 + row - 1;
            half4 lo = half4(0), hi = half4(0);
            if (gx >= 0 && gx < (int)p.w && gy >= 0 && gy < (int)p.h) {
                device const half *src = feat + ((uint)gy * p.w + (uint)gx) * 16 + 8 * g;
                lo = *(device const half4 *)src; hi = *(device const half4 *)(src + 4);
            }
            *(threadgroup half4 *)(halo + (row * HW + col) * KC + 8 * g) = lo;
            *(threadgroup half4 *)(halo + (row * HW + col) * KC + 8 * g + 4) = hi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr auto d = matmul2d_descriptor(TW, 32, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroup> op;
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        auto bT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        auto hT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        bT.load(BT((device half *)b0, dextents<int32_t, 2>(32, TW), array<int32_t, 2>{1, 0}));
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        #pragma unroll
        for (int tap = 0; tap < 9; tap++) {
            int ky = tap / 3, kx = tap % 3;
            AT A(halo + (((int)sg + ky) * HW + kx) * KC, dextents<int32_t, 2>(KC, TW), array<int32_t, 2>{1, KC});
            BT B((device half *)w0 + (size_t)tap * 16, dextents<int32_t, 2>(KC, 32), array<int32_t, 2>{1, 144});
            op.run(A, B, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // epilogue in the matrix unit's layout: bias as a cooperative tensor (broadcast view), one cooperative store of the rounded values
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            const half v = (half)(cT[i] + (float)bT[i]);
            hT[i] = v > (half)0 ? v : (half)0;
        }
        hT.store(AT(atile + (int)sg * TW * 32, dextents<int32_t, 2>(32, TW), array<int32_t, 2>{1, 32}));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- pool A: s0 = E(A) and P(A) (32 pooled pixels x 4 channel groups = 128 items) ----
    if (tid < 128) {
        int g = (int)tid & 3, pp = (int)tid >> 2;                   // pooled pixel 0..31 in row order
        int pr = pp / PW, pc = pp % PW;
        half4 mlo = half4(-INFINITY), mhi = half4(-INFINITY);
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int r = 2 * pr + dy, c = 2 * pc + dx;
            half4 lo = *(threadgroup half4 *)(atile + (r * TW + c) * 32 + 8 * g), hi = *(threadgroup half4 *)(atile + (r * TW + c) * 32 + 8 * g + 4);
            mlo = max(mlo, lo); mhi = max(mhi, hi);
            int gx = x0 + c, gy = y0 + r;
            if (gx < (int)p.w && gy < (int)p.h) {
                ulong raw = 0;
                for (int k = 0; k < 4; k++) raw |= ((ulong)m8e(lo[k]) << (8 * k)) | ((ulong)m8e(hi[k]) << (8 * (k + 4)));
                *(device ulong *)(s0 + ((uint)gy * p.w + (uint)gx) * 32 + 8 * g) = raw;
            }
        }
        *(threadgroup half4 *)(pat + pp * 40 + 8 * g) = mlo;
        *(threadgroup half4 *)(pat + pp * 40 + 8 * g + 4) = mhi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 2: 1x1 conv 32 -> 64 on P(A): simdgroup s takes pooled pixels (s & 1) * 16.., channels (s >> 1) * N2.. ----
    {
        const int mt = (int)sg & 1, nt = (int)sg >> 1;
        constexpr auto d = matmul2d_descriptor(16, N2, 32, false, true, false);
        matmul2d<d, execution_simdgroup> op;
        AT A(pat + mt * 16 * 40, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 40});
        BT B((device half *)w1 + (size_t)nt * N2 * 32, dextents<int32_t, 2>(32, N2));
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        auto bT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        auto hT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        bT.load(BT((device half *)b1 + nt * N2, dextents<int32_t, 2>(N2, 16), array<int32_t, 2>{1, 0}));
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        op.run(A, B, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            const half v = (half)(cT[i] + (float)bT[i]);
            hT[i] = v > (half)0 ? v : (half)0;
        }
        hT.store(AT(bt + mt * 16 * 72 + nt * N2, dextents<int32_t, 2>(N2, 16), array<int32_t, 2>{1, 72}));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stores: s1 = E(B) (32 L1 pixels x 8 groups), q2 = E(P(B)) (8 L2 pixels x 8 groups) ----
    const int w1s = (int)p.w / 2, h1s = (int)p.h / 2, w2s = (int)p.w / 4;
    for (int it = (int)tid; it < 256; it += NT) {
        int g = it & 7, pp = it >> 3;                                 // 32 pixels, 8 channel groups = 256 items
        int gx = (int)tg.x * PW + (pp % PW), gy = (int)tg.y * (TH / 2) + (pp / PW);
        if (gx < w1s && gy < h1s) {
            half4 lo = *(threadgroup half4 *)(bt + pp * 72 + 8 * g), hi = *(threadgroup half4 *)(bt + pp * 72 + 8 * g + 4);
            ulong raw = 0;
            for (int k = 0; k < 4; k++) raw |= ((ulong)m8e(lo[k]) << (8 * k)) | ((ulong)m8e(hi[k]) << (8 * (k + 4)));
            *(device ulong *)(s1 + ((uint)gy * w1s + (uint)gx) * 64 + 8 * g) = raw;
        }
    }
    if (tid < (uint)(QN * 8)) {
        int g = (int)tid & 7, qp = (int)tid >> 3;                    // 8 L2 pixels
        int qc = qp % QW, qr = qp / QW;
        half4 mlo = half4(-INFINITY), mhi = half4(-INFINITY);
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int pp = (2 * qr + dy) * PW + (2 * qc + dx);
            mlo = max(mlo, *(threadgroup half4 *)(bt + pp * 72 + 8 * g));
            mhi = max(mhi, *(threadgroup half4 *)(bt + pp * 72 + 8 * g + 4));
        }
        int gx = (int)tg.x * QW + qc, gy = (int)tg.y * (TH / 4) + qr;
        if (gx < w2s && gy < (int)p.h / 4) {
            ulong raw = 0;
            for (int k = 0; k < 4; k++) raw |= ((ulong)m8e(mlo[k]) << (8 * k)) | ((ulong)m8e(mhi[k]) << (8 * (k + 4)));
            *(device ulong *)(q2 + ((uint)gy * w2s + (uint)gx) * 64 + 8 * g) = raw;
        }
    }
}

#define P2_KERNEL(NAME, TW, TH) \
kernel void NAME(device const half *feat [[buffer(0)]], \
                 device const half *w0 [[buffer(1)]], device const half *b0 [[buffer(2)]],      /* enc1a [32][3][3][16], [32] */ \
                 device const half *w1 [[buffer(3)]], device const half *b1 [[buffer(4)]],      /* enc1b [64][32], [64] */ \
                 device uchar *s0 [[buffer(5)]], device uchar *s1 [[buffer(6)]], device uchar *q2 [[buffer(7)]], \
                 constant P2fParams &p [[buffer(8)]], \
                 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup half atile[TH * TW * 32];   /* A (8192 B); the halo and B alias it */ \
    threadgroup half pat[32 * 40];          /* P(A): [pooled px][32 + 8 pad] */ \
    p2_impl<TW, TH>(feat, w0, b0, w1, b1, s0, s1, q2, p, tg, tid, sg, atile, pat); \
}
// 32 x 4 (128 threads, M = 32 per product: half the weight fetches per pixel of 16 x 8); 16 x 8 measured 3-4 % slower per call
P2_KERNEL(conv_tile_p2f, 32, 4)
