// Head network of P12 (sr.head_filter) for one tile of 32 L0 pixels, as used by the fused P12 kernel (T2): bilinear 2x upsample of d1 plus the
// skip s0 (both 8-bit minifloat) staged in threadgroup memory, the two chained 1x1 convolutions on the matrix units with the two-rounding epilogue,
// the result left in threadgroup memory (40 values per pixel, row stride kOStride). Same arithmetic as dec_impl in dec_fused.metal (the test
// entry `dec_head`); the fused kernel's output is checked bit for bit against that entry.
// The staging is split in two (head_issue: the five loads of a thread; head_store: the upsample and the sum into the tile) so the caller can issue
// its other independent reads in between. The tile holds x, then the hidden values, then the output, all in place: a simdgroup owns 16 rows and
// runs both layers on them, so only simdgroup barriers separate the layers.
#ifndef SR_HEAD_NET_H
#define SR_HEAD_NET_H
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "sr_math.h"
using namespace metal;
using namespace mpp::tensor_ops;

constant int kHeadTP = 32;                       // L0 pixels per tile: 8 wide, 4 tall
constant int kHeadC = 32;                        // input and hidden channels
constant int kOStride = 48 + 8;                  // row stride (halves) of the tile (x, hidden, then 40 output values; 48 matmul columns)
constant int kHeadHalves = kHeadTP * kOStride;

struct H8h { half4 lo, hi; };

static inline H8h head_dec8(ulong raw) {
    H8h r;
    for (int c = 0; c < 4; c++) {
        r.lo[c] = as_type<half>((ushort)(((raw >> (8 * c)) & 0xFF) << 7));
        r.hi[c] = as_type<half>((ushort)(((raw >> (8 * (c + 4))) & 0xFF) << 7));
    }
    return r;
}
template <bool WIDE>
static inline half4 head_lerp4(half4 n, half4 f) {
    if (!WIDE) return fma(half4(0.75h), n - f, f);
    return fma(half4(0.75h), n, f * half4(0.25h));
}
template <bool WIDE>
static inline H8h head_lerp8(H8h n, H8h f) { H8h r; r.lo = head_lerp4<WIDE>(n.lo, f.lo); r.hi = head_lerp4<WIDE>(n.hi, f.hi); return r; }

// the raw 8-bit values of one item (pixel, group of 8 channels) of thread tid < 128: four d1 neighbours and the skip
struct HeadRaw { ulong a, b, d, e, s; };

// d1: 8-bit minifloat, W1 x H1 x 32; s0: 8-bit minifloat, W0 x H0 x 32; the tile covers the L0 pixels (X0 .. X0 + 7, Y0 .. Y0 + 3).
static inline HeadRaw head_issue(device const uchar *d1, device const uchar *s0, uint W0, uint W1, uint H1, uint X0, uint Y0, uint tid) {
    const int pl = (int)tid / 4, g = (int)tid % 4;
    const int X = (int)X0 + (pl & 7), Y = (int)Y0 + (pl >> 3);
    const int nx = X >> 1, ny = Y >> 1;
    int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
    fx = clamp(fx, 0, (int)W1 - 1); fy = clamp(fy, 0, (int)H1 - 1);
    const uint ch = (uint)(8 * g);
    HeadRaw r;
    r.a = *(device const ulong *)(d1 + ((uint)ny * W1 + (uint)nx) * kHeadC + ch);
    r.b = *(device const ulong *)(d1 + ((uint)ny * W1 + (uint)fx) * kHeadC + ch);
    r.d = *(device const ulong *)(d1 + ((uint)fy * W1 + (uint)nx) * kHeadC + ch);
    r.e = *(device const ulong *)(d1 + ((uint)fy * W1 + (uint)fx) * kHeadC + ch);
    r.s = *(device const ulong *)(s0 + ((uint)Y * W0 + (uint)X) * kHeadC + ch);
    return r;
}

// x = rn16(up2(d1) + s0) of the item into the tile
template <bool WIDE>
static inline void head_store(HeadRaw r, uint tid, threadgroup half *xt) {
    const int pl = (int)tid / 4, g = (int)tid % 4;
    H8h hn = head_lerp8<WIDE>(head_dec8(r.a), head_dec8(r.b)), hf = head_lerp8<WIDE>(head_dec8(r.d), head_dec8(r.e));
    H8h up = head_lerp8<WIDE>(hn, hf);
    H8h s = head_dec8(r.s);
    *(threadgroup half4 *)(xt + pl * kOStride + 8 * g) = up.lo + s.lo;
    *(threadgroup half4 *)(xt + pl * kOStride + 8 * g + 4) = up.hi + s.hi;
}

// both layers for the 16 rows of simdgroup sg < 2 (the staged tile is visible to the group); afterwards the output of pixel pl is at
// xt + pl * kOStride. The epilogues stay in the matrix unit's register layout: the bias is loaded as a cooperative tensor of the same layout
// (a broadcast view over the bias vector), the rounded results go to a half cooperative tensor of that layout and are stored as a whole
// (no per-element index arithmetic or scattered stores). Layer 2 stores its 48 matmul columns; columns 40 to 47 fall in the row padding.
static inline void head_layers(device const half *wa, device const half *ba, device const half *wb, device const half *bb, uint sg, threadgroup half *xt) {
    using AT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using BT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    threadgroup half *rows = xt + sg * 16 * kOStride;
    AT A(rows, dextents<int32_t, 2>(kHeadC, 16), array<int32_t, 2>{1, kOStride});
    {
        constexpr auto d1d = matmul2d_descriptor(16, kHeadC, kHeadC, false, true, false);
        matmul2d<d1d, execution_simdgroup> op;
        BT B((device half *)wa, dextents<int32_t, 2>(kHeadC, kHeadC));
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        auto bT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        auto hT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        bT.load(BT((device half *)ba, dextents<int32_t, 2>(kHeadC, 16), array<int32_t, 2>{1, 0}));
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        op.run(A, B, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            const half v = sr_two_round_add(cT[i], bT[i]);
            hT[i] = v > (half)0 ? v : (half)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);   // every lane has read x before the hidden values overwrite it
        hT.store(A);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
        constexpr auto d2d = matmul2d_descriptor(16, 48, kHeadC, false, true, false);
        matmul2d<d2d, execution_simdgroup> op;
        BT B((device half *)wb, dextents<int32_t, 2>(kHeadC, 48));
        auto cT = op.template get_destination_cooperative_tensor<AT, BT, float>();
        auto bT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        auto hT = op.template get_destination_cooperative_tensor<AT, BT, half>();
        bT.load(BT((device half *)bb, dextents<int32_t, 2>(48, 16), array<int32_t, 2>{1, 0}));
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        op.run(A, B, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) hT[i] = sr_two_round_add(cT[i], bT[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        hT.store(AT(rows, dextents<int32_t, 2>(48, 16), array<int32_t, 2>{1, kOStride}));
    }
}
#endif
