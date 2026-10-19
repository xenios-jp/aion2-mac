// Matrix-unit kernels (MPP matmul2d, simdgroup scope, fp16 operands, fp32 cooperative destination).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "sr_math.h"
using namespace metal;
using namespace mpp::tensor_ops;

struct MmParams {
    uint npix;   // pixels (M)
    uint cin;    // K, multiple of 16
    uint cout;   // valid output channels
    uint flags;  // 1 = relu, 2 = head epilogue (rn16(acc) first, then an fp16 bias add)
    uint ncols;  // output channels covered by the weight view (multiple of TN, rows beyond cout are zero)
};

constant constexpr int TM = 16;   // pixels per simdgroup
constant constexpr int TN = 32;   // output channels per simdgroup (default instance)

// Per-pixel 1x1 convolution on the matrix units. Grid: (ncols / TNV, ceil(npix / TM)) threadgroups of one simdgroup.
template <int TNV>
static inline void conv1x1_mm_impl(device half *x, device half *wt, device half *bias, device half *y, constant MmParams &p, uint2 tg) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(x, dextents<int32_t, 2>((int)p.cin, (int)p.npix));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B(wt, dextents<int32_t, 2>((int)p.cin, (int)p.ncols));

    constexpr auto desc = matmul2d_descriptor(TM, TNV, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroup> op;

    int m0 = (int)tg.y * TM, n0 = (int)tg.x * TNV;
    auto cT = op.template get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    auto tA = A.slice(0, m0);
    auto tB = B.slice(0, n0);
    op.run(tA, tB, cT);

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);   // (n, m) in the tile
        int n = n0 + (int)idx[0], m = m0 + (int)idx[1];
        if (m >= (int)p.npix || n >= (int)p.cout) continue;
        half v = (p.flags & 2u) ? sr_two_round_add(cT[i], bias[n]) : (half)(cT[i] + (float)bias[n]);
        if (p.flags & 1u) v = max(v, (half)0);
        y[(uint)m * p.cout + (uint)n] = v;
    }
}

kernel void conv1x1_mm(device half *x [[buffer(0)]], device half *wt [[buffer(1)]], device half *bias [[buffer(2)]], device half *y [[buffer(3)]],
                       constant MmParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
    conv1x1_mm_impl<32>(x, wt, bias, y, p, tg);
}
// Bottleneck 1x1 with two K slices (sr.bottleneck_1x1): each half of the input channels accumulates in its own fp32 accumulator, the result is
// their sum (slice 0 + slice 1). K = cin must be even and a multiple of 32. The two slices run on the two simdgroups of a group at the same time
// (the serial chain of each is half as long as the chain of both); simdgroup 1 hands its accumulator through threadgroup memory and simdgroup 0
// forms the sum c0 + c1 and the epilogue. The operands and the order of every sum are those of the sequential kernel.
kernel void conv1x1_splitk(device half *x [[buffer(0)]], device half *wt [[buffer(1)]], device half *bias [[buffer(2)]], device half *y [[buffer(3)]],
                           constant MmParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    using TT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    threadgroup float part[TM * 32];
    const int half_k = (int)p.cin / 2;
    TT A(x + sg * half_k, dextents<int32_t, 2>(half_k, (int)p.npix), array<int32_t, 2>{1, (int)p.cin});
    TT B(wt + sg * half_k, dextents<int32_t, 2>(half_k, (int)p.ncols), array<int32_t, 2>{1, (int)p.cin});
    constexpr auto desc = matmul2d_descriptor(TM, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroup> op;
    int m0 = (int)tg.y * TM, n0 = (int)tg.x * 32;
    auto c = op.template get_destination_cooperative_tensor<TT, TT, float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;
    auto a = A.slice(0, m0); auto b = B.slice(0, n0);
    op.run(a, b, c);
    if (sg == 1) {
        #pragma unroll
        for (uint16_t i = 0; i < c.get_capacity(); ++i) {
            if (!c.is_valid_element(i)) continue;
            auto idx = c.get_multidimensional_index(i);
            part[(int)idx[1] * 32 + (int)idx[0]] = c[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        int n = n0 + (int)idx[0], m = m0 + (int)idx[1];
        if (m >= (int)p.npix || n >= (int)p.cout) continue;
        float acc = c[i] + part[(int)idx[1] * 32 + (int)idx[0]];
        half v = (half)(acc + (float)bias[n]);
        if (p.flags & 1u) v = max(v, (half)0);
        y[(uint)m * p.cout + (uint)n] = v;
    }
}

// N = 48: the 40 head outputs padded with zero rows
kernel void conv1x1_mm48(device half *x [[buffer(0)]], device half *wt [[buffer(1)]], device half *bias [[buffer(2)]], device half *y [[buffer(3)]],
                         constant MmParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
    conv1x1_mm_impl<48>(x, wt, bias, y, p, tg);
}

struct Conv3Params {
    uint w, h, cin, cout;
    uint m8_in;
};

// 3x3 convolution + ReLU on the matrix units. One simdgroup computes 16 pixels of one row x 32 output
// channels. For each tap and 32-channel K chunk the shifted, zero-padded activation tile is staged in
// threadgroup memory and fed to a dynamic-K matmul accumulating into one fp32 cooperative tensor.
// Grid: (ceil(w/16), h, cout/32) threadgroups of 32 threads. Weights [cout][3][3][cin]; cin multiple of 32.
kernel void conv3x3_mm(device const void *x [[buffer(0)]],
                       device half *wt [[buffer(1)]],
                       device half *bias [[buffer(2)]],
                       device half *y [[buffer(3)]],
                       constant Conv3Params &p [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    threadgroup half tile[16 * 32];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> A(tile, dextents<int32_t, 2>(32, 16));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    int x0 = (int)tg.x * 16, yy = (int)tg.y, n0 = (int)tg.z * 32;
    using BT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto cT = op.get_destination_cooperative_tensor<decltype(A), BT, float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    for (int ky = 0; ky < 3; ky++) {
        int sy = yy + ky - 1;
        if (sy < 0 || sy >= (int)p.h) continue;
        for (int kx = 0; kx < 3; kx++) {
            for (int k0 = 0; k0 < (int)p.cin; k0 += 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 16 * 32; e += 32) {
                    int px = (int)(e >> 5), c = (int)(e & 31);
                    int sx = x0 + px + kx - 1;
                    half v = 0;
                    if (sx >= 0 && sx < (int)p.w) {
                        uint idx = ((uint)sy * p.w + (uint)sx) * p.cin + (uint)(k0 + c);
                        if (p.m8_in) v = as_type<half>((ushort)(((device const uchar *)x)[idx] << 7));
                        else v = ((device const half *)x)[idx];
                    }
                    tile[e] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // B[n][k] for this tap/chunk: element (k, n) at wt[((n*3+ky)*3+kx)*cin + k0 + k]
                device half *wb = wt + ((size_t)(n0 * 9 + ky * 3 + kx)) * p.cin + k0;
                BT B(wb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, (int)(9 * p.cin)});
                op.run(A, B, cT);
            }
        }
    }

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        int n = n0 + (int)idx[0], px = x0 + (int)idx[1];
        if (px >= (int)p.w) continue;
        half v = (half)(cT[i] + (float)bias[n]);
        y[((uint)yy * p.w + (uint)px) * p.cout + (uint)n] = max(v, (half)0);
    }
}
