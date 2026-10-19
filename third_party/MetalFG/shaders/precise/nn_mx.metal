// Matrix-unit convolution (fg.arithmetic_tolerance_r1, fg.metal_techniques_r4 T1): Metal 4 tensor operations (matmul2d, binary16 x binary16 -> binary32 accumulate).
// One simdgroup computes a tile of 32 outputs positions x 32 output channels as an implicit matrix product, K = (taps x channels) in chunks of sixteen channels of one
// tap. The sum is formed in binary32 in the order of the unit; the epilogue (bias, leaky rule, saturation, explicit zero, residual) runs on the binary32 tile and the
// result is rounded to binary16 once. The pooling layers add the four positions of a block as three binary16 additions and the quarter (rule D), on the stored
// binary16 values of the four positions, as in the pinned kernels. Layers with cin a multiple of sixteen.
// The structure ConvParams is that of nn.metal (the host fills one structure for both).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
#include "nn_common.h"

struct ConvParams {
    uint W, H;
    uint cin, in_stride, in_base;
    uint cout, out_stride, out_base;
    uint k;
    uint row0;
    uint cin_w;
    uint flags;               // 1 act, 2 bound, 4 pool, 8 residual add, 16 explicit zero add, 128 the full output and its 2 x 2 maximum (buffer 6)
    float leak, bound;
    uint res_stride, res_base;
    uint bound_from;
    uint nest;                // order of the pairs (pinned-rounding variant)
    uint up_ws, up_hs;        // non-zero (1 x 1 layers): the input pixel is the net1 decoder sum up_corner(src)(x, y) + in(x, y), src of up_ws x up_hs pixels with cin
                              // channels at buffer 7, formed while loading (the tensor up_corner_add would store)
};

// Specialisation (fg.metal_techniques_r4 T6): the host builds one pipeline of conv_mx_t per layer shape with these constants (flags, kernel size, channel groups,
// decoder fold); a pipeline built without them reads the values from the parameters.
constant uint FC_FLAGS [[function_constant(0)]];
constant uint FC_K [[function_constant(1)]];
constant uint FC_NCH [[function_constant(2)]];
constant bool FC_UP [[function_constant(3)]];
constant bool FC_SPEC = is_function_constant_defined(FC_FLAGS);
static inline uint cflags(constant ConvParams &P) { return FC_SPEC ? FC_FLAGS : P.flags; }

constant uint TS = 33;        // row stride of the binary32 tile (odd: no bank conflicts in the epilogue)

static inline float epilogue_value(constant ConvParams &P, float v, uint ch) {
    if (cflags(P) & 1u) v = max(v, v * float(half(P.leak)));
    if ((cflags(P) & 2u) && ch >= P.bound_from) v = min(max(v, -P.bound), P.bound);
    return v;
}

// Pinned-rounding variant (RND): the running total starts at the binary16 bias and is rounded to binary16 after every (tap, 16-channel group) pair, the pairs follow the
// order of the pin (nest), and the leaky rule and the saturation run on the binary16 total, the residual add after them: the pin's rules A to C on the unit's sum, so that
// the result differs from the pinned kernels only where the unit's float sum of a pair lands on the other side of a binary16 rounding boundary.
static inline half finish_pin(constant ConvParams &P, half v, uint ch) {
    if (cflags(P) & 1u) { const half p = half(float(v) * float(half(P.leak))); v = max(v, p); }
    if ((cflags(P) & 2u) && ch >= P.bound_from) v = half(min(max(float(v), -P.bound), P.bound));
    return v;
}

template <uint NT, bool RND, uint SG = 4>
kernel void conv_mx_t(constant ConvParams &P [[buffer(0)]],
                    device const half *in [[buffer(1)]],
                    device half *out [[buffer(2)]],
                    device const half *wgt [[buffer(3)]],
                    device const half *bias [[buffer(4)]],
                    device const half *res [[buffer(5)]],
                    device half *pmax [[buffer(6)]],
                    device const half4 *usrc [[buffer(7)]],
                    constant uint *pairs [[buffer(9)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[SG][32 * TS];
    const uint flags = cflags(P), kk = FC_SPEC ? FC_K : P.k;
    const bool up = FC_SPEC ? FC_UP : P.up_ws != 0;
    const bool mpool = (flags & 128u) != 0;                            // full output plus the 2 x 2 maximum: the rows are the blocks' positions, as for pooling
    const bool pool = (flags & 4u) != 0 || mpool;
    const uint OW = P.W / 2, OH = P.H / 2;
    const uint npos = pool ? OW * OH : P.W * P.H;
    const uint per = pool ? 8 : 32;
    const uint ntm = (npos + per - 1) / per, ntn = (P.cout + NT - 1) / NT;
    const uint tile = tgid * SG + sg;
    if (tile >= ntm * ntn) return;
    const uint tm = tile % ntm, tn = tile / ntm;
    const uint u = tm * per + (pool ? lane / 4 : lane);                  // position, or pooling block, of this lane's A row
    const bool valid = u < npos;
    int x, y;
    if (pool) { x = int(u % OW) * 2 + int(lane & 1); y = int(u / OW) * 2 + int((lane >> 1) & 1); }
    else { x = int(u % P.W); y = int(u / P.W); }
    const uint wrow = P.row0 + min(tn * NT + (lane % NT), P.cout - 1);          // this lane's B row (rows past the output count are computed and dropped)
    const uint ktot = kk * kk * P.cin_w;
    threadgroup uint4 *sA = (threadgroup uint4 *)scratch[sg];
    threadgroup uint4 *sB = sA + 64;                                     // NT rows x 16 halves = 2 NT uint4 (A takes 64)

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, NT, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<1>> op;
    array<int, 2> st = {1, 16};
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> tA((threadgroup half *)sA, dextents<int, 2>(16, 32), st);
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> tB((threadgroup half *)sB, dextents<int, 2>(16, NT), st);
    auto c = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    threadgroup float *tile32 = scratch[sg];
    array<int, 2> ss = {1, int(TS)};
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> tD(tile32, dextents<int, 2>(NT, 32), ss);
    if (RND) {
        for (uint n = 0; n < NT; n++) tile32[lane * TS + n] = float(bias[P.row0 + min(tn * NT + n, P.cout - 1)]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        c.load(tD);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (uint i = 0; i < c.get_capacity(); i++) c[i] = 0.0f;
    }

    const uint taps = kk == 3 ? 9 : 1, nch = FC_SPEC ? FC_NCH : P.cin / 16;
    const uint npair = taps * nch;
    // the A row and B row of pair si (zero A outside the image), loaded into registers
    auto fetch = [&](uint si, thread uint4 &a0, thread uint4 &a1, thread uint4 &b0, thread uint4 &b1) {
#if defined(METALFG_CTL_ORDER_ROT)
        const uint s = (flags & 64u) ? npair - 1 - si : si;            // control of tol-det: the order of the sum follows the run counter the host passes
#else
        const uint s = si;
#endif
        // the pair order of the pin (rule B; nest 1 groups, x, y; nest 2 x, groups, y; nest 3 y, x, groups) as a table from the host: entry s holds the channel group
        // (bits 0..7), the tap's column and row offsets plus one (bits 8..9, 10..11) and the tap's first weight column / 8 (bits 12..31) of pair s (the order of the
        // unit variant is tap-major)
        const uint e = pairs[s], g = e & 0xFFu;
        const int px = x + int((e >> 8) & 3u) - 1, py = y + int((e >> 10) & 3u) - 1;
        const bool ok = valid && px >= 0 && py >= 0 && px < int(P.W) && py < int(P.H);
        device const uint4 *ip = (device const uint4 *)(in + (ulong)(ok ? uint(py) * P.W + uint(px) : 0u) * P.in_stride + P.in_base);
        device const uint4 *wp = (device const uint4 *)(wgt + (ulong)wrow * ktot) + (e >> 12);
        a0 = uint4(0); a1 = uint4(0);
        if (ok && up) {
            device const half4 *sk = (device const half4 *)ip;
            half4 q[4];
            for (uint j = 0; j < 4; j++) q[j] = up_add4(up_corner4(usrc, P.up_ws, P.up_hs, P.cin / 4, uint(px), uint(py), 4 * g + j), sk[4 * g + j]);
            a0 = uint4(as_type<uint2>(q[0]), as_type<uint2>(q[1]));
            a1 = uint4(as_type<uint2>(q[2]), as_type<uint2>(q[3]));
        } else if (ok) { a0 = ip[2 * g]; a1 = ip[2 * g + 1]; }
        if (lane < NT) { b0 = wp[2 * g]; b1 = wp[2 * g + 1]; }
    };
    uint4 a0, a1, b0 = uint4(0), b1 = uint4(0);
    fetch(0, a0, a1, b0, b1);
    for (uint si = 0; si < npair; si++) {
        sA[2 * lane] = a0; sA[2 * lane + 1] = a1;
        if (lane < NT) { sB[2 * lane] = b0; sB[2 * lane + 1] = b1; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (si + 1 < npair) fetch(si + 1, a0, a1, b0, b1);            // the next pair's loads are in flight during this pair's product
        op.run(tA, tB, c);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (RND) for (uint i = 0; i + 1 < c.get_capacity(); i += 2) { const half2 h = half2(float2(c[i], c[i + 1])); c[i] = float(h.x); c[i + 1] = float(h.y); }   // capacity: 32 (NT = 32) or 8 (NT = 8), even
    }
    c.store(tD);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (mpool) {
        // the lane's own position, stored as the plain layer does it; then the maximum of the block's four stored values, in the order of pool_max2
        // (max(max(a, b), max(d, e)), a and b the upper row); quarter q of the block owns channels 8 q .. 8 q + 7
        if (!valid) return;
        device half *o = out + ((ulong)(uint(y) * P.W + uint(x)) * P.out_stride + P.out_base + tn * NT);
        for (uint n = 0; n < NT; n++) {
            const uint ch = tn * NT + n;
            if (ch >= P.cout) break;
            o[n] = RND ? finish_pin(P, half(tile32[lane * TS + n]), ch) : half(epilogue_value(P, tile32[lane * TS + n] + float(bias[P.row0 + ch]), ch));
        }
        const uint blk = lane / 4, q = lane & 3;
        const uint ub = tm * per + blk;
        device half *po = pmax + (ulong)ub * P.out_stride + P.out_base + tn * NT;
        for (uint n = q * 8; n < q * 8 + 8 && n < NT; n++) {
            const uint ch = tn * NT + n;
            if (ch >= P.cout) break;
            half v[4];
            for (uint j = 0; j < 4; j++)
                v[j] = RND ? finish_pin(P, half(tile32[(blk * 4 + j) * TS + n]), ch) : half(epilogue_value(P, tile32[(blk * 4 + j) * TS + n] + float(bias[P.row0 + ch]), ch));
            po[n] = max(max(v[0], v[1]), max(v[2], v[3]));
        }
        return;
    }
    if (!pool) {
        if (!valid) return;
        device half *o = out + ((ulong)(uint(y) * P.W + uint(x)) * P.out_stride + P.out_base + tn * NT);
        device const half *rp = res + ((ulong)(uint(y) * P.W + uint(x)) * P.res_stride + P.res_base + tn * NT);
        for (uint n = 0; n < NT; n++) {
            const uint ch = tn * NT + n;
            if (ch >= P.cout) break;
            half hv;
            if (RND) {
                hv = finish_pin(P, half(tile32[lane * TS + n]), ch);
                if (flags & 16u) hv = half(float(hv) + 0.0f);                       // explicit zero: a negative zero becomes positive
                if (flags & 8u) hv = half(float(hv) + float(rp[n]));
                // 256: the head's rows go straight into the net2 output N = k + 2 u (rows 0..3) or k + u (rows 4..7), the form of n2_final (rule G), u from buffer 5
                if (flags & 256u) hv = half(float(hv) + ((P.out_base + ch) < 4u ? 2.0f : 1.0f) * float(rp[n]));
            } else {
                float v = epilogue_value(P, tile32[lane * TS + n] + float(bias[P.row0 + ch]), ch);
                // rule: the first convolution of a net2 pair has no negative zero (the explicit zero add of the pinned kernels, applied to the rounded value)
                if (flags & 8u) v += float(rp[n]);
                hv = half(v);
                if (flags & 16u) hv = half(float(hv) + 0.0f);
                if (flags & 256u) hv = half(float(hv) + ((P.out_base + ch) < 4u ? 2.0f : 1.0f) * float(rp[n]));
            }
            o[n] = hv;
        }
    } else {
        // lane = 4 * block + quarter: the quarter owns eight channels; the block's four positions are rows 4 * block .. 4 * block + 3 of the tile in raster order
        const uint blk = lane / 4, q = lane & 3;
        const uint ub = tm * per + blk;
        if (ub >= npos) return;
        for (uint n = q * 8; n < q * 8 + 8 && n < NT; n++) {
            const uint ch = tn * NT + n;
            if (ch >= P.cout) break;
            const float b = RND ? 0.0f : float(bias[P.row0 + ch]);
            half s = RND ? finish_pin(P, half(tile32[(blk * 4) * TS + n]), ch) : half(epilogue_value(P, tile32[(blk * 4) * TS + n] + b, ch));
            for (uint j = 1; j < 4; j++)
                s = half(float(s) + float(RND ? finish_pin(P, half(tile32[(blk * 4 + j) * TS + n]), ch) : half(epilogue_value(P, tile32[(blk * 4 + j) * TS + n] + b, ch))));
            out[(ulong)ub * P.out_stride + P.out_base + ch] = half(float(s) * 0.25f);
        }
    }
}

template [[host_name("conv_mx")]] kernel void conv_mx_t<32, false>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
// narrow heads (4, 1 and 3 outputs): a tile of 32 positions x 8 channels; no pooling
template [[host_name("conv_mx8")]] kernel void conv_mx_t<8, false>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
// pinned-rounding variants (default path of the engine)
template [[host_name("conv_mxr")]] kernel void conv_mx_t<32, true>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
// sixteen-output layers (n2.b.d0, n1.head): a tile of 32 positions x 16 channels (the unit's sum of an output does not depend on the tile width)
template [[host_name("conv_mxr16")]] kernel void conv_mx_t<16, true>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
// two simdgroups per threadgroup (the second network: measured faster than four)
template [[host_name("conv_mxr_s2")]] kernel void conv_mx_t<32, true, 2>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
template [[host_name("conv_mxr16_s2")]] kernel void conv_mx_t<16, true, 2>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
template [[host_name("conv_mxr8_s2")]] kernel void conv_mx_t<8, true, 2>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);
template [[host_name("conv_mxr8")]] kernel void conv_mx_t<8, true>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, device const half *, device half *, device const half4 *, constant uint *, uint, uint, uint);

// Two 1 x 1 layers in one pass (n1.dec0a then n1.head; n1.dec1a then n1.dec1b): the first (32 M1 outputs, M1 tiles of 32 in one simdgroup) exactly as conv_mxr forms
// a 1 x 1 layer (with the decoder fold of up_ws when set); its rounded output row of the lane's pixel stays in registers and is staged as the A tile of the second
// layer (32 M1 inputs, at most NT2 outputs), formed as conv_mxr / conv_mxr16 forms it. The intermediate tensor is not stored.
template <uint M1, uint NT2>
kernel void conv_pair1_t(constant ConvParams &P [[buffer(0)]],
                         device const half *in [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         device const half *wgt [[buffer(3)]],
                         device const half *bias [[buffer(4)]],
                         constant ConvParams &P2 [[buffer(5)]],
                         device const half *wgt2 [[buffer(6)]],
                         device const half4 *usrc [[buffer(7)]],
                         device const half *bias2 [[buffer(8)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[4][32 * TS];
    const uint npos = P.W * P.H, ntm = (npos + 31) / 32;
    const uint tile = tgid * 4 + sg;
    if (tile >= ntm) return;
    const uint u = tile * 32 + lane;
    const bool valid = u < npos;
    const int x = int(u % P.W), y = int(u / P.W);
    threadgroup uint4 *sA = (threadgroup uint4 *)scratch[sg];
    threadgroup uint4 *sB = sA + 64;                                     // M1 B tiles of 64 uint4 for layer 1, one tile for layer 2
    threadgroup float *tile32 = scratch[sg];
    array<int, 2> st = {1, 16}, ss = {1, int(TS)};
    using TT = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
    // layer 1: M1 tiles of 32 outputs
    constexpr auto d1 = mpp::tensor_ops::matmul2d_descriptor(32, 32, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<d1, execution_simdgroups<1>> op1;
    TT tA((threadgroup half *)sA, dextents<int, 2>(16, 32), st), tB1a((threadgroup half *)sB, dextents<int, 2>(16, 32), st),
       tB1b((threadgroup half *)(sB + 64), dextents<int, 2>(16, 32), st);
    auto c1a = op1.template get_destination_cooperative_tensor<TT, TT, float>();
    auto c1b = op1.template get_destination_cooperative_tensor<TT, TT, float>();
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> tD1(tile32, dextents<int, 2>(32, 32), ss);
    for (uint m = 0; m < M1; m++) {
        for (uint n = 0; n < 32; n++) tile32[lane * TS + n] = float(bias[P.row0 + min(m * 32 + n, P.cout - 1)]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (m == 0) c1a.load(tD1); else c1b.load(tD1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint nch = P.cin / 16;
    device const uint4 *ip = (device const uint4 *)(in + (ulong)(valid ? u : 0u) * P.in_stride + P.in_base);
    for (uint g = 0; g < nch; g++) {
        uint4 a0 = uint4(0), a1 = uint4(0);
        if (valid && P.up_ws != 0) {
            device const half4 *sk = (device const half4 *)ip;
            half4 q[4];
            for (uint j = 0; j < 4; j++) q[j] = up_add4(up_corner4(usrc, P.up_ws, P.up_hs, P.cin / 4, uint(x), uint(y), 4 * g + j), sk[4 * g + j]);
            a0 = uint4(as_type<uint2>(q[0]), as_type<uint2>(q[1]));
            a1 = uint4(as_type<uint2>(q[2]), as_type<uint2>(q[3]));
        } else if (valid) { a0 = ip[2 * g]; a1 = ip[2 * g + 1]; }
        sA[2 * lane] = a0; sA[2 * lane + 1] = a1;
        for (uint m = 0; m < M1; m++) {
            device const uint4 *wp = (device const uint4 *)(wgt + (ulong)(P.row0 + min(m * 32 + lane, P.cout - 1)) * P.cin_w);
            sB[64 * m + 2 * lane] = wp[2 * g]; sB[64 * m + 2 * lane + 1] = wp[2 * g + 1];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        op1.run(tA, tB1a, c1a);
        if (M1 > 1) op1.run(tA, tB1b, c1b);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i + 1 < c1a.get_capacity(); i += 2) {
            half2 h = half2(float2(c1a[i], c1a[i + 1])); c1a[i] = float(h.x); c1a[i + 1] = float(h.y);
            if (M1 > 1) { h = half2(float2(c1b[i], c1b[i + 1])); c1b[i] = float(h.x); c1b[i + 1] = float(h.y); }
        }
    }
    half v[32 * M1];
    for (uint m = 0; m < M1; m++) {
        if (m == 0) c1a.store(tD1); else c1b.store(tD1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < 32; n++) v[32 * m + n] = finish_pin(P, half(tile32[lane * TS + n]), 32 * m + n);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // layer 2: A = v (2 M1 groups of sixteen), at most NT2 outputs
    constexpr auto d2 = mpp::tensor_ops::matmul2d_descriptor(32, NT2, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<d2, execution_simdgroups<1>> op2;
    TT tB2((threadgroup half *)sB, dextents<int, 2>(16, NT2), st);
    auto c2 = op2.template get_destination_cooperative_tensor<TT, TT, float>();
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> tD2(tile32, dextents<int, 2>(NT2, 32), ss);
    for (uint n = 0; n < NT2; n++) tile32[lane * TS + n] = float(bias2[P2.row0 + min(n, P2.cout - 1)]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    c2.load(tD2);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint wrow2 = P2.row0 + min(lane % NT2, P2.cout - 1);
    for (uint g = 0; g < 2 * M1; g++) {
        uint4 a0 = uint4(0), a1 = uint4(0);
        if (valid) {
            for (uint j = 0; j < 4; j++) { a0[j] = as_type<uint>(half2(v[16 * g + 2 * j], v[16 * g + 2 * j + 1])); a1[j] = as_type<uint>(half2(v[16 * g + 8 + 2 * j], v[16 * g + 9 + 2 * j])); }
        }
        sA[2 * lane] = a0; sA[2 * lane + 1] = a1;
        if (lane < NT2) { device const uint4 *wp = (device const uint4 *)(wgt2 + (ulong)wrow2 * P2.cin_w); sB[2 * lane] = wp[2 * g]; sB[2 * lane + 1] = wp[2 * g + 1]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        op2.run(tA, tB2, c2);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i + 1 < c2.get_capacity(); i += 2) { const half2 h = half2(float2(c2[i], c2[i + 1])); c2[i] = float(h.x); c2[i + 1] = float(h.y); }
    }
    c2.store(tD2);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    device half *o = out + (ulong)u * P2.out_stride + P2.out_base;
    for (uint n = 0; n < NT2 && n < P2.cout; n++) o[n] = finish_pin(P2, half(tile32[lane * TS + n]), n);
}

template [[host_name("conv_pair1_16")]] kernel void conv_pair1_t<1, 16>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, constant ConvParams &, device const half *, device const half4 *, device const half *, uint, uint, uint);
template [[host_name("conv_pair2_32")]] kernel void conv_pair1_t<2, 32>(constant ConvParams &, device const half *, device half *, device const half *, device const half *, constant ConvParams &, device const half *, device const half4 *, device const half *, uint, uint, uint);
