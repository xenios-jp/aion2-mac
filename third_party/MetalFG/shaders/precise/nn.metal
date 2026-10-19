// Network kernels shared by net1 and net2 (specs fg.net1_forward, fg.net2_forward, fg.numerics) with the arithmetic of fg.arithmetic_pin_r4: every value is
// defined by the CPU model of that spec and reproduced bit for bit, except the two approximate functions (fast divide and fast exp2) of rules H and I.
// All tensors are binary16, channels last: element (x, y, c) of a tensor of width W and channel stride S is at (y * W + x) * S + base + c.
// This file is compiled without fast math; contraction is off, the fused operations are written with fma.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
#include "nn_common.h"

struct ConvParams {
    uint W, H;                // spatial size of the input (and of the convolution output before pooling)
    uint cin, in_stride, in_base;
    uint cout, out_stride, out_base;
    uint k;                   // 1 or 3
    uint row0;                // first weight row used (grouped layers)
    uint cin_w;               // input channels per tap as stored in the weight tensor
    uint flags;               // 1 act, 2 bound, 4 pool (rule D), 8 residual add, 16 explicit zero add (first convolution of a net2 pair)
    float leak, bound;
    uint res_stride, res_base;
    uint bound_from;          // the bound applies to output channels >= bound_from
    uint nest;                // order of the (tap, channel group) pairs of a 3x3 layer (rule B): 1 groups, x, y; 2 x, groups, y; 3 y, x, groups
    uint up_ws, up_hs;        // matrix kernels only (nn_mx.metal): input formed as an enlargement plus skip
};

// Rule A: one (filter tap, channel group) pair. The sixteen products are exact in binary32; they are added one after another, in ascending channel order, to the
// running total widened to binary32, and the total is rounded to binary16 before the next pair. Taps outside the image contribute nothing.
static inline half pair_acc(constant ConvParams &P, device const half *in, device const half *wrow, int x, int y, int kx, int ky, uint g, half tot) {
    const int px = P.k == 3 ? x + kx - 1 : x, py = P.k == 3 ? y + ky - 1 : y;
    if (px < 0 || py < 0 || px >= int(P.W) || py >= int(P.H)) return tot;
    device const half *ip = in + ((ulong)uint(py) * P.W + uint(px)) * P.in_stride + P.in_base;
    device const half *wp = wrow + (P.k == 3 ? uint(ky * 3 + kx) : 0u) * P.cin_w;
    float t = float(tot);
    const uint c1 = min(g * 16 + 16, P.cin);
#if defined(METALFG_CTL_ACC3)
    // control of tol-budget: the running total is rounded to three mantissa bits after every addition
    for (uint c = g * 16; c < c1; c++) { t = fma(float(wp[c]), float(ip[c]), t); t = as_type<float>((as_type<uint>(t) + (1u << 19)) & ~((1u << 20) - 1u)); }
#else
    for (uint c = g * 16; c < c1; c++) t = fma(float(wp[c]), float(ip[c]), t);
#endif
    return half(t);
}

// Rule C: the leaky rule on binary16 values and the saturation, applied to the final running total of output channel ch.
static inline half finish_value(constant ConvParams &P, half v, uint ch) {
    if (P.flags & 1u) {
        const half p = half(float(v) * float(half(P.leak)));      // the binary16 product with the binary16 slope, then the larger of the two
        v = max(v, p);
    }
    if ((P.flags & 2u) && ch >= P.bound_from) v = half(min(max(float(v), -P.bound), P.bound));
    return v;
}

// Rules A to C for one output value of one position: the running total, the leaky rule on binary16 values, the saturation.
static inline half conv_value(constant ConvParams &P, device const half *in, device const half *wgt, device const half *bias, int x, int y, uint ch) {
    half tot = bias[P.row0 + ch];
    device const half *wrow = wgt + (ulong)(P.row0 + ch) * (P.k * P.k * P.cin_w);
    const uint ng = (P.cin + 15) / 16;
    if (P.k == 1) {
        for (uint g = 0; g < ng; g++) tot = pair_acc(P, in, wrow, x, y, 0, 0, g, tot);
    } else if (P.nest == 1) {
        for (uint g = 0; g < ng; g++) for (int kx = 0; kx < 3; kx++) for (int ky = 0; ky < 3; ky++) tot = pair_acc(P, in, wrow, x, y, kx, ky, g, tot);
    } else if (P.nest == 2) {
        for (int kx = 0; kx < 3; kx++) for (uint g = 0; g < ng; g++) for (int ky = 0; ky < 3; ky++) tot = pair_acc(P, in, wrow, x, y, kx, ky, g, tot);
    } else {
#if defined(METALFG_CTL_ACC_ORDER)
        // control 3 of fg.performance_r1: the channel groups of the layer the host marks (flag 32) taken in the opposite order; same speed, other last bits
        if (P.flags & 32u) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (uint g = ng; g-- > 0;) tot = pair_acc(P, in, wrow, x, y, kx, ky, g, tot);
        else
#endif
        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (uint g = 0; g < ng; g++) tot = pair_acc(P, in, wrow, x, y, kx, ky, g, tot);
    }
    return finish_value(P, tot, ch);
}

// One thread per output element (position, channel); channel fastest.
kernel void conv_pin(constant ConvParams &P [[buffer(0)]],
                     device const half *in [[buffer(1)]],
                     device half *out [[buffer(2)]],
                     device const half *wgt [[buffer(3)]],
                     device const half *bias [[buffer(4)]],
                     device const half *res [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    const uint ch = gid % P.cout, pix = gid / P.cout;
    if (!(P.flags & 4u)) {
        if (pix >= P.W * P.H) return;
        const uint x = pix % P.W, y = pix / P.W;
        half v = conv_value(P, in, wgt, bias, int(x), int(y), ch);
        if (P.flags & 16u) v = half(float(v) + 0.0f);                                  // explicit zero: a negative zero becomes positive
        if (P.flags & 8u) v = half(float(v) + float(res[(y * P.W + x) * P.res_stride + P.res_base + ch]));
        out[(y * P.W + x) * P.out_stride + P.out_base + ch] = v;
    } else {
        // rule D: the convolution, activation and saturation run at full resolution; the 2 x 2 block is added in raster order, one binary16 addition after
        // the other, and multiplied by the quarter in binary16
        const uint OW = P.W / 2, OH = P.H / 2;
        if (pix >= OW * OH) return;
        const int ox = int(pix % OW) * 2, oy = int(pix / OW) * 2;
        half s = conv_value(P, in, wgt, bias, ox, oy, ch);
        s = half(float(s) + float(conv_value(P, in, wgt, bias, ox + 1, oy, ch)));
        s = half(float(s) + float(conv_value(P, in, wgt, bias, ox, oy + 1, ch)));
        s = half(float(s) + float(conv_value(P, in, wgt, bias, ox + 1, oy + 1, ch)));
        out[(pix * P.out_stride) + P.out_base + ch] = half(float(s) * 0.25f);
    }
}

// Register-blocked scalar path of the same convolution (fg.metal_techniques_r4 T5, T11). One thread computes TP positions x TC output channels as TP TC running totals
// held in registers. Every output keeps its own sequential chain exactly as rule A: per (tap, group) pair the sixteen products are added with fma in ascending channel
// order to the running total, which is rounded to binary16 after the pair; only the mapping of outputs to threads differs from conv_pin. Positions are TP consecutive
// columns of a row, or TP / 4 consecutive pooling blocks of a row (each block's 4 positions in raster order) when the layer pools. Serves layers with cin a multiple of
// 16; an output count that is not a multiple of TC is predicated (the extra channels read the last valid row and are not stored).
template <uint TP, uint TC>
static inline void blk_pair(constant ConvParams &P, device const half *in, device const half *wgt, uint ch0, uint ktot, thread int2 (&pos)[TP], thread bool (&valid)[TP],
                            int kx, int ky, uint g, thread float (&acc)[TP][TC]) {
    const uint c0 = g * 16;
    device const uint4 *wp[TC];
    for (uint ch = 0; ch < TC; ch++) wp[ch] = (device const uint4 *)(wgt + (ulong)(P.row0 + min(ch0 + ch, P.cout - 1)) * ktot + (P.k == 3 ? uint(ky * 3 + kx) : 0u) * P.cin_w + c0);
    for (uint j = 0; j < TP; j++) {
        const int px = P.k == 3 ? pos[j].x + kx - 1 : pos[j].x, py = P.k == 3 ? pos[j].y + ky - 1 : pos[j].y;
        if (!valid[j] || px < 0 || py < 0 || px >= int(P.W) || py >= int(P.H)) continue;
        device const uint4 *ip = (device const uint4 *)(in + ((ulong)uint(py) * P.W + uint(px)) * P.in_stride + P.in_base + c0);
        for (uint q = 0; q < 2; q++) {
            const uint4 xr = ip[q];                                    // eight channels as four half pairs, ascending
            const float2 x0 = float2(as_type<half2>(xr.x)), x1 = float2(as_type<half2>(xr.y)), x2 = float2(as_type<half2>(xr.z)), x3 = float2(as_type<half2>(xr.w));
            for (uint ch = 0; ch < TC; ch++) {
                const uint4 wr = wp[ch][q];
                const float2 w0 = float2(as_type<half2>(wr.x)), w1 = float2(as_type<half2>(wr.y)), w2 = float2(as_type<half2>(wr.z)), w3 = float2(as_type<half2>(wr.w));
                float a = acc[j][ch];
                a = fma(w0.x, x0.x, a); a = fma(w0.y, x0.y, a); a = fma(w1.x, x1.x, a); a = fma(w1.y, x1.y, a);
                a = fma(w2.x, x2.x, a); a = fma(w2.y, x2.y, a); a = fma(w3.x, x3.x, a); a = fma(w3.y, x3.y, a);
                acc[j][ch] = a;
            }
        }
        for (uint ch = 0; ch < TC; ch++) acc[j][ch] = float(half(acc[j][ch]));
    }
}

template <uint TP, uint TC>
static inline void conv_blk_body(constant ConvParams &P, device const half *in, device half *out, device const half *wgt, device const half *bias,
                                 device const half *res, uint gid) {
    const bool pool = (P.flags & 4u) != 0;
    const uint OW = P.W / 2, OH = P.H / 2;
    const uint per = pool ? TP / 4 : TP;                        // pooled pixels or columns per thread
    const uint nbx = ((pool ? OW : P.W) + per - 1) / per;
    const uint npb = nbx * (pool ? OH : P.H);
    const uint ncb = (P.cout + TC - 1) / TC;
    if (gid >= npb * ncb) return;
    const uint pb = gid % npb, cb = gid / npb, ch0 = cb * TC;
    const uint by = pb / nbx, bx = pb % nbx;
    int2 pos[TP];
    bool valid[TP];
    for (uint j = 0; j < TP; j++) {
        if (pool) {
            const uint ox = bx * per + j / 4, jj = j % 4;
            pos[j] = int2(int(2 * ox + (jj & 1)), int(2 * by + (jj >> 1)));
            valid[j] = ox < OW;
        } else {
            pos[j] = int2(int(bx * TP + j), int(by));
            valid[j] = pos[j].x < int(P.W);
        }
    }
    const uint ktot = P.k * P.k * P.cin_w, ng = P.cin / 16;
    float acc[TP][TC];
    for (uint ch = 0; ch < TC; ch++) {
        const float b = float(bias[P.row0 + min(ch0 + ch, P.cout - 1)]);        // predicated channels repeat the last valid row
        for (uint j = 0; j < TP; j++) acc[j][ch] = b;
    }
    if (P.k == 1) {
        for (uint g = 0; g < ng; g++) blk_pair<TP, TC>(P, in, wgt, ch0, ktot, pos, valid, 0, 0, g, acc);
    } else if (P.nest == 1) {
        for (uint g = 0; g < ng; g++) for (int kx = 0; kx < 3; kx++) for (int ky = 0; ky < 3; ky++) blk_pair<TP, TC>(P, in, wgt, ch0, ktot, pos, valid, kx, ky, g, acc);
    } else if (P.nest == 2) {
        for (int kx = 0; kx < 3; kx++) for (uint g = 0; g < ng; g++) for (int ky = 0; ky < 3; ky++) blk_pair<TP, TC>(P, in, wgt, ch0, ktot, pos, valid, kx, ky, g, acc);
    } else {
        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (uint g = 0; g < ng; g++) blk_pair<TP, TC>(P, in, wgt, ch0, ktot, pos, valid, kx, ky, g, acc);
    }
    for (uint ch = 0; ch < TC; ch++) {
        const uint c = ch0 + ch;
        if (c >= P.cout) continue;
        half v[TP];
        for (uint j = 0; j < TP; j++) v[j] = finish_value(P, half(acc[j][ch]), c);
        if (pool) {
            for (uint i = 0; i < per; i++) {
                const uint ox = bx * per + i;
                if (ox >= OW) continue;
                half s = v[4 * i];
                s = half(float(s) + float(v[4 * i + 1]));
                s = half(float(s) + float(v[4 * i + 2]));
                s = half(float(s) + float(v[4 * i + 3]));
                out[(by * OW + ox) * P.out_stride + P.out_base + c] = half(float(s) * 0.25f);
            }
        } else {
            for (uint j = 0; j < TP; j++) {
                if (!valid[j]) continue;
                half r = v[j];
                if (P.flags & 16u) r = half(float(r) + 0.0f);
                const uint x = uint(pos[j].x), y = uint(pos[j].y);
                if (P.flags & 8u) r = half(float(r) + float(res[(y * P.W + x) * P.res_stride + P.res_base + c]));
                out[(y * P.W + x) * P.out_stride + P.out_base + c] = r;
            }
        }
    }
}

#define BLK_KERNEL(NAME, TP, TC) \
kernel void NAME(constant ConvParams &P [[buffer(0)]], device const half *in [[buffer(1)]], device half *out [[buffer(2)]], device const half *wgt [[buffer(3)]], \
                 device const half *bias [[buffer(4)]], device const half *res [[buffer(5)]], uint gid [[thread_position_in_grid]]) { \
    conv_blk_body<TP, TC>(P, in, out, wgt, bias, res, gid); \
}
BLK_KERNEL(conv_blk4x8, 4, 8)
BLK_KERNEL(conv_blk4x4, 4, 4)

struct PoolParams { uint W, H, C; };   // input size; output is W/2 x H/2

kernel void pool_max2(constant PoolParams &P [[buffer(0)]],
                      device const half4 *in [[buffer(1)]],
                      device half4 *out [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    // four channels per thread (C is a multiple of four)
    const uint OW = P.W / 2, OH = P.H / 2, C4 = P.C / 4;
    if (gid >= OW * OH * C4) return;
    const uint c = gid % C4, p = gid / C4, x = p % OW, y = p / OW;
    const uint i = ((2 * y) * P.W + 2 * x) * C4 + c;
    half4 a = in[i], b = in[i + C4], d = in[i + P.W * C4], e = in[i + P.W * C4 + C4];
    out[gid] = max(max(a, b), max(d, e));   // exact maximum
}

struct UpParams { uint Ws, Hs, C; };   // source size; destination is 2 Ws x 2 Hs

// Interpolation step in binary16 (rules E and H): weights (1 - f) and f rounded to binary16, each product rounded, the sum rounded.
static inline half lerp16(half a, half b, float f) {
    const half w1 = half(f), w0 = half(1.0f - f);
    const half t0 = half(float(a) * float(w0)), t1 = half(float(b) * float(w1));
    return half(float(t0) + float(t1));
}

// Net1 (rule H): corner-aligned bilinear x2. The coordinate scale (Ws - 1) / (Wd - 1) is the binary32 quotient of fast::divide; coordinates are binary32; the
// weights and both interpolation steps are binary16, horizontal first; the skip sum is formed in binary32 and rounded once.
kernel void up_corner_add(constant UpParams &P [[buffer(0)]],
                          device const half4 *src [[buffer(1)]],
                          device const half4 *skip [[buffer(2)]],
                          device half4 *out [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    // four channels per thread (C is a multiple of four)
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2, C4 = P.C / 4;
    if (gid >= Wd * Hd * C4) return;
    const uint c = gid % C4, p = gid / C4, X = p % Wd, Y = p / Wd;
    out[gid] = up_add4(up_corner4(src, P.Ws, P.Hs, C4, X, Y, c), skip[gid]);
}

// Net2: half-pixel bilinear x2 (weights 3/4 and 1/4, edges repeated), rounded once.
kernel void up2_half(constant UpParams &P [[buffer(0)]],
                     device const half4 *src [[buffer(1)]],
                     device half4 *out [[buffer(2)]],
                     uint3 g3 [[thread_position_in_grid]]) {
    // four channels per thread (C is a multiple of four); grid (C / 4, 2 Ws, 2 Hs)
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2, C4 = P.C / 4;
    const uint c = g3.x, X = g3.y, Y = g3.z;
    if (c >= C4 || X >= Wd || Y >= Hd) return;
    const uint gid = (Y * Wd + X) * C4 + c;
    const int nx = int(X / 2), ny = int(Y / 2);
    const int fxi = clamp((X & 1) ? nx + 1 : nx - 1, 0, int(P.Ws) - 1);
    const int fyi = clamp((Y & 1) ? ny + 1 : ny - 1, 0, int(P.Hs) - 1);
    const float4 h0 = 0.75f * float4(src[(uint(ny) * P.Ws + uint(nx)) * C4 + c]) + 0.25f * float4(src[(uint(ny) * P.Ws + uint(fxi)) * C4 + c]);
    const float4 h1 = 0.75f * float4(src[(uint(fyi) * P.Ws + uint(nx)) * C4 + c]) + 0.25f * float4(src[(uint(fyi) * P.Ws + uint(fxi)) * C4 + c]);
    out[gid] = half4(0.75f * h0 + 0.25f * h1);
}

// Net2 (rule E), the enlargement that feeds the two final heads: source coordinate (index - 1/2) / 2 in binary32; where it is negative both taps address the first
// column and the fraction is the negative coordinate itself (extrapolation). Weights and both steps in binary16, horizontal first.
static inline void up2x_axis(uint X, uint Ns, thread uint &i0, thread uint &i1, thread float &f) {
    const float s = (float(X) - 0.5f) * 0.5f;
    if (s < 0.0f) { i0 = i1 = 0; f = s; }
    else { i0 = min(uint(floor(s)), Ns - 1); i1 = min(i0 + 1, Ns - 1); f = s - float(i0); }
}

kernel void up2_extrap(constant UpParams &P [[buffer(0)]],
                       device const half4 *src [[buffer(1)]],
                       device half4 *out [[buffer(2)]],
                       uint3 g3 [[thread_position_in_grid]]) {
    // four channels per thread (C is a multiple of four); grid (C / 4, 2 Ws, 2 Hs)
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2, C4 = P.C / 4;
    const uint c = g3.x, X = g3.y, Y = g3.z;
    if (c >= C4 || X >= Wd || Y >= Hd) return;
    const uint gid = (Y * Wd + X) * C4 + c;
    uint ix0, ix1, iy0, iy1;
    float fx, fy;
    up2x_axis(X, P.Ws, ix0, ix1, fx);
    up2x_axis(Y, P.Hs, iy0, iy1, fy);
    const half fxw1 = half(fx), fxw0 = half(1.0f - fx), fyw1 = half(fy), fyw0 = half(1.0f - fy);
    const half4 h0 = lerp16v(src[(iy0 * P.Ws + ix0) * C4 + c], src[(iy0 * P.Ws + ix1) * C4 + c], fxw0, fxw1);
    const half4 h1 = lerp16v(src[(iy1 * P.Ws + ix0) * C4 + c], src[(iy1 * P.Ws + ix1) * C4 + c], fxw0, fxw1);
    out[gid] = lerp16v(h0, h1, fyw0, fyw1);
}

// Scalar forms (one element per thread) for channel counts that are not a multiple of four; the engine's tensors always are, the tests use other counts.
kernel void pool_max2_s(constant PoolParams &P [[buffer(0)]],
                      device const half *in [[buffer(1)]],
                      device half *out [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint OW = P.W / 2, OH = P.H / 2;
    if (gid >= OW * OH * P.C) return;
    const uint c = gid % P.C, p = gid / P.C, x = p % OW, y = p / OW;
    const uint i = ((2 * y) * P.W + 2 * x) * P.C + c;
    half a = in[i], b = in[i + P.C], d = in[i + P.W * P.C], e = in[i + P.W * P.C + P.C];
    out[gid] = max(max(a, b), max(d, e));   // exact maximum
}

kernel void up_corner_add_s(constant UpParams &P [[buffer(0)]],
                          device const half *src [[buffer(1)]],
                          device const half *skip [[buffer(2)]],
                          device half *out [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2;
    if (gid >= Wd * Hd * P.C) return;
    const uint c = gid % P.C, p = gid / P.C, X = p % Wd, Y = p / Wd;
    const float scx = Wd > 1 ? fast::divide(float(P.Ws - 1), float(Wd - 1)) : 0.0f;
    const float scy = Hd > 1 ? fast::divide(float(P.Hs - 1), float(Hd - 1)) : 0.0f;
    const float sx = float(X) * scx, sy = float(Y) * scy;
    const uint ix0 = min(uint(max(floor(sx), 0.0f)), P.Ws - 1), iy0 = min(uint(max(floor(sy), 0.0f)), P.Hs - 1);
    const uint ix1 = min(ix0 + 1, P.Ws - 1), iy1 = min(iy0 + 1, P.Hs - 1);
    const float fx = sx - float(ix0), fy = sy - float(iy0);
    const half h0 = lerp16(src[(iy0 * P.Ws + ix0) * P.C + c], src[(iy0 * P.Ws + ix1) * P.C + c], fx);
    const half h1 = lerp16(src[(iy1 * P.Ws + ix0) * P.C + c], src[(iy1 * P.Ws + ix1) * P.C + c], fx);
    const half u = lerp16(h0, h1, fy);
    out[gid] = half(float(u) + float(skip[gid]));
}

kernel void up2_half_s(constant UpParams &P [[buffer(0)]],
                     device const half *src [[buffer(1)]],
                     device half *out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2;
    if (gid >= Wd * Hd * P.C) return;
    const uint c = gid % P.C, p = gid / P.C, X = p % Wd, Y = p / Wd;
    const int nx = int(X / 2), ny = int(Y / 2);
    const int fxi = clamp((X & 1) ? nx + 1 : nx - 1, 0, int(P.Ws) - 1);
    const int fyi = clamp((Y & 1) ? ny + 1 : ny - 1, 0, int(P.Hs) - 1);
    const float h0 = 0.75f * float(src[(uint(ny) * P.Ws + uint(nx)) * P.C + c]) + 0.25f * float(src[(uint(ny) * P.Ws + uint(fxi)) * P.C + c]);
    const float h1 = 0.75f * float(src[(uint(fyi) * P.Ws + uint(nx)) * P.C + c]) + 0.25f * float(src[(uint(fyi) * P.Ws + uint(fxi)) * P.C + c]);
    out[gid] = half(0.75f * h0 + 0.25f * h1);
}

kernel void up2_extrap_s(constant UpParams &P [[buffer(0)]],
                       device const half *src [[buffer(1)]],
                       device half *out [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2;
    if (gid >= Wd * Hd * P.C) return;
    const uint c = gid % P.C, p = gid / P.C, X = p % Wd, Y = p / Wd;
    uint ix0, ix1, iy0, iy1;
    float fx, fy;
    up2x_axis(X, P.Ws, ix0, ix1, fx);
    up2x_axis(Y, P.Hs, iy0, iy1, fy);
    const half h0 = lerp16(src[(iy0 * P.Ws + ix0) * P.C + c], src[(iy0 * P.Ws + ix1) * P.C + c], fx);
    const half h1 = lerp16(src[(iy1 * P.Ws + ix0) * P.C + c], src[(iy1 * P.Ws + ix1) * P.C + c], fx);
    out[gid] = lerp16(h0, h1, fy);
}

#if defined(METALFG_CTL_UP_HALF)
// Control: the half-pixel-centre 2x upsampling of the second network used in the first network's decoder, then the skip sum.
kernel void up_half_add(constant UpParams &P [[buffer(0)]], device const half *src [[buffer(1)]], device const half *skip [[buffer(2)]],
                        device half *out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint Wd = P.Ws * 2, Hd = P.Hs * 2;
    if (gid >= Wd * Hd * P.C) return;
    const uint c = gid % P.C, p = gid / P.C, X = p % Wd, Y = p / Wd;
    const int nx = int(X / 2), ny = int(Y / 2);
    const int fxi = clamp((X & 1) ? nx + 1 : nx - 1, 0, int(P.Ws) - 1);
    const int fyi = clamp((Y & 1) ? ny + 1 : ny - 1, 0, int(P.Hs) - 1);
    const float h0 = 0.75f * float(src[(uint(ny) * P.Ws + uint(nx)) * P.C + c]) + 0.25f * float(src[(uint(ny) * P.Ws + uint(fxi)) * P.C + c]);
    const float h1 = 0.75f * float(src[(uint(fyi) * P.Ws + uint(nx)) * P.C + c]) + 0.25f * float(src[(uint(fyi) * P.Ws + uint(fxi)) * P.C + c]);
    const half u = half(0.75f * h0 + 0.25f * h1);
    out[gid] = half(float(u) + float(skip[gid]));
}
#endif

struct N2Params { uint Nw, Nh; float t; };

// x0 = [CA (4), CB (4), HM, t, six zero columns]: 16 channels per pixel.
kernel void n2_pack_x0(constant N2Params &P [[buffer(0)]],
                       device const half4 *ca [[buffer(1)]],
                       device const half4 *cb [[buffer(2)]],
                       device const half *hm [[buffer(3)]],
                       device half *out [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Nw || gid.y >= P.Nh) return;
    const uint p = gid.y * P.Nw + gid.x;
    device half4 *o4 = (device half4 *)(out + p * 16);
    o4[0] = ca[p];
    o4[1] = cb[p];
    o4[2] = half4(hm[p], half(P.t), 0, 0);
    o4[3] = half4(0);
}

// Stage join sampling (rule F): the position is the binary16 sum of the binary16 pixel index and the binary16 doubled offset; the fraction is the position minus the
// binary16 floor; the four weighted terms are each rounded to binary16 and summed in the order upper-left, upper-right, lower-left, lower-right.
static inline void join_axis(uint idx, half off, thread int &i0, thread int &i1, thread float &a, uint N) {
    const half pos = half(float(half(float(idx))) + float(half(2.0f * float(off))));
    float fl = floor(float(pos));
    a = float(pos) - fl;
    if (isnan(fl)) { fl = 0.0f; a = 0.0f; }
    fl = clamp(fl, -65536.0f, 65536.0f);
    i0 = clamp(int(fl), 0, int(N) - 1);
    i1 = clamp(int(fl) + 1, 0, int(N) - 1);
}

static inline half4 gather4(device const half4 *img, uint W, uint H, uint2 gid, half2 off) {
    int x0, x1, y0, y1;
    float ax, ay;
    join_axis(gid.x, off.x, x0, x1, ax, W);
    join_axis(gid.y, off.y, y0, y1, ay, H);
    const float wx0 = 1.0f - ax, wx1 = ax, wy0 = 1.0f - ay, wy1 = ay;
    const float4 a = float4(img[uint(y0) * W + uint(x0)]), b = float4(img[uint(y0) * W + uint(x1)]);
    const float4 c = float4(img[uint(y1) * W + uint(x0)]), d = float4(img[uint(y1) * W + uint(x1)]);
    half4 s = half4(a * (wx0 * wy0));
    s = half4(float4(s) + float4(half4(b * (wx1 * wy0))));
    s = half4(float4(s) + float4(half4(c * (wx0 * wy1))));
    s = half4(float4(s) + float4(half4(d * (wx1 * wy1))));
    return s;
}

// Stage join: Z has 18 real channels (stored with a stride of 32, the rest zero).
kernel void n2_join(constant N2Params &P [[buffer(0)]],
                    device const half4 *ca [[buffer(1)]],
                    device const half4 *cb [[buffer(2)]],
                    device const half *hm [[buffer(3)]],
                    device const half *u [[buffer(4)]],      // 8 channels per pixel
                    device half *z [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Nw || gid.y >= P.Nh) return;
    const uint p = gid.y * P.Nw + gid.x;
    device const half *up = u + p * 8;
    device half4 *o4 = (device half4 *)(z + p * 32);
    o4[0] = gather4(ca, P.Nw, P.Nh, gid, half2(up[0], up[1]));
    o4[1] = gather4(cb, P.Nw, P.Nh, gid, half2(up[2], up[3]));
    o4[2] = half4(up[0], up[1], up[2], up[3]);
    o4[3] = half4(up[4], hm[p], up[5], up[6]);
    o4[4] = half4(up[7], half(P.t), 0, 0);
    for (uint i = 5; i < 8; i++) o4[i] = half4(0);
}

// N = [k0 + 2 u0 (4), k1 + u1, k2 + u2 (3)] (rule G): the binary16 sum of the head value and the binary16 product of two and the joined sample (the heads one and two
// were saturated before); the sum of two binary16 values in binary32 and one rounding equals the binary16 sum.
kernel void n2_final(constant N2Params &P [[buffer(0)]],
                     device const half *k [[buffer(1)]],
                     device const half *u [[buffer(2)]],
                     device half *out [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= P.Nw * P.Nh * 8) return;
    const uint c = gid % 8;
    const float m = c < 4 ? 2.0f : 1.0f;
    out[gid] = half(float(k[gid]) + m * float(u[gid]));
}
