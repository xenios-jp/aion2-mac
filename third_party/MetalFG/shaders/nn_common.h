// Interpolation helpers shared by the network kernels (nn.metal) and the matrix kernels that fold an enlargement into their input loads (nn_mx.metal), so that both
// evaluate the same expressions (fg.arithmetic_pin_r4 rules E and H). Contraction is off: every product and sum is rounded where written.
#ifndef METALFG_NN_COMMON_H
#define METALFG_NN_COMMON_H
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

// The interpolation step on four channels with the weights of the pixel formed once: weights (1 - f) and f rounded to binary16, each product rounded, the sum rounded.
static inline half4 lerp16v(half4 a, half4 b, half w0, half w1) {
    const half4 t0 = half4(float4(a) * float(w0)), t1 = half4(float4(b) * float(w1));
    return half4(float4(t0) + float4(t1));
}

// Net1 (rule H): channels 4 c .. 4 c + 3 of the corner-aligned bilinear x2 enlargement of src (Ws x Hs, C4 groups of four channels per pixel) at destination
// pixel (X, Y). The coordinate scale (Ws - 1) / (Wd - 1) is the binary32 quotient of fast::divide; coordinates are binary32; the weights and both interpolation
// steps are binary16, horizontal first.
static inline half4 up_corner4(device const half4 *src, uint Ws, uint Hs, uint C4, uint X, uint Y, uint c) {
    const uint Wd = Ws * 2, Hd = Hs * 2;
    const float scx = Wd > 1 ? fast::divide(float(Ws - 1), float(Wd - 1)) : 0.0f;
    const float scy = Hd > 1 ? fast::divide(float(Hs - 1), float(Hd - 1)) : 0.0f;
    const float sx = float(X) * scx, sy = float(Y) * scy;
    const uint ix0 = min(uint(max(floor(sx), 0.0f)), Ws - 1), iy0 = min(uint(max(floor(sy), 0.0f)), Hs - 1);
    const uint ix1 = min(ix0 + 1, Ws - 1), iy1 = min(iy0 + 1, Hs - 1);
    const float fx = sx - float(ix0), fy = sy - float(iy0);
    const half fxw1 = half(fx), fxw0 = half(1.0f - fx), fyw1 = half(fy), fyw0 = half(1.0f - fy);
    const half4 h0 = lerp16v(src[(iy0 * Ws + ix0) * C4 + c], src[(iy0 * Ws + ix1) * C4 + c], fxw0, fxw1);
    const half4 h1 = lerp16v(src[(iy1 * Ws + ix0) * C4 + c], src[(iy1 * Ws + ix1) * C4 + c], fxw0, fxw1);
    return lerp16v(h0, h1, fyw0, fyw1);
}

// The skip sum of the decoder: formed in binary32 and rounded once.
static inline half4 up_add4(half4 u, half4 skip) { return half4(float4(u) + float4(skip)); }
#endif
