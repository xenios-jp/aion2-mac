// Helpers shared by the motion-chain kernels (spec fg.motion_prep notation, fg.numerics).
#ifndef METALFG_COMMON_H
#define METALFG_COMMON_H
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constexpr sampler lin_clamp(filter::linear, address::clamp_to_edge, coord::normalized);

static inline float sat1(float v) { return fmin(fmax(v, 0.0f), 1.0f); }
static inline float sigma1(float v) { return v > 0.0f ? 1.0f : -1.0f; }
// Luminance (fg.arithmetic_pin_r4 rule J): the green term first, the red term fused into it, then the blue term fused into that.
static inline float lum3(float3 c) { return fma(0.114f, c.z, fma(0.299f, c.x, 0.587f * c.y)); }

// Tap offsets of the depth dilation (fg.motion_prep): the tap code is the index.
constant int2 TAPS[5] = {int2(0, 0), int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)};

// Packed depth record: fp32 depth bits with the three lowest mantissa bits replaced by the tap code.
// Scatter keys. Release: the key of a surface is its depth bits (smaller wins). METALFG_CTL_DEPTH_FLIP resolves every conflict with the
// opposite depth order (the farther source wins): the depth bits are mirrored inside the range of finite non-negative floats.
static inline uint depth_key(uint bits) {
#if defined(METALFG_CTL_DEPTH_FLIP)
    return 0x7F800000u - min(bits, 0x7F800000u);
#else
    return bits;
#endif
}
static inline uint depth_unkey(uint key) {
#if defined(METALFG_CTL_DEPTH_FLIP)
    return 0x7F800000u - key;
#else
    return key;
#endif
}

// Depth record (fg.interm_motion_pin_r4, rule A): the tap selector (0..4) is stored in the top three bits (sign, top exponent bit and the next exponent bit);
// the lower 29 bits hold the depth's own bits. A depth in the unit range has the top two bits zero and the third bit one from 2^-63 upward, so the third bit is
// restored to one; a record whose lower 29 bits are zero is the depth zero. A positive depth below 2^-62 is not representable and is stored as zero.
constant uint REC_LOW = 0x1FFFFFFFu;
static inline uint rec_pack(float d, uint code) {
    const uint b = as_type<uint>(d);
    const uint low = (d >= 0x1p-62f) ? (b & REC_LOW) : 0u;
    return (code << 29) | low;
}
static inline uint rec_code(uint rec) { return rec >> 29; }
static inline float rec_depth(uint rec) {
    const uint low = rec & REC_LOW;
    return as_type<float>(low == 0u ? 0u : (low | 0x20000000u));
}
// Position q of the depth record of the Mot pixel gid (fg.motion_prep_r2): the centre of the texel c + T_code, c = ceil((x + 1/2) Wz / Wm) - 1 (integer form),
// clamped to the depth rectangle; the record's offset from uv(p) is q - uv(p).
static inline float2 rec_pos(uint rec, uint2 gid, uint2 mot, float2 depth_dim) {
    const uint2 dd = uint2(depth_dim);
    const int2 c = clamp(int2(((2u * gid + 1u) * dd + 2u * mot - 1u) / (2u * mot)) - 1, int2(0), int2(dd) - 1);
    const int2 t = clamp(c + TAPS[min(rec_code(rec), 4u)], int2(0), int2(dd) - 1);
    return clamp((float2(t) + 0.5f) / depth_dim, 0.0f, 1.0f);
}

struct DepthConst {
    float Zs, Zp, Zsep, eps_n;
    uint reversed;
};

static inline float lambda_of(constant DepthConst &D, float d) {
    const float n = fmax(D.reversed ? d : 1.0f - d, D.eps_n);
    return D.Zs / n;
}

// Camera displacement in Mot pixels (fg.prev2curr). M is a row-major 4 x 4 in the row-vector convention.
// w <= 0 returns a large vector (4096, 4096) and the caller sees w.
static inline float2 cam_disp(float2 u, float d, constant float *M, float2 mot, thread float &w) {
    const float nx = 2.0f * u.x - 1.0f, ny = 1.0f - 2.0f * u.y;
    const float cx = nx * M[0] + ny * M[4] + d * M[8] + M[12];
    const float cy = nx * M[1] + ny * M[5] + d * M[9] + M[13];
    w = nx * M[3] + ny * M[7] + d * M[11] + M[15];
    if (w > 0.0f) return float2((cx / w - nx) * mot.x / 2.0f, (ny - cy / w) * mot.y / 2.0f);
    return float2(4096.0f, 4096.0f);
}

// Safe logistic (fg.numerics).
static inline float sig_safe(float v) {
    if (v >= 0.0f) return 1.0f / (1.0f + exp(-v));
    const float e = exp(v);
    return e / (1.0f + e);
}

#endif
