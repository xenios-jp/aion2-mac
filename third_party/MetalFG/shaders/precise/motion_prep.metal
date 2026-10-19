// F05 motion preparation, F06 distortion field downsample, F12 distortion correction (spec fg.motion_prep).
#include "fgcommon.h"
using namespace metal;

constexpr sampler lin(filter::linear, address::clamp_to_edge, coord::normalized);

static inline float sat(float v) { return fmin(fmax(v, 0.0f), 1.0f); }

struct F05Params {
    uint Wm, Hm;
    uint dox, doy, dw, dh;      // depth rectangle
    uint mox, moy, mw, mh;      // motion rectangle
    float sx, sy;               // motion scale
    float mtw, mth;             // motion texture size
    uint reversed;
    float eps_n;
};

// Subrect sampling (fg.numerics): the sampling position is limited to the centres of the rectangle's edge pixels.
static inline float2 subrect_uv(float2 uv, float2 origin, float2 size, float2 texsize) {
    const float2 lo = origin + 0.5f, hi = max(origin + size - 0.5f, lo);
    return clamp(origin + uv * size, lo, hi) / texsize;
}

static inline float fetch(texture2d<float, access::read> t, uint2 p) { return t.read(p).x; }
static inline float fetch(depth2d<float, access::read> t, uint2 p) { return t.read(p); }

template <typename DT>
static void f05_body(constant F05Params &P, DT depth, texture2d<float> motion, texture2d<uint, access::write> dep_out,
                     texture2d<float, access::write> mvo_out, uint2 gid) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const int2 dim = int2(P.dw, P.dh);
    // fg.motion_prep_r2: the centre texel c satisfies c < P <= c + 1 for P = ((x + 1/2) Wz / Wm, (y + 1/2) Hz / Hm), i.e. ceil(P) - 1
    // (a point on the line between two texels belongs to the lower one); exact integer form c = ceil((2 x + 1) Wz / (2 Wm)) - 1.
    const uint2 num = (2u * gid + 1u) * uint2(P.dw, P.dh), den = 2u * uint2(P.Wm, P.Hm);
    const int2 c = clamp(int2((num + den - 1u) / den) - 1, int2(0), dim - 1);
    constexpr int2 T[5] = {int2(0, 0), int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)};
    float best_d = 0.0f, best_n = -1.0f;
    uint code = 0;
    for (uint k = 0; k < 5; k++) {
        const int2 tp = clamp(c + T[k], int2(0), dim - 1);
        const float d = fetch(depth, uint2(P.dox + uint(tp.x), P.doy + uint(tp.y)));
        const float n = fmax(P.reversed ? d : 1.0f - d, P.eps_n);     // a smaller proxy is a larger nearness
        if (k == 0 || n > best_n) { best_n = n; best_d = d; code = k; }  // ties keep the earlier tap
    }

    dep_out.write(uint4(rec_pack(sat(best_d), code), 0, 0, 0), gid);
    // the motion is read at the centre of the texel that supplied the winning depth, in normalised coordinates of the depth rectangle
    const int2 tw = clamp(c + T[code], int2(0), dim - 1);
    const float2 qd = clamp((float2(tw) + 0.5f) / float2(dim), 0.0f, 1.0f);
    const float2 q = subrect_uv(qd, float2(P.mox, P.moy), float2(P.mw, P.mh), float2(P.mtw, P.mth));
    const float2 mv = motion.sample(lin, q).xy * float2(P.sx, P.sy) / float2(P.mw, P.mh);
    mvo_out.write(float4(mv, 0, 0), gid);
}

kernel void f05_motion_prep(constant F05Params &P [[buffer(0)]], texture2d<float, access::read> depth [[texture(0)]], texture2d<float> motion [[texture(1)]],
                            texture2d<uint, access::write> dep_out [[texture(2)]], texture2d<float, access::write> mvo_out [[texture(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    f05_body(P, depth, motion, dep_out, mvo_out, gid);
}

kernel void f05_motion_prep_depth(constant F05Params &P [[buffer(0)]], depth2d<float, access::read> depth [[texture(0)]], texture2d<float> motion [[texture(1)]],
                                  texture2d<uint, access::write> dep_out [[texture(2)]], texture2d<float, access::write> mvo_out [[texture(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    f05_body(P, depth, motion, dep_out, mvo_out, gid);
}

struct F06Params {
    uint Wm, Hm;
    uint ox, oy, w, h;          // distortion rectangle
    float tw, th;               // texture size
    uint chans;                 // 2 or 4 channels carried by the texture format
};

kernel void f06_distortion(constant F06Params &P [[buffer(0)]], texture2d<float> field [[texture(0)]],
                           texture2d<float, access::write> out [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const float2 uv = (float2(gid) + 0.5f) / float2(P.Wm, P.Hm);
    float4 v = field.sample(lin, subrect_uv(uv, float2(P.ox, P.oy), float2(P.w, P.h), float2(P.tw, P.th)));
    if (P.chans < 4) v.zw = float2(0);
    out.write(v, gid);
}

struct F12Params { uint Wm, Hm; };

kernel void f12_distortion_correction(constant F12Params &P [[buffer(0)]],
                                      texture2d<float> b1 [[texture(0)]],      // bdf.mot.cur
                                      texture2d<float> b0 [[texture(1)]],      // bdf.mot.old
                                      texture2d<float> mvo [[texture(2)]],     // normalised current to previous
                                      texture2d<float> mvf [[texture(3)]],     // Mot pixels, previous to current
                                      texture2d<uint, access::read> dep [[texture(4)]],
                                      texture2d<float, access::write> mvod [[texture(5)]],
                                      texture2d<float, access::write> mvfd [[texture(6)]],
                                      texture2d<uint, access::write> dep2 [[texture(7)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const float2 sz = float2(P.Wm, P.Hm);
    const float2 u = (float2(gid) + 0.5f) / sz;
    // current to previous: enters through the current field, leaves through the previous one
    const float2 a = u + b1.sample(lin, u).xy;
    const float2 b = a + mvo.sample(lin, a).xy;
    const float2 c = b + b0.sample(lin, b).zw;
    mvod.write(float4(c - u, 0, 0), gid);
    // previous to current: the reverse
    const float2 a2 = u + b0.sample(lin, u).xy;
    const float2 b2 = a2 + mvf.sample(lin, a2).xy / sz;
    const float2 c2 = b2 + b1.sample(lin, b2).zw;
    mvfd.write(float4((c2 - u) * sz, 0, 0), gid);
    // packed depth at the undistorted position
    const int2 t = clamp(int2(floor(a * sz)), int2(0), int2(sz) - 1);
    dep2.write(dep.read(uint2(t)), gid);
}
