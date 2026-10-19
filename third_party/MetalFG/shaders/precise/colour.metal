// F04 colour reduction and UI estimate, F13 disocclusion scatter, F14 warp-error seed (spec fg.colour_disocclusion).
#include "fgcommon.h"

enum { MODE_P = 0, MODE_D = 1, MODE_A = 2, MODE_R = 3 };

struct F04Params {
    uint W, H;
    uint ox, oy;          // colour rectangle origin (hudless and coverage share it)
    uint mode;
    uint ui;              // create bit 0
    uint cov_from_colour; // coverage from the alpha channel of ui_colour instead of the ui_alpha plane
    uint df;
    float th_ui, th_uia;
};

struct PixelEval {
    float3 S;
    float alpha;
    float4 L;
    bool flag;
};

static PixelEval eval_pixel(constant F04Params &P, texture2d<float, access::read> C, texture2d<float, access::read> Hl, texture2d<float, access::read> A, uint2 p) {
    PixelEval r;
    const uint2 q = uint2(P.ox, P.oy) + p;
    const float3 c = C.read(q).rgb;
    r.S = c; r.alpha = 0.0f; r.L = float4(0.0f); r.flag = false;
    float a = 0.0f;
    if (P.mode == MODE_A || P.mode == MODE_R) {
        const float4 av = A.read(q);
        a = P.cov_from_colour ? av.w : av.x;
    }
    if (P.mode == MODE_R) {
        const float3 h = Hl.read(q).rgb;
        r.S = h; r.alpha = a;
        const float3 E = c + (a - 1.0f) * h;
        const bool bad = !all(isfinite(E)) || any(E <= -1.0f) || a == 1.0f || any(E < c - 2.0f * h);
        r.L = bad ? float4(c, 1.0f) : float4(E, a);
    } else if (P.mode == MODE_A) {
        r.alpha = a;
        r.flag = a > P.th_uia;
    } else if (P.mode == MODE_D) {
        const float3 h = Hl.read(q).rgb;
        const float3 df = abs(c - h);
        r.flag = fmax(fmax(df.x, df.y), df.z) > P.th_ui;
    }
    return r;
}

kernel void f04_full(constant F04Params &P [[buffer(0)]], texture2d<float, access::read> C [[texture(0)]], texture2d<float, access::read> Hl [[texture(1)]],
                     texture2d<float, access::read> A [[texture(2)]], texture2d<float, access::write> colCur [[texture(3)]],
                     texture2d<float, access::write> uiCur [[texture(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.W || gid.y >= P.H) return;
    const PixelEval e = eval_pixel(P, C, Hl, A, gid);
    colCur.write(float4(e.S, e.alpha), gid);
    if (P.ui) uiCur.write(e.L, gid);
}

kernel void f04_reduce(constant F04Params &P [[buffer(0)]], texture2d<float, access::read> C [[texture(0)]], texture2d<float, access::read> Hl [[texture(1)]],
                       texture2d<float, access::read> A [[texture(2)]], texture2d<float, access::read> colOld [[texture(3)]],
                       texture2d<float, access::write> colFeat [[texture(4)]], texture2d<float, access::write> uim [[texture(5)]],
                       texture2d<float, access::write> dY [[texture(6)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= colFeat.get_width() || gid.y >= colFeat.get_height()) return;
    const uint df = P.df;
    float3 sum = float3(0.0f);
    bool flag = false;
    float lam[16];
    for (uint j = 0; j < df; j++)
        for (uint i = 0; i < df; i++) {
            const uint2 p = uint2(gid.x * df + i, gid.y * df + j);
            const PixelEval e = eval_pixel(P, C, Hl, A, p);
            sum += e.S;
            flag = flag || e.flag;
            const float yn = sat1(lum3(e.S)), yo = sat1(lum3(colOld.read(p).rgb));
            lam[j * df + i] = fabs(yn - yo);
        }
    float d;
    if (df == 2) {
        d = fmin(fmin(lam[0], lam[1]), fmin(lam[2], lam[3]));
    } else {
        d = 0.0f;
        for (uint qy = 0; qy < 2; qy++)
            for (uint qx = 0; qx < 2; qx++) {
                const uint b = (qy * 2) * 4 + qx * 2;
                d += fmin(fmin(lam[b], lam[b + 1]), fmin(lam[b + 4], lam[b + 5]));
            }
        d = d / 4.0f;
    }
    colFeat.write(float4(sum / float(df * df), 1.0f), gid);
    uim.write(float4(flag ? 1.0f : 0.0f, 0, 0, 0), gid);
    dY.write(float4(d, 0, 0, 0), gid);
}


// Fused F04 (same values as f04_full followed by f04_reduce, the colour read once): one thread per block of df x df positions evaluates each position, stores its
// colour and UI estimate (positions inside the display only) and accumulates the reduction as f04_reduce does. Used when the blocks cover the display.
kernel void f04_fused(constant F04Params &P [[buffer(0)]], texture2d<float, access::read> C [[texture(0)]], texture2d<float, access::read> Hl [[texture(1)]],
                      texture2d<float, access::read> A [[texture(2)]], texture2d<float, access::read> colOld [[texture(3)]],
                      texture2d<float, access::write> colFeat [[texture(4)]], texture2d<float, access::write> uim [[texture(5)]],
                      texture2d<float, access::write> dY [[texture(6)]], texture2d<float, access::write> colCur [[texture(7)]],
                      texture2d<float, access::write> uiCur [[texture(8)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= colFeat.get_width() || gid.y >= colFeat.get_height()) return;
    const uint df = P.df;
    float3 sum = float3(0.0f);
    bool flag = false;
    float lam[16];
    for (uint j = 0; j < df; j++)
        for (uint i = 0; i < df; i++) {
            const uint2 p = uint2(gid.x * df + i, gid.y * df + j);
            const PixelEval e = eval_pixel(P, C, Hl, A, p);
            if (p.x < P.W && p.y < P.H) {
                colCur.write(float4(e.S, e.alpha), p);
                if (P.ui) uiCur.write(e.L, p);
            }
            sum += e.S;
            flag = flag || e.flag;
            const float yn = sat1(lum3(e.S)), yo = sat1(lum3(colOld.read(p).rgb));
            lam[j * df + i] = fabs(yn - yo);
        }
    float d;
    if (df == 2) {
        d = fmin(fmin(lam[0], lam[1]), fmin(lam[2], lam[3]));
    } else {
        d = 0.0f;
        for (uint qy = 0; qy < 2; qy++)
            for (uint qx = 0; qx < 2; qx++) {
                const uint b = (qy * 2) * 4 + qx * 2;
                d += fmin(fmin(lam[b], lam[b + 1]), fmin(lam[b + 4], lam[b + 5]));
            }
        d = d / 4.0f;
    }
    colFeat.write(float4(sum / float(df * df), 1.0f), gid);
    uim.write(float4(flag ? 1.0f : 0.0f, 0, 0, 0), gid);
    dY.write(float4(d, 0, 0, 0), gid);
}

// Tiled F04 (the values of f04_fused): one thread per position, so that neighbouring lanes read and write neighbouring texels; the per-position colour, flag and
// luminance change are staged in a 16 x 16 tile and one thread per block of df x df positions reduces them in the order of f04_fused.
// The reduction of one block of DF x DF positions staged in a 16 x 16 tile (the order of f04_fused: raster order sums, the minimum of each 2 x 2 quarter).
template <uint DF>
static inline void f04_block(threadgroup const float3 *tS, threadgroup const uchar *tF, threadgroup const float *tL, uint2 lid,
                             thread float3 &sum, thread bool &flag, thread float &d) {
    sum = float3(0.0f);
    flag = false;
    float lam[DF * DF];
    for (uint j = 0; j < DF; j++)
        for (uint i = 0; i < DF; i++) {
            const uint k = (lid.y * DF + j) * 16 + lid.x * DF + i;
            sum += tS[k];
            flag = flag || tF[k] != 0;
            lam[j * DF + i] = tL[k];
        }
    if (DF == 2) {
        d = fmin(fmin(lam[0], lam[1]), fmin(lam[2], lam[3]));
    } else {
        d = 0.0f;
        for (uint qy = 0; qy < 2; qy++)
            for (uint qx = 0; qx < 2; qx++) {
                const uint b = (qy * 2) * DF + qx * 2;
                d += fmin(fmin(lam[b], lam[b + 1]), fmin(lam[b + DF], lam[b + DF + 1]));
            }
        d = d / 4.0f;
    }
}

kernel void f04_tiled(constant F04Params &P [[buffer(0)]], texture2d<float, access::read> C [[texture(0)]], texture2d<float, access::read> Hl [[texture(1)]],
                      texture2d<float, access::read> A [[texture(2)]], texture2d<float, access::read> colOld [[texture(3)]],
                      texture2d<float, access::write> colFeat [[texture(4)]], texture2d<float, access::write> uim [[texture(5)]],
                      texture2d<float, access::write> dY [[texture(6)]], texture2d<float, access::write> colCur [[texture(7)]],
                      texture2d<float, access::write> uiCur [[texture(8)]], uint2 gid [[thread_position_in_grid]],
                      uint2 lid [[thread_position_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]], uint2 tgs [[threads_per_threadgroup]]) {
    threadgroup float3 tS[256];
    threadgroup float tL[256];
    threadgroup uchar tF[256];
    const uint df = P.df;
    const PixelEval e = eval_pixel(P, C, Hl, A, gid);
    if (gid.x < P.W && gid.y < P.H) {
        colCur.write(float4(e.S, e.alpha), gid);
        if (P.ui) uiCur.write(e.L, gid);
    }
    const uint li = lid.y * 16 + lid.x;
    const float yn = sat1(lum3(e.S)), yo = sat1(lum3(colOld.read(gid).rgb));
    tS[li] = e.S;
    tF[li] = e.flag ? 1 : 0;
    tL[li] = fabs(yn - yo);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.x * df >= tgs.x || lid.y * df >= tgs.y) return;
    // the block reduction of f04_fused, with the block size known at compile time (df is 2 or 4)
    float3 sum;
    bool flag;
    float d;
    if (df == 2) f04_block<2>(tS, tF, tL, lid, sum, flag, d);
    else f04_block<4>(tS, tF, tL, lid, sum, flag, d);
    const uint2 gb = tgp * (16 / df) + lid;
    colFeat.write(float4(sum / float(df * df), 1.0f), gb);
    uim.write(float4(flag ? 1.0f : 0.0f, 0, 0, 0), gb);
    dY.write(float4(d, 0, 0, 0), gb);
}

struct F13Params {
    uint Wo, Ho, Wm, Hm;
    uint normalised;
};

// The output image is prefilled with one; this pass only writes zeros.
kernel void f13_scatter(constant F13Params &P [[buffer(0)]], texture2d<float> G [[texture(0)]],
                        texture2d<float, access::write> O [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wo || gid.y >= P.Ho) return;
    const float2 uv = (float2(gid) + 0.5f) / float2(P.Wo, P.Ho);
    const float2 g = G.sample(lin_clamp, uv).xy;
    const float2 d = P.normalised ? g * float2(P.Wo, P.Ho) : g / float2(P.Wm, P.Hm) * float2(P.Wo, P.Ho);
    if (!all(isfinite(d)) || any(abs(d) > 1.0e6f)) return;
    const int2 off = int2(trunc(d));
    const int2 s = int2(sigma1(d.x), sigma1(d.y));
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const int2 c = int2(gid) + off + int2(i * s.x, j * s.y);
            if (c.x >= 0 && c.y >= 0 && c.x < int(P.Wo) && c.y < int(P.Ho)) O.write(float4(0.0f), uint2(c));
        }
}

kernel void f14_seed(texture2d<float, access::read> colFeat [[texture(0)]], texture2d<float, access::write> err [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= err.get_width() || gid.y >= err.get_height()) return;
    const float3 c = colFeat.read(gid).rgb;
    err.write(float4(sat1((c.x + c.y + c.z) / 3.0f), 0.0f, 0, 0), gid);
}
