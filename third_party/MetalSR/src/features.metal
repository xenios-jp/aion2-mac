// Pass P1 (sr.input_features_r2): for every output pixel, the reprojected and warped history, the warped image R, the
// motion memory and four features; the features of each 2 x 2 block of output pixels are averaged and stored as one
// 8-byte record of the network input feat. One thread handles one 2 x 2 block.
#include <metal_stdlib>
#include "sr_math.h"
#include "shared_types.h"
using namespace metal;

constexpr sampler kLinear(coord::normalized, address::clamp_to_edge, filter::linear);

// motion in output pixels and reprojected position; the product and the sum are separate fp32 roundings
static inline float2 p1_reproject(int x, int y, float2 S, float2 V, thread float2 &m) {
    SR_STRICT
    m = float2(S.x * V.x, S.y * V.y);
    return float2(((float)x + 0.5f) + m.x, ((float)y + 0.5f) + m.y);
}

// exact ownership tests of the motion memory: cell i belongs to pixel x when s x < i + 1/2 <= s (x + 1). The sign of the
// fused operation is the sign of the exact difference.
static inline bool cell_le_hi(float s, int x, int i) { SR_STRICT return fma(s, (float)(x + 1), -((float)i + 0.5f)) >= 0.0f; }
static inline bool cell_gt_lo(float s, int x, int i) { SR_STRICT return fma(s, (float)x, -((float)i + 0.5f)) < 0.0f; }
// largest cell with i + 1/2 <= s (x + 1), from an estimate corrected by the exact test (-1 when there is none)
static inline int cell_highest(float s, int x) {
    int i = (int)floor(s * (float)(x + 1) - 0.5f);
    while (i >= 0 && !cell_le_hi(s, x, i)) i--;
    while (cell_le_hi(s, x, i + 1)) i++;
    return i;
}

// Catmull-Rom weights of one axis for a fractional position t, with the middle pair merged (sr.input_features_r2)
struct Cubic { float w0, w3, w12, off; };
static inline Cubic cubic_axis(float t) {
    float t2 = t * t, t3 = t2 * t;
    float w0 = t2 - 0.5f * (t + t3), w1 = 1.5f * t3 - 2.5f * t2 + 1.0f, w3 = 0.5f * (t3 - t2);
    float w2 = 1.0f - w0 - w1 - w3;
    Cubic c; c.w0 = w0; c.w3 = w3; c.w12 = w1 + w2; c.off = w2 / (w1 + w2);
    return c;
}

// sqrt(fma32(d_x, d_x, d_y d_y)): the fused operation and the rounded product are fixed by the spec
static inline float motion_length(float2 d) { SR_STRICT return sqrt(fma(d.x, d.x, d.y * d.y)); }

static inline float3 luma_limit(float3 z) { float y = sr_luma(z); return y > 1.0f ? z / y : z; }
static inline float3 tonemap_inv(float3 c, float phi) { return c / max(1.0f - sr_luma(c), phi); }
static inline float tonemap_scalar(float3 c, float phi) { float y = sr_luma(c); return y * max(1.0f / (y + 1.0f), phi); }

// One thread per output pixel (T4). The four pixels of a 2 x 2 block sit in adjacent lanes of a quad (lane = 2 b + a, the order in which the
// block's features are summed), so the block average is formed from the four lanes' values by quad shuffles. A group of 128 threads covers 16 x 8
// pixels = 8 x 4 blocks.
template <bool HDR, bool MVR, bool DINV>
static inline void p1_body(texture2d<float> colour, texture2d<float> depth, texture2d<float> motion, texture2d<float> hist, texture2d<float> mem_read,
                           texture2d<float> e_now_t, texture2d<float> e_prev_t, texture2d<half, access::write> warped, texture2d<half, access::write> mem_write,
                           device half *feat, constant FeatParams &p, uint2 tg, uint lin) {
    const uint sub = lin & 3u, q = lin >> 2;
    const int I = (int)(tg.x * 8 + (q & 7u)), J = (int)(tg.y * 4 + (q >> 3));
    const int a = (int)(sub & 1u), b = (int)(sub >> 1);
    const int x = 2 * I + a, y = 2 * J + b;
    const int Wr = (int)p.Wr, Hr = (int)p.Hr, Wo = (int)p.Wo, Ho = (int)p.Ho;
    const bool inside = x < Wo && y < Ho;
    float4 f = float4(0.0f);

    if (inside) {
        float en = e_now_t.read(uint2(0)).x, ep = e_prev_t.read(uint2(0)).x;
        en = en == 0.0f ? 1.0f : en; ep = ep == 0.0f ? 1.0f : ep;
        float k = p.invp * en;
        k = k == 0.0f ? 1.0f : k;
        const float r_e = en / ep;

        // nearest render sample
        const float u = sr_coord_a(p.sx, p.jx, x), v = sr_coord_a(p.sy, p.jy, y);
        const int2 n = int2(clamp((int)sr_round(u), 0, Wr - 1), clamp((int)sr_round(v), 0, Hr - 1));
        const float3 C = sr_san3(colour.read(uint2(n)).rgb);

        // motion vector
        float2 V;
        if (MVR) {
            const int2 cand[4] = {int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)};
            float dv[5];
            dv[0] = sr_san(depth.read(uint2(n)).x);
            #pragma unroll
            for (int d = 0; d < 4; d++)
#ifdef SR_ABLATE_P1_DEPTH
                dv[d + 1] = dv[0] + (float)d;
#else
                dv[d + 1] = sr_san(depth.read(uint2(clamp(n.x + cand[d].x, 0, Wr - 1), clamp(n.y + cand[d].y, 0, Hr - 1))).x);
#endif
            int2 best = int2(0);
            float bd = dv[0];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                const float dd = dv[d + 1];
                if (DINV ? dd > bd : dd < bd) { bd = dd; best = cand[d]; }
            }
            const float4 mv = motion.read(uint2(clamp(n.x + best.x, 0, Wr - 1), clamp(n.y + best.y, 0, Hr - 1)));
            V = float2(sr_san(mv.x), sr_san(mv.y));
        } else {
            float4 mv = motion.read(uint2(x, y));
            V = float2(sr_san(mv.x), sr_san(mv.y));
        }
        float2 m;
        const float2 P = p1_reproject(x, y, float2(p.Sx, p.Sy), V, m);

        // the previous motion at the reprojected position depends on P only: it is fetched together with the history samples (T4)
        float2 m_prev = m;
#ifndef SR_ABLATE_P1_MEM
        if (!p.reset) {
            const float2 Qr = clamp(float2(P.x * p.sx, P.y * p.sy), float2(0.5f), float2((float)Wr - 0.5f, (float)Hr - 0.5f));
            m_prev = mem_read.sample(kLinear, Qr * float2(p.inv_wo, p.inv_ho)).xy;
        }
#endif

        // current colour and its feature
        float3 craw = C, c = C;
        if (HDR) { craw = k * C; c = craw; } else { c = tonemap_inv(C, p.phi_hi); }
        const float F0 = tonemap_scalar(c, p.phi_lo);

        // warped history
        const bool valid = !p.reset && P.x >= 0.0f && P.x < (float)Wo && P.y >= 0.0f && P.y < (float)Ho;
        float3 Rv = max(craw, 0.0f);
        float a_h = 0.0f;
        if (valid) {
            const float2 Pc = clamp(P, float2(0.5f), float2((float)Wo - 0.5f, (float)Ho - 0.5f));
            const float2 bpos = floor(Pc - 0.5f) + 0.5f;
            const float2 t = clamp(Pc - bpos, 0.0f, 1.0f);
            const Cubic cx = cubic_axis(t.x), cy = cubic_axis(t.y);
            const float2 inv = float2(p.inv_wo, p.inv_ho);
#ifdef SR_ABLATE_P1_HIST
            const float4 sM = hist.sample(kLinear, float2(bpos.x + cx.off, bpos.y + cy.off) * inv), sL = sM * 0.9f, sR = sM * 0.8f, sT = sM * 0.7f, sB = sM * 0.6f;
#else
            const float4 sL = hist.sample(kLinear, float2(bpos.x - 1.0f, bpos.y + cy.off) * inv);
            const float4 sR = hist.sample(kLinear, float2(bpos.x + 2.0f, bpos.y + cy.off) * inv);
            const float4 sT = hist.sample(kLinear, float2(bpos.x + cx.off, bpos.y - 1.0f) * inv);
            const float4 sB = hist.sample(kLinear, float2(bpos.x + cx.off, bpos.y + 2.0f) * inv);
            const float4 sM = hist.sample(kLinear, float2(bpos.x + cx.off, bpos.y + cy.off) * inv);
#endif
            const float WL = cx.w0 * cy.w12, WR = cx.w3 * cy.w12, WT = cy.w0 * cx.w12, WB = cy.w3 * cx.w12, WM = cx.w12 * cy.w12;
            const float Sn = 1.0f - (cx.w0 + cx.w3) * (cy.w0 + cy.w3);
            const float3 lm = luma_limit(sM.rgb);
            float3 N = Sn * sM.rgb + WL * (luma_limit(sL.rgb) - lm) + WR * (luma_limit(sR.rgb) - lm) + WT * (luma_limit(sT.rgb) - lm) + WB * (luma_limit(sB.rgb) - lm);
            float3 h = N / Sn;
            a_h = (WL * sL.a + WR * sR.a + WT * sT.a + WB * sB.a + WM * sM.a) / Sn;
            if (HDR) h = r_e * tonemap_inv(h, p.phi_hi);
            Rv = max(h, 0.0f);
        }
#ifdef SR_ABLATE_P1_WARPED
        if (Rv.x == 12345.0f)
#endif
        warped.write(half4(sr_rz16(Rv.x), sr_rz16(Rv.y), sr_rz16(Rv.z), (half)0.0f), uint2(x, y));

        // features
        const float3 e = HDR ? Rv : tonemap_inv(Rv, p.phi_hi);
        const float F1 = tonemap_scalar(e, p.phi_lo);
        float F2 = valid ? a_h : F0;
        F2 = sr_san(F2);
        const float2 d = m - m_prev;
#ifdef SR_ABLATE_P1_F3
        const float F3 = d.x + d.y;
#else
        const float F3 = sr_san(sr_log1p(p.g_mv * motion_length(d)));
#endif
        f = float4(fmax(fmin(F0, 1.0f), 0.0f), fmax(fmin(F1, 1.0f), 0.0f), F2, F3);

        // motion memory: the render-grid cells owned by this pixel
        const int ix = cell_highest(p.sx, x), iy = cell_highest(p.sy, y);
        const half4 mw = half4((half)m.x, (half)m.y, (half)0.0f, (half)0.0f);
#ifndef SR_ABLATE_P1_MEM
        if (p.sx <= 1.0f && p.sy <= 1.0f) {
            // render no larger than output (the size contract): the interval of a pixel holds at most one cell per axis, so the loops below
            // reduce to one test of the highest candidate and at most one write
            if (ix >= 0 && iy >= 0 && ix < Wr && iy < Hr && cell_gt_lo(p.sx, x, ix) && cell_gt_lo(p.sy, y, iy)) mem_write.write(mw, uint2(ix, iy));
        } else {
            for (int j = iy; j >= 0 && cell_gt_lo(p.sy, y, j); j--) {
                if (j >= Hr) continue;
                for (int i = ix; i >= 0 && cell_gt_lo(p.sx, x, i); i--)
                    if (i < Wr) mem_write.write(mw, uint2(i, j));
            }
        }
#else
        if (ix + iy == 12345) mem_write.write(mw, uint2(0, 0));
#endif
    }

    // the block average: the features of the pixels inside the output, summed in the order of lanes 0 to 3 from zero
    float4 sum = float4(0.0f);
    for (uint s = 0; s < 4; s++) {
        const float4 fs = quad_shuffle(f, (ushort)s);
        const int xs = 2 * I + (int)(s & 1u), ys = 2 * J + (int)(s >> 1);
        if (xs < Wo && ys < Ho) sum += fs;
    }
    if (sub == 0 && I < (int)((p.Wo + 1) / 2) && J < (int)((p.Ho + 1) / 2)) {
        const half4 rec = half4(sum * 0.25f);
        *(device half4 *)(feat + ((size_t)((uint)J >> 1) * p.W0 + ((uint)I >> 1)) * 16 + 8 * (J & 1) + 4 * (I & 1)) = rec;
    }
}

// T1: the colour mode, the motion-vector resolution and the depth direction are fixed at create; each combination is a kernel
#define P1_KERNEL(NAME, HDR, MVR, DINV) \
kernel void NAME(texture2d<float> colour [[texture(0)]], texture2d<float> depth [[texture(1)]], texture2d<float> motion [[texture(2)]], \
                 texture2d<float> hist [[texture(3)]], texture2d<float> mem_read [[texture(4)]], \
                 texture2d<float> e_now_t [[texture(5)]], texture2d<float> e_prev_t [[texture(6)]], \
                 texture2d<half, access::write> warped [[texture(7)]], texture2d<half, access::write> mem_write [[texture(8)]], \
                 device half *feat [[buffer(0)]], constant FeatParams &p [[buffer(1)]], uint2 tg [[threadgroup_position_in_grid]], uint lin [[thread_index_in_threadgroup]]) { \
    p1_body<HDR, MVR, DINV>(colour, depth, motion, hist, mem_read, e_now_t, e_prev_t, warped, mem_write, feat, p, tg, lin); \
}
P1_KERNEL(p1_features_000, false, false, false)
P1_KERNEL(p1_features_001, false, false, true)
P1_KERNEL(p1_features_010, false, true, false)
P1_KERNEL(p1_features_011, false, true, true)
P1_KERNEL(p1_features_100, true, false, false)
P1_KERNEL(p1_features_101, true, false, true)
P1_KERNEL(p1_features_110, true, true, false)
P1_KERNEL(p1_features_111, true, true, true)
