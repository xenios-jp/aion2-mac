// F18 projection of the two motion fields to the intermediate time, F19 blend weights (spec fg.interm_motion).
// Scatter cell (8 bytes, little endian words): word 0 = payload, word 1 = key; both all ones when empty.
#include "fgcommon.h"

constant uint EMPTY = 0xFFFFFFFFu;     // initial pattern of every scatter word (fg.interm_motion_pin_r4, rule D)

struct F18Params {
    uint Wm, Hm, mode;       // mode 0: key pass, 1: payload pass
    uint use_zn;             // 1: the scale-distance admission applies (projection of the current-call field only)
    float hsx, hsy;          // h = field.xy * hs, in Mot pixels
    float H;
    float d_sim, th_zn, Z_sep;
    int sp_min, sp_max;
    DepthConst D;
    uint y0;                 // first row of this dispatch (the scatter runs in row slices woven into the first network)
};

static inline void offer(device atomic_uint *sc, uint ci, uint key, uint payload, uint mode) {
#if !defined(METALFG_CTL_KEY_RECOMPUTE)
    // One pass: the cell is the little-endian pair (payload, key), so the 64-bit minimum of (key << 32 | payload) is the smallest key and, among equal keys, the smallest payload.
    atomic_min_explicit((device atomic_ulong *)(sc + 2 * ci), (ulong(key) << 32) | ulong(payload), memory_order_relaxed);
#else
    if (mode == 0) {
        atomic_fetch_min_explicit(&sc[2 * ci + 1], key, memory_order_relaxed);
    } else if (atomic_load_explicit(&sc[2 * ci + 1], memory_order_relaxed) == key) {
        uint cur = atomic_load_explicit(&sc[2 * ci], memory_order_relaxed);
        while (cur == EMPTY || payload < cur) {
            if (atomic_compare_exchange_weak_explicit(&sc[2 * ci], &cur, payload, memory_order_relaxed, memory_order_relaxed)) break;
        }
    }
#endif
}

static inline float2 raw(constant F18Params &P, texture2d<float, access::read> field, int2 q) {
#if defined(METALFG_CTL_ZERO_BORDER)
    if (any(q < 0) || any(q >= int2(P.Wm, P.Hm))) return float2(0.0f);
#endif
    return field.read(uint2(clamp(q, int2(0), int2(P.Wm - 1, P.Hm - 1)))).xy;
}

static inline float2 hat(constant F18Params &P, texture2d<float, access::read> field, int2 q) {
#if defined(METALFG_CTL_ZERO_BORDER)
    if (any(q < 0) || any(q >= int2(P.Wm, P.Hm))) return float2(0.0f);   // control: zero motion outside the grid
#endif
    const int2 c = clamp(q, int2(0), int2(P.Wm - 1, P.Hm - 1));
    return field.read(uint2(c)).xy * float2(P.hsx, P.hsy);
}

static inline float kat(constant F18Params &P, texture2d<uint, access::read> dep, int2 q) {
#if defined(METALFG_CTL_ZERO_BORDER)
    if (any(q < 0) || any(q >= int2(P.Wm, P.Hm))) return lambda_of(P.D, 0.0f);   // control: zero depth record outside the grid
#endif
    const int2 c = clamp(q, int2(0), int2(P.Wm - 1, P.Hm - 1));
    // the scale of the cell is the depth scale times the (fast) reciprocal of the clamped nearness; its bit pattern is the sort key (fg.interm_motion_pin_r4, rule B)
    const float d = rec_depth(dep.read(uint2(c)).x);
    const float n = fmax(P.D.reversed ? d : 1.0f - d, P.D.eps_n);
#if defined(METALFG_CTL_KEY_RECOMPUTE)
    if (P.mode == 1) return P.D.Zs / n;       // control: the payload pass recomputes the key with the correctly rounded division
#endif
    return P.D.Zs * fast::divide(1.0f, n);
}

kernel void f18_scatter(constant F18Params &P [[buffer(0)]], device atomic_uint *sc [[buffer(1)]], texture2d<float, access::read> field [[texture(0)]],
                        texture2d<uint, access::read> dep [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
    // The 3 x 3 neighbourhood of the analysis (field and key scale) is read into registers.
    const uint2 gid = uint2(g.x, g.y + P.y0);
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const int2 p = int2(gid);
    float3 nb[3][3];
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) { const int2 q = p + int2(dx, dy); nb[dy + 1][dx + 1] = float3(raw(P, field, q), kat(P, dep, q)); }
    const float3 tp = nb[1][1];
    const float2 hs = float2(P.hsx, P.hsy);
    const float2 hp = tp.xy * hs;
    const float kp = tp.z;
    float2 maxdev = float2(0.0f);
    float spread = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            const float3 tq = nb[dy + 1][dx + 1];
            maxdev = fmax(maxdev, abs(fma(tq.xy, hs, float2(dx, dy)) - hp));   // the fused form of the former hat() sum
            spread = fmax(spread, fabs(tq.z - kp));
        }
    int rx, ry;
    if (spread > P.Z_sep) { rx = ry = P.sp_min; }
    else {
        rx = int(fmin(fmax(ceil(maxdev.x + 0.5f), float(P.sp_min)), float(P.sp_max)));
        ry = int(fmin(fmax(ceil(maxdev.y + 0.5f), float(P.sp_min)), float(P.sp_max)));
    }
    const int2 s = int2(sigma1(hp.x), sigma1(hp.y));
    const int2 t0 = int2(trunc(hp));
    const float thr = fmax(1.0f, dot(hp, hp) * P.d_sim / (P.H * P.H));
    const float2 o = -hp / float2(P.Wm, P.Hm);
    const uint key = depth_key(as_type<uint>(kp));
    const uint payload = (uint(as_type<ushort>(half(o.x))) << 16) | uint(as_type<ushort>(half(o.y)));
    if (rx <= 2 && ry <= 2) {
        // footprints of at most 2 x 2 (the usual case): every source q lies in the 3 x 3 neighbourhood, so the staged tile holds its field value and key scale
        // (the values hat() and kat() read for a q inside the grid); no texture is read in this loop
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) {
                if (i >= rx || j >= ry) continue;
                const int2 q = p + int2(i * s.x, j * s.y);
                const int2 T = q + t0;
                if (T.x < 0 || T.y < 0 || T.x >= int(P.Wm) || T.y >= int(P.Hm)) continue;
                if (q.x < 0 || q.y < 0 || q.x >= int(P.Wm) || q.y >= int(P.Hm)) continue;
                const float3 tq = nb[1 + j * s.y][1 + i * s.x];
                bool take = (i == 0 && j == 0);
                if (!take) {
                    const float2 dq = tq.xy * hs - hp;
                    take = dot(dq, dq) < thr;
                }
                if (!take && P.use_zn) {
                    const float tq2 = fmin(trunc(tq.z), 4294967295.0f);
                    take = fabs(tq2 - kp) < P.th_zn;
                }
                if (take) offer(sc, uint(T.y) * P.Wm + uint(T.x), key, payload, P.mode);
            }
        return;
    }
    for (int j = 0; j < ry; j++)
        for (int i = 0; i < rx; i++) {
            const int2 q = p + int2(i * s.x, j * s.y);
            const int2 T = q + t0;
            if (T.x < 0 || T.y < 0 || T.x >= int(P.Wm) || T.y >= int(P.Hm)) continue;
            if (q.x < 0 || q.y < 0 || q.x >= int(P.Wm) || q.y >= int(P.Hm)) continue;
            bool take = (i == 0 && j == 0);
            if (!take) {
                const float2 dq = hat(P, field, q) - hp;
                take = dot(dq, dq) < thr;
            }
            if (!take && P.use_zn) {
                const float kq = kat(P, dep, q);
                const float tq = fmin(trunc(kq), 4294967295.0f);
                take = fabs(tq - kp) < P.th_zn;
            }
            if (take) offer(sc, uint(T.y) * P.Wm + uint(T.x), key, payload, P.mode);
        }
}

// ---- F19 ----------------------------------------------------------------------------------------------

struct F19Params {
    uint Wm, Hm, Wf, Hf;
};

static inline float2 decode_offset(device const uint2 *sc, uint ci) {
    const uint2 e = sc[ci];
    if (e.y == EMPTY) return float2(0.0f);
    return float2(as_type<half>(ushort(e.x >> 16)), as_type<half>(ushort(e.x & 0xFFFFu)));
}

#if defined(METALFG_CTL_FEAT_WEIGHTS)
// Control (fg.test_scenes_r2 "Ctl-feat-weights"): the weights on the Feat grid, one evaluation per Feat texel centre with the offsets of the Mot
// cell floor((X + 1/2) Wm / Wf) and the second logit components at X; F20 then samples this image bilinearly.
kernel void f19_weights_feat(constant F19Params &P [[buffer(0)]], device const uint2 *scp [[buffer(1)]], device const uint2 *scc [[buffer(2)]],
                             texture2d<float> lgP [[texture(0)]], texture2d<float> lgC [[texture(1)]],
                             texture2d<float, access::write> bw [[texture(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wf || gid.y >= P.Hf) return;
    const float2 u = (float2(gid) + 0.5f) / float2(P.Wf, P.Hf);
    const uint2 m = min(uint2(floor((float2(gid) + 0.5f) * float2(P.Wm, P.Hm) / float2(P.Wf, P.Hf))), uint2(P.Wm - 1, P.Hm - 1));
    const uint ci = m.y * P.Wm + m.x;
    const float2 oP = decode_offset(scp, ci), oC = decode_offset(scc, ci);
    const float xp = lgP.sample(lin_clamp, u + oP).x, xc = lgC.sample(lin_clamp, u + oC).x;
    bw.write(float4(sig_safe(xp - lgP.read(gid).y), sig_safe(xc - lgC.read(gid).y), 0, 0), gid);
}
#endif

kernel void f19_weights(constant F19Params &P [[buffer(0)]], device const uint2 *scp [[buffer(1)]], device const uint2 *scc [[buffer(2)]],
                        texture2d<float> lgP [[texture(0)]], texture2d<float> lgC [[texture(1)]],
                        texture2d<float, access::write> bw [[texture(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const uint ci = gid.y * P.Wm + gid.x;
    const float2 u = (float2(gid) + 0.5f) / float2(P.Wm, P.Hm);
    const float2 oP = decode_offset(scp, ci), oC = decode_offset(scc, ci);
    const float2 N = float2(P.Wf, P.Hf);
    const int2 f = clamp(int2(ceil(u * N)) - 1, int2(0), int2(P.Wf - 1, P.Hf - 1));
    const float xp = lgP.sample(lin_clamp, u + oP).x, xc = lgC.sample(lin_clamp, u + oC).x;
    const float yp = lgP.read(uint2(f)).y, yc = lgC.read(uint2(f)).y;
    bw.write(float4(sig_safe(xp - yp), sig_safe(xc - yc), 0, 0), gid);
}
