// F07 previous-to-current scatter, F08 motion judgement, F09 to F11 coverage pyramid and hole fill (spec fg.prev2curr).
// Scatter cell (8 bytes, little endian words): word 0 = payload, word 1 = key; both 0x7FFFFFFF when empty.
#include "fgcommon.h"

constant uint EMPTY = 0x7FFFFFFFu;

struct P2CParams {
    uint Wm, Hm, mode;       // mode: unused (one pass)
    float Wz, Hz;            // depth rectangle size (tap offsets)
    float H;                 // display height
    float d_dyn, d_sim, th_cam, eps_c;
    DepthConst D;
    float c2p[16];
    float p2c[16];
};

static inline float2 read_mv(texture2d<float, access::read> mvo, int2 q, float2 mot) { return mvo.read(uint2(q)).xy * mot; }

// One pass: the cell is the little-endian pair (payload, key), so the 64-bit minimum of (key << 32 | payload) is the smallest key and, among equal keys, the
// smallest payload (the same winner as a key pass followed by a payload pass; a key of EMPTY or more never replaces an empty cell's key).
static inline void offer(device atomic_uint *sc, uint ci, uint key, uint payload) {
    atomic_min_explicit((device atomic_ulong *)(sc + 2 * ci), (ulong(key) << 32) | ulong(payload), memory_order_relaxed);
}

static inline uint pack_half2(float2 v) {
    return (uint(as_type<ushort>(half(v.x))) << 16) | uint(as_type<ushort>(half(v.y)));
}

kernel void f07_scatter(constant P2CParams &P [[buffer(0)]], device atomic_uint *sc [[buffer(1)]], texture2d<uint, access::read> dep [[texture(0)]],
                        texture2d<float, access::read> mvo [[texture(1)]], texture2d<float, access::write> dynOut [[texture(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const float2 mot = float2(P.Wm, P.Hm);
    const int2 p = int2(gid);
    const uint rec = dep.read(gid).x;
    const float d = rec_depth(rec);
    const float2 u = rec_pos(rec, gid, uint2(P.Wm, P.Hm), float2(P.Wz, P.Hz));
    const float2 mv = read_mv(mvo, p, mot);
    float w;
    const float2 cam = cam_disp(u, d, P.c2p, mot, w);
    const float2 df = mv - cam;
    const bool dyn = (w <= 0.0f) || (dot(df, df) > fmax(1.0f, dot(mv, mv) * P.d_dyn / (P.H * P.H)));
    dynOut.write(float4(dyn ? 1.0f : 0.0f, 0, 0, 0), gid);
    const float lam = lambda_of(P.D, d);
    const float k = dyn ? lam : lam * w;
    const uint key = (depth_key(as_type<uint>(k) & ~1u) & ~1u) | (dyn ? 0u : 1u);
    const uint payload = pack_half2(mv);
    const int2 s = int2(sigma1(mv.x), sigma1(mv.y));
    const int2 t = int2(trunc(mv));
    const float thr = fmax(1.0f, dot(mv, mv) * P.d_sim / (P.H * P.H));
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const int2 qn = p + int2(i * s.x, j * s.y);
            const int2 q = clamp(qn, int2(0), int2(P.Wm - 1, P.Hm - 1));
            const int2 T = q + t;
            if (T.x < 0 || T.y < 0 || T.x >= int(P.Wm) || T.y >= int(P.Hm)) continue;
            if (i != 0 || j != 0) {
#if defined(METALFG_CTL_ZERO_BORDER)
                const float2 dq = (any(qn != q) ? float2(0.0f) : read_mv(mvo, q, mot)) - mv;   // control: zero motion outside the grid
#else
                const float2 dq = read_mv(mvo, q, mot) - mv;
#endif
                if (!(dot(dq, dq) < thr)) continue;
            }
            offer(sc, uint(T.y) * P.Wm + uint(T.x), key, payload);
        }
}

// ---- F08 ----------------------------------------------------------------------------------------------

// Values: mvf (raw v), nsc, hb bits 2 (b1) and 3 (b0) for the closing.
kernel void f08_values(constant P2CParams &P [[buffer(0)]], device const uint2 *sc [[buffer(1)]], device uchar *hb [[buffer(2)]],
                       texture2d<uint> depOld [[texture(0)]], texture2d<float> dynOld [[texture(1)]], texture2d<float> mvo [[texture(2)]],
                       texture2d<float, access::write> mvf [[texture(3)]], texture2d<float, access::write> nsc [[texture(4)]],
                       constant uint &writeNsc [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    const float2 mot = float2(P.Wm, P.Hm);
    const uint ci = gid.y * P.Wm + gid.x;
    const uint2 e = sc[ci];     // x = payload, y = key
    const bool empty = (e.y == EMPTY);
    float2 m = float2(0.0f);
    if (!empty) m = float2(as_type<half>(ushort(e.x >> 16)), as_type<half>(ushort(e.x & 0xFFFFu)));
    const bool kind_dyn = !empty && ((e.y & 1u) == 0u);
    const float clsk = as_type<float>(depth_unkey(e.y & ~1u));
    const bool pd = dynOld.read(gid).x > 0.5f;

    const float2 uvq = (float2(gid) + 0.5f) / mot;
    const uint orec = depOld.read(gid).x;
    const float dold = rec_depth(orec);
    const float2 uold = rec_pos(orec, gid, uint2(P.Wm, P.Hm), float2(P.Wz, P.Hz));
    float wold;
    const float2 camp = cam_disp(uold, dold, P.p2c, mot, wold);
    const float lold = pd ? lambda_of(P.D, dold) : lambda_of(P.D, dold) * wold;

    const float2 r = uvq + camp / mot;
    const float2 ev = mvo.sample(lin_clamp, r).xy * mot + camp;
    const bool camOK = dot(ev, ev) < P.th_cam;
    const bool mismatch = fabs(fmin(fmax(clsk, 0.0f), P.D.Zp) - fmin(fmax(lold, 0.0f), P.D.Zp)) > P.D.Zsep;
    const bool bad = mismatch || empty;
    float2 v;
    if (pd) v = -m;
    else if (!bad && !camOK && kind_dyn) v = -m;
    else v = camp;
    mvf.write(float4(v, 0, 0), gid);
    if (writeNsc) nsc.write(float4(empty ? 1.0f : 0.0f, 0, 0, 0), gid);
    hb[ci] = uchar((pd && bad ? 4 : 0) | (pd ? 8 : 0));
}

// Closing of b1 (dilation then erosion by 3 x 3, zero outside before the dilation), weights, hole bit. One threadgroup of 16 x 16 cells (tg = 16 x 16): the b1 flags of
// the tile with a two-cell rim are staged once (zero outside the grid), the dilation is formed on the tile with a one-cell rim (positions outside the grid included, as
// the erosion reads them), then the erosion. The in-place update keeps bits 4 and 8 (b1, b0), so staging reads the same values in any order.
kernel void f08_hole(constant P2CParams &P [[buffer(0)]], device uchar *hb [[buffer(2)]], uint2 gid [[thread_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]], uint2 tgs [[threads_per_threadgroup]]) {
    threadgroup uchar b1[20][20];      // cells x0 - 2 .. x0 + 17
    threadgroup uchar dl[18][18];      // dilation at cells x0 - 1 .. x0 + 16
    const int x0 = int(tgp.x) * 16 - 2, y0 = int(tgp.y) * 16 - 2;
    const int W = int(P.Wm), H = int(P.Hm);
    const uint li = lid.y * tgs.x + lid.x, nt = tgs.x * tgs.y;
    for (uint i = li; i < 400; i += nt) {
        const int x = x0 + int(i % 20), y = y0 + int(i / 20);
        b1[i / 20][i % 20] = (x >= 0 && y >= 0 && x < W && y < H) ? ((hb[uint(y) * P.Wm + uint(x)] & 4) ? 1 : 0) : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = li; i < 324; i += nt) {
        const uint r = i / 18, c = i % 18;
        dl[r][c] = b1[r][c] | b1[r][c + 1] | b1[r][c + 2] | b1[r + 1][c] | b1[r + 1][c + 1] | b1[r + 1][c + 2] | b1[r + 2][c] | b1[r + 2][c + 1] | b1[r + 2][c + 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid.x >= P.Wm || gid.y >= P.Hm) return;
    bool closed = true;
    for (uint ey = 0; ey < 3; ey++)
        for (uint ex = 0; ex < 3; ex++) closed = closed && dl[lid.y + ey][lid.x + ex] != 0;
    const uint ci = gid.y * P.Wm + gid.x;
    const uchar old = hb[ci];
    const bool b0 = (old & 8) != 0;
    hb[ci] = uchar((old & 12) | (closed ? 1 : 0) | ((b0 && !closed) ? 2 : 0));
}

// ---- pyramid ------------------------------------------------------------------------------------------

struct PyrParams {
    uint srcW, srcH, dstW, dstH;
    uint srcOff, dstOff, upOff, upW, upH;   // upOff/up*: the coarser filled level (fill kernels)
    uint last;                               // fill: this is the top level (no coarser level)
    uint level;
    float eps_c;
};

// Level 1 from level 0 = (v wt, wt) per pixel.
kernel void pyr_level1(constant PyrParams &Q [[buffer(0)]], device const uchar *hb [[buffer(2)]], texture2d<float, access::read> mvf [[texture(0)]],
                       device float4 *S [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= Q.dstW || gid.y >= Q.dstH) return;
    float3 s = float3(0.0f);
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++) {
            const uint2 p = uint2(gid.x * 2 + i, gid.y * 2 + j);
            if (p.x >= Q.srcW || p.y >= Q.srcH) continue;
            if (hb[p.y * Q.srcW + p.x] & 2) {
                const float2 v = mvf.read(p).xy;
                s += float3(v, 1.0f);
            }
        }
    S[Q.dstOff + gid.y * Q.dstW + gid.x] = float4(s, 0.0f);
}

static inline void down_px(constant PyrParams &Q, device float4 *S, uint2 gid) {
    float3 s = float3(0.0f);
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++) {
            const uint2 p = uint2(gid.x * 2 + i, gid.y * 2 + j);
            if (p.x >= Q.srcW || p.y >= Q.srcH) continue;
            s += S[Q.srcOff + p.y * Q.srcW + p.x].xyz;
        }
    S[Q.dstOff + gid.y * Q.dstW + gid.x] = float4(s, 0.0f);
}

kernel void pyr_down(constant PyrParams &Q [[buffer(0)]], device float4 *S [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= Q.dstW || gid.y >= Q.dstH) return;
    down_px(Q, S, gid);
}

// Upsample of the filled coarse level F (pairs (V, c)) to the fine element (X, Y).
static float3 upsample(device const float4 *F, uint off, uint cw, uint ch, uint X, uint Y, float eps_c) {
    const int nx = int(X >> 1), ny = int(Y >> 1);
    int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
    if (fx < 0) fx = nx;
    if (fy < 0) fy = ny;
    float upc = 0.0f;
    float2 upvc = float2(0.0f);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const int cx = i ? fx : nx, cy = j ? fy : ny;
            if (cx >= int(cw) || cy >= int(ch)) continue;
            const float a = (i ? 0.25f : 0.75f) * (j ? 0.25f : 0.75f);
            const float4 f = F[off + uint(cy) * cw + uint(cx)];   // (V.x, V.y, c)
            upc += a * f.z;
            upvc += a * f.z * f.xy;
        }
    return float3(upvc / fmin(fmax(upc, eps_c), 1.0f), upc);
}

// Filled level k from its own block sums (S) and the filled coarser level (F).
static inline void fill_px(constant PyrParams &Q, device const float4 *S, device float4 *F, uint2 gid) {
    const float4 s = S[Q.dstOff + gid.y * Q.dstW + gid.x];
    const float area = exp2(2.0f * float(Q.level));
    float3 out;
    if (s.z > 0.0f || Q.last) out = float3(s.xy / fmax(s.z, Q.eps_c), s.z / area);
    else out = upsample(F, Q.upOff, Q.upW, Q.upH, gid.x, gid.y, Q.eps_c);
    F[Q.dstOff + gid.y * Q.dstW + gid.x] = float4(out, 0.0f);
}

kernel void pyr_fill(constant PyrParams &Q [[buffer(0)]], device const float4 *S [[buffer(4)]], device float4 *F [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= Q.dstW || gid.y >= Q.dstH) return;
    fill_px(Q, S, F, gid);
}

// The small levels first..K of both sweeps (block sums upwards, fill downwards) in one threadgroup: one dispatch instead of 2 (K - first + 1) tiny ones.
// down[k - first] and fill[k - first] are the parameter blocks the separate kernels get for level k.
struct TailParams {
    PyrParams down[8];
    PyrParams fill[8];
    uint n;                  // number of levels
};

kernel void pyr_tail(constant TailParams &T [[buffer(0)]], device float4 *S [[buffer(4)]], device float4 *F [[buffer(5)]],
                     uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    for (uint l = 0; l < T.n; l++) {
        const uint w = T.down[l].dstW, cnt = w * T.down[l].dstH;
        for (uint i = tid; i < cnt; i += nt) down_px(T.down[l], S, uint2(i % w, i / w));
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (uint l = T.n; l-- > 0;) {
        const uint w = T.fill[l].dstW, cnt = w * T.fill[l].dstH;
        for (uint i = tid; i < cnt; i += nt) fill_px(T.fill[l], S, F, uint2(i % w, i / w));
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Level 0: holes take the upsampled pair of F_1.
kernel void pyr_final(constant PyrParams &Q [[buffer(0)]], device const uchar *hb [[buffer(2)]], device const float4 *F [[buffer(5)]],
                      texture2d<float, access::write> mvf [[texture(0)]], constant uint &haveLevel1 [[buffer(6)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= Q.srcW || gid.y >= Q.srcH) return;
    if (!(hb[gid.y * Q.srcW + gid.x] & 1)) return;
    float2 v = float2(0.0f);
    if (haveLevel1) v = upsample(F, Q.upOff, Q.upW, Q.upH, gid.x, gid.y, Q.eps_c).xy;
    mvf.write(float4(v, 0, 0), gid);
}
