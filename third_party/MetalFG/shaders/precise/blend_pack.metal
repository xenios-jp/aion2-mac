// F20 candidate blend and pack (specs fg.blend_pack, fg.blend_pack_pin_r3).
#include <metal_stdlib>
using namespace metal;

constexpr sampler lin(filter::linear, address::clamp_to_edge, coord::normalized);

static inline float sat(float v) { return fmin(fmax(v, 0.0f), 1.0f); }

struct F20Params {
    uint W, H, Wm, Hm;
    uint ui;          // create bit 0: UI recomposition
    float tau;        // "alpha is not zero" means alpha > tau
};

// Decoded view of a scatter cell (fg.interm_motion): word 1 is the key and reads all ones when the cell is empty;
// word 0 is the payload, the binary16 pattern of the x offset in its upper half and of y in its lower half.
static inline bool decode_entry(uint2 e, thread float2 &off) {
    if (e.y == 0xFFFFFFFFu) { off = float2(0); return true; }
    off = float2(as_type<half>(ushort(e.x >> 16)), as_type<half>(ushort(e.x & 0xFFFFu)));
    return false;
}

// Specialisation of f20_blend_pack (fg.metal_techniques_r4 T6): the net2 factor and the UI recomposition flag as function constants (read from the parameters
// when not set).
constant uint FC20_S [[function_constant(10)]];
constant bool FC20_UI [[function_constant(11)]];
constant bool FC20_SPEC = is_function_constant_defined(FC20_S);

struct BlendOut { float4 A, B; bool hole; };

// The binary32 candidate values of one display position (fg.blend_pack_pin_r3, rule A). The blend pass stores them in binary16 for the synthesis; the pack
// pass computes them again with the same code and reduces the binary32 values, so the network input never derives from the stored images.
template <typename UiTex>
static inline BlendOut blend_at(constant F20Params &P, device const uint2 *scp, device const uint2 *scc, texture2d<float> Pt, texture2d<float> Ct,
                                texture2d<float> bw, texture2d<float> qt, UiTex uiP, UiTex uiC, uint2 gid) {
    const float2 uv = (float2(gid) + 0.5f) / float2(P.W, P.H);
    const uint2 mt = min(uint2(uv * float2(P.Wm, P.Hm)), uint2(P.Wm - 1, P.Hm - 1));
    const uint mi = mt.y * P.Wm + mt.x;
    float2 oP, oC;
    const bool eP = decode_entry(scp[mi], oP), eC = decode_entry(scc[mi], oC);

    const float3 p0 = Pt.read(gid).rgb, c0 = Ct.read(gid).rgb;
    const float3 pw = Pt.sample(lin, uv + oP).rgb, cw = Ct.sample(lin, uv + oC).rgb;
    const float3 cA = eP ? cw : pw;
    const float3 cB = eC ? pw : cw;
    const float2 w = bw.sample(lin, uv).xy;           // (wp, wc)
    float3 RA = p0 + w.x * (cA - p0);
    float3 RB = c0 + w.y * (cB - c0);
    const float2 q0 = qt.sample(lin, uv).xy;
    const float qa = q0.x + w.x * (qt.sample(lin, uv + oP).x - q0.x);
    const float qb = q0.y + w.y * (qt.sample(lin, uv + oC).y - q0.y);
    float alphaA = qa, alphaB = qb;
    bool isHole = eP && eC && !(w.x < 0.5f && w.y < 0.5f);   // a NaN weight is not below one half

    if (FC20_SPEC ? FC20_UI : (P.ui != 0)) {
        const float4 up = uiP.read(gid), uc = uiC.read(gid);
        RA = RA * (1.0f - up.w) + up.xyz;
        RB = RB * (1.0f - uc.w) + uc.xyz;
        const float3 df = abs(up.xyz - uc.xyz);
        const float d = sat((df.x + df.y + df.z) / 3.0f);
        const bool upa = up.w > P.tau, uca = uc.w > P.tau;
        if (upa) alphaA = d;
        if (uca) alphaB = d;
        if (upa || uca) isHole = false;
    }
    BlendOut o;
    o.A = float4(RA, alphaA);
    o.B = float4(RB, alphaB);
    o.hole = isHole;
    return o;
}

#if defined(METALFG_CTL_BP_TRUNC)
// Control of fg.blend_pack_pin_r3 (bp2-inject): the stored candidate images are written with an explicit truncation toward zero, on any platform.
static inline float4 trunc_to_half(float4 v) {
    float4 r;
    for (int i = 0; i < 4; i++) {
        half h = half(v[i]);                                     // nearest even
        if (isfinite(v[i]) && fabs(float(h)) > fabs(v[i])) h = as_type<half>(ushort(as_type<ushort>(h) - 1));
        r[i] = float(h);
    }
    return r;
}
#endif

kernel void f20_blend(constant F20Params &P [[buffer(0)]],
                      device const uint2 *scp [[buffer(1)]],    // sc.ip: intermediate to previous
                      device const uint2 *scc [[buffer(2)]],    // sc.ic: intermediate to current
                      texture2d<float> Pt [[texture(0)]],       // col.full.old
                      texture2d<float> Ct [[texture(1)]],       // col.full.cur
                      texture2d<float> bw [[texture(2)]],       // bw.feat (previous, current)
                      texture2d<float> qt [[texture(3)]],       // err.feat.cur (Ec, Ep)
                      texture2d<float, access::read> uiP [[texture(4)]],      // ui.full.old
                      texture2d<float, access::read> uiC [[texture(5)]],      // ui.full.cur
                      texture2d<float, access::write> candA [[texture(6)]],
                      texture2d<float, access::write> candB [[texture(7)]],
                      texture2d<float, access::write> hole [[texture(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.W || gid.y >= P.H) return;
    const BlendOut o = blend_at(P, scp, scc, Pt, Ct, bw, qt, uiP, uiC, gid);
#if defined(METALFG_CTL_BP_TRUNC)
    candA.write(trunc_to_half(o.A), gid);
    candB.write(trunc_to_half(o.B), gid);
#else
    candA.write(o.A, gid);
    candB.write(o.B, gid);
#endif
    hole.write(float4(o.hole ? 1.0f : 0.0f, 0, 0, 0), gid);
}

struct PackParams { F20Params b; uint Nw, Nh, s; };

// Reduction to the net2 grid (rules B and C): for a factor of one the binary32 candidate value rounded once to binary16 (nearest even); for larger factors the
// s x s block summed in binary32 in raster order (a single running sum starting with the first value, positions outside the display count as zero), multiplied by
// the reciprocal of the block area (a power of two), rounded once. The hole mask is the OR over the block, as before.
kernel void f20_pack_n2(constant PackParams &P [[buffer(0)]],
                        device half4 *n2a [[buffer(1)]],
                        device half4 *n2b [[buffer(2)]],
                        device half *n2h [[buffer(3)]],
                        device const uint2 *scp [[buffer(4)]],
                        device const uint2 *scc [[buffer(5)]],
                        texture2d<float> Pt [[texture(0)]],
                        texture2d<float> Ct [[texture(1)]],
                        texture2d<float> bw [[texture(2)]],
                        texture2d<float> qt [[texture(3)]],
                        texture2d<float, access::read> uiP [[texture(4)]],
                        texture2d<float, access::read> uiC [[texture(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Nw || gid.y >= P.Nh) return;

    float4 sa = float4(0), sb = float4(0);
    bool h = false, first = true;
    for (uint dy = 0; dy < P.s; dy++)
        for (uint dx = 0; dx < P.s; dx++) {
            const uint2 p = uint2(gid.x * P.s + dx, gid.y * P.s + dy);
            if (p.x >= P.b.W || p.y >= P.b.H) continue;
            const BlendOut o = blend_at(P.b, scp, scc, Pt, Ct, bw, qt, uiP, uiC, p);
            if (first) { sa = o.A; sb = o.B; first = false; }
            else { sa += o.A; sb += o.B; }
            h = h || o.hole;
        }
    const uint i = gid.y * P.Nw + gid.x;
    if (P.s == 1) {
        n2a[i] = half4(sa);
        n2b[i] = half4(sb);
    } else {
        const float inv = 1.0f / float(P.s * P.s);
        n2a[i] = half4(sa * inv);
        n2b[i] = half4(sb * inv);
    }
    n2h[i] = half(h ? 1.0f : 0.0f);
}

// The raster-order sum of channel ch over an S x S block of a tile of row length T, with S known at compile time (unrolled: the loads do not wait for the loop).
template <uint S>
static inline float block_sum(threadgroup const float4 *blk, uint T, uint ch) {
    float v[S * S];
    for (uint k = 0; k < S * S; k++) v[k] = blk[(k / S) * T + (k % S)][ch];
    float sum = v[0];
    for (uint k = 1; k < S * S; k++) sum += v[k];
    return sum;
}

// The OR of the flags of an S x S block (S known at compile time).
template <uint S>
static inline uchar block_or(threadgroup const uchar *fb, uint T) {
    uchar hh = 0;
    for (uint k = 0; k < S * S; k++) hh |= fb[(k / S) * T + (k % S)];
    return hh;
}

// Fused F20 (fg.performance_r1: the same values as f20_blend followed by f20_pack_n2, computed once). One thread per display position, as f20_blend: it blends and stores
// its position (binary16 candidate images, hole mask) and parks the binary32 values in threadgroup memory. A threadgroup is a square tile of T = max(s, 8) positions per
// side holding (T / s)^2 blocks of the net2 grid; after the barrier one thread per block adds the values of its block in raster order (positions outside the display are
// absent), multiplies by the reciprocal of the area and writes the reduction exactly as f20_pack_n2 does (rules B and C of fg.blend_pack_pin_r3).
kernel void f20_blend_pack(constant PackParams &P [[buffer(0)]],
                           device half4 *n2a [[buffer(1)]],
                           device half4 *n2b [[buffer(2)]],
                           device half *n2h [[buffer(3)]],
                           device const uint2 *scp [[buffer(4)]],
                           device const uint2 *scc [[buffer(5)]],
                           texture2d<float> Pt [[texture(0)]],
                           texture2d<float> Ct [[texture(1)]],
                           texture2d<float> bw [[texture(2)]],
                           texture2d<float> qt [[texture(3)]],
                           texture2d<float, access::read> uiP [[texture(4)]],
                           texture2d<float, access::read> uiC [[texture(5)]],
                           texture2d<float, access::write> candA [[texture(6)]],
                           texture2d<float, access::write> candB [[texture(7)]],
                           texture2d<float, access::write> hole [[texture(8)]],
                           uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    const uint S = FC20_SPEC ? FC20_S : P.s;
    threadgroup float4 sA[256], sB[256];
    threadgroup uchar sF[256];
    const uint T = max(S, 16u);
    const uint li = lid.y * T + lid.x;
    uchar flag = 0;
    if (gid.x < P.b.W && gid.y < P.b.H) {
        const BlendOut o = blend_at(P.b, scp, scc, Pt, Ct, bw, qt, uiP, uiC, gid);
#if defined(METALFG_CTL_BP_TRUNC)
        candA.write(trunc_to_half(o.A), gid);
        candB.write(trunc_to_half(o.B), gid);
#else
        candA.write(o.A, gid);
        candB.write(o.B, gid);
#endif
        hole.write(float4(o.hole ? 1.0f : 0.0f, 0, 0, 0), gid);
        sA[li] = o.A; sB[li] = o.B;
        flag = o.hole ? 3 : 1;
    }
    sF[li] = flag;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bpt = T / S;
    if (S >= 4) {
        // nine lanes per block: one per channel of A and B (each channel's sum keeps the raster order) and one for the hole OR; the valid range of the block is arithmetic
        const uint blk = li / 9, c = li % 9;
        if (blk >= bpt * bpt) return;
        const uint lx = blk % bpt, ly = blk / bpt;
        const uint bx = tg.x * bpt + lx, by = tg.y * bpt + ly;
        if (bx >= P.Nw || by >= P.Nh) return;
        const uint nx = min(S, P.b.W - min(bx * S, P.b.W)), ny = min(S, P.b.H - min(by * S, P.b.H));
        const uint i = by * P.Nw + bx;
        if (c == 8) {
            uchar hh = 0;
            threadgroup const uchar *fb = sF + (ly * S) * T + lx * S;
            if (nx == S && ny == S && S == 8) hh = block_or<8>(fb, T);
            else if (nx == S && ny == S && S == 4) hh = block_or<4>(fb, T);
            else for (uint dy = 0; dy < ny; dy++) for (uint dx = 0; dx < nx; dx++) hh |= fb[dy * T + dx];
            n2h[i] = half((hh & 2) ? 1.0f : 0.0f);
            return;
        }
        threadgroup const float4 *src = c < 4 ? sA : sB;
        const uint ch = c & 3;
        float sum = 0.0f;
        threadgroup const float4 *bp = src + (ly * S) * T + lx * S;
        if (nx == S && ny == S && S == 8) sum = block_sum<8>(bp, T, ch);
        else if (nx == S && ny == S && S == 4) sum = block_sum<4>(bp, T, ch);
        else if (nx == S && ny == S && S == 16) sum = block_sum<16>(bp, T, ch);
        else if (nx == S && ny == S) {
            sum = src[(ly * S) * T + lx * S][ch];
            for (uint dy = 0; dy < S; dy++) {
                threadgroup const float4 *row = src + (ly * S + dy) * T + lx * S;
                for (uint dx = (dy == 0 ? 1u : 0u); dx < S; dx++) sum += row[dx][ch];     // raster order
            }
        } else {
            for (uint dy = 0; dy < ny; dy++)
                for (uint dx = 0; dx < nx; dx++) {
                    const float v = src[(ly * S + dy) * T + lx * S + dx][ch];
                    sum = (dy == 0 && dx == 0) ? v : sum + v;                // the sum starts with the first value of the block
                }
        }
        const float inv = 1.0f / float(S * S);
        device half *dst = (device half *)(c < 4 ? n2a : n2b);
        dst[i * 4 + ch] = half(nx * ny ? sum * inv : 0.0f);
        return;
    }
    if (lid.x >= bpt || lid.y >= bpt) return;
    const uint bx = tg.x * bpt + lid.x, by = tg.y * bpt + lid.y;
    if (bx >= P.Nw || by >= P.Nh) return;
    float4 sa = float4(0), sb = float4(0);
    bool h = false, first = true;
    for (uint dy = 0; dy < S; dy++)
        for (uint dx = 0; dx < S; dx++) {
            const uint k = (lid.y * S + dy) * T + lid.x * S + dx;
            if (!(sF[k] & 1)) continue;
            if (first) { sa = sA[k]; sb = sB[k]; first = false; }
            else { sa += sA[k]; sb += sB[k]; }
            h = h || (sF[k] & 2);
        }
    const uint i = by * P.Nw + bx;
    if (S == 1) {
        n2a[i] = half4(sa);
        n2b[i] = half4(sb);
    } else {
        const float inv = 1.0f / float(S * S);
        n2a[i] = half4(sa * inv);
        n2b[i] = half4(sb * inv);
    }
    n2h[i] = half(h ? 1.0f : 0.0f);
}
