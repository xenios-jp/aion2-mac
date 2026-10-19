// F22 direct output synthesis and the hole-fill chain F23..F26 (spec fg.output_synthesis).
#include <metal_stdlib>
using namespace metal;

constant float TAU = 0x1p-34f;      // positive guard, not above 2^-32

static inline float sat(float v) { return fmin(fmax(v, 0.0f), 1.0f); }

static inline float sig(float v) {
    if (v >= 0.0f) { const float e = precise::exp(-v); return 1.0f / (1.0f + e); }
    const float e = precise::exp(v);
    return e / (1.0f + e);
}

struct SynthParams {
    uint W, H, Nw, Nh, s;
    uint ox, oy;                 // output rectangle origin
};

// The bilinear tap of a candidate image at q (display pixels; texel x0 = floor(q.x) and x0 + 1, clamped to the image, likewise y). The four texels come from three
// gathers, one per channel (xyzw = (x0, y1), (x1, y1), (x1, y0), (x0, y0)): nearest sampling at half-pixel deltas around (x0 + 1, y0 + 1) in pixel coordinates
// lands on the texel centres, and the edge clamp is the clamp of the taps. Zero-weight taps are left out so that an integer position copies its texel exactly
// (no 0 * NaN or 0 * inf).
static inline float3 gather3g(texture2d<float> img, float2 q) {
    constexpr sampler gs(coord::pixel, address::clamp_to_edge, filter::nearest);
    const float fx = floor(q.x), fy = floor(q.y);
    const float ax = q.x - fx, ay = q.y - fy;
    const float2 c = float2(fx + 1.0f, fy + 1.0f);
    const float4 R = img.gather(gs, c, int2(0), component::x), G = img.gather(gs, c, int2(0), component::y), B = img.gather(gs, c, int2(0), component::z);
    float3 a = float3(R.w, G.w, B.w), b = float3(R.x, G.x, B.x);
    if (ax != 0.0f) {
        a = (1.0f - ax) * a + ax * float3(R.z, G.z, B.z);
        b = (1.0f - ax) * b + ax * float3(R.y, G.y, B.y);
    }
    return ay != 0.0f ? (1.0f - ay) * a + ay * b : a;
}

// Steps 1 to 5 of F22 for one display pixel; returns the stored (sanitised, saturated) colour. S4 non-null: the logistic of channel 4 of every cell, precomputed
// by f23_sig with the same function (sig) on the same values.
static float3 synthesize(constant SynthParams &P, uint2 p, device const half *N, texture2d<float> cA, texture2d<float> cB,
                         bool is_hole, device const float *S4 = nullptr) {
    const float s = float(P.s);
    const float2 g = min(float2(p) / s, float2(P.Nw - 1, P.Nh - 1));
    const uint i0 = uint(floor(g.x)), j0 = uint(floor(g.y));
    const uint i1 = min(i0 + 1, P.Nw - 1), j1 = min(j0 + 1, P.Nh - 1);
    const float fx = g.x - float(i0), fy = g.y - float(j0);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    device const half *c00 = N + (j0 * P.Nw + i0) * 8, *c10 = N + (j0 * P.Nw + i1) * 8;
    device const half *c01 = N + (j1 * P.Nw + i0) * 8, *c11 = N + (j1 * P.Nw + i1) * 8;
    float v[8];
    for (uint c = 0; c < 8; c++) {
        if (c == 4 && S4)
            v[c] = w00 * S4[j0 * P.Nw + i0] + w10 * S4[j0 * P.Nw + i1] + w01 * S4[j1 * P.Nw + i0] + w11 * S4[j1 * P.Nw + i1];
        else if (c == 4)
            v[c] = w00 * sig(float(c00[c])) + w10 * sig(float(c10[c])) + w01 * sig(float(c01[c])) + w11 * sig(float(c11[c]));
        else
            v[c] = w00 * float(c00[c]) + w10 * float(c10[c]) + w01 * float(c01[c]) + w11 * float(c11[c]);
    }
    float3 col;
    if (is_hole) {
        col =float3(sig(s * v[5]), sig(s * v[6]), sig(s * v[7]));
    } else {
        const float3 SA = gather3g(cA, float2(p) + s * float2(v[0], v[1]));
        const float3 SB = gather3g(cB, float2(p) + s * float2(v[2], v[3]));
        col = v[4] * SA + (1.0f - v[4]) * SB;
    }
#if defined(METALFG_CTL_NO_SANITISE)
    return col;                                                   // control: no marker, no saturation
#else
    if (isnan(col.x * col.y * col.z)) col = float3(0, 1, 1);     // debugging marker
    return float3(sat(col.x), sat(col.y), sat(col.z));
#endif
}

// synthesize for the fused chain, with the logistic of channel 4 precomputed per cell (S4): the same expressions channel by channel, written out so that each cell is
// loaded as two half4 vectors and no loop or branch over the channels remains; p / s is formed as an exact power-of-two scaling (s is a power of two).
static float3 synthesize_s4(constant SynthParams &P, uint2 p, device const half4 *N4, texture2d<float> cA, texture2d<float> cB,
                            bool is_hole, device const float *S4) {
    const float s = float(P.s);
    const float2 g = min(ldexp(float2(p), -int(ctz(P.s))), float2(P.Nw - 1, P.Nh - 1));
    const uint i0 = uint(floor(g.x)), j0 = uint(floor(g.y));
    const uint i1 = min(i0 + 1, P.Nw - 1), j1 = min(j0 + 1, P.Nh - 1);
    const float fx = g.x - float(i0), fy = g.y - float(j0);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    const uint k00 = j0 * P.Nw + i0, k10 = j0 * P.Nw + i1, k01 = j1 * P.Nw + i0, k11 = j1 * P.Nw + i1;
    const half4 a00 = N4[2 * k00], a10 = N4[2 * k10], a01 = N4[2 * k01], a11 = N4[2 * k11];
    const half4 b00 = N4[2 * k00 + 1], b10 = N4[2 * k10 + 1], b01 = N4[2 * k01 + 1], b11 = N4[2 * k11 + 1];
    const float v0 = w00 * float(a00.x) + w10 * float(a10.x) + w01 * float(a01.x) + w11 * float(a11.x);
    const float v1 = w00 * float(a00.y) + w10 * float(a10.y) + w01 * float(a01.y) + w11 * float(a11.y);
    const float v2 = w00 * float(a00.z) + w10 * float(a10.z) + w01 * float(a01.z) + w11 * float(a11.z);
    const float v3 = w00 * float(a00.w) + w10 * float(a10.w) + w01 * float(a01.w) + w11 * float(a11.w);
    const float v4 = w00 * S4[k00] + w10 * S4[k10] + w01 * S4[k01] + w11 * S4[k11];
    float3 col;
    if (is_hole) {
        const float v5 = w00 * float(b00.y) + w10 * float(b10.y) + w01 * float(b01.y) + w11 * float(b11.y);
        const float v6 = w00 * float(b00.z) + w10 * float(b10.z) + w01 * float(b01.z) + w11 * float(b11.z);
        const float v7 = w00 * float(b00.w) + w10 * float(b10.w) + w01 * float(b01.w) + w11 * float(b11.w);
        col = float3(sig(s * v5), sig(s * v6), sig(s * v7));
    } else {
        const float3 SA = gather3g(cA, float2(p) + s * float2(v0, v1));
        const float3 SB = gather3g(cB, float2(p) + s * float2(v2, v3));
        col = v4 * SA + (1.0f - v4) * SB;
    }
#if defined(METALFG_CTL_NO_SANITISE)
    return col;                                                   // control: no marker, no saturation
#else
    if (isnan(col.x * col.y * col.z)) col = float3(0, 1, 1);     // debugging marker
    return float3(sat(col.x), sat(col.y), sat(col.z));
#endif
}

kernel void f22_synth(constant SynthParams &P [[buffer(0)]], device const half *N [[buffer(1)]],
                      texture2d<float> cA [[texture(0)]], texture2d<float> cB [[texture(1)]], texture2d<float, access::read> hole [[texture(2)]],
                      texture2d<float, access::write> out [[texture(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.W || gid.y >= P.H) return;
    out.write(float4(synthesize(P, gid, N, cA, cB, hole.read(gid).x > 0.0f), 1.0f), uint2(P.ox, P.oy) + gid);
}

// The binary16 value a texture store of a finite float produces: the store of the former intermediate image truncates (measured on this device: it differs from
// the round-to-nearest conversion on half of the inputs). Inputs here are saturated, in 0..1.
static inline half half_store(float v) {
    const uint b = as_type<uint>(v);
    const uint e = (b >> 23) & 0xffu;
    const uint h = e >= 113u ? ((b & 0x7fffffffu) - 0x38000000u) >> 13 : uint(fabs(v) * 16777216.0f);     // normal range, else the binary16 subnormal grid 2^-24
    return as_type<half>(ushort(h | ((b >> 16) & 0x8000u)));
}

// F23 to F24 first levels in one kernel. One threadgroup of 16 x 16 pixels: the bad flags (hole, or outside the display) of the tile and a two-texel halo are staged
// once, the synthesis of F22 runs for the tile (the colour is rounded to binary16 and kept in threadgroup memory, as the former intermediate texture stored it), the
// fill mask F follows (1 unless one of the nine 3x3 windows containing the pixel consists of non-hole texels inside the display: a window is good when its three row
// segments are good), then level 1 (8 x 8 cells of the tile) and level 2 (4 x 4 cells), written to the buffers of the levels. The mask is also stored for the final step.
kernel void f23_fused(constant SynthParams &P [[buffer(0)]], device const half *N [[buffer(1)]], device float4 *lv1 [[buffer(2)]], device float4 *lv2 [[buffer(3)]],
                      constant uint &have2 [[buffer(4)]], device atomic_uint *tiles [[buffer(5)]], device const float *S4 [[buffer(6)]],
                      texture2d<float> cA [[texture(0)]], texture2d<float> cB [[texture(1)]], texture2d<float> hole [[texture(2)]],
                      texture2d<float, access::write> out [[texture(3)]], texture2d<float, access::write> fmask [[texture(4)]],
                      uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]],
                      uint2 ntg [[threadgroups_per_grid]]) {
    threadgroup uchar bad[20][20];      // texels x0 - 2 .. x0 + 17
    threadgroup uchar seg[20][18];      // bad flag of the three texels starting at column c
    threadgroup uchar win[18][18];      // window with top-left corner (x0 - 2 + c, y0 - 2 + r) is good
    threadgroup half4 synt[16][16];
    threadgroup uchar Fm[16][16];
    threadgroup uint anyfill, anybad;
    const int x0 = int(tg.x) * 16 - 2, y0 = int(tg.y) * 16 - 2;
    const uint li = lid.y * 16 + lid.x;
    if (li == 0) { anyfill = 0; anybad = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li < 100) {                                   // one gather per 2 x 2 block of texels (xyzw = (x0, y1), (x1, y1), (x1, y0), (x0, y0)); texels outside the display are bad
        constexpr sampler gs(coord::pixel, address::clamp_to_edge, filter::nearest);
        const uint bx = 2 * (li % 10), by = 2 * (li / 10);
        const int x = x0 + int(bx), y = y0 + int(by);
        const float4 h = hole.gather(gs, float2(float(x + 1), float(y + 1)));
        const bool ox0 = x < 0 || x >= int(P.W), ox1 = x + 1 < 0 || x + 1 >= int(P.W), oy0 = y < 0 || y >= int(P.H), oy1 = y + 1 < 0 || y + 1 >= int(P.H);
        const uchar b00 = (ox0 || oy0 || h.w > 0.0f) ? 1 : 0, b10 = (ox1 || oy0 || h.z > 0.0f) ? 1 : 0;
        const uchar b01 = (ox0 || oy1 || h.x > 0.0f) ? 1 : 0, b11 = (ox1 || oy1 || h.y > 0.0f) ? 1 : 0;
        bad[by][bx] = b00; bad[by][bx + 1] = b10; bad[by + 1][bx] = b01; bad[by + 1][bx + 1] = b11;
        if (b00 | b10 | b01 | b11) anybad = 1;        // benign race: every writer stores one
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool clean = anybad == 0;                   // no bad texel in the staged region: every window is good, F = 0 on the whole tile
    const bool inside = gid.x < P.W && gid.y < P.H;
    if (inside) {
        const float3 c = synthesize_s4(P, gid, (device const half4 *)N, cA, cB, bad[lid.y + 2][lid.x + 2] != 0, S4);
        out.write(float4(c, 1.0f), uint2(P.ox, P.oy) + gid);
        synt[lid.y][lid.x] = half4(half_store(c.x), half_store(c.y), half_store(c.z), 1.0h);
    }
    if (!clean) {
        for (uint i = li; i < 360; i += 256) {
            const uint r = i / 18, c = i % 18;
            seg[r][c] = bad[r][c] | bad[r][c + 1] | bad[r][c + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = li; i < 324; i += 256) {
            const uint r = i / 18, c = i % 18;
            win[r][c] = (seg[r][c] | seg[r + 1][c] | seg[r + 2][c]) ? 0 : 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        bool trusted = clean;
        if (!clean) for (uint j = 0; j < 3; j++) for (uint i = 0; i < 3; i++) trusted = trusted || win[lid.y + j][lid.x + i];
        Fm[lid.y][lid.x] = trusted ? 0 : 1;
        if (inside) fmask.write(float4(trusted ? 0.0f : 1.0f, 0, 0, 0), gid);
        if (inside && !trusted) anyfill = 1;          // benign race: every writer stores one
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // a tile holding a pixel to fill goes on the list of the final step (word 0 counts the tiles, the indices start at word 4)
    if (li == 0 && anyfill) {
        const uint k = atomic_fetch_add_explicit(&tiles[0], 1u, memory_order_relaxed);
        atomic_store_explicit(&tiles[4 + k], tg.y * ntg.x + tg.x, memory_order_relaxed);
    }
    const uint W1 = (P.W + 1) / 2, H1 = (P.H + 1) / 2;
    if (li < 64) {
        const uint cx = li % 8, cy = li / 8;
        const uint gx = tg.x * 8 + cx, gy = tg.y * 8 + cy;
        float4 r = float4(0);
        if (gx < W1 && gy < H1) {
            float sc = 0.0f;
            float3 sr = float3(0);
            for (uint dy = 0; dy < 2; dy++)
                for (uint dx = 0; dx < 2; dx++) {
                    if (2 * gx + dx >= P.W || 2 * gy + dy >= P.H) continue;
                    const float c = 1.0f - float(Fm[2 * cy + dy][2 * cx + dx]);
                    sc += c;
                    sr += c * float3(synt[2 * cy + dy][2 * cx + dx].rgb);
                }
            r = float4(sr / fmax(sc, TAU), sc / 4.0f);
            lv1[gy * W1 + gx] = r;
        }
        // level 2 from the 2 x 2 level-1 cells (cx, cy) .. (cx + 1, cy + 1): for even cx and cy they are lanes li, li + 1, li + 8, li + 9 of the same simdgroup
        // (threads 0..31 hold cells rows 0..3, threads 32..63 rows 4..7); added in the order of the former loop
        if (have2 != 0) {
            const float4 rb = simd_shuffle_down(r, 1), rc = simd_shuffle_down(r, 8), rd = simd_shuffle_down(r, 9);
            const uint ex = cx / 2, ey = cy / 2, W2 = (W1 + 1) / 2, H2 = (H1 + 1) / 2;
            const uint g2x = tg.x * 4 + ex, g2y = tg.y * 4 + ey;
            if ((cx & 1) == 0 && (cy & 1) == 0 && g2x < W2 && g2y < H2) {
                const float4 cs[4] = {r, rb, rc, rd};
                float sc = 0.0f;
                float3 sr = float3(0);
                for (uint dy = 0; dy < 2; dy++)
                    for (uint dx = 0; dx < 2; dx++) {
                        if (2 * g2x + dx >= W1 || 2 * g2y + dy >= H1) continue;
                        const float4 c = cs[dy * 2 + dx];
                        sc += c.w;
                        sr += c.w * c.rgb;
                    }
                lv2[g2y * W2 + g2x] = float4(sr / fmax(sc, TAU), sc / 4.0f);
            }
        }
    }
}

struct LevelParams { uint Wf, Hf, Wc, Hc; };    // fine level size, coarse level size

// Pull: level k+1 from level k, a coverage-weighted 2x2 average.
static inline void pull_px(constant LevelParams &P, device const float4 *fine, device float4 *coarse, uint2 gid) {
    float sc = 0.0f;
    float3 sr = float3(0);
    for (uint dy = 0; dy < 2; dy++)
        for (uint dx = 0; dx < 2; dx++) {
            const uint x = 2 * gid.x + dx, y = 2 * gid.y + dy;
            if (x >= P.Wf || y >= P.Hf) continue;
            const float4 c = fine[y * P.Wf + x];
            sc += c.w;
            sr += c.w * c.rgb;
        }
    coarse[gid.y * P.Wc + gid.x] = float4(sr / fmax(sc, TAU), sc / 4.0f);
}

kernel void f24_pull(constant LevelParams &P [[buffer(0)]], device const float4 *fine [[buffer(1)]], device float4 *coarse [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wc || gid.y >= P.Hc) return;
    pull_px(P, fine, coarse, gid);
}

// Push weights: 3/4 on the nearest coarse index, 1/4 on the neighbour at the side of the fine centre.
template <typename Fetch>
static inline float4 push_gen(Fetch fetch, uint Wc, uint Hc, uint ax, uint ay) {
    // the second index is the neighbour on the side of the fine cell's centre: a - 1 for even a, a + 1 for odd a
    const int mx[2] = {int(ax / 2), (ax & 1) ? int(ax / 2) + 1 : int(ax / 2) - 1};
    const int ny[2] = {int(ay / 2), (ay & 1) ? int(ay / 2) + 1 : int(ay / 2) - 1};
    float V = 0.0f;
    float3 Uc = float3(0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const int x = mx[i], y = ny[j];
            if (x < 0 || y < 0 || x >= int(Wc) || y >= int(Hc)) continue;      // outside the level: nothing
            const float w = (i == 0 ? 0.75f : 0.25f) * (j == 0 ? 0.75f : 0.25f);
            const float4 c = fetch(uint(x), uint(y));
            V += w * c.w;
            Uc += w * c.w * c.rgb;
        }
    return float4(Uc / clamp(V, TAU, 1.0f), V);
}

static inline float4 push_sample(device const float4 *coarse, uint Wc, uint Hc, uint ax, uint ay) {
    return push_gen([=](uint x, uint y) { return coarse[y * Wc + x]; }, Wc, Hc, ax, ay);
}

static inline void push_px(constant LevelParams &P, device float4 *fine, device const float4 *coarse, uint2 gid) {
    const float4 c = fine[gid.y * P.Wf + gid.x];
    if (c.w > 0.0f) return;                                                   // covered cells keep their values
    fine[gid.y * P.Wf + gid.x] = push_sample(coarse, P.Wc, P.Hc, gid.x, gid.y);
}

kernel void f25_push(constant LevelParams &P [[buffer(0)]], device float4 *fine [[buffer(1)]], device const float4 *coarse [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wf || gid.y >= P.Hf) return;
    push_px(P, fine, coarse, gid);
}

// The small levels (at most 1024 cells each, at most 8 levels): the pulls up to the top and the pushes back down to the first small level in one threadgroup,
// with a device-memory barrier between levels; each cell as the separate kernels compute it. Offsets in cells of the single level buffer.
struct TailLevel { LevelParams p; uint offF, offC; };
struct TailParamsOut { TailLevel pull[8]; TailLevel push[8]; uint npull, npush; };

kernel void f24_tail(constant TailParamsOut &T [[buffer(0)]], device float4 *lv [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    for (uint l = 0; l < T.npull; l++) {
        constant LevelParams &P = T.pull[l].p;
        const uint cnt = P.Wc * P.Hc;
        for (uint i = tid; i < cnt; i += nt) pull_px(P, lv + T.pull[l].offF, lv + T.pull[l].offC, uint2(i % P.Wc, i / P.Wc));
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (uint l = 0; l < T.npush; l++) {
        constant LevelParams &P = T.push[l].p;
        const uint cnt = P.Wf * P.Hf;
        for (uint i = tid; i < cnt; i += nt) push_px(P, lv + T.push[l].offF, lv + T.push[l].offC, uint2(i % P.Wf, i / P.Wf));
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Final step: pixels with F = 1 take the pushed value of the filled level 1 (alpha 1). The push of level 2 into level 1 is evaluated on the cells read
// (covered cells keep their values), so level 1 is not written back; have2 = 0 when the pyramid has a single level.
static inline void final_px(constant SynthParams &P, device const float4 *lv1, device const float4 *lv2, uint have2, texture2d<float, access::read> fmask,
                            texture2d<float, access::write> out, uint2 gid) {
    if (gid.x >= P.W || gid.y >= P.H) return;
    if (fmask.read(gid).x == 0.0f) return;
    const uint W1 = (P.W + 1) / 2, H1 = (P.H + 1) / 2, W2 = (W1 + 1) / 2, H2 = (H1 + 1) / 2;
    const uint h2 = have2;
    const float4 c = push_gen([=](uint x, uint y) {
        const float4 f = lv1[y * W1 + x];
        return (h2 != 0 && !(f.w > 0.0f)) ? push_sample(lv2, W2, H2, x, y) : f;
    }, W1, H1, gid.x, gid.y);
    out.write(float4(c.rgb, 1.0f), uint2(P.ox, P.oy) + gid);
}

kernel void f26_final(constant SynthParams &P [[buffer(0)]], device const float4 *lv1 [[buffer(1)]], device const float4 *lv2 [[buffer(2)]], constant uint &have2 [[buffer(3)]],
                      texture2d<float, access::read> fmask [[texture(0)]], texture2d<float, access::write> out [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    final_px(P, lv1, lv2, have2, fmask, out, gid);
}

// The final step on the tiles listed by f23_fused (one threadgroup of 16 x 16 pixels per listed tile, dispatched with the count of the list).
kernel void f26_final_tiles(constant SynthParams &P [[buffer(0)]], device const float4 *lv1 [[buffer(1)]], device const float4 *lv2 [[buffer(2)]],
                            constant uint &have2 [[buffer(3)]], device const uint *tiles [[buffer(4)]],
                            texture2d<float, access::read> fmask [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                            uint2 tgi [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]]) {
    const uint t = tiles[4 + tgi.x], ntx = (P.W + 15) / 16;
    final_px(P, lv1, lv2, have2, fmask, out, uint2(t % ntx, t / ntx) * 16 + lid);
}

// The logistic of channel 4 of every net2 cell (the blend weight of step 2 before the interpolation), once per cell for the fused synthesis.
kernel void f23_sig(constant SynthParams &P [[buffer(0)]], device const half *N [[buffer(1)]], device float *S4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= P.Nw * P.Nh) return;
    S4[gid] = sig(float(N[gid * 8 + 4]));
}

kernel void f23_tiles_reset(device uint *tiles [[buffer(0)]]) {
    tiles[0] = 0; tiles[1] = 1; tiles[2] = 1;
}
