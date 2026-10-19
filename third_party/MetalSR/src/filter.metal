// Reconstruction filter and history blend of pass P12 (sr.head_filter, sr.head_filter_r4). One thread handles one quad
// (2 x 2 output pixels sharing the ten parameters v_0..v_9 of the network head). The parameters come from the head network
// output `o` (fp16, 40 channels per L0 pixel, row order of sr.weights_format).
#include <metal_stdlib>
#include "sr_math.h"
#include "shared_types.h"
#include "head_net.h"
using namespace metal;

constant float kLog2e = 1.44269504088896341f;

// parameters of one quad after the derivations of sr.head_filter ("Per-quad parameters")
struct QuadParams {
    float sigma_x, sigma_y, rho, alpha_s, beta;
    float k_xx, k_xy, k_yy;      // -1/2 of the inverse covariance, multiplied by log2(e)
    half a16;                    // mean of the five tanh values
    bool colour_ok, rec_ok;      // false when a NaN parameter makes the colour or the recurrent scalar unusable
};

template <typename PTR>
static inline QuadParams quad_params(PTR o, int t, constant FiltParams &p, device const float *tanh_tab) {
    SR_STRICT
    half v[10];
    for (int q = 0; q < 10; q++) v[q] = o[8 * (q / 2) + 2 * t + (q % 2)];
    QuadParams r;
    r.colour_ok = !(sr_isnan_bits(v[0]) || sr_isnan_bits(v[1]) || sr_isnan_bits(v[2]) || sr_isnan_bits(v[8]) || sr_isnan_bits(v[9]));
    r.rec_ok = !(sr_isnan_bits(v[3]) || sr_isnan_bits(v[4]) || sr_isnan_bits(v[5]) || sr_isnan_bits(v[6]) || sr_isnan_bits(v[7]));
    r.sigma_x = r.sigma_y = r.rho = r.alpha_s = r.beta = 0.0f;
    r.k_xx = r.k_xy = r.k_yy = 0.0f;
    r.a16 = (half)0.0f;
    if (r.colour_ok) {
        r.sigma_x = fma(p.sigma_span, sr_sig((float)v[0]), p.sigma_min);
        r.sigma_y = fma(p.sigma_span, sr_sig((float)v[1]), p.sigma_min);
        const float ax = r.sigma_x * r.sigma_x, ay = r.sigma_y * r.sigma_y;
        r.rho = p.rho_max * sr_qfun((float)v[2], p.q_a, p.q_b, p.q_clip);
        const float cxy = r.rho * sqrt(ax * ay);
        const float inv = 1.0f / (ax * ay - cxy * cxy);
        r.k_xx = (-(ay / 2.0f) * inv) * kLog2e;
        r.k_xy = (cxy * inv) * kLog2e;
        r.k_yy = (-(ax / 2.0f) * inv) * kLog2e;
        r.alpha_s = p.A * sr_sig((float)v[8]);
        r.beta = clamp(0.5f + p.g_beta * (sr_sig((float)v[9]) - 0.5f), 0.0f, 1.0f);
    }
    if (r.rec_ok) {
        float s = tanh_tab[as_type<ushort>(v[3])];   // tanh of every fp16 bit pattern, correctly rounded (the engine's table)
        s = s + tanh_tab[as_type<ushort>(v[4])];
        s = s + tanh_tab[as_type<ushort>(v[5])];
        s = s + tanh_tab[as_type<ushort>(v[6])];
        s = s + tanh_tab[as_type<ushort>(v[7])];
        r.a16 = (half)(s * 0.2f);
    }
    return r;
}

// colour of one tap as the fp16 value of the shared tile (sr.head_filter, Per-pixel filter and blend)
static inline half4 tap_value(float4 C, float k_c, float phi_lo, bool hdr) {
    SR_STRICT
    if (!hdr) return half4(C);
    const float3 s = k_c * C.rgb;
    const float y = sr_luma(s);
    return half4(half3(s * max(1.0f / (1.0f + y), phi_lo)), (half)C.a);
}

static inline bool coverage(float cov_x, float cov_y, float dx, float dy) {
    SR_STRICT
    return (cov_y * fabs(dy) < 0.5f) && (cov_x * fabs(dx) < 0.5f);
}

static inline bool finite3(half3 v) { return sr_finite_bits(v.x) && sr_finite_bits(v.y) && sr_finite_bits(v.z); }

// history value: non-finite components read as 0, HDR tonemap with IEEE division (1 / 0 = +infinity)
static inline half3 history_value(float3 g, float phi_lo, bool hdr) {
    SR_STRICT
    g = sr_san3(g);
    if (hdr) g = g * max(1.0f / (sr_luma(g) + 1.0f), phi_lo);
    return half3(g);
}

// two-stage blend, each stage one fp16 fused multiply-add of an fp16 difference (sr.head_filter)
static inline half3 blend(half3 Cn, half3 H, half3 Cg, half lambda, half beta) {
    SR_STRICT
    const half3 M = fma(half3(lambda), Cn - H, H);
    return fma(half3(beta), Cg - M, M);
}

struct Stored { half4 out, hist; };

// colour after the blend (table of sr.head_filter_r4 section 3), the tonemaps and the saturating stores
template <bool HDR>
static inline Stored finish(half3 Q, bool q_usable, half3 H, float c_a, half rec, float e, constant FiltParams &p) {
    SR_STRICT
    const half3 Qp = (q_usable && finite3(Q)) ? Q : (finite3(H) ? H : half3(0.0h));
    float3 O = float3(Qp), K = float3(Qp);
    if (HDR) {
        const float3 q = float3(Qp);
        const float3 Z = q * (1.0f / max(1.0f - sr_luma(q), p.phi_lo));
        O = max((p.pre / e) * Z, 0.0f);
        K = Z * max(1.0f / (sr_luma(Z) + 1.0f), p.phi_hi);
    } else {
        O = max(O, 0.0f);
    }
    Stored s;
    s.out = half4(sr_sat16(O.x), sr_sat16(O.y), sr_sat16(O.z), sr_sat16(c_a));
    s.hist = half4(sr_sat16(K.x), sr_sat16(K.y), sr_sat16(K.z), rec);
    return s;
}

static inline half3 filtered(float3 acc, float Ws, bool bad) {
    SR_STRICT
    return bad ? half3(NAN) : half3(acc * (1.0f / Ws));
}

// Group of 16 x 8 quads = 32 x 16 output pixels = 8 x 4 L0 pixels. The colour taps of the group are staged once: every texel of the footprint is
// read, sanitised and tonemapped (tap_value) once instead of once per tap of every pixel that covers it. The footprint of one group at a render/output scale up to 1 is at most 35 x 19 texels (path N) or 37 x 20 (path W); a group
// whose footprint is larger (a huge jitter) reads the texture directly, with the same values.
// Path N: fp16 texels, 40 x 20. Path W (ratio above r_wide, so a smaller footprint): fp32 texels holding the same fp16 values exactly, 24 x 16,
// so a tap needs no conversion (the W filter has 25 taps per pixel).
template <bool WIDE> struct TileCfg { using T = half4; static constexpr constant int W = 40, H = 20; };
template <> struct TileCfg<true> { using T = float4; static constexpr constant int W = 24, H = 16; };

static inline half4 tile_value(float4 raw, float k_c, float phi_lo, bool hdr) {
    const half4 c = tap_value(sr_san4(raw), k_c, phi_lo, hdr);
    return half4(c.rgb, (half)0.0h);
}
static inline half4 tile_texel(texture2d<float> colour, int x, int y, int Wr, int Hr, float k_c, float phi_lo, bool hdr) {
    return tile_value(colour.read(uint2(clamp(x, 0, Wr - 1), clamp(y, 0, Hr - 1))), k_c, phi_lo, hdr);
}

struct Footprint { bool fits; int tx0, ty0, tw, th; };

// footprint of the group: the first and the last output pixel of its rectangle (the coordinates are monotone in the pixel index)
template <bool WIDE>
static inline Footprint p12_footprint(constant FiltParams &p, uint2 tg) {
    constexpr int R = WIDE ? 2 : 1;
    const int xa = (int)(tg.x * 32), ya = (int)(tg.y * 16);
    const int xb = min(xa + 31, (int)p.Wo - 1), yb = min(ya + 15, (int)p.Ho - 1);
    const float ua = WIDE ? sr_coord_b(p.sx, p.jx, xa) : sr_coord_a(p.sx, p.jx, xa), ub = WIDE ? sr_coord_b(p.sx, p.jx, xb) : sr_coord_a(p.sx, p.jx, xb);
    const float va = WIDE ? sr_coord_b(p.sy, p.jy, ya) : sr_coord_a(p.sy, p.jy, ya), vb = WIDE ? sr_coord_b(p.sy, p.jy, yb) : sr_coord_a(p.sy, p.jy, yb);
    const long nxa = (long)sr_round(ua), nxb = (long)sr_round(ub), nya = (long)sr_round(va), nyb = (long)sr_round(vb);
    const long twl = nxb - nxa + 2 * R + 1, thl = nyb - nya + 2 * R + 1;
    Footprint f;
    f.fits = twl > 0 && thl > 0 && twl <= TileCfg<WIDE>::W && thl <= TileCfg<WIDE>::H && nxa > -1000000000L && nxa < 1000000000L && nya > -1000000000L && nya < 1000000000L;
    f.tx0 = (int)nxa - R; f.ty0 = (int)nya - R; f.tw = (int)twl; f.th = (int)thl;
    return f;
}

template <bool WIDE, bool HDR>
static inline void p12_fill_tile(texture2d<float> colour, constant FiltParams &p, float k_c, Footprint f, uint lin, threadgroup typename TileCfg<WIDE>::T *tile) {
    constexpr int kTileW = TileCfg<WIDE>::W, kTileH = TileCfg<WIDE>::H;
    if (!f.fits) return;
    const int Wr = (int)p.Wr, Hr = (int)p.Hr, n = f.tw * f.th;
    // row of item idx = idx / tw without an integer division: (idx * ceil(65536 / tw)) >> 16 is exact for idx < kTileW * kTileH and
    // tw <= kTileW, as idx * (ceil(65536 / tw) * tw - 65536) < kTileW * kTileH * kTileW < 65536
    static_assert(kTileW * kTileH * kTileW < 65536, "exact reciprocal row index");
    const uint rm = (65535u + (uint)f.tw) / (uint)f.tw;
    // every read of the thread is issued before the first tonemap (T4)
    constexpr int kIt = (kTileW * kTileH + 127) / 128;
    float4 raw[kIt];
    int slot[kIt];
    #pragma unroll
    for (int k = 0; k < kIt; k++) {
        const int idx = (int)lin + 128 * k, cy = (int)(((uint)idx * rm) >> 16), cx = idx - cy * f.tw;
        slot[k] = cy * kTileW + cx;
        if (idx < n) raw[k] = colour.read(uint2(clamp(f.tx0 + cx, 0, Wr - 1), clamp(f.ty0 + cy, 0, Hr - 1)));
    }
    #pragma unroll
    for (int k = 0; k < kIt; k++)
        if ((int)lin + 128 * k < n) tile[slot[k]] = typename TileCfg<WIDE>::T(tile_value(raw[k], k_c, p.phi_lo, HDR));
}

// the exposure scalars of the call
static inline void p12_exposure(texture2d<float> e_now_t, constant FiltParams &p, thread float &e, thread float &k_c) {
    e = e_now_t.read(uint2(0)).x;
    e = e == 0.0f ? 1.0f : e;
    k_c = p.invp * e;
    k_c = k_c == 0.0f ? 1.0f : k_c;
}

// the texture reads of the four pixels of quad (qx, qy), issued before any arithmetic that consumes them (T4); the warped image is read in
// its own precision (TW = half in the fused pass: the values of the RGBA16F image, exactly)
template <typename TW>
struct PixIn { float2 dc[4]; int2 nn[4]; vec<TW, 4> wv[4]; float ca[4]; };

template <bool WIDE, typename TW>
static inline PixIn<TW> p12_reads(texture2d<float> colour, texture2d<TW> warped, constant FiltParams &p, int qx, int qy) {
    const int Wr = (int)p.Wr, Hr = (int)p.Hr;
    PixIn<TW> r;
    #pragma unroll
    for (int b = 0; b < 2; b++) {
        #pragma unroll
        for (int a = 0; a < 2; a++) {
            const int k = 2 * b + a, x = 2 * qx + a, y = 2 * qy + b;
            const float u = WIDE ? sr_coord_b(p.sx, p.jx, x) : sr_coord_a(p.sx, p.jx, x);
            const float v = WIDE ? sr_coord_b(p.sy, p.jy, y) : sr_coord_a(p.sy, p.jy, y);
            const int nx = (int)sr_round(u), ny = (int)sr_round(v);
            r.nn[k] = int2(nx, ny); r.dc[k] = float2((float)nx - u, (float)ny - v);
#ifdef SR_ABLATE_P12_WARPED
            r.wv[k] = vec<TW, 4>(float4(r.dc[k], u, 0.0f));
#else
            r.wv[k] = warped.read(uint2(x, y));
#endif
            r.ca[k] = colour.read(uint2(clamp(nx, 0, Wr - 1), clamp(ny, 0, Hr - 1))).a;
        }
    }
    return r;
}

// One tap of the filter (sr.head_filter): the exponent of the weight is (k_xx dx + k_xy dy) dx + k_yy dy^2 with the row terms b = dy k_xy and
// c = (dy dy) k_yy, two fused multiply-adds, and the weighted colour is added with a fused multiply-add. These are the roundings the fast-math
// build always produced; they are written out (no contraction left to the compiler) so that the code around the tap loop cannot change them.
static inline float tap_arg(float k_xx, float dx, float b, float c) { SR_STRICT return fma(fma(k_xx, dx, b), dx, c); }
static inline float tap_row_b(float k_xy, float dy) { SR_STRICT return dy * k_xy; }
static inline float tap_row_c(float k_yy, float dy) { SR_STRICT return (dy * dy) * k_yy; }
static inline float tap_sum(float s, float w) { SR_STRICT return s + w; }
static inline float3 tap_acc(float3 acc, float w, half3 c) { SR_STRICT return fma(float3(w), float3(c), acc); }
static inline float3 tap_acc(float3 acc, float w, float3 c) { SR_STRICT return fma(float3(w), c, acc); }

struct TapSum { float3 acc; float Ws; half3 Cn; };

// Half-precision filter weights (sr.metal_techniques_r3 section 3; rounding-changing, allowed by the owner's decision; a build switch, off by default).
// SR_HW_N / SR_HW_W select the variant per network path (default SR_HW, default 0 = the fp32 weights of sr.head_filter):
//   1 = V1: w' = rn16(2^(a - a_max)), the exponential in fp32, sums in fp32;
//   2 = control for section 7 item 3: V1 without the scaling by a_max (the scale of the largest weight is lost);
//   3 = control for section 7 item 6: the weights rounded to 3 mantissa bits instead of binary16.
// SR_TESTOUT (test builds only): the filter writes the filtered colour C'_g (before the blend) to the output texture and stores nothing else.
#ifndef SR_HW
#define SR_HW 0
#endif
#ifndef SR_HW_N
#define SR_HW_N SR_HW
#endif
#ifndef SR_HW_W
#define SR_HW_W SR_HW
#endif

static inline float hw_quant3(float w) {
    SR_STRICT
    return as_type<float>((as_type<uint>(w) + 0x80000u) & ~0xFFFFFu);   // keep 3 of the 23 mantissa bits, rounding half up (a control, not a rounding mode of any spec)
}

// V1 and its controls: all arguments first, then the weights relative to the largest
template <bool WIDE, bool HDR, bool FITS, int HW>
static inline TapSum p12_taps_h(texture2d<float> colour, constant FiltParams &p, QuadParams qp, int nx, int ny, float dxc, float dyc, float k_c, int tx0, int ty0,
                                threadgroup typename TileCfg<WIDE>::T *tile) {
    constexpr int R = WIDE ? 2 : 1, NT = 2 * R + 1;
    constexpr int kTileW = TileCfg<WIDE>::W;
    using T = typename TileCfg<WIDE>::T;
    const int Wr = (int)p.Wr, Hr = (int)p.Hr;
    float arg[NT * NT];
    float amax = -INFINITY;
    #pragma unroll
    for (int j = -R; j <= R; j++) {
        const float dy = dyc + (float)j;
        const float rb = tap_row_b(qp.k_xy, dy), rc = tap_row_c(qp.k_yy, dy);
        #pragma unroll
        for (int i = -R; i <= R; i++) {
            const float a = tap_arg(qp.k_xx, dxc + (float)i, rb, rc);
            arg[(j + R) * NT + (i + R)] = a;
            amax = fmax(amax, a);
        }
    }
    const float scale = HW == 2 ? 0.0f : amax;   // control 2: no scaling
    float3 acc = float3(0.0f);
    float Ws = 0.0f;
    half3 Cn = half3(0.0h);
    const int bx = nx - R - tx0, by = ny - R - ty0;
    #pragma unroll
    for (int j = -R; j <= R; j++) {
        #pragma unroll
        for (int i = -R; i <= R; i++) {
            const T c = FITS ? tile[(by + j + R) * kTileW + (bx + i + R)]
                             : T(tile_texel(colour, nx + i, ny + j, Wr, Hr, k_c, p.phi_lo, HDR));
            const float d = fmax(arg[(j + R) * NT + (i + R)] - scale, -126.0f);   // below 2^-126 the weight is 0 in binary16 (any weight below 2^-25 may be 0)
            const float wf = fast::exp2(d);
            const float w = HW == 3 ? hw_quant3(wf) : (float)(half)wf;
            acc = tap_acc(acc, w, c.rgb);
            Ws = tap_sum(Ws, w);
            if (i == 0 && j == 0) Cn = half3(c.rgb);
        }
    }
    TapSum t;
    t.acc = acc; t.Ws = Ws; t.Cn = Cn;
    return t;
}

// the weighted sum over the taps of one pixel; FITS (uniform in a group): the taps come from the staged tile, else from the colour texture
template <bool WIDE, bool HDR, bool FITS>
static inline TapSum p12_taps(texture2d<float> colour, constant FiltParams &p, QuadParams qp, int nx, int ny, float dxc, float dyc, float k_c, int tx0, int ty0,
                              threadgroup typename TileCfg<WIDE>::T *tile) {
    constexpr int R = WIDE ? 2 : 1;
    constexpr int kTileW = TileCfg<WIDE>::W;
    constexpr int HWV = WIDE ? SR_HW_W : SR_HW_N;
    if (HWV != 0) return p12_taps_h<WIDE, HDR, FITS, HWV>(colour, p, qp, nx, ny, dxc, dyc, k_c, tx0, ty0, tile);
    using T = typename TileCfg<WIDE>::T;
    const int Wr = (int)p.Wr, Hr = (int)p.Hr;
    float3 acc = float3(0.0f);
    float Ws = 0.0f;
    half3 Cn = half3(0.0h);
    const int bx = nx - R - tx0, by = ny - R - ty0;
#ifdef SR_ABLATE_P12_TAPS
    #pragma unroll
    for (int j = 0; j <= 0; j++) {
#else
    #pragma unroll
    for (int j = -R; j <= R; j++) {
#endif
        const float dy = dyc + (float)j;
        const float rb = tap_row_b(qp.k_xy, dy), rc = tap_row_c(qp.k_yy, dy);
#ifdef SR_ABLATE_P12_TAPS
        #pragma unroll
        for (int i = 0; i <= 0; i++) {
#else
        #pragma unroll
        for (int i = -R; i <= R; i++) {
#endif
            const float dx = dxc + (float)i;
            const T c = FITS ? tile[(by + j + R) * kTileW + (bx + i + R)]
                             : T(tile_texel(colour, nx + i, ny + j, Wr, Hr, k_c, p.phi_lo, HDR));
#ifdef SR_ABLATE_P12_EXP
            const float w = qp.k_xx * dx * dx + qp.k_yy * dy * dy + qp.k_xy * dx * dy;
#else
            const float w = fast::exp2(tap_arg(qp.k_xx, dx, rb, rc));
#endif
            acc = tap_acc(acc, w, c.rgb);
            Ws = tap_sum(Ws, w);
            if (i == 0 && j == 0) Cn = half3(c.rgb);
        }
    }
    TapSum t;
    t.acc = acc; t.Ws = Ws; t.Cn = Cn;
    return t;
}

// the filter and blend of the four pixels of quad (qx, qy) from their reads; the tile is filled and visible to the group
template <bool WIDE, bool HDR, typename TW>
static inline void p12_pixels(texture2d<float> colour, texture2d<half, access::write> out, texture2d<half, access::write> hist,
                              constant FiltParams &p, QuadParams qp, int qx, int qy, float e, float k_c, Footprint fp, threadgroup typename TileCfg<WIDE>::T *tile, PixIn<TW> in) {
    constexpr int R = WIDE ? 2 : 1;
    const int Wr = (int)p.Wr, Hr = (int)p.Hr;
    const half rec = qp.rec_ok ? (half)fmin(fmax((float)qp.a16, -1.0f), 1.0f) : (half)0.0f;
    const bool fits = fp.fits;
    const int tx0 = fp.tx0, ty0 = fp.ty0;
    const thread float2 *dc = in.dc;
    const thread int2 *nn = in.nn;
    #pragma unroll
    for (int b = 0; b < 2; b++) {
        #pragma unroll
        for (int a = 0; a < 2; a++) {
            const int k = 2 * b + a, x = 2 * qx + a, y = 2 * qy + b;
            if (x >= (int)p.Wo || y >= (int)p.Ho) continue;
            const int nx = nn[k].x, ny = nn[k].y;
            const float dxc = dc[k].x, dyc = dc[k].y;
            const half3 H = history_value(float3(in.wv[k].rgb), p.phi_lo, HDR);

            half3 Q = half3(0.0h);
            float c_a = sr_san(in.ca[k]);
            if (qp.colour_ok) {
                TapSum ts;
                if (fits) ts = p12_taps<WIDE, HDR, true>(colour, p, qp, nx, ny, dxc, dyc, k_c, tx0, ty0, tile);
                else ts = p12_taps<WIDE, HDR, false>(colour, p, qp, nx, ny, dxc, dyc, k_c, tx0, ty0, tile);
                const float3 acc = ts.acc;
                const float Ws = ts.Ws;
                const half3 Cn = ts.Cn;
                const half lambda = coverage(p.cov_x, p.cov_y, dxc, dyc) ? (half)qp.alpha_s : (half)0.0f;
                // A tap with a non-finite value makes the weighted sum non-finite (every weight is finite and positive or zero: infinity times zero is NaN),
                // so the "some tap is not finite" test of sr.head_filter is a bit-level test of the sum, once per pixel instead of once per tap.
                const uint3 ab = as_type<uint3>(acc) & 0x7F800000u;
                const bool bad = ab.x == 0x7F800000u || ab.y == 0x7F800000u || ab.z == 0x7F800000u;
#ifdef SR_TESTOUT
                out.write(half4(filtered(acc, Ws, bad), (half)1.0h), uint2(x + p.ox, y + p.oy));
                continue;
#endif
                Q = blend(Cn, H, filtered(acc, Ws, bad), lambda, (half)qp.beta);
            }
            Stored s = finish<HDR>(Q, qp.colour_ok, H, c_a, rec, e, p);
#ifdef SR_INJECT_NONFINITE
            // test builds only (the control of tests/finite_gpu.m): corrupt some stores past sat16, as an upstream defect would
            { const uint h = (uint)x * 7u + (uint)y * 13u; if (h % 61u == 0u) s.out.x = as_type<half>((ushort)(0x7C00u | ((h & 64u) << 9))); if (h % 61u == 1u) s.out.y = as_type<half>((ushort)0x7E00u); }
#endif
#ifndef SR_NO_OUTPUT_GUARD
            s.out = sr_guard16(s.out);   // last line before the store that leaves the engine: no infinity or NaN is ever written
#endif
            out.write(s.out, uint2(x + p.ox, y + p.oy));
#ifndef SR_ABLATE_P12_HIST
            hist.write(s.hist, uint2(x, y));
#endif
        }
    }
}

// test entry (the separate filter of the unfused pass; the head output o comes from a device buffer, `dec_head`)
template <bool WIDE, bool HDR>
static inline void p12_filter_body(texture2d<float> colour, texture2d<float> warped, texture2d<float> e_now_t, texture2d<half, access::write> out,
                                   texture2d<half, access::write> hist, device const half *o, constant FiltParams &p, uint W0, device const float *tanh_tab,
                                   uint2 tg, uint2 tid, threadgroup typename TileCfg<WIDE>::T *tile) {
    float e, k_c;
    p12_exposure(e_now_t, p, e, k_c);
    const Footprint fp = p12_footprint<WIDE>(p, tg);
    p12_fill_tile<WIDE, HDR>(colour, p, k_c, fp, tid.y * 16 + tid.x, tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int qx = (int)(tg.x * 16 + tid.x), qy = (int)(tg.y * 8 + tid.y);
    if (qx >= (int)((p.Wo + 1) / 2) || qy >= (int)((p.Ho + 1) / 2)) return;
    const int X = qx >> 1, Y = qy >> 1, t = (qx & 1) + 2 * (qy & 1);
    const QuadParams qp = quad_params(o + ((size_t)Y * W0 + (size_t)X) * 40, t, p, tanh_tab);
    p12_pixels<WIDE, HDR>(colour, out, hist, p, qp, qx, qy, e, k_c, fp, tile, p12_reads<WIDE>(colour, warped, p, qx, qy));
}

#define P12_KERNEL(NAME, WIDE, HDR) \
kernel void NAME(texture2d<float> colour [[texture(0)]], texture2d<float> warped [[texture(1)]], texture2d<float> e_now_t [[texture(2)]], \
                 texture2d<half, access::write> out [[texture(3)]], texture2d<half, access::write> hist [[texture(4)]], \
                 device const half *o [[buffer(0)]], constant FiltParams &p [[buffer(1)]], constant uint &W0 [[buffer(2)]], device const float *tanh_tab [[buffer(3)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) { \
    threadgroup TileCfg<WIDE>::T tile[TileCfg<WIDE>::W * TileCfg<WIDE>::H]; \
    p12_filter_body<WIDE, HDR>(colour, warped, e_now_t, out, hist, o, p, W0, tanh_tab, tg, tid, tile); \
}
P12_KERNEL(p12_filter_n0, false, false)
P12_KERNEL(p12_filter_n1, false, true)
P12_KERNEL(p12_filter_w0, true, false)
P12_KERNEL(p12_filter_w1, true, true)

// pass P12 as one dispatch (T2, T3): the head network of a tile of 8 x 4 L0 pixels in threadgroup memory, the quad parameters from it, then the
// filter of the 32 x 16 output pixels. The head output never reaches device memory. P13 (E_prev = E_now) is done by one thread of the first group:
// the only reader of E_prev is P1, which has finished.
// Two group barriers: every thread first issues all its reads (head staging, colour tile, the warped and colour values of its pixels) and stores
// the staged tile and the colour tile (separate regions); then two simdgroups run both head layers on their own rows; then the filter.
template <bool WIDE, bool HDR>
static inline void p12_fused_body(texture2d<float> colour, texture2d<half> warped, texture2d<float> e_now_t, texture2d<half, access::write> out,
                                  texture2d<half, access::write> hist, texture2d<float, access::write> e_prev, device const uchar *d1, device const uchar *s0,
                                  device const half *wa, device const half *ba, device const half *wb, device const half *bb,
                                  constant FiltParams &p, constant uint &W0, constant uint &W1H1, device const float *tanh_tab,
                                  uint2 tg, uint2 tid, uint sg, threadgroup typename TileCfg<WIDE>::T *tile, threadgroup half *ht) {
    const uint lin = tid.y * 16 + tid.x;
    const HeadRaw hr = head_issue(d1, s0, W0, W1H1 & 0xFFFFu, W1H1 >> 16, tg.x * 8, tg.y * 4, lin);
    const int qx = (int)(tg.x * 16 + tid.x), qy = (int)(tg.y * 8 + tid.y);
    const bool live = qx < (int)((p.Wo + 1) / 2) && qy < (int)((p.Ho + 1) / 2);
    PixIn<half> in;
    if (live) in = p12_reads<WIDE>(colour, warped, p, qx, qy);
    float e, k_c;
    p12_exposure(e_now_t, p, e, k_c);
    if (tg.x == 0 && tg.y == 0 && lin == 0) e_prev.write(e_now_t.read(uint2(0, 0)), uint2(0, 0));
    const Footprint fp = p12_footprint<WIDE>(p, tg);
    p12_fill_tile<WIDE, HDR>(colour, p, k_c, fp, lin, tile);
    head_store<WIDE>(hr, lin, ht);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 2) head_layers(wa, ba, wb, bb, sg, ht);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) return;
    const int pl = (tid.y >> 1) * 8 + (tid.x >> 1), t = (tid.x & 1) + 2 * (tid.y & 1);
    const QuadParams qp = quad_params((threadgroup const half *)ht + pl * kOStride, t, p, tanh_tab);
#ifdef SR_ABLATE_P12_FILTER
    out.write(half4((half)qp.k_xx, (half)qp.k_xy, (half)qp.beta, qp.a16), uint2(qx, qy)); return;
#endif
    p12_pixels<WIDE, HDR>(colour, out, hist, p, qp, qx, qy, e, k_c, fp, tile, in);
}

#define P12_FUSED_KERNEL(NAME, WIDE, HDR) \
kernel void NAME(texture2d<float> colour [[texture(0)]], texture2d<half> warped [[texture(1)]], texture2d<float> e_now_t [[texture(2)]], \
                 texture2d<half, access::write> out [[texture(3)]], texture2d<half, access::write> hist [[texture(4)]], \
                 texture2d<float, access::write> e_prev [[texture(5)]], \
                 device const uchar *d1 [[buffer(0)]], device const uchar *s0 [[buffer(1)]], device const half *wa [[buffer(2)]], device const half *ba [[buffer(3)]], \
                 device const half *wb [[buffer(4)]], device const half *bb [[buffer(5)]], constant FiltParams &p [[buffer(6)]], constant uint &W0 [[buffer(7)]], \
                 constant uint &W1H1 [[buffer(8)]], device const float *tanh_tab [[buffer(9)]], \
                 uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup TileCfg<WIDE>::T tile[TileCfg<WIDE>::W * TileCfg<WIDE>::H]; \
    threadgroup half ht[kHeadHalves]; \
    p12_fused_body<WIDE, HDR>(colour, warped, e_now_t, out, hist, e_prev, d1, s0, wa, ba, wb, bb, p, W0, W1H1, tanh_tab, tg, tid, sg, tile, ht); \
}
P12_FUSED_KERNEL(p12_fused_n0, false, false)
P12_FUSED_KERNEL(p12_fused_n1, false, true)
P12_FUSED_KERNEL(p12_fused_w0, true, false)
P12_FUSED_KERNEL(p12_fused_w1, true, true)

// test hook: the derived parameters of every quad of the head output (head-quads). Eight floats per quad, quads in
// row-major order of a (2 W0) x (2 H0) grid: sigma_x, sigma_y, rho, alpha_s, beta, a16, colour usable, recurrent usable.
kernel void p12_quad_debug(device const half *o [[buffer(0)]], constant FiltParams &p [[buffer(1)]], constant uint &W0 [[buffer(2)]], device const float *tanh_tab [[buffer(4)]],
                           device float *dbg [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const int X = (int)(gid.x >> 1), Y = (int)(gid.y >> 1), t = (int)(gid.x & 1) + 2 * (int)(gid.y & 1);
    const QuadParams q = quad_params(o + ((size_t)Y * W0 + (size_t)X) * 40, t, p, tanh_tab);
    device float *d = dbg + ((size_t)gid.y * (2 * W0) + gid.x) * 8;
    d[0] = q.sigma_x; d[1] = q.sigma_y; d[2] = q.rho; d[3] = q.alpha_s; d[4] = q.beta; d[5] = (float)q.a16; d[6] = q.colour_ok ? 1.0f : 0.0f; d[7] = q.rec_ok ? 1.0f : 0.0f;
}
