/* CPU model of the network operators (specs fg.net1_forward, fg.net2_forward, fg.numerics and the arithmetic of fg.arithmetic_pin_r4): binary16 running totals
   with a rounding after every (tap, channel group) pair, the order of the pairs of rule B, binary16 activation and reduction steps. Written independently of the
   kernels; an engine passes when it reproduces it bit for bit (rules H and I excepted). */
#pragma once
#pragma clang fp contract(off)
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "weights.h"

typedef _Float16 h16;

static inline double rn16(double v) { return (double)(h16)v; }

static inline uint16_t bits16(h16 h) { uint16_t b; memcpy(&b, &h, 2); return b; }

/* order of the pairs of a 3x3 layer (rule B): 1 channel groups, horizontal tap, vertical tap; 2 horizontal tap, groups, vertical tap; 3 vertical tap, horizontal tap,
   groups. The model reads it from this variable; the tests set it for the layer they model (net2 and the default: 3). */
static int g_ref_nest = 3;

/* the order of the pairs (rule B) of a named layer: net1's first, third and fourth encoder convolutions 1, the fifth 2, every other 3x3 layer (net2) 3 */
static int nest_for_name(const char *n) {
    if (strstr(n, "n1.enc0") || strstr(n, "n1.enc2") || strstr(n, "n1.enc3")) return 1;
    if (strstr(n, "n1.enc4")) return 2;
    return 3;
}

/* rule A: one (tap, group) pair */
static h16 ref_pair(const h16 *in, int W, int H, int in_stride, int in_base, int cin, const h16 *wrow, int k, int cin_w, int x, int y, int kx, int ky, int g, h16 tot) {
    int px = k == 3 ? x + kx - 1 : x, py = k == 3 ? y + ky - 1 : y;
    if (px < 0 || py < 0 || px >= W || py >= H) return tot;
    const h16 *ip = in + ((size_t)py * W + px) * in_stride + in_base;
    const h16 *wp = wrow + (k == 3 ? (ky * 3 + kx) : 0) * cin_w;
    float t = (float)tot;
    int c1 = g * 16 + 16 < cin ? g * 16 + 16 : cin;
    for (int c = g * 16; c < c1; c++) t = fmaf((float)wp[c], (float)ip[c], t);
    return (h16)t;
}

/* rules A to C for one output value */
static h16 ref_conv_value(const h16 *in, int W, int H, int in_stride, int in_base, int cin, const h16 *wgt, const h16 *bias, int k, int cin_w, int row0, int o,
                          int flags, double leak, double bound, int bound_from, int x, int y, int nest) {
    h16 tot = bias[row0 + o];
    const h16 *wrow = wgt + (size_t)(row0 + o) * k * k * cin_w;
    int ng = (cin + 15) / 16;
    if (k == 1) { for (int g = 0; g < ng; g++) tot = ref_pair(in, W, H, in_stride, in_base, cin, wrow, k, cin_w, x, y, 0, 0, g, tot); }
    else if (nest == 1) { for (int g = 0; g < ng; g++) for (int kx = 0; kx < 3; kx++) for (int ky = 0; ky < 3; ky++) tot = ref_pair(in, W, H, in_stride, in_base, cin, wrow, k, cin_w, x, y, kx, ky, g, tot); }
    else if (nest == 2) { for (int kx = 0; kx < 3; kx++) for (int g = 0; g < ng; g++) for (int ky = 0; ky < 3; ky++) tot = ref_pair(in, W, H, in_stride, in_base, cin, wrow, k, cin_w, x, y, kx, ky, g, tot); }
    else { for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (int g = 0; g < ng; g++) tot = ref_pair(in, W, H, in_stride, in_base, cin, wrow, k, cin_w, x, y, kx, ky, g, tot); }
    h16 v = tot;
    if (flags & 1) {
        h16 p = (h16)((float)v * (float)(h16)(float)leak);
        v = (float)v >= (float)p ? v : p;                  /* the larger of the value and the binary16 product */
        if (isnan((float)p) && !isnan((float)v)) v = v;
    }
    if ((flags & 2) && o >= bound_from) { float f = (float)v; f = f < -(float)bound ? -(float)bound : f; f = f > (float)bound ? (float)bound : f; v = (h16)f; }
    return v;
}

/* One convolution layer. in: W x H, channel stride in_stride, offset in_base, cin channels. Weights [row][tap][cin_w].
   Output channels row0 .. row0 + cout - 1 of the weight tensor go to out channels out_base ... with stride out_stride.
   flags as in nn.h: 1 act, 2 bound, 4 pool (rule D), 8 residual, 16 explicit zero add. The order of the pairs is g_ref_nest. */
static void ref_conv(const h16 *in, int W, int H, int in_stride, int in_base, int cin, const h16 *wgt, const h16 *bias,
                     int k, int cin_w, int row0, int cout, int flags, double leak, double bound, int bound_from,
                     h16 *out, int out_stride, int out_base, const h16 *res, int res_stride, int res_base) {
    int OW = (flags & 4) ? W / 2 : W, OH = (flags & 4) ? H / 2 : H;
    for (int y = 0; y < OH; y++)
        for (int x = 0; x < OW; x++)
            for (int o = 0; o < cout; o++) {
#define CV(xx, yy) ref_conv_value(in, W, H, in_stride, in_base, cin, wgt, bias, k, cin_w, row0, o, flags, leak, bound, bound_from, xx, yy, g_ref_nest)
                h16 r;
                if (flags & 4) {
                    h16 s = CV(2 * x, 2 * y);
                    s = (h16)((float)s + (float)CV(2 * x + 1, 2 * y));
                    s = (h16)((float)s + (float)CV(2 * x, 2 * y + 1));
                    s = (h16)((float)s + (float)CV(2 * x + 1, 2 * y + 1));
                    r = (h16)((float)s * 0.25f);
                } else {
                    r = CV(x, y);
                    if (flags & 16) r = (h16)((float)r + 0.0f);
                    if (flags & 8) r = (h16)((float)r + (float)res[((size_t)y * W + x) * res_stride + res_base + o]);
                }
#undef CV
                out[((size_t)y * OW + x) * out_stride + out_base + o] = r;
            }
}

static void ref_pool_max(const h16 *in, int W, int H, int C, h16 *out) {
    for (int y = 0; y < H / 2; y++)
        for (int x = 0; x < W / 2; x++)
            for (int c = 0; c < C; c++) {
                double m = -INFINITY;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        double v = (double)in[((size_t)(2 * y + dy) * W + 2 * x + dx) * C + c];
                        if (v > m) m = v;
                    }
                out[((size_t)y * (W / 2) + x) * C + c] = (h16)m;
            }
}

/* interpolation step in binary16 (rules E and H) */
static h16 ref_lerp16(h16 a, h16 b, float f) {
    h16 w1 = (h16)f, w0 = (h16)(1.0f - f);
    h16 t0 = (h16)((float)a * (float)w0), t1 = (h16)((float)b * (float)w1);
    return (h16)((float)t0 + (float)t1);
}

/* rule H: net1 enlargement with an explicit coordinate scale (the quotient (Ws - 1) / (Wd - 1) as a binary32 number); scale < 0 selects the correctly rounded one.
   out = rn16(float(up(src)) + skip) when skip is given, else up(src). */
static void ref_up_corner_scaled(const h16 *src, int Ws, int Hs, int C, const h16 *skip, h16 *out, float scx_in, float scy_in) {
    int Wd = 2 * Ws, Hd = 2 * Hs;
    float scx = scx_in >= 0 ? scx_in : (Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.0f);
    float scy = scy_in >= 0 ? scy_in : (Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.0f);
    for (int Y = 0; Y < Hd; Y++)
        for (int X = 0; X < Wd; X++) {
            float sx = (float)X * scx, sy = (float)Y * scy;
            int x0 = (int)floorf(sx > 0 ? sx : 0), y0 = (int)floorf(sy > 0 ? sy : 0);
            x0 = x0 > Ws - 1 ? Ws - 1 : x0; y0 = y0 > Hs - 1 ? Hs - 1 : y0;
            int x1 = x0 + 1 < Ws - 1 ? x0 + 1 : Ws - 1, y1 = y0 + 1 < Hs - 1 ? y0 + 1 : Hs - 1;
            float fx = sx - (float)x0, fy = sy - (float)y0;
            for (int c = 0; c < C; c++) {
#define S(yy, xx) src[((size_t)(yy) * Ws + (xx)) * C + c]
                h16 h0 = ref_lerp16(S(y0, x0), S(y0, x1), fx), h1 = ref_lerp16(S(y1, x0), S(y1, x1), fx);
#undef S
                h16 u = ref_lerp16(h0, h1, fy);
                if (skip) u = (h16)((float)u + (float)skip[((size_t)Y * Wd + X) * C + c]);
                out[((size_t)Y * Wd + X) * C + c] = u;
            }
        }
}

static void ref_up_corner(const h16 *src, int Ws, int Hs, int C, const h16 *skip, h16 *out) { ref_up_corner_scaled(src, Ws, Hs, C, skip, out, -1.0f, -1.0f); }

/* net2 half-pixel x2 (edges repeated; the enlargements that do not feed the final heads) */
static void ref_up2(const h16 *src, int Ws, int Hs, int C, h16 *out) {
    int Wd = 2 * Ws, Hd = 2 * Hs;
    for (int Y = 0; Y < Hd; Y++)
        for (int X = 0; X < Wd; X++) {
            int nx = X / 2, ny = Y / 2;
            int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
            fx = fx < 0 ? 0 : fx > Ws - 1 ? Ws - 1 : fx;
            fy = fy < 0 ? 0 : fy > Hs - 1 ? Hs - 1 : fy;
            for (int c = 0; c < C; c++) {
#define S(yy, xx) (float)src[((size_t)(yy) * Ws + (xx)) * C + c]
                float h0 = 0.75f * S(ny, nx) + 0.25f * S(ny, fx);
                float h1 = 0.75f * S(fy, nx) + 0.25f * S(fy, fx);
#undef S
                out[((size_t)Y * Wd + X) * C + c] = (h16)(0.75f * h0 + 0.25f * h1);
            }
        }
}

/* rule E: the enlargement that feeds the two final heads (extrapolating at the left and top edge) */
static void ref_up2x_axis(int X, int Ns, int *i0, int *i1, float *f) {
    float s = ((float)X - 0.5f) * 0.5f;
    if (s < 0.0f) { *i0 = *i1 = 0; *f = s; }
    else { int a = (int)floorf(s); *i0 = a < Ns - 1 ? a : Ns - 1; *i1 = *i0 + 1 < Ns - 1 ? *i0 + 1 : Ns - 1; *f = s - (float)*i0; }
}
static void ref_up2x(const h16 *src, int Ws, int Hs, int C, h16 *out) {
    int Wd = 2 * Ws, Hd = 2 * Hs;
    for (int Y = 0; Y < Hd; Y++)
        for (int X = 0; X < Wd; X++) {
            int x0, x1, y0, y1;
            float fx, fy;
            ref_up2x_axis(X, Ws, &x0, &x1, &fx);
            ref_up2x_axis(Y, Hs, &y0, &y1, &fy);
            for (int c = 0; c < C; c++) {
#define S(yy, xx) src[((size_t)(yy) * Ws + (xx)) * C + c]
                h16 h0 = ref_lerp16(S(y0, x0), S(y0, x1), fx), h1 = ref_lerp16(S(y1, x0), S(y1, x1), fx);
#undef S
                out[((size_t)Y * Wd + X) * C + c] = ref_lerp16(h0, h1, fy);
            }
        }
}

/* rule F: one gathered pixel of the stage join. img: Nw x Nh x 4. */
static void ref_join_axis(int idx, h16 off, int N, int *i0, int *i1, float *a) {
    h16 pos = (h16)((float)(h16)(float)idx + (float)(h16)(2.0f * (float)off));
    float fl = floorf((float)pos);
    *a = (float)pos - fl;
    if (isnan(fl)) { fl = 0; *a = 0; }
    fl = fl < -65536.0f ? -65536.0f : fl > 65536.0f ? 65536.0f : fl;
    int f = (int)fl;
    *i0 = f < 0 ? 0 : f > N - 1 ? N - 1 : f;
    *i1 = f + 1 < 0 ? 0 : f + 1 > N - 1 ? N - 1 : f + 1;
}
static void ref_gather4(const h16 *img, int W, int H, int x, int y, h16 ox, h16 oy, h16 *out) {
    int x0, x1, y0, y1;
    float ax, ay;
    ref_join_axis(x, ox, W, &x0, &x1, &ax);
    ref_join_axis(y, oy, H, &y0, &y1, &ay);
    float wx0 = 1.0f - ax, wx1 = ax, wy0 = 1.0f - ay, wy1 = ay;
    for (int c = 0; c < 4; c++) {
        float a = (float)img[((size_t)y0 * W + x0) * 4 + c], b = (float)img[((size_t)y0 * W + x1) * 4 + c];
        float cc = (float)img[((size_t)y1 * W + x0) * 4 + c], d = (float)img[((size_t)y1 * W + x1) * 4 + c];
        h16 s = (h16)(a * (wx0 * wy0));
        s = (h16)((float)s + (float)(h16)(b * (wx1 * wy0)));
        s = (h16)((float)s + (float)(h16)(cc * (wx0 * wy1)));
        s = (h16)((float)s + (float)(h16)(d * (wx1 * wy1)));
        out[c] = s;
    }
}

/* comparison helpers */
typedef struct { size_t n, exact, within1, bad; double max_abs; double rms_err, rms_ref, max_ref; } Cmp;

static int ulp_dist(h16 a, h16 b) {
    uint16_t x = bits16(a), y = bits16(b);
    int ix = (x & 0x8000) ? -(int)(x & 0x7fff) : (int)x, iy = (y & 0x8000) ? -(int)(y & 0x7fff) : (int)y;
    return abs(ix - iy);
}

static Cmp cmp_tensors_tol(const h16 *got, const h16 *ref, size_t n, double abs_tol) {
    Cmp c;
    memset(&c, 0, sizeof c);
    c.n = n;
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)got[i], r = (double)ref[i];
        if (isnan(g) || isinf(g)) { c.bad++; continue; }
        int u = ulp_dist(got[i], ref[i]);
        if (u == 0) c.exact++;
        double e = fabs(g - r);
        /* A result that is the difference of large terms can sit a few binary16 ulps away from the float64 reference
           when it is tiny (the fp32 accumulation error is absolute, about 1e-7 here); those count as within one ulp
           when the absolute difference is below 2e-6. */
        if (u <= 1 || e <= abs_tol) c.within1++;
        if (e > c.max_abs) c.max_abs = e;
        if (fabs(r) > c.max_ref) c.max_ref = fabs(r);
        se += e * e; sr += r * r;
    }
    c.rms_err = sqrt(se / (double)n);
    c.rms_ref = sqrt(sr / (double)n);
    return c;
}

static Cmp cmp_tensors(const h16 *got, const h16 *ref, size_t n) { return cmp_tensors_tol(got, ref, n, 2e-6); }
