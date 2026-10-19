/* CPU model of pass P1 (sr.input_features_r2): a per-pixel implementation of section 3 that evaluates every expression in
   float64 except the rules fixed in "Required evaluation rules" (nearest-sample coordinates, luma Y, the sum forming P,
   the fma32 of F3, the roundings rz16 / rn16). Bilinear samples emulate the fixed-point sub-texel weights of the GPU.
   The option flags build the wrong variants used as test controls. */
#ifndef SR_REF_P1_H
#define SR_REF_P1_H
#include <stdlib.h>
#include "numerics.h"
#include "../src/shared_types.h"

typedef struct {
    int no_limit;        /* control: no luma limiting of the cubic correction */
    int taps16;          /* control: all 16 taps of the cubic footprint */
    int motion_at_n;     /* control: motion read at the nearest sample, not at the dilated one */
    int floor_round;     /* control: nearest sample by floor(v + 1/2) */
    int own_left;        /* control: motion-memory ownership by the cell's left edge */
    int sub_bits;        /* fractional bits of the sub-texel weights of bilinear samples (0: exact) */
    int smp_round16;     /* bilinear samples of fp16 images are rounded to fp16 (what the sampler returns) */
} RefP1Opts;

typedef struct {
    const float *col, *depth, *mot;   /* colour rgba (Wr x Hr), depth, motion rg (render or output size) */
    const f16 *hist, *memr;           /* H (Wo x Ho x 4), M_read (Wo x Ho x 2) */
    float e_now, e_prev;              /* stored texel values, 0 reads as 1 */
} RefP1In;

static inline double r1_san(double v) { return isfinite(v) ? v : 0.0; }
static inline double r1_luma(double r, double g, double b) { return (double)lumaY((float)r, (float)g, (float)b); }
static inline int r1_clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* nearest render sample and motion in output pixels of output pixel (x, y) */
static inline void ref_p1_motion(const FeatParams *p, const RefP1Opts *o, const RefP1In *in, int x, int y, int n[2], double m[2]) {
    float u = coord_formA(p->sx, p->jx, x), v = coord_formA(p->sy, p->jy, y);
    n[0] = r1_clampi(o->floor_round ? (int)floorf(u + 0.5f) : (int)roundf(u), 0, (int)p->Wr - 1);
    n[1] = r1_clampi(o->floor_round ? (int)floorf(v + 0.5f) : (int)roundf(v), 0, (int)p->Hr - 1);
    double V[2];
    if (p->mv_render) {
        static const int cand[5][2] = {{0, 0}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
        int best = 0; double bd = 0;
        for (int d = 0; d < 5; d++) {
            int qx = r1_clampi(n[0] + cand[d][0], 0, (int)p->Wr - 1), qy = r1_clampi(n[1] + cand[d][1], 0, (int)p->Hr - 1);
            double dd = r1_san(in->depth[(size_t)qy * p->Wr + qx]);
            if (d == 0 || (p->depth_inv ? dd > bd : dd < bd)) { bd = dd; if (d) best = d; }
        }
        if (o->motion_at_n) best = 0;
        int qx = r1_clampi(n[0] + cand[best][0], 0, (int)p->Wr - 1), qy = r1_clampi(n[1] + cand[best][1], 0, (int)p->Hr - 1);
        V[0] = r1_san(in->mot[((size_t)qy * p->Wr + qx) * 2]); V[1] = r1_san(in->mot[((size_t)qy * p->Wr + qx) * 2 + 1]);
    } else {
        V[0] = r1_san(in->mot[((size_t)y * p->Wo + x) * 2]); V[1] = r1_san(in->mot[((size_t)y * p->Wo + x) * 2 + 1]);
    }
    m[0] = (double)(p->Sx * (float)V[0]); m[1] = (double)(p->Sy * (float)V[1]);
}

/* round to binary16 with ties away from zero (what the sampler does with the exact mid-point of two fp16 values) */
static inline double r1_round_ta(double x) {
    double a = fabs(x);
    if (!(a > 0) || a >= 65520.0) return (f16)x;
    int e; frexp(a, &e);                               /* a = m 2^e, m in [0.5, 1) */
    double ulp = ldexp(1.0, (e - 11 < -24) ? -24 : e - 11);
    return copysign(floor(a / ulp + 0.5) * ulp, x);
}

/* bilinear sample of an fp16 image at a texel-space position (centres at i + 1/2); clamp to edge; the normalised
   coordinate is formed in fp32 from the fp32 inverse size, as the engine does */
static inline void r1_bilinear(const f16 *img, int W, int H, int C, double px, double py, float invw, float invh, int bits, int rnd, double *out) {
    float u = (float)px * invw, v = (float)py * invh;
    double tx = (double)u * W - 0.5, ty = (double)v * H - 0.5;
    int x0 = (int)floor(tx), y0 = (int)floor(ty);
    double fx = tx - x0, fy = ty - y0;
    if (bits > 0) { double s = (double)(1 << bits); fx = floor(fx * s + 0.5) / s; fy = floor(fy * s + 0.5) / s; }
    else if (bits < 0) { double s = (double)(1 << -bits); fx = floor(fx * s) / s; fy = floor(fy * s) / s; }   /* negative: truncated weights */
    for (int c = 0; c < C; c++) {
        double a = (double)img[((size_t)r1_clampi(y0, 0, H - 1) * W + r1_clampi(x0, 0, W - 1)) * C + c], b = (double)img[((size_t)r1_clampi(y0, 0, H - 1) * W + r1_clampi(x0 + 1, 0, W - 1)) * C + c];
        double d = (double)img[((size_t)r1_clampi(y0 + 1, 0, H - 1) * W + r1_clampi(x0, 0, W - 1)) * C + c], e = (double)img[((size_t)r1_clampi(y0 + 1, 0, H - 1) * W + r1_clampi(x0 + 1, 0, W - 1)) * C + c];
        out[c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
        if (rnd == 1) out[c] = r1_round_ta(out[c]);   /* the sampler returns fp16-precision results for fp16 textures, ties away from zero */
    }
}

static inline void r1_lim(const double *z, double *o, int no_limit) {
    double y = r1_luma(z[0], z[1], z[2]);
    for (int k = 0; k < 3; k++) o[k] = (!no_limit && y > 1) ? z[k] / y : z[k];
}
static inline void r1_tinv(const double *c, double phi, double *o) {
    double d = 1 - r1_luma(c[0], c[1], c[2]); if (d < phi) d = phi;
    for (int k = 0; k < 3; k++) o[k] = c[k] / d;
}
static inline double r1_tscalar(const double *c, double phi) {
    double y = r1_luma(c[0], c[1], c[2]), s = 1 / (y + 1); if (s < phi) s = phi;
    return y * s;
}
static inline void r1_cubic(double t, double w[4]) {
    w[0] = t * t - (t + t * t * t) / 2; w[1] = 1.5 * t * t * t - 2.5 * t * t + 1; w[3] = (t * t * t - t * t) / 2; w[2] = 1 - w[0] - w[1] - w[3];
}

/* One output pixel: R (rz16 of the warped image value, 4 values with alpha 0), the four unsaturated features F0..F3 as
   saturated for feat (F0, F1 saturated), m (motion), valid flag. */
static inline void ref_p1_pixel(const FeatParams *p, const RefP1Opts *o, const RefP1In *in, int x, int y, f16 *R, double F[4], double m[2], int *valid) {
    int n[2]; ref_p1_motion(p, o, in, x, y, n, m);
    double en = in->e_now == 0 ? 1 : in->e_now, ep = in->e_prev == 0 ? 1 : in->e_prev;
    double k = (double)p->invp * en; if (k == 0) k = 1;
    double r_e = en / ep;
    double C[3], craw[3], c[3];
    for (int q = 0; q < 3; q++) C[q] = r1_san(in->col[((size_t)n[1] * p->Wr + n[0]) * 4 + q]);
    if (p->hdr) { for (int q = 0; q < 3; q++) craw[q] = c[q] = k * C[q]; }
    else { for (int q = 0; q < 3; q++) craw[q] = C[q]; r1_tinv(C, p->phi_hi, c); }
    double F0 = r1_tscalar(c, p->phi_lo);
    float Px = ((float)x + 0.5f) + (float)m[0], Py = ((float)y + 0.5f) + (float)m[1];
    int vld = !p->reset && Px >= 0 && Px < (float)p->Wo && Py >= 0 && Py < (float)p->Ho;
    *valid = vld;
    double Rv[3], a_h = 0;
    for (int q = 0; q < 3; q++) Rv[q] = craw[q] > 0 ? craw[q] : 0;
    if (vld) {
        double pc[2] = {fmin(fmax((double)Px, 0.5), p->Wo - 0.5), fmin(fmax((double)Py, 0.5), p->Ho - 0.5)}, b[2], t[2], w[2][4];
        for (int a = 0; a < 2; a++) { b[a] = floor(pc[a] - 0.5) + 0.5; t[a] = fmin(fmax(pc[a] - b[a], 0), 1); r1_cubic(t[a], w[a]); }
        double h[3], ah;
        if (o->taps16) {
            double acc[4] = {0, 0, 0, 0};
            for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
                int tx = r1_clampi((int)(b[0] - 0.5) - 1 + i, 0, p->Wo - 1), ty = r1_clampi((int)(b[1] - 0.5) - 1 + j, 0, p->Ho - 1);
                for (int q = 0; q < 4; q++) acc[q] += w[0][i] * w[1][j] * (double)in->hist[((size_t)ty * p->Wo + tx) * 4 + q];
            }
            for (int q = 0; q < 3; q++) h[q] = acc[q];
            ah = acc[3];
        } else {
            double w12x = w[0][1] + w[0][2], w12y = w[1][1] + w[1][2], ox = w[0][2] / w12x, oy = w[1][2] / w12y;
            double pos[5][2] = {{b[0] - 1, b[1] + oy}, {b[0] + 2, b[1] + oy}, {b[0] + ox, b[1] - 1}, {b[0] + ox, b[1] + 2}, {b[0] + ox, b[1] + oy}};
            double W5[5] = {w[0][0] * w12y, w[0][3] * w12y, w[1][0] * w12x, w[1][3] * w12x, w12x * w12y}, s[5][4];
            for (int q = 0; q < 5; q++) r1_bilinear(in->hist, p->Wo, p->Ho, 4, pos[q][0], pos[q][1], p->inv_wo, p->inv_ho, o->sub_bits, o->smp_round16, s[q]);
            double Sn = 1 - (w[0][0] + w[0][3]) * (w[1][0] + w[1][3]), lm[3], lk[3];
            r1_lim(s[4], lm, o->no_limit);
            for (int q = 0; q < 3; q++) h[q] = Sn * s[4][q];
            ah = W5[4] * s[4][3];
            for (int kk = 0; kk < 4; kk++) {
                r1_lim(s[kk], lk, o->no_limit);
                for (int q = 0; q < 3; q++) h[q] += W5[kk] * (lk[q] - lm[q]);
                ah += W5[kk] * s[kk][3];
            }
            for (int q = 0; q < 3; q++) h[q] /= Sn;
            ah /= Sn;
        }
        a_h = ah;
        if (p->hdr) { double ti[3]; r1_tinv(h, p->phi_hi, ti); for (int q = 0; q < 3; q++) h[q] = r_e * ti[q]; }
        for (int q = 0; q < 3; q++) Rv[q] = h[q] > 0 ? h[q] : 0;
    }
    for (int q = 0; q < 3; q++) R[q] = rz16((float)Rv[q]);
    R[3] = (f16)0;
    double e[3]; if (p->hdr) { for (int q = 0; q < 3; q++) e[q] = Rv[q]; } else r1_tinv(Rv, p->phi_hi, e);
    double F1 = r1_tscalar(e, p->phi_lo), F2 = vld ? a_h : F0;
    if (!isfinite(F2)) F2 = 0;
    double mp[2] = {m[0], m[1]};
    if (!p->reset) {
        double qr[2] = {fmin(fmax((double)((float)Px * p->sx), 0.5), p->Wr - 0.5), fmin(fmax((double)((float)Py * p->sy), 0.5), p->Hr - 0.5)};
        r1_bilinear(in->memr, p->Wo, p->Ho, 2, qr[0], qr[1], p->inv_wo, p->inv_ho, o->sub_bits, o->smp_round16, mp);
    }
    float dx = (float)(m[0] - mp[0]), dy = (float)(m[1] - mp[1]);
    double F3 = log1p((double)p->g_mv * sqrt((double)fmaf(dx, dx, dy * dy)));
    if (!isfinite(F3)) F3 = 0;
    F[0] = fmin(fmax(F0, 0), 1); F[1] = fmin(fmax(F1, 0), 1); F[2] = F2; F[3] = F3;
}

/* Whole pass for the L0 rows Y with Y % row_stride == 0 (all rows for 1): feat [H0][W0][16] and R [Ho][Wo][4] for the
   pixels of those rows, written into zero-initialised arrays. Returns the per-pixel validity in `valid` (Wo x Ho, may be NULL). */
static inline void ref_p1(const FeatParams *p, const RefP1Opts *o, const RefP1In *in, int H0, int row_stride, f16 *feat, f16 *R, uint8_t *valid) {
    for (int J = 0; J < (int)((p->Ho + 1) / 2); J++) {
        if (((J >> 1) % row_stride) != 0) continue;
        for (int I = 0; I < (int)((p->Wo + 1) / 2); I++) {
            double sum[4] = {0, 0, 0, 0};
            for (int b = 0; b < 2; b++) for (int a = 0; a < 2; a++) {
                int x = 2 * I + a, y = 2 * J + b;
                if (x >= (int)p->Wo || y >= (int)p->Ho) continue;
                double F[4], m[2]; int v;
                ref_p1_pixel(p, o, in, x, y, R + ((size_t)y * p->Wo + x) * 4, F, m, &v);
                if (valid) valid[(size_t)y * p->Wo + x] = (uint8_t)v;
                for (int q = 0; q < 4; q++) sum[q] += F[q];
            }
            for (int q = 0; q < 4; q++) feat[(((size_t)(J >> 1) * p->W0 + (I >> 1)) * 16) + 8 * (J & 1) + 4 * (I & 1) + q] = (f16)(sum[q] / 4);
        }
    }
    (void)H0;
}

static inline int r1_owner(float s, int i, int n, int own_left);
/* motion memory written by one call: every cell (i, j) of the render rectangle receives rn16 of the vector of its owner
   pixel; cells outside the rectangle are not touched (mw is Wo x Ho x 2) */
static inline void ref_p1_memory(const FeatParams *p, const RefP1Opts *o, const RefP1In *in, f16 *mw) {
    double *mm = malloc((size_t)p->Wo * p->Ho * 16);
    for (int y = 0; y < (int)p->Ho; y++) for (int x = 0; x < (int)p->Wo; x++) { int n[2]; ref_p1_motion(p, o, in, x, y, n, mm + ((size_t)y * p->Wo + x) * 2); }
    for (int j = 0; j < (int)p->Hr; j++) for (int i = 0; i < (int)p->Wr; i++) {
        int ox = r1_owner(p->sx, i, p->Wo, o->own_left), oy = r1_owner(p->sy, j, p->Ho, o->own_left);
        if (ox < 0 || oy < 0) continue;
        for (int k = 0; k < 2; k++) mw[((size_t)j * p->Wo + i) * 2 + k] = (f16)mm[((size_t)oy * p->Wo + ox) * 2 + k];
    }
    free(mm);
}

/* Owner pixel of render-grid cell i along one axis for scale s (fp32) and pixel count n: s x < i + 1/2 <= s (x + 1), decided in
   exact arithmetic (s has 24 significant bits and x fits in 16, so the products are exact in float64). Returns -1 when none.
   own_left: the control that uses the cell's left edge i in place of its centre. */
static inline int r1_owner(float s, int i, int n, int own_left) {
    double c = own_left ? (double)i : (double)i + 0.5;
    int x0 = (int)floor(c / (double)s) - 2;
    for (int x = x0 < 0 ? 0 : x0; x < x0 + 6 && x < n; x++)
        if ((double)s * x < c && c <= (double)s * (x + 1)) return x;
    return -1;
}
#endif
