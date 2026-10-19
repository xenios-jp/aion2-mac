/* CPU model of pass P12 (sr.head_filter, sr.head_filter_r4): the network head, the per-quad parameters, the filter, the
   two-stage blend with the exceptional-value rules, and the stores. The filter is evaluated in float64, the blend with
   fp16 emulation (rn16, fma16, finiteness tests), IEEE fp32 elsewhere. The option flags build the wrong variants used as
   test controls. */
#ifndef SR_REF_P12_H
#define SR_REF_P12_H
#include <stdlib.h>
#include "ref.h"
#include "../src/shared_types.h"

typedef struct {
    int old_overflow;    /* control: the superseded overflow rule (product of the components) in place of the table */
    int rn16_store;      /* control: rn16 instead of sat16 at the stores */
    int rec_clamp;       /* control: NaN recurrent scalar through NaN-suppressing min and max */
    int no_unusable;     /* control: no unusable-colour rule, NaN parameters reach the blend through suppressing clamps */
    int taps_n;          /* control (path W): 3 x 3 taps (1) */
    int form_swap;       /* control: form A coordinates on path W (form B on path N) */
    int *kinds;          /* optional counters of exceptional pixels: Q kept with a component >= 64, Q' = H, Q' = 0, saturated store, rec = 0 from a NaN parameter */
} RefP12Opts;

typedef struct {
    double sigma_x, sigma_y, rho, alpha_s, beta, k_xx, k_xy, k_yy;   /* k: -1/2 inverse covariance (natural-exponent form) */
    f16 a16;
    int colour_ok, rec_ok;
} RefQuad;

static inline double r12_sig(double v) { return 1.0 / (1.0 + exp(-v)); }
static inline double r12_q(double v, const FiltParams *p) {
    return fabs(v) <= p->q_clip ? v * (p->q_a + v * v) / (p->q_a + p->q_b * (v * v)) : copysign(1.0, v);
}

/* the ten parameters v_0..v_9 of quad t of an L0 pixel with head output `o40` */
static inline void ref_quad_v(const f16 *o40, int t, f16 v[10]) { for (int q = 0; q < 10; q++) v[q] = o40[8 * (q / 2) + 2 * t + (q % 2)]; }

static inline RefQuad ref_quad(const f16 v[10], const FiltParams *p, const RefP12Opts *o) {
    RefQuad r; memset(&r, 0, sizeof r);
    r.colour_ok = !(isnan((float)v[0]) || isnan((float)v[1]) || isnan((float)v[2]) || isnan((float)v[8]) || isnan((float)v[9]));
    r.rec_ok = !(isnan((float)v[3]) || isnan((float)v[4]) || isnan((float)v[5]) || isnan((float)v[6]) || isnan((float)v[7]));
    if (r.colour_ok || o->no_unusable) {
        r.sigma_x = (double)p->sigma_min + (double)p->sigma_span * r12_sig((double)v[0]);
        r.sigma_y = (double)p->sigma_min + (double)p->sigma_span * r12_sig((double)v[1]);
        double ax = r.sigma_x * r.sigma_x, ay = r.sigma_y * r.sigma_y;
        r.rho = (double)p->rho_max * r12_q((double)v[2], p);
        double cxy = r.rho * sqrt(ax * ay), inv = 1 / (ax * ay - cxy * cxy);
        r.k_xx = -(ay / 2) * inv; r.k_xy = cxy * inv; r.k_yy = -(ax / 2) * inv;
        r.alpha_s = (double)p->A * r12_sig((double)v[8]);
        double bb = 0.5 + (double)p->g_beta * (r12_sig((double)v[9]) - 0.5);
        r.beta = o->no_unusable ? fmin(fmax(bb, 0), 1) : (bb < 0 ? 0 : bb > 1 ? 1 : bb);
    }
    float s = (float)tanh((double)v[3]);
    for (int q = 4; q < 8; q++) s = s + (float)tanh((double)v[q]);
    r.a16 = (f16)(s * 0.2f);
    return r;
}

static inline f16 r12_sat16(float w) { if (isnan(w)) return (f16)0; if (w > 65504.0f) w = 65504.0f; if (w < -65504.0f) w = -65504.0f; return (f16)w; }
static inline int r12_fin(f16 v) { return isfinite((float)v); }
static inline float r12_san(float v) { return isfinite(v) ? v : 0.0f; }

/* Tap value of the shared tile for a colour texel (already read as fp32 rgba) */
static inline void r12_tap(const float *C, float k_c, const FiltParams *p, f16 *c) {
    float s[4]; for (int q = 0; q < 4; q++) s[q] = r12_san(C[q]);
    if (p->hdr) {
        float r = k_c * s[0], g = k_c * s[1], b = k_c * s[2];
        float y = lumaY(r, g, b), t = 1.0f / (1.0f + y); t = t > p->phi_lo ? t : p->phi_lo;
        c[0] = (f16)(r * t); c[1] = (f16)(g * t); c[2] = (f16)(b * t); c[3] = (f16)s[3];
    } else for (int q = 0; q < 4; q++) c[q] = (f16)s[q];
}

/* Filter and blend for the pixels of the output rows y with y % row_stride == 0. out and hist_new are Wo x Ho x 4. */
static inline void ref_p12(const FiltParams *p, const RefP12Opts *o, int W0, const f16 *oh, const float *col, const f16 *R, float e_now, int row_stride,
                           f16 *out, f16 *hist_new) {
    float e = e_now == 0 ? 1.0f : e_now, k_c = p->invp * e; if (k_c == 0) k_c = 1.0f;
    int Rr = (p->wide && !o->taps_n) ? 2 : 1;
    for (int y = 0; y < (int)p->Ho; y += 1) {
        if (y % row_stride) continue;
        for (int x = 0; x < (int)p->Wo; x++) {
            int X = x >> 2, Y = y >> 2, t = ((x >> 1) & 1) + 2 * ((y >> 1) & 1);
            f16 v[10]; ref_quad_v(oh + ((size_t)Y * W0 + X) * 40, t, v);
            RefQuad q = ref_quad(v, p, o);
            int useB = o->form_swap ? !p->wide : p->wide;
            float u = useB ? coord_formB(p->sx, p->jx, x) : coord_formA(p->sx, p->jx, x);
            float vv = useB ? coord_formB(p->sy, p->jy, y) : coord_formA(p->sy, p->jy, y);
            int nx = (int)roundf(u), ny = (int)roundf(vv);
            float dxc = (float)nx - u, dyc = (float)ny - vv;
            int covered = (p->cov_y * fabsf(dyc) < 0.5f) && (p->cov_x * fabsf(dxc) < 0.5f);
            /* history value */
            float g[3]; for (int k = 0; k < 3; k++) g[k] = r12_san((float)R[((size_t)y * p->Wo + x) * 4 + k]);
            if (p->hdr) { float d = 1.0f / (lumaY(g[0], g[1], g[2]) + 1.0f); d = d > p->phi_lo ? d : p->phi_lo; for (int k = 0; k < 3; k++) g[k] *= d; }
            f16 H[3] = {(f16)g[0], (f16)g[1], (f16)g[2]};
            /* filter */
            f16 Q[3] = {0, 0, 0};
            const float *Cn_px = col + ((size_t)(ny < 0 ? 0 : ny >= (int)p->Hr ? (int)p->Hr - 1 : ny) * p->Wr + (nx < 0 ? 0 : nx >= (int)p->Wr ? (int)p->Wr - 1 : nx)) * 4;
            float c_a = r12_san(Cn_px[3]);
            if (q.colour_ok || o->no_unusable) {
                double acc[3] = {0, 0, 0}, Ws = 0; f16 Cn[3] = {0, 0, 0};
                for (int j = -Rr; j <= Rr; j++) for (int i = -Rr; i <= Rr; i++) {
                    int tx = nx + i, ty = ny + j;
                    tx = tx < 0 ? 0 : tx >= (int)p->Wr ? (int)p->Wr - 1 : tx; ty = ty < 0 ? 0 : ty >= (int)p->Hr ? (int)p->Hr - 1 : ty;
                    f16 c[4]; r12_tap(col + ((size_t)ty * p->Wr + tx) * 4, k_c, p, c);
                    double dx = (double)dxc + i, dy = (double)dyc + j;
                    double w = exp(q.k_xx * dx * dx + q.k_yy * dy * dy + q.k_xy * dx * dy);
                    for (int k = 0; k < 3; k++) acc[k] += w * (double)c[k];
                    Ws += w;
                    if (!i && !j) for (int k = 0; k < 3; k++) Cn[k] = c[k];
                }
                f16 Cg[3]; for (int k = 0; k < 3; k++) Cg[k] = (f16)(acc[k] * (1 / Ws));
                f16 lambda = covered ? (f16)q.alpha_s : (f16)0, beta = (f16)q.beta;
                for (int k = 0; k < 3; k++) {
                    f16 M = fma16((double)lambda, (double)(f16)((double)Cn[k] - (double)H[k]), (double)H[k]);
                    Q[k] = fma16((double)beta, (double)(f16)((double)Cg[k] - (double)M), (double)M);
                }
            }
            /* colour after the blend */
            f16 Qp[3];
            int q_ok = (q.colour_ok || o->no_unusable) && r12_fin(Q[0]) && r12_fin(Q[1]) && r12_fin(Q[2]);
            int h_ok = r12_fin(H[0]) && r12_fin(H[1]) && r12_fin(H[2]);
            if (o->old_overflow) {
                f16 prod = (f16)((float)Q[0] * (float)Q[1] * (float)Q[2]);
                int ovf = isinf((float)prod);
                for (int k = 0; k < 3; k++) Qp[k] = ovf ? H[k] : Q[k];
            } else for (int k = 0; k < 3; k++) Qp[k] = q_ok ? Q[k] : h_ok ? H[k] : (f16)0;
            f16 rec;
            if (o->rec_clamp) { float a = (float)q.a16; rec = (f16)fminf(fmaxf(a, -1.0f), 1.0f); }
            else rec = q.rec_ok ? (f16)fminf(fmaxf((float)q.a16, -1.0f), 1.0f) : (f16)0;
            /* tonemaps and stores */
            float Z[3] = {(float)Qp[0], (float)Qp[1], (float)Qp[2]}, O[3], K[3];
            if (p->hdr) {
                float d = 1.0f / fmaxf(1.0f - lumaY(Z[0], Z[1], Z[2]), p->phi_lo);
                for (int k = 0; k < 3; k++) Z[k] *= d;
                float o_s = p->pre / e, ky = 1.0f / (lumaY(Z[0], Z[1], Z[2]) + 1.0f); ky = fmaxf(ky, p->phi_hi);
                for (int k = 0; k < 3; k++) { O[k] = fmaxf(o_s * Z[k], 0.0f); K[k] = Z[k] * ky; }
            } else for (int k = 0; k < 3; k++) { O[k] = fmaxf(Z[k], 0.0f); K[k] = Z[k]; }
            f16 *po = out + ((size_t)y * p->Wo + x) * 4, *ph = hist_new + ((size_t)y * p->Wo + x) * 4;
            for (int k = 0; k < 3; k++) { po[k] = o->rn16_store ? (f16)O[k] : r12_sat16(O[k]); ph[k] = o->rn16_store ? (f16)K[k] : r12_sat16(K[k]); }
            po[3] = o->rn16_store ? (f16)c_a : r12_sat16(c_a);
            ph[3] = rec;
            if (o->kinds) {
                float w[7] = {O[0], O[1], O[2], c_a, K[0], K[1], K[2]};
                int sat = 0;
                for (int k = 0; k < 7; k++) if (!isnan(w[k]) && isinf((float)(f16)w[k])) sat = 1;
                if (q_ok && fmaxf(fmaxf(fabsf((float)Q[0]), fabsf((float)Q[1])), fabsf((float)Q[2])) >= 64.0f) o->kinds[0]++;
                if (!q_ok && h_ok) o->kinds[1]++;
                if (!q_ok && !h_ok) o->kinds[2]++;
                if (sat) o->kinds[3]++;
                if (!q.rec_ok) o->kinds[4]++;
            }
        }
    }
}

/* Network head of P12 on decoded inputs: d1 (m8 codes at L1), s0 (m8 codes at L0), result o [H0][W0][40] */
static inline void ref_head(const uint8_t *img, const RefLevels *L, int variant, const uint8_t *d1, const uint8_t *s0, f16 *o, int two_round) {
    size_t n0 = (size_t)L->W[0] * L->H[0], n1 = (size_t)L->W[1] * L->H[1];
    f16 *d1d = malloc(n1 * 64), *s0d = malloc(n0 * 64), *x = malloc(n0 * 64), *h = malloc(n0 * 64);
    m8_decode_buf(d1, n1 * 32, d1d); m8_decode_buf(s0, n0 * 32, s0d);
    ref_up2_add(d1d, L->W[1], L->H[1], 32, s0d, variant, x);
    ref_conv1x1(x, (int)n0, 32, wt_of(img, 30), wt_of(img, 31), 32, h, two_round ? 2 | 4 : 2);
    ref_conv1x1(h, (int)n0, 32, wt_of(img, 32), wt_of(img, 33), 40, o, two_round ? 4 : 0);
    free(d1d); free(s0d); free(x); free(h);
}
#endif
