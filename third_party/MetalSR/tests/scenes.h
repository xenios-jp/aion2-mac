/* Synthetic analytic scenes and rendering rules of sr.test_scenes. */
#ifndef SR_SCENES_H
#define SR_SCENES_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

enum { SC_SMOOTH, SC_STRIPES, SC_MIX, SC_DISCS, SC_DISCS_INF, SC_THIN, SC_PAN, SC_DISOCC, SC_PARALLAX };

static inline double frac_(double v) { return v - floor(v); }
static inline double halton(int i, int b) { double f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

static inline void sc_smooth(double x, double y, double *c) {
    c[0] = 0.5 + 0.4 * sin(2 * M_PI * x / 97); c[1] = 0.5 + 0.4 * sin(2 * M_PI * y / 61); c[2] = 0.5 + 0.4 * sin(2 * M_PI * (x + y) / 131);
}
static inline void sc_stripes(double x, double y, double *c) {
    double th = 20 * M_PI / 180, u = x * cos(th) + y * sin(th);
    double v = frac_(u / 7.3) < 0.5 ? 0.9 : 0.05;
    c[0] = c[1] = c[2] = v;
}
static inline void sc_discs(double x, double y, double Wo, double Ho, int inf, double *c) {
    c[0] = c[1] = c[2] = 0.02;
    for (int k = 0; k < 24; k++) {
        double cx = Wo * frac_(0.1 + 0.6180339887 * k), cy = Ho * frac_(0.2 + 0.7548776662 * k), r = 3 + 27 * frac_(0.4142135624 * k);
        double I = pow(2000.0, k / 23.0);
        if (inf && k == 23) I = 1e5;
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r) { c[0] = I; c[1] = 0.8 * I; c[2] = 0.6 * I; }
    }
}
static inline void sc_thin(double x, double y, double Wo, double Ho, double *c) {
    c[0] = c[1] = c[2] = 0.3;
    for (int k = 0; k < 8; k++) {
        double px = Wo * (k + 1) / 9.0, py = Ho / 2, a = 22.5 * k * M_PI / 180, w = (k % 2 == 0) ? 1.0 : 0.5;
        double dist = fabs(-(x - px) * sin(a) + (y - py) * cos(a));
        if (dist < w / 2) c[0] = c[1] = c[2] = 1.0;
    }
}
static inline void sc_eval(int kind, double x, double y, double Wo, double Ho, double *c) {
    double a[3], b[3];
    switch (kind) {
    case SC_SMOOTH: sc_smooth(x, y, c); break;
    case SC_STRIPES: sc_stripes(x, y, c); break;
    case SC_MIX: sc_smooth(x, y, a); sc_stripes(x, y, b); for (int i = 0; i < 3; i++) c[i] = 0.5 * (a[i] + b[i]); break;
    case SC_DISCS: sc_discs(x, y, Wo, Ho, 0, c); break;
    case SC_DISCS_INF: sc_discs(x, y, Wo, Ho, 1, c); break;
    case SC_THIN: sc_thin(x, y, Wo, Ho, c); break;
    default: c[0] = c[1] = c[2] = 0;
    }
}

/* sample of a full scene at output-plane position (x, y), frame t: colour, depth (standard), motion in output pixels */
static inline void scene_sample(int kind, int inner, double x, double y, int t, double Wo, double Ho, double *c, double *depth, double *mv) {
    const double vx = 1.75, vy = -0.6;
    *depth = 0.5; mv[0] = mv[1] = 0;
    if (kind == SC_PAN) { sc_eval(inner, x - vx * t, y - vy * t, Wo, Ho, c); mv[0] = -vx; mv[1] = -vy; return; }
    if (kind == SC_DISOCC) {
        double side = Ho / 4, cx = Wo / 4 + 3.0 * t, cy = Ho / 2;
        if (fabs(x - cx) <= side / 2 && fabs(y - cy) <= side / 2) { sc_smooth(x - (cx - side / 2), y - (cy - side / 2), c); *depth = 0.1; mv[0] = -3; return; }
        sc_stripes(x, y, c); *depth = 0.9; return;
    }
    if (kind == SC_PARALLAX) {
        if (y < Ho / 2) { sc_stripes(x - 2.0 * t, y, c); *depth = 0.2; mv[0] = -2; }
        else { sc_smooth(x - 0.5 * t, y, c); *depth = 0.8; mv[0] = -0.5; }
        return;
    }
    sc_eval(kind, x, y, Wo, Ho, c);
}

/* render frame f at Wr x Hr: colour rgba float, depth, motion (render- or output-resolution vectors, in pixels of that resolution) */
static inline void scene_render(int kind, int inner, int f, int Wo, int Ho, int Wr, int Hr, int mv_render, int depth_inv,
                                float jx, float jy, float *col, float *depth, float *mot) {
    double sx = (double)Wr / Wo, sy = (double)Hr / Ho;
    for (int j = 0; j < Hr; j++) for (int i = 0; i < Wr; i++) {
        double c[3], d, mv[2];
        scene_sample(kind, inner, (i + 0.5 - jx) / sx, (j + 0.5 - jy) / sy, f, Wo, Ho, c, &d, mv);
        for (int a = 0; a < 3; a++) col[((size_t)j * Wr + i) * 4 + a] = (float)c[a];
        col[((size_t)j * Wr + i) * 4 + 3] = 1;
        depth[(size_t)j * Wr + i] = (float)(depth_inv ? 1 - d : d);
        if (mv_render) { mot[((size_t)j * Wr + i) * 2] = (float)(mv[0] * sx); mot[((size_t)j * Wr + i) * 2 + 1] = (float)(mv[1] * sy); }
    }
    if (!mv_render) for (int j = 0; j < Ho; j++) for (int i = 0; i < Wo; i++) {
        double c[3], d, mv[2];
        scene_sample(kind, inner, (i + 0.5), (j + 0.5), f, Wo, Ho, c, &d, mv);
        mot[((size_t)j * Wo + i) * 2] = (float)mv[0]; mot[((size_t)j * Wo + i) * 2 + 1] = (float)mv[1];
    }
}
/* ground truth: average of 3x3 point samples */
static inline void scene_truth(int kind, int inner, int f, int Wo, int Ho, float *out /* Wo*Ho*3 */) {
    for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) {
        double acc[3] = {0, 0, 0};
        for (int b = 0; b < 3; b++) for (int a = 0; a < 3; a++) {
            double c[3], d, mv[2];
            scene_sample(kind, inner, x + (a + 0.5) / 3, y + (b + 0.5) / 3, f, Wo, Ho, c, &d, mv);
            for (int k = 0; k < 3; k++) acc[k] += c[k] / 9;
        }
        for (int k = 0; k < 3; k++) out[((size_t)y * Wo + x) * 3 + k] = (float)acc[k];
    }
}
#endif
