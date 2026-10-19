/* api-layout (with the interface-version-2 fields of fg.engine_api_r2), api-depth-config, api-camera-fallback (a), api-camera-motion (spec fg.engine_api section 7). */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "metalfg.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* The header's static assertions are the layout test; this repeats a few on literal numbers so that a header edit
   which changes both sides of an assertion is still caught. */
static void layout(void) {
    CHECK(sizeof(MetalFGCreateInfo) == 56, "create size");
    CHECK(sizeof(MetalFGFrameInfo) == 352, "frame size (interface version 2)");
    CHECK(offsetof(MetalFGFrameInfo, depth_size) == 336, "depth_size at 336");
    CHECK(offsetof(MetalFGFrameInfo, motion_size) == 344, "motion_size at 344");
    CHECK(METALFG_FRAME_RECORD_V1 == 336 && METALFG_FRAME_RECORD_V2 == 352 && METALFG_INTERFACE_VERSION == 2, "record lengths and version");
    CHECK(offsetof(MetalFGCreateInfo, depth_scale) == 40, "depth_scale at 40");
    CHECK(offsetof(MetalFGFrameInfo, world_to_view) == 208, "w2v");
    CHECK(offsetof(MetalFGFrameInfo, view_to_clip) == 272, "v2c");
    CHECK(offsetof(MetalFGFrameInfo, out_off) == 144, "out_off");
    CHECK(offsetof(MetalFGFrameInfo, frame_index) == 200, "frame_index");
    /* exported helper signatures (assignment fails to compile on a mismatch) */
    int (*f1)(float, float, uint32_t, float *) = metalfg_depth_config;
    int (*f2)(const float *, const float *, const float *, const float *, float, float, float, float, uint32_t,
              uint32_t, float *) = metalfg_camera_matrices;
    CHECK(f1 && f2, "signatures");
}

static void depth_config(void) {
    float o[3];
    struct { float n, f, e[3]; } rows[] = {
        {0.5f, 64.0f, {0.5f, 64.0f, 1.0f / 16}},
        {0.5f, INFINITY, {0.5f, 512.0f, 0.5f}},
        {0.0f, 64.0f, {1.0f, 64.0f, 1.0f / 16}},
        {0.5f, 0.25f, {0.5f, 512.0f, 0.5f}},
    };
    for (int r = 0; r < 4; r++)
        for (uint32_t rev = 0; rev < 2; rev++) {
            CHECK(metalfg_depth_config(rows[r].n, rows[r].f, rev, o) == 0, "ret");
            CHECK(memcmp(o, rows[r].e, sizeof o) == 0, "row %d rev %u: %g %g %g", r, rev, o[0], o[1], o[2]);
        }
}

static int is_identity(const float *m) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            if (m[r * 4 + c] != (r == c ? 1.0f : 0.0f)) return 0;
    return 1;
}

static void camera_fallback(void) {
    float out[32], z[16] = {0}, Pc[16] = {0}, Vi[16] = {0};
    for (uint32_t rev = 0; rev < 2; rev++)
        for (uint32_t reset = 0; reset < 2; reset++) {
            CHECK(metalfg_camera_matrices(NULL, NULL, NULL, NULL, 60, 16.0f / 9, 0.5f, 64, rev, reset, out) == 0, "ret");
            CHECK(is_identity(out) && is_identity(out + 16), "null matrices -> identity");
            CHECK(metalfg_camera_matrices(z, z, z, z, 60, 16.0f / 9, 0.5f, 64, rev, reset, out) == 0, "ret");
            CHECK(is_identity(out) && is_identity(out + 16), "zero matrices -> identity");
            /* only one of V, P zero: fallback replaces both */
            Vi[0] = Vi[5] = Vi[10] = Vi[15] = 1;
            CHECK(metalfg_camera_matrices(Vi, z, Vi, z, 60, 16.0f / 9, 0.5f, 64, rev, reset, out) == 0, "ret");
            CHECK(is_identity(out) && is_identity(out + 16), "half-zero pair -> identity");
        }
    /* constant perspective with non-power-of-two diagonal entries */
    Pc[0] = 7.0f / 5; Pc[5] = 9.0f / 4; Pc[10] = -1.01f; Pc[11] = -1; Pc[14] = -0.505f;
    CHECK(metalfg_camera_matrices(Vi, Pc, Vi, Pc, 60, 16.0f / 9, 0.5f, 64, 0, 0, out) == 0, "ret");
    CHECK(is_identity(out) && is_identity(out + 16), "equal supplied pairs -> exact identity");
}

static void camera_motion(void) {
    float Vp[16] = {0}, Vc[16] = {0}, P[16] = {0}, out[32];
    Vp[0] = Vp[5] = Vp[10] = Vp[15] = 1;
    Vc[0] = Vc[5] = Vc[10] = Vc[15] = 1;
    Vc[12] = 0.5f; /* translation (1/2, 0, 0), column-major, last column */
    P[0] = 7.0f / 5; P[5] = 9.0f / 4; P[10] = -1.01f; P[11] = -1; P[14] = -0.505f;
    CHECK(metalfg_camera_matrices(Vc, P, Vp, P, 60, 16.0f / 9, 0.5f, 64, 0, 0, out) == 0, "ret");
    /* fp64 model: A = P V_prev inv(V_cur) inv(P) = P inv(V_cur) inv(P) (column vectors), then transposed */
    double Pd[16], iP[16] = {0}, T[16] = {0}, A[16], tmp[16];
    /* host column-major -> row-major column-vector form */
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) Pd[r * 4 + c] = P[c * 4 + r];
    extern int metalfg_invert4_d(const double *, double *);
    metalfg_invert4_d(Pd, iP);
    /* inv(V_cur): translation by (-1/2, 0, 0) */
    for (int i = 0; i < 4; i++) T[i * 5] = 1;
    T[3] = -0.5;
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { double s = 0; for (int k = 0; k < 4; k++) s += Pd[r*4+k]*T[k*4+c]; tmp[r*4+c] = s; }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { double s = 0; for (int k = 0; k < 4; k++) s += tmp[r*4+k]*iP[k*4+c]; A[r*4+c] = s; }
    double worst = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double e = A[c * 4 + r], g = out[r * 4 + c];
            double scale = fabs(e) > 1e-30 ? fabs(e) : 1.0;
            double rel = fabs(g - e) / scale;
            if (rel > worst) worst = rel;
        }
    CHECK(worst <= 0x1p-20, "cur->prev relative error %g", worst);
    /* the second matrix is the inverse of the first (row-vector forms multiply to identity) */
    double m1[16], m2[16], pr[16];
    for (int i = 0; i < 16; i++) { m1[i] = out[i]; m2[i] = out[16 + i]; }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { double s = 0; for (int k = 0; k < 4; k++) s += m1[r*4+k]*m2[k*4+c]; pr[r*4+c] = s; }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) CHECK(fabs(pr[r*4+c] - (r == c)) < 1e-5, "inverse (%d,%d)=%g", r, c, pr[r*4+c]);
    /* translation entries sit in the last row (row-vector convention): not in the last column */
    CHECK(fabs(out[12]) > 1e-3 || fabs(out[13]) > 1e-3 || fabs(out[14]) > 1e-3, "translation in last row");
}

int main(void) {
    layout(); depth_config(); camera_fallback(); camera_motion();
    printf(fails ? "t_api_layout: %d FAILURES\n" : "t_api_layout: ok\n", fails);
    return fails != 0;
}
