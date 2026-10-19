/* Depth-proxy defaults and camera matrices (spec fg.engine_api sections 2-3, fg.numerics clip space). */
#include <math.h>
#include <string.h>
#include "metalfg.h"
#include "camera.h"

int metalfg_depth_config(float near_plane, float far_plane, uint32_t depth_reversed, float *out3) {
    (void)depth_reversed; /* the direction enters the proxy elsewhere */
    if (!out3) return 1;
    float scale = (isfinite(near_plane) && near_plane > 0.0f) ? near_plane : 1.0f;
    float part = (isfinite(far_plane) && far_plane > near_plane) ? far_plane : 1024.0f * scale;
    out3[0] = scale;
    out3[1] = part;
    out3[2] = part / 1024.0f;
    return 0;
}

static void mul4(const double a[16], const double b[16], double o[16]) {
    double t[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
            t[r * 4 + c] = s;
        }
    memcpy(o, t, sizeof t);
}

/* Gauss-Jordan inverse with partial pivoting; returns 0 when singular. */
int metalfg_invert4_d(const double m[16], double out[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            a[r][c] = m[r * 4 + c];
            a[r][4 + c] = (r == c) ? 1.0 : 0.0;
        }
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (a[piv][col] == 0.0 || !isfinite(a[piv][col])) return 0;
        if (piv != col)
            for (int c = 0; c < 8; c++) { double t = a[col][c]; a[col][c] = a[piv][c]; a[piv][c] = t; }
        double d = 1.0 / a[col][col];
        for (int c = 0; c < 8; c++) a[col][c] *= d;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double f = a[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r * 4 + c] = a[r][4 + c];
    return 1;
}

static int all_zero(const float *m) {
    if (!m) return 1;
    for (int i = 0; i < 16; i++)
        if (m[i] != 0.0f) return 0;
    return 1;
}

/* Standard perspective matrix in host (column-vector, column-major) form. */
void metalfg_fallback_perspective(float fov_deg, float aspect, float nearp, float farp, int reversed, float *m) {
    memset(m, 0, 16 * sizeof(float));
    double f = 1.0 / tan(0.5 * (double)fov_deg * M_PI / 180.0);
    double zz, zw;
    double n = nearp, fr = farp;
    if (!isfinite(fr)) { /* infinite far plane */
        zz = reversed ? 0.0 : -1.0;
        zw = reversed ? n : -n;
    } else if (reversed) {
        zz = n / (fr - n);
        zw = n * fr / (fr - n);
    } else {
        zz = fr / (n - fr);
        zw = fr * n / (n - fr);
    }
    m[0] = (float)(f / aspect);
    m[5] = (float)f;
    m[10] = (float)zz;   /* column 2, row 2 */
    m[11] = -1.0f;       /* column 2, row 3 */
    m[14] = (float)zw;   /* column 3, row 2 */
}

static void identity16(float *m) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

int metalfg_camera_matrices(const float *cv, const float *cp, const float *pv, const float *pp,
                        float fov_deg, float aspect, float nearp, float farp,
                        uint32_t depth_reversed, uint32_t reset, float *out) {
    if (!out) return 1;
    float cV[16], cP[16], pV[16], pP[16];
    int rev = depth_reversed ? 1 : 0;
    if (all_zero(cv) || all_zero(cp)) {
        identity16(cV);
        metalfg_fallback_perspective(fov_deg, aspect, nearp, farp, rev, cP);
    } else { memcpy(cV, cv, sizeof cV); memcpy(cP, cp, sizeof cP); }
    if (all_zero(pv) || all_zero(pp)) {
        identity16(pV);
        metalfg_fallback_perspective(fov_deg, aspect, nearp, farp, rev, pP);
    } else { memcpy(pV, pv, sizeof pV); memcpy(pP, pp, sizeof pP); }

    if (reset || (memcmp(cV, pV, sizeof cV) == 0 && memcmp(cP, pP, sizeof cP) == 0)) {
        identity16(out);
        identity16(out + 16);
        return 0;
    }
    /* host matrices are column-major: element (r, c) at [c * 4 + r]; build row-major column-vector forms */
    double Vc[16], Pc[16], Vp[16], Pp[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            Vc[r * 4 + c] = cV[c * 4 + r]; Pc[r * 4 + c] = cP[c * 4 + r];
            Vp[r * 4 + c] = pV[c * 4 + r]; Pp[r * 4 + c] = pP[c * 4 + r];
        }
    double iVc[16], iPc[16], iVp[16], iPp[16];
    if (!metalfg_invert4_d(Vc, iVc) || !metalfg_invert4_d(Pc, iPc) || !metalfg_invert4_d(Vp, iVp) || !metalfg_invert4_d(Pp, iPp))
        return 2;
    double A[16], B[16], t[16];
    /* A = P_prev V_prev inv(V_cur) inv(P_cur) */
    mul4(Pp, Vp, t); mul4(t, iVc, t); mul4(t, iPc, A);
    /* inverse: P_cur V_cur inv(V_prev) inv(P_prev) */
    mul4(Pc, Vc, t); mul4(t, iVp, t); mul4(t, iPp, B);
    /* transpose into the engine's row-vector convention */
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            out[r * 4 + c] = (float)A[c * 4 + r];
            out[16 + r * 4 + c] = (float)B[c * 4 + r];
        }
    return 0;
}
