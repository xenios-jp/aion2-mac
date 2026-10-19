/* F07 to F11 (spec fg.prev2curr section 7): pc-static, pc2-slow-background, pc2-fast-background, pc-camera, pc-behind, pc-block, pc-newly-dynamic, pc-hole-fill,
   pc-gap, pc-tie, pc-nearest, pc-zero-sign, pc-reset, pc-random. A CPU reference of section 3 (IEEE fp32 for the decisions,
   double for the pyramid) is compared with the engine on every scene; the explicit expectations of the spec are checked on
   top, and the controls are mutations of the reference that must differ from the engine's output. */
#import <Foundation/Foundation.h>
#pragma clang fp contract(off)
#include "texutil.h"
#include "chain.h"

static int fails;
static NnRig R;

enum { MUT_NONE = 0, MUT_PAYLOAD_REV, MUT_FAR_CLAMP, MUT_SIGMA0_POS, MUT_NO_SIM, MUT_NO_KIND, MUT_FAR_WINS, MUT_NO_FILL, MUT_NO_CAMOK,
       MUT_NO_DEPTH_AGREE, MUT_ONE_CELL, MUT_NO_MATRICES, MUT_DYN_GE, MUT_DYN_DISPLAY, MUT_DYN_H48, MUT_REC_UV };

#define WM 320
#define HM 180

typedef struct {
    int Wm, Hm;
    float H;
    int zr;                       /* depth rectangle = zr x the Mot grid (0 or 1: equal); fg.motion_prep_r2 record positions */
    float Zs, Zp, Zsep, eps_n;
    int reversed;
    float d_dyn, d_sim, th_cam;
    float c2p[16], p2c[16];
    float *depth, *mvx, *mvy;     /* current depth (value d) and mvo in Mot pixels */
    float *odepth;                /* previous depth */
    uint8_t *dynold;
} Scene;

typedef struct {
    uint8_t *dyn, *nsc;
    float *vx, *vy;
} Res;

/* position of the depth record of Mot index i of n (code 0): the centre of texel zr i (fg.motion_prep_r2, ceil(P) - 1 for the ratio zr = 2), in normalised
   coordinates of the depth rectangle; MUT_REC_UV is the former uv(p) */
static float rpos(const Scene *s, int i, int n, int mut) {
    if (s->zr == 2 && mut != MUT_REC_UV) return ((float)(2 * i) + 0.5f) / (float)(2 * n);
    return ((float)i + 0.5f) / (float)n;
}

static float lam(const Scene *s, float d) {
    float n = s->reversed ? d : 1.0f - d;
    n = fmaxf(n, s->eps_n);
    return s->Zs / n;
}

static float cam(const Scene *s, float ux, float uy, float d, const float *M, float *cx, float *cy, int mut) {
    float nx = 2.0f * ux - 1.0f, ny = 1.0f - 2.0f * uy;
    if (mut == MUT_NO_MATRICES) { *cx = *cy = 0; return 1; }
    float qx = nx * M[0] + ny * M[4] + d * M[8] + M[12];
    float qy = nx * M[1] + ny * M[5] + d * M[9] + M[13];
    float w = nx * M[3] + ny * M[7] + d * M[11] + M[15];
    if (w > 0.0f) { *cx = (qx / w - nx) * (float)s->Wm / 2.0f; *cy = (ny - qy / w) * (float)s->Hm / 2.0f; }
    else { *cx = *cy = 4096.0f; }
    return w;
}

static uint16_t hb16(float v) { h16 h = (h16)v; uint16_t b; memcpy(&b, &h, 2); return b; }
static float sg(float v) { return v > 0 ? 1.0f : -1.0f; }
static float rh(float v) { return (float)(h16)v; }

static float bil(const float *img, int W, int H, float u, float v) {
    float x = u * W - 0.5f, y = v * H - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
#define AT(X, Y) img[(Y < 0 ? 0 : Y >= H ? H - 1 : Y) * W + (X < 0 ? 0 : X >= W ? W - 1 : X)]
    return (AT(x0, y0) * (1 - fx) + AT(x0 + 1, y0) * fx) * (1 - fy) + (AT(x0, y0 + 1) * (1 - fx) + AT(x0 + 1, y0 + 1) * fx) * fy;
}

static void reference(const Scene *s, int mut, Res *r) {
    int W = s->Wm, H = s->Hm;
    size_t n = (size_t)W * H;
    uint32_t *key = malloc(n * 4), *pay = malloc(n * 4);
    for (size_t i = 0; i < n; i++) key[i] = pay[i] = 0x7FFFFFFFu;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            float d = s->depth[i], mx = s->mvx[i], my = s->mvy[i];
            float ux = rpos(s, x, W, mut), uy = rpos(s, y, H, mut), cx, cy;
            float w = cam(s, ux, uy, d, s->c2p, &cx, &cy, mut);
            float dx = mx - cx, dy = my - cy;
            /* controls of fg.prev2curr_r2: >= instead of >, the motion measured in display pixels (twice the Mot pixels), H taken as the Mot height 48 of the spec's grid */
            float d2 = dx * dx + dy * dy, dh = mut == MUT_DYN_H48 ? 48.0f : s->H;
            if (mut == MUT_DYN_DISPLAY) d2 *= 4.0f;
            float thr_dyn = fmaxf(1.0f, (mx * mx + my * my) * s->d_dyn / (dh * dh));
            int dyn = (w <= 0) || (mut == MUT_DYN_GE ? d2 >= thr_dyn : d2 > thr_dyn);
            r->dyn[i] = dyn;
            float l = lam(s, d), k = dyn ? l : l * w;
            uint32_t kb;
            memcpy(&kb, &k, 4);
            uint32_t kk = (kb & ~1u) | (dyn ? 0u : 1u);
            uint32_t pl = ((uint32_t)hb16(mx) << 16) | hb16(my);
            float sx = mut == MUT_SIGMA0_POS ? (mx >= 0 ? 1 : -1) : sg(mx), sy = mut == MUT_SIGMA0_POS ? (my >= 0 ? 1 : -1) : sg(my);
            int tx = (int)truncf(mx), ty = (int)truncf(my);
            float thr = fmaxf(1.0f, (mx * mx + my * my) * s->d_sim / (s->H * s->H));
            for (int j = 0; j < 2; j++)
                for (int a = 0; a < 2; a++) {
                    if (mut == MUT_ONE_CELL && (a || j)) continue;
                    int qx = x + a * (int)sx, qy = y + j * (int)sy;
                    qx = qx < 0 ? 0 : qx >= W ? W - 1 : qx; qy = qy < 0 ? 0 : qy >= H ? H - 1 : qy;
                    int Tx = qx + tx, Ty = qy + ty;
                    if (Tx < 0 || Ty < 0 || Tx >= W || Ty >= H) continue;
                    if (a || j) {
                        float ex = s->mvx[(size_t)qy * W + qx] - mx, ey = s->mvy[(size_t)qy * W + qx] - my;
                        if (mut != MUT_NO_SIM && !(ex * ex + ey * ey < thr)) continue;
                    }
                    size_t c = (size_t)Ty * W + Tx;
                    uint32_t k2 = mut == MUT_NO_KIND ? (kk & ~1u) : kk;
                    uint32_t p2 = mut == MUT_PAYLOAD_REV ? ~pl : pl;
                    int better;
                    if (mut == MUT_FAR_WINS) better = key[c] == 0x7FFFFFFFu || k2 > key[c] || (k2 == key[c] && p2 < pay[c]);
                    else better = k2 < key[c] || (k2 == key[c] && p2 < pay[c]);
                    if (better) { key[c] = k2; pay[c] = p2; }
                }
        }
    /* F08 */
    uint8_t *b1 = calloc(n, 1), *b0 = calloc(n, 1);
    float *vx = r->vx, *vy = r->vy;
    float *emx = malloc(n * 4), *emy = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { emx[i] = s->mvx[i]; emy[i] = s->mvy[i]; }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            int empty = key[i] == 0x7FFFFFFFu;
            float mx = 0, my = 0;
            if (!empty) {
                uint32_t pl = mut == MUT_PAYLOAD_REV ? ~pay[i] : pay[i];
                uint16_t bx = pl >> 16, by = pl & 0xFFFF;
                h16 hx, hy;
                memcpy(&hx, &bx, 2); memcpy(&hy, &by, 2);
                mx = (float)hx; my = (float)hy;
            }
            int kind_dyn = !empty && (key[i] & 1u) == 0;
            uint32_t cb = key[i] & ~1u;
            float clsk;
            memcpy(&clsk, &cb, 4);
            int pd = s->dynold[i] != 0;
            float uvx = ((float)x + 0.5f) / W, uvy = ((float)y + 0.5f) / H;
            float dold = s->odepth[i], pcx, pcy, rux = rpos(s, x, W, mut), ruy = rpos(s, y, H, mut);
            float wold = cam(s, rux, ruy, dold, s->p2c, &pcx, &pcy, mut);
            float lold = pd ? lam(s, dold) : lam(s, dold) * wold;
            float rx = uvx + pcx / W, ry = uvy + pcy / H;
            float ex = bil(emx, W, H, rx, ry) + pcx, ey = bil(emy, W, H, rx, ry) + pcy;
            int camOK = ex * ex + ey * ey < s->th_cam;
            if (mut == MUT_NO_CAMOK) camOK = 0;
            float a1 = fminf(fmaxf(clsk, 0), s->Zp), a2 = fminf(fmaxf(lold, 0), s->Zp);
            int mismatch = fabsf(a1 - a2) > s->Zsep;
            if (mut == MUT_NO_DEPTH_AGREE) mismatch = 0;
            int bad = mismatch || empty;
            float px, py;
            if (pd) { px = -mx; py = -my; }
            else if (!bad && !camOK && kind_dyn) { px = -mx; py = -my; }
            else { px = pcx; py = pcy; }
            vx[i] = rh(px); vy[i] = rh(py);
            r->nsc[i] = empty;
            b1[i] = pd && bad; b0[i] = pd;
        }
    uint8_t *hole = calloc(n, 1), *wt = calloc(n, 1);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int closed = 1;
            for (int ey = -1; ey <= 1 && closed; ey++) for (int ex = -1; ex <= 1 && closed; ex++) {
                int dil = 0;
                for (int dy = -1; dy <= 1 && !dil; dy++) for (int dx = -1; dx <= 1 && !dil; dx++) {
                    int qx = x + ex + dx, qy = y + ey + dy;
                    if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
                    dil = b1[(size_t)qy * W + qx];
                }
                closed = dil;
            }
            hole[(size_t)y * W + x] = closed;
            wt[(size_t)y * W + x] = b0[(size_t)y * W + x] && !closed;
        }
    /* pyramid in double: S_v, S_c per level */
    int lw[16], lh[16], K = 0;
    double *S[16];
    lw[0] = W; lh[0] = H;
    while (lw[K] > 1 || lh[K] > 1) { K++; lw[K] = (lw[K - 1] + 1) / 2; lh[K] = (lh[K - 1] + 1) / 2; }
    S[0] = calloc(n * 3, 8);
    for (size_t i = 0; i < n; i++) if (wt[i]) { S[0][i * 3] = vx[i]; S[0][i * 3 + 1] = vy[i]; S[0][i * 3 + 2] = 1; }
    for (int k = 1; k <= K; k++) {
        S[k] = calloc((size_t)lw[k] * lh[k] * 3, 8);
        for (int y = 0; y < lh[k - 1]; y++) for (int x = 0; x < lw[k - 1]; x++)
            for (int c = 0; c < 3; c++) S[k][((size_t)(y / 2) * lw[k] + x / 2) * 3 + c] += S[k - 1][((size_t)y * lw[k - 1] + x) * 3 + c];
    }
    double *F[16];
    for (int k = K; k >= 1; k--) {
        F[k] = calloc((size_t)lw[k] * lh[k] * 3, 8);
        for (int y = 0; y < lh[k]; y++) for (int x = 0; x < lw[k]; x++) {
            double *s3 = &S[k][((size_t)y * lw[k] + x) * 3], *o = &F[k][((size_t)y * lw[k] + x) * 3];
            if (s3[2] > 0 || k == K) {
                o[0] = s3[0] / fmax(s3[2], 1e-12); o[1] = s3[1] / fmax(s3[2], 1e-12); o[2] = s3[2] / pow(4.0, k);
            } else {
                int nx = x >> 1, ny = y >> 1, fx = (x & 1) ? nx + 1 : nx - 1, fy = (y & 1) ? ny + 1 : ny - 1;
                if (fx < 0) fx = nx;
                if (fy < 0) fy = ny;
                if (mut == MUT_FAR_CLAMP) { if (fx >= lw[k + 1]) fx = lw[k + 1] - 1; if (fy >= lh[k + 1]) fy = lh[k + 1] - 1; }
                double uc = 0, uvx = 0, uvy = 0;
                for (int j = 0; j < 2; j++) for (int a = 0; a < 2; a++) {
                    int cx = a ? fx : nx, cy = j ? fy : ny;
                    if (cx >= lw[k + 1] || cy >= lh[k + 1]) continue;
                    double wgt = (a ? 0.25 : 0.75) * (j ? 0.25 : 0.75);
                    double *f = &F[k + 1][((size_t)cy * lw[k + 1] + cx) * 3];
                    uc += wgt * f[2]; uvx += wgt * f[2] * f[0]; uvy += wgt * f[2] * f[1];
                }
                double dv = fmin(fmax(uc, 1e-12), 1.0);
                o[0] = uvx / dv; o[1] = uvy / dv; o[2] = uc;
            }
        }
    }
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        size_t i = (size_t)y * W + x;
        if (!hole[i] || mut == MUT_NO_FILL) continue;
        double ux = 0, uy = 0;
        if (K >= 1) {
            int nx = x >> 1, ny = y >> 1, fx = (x & 1) ? nx + 1 : nx - 1, fy = (y & 1) ? ny + 1 : ny - 1;
            if (fx < 0) fx = nx;
            if (fy < 0) fy = ny;
            if (mut == MUT_FAR_CLAMP) { if (fx >= lw[1]) fx = lw[1] - 1; if (fy >= lh[1]) fy = lh[1] - 1; }
            double uc = 0, uvx = 0, uvy = 0;
            for (int j = 0; j < 2; j++) for (int a = 0; a < 2; a++) {
                int cx = a ? fx : nx, cy = j ? fy : ny;
                if (cx >= lw[1] || cy >= lh[1]) continue;
                double wgt = (a ? 0.25 : 0.75) * (j ? 0.25 : 0.75);
                double *f = &F[1][((size_t)cy * lw[1] + cx) * 3];
                uc += wgt * f[2]; uvx += wgt * f[2] * f[0]; uvy += wgt * f[2] * f[1];
            }
            double dv = fmin(fmax(uc, 1e-12), 1.0);
            ux = uvx / dv; uy = uvy / dv;
        }
        vx[i] = rh((float)ux); vy[i] = rh((float)uy);
    }
    for (int k = 0; k <= K; k++) { free(S[k]); if (k >= 1) free(F[k]); }
    free(key); free(pay); free(b1); free(b0); free(emx); free(emy); free(hole); free(wt);
}

/* ---- engine side ---- */
static MetalFGChain *CH;

static Res *new_res(void) {
    Res *r = calloc(1, sizeof *r);
    r->dyn = calloc(WM * HM, 1); r->nsc = calloc(WM * HM, 1);
    r->vx = calloc(WM * HM, 4); r->vy = calloc(WM * HM, 4);
    return r;
}
static void free_res(Res *r) { free(r->dyn); free(r->nsc); free(r->vx); free(r->vy); free(r); }

static Scene *new_scene(void) {
    Scene *s = calloc(1, sizeof *s);
    s->Wm = WM; s->Hm = HM; s->H = 360;
    s->Zs = 4; s->Zp = 64; s->Zsep = 2; s->eps_n = 1e-7f;
    s->d_dyn = s->d_sim = 1000; s->th_cam = 1;
    for (int i = 0; i < 16; i++) s->c2p[i] = s->p2c[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    s->depth = malloc(WM * HM * 4); s->mvx = calloc(WM * HM, 4); s->mvy = calloc(WM * HM, 4);
    s->odepth = malloc(WM * HM * 4); s->dynold = calloc(WM * HM, 1);
    for (int i = 0; i < WM * HM; i++) s->depth[i] = s->odepth[i] = 0.75f;
    return s;
}
static void free_scene(Scene *s) { free(s->depth); free(s->mvx); free(s->mvy); free(s->odepth); free(s->dynold); free(s); }

static uint32_t pack_depth(float d) { return rec_pack_c(d, 0); }

static void run_engine(const Scene *s, Res *o) {
    id<MTLTexture> dep = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, WM, HM), dold = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, WM, HM);
    id<MTLTexture> mvo = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, WM, HM), mvf = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    id<MTLTexture> dynC = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, WM, HM), dynO = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, WM, HM);
    id<MTLTexture> nsc = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, WM, HM);
    uint32_t *words = malloc(WM * HM * 4);
    for (int i = 0; i < WM * HM; i++) words[i] = pack_depth(s->depth[i]);
    gpu_upload(dep, words, 4);
    for (int i = 0; i < WM * HM; i++) words[i] = pack_depth(s->odepth[i]);
    gpu_upload(dold, words, 4);
    free(words);
    tex_set(mvo, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->mvx[y * WM + x] / WM; v[1] = s->mvy[y * WM + x] / HM; });
    tex_set(dynO, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->dynold[y * WM + x]; });
    id<MTLBuffer> sc = [R.gpu.dev newBufferWithLength:WM * HM * 8 options:MTLResourceStorageModeShared];
    memset(sc.contents, 0xFF, sc.length);
    uint32_t *p = sc.contents;
    for (int i = 0; i < WM * HM * 2; i++) p[i] = 0x7FFFFFFFu;
    id<MTLBuffer> hb = [R.gpu.dev newBufferWithLength:WM * HM options:MTLResourceStorageModeShared];
    MetalFGP2CArgs a;
    memset(&a, 0, sizeof a);
    a.Wm = WM; a.Hm = HM; a.depthW = WM * (s->zr ? s->zr : 1); a.depthH = HM * (s->zr ? s->zr : 1); a.H = s->H; a.d_dyn = s->d_dyn; a.d_sim = s->d_sim; a.th_cam = s->th_cam;
    a.D = (MetalFGDepthConst){s->Zs, s->Zp, s->Zsep, s->eps_n, (uint32_t)s->reversed};
    memcpy(a.c2p, s->c2p, 64); memcpy(a.p2c, s->p2c, 64);
    a.writeNsc = 1;
    nn_run(&R, ^(MetalFGEnc *e) { [CH p2c:e args:a sc:sc hb:hb depCur:dep depOld:dold dynCur:dynC dynOld:dynO mvo:mvo mvf:mvf nsc:nsc]; });
    float *v = tex_get(mvf, 2), *d = tex_get(dynC, 1), *n = tex_get(nsc, 1);
    for (int i = 0; i < WM * HM; i++) { o->vx[i] = v[2 * i]; o->vy[i] = v[2 * i + 1]; o->dyn[i] = d[i] > 0.5f; o->nsc[i] = n[i] > 0.5f; }
    free(v); free(d); free(n);
}

/* returns the number of pixels where the engine differs from the reference */
static int diff(const Res *a, const Res *b, double tol, int mv_only_nonexcluded) {
    int bad = 0;
    for (int i = 0; i < WM * HM; i++)
        bad += a->dyn[i] != b->dyn[i] || a->nsc[i] != b->nsc[i] || fabs(a->vx[i] - b->vx[i]) > tol || fabs(a->vy[i] - b->vy[i]) > tol;
    (void)mv_only_nonexcluded;
    return bad;
}

static void judge(const char *name, const Scene *s, const int *ctl, int nctl, int max_bad) {
    Res *g = new_res(), *r = new_res(), *c = new_res();
    run_engine(s, g);
    reference(s, MUT_NONE, r);
    int bad = diff(g, r, 0x1p-5, 0);
    CHECKC(bad <= max_bad, "%s: %d pixels differ from the reference (allowed %d)", name, bad, max_bad);
    if (bad > max_bad) { int shown = 0; for (int i = 0; i < WM * HM && shown < 6; i++) if (g->dyn[i] != r->dyn[i] || g->nsc[i] != r->nsc[i] || fabs(g->vx[i] - r->vx[i]) > 0x1p-5 || fabs(g->vy[i] - r->vy[i]) > 0x1p-5) { printf("    (%d,%d) engine dyn %d nsc %d v (%g,%g)  ref dyn %d nsc %d v (%g,%g)\n", i % WM, i / WM, g->dyn[i], g->nsc[i], g->vx[i], g->vy[i], r->dyn[i], r->nsc[i], r->vx[i], r->vy[i]); shown++; } }
    for (int i = 0; i < nctl; i++) {
        reference(s, ctl[i], c);
        int d = diff(g, c, 0x1p-5, 0);
        CHECKC(d >= (max_bad ? 10 : 1), "%s: control %d differs at only %d pixels", name, ctl[i], d);
    }
    free_res(g); free_res(r); free_res(c);
}

#define IDX(x, y) ((y) * WM + (x))

static void block(Scene *s, int x0, int x1, int y0, int y1, float depth, float mx, float my) {
    for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) { s->depth[IDX(x, y)] = depth; s->mvx[IDX(x, y)] = mx; s->mvy[IDX(x, y)] = my; }
}

static Res *eval(const Scene *s) { Res *r = new_res(); run_engine(s, r); return r; }

static void run_cases(void) {
    Res *g;
    /* pc-static */
    { Scene *s = new_scene(); judge("pc-static", s, NULL, 0, 0); g = eval(s);
      int bad = 0; for (int i = 0; i < WM * HM; i++) bad += g->dyn[i] || g->nsc[i] || g->vx[i] != 0 || g->vy[i] != 0;
      CHECKC(bad == 0, "pc-static explicit: %d", bad); free_res(g); free_scene(s); }
    /* pc2-slow-background, pc2-fast-background (fg.prev2curr_r2): d_dyn 2400 and H 96 as in the spec (threshold max(1, m2 x 0.26)); a motion of one Mot pixel
       is static under the zero camera, a motion of two is dynamic. The Mot grid of this harness is larger than the spec's, the threshold does not depend on it. */
    for (int fast = 0; fast < 2; fast++) {
        Scene *s = new_scene();
        s->H = 96; s->d_dyn = 2400;
        for (int i = 0; i < WM * HM; i++) s->mvx[i] = fast ? -2.0f : -1.0f;
        int ctl_slow[] = {MUT_DYN_GE, MUT_DYN_DISPLAY}, ctl_fast[] = {MUT_DYN_H48};
        judge(fast ? "pc2-fast-background" : "pc2-slow-background", s, fast ? ctl_fast : ctl_slow, fast ? 1 : 2, 0);
        g = eval(s);
        int bad = 0;
        for (int i = 0; i < WM * HM; i++) {
            if (fast) bad += g->dyn[i] != 1;
            else bad += g->dyn[i] || g->vx[i] != 0 || g->vy[i] != 0;
        }
        CHECKC(bad == 0, "%s explicit: %d pixels wrong", fast ? "pc2-fast-background" : "pc2-slow-background", bad);
        free_res(g); free_scene(s);
    }
    /* pc2-record-position (fg.motion_prep_r2, tau(p) = q - uv(p)): the depth rectangle is twice the Mot grid, so the record of a pixel sits on texel 2 x, a quarter
       Mot pixel to the left of uv(p). A horizontal scale in the camera makes the displacement depend on the position; the flag of Mot column 163 flips with it. */
    { Scene *s = new_scene(); s->zr = 2; s->c2p[0] = 1.5f;
      for (int i = 0; i < WM * HM; i++) s->mvx[i] = 0.7f;
      int ctl[] = {MUT_REC_UV};
      judge("pc2-record-position", s, ctl, 1, 0); free_scene(s); }
    /* pc-camera */
    { Scene *s = new_scene(); float d = 5.0f / 320;
      s->c2p[12] = d; s->p2c[12] = -d;
      for (int i = 0; i < WM * HM; i++) s->mvx[i] = 2.5f;
      int ctl[] = {MUT_NO_MATRICES};
      judge("pc-camera", s, ctl, 1, 0); g = eval(s);
      int bad = 0;
      for (int y = 0; y < HM; y++) for (int x = 0; x < WM; x++) {
          int i = IDX(x, y);
          bad += g->dyn[i] || fabs(g->vx[i] + 2.5) > 0x1p-7 || g->nsc[i] != (x < 2);
      }
      CHECKC(bad == 0, "pc-camera explicit: %d", bad); free_res(g); free_scene(s); }
    /* pc-behind */
    { Scene *s = new_scene(); s->c2p[15] = -1;
      judge("pc-behind", s, NULL, 0, 0); g = eval(s);
      int bad = 0; for (int i = 0; i < WM * HM; i++) bad += !g->dyn[i];
      CHECKC(bad == 0, "pc-behind explicit: %d", bad); free_res(g); free_scene(s); }
    /* pc-block, pc-newly-dynamic, pc-hole-fill, pc-reset */
    for (int variant = 0; variant < 4; variant++) {
        Scene *s = new_scene();
        block(s, 100, 107, 60, 67, 0.5f, -3.25f, 0);
        for (int y = 60; y <= 67; y++) for (int x = 97; x <= 104; x++) { s->dynold[IDX(x, y)] = 1; s->odepth[IDX(x, y)] = 0.5f; }
        const char *nm = "pc-block";
        if (variant == 1) {
            nm = "pc-newly-dynamic";
            memset(s->dynold, 0, WM * HM);
            for (int y = 60; y <= 67; y++) s->odepth[IDX(104, y)] = 0.75f;
        } else if (variant == 2) {
            nm = "pc-hole-fill";
            for (int y = 60; y <= 67; y++) for (int x = 97; x <= 107; x++) { s->dynold[IDX(x, y)] = 1; s->odepth[IDX(x, y)] = x <= 104 ? 0.5f : 0.75f; }
        } else if (variant == 3) {
            nm = "pc-reset";
            memset(s->dynold, 1, WM * HM);
            for (int i = 0; i < WM * HM; i++) s->odepth[i] = 0;
        }
        int ctl_b[] = {MUT_NO_SIM}, ctl_n[] = {MUT_NO_CAMOK, MUT_NO_DEPTH_AGREE}, ctl_h[] = {MUT_NO_FILL}, ctl_r[] = {MUT_NO_MATRICES};
        judge(nm, s, variant == 0 ? ctl_b : variant == 1 ? ctl_n : variant == 2 ? ctl_h : NULL, variant == 0 ? 1 : variant == 1 ? 2 : variant == 2 ? 1 : 0, 0);
        (void)ctl_r;
        g = eval(s);
        int bad = 0;
        for (int y = 0; y < HM; y++) for (int x = 0; x < WM; x++) {
            int i = IDX(x, y), in = y >= 60 && y <= 67;
            double want = 0;
            if (variant == 0 || variant == 2) { if (in && x >= 97 && x <= (variant == 0 ? 104 : 107)) want = 3.25; }
            else if (variant == 1) { if (in && x >= 100 && x <= 103) want = 3.25; }
            if (fabs(g->vx[i] - want) > 0x1p-5 || g->vy[i] != 0) bad++;
            if (variant != 1 && (g->nsc[i] != (in && x >= 105 && x <= 107))) bad++;
            if (variant != 3 || 1) if (variant != 1 && g->dyn[i] != (in && x >= 100 && x <= 107)) bad++;
        }
        CHECKC(bad == 0, "%s explicit: %d", nm, bad);
        free_res(g); free_scene(s);
    }
    /* pc-gap */
    { Scene *s = new_scene();
      block(s, 100, 120, 60, 67, 0.5f, -3.25f, 0); block(s, 121, 140, 60, 67, 0.5f, -2.5f, 0);
      for (int y = 60; y <= 67; y++) for (int x = 100; x <= 140; x++) s->odepth[IDX(x, y)] = 0.5f;
      int ctl[] = {MUT_ONE_CELL};
      judge("pc-gap", s, ctl, 1, 0); g = eval(s);
      int bad = 0; for (int y = 60; y <= 67; y++) bad += g->nsc[IDX(118, y)] || fabs(g->vx[IDX(118, y)] - 2.5) > 0x1p-5;
      CHECKC(bad == 0, "pc-gap explicit: %d", bad); free_res(g); free_scene(s); }
    /* pc-tie */
    { Scene *s = new_scene();
      s->mvx[IDX(50, 50)] = 2.5f; s->mvx[IDX(54, 50)] = -2.5f; s->dynold[IDX(52, 50)] = 1;
      int ctl[] = {MUT_NO_KIND, MUT_PAYLOAD_REV};
      judge("pc-tie", s, ctl, 2, 0); g = eval(s);
      CHECKC(fabs(g->vx[IDX(52, 50)] + 2.5) < 0x1p-7 && g->vy[IDX(52, 50)] == 0, "pc-tie explicit: %g", g->vx[IDX(52, 50)]); free_res(g); free_scene(s); }
    /* pc-nearest */
    { Scene *s = new_scene();
      s->depth[IDX(50, 50)] = 0.5f; s->mvx[IDX(50, 50)] = 2.5f; s->dynold[IDX(52, 50)] = 1; s->odepth[IDX(52, 50)] = 0.5f;
      judge("pc-nearest", s, NULL, 0, 0); g = eval(s);
      CHECKC(fabs(g->vx[IDX(52, 50)] + 2.5) < 0x1p-7, "pc-nearest explicit: %g", g->vx[IDX(52, 50)]);
      Res *c = new_res(); reference(s, MUT_FAR_WINS, c);
      CHECKC(fabs(c->vx[IDX(52, 50)]) < 0x1p-7, "pc-nearest control is sensitive (%g)", c->vx[IDX(52, 50)]);
      free_res(c); free_res(g); free_scene(s); }
    /* pc-zero-sign */
    { Scene *s = new_scene();
      s->depth[IDX(70, 40)] = s->depth[IDX(72, 40)] = 0.75f; s->mvy[IDX(70, 40)] = s->mvy[IDX(72, 40)] = 2.75f;
      s->depth[IDX(71, 40)] = 0.5f; s->mvy[IDX(71, 40)] = 2.5f;
      for (int x = 70; x <= 72; x++) { s->dynold[IDX(x, 42)] = 1; s->odepth[IDX(x, 42)] = x < 72 ? 0.5f : 0.75f; }
      int ctl[] = {MUT_SIGMA0_POS};
      judge("pc-zero-sign", s, ctl, 1, 0); g = eval(s);
      CHECKC(fabs(g->vy[IDX(70, 42)] + 2.5) < 0x1p-7 && fabs(g->vy[IDX(71, 42)] + 2.5) < 0x1p-7 && fabs(g->vy[IDX(72, 42)] + 2.75) < 0x1p-7, "pc-zero-sign explicit: %g %g %g",
             g->vy[IDX(70, 42)], g->vy[IDX(71, 42)], g->vy[IDX(72, 42)]);
      free_res(g); free_scene(s); }
    /* pc-random */
    { Scene *s = new_scene();
      uint64_t st = 1;
      const float mset[4] = {-2.5f, 0, 0.75f, 2.25f};
#define NEXT() (st = st * 6364136223846793005ull + 1442695040888963407ull, (uint32_t)(st >> 33))
      for (int ty = 0; ty < (HM + 7) / 8; ty++) for (int tx = 0; tx < WM / 8; tx++) {
          float d = (NEXT() & 1) ? 0.5f : 0.75f;
          float mx = mset[NEXT() & 3], my = mset[NEXT() & 3];
          for (int y = ty * 8; y < ty * 8 + 8 && y < HM; y++) for (int x = tx * 8; x < tx * 8 + 8; x++) { s->depth[IDX(x, y)] = d; s->mvx[IDX(x, y)] = mx; s->mvy[IDX(x, y)] = my; }
          int half = NEXT() & 1;
          for (int y = ty * 8; y < ty * 8 + 8 && y < HM; y++) for (int x = tx * 8; x < tx * 8 + 8; x++) s->odepth[IDX(x, y)] = half ? d : (d == 0.5f ? 0.75f : 0.5f);
      }
      for (int ty = 0; ty < (HM + 3) / 4; ty++) for (int tx = 0; tx < WM / 4; tx++) {
          int dyn = NEXT() & 1;
          for (int y = ty * 4; y < ty * 4 + 4 && y < HM; y++) for (int x = tx * 4; x < tx * 4 + 4; x++) s->dynold[IDX(x, y)] = dyn;
      }
      int ctl[] = {MUT_PAYLOAD_REV, MUT_FAR_CLAMP};
      judge("pc-random", s, ctl, 2, WM * HM * 2 / 100);
      free_scene(s); }
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        CH = [[MetalFGChain alloc] initWithDevice:g.dev library:R.lib Wm:WM Hm:HM];
        if (!CH) { printf("chain kernels missing\n"); return 2; }
        run_cases();
    }
    printf(fails ? "g_p2c: %d FAILURES\n" : "g_p2c: ok\n", fails);
    return fails != 0;
}
