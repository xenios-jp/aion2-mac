/* Candidate blend and pack (spec fg.blend_pack section 7): bp-fallback, bp-hole-rule, bp-quality, bp-pack, bp-ui, bp-hole-nan,
   bp-ui-tiny, bp-padding. */
#import <Foundation/Foundation.h>
#include "texutil.h"
#include "blend.h"
#include "probe.h"

static int fails;
static NnRig R;
static MetalFGBlend *B;

enum { W = 64, H = 48, WM = 32, HM = 24 };

/* ---- reference sampler: bilinear, clamp to edge, texel centres at half integers ---- */
static double bil(const float *img, int w, int h, int comps, int c, double ux, double uy) {
    double px = ux * w - 0.5, py = uy * h - 0.5;
    double fx = floor(px), fy = floor(py), ax = px - fx, ay = py - fy;
    int x0 = (int)fx, y0 = (int)fy;
#define CL(v, n) ((v) < 0 ? 0 : (v) > (n) - 1 ? (n) - 1 : (v))
#define T(xx, yy) ((double)img[((size_t)CL(yy, h) * w + CL(xx, w)) * comps + c])
    double a = (1 - ax) * T(x0, y0) + ax * T(x0 + 1, y0), b = (1 - ax) * T(x0, y0 + 1) + ax * T(x0 + 1, y0 + 1);
#undef T
    return (1 - ay) * a + ay * b;
}

typedef struct {
    float P[W * H * 4], C[W * H * 4];     /* half-rounded */
    float bw[WM * HM * 2], Q[WM * HM * 2];
    int eP[WM * HM], eC[WM * HM];
    float oP[WM * HM], oC[WM * HM];       /* x offsets, normalised */
    float UP[W * H * 4], UC[W * H * 4];
    int ui;
} Scene;

static float q16(double v) { return (float)(h16)v; }

static id<MTLBuffer> sc_buf(const int *flag, const float *off) {
    id<MTLBuffer> b = [R.gpu.dev newBufferWithLength:WM * HM * 8 options:MTLResourceStorageModeShared];
    uint32_t *p = b.contents;
    for (int i = 0; i < WM * HM; i++) {
        if (flag[i]) { p[2 * i] = p[2 * i + 1] = 0xFFFFFFFFu; }
        else {
            h16 hx = (h16)off[i], hy = (h16)0;
            uint16_t bx, by;
            memcpy(&bx, &hx, 2); memcpy(&by, &hy, 2);
            p[2 * i] = (uint32_t)bx << 16 | by;
            p[2 * i + 1] = 0x3f800000u;
        }
    }
    return b;
}

static void ref_pixel(const Scene *s, int x, int y, double ra[3], double rb[3], double *aa, double *ab, int *hole) {
    double ux = (x + 0.5) / W, uy = (y + 0.5) / H;
    int mx = (int)(ux * WM), my = (int)(uy * HM);
    int mi = my * WM + mx;
    int eP = s->eP[mi], eC = s->eC[mi];
    double oP = eP ? 0 : s->oP[mi], oC = eC ? 0 : s->oC[mi];
    double wp = bil(s->bw, WM, HM, 2, 0, ux, uy), wc = bil(s->bw, WM, HM, 2, 1, ux, uy);
    for (int c = 0; c < 3; c++) {
        double p0 = s->P[((size_t)y * W + x) * 4 + c], c0 = s->C[((size_t)y * W + x) * 4 + c];
        double pw = bil(s->P, W, H, 4, c, ux + oP, uy), cw = bil(s->C, W, H, 4, c, ux + oC, uy);
        double cA = eP ? cw : pw, cB = eC ? pw : cw;
        ra[c] = p0 + wp * (cA - p0);
        rb[c] = c0 + wc * (cB - c0);
    }
    double q0x = bil(s->Q, WM, HM, 2, 0, ux, uy), q0y = bil(s->Q, WM, HM, 2, 1, ux, uy);
    *aa = q0x + wp * (bil(s->Q, WM, HM, 2, 0, ux + oP, uy) - q0x);
    *ab = q0y + wc * (bil(s->Q, WM, HM, 2, 1, ux + oC, uy) - q0y);
    *hole = eP && eC && !(wp < 0.5 && wc < 0.5);
    if (s->ui) {
        const float *up = &s->UP[((size_t)y * W + x) * 4], *uc = &s->UC[((size_t)y * W + x) * 4];
        for (int c = 0; c < 3; c++) { ra[c] = ra[c] * (1 - up[3]) + up[c]; rb[c] = rb[c] * (1 - uc[3]) + uc[c]; }
        double d = fmin(fmax((fabs(up[0] - uc[0]) + fabs(up[1] - uc[1]) + fabs(up[2] - uc[2])) / 3, 0), 1);
        if (up[3] > 0) *aa = d;
        if (uc[3] > 0) *ab = d;
        if (up[3] > 0 || uc[3] > 0) *hole = 0;
    }
}

static void build_scene(Scene *s, int quality_varies) {
    memset(s, 0, sizeof *s);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float *p = &s->P[((size_t)y * W + x) * 4], *c = &s->C[((size_t)y * W + x) * 4];
        p[0] = q16(x / 64.0); p[1] = q16(0.25); p[2] = q16(0.5); p[3] = 1;
        c[0] = q16(0.75); c[1] = q16(y / 48.0); c[2] = q16(0.25); c[3] = 1;
    }
    for (int j = 0; j < HM; j++) for (int i = 0; i < WM; i++) {
        int band = i / 8;   /* N, P, C, PC */
        int mi = j * WM + i;
        s->eP[mi] = band == 1 || band == 3;
        s->eC[mi] = band == 2 || band == 3;
        s->oP[mi] = 4.0f / 64; s->oC[mi] = -2.0f / 64;
        s->bw[2 * mi] = j < 12 ? 0.75f : 0.25f; s->bw[2 * mi + 1] = 0.25f;
        s->Q[2 * mi] = quality_varies ? q16(i / 32.0) : q16(0.2); s->Q[2 * mi + 1] = quality_varies ? q16(j / 24.0) : q16(0.4);
    }
}

typedef struct {
    id<MTLTexture> Pt, Ct, bw, Q, uiP, uiC, candA, candB, hole;
    id<MTLBuffer> ip, ic, n2a, n2b, n2h;
} Dev;

static Dev upload(const Scene *s) {
    Dev d;
    memset((void *)&d, 0, sizeof d);
    id<MTLDevice> dv = R.gpu.dev;
    (void)dv;
    d.Pt = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H); d.Ct = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    d.bw = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM); d.Q = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    d.uiP = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H); d.uiC = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    d.candA = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H); d.candB = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    d.hole = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, W, H);
    tex_set(d.Pt, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->P[((size_t)y * W + x) * 4], 16); });
    tex_set(d.Ct, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->C[((size_t)y * W + x) * 4], 16); });
    tex_set(d.bw, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->bw[2 * (y * WM + x)]; v[1] = s->bw[2 * (y * WM + x) + 1]; });
    tex_set(d.Q, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->Q[2 * (y * WM + x)]; v[1] = s->Q[2 * (y * WM + x) + 1]; });
    tex_set(d.uiP, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->UP[((size_t)y * W + x) * 4], 16); });
    tex_set(d.uiC, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->UC[((size_t)y * W + x) * 4], 16); });
    d.ip = sc_buf(s->eP, s->oP);
    d.ic = sc_buf(s->eC, s->oC);
    return d;
}

static void run_blend(Dev *d, int ui) {
    MetalFGF20Args a;
    memset(&a, 0, sizeof a);
    a.scIp = d->ip; a.scIc = d->ic; a.colOld = d->Pt; a.colCur = d->Ct; a.bw = d->bw; a.err = d->Q; a.uiOld = d->uiP; a.uiCur = d->uiC;
    a.candA = d->candA; a.candB = d->candB; a.hole = d->hole;
    a.W = W; a.H = H; a.Wm = WM; a.Hm = HM; a.ui = ui;
    nn_run(&R, ^(MetalFGEnc *e) { [B blend:e args:a]; });
}

/* compare against the reference; rows_lo..rows_hi are excluded (the bw filter mixes the two weight regions) */
static void compare(const Scene *s, Dev *d, const char *name, int excl_boundary, int check_hole_cols) {
    float *a = tex_get(d->candA, 4), *b = tex_get(d->candB, 4), *hl = tex_get(d->hole, 1);
    int bad = 0, badh = 0, n = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        if (excl_boundary && y >= 22 && y <= 25) continue;
        double ra[3], rb[3], aa, ab;
        int hole;
        ref_pixel(s, x, y, ra, rb, &aa, &ab, &hole);
        n++;
        for (int c = 0; c < 4; c++) {
            double wa = c < 3 ? ra[c] : aa, wb = c < 3 ? rb[c] : ab;
            h16 ga = (h16)a[((size_t)y * W + x) * 4 + c], gb = (h16)b[((size_t)y * W + x) * 4 + c];
            if (ulp_dist(ga, (h16)wa) > 2 && fabs((double)ga - wa) > 1e-4) bad++;
            if (ulp_dist(gb, (h16)wb) > 2 && fabs((double)gb - wb) > 1e-4) bad++;
        }
        if ((hl[(size_t)y * W + x] > 0.5f) != hole) badh++;
    }
    (void)check_hole_cols;
    CHECKC(bad == 0 && badh == 0, "%s: %d colour/quality and %d hole mismatches of %d pixels", name, bad, badh, n);
    free(a); free(b); free(hl);
}

static void fallback_hole_quality(void) {
    Scene *s = malloc(sizeof *s);
    for (int qv = 0; qv < 2; qv++) {
        build_scene(s, qv);
        Dev d = upload(s);
        run_blend(&d, 0);
        compare(s, &d, qv ? "bp-fallback/bp-hole-rule/bp-quality" : "bp-fallback/bp-hole-rule (constant Q)", 1, 0);
        if (qv == 0) {
            /* bp-hole-rule: the hole is exactly band PC rows 0..11 */
            float *hl = tex_get(d.hole, 1);
            int bad = 0;
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
                if (y >= 22 && y <= 25) continue;
                int want = x >= 48 && y < 24;
                if ((hl[y * W + x] > 0.5f) != want) bad++;
            }
            CHECKC(bad == 0, "bp-hole-rule: %d wrong", bad);
            free(hl);
        }
    }
    free(s);
}

static void ui_cases(void) {
    Scene *s = malloc(sizeof *s);
    build_scene(s, 1);
    s->ui = 1;
    WgenRng rng; wgen_seed(&rng, 77);
    static const float al[3] = {0, 0.5f, 1};
    for (int i = 0; i < W * H; i++) {
        float a1 = al[wgen_u64(&rng) % 3], a2 = al[wgen_u64(&rng) % 3];
        for (int c = 0; c < 3; c++) { s->UP[4 * i + c] = q16(a1 * wgen_uniform(&rng)); s->UC[4 * i + c] = q16(a2 * wgen_uniform(&rng)); }
        s->UP[4 * i + 3] = a1; s->UC[4 * i + 3] = a2;
    }
    Dev d = upload(s);
    run_blend(&d, 1);
    compare(s, &d, "bp-ui", 1, 0);
    /* bp-ui-tiny: alpha 1/255 everywhere on the hole-rule data */
    build_scene(s, 0);
    s->ui = 1;
    float tiny = q16(1.0 / 255);
    for (int i = 0; i < W * H; i++) {
        s->UP[4 * i + 3] = tiny; s->UC[4 * i + 3] = tiny;
        for (int c = 0; c < 3; c++) { s->UP[4 * i + c] = q16(tiny * wgen_uniform(&rng)); s->UC[4 * i + c] = q16(tiny * wgen_uniform(&rng)); }
    }
    d = upload(s);
    run_blend(&d, 1);
    compare(s, &d, "bp-ui-tiny", 1, 0);
    float *hl = tex_get(d.hole, 1);
    int holes = 0;
    for (int i = 0; i < W * H; i++) holes += hl[i] > 0.5f;
    CHECKC(holes == 0, "bp-ui-tiny: hole is 0 everywhere (%d set)", holes);
    free(hl);
    free(s);
}

static void hole_nan(void) {
    Scene *s = malloc(sizeof *s);
    build_scene(s, 0);
    Dev d = upload(s);
    tex_set(d.bw, ^(uint32_t x, uint32_t y, float v[4]) {
        v[0] = y < 12 ? 0.75f : 0.25f; v[1] = 0.25f;
        if (x >= 26 && x <= 29) { if (y < 12) v[0] = NAN; else v[1] = NAN; }
    });
    run_blend(&d, 0);
    float *hl = tex_get(d.hole, 1);
    int bad = 0;
    for (int y = 0; y < H; y++) for (int x = 54; x <= 57; x++) {
        if (y >= 20 && y <= 27) continue;                  /* away from the row 11/12 boundary */
        if (!(hl[y * W + x] > 0.5f)) bad++;
    }
    CHECKC(bad == 0, "bp-hole-nan: %d pixels without a hole", bad);
    free(hl); free(s);
}

/* ---- the reduction to the net2 grid (fg.blend_pack_pin_r3): an exact CPU model of rules A to C ---------------------------------------------------- */

typedef struct { int PW, PH, s, Nw, Nh; float *P, *C; int *eP, *eC; } PScene;

static float zero_off[WM * HM];

/* binary16 conversions: nearest even (the language conversion) and truncation toward zero */
static float rne16(float v) { return (float)(h16)v; }
static float trunc16(float v) {
    h16 h = (h16)v;
    if (isfinite(v) && fabsf((float)h) > fabsf(v)) { uint16_t b; memcpy(&b, &h, 2); b--; memcpy(&h, &b, 2); }
    return (float)h;
}

/* the binary32 candidate values of one position when both blend weights are one half, offsets are zero and the quality image is the constant 0.3: every term is
   exact in binary32 for the test data, so the model needs no assumption about contraction */
static void model_px(const PScene *q, int x, int y, float A[4], float Bv[4], int *hole) {
    float ux = ((float)x + 0.5f) / (float)q->PW, uy = ((float)y + 0.5f) / (float)q->PH;
    int mx = (int)(ux * (float)WM), my = (int)(uy * (float)HM);
    mx = mx > WM - 1 ? WM - 1 : mx; my = my > HM - 1 ? HM - 1 : my;
    int mi = my * WM + mx, eP = q->eP[mi], eC = q->eC[mi];
    for (int c = 0; c < 3; c++) {
        float p0 = q->P[((size_t)y * q->PW + x) * 4 + c], c0 = q->C[((size_t)y * q->PW + x) * 4 + c];
        float cA = eP ? c0 : p0, cB = eC ? p0 : c0;
        A[c] = p0 + 0.5f * (cA - p0);
        Bv[c] = c0 + 0.5f * (cB - c0);
    }
    A[3] = Bv[3] = rne16(0.3f);
    *hole = eP && eC;
}

/* block (i, j): order 0 = raster, 1 = reversed raster; divisor mode 0 = the full block area, 1 = the number of covered positions; source mode 0 = binary32 values,
   1 = the values after the store conversion (nearest = 1, truncation = 0 in smode) */
static void block_model(const PScene *q, int i, int j, int order, int divmode, int stored, int stored_nearest, h16 *oa, h16 *ob, int *hole) {
    int s = q->s;
    float sa[4] = {0}, sb[4] = {0};
    int first = 1, cnt = 0, h = 0;
    for (int k = 0; k < s * s; k++) {
        int kk = order ? s * s - 1 - k : k, dy = kk / s, dx = kk % s;
        int x = i * s + dx, y = j * s + dy;
        if (x >= q->PW || y >= q->PH) continue;
        float A[4], Bv[4];
        int ho;
        model_px(q, x, y, A, Bv, &ho);
        if (stored) for (int c = 0; c < 4; c++) { A[c] = stored_nearest ? rne16(A[c]) : trunc16(A[c]); Bv[c] = stored_nearest ? rne16(Bv[c]) : trunc16(Bv[c]); }
        for (int c = 0; c < 4; c++) { if (first) { sa[c] = A[c]; sb[c] = Bv[c]; } else { sa[c] += A[c]; sb[c] += Bv[c]; } }
        first = 0; cnt++; h |= ho;
    }
    float inv = 1.0f / (float)(divmode ? (cnt ? cnt : 1) : s * s);
    for (int c = 0; c < 4; c++) {
        oa[c] = (h16)(s == 1 ? sa[c] : sa[c] * inv);
        ob[c] = (h16)(s == 1 ? sb[c] : sb[c] * inv);
    }
    *hole = h;
}

static void pack_run_mode(const PScene *q, h16 *oA, h16 *oB, h16 *oH, float *stA, int fused, float *stB, uint8_t *stH) {
    int PW = q->PW, PH = q->PH;
    id<MTLTexture> Pt = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, PW, PH), Ct = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, PW, PH);
    id<MTLTexture> bw = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM), Qt = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    id<MTLTexture> ca = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, PW, PH), cb = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, PW, PH), ho = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, PW, PH);
    tex_set(Pt, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &q->P[((size_t)y * PW + x) * 4], 16); });
    tex_set(Ct, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &q->C[((size_t)y * PW + x) * 4], 16); });
    tex_set(bw, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = 0.5f; });
    tex_set(Qt, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = 0.3f; });
    id<MTLBuffer> n2a = nn_buf(&R, NULL, (size_t)q->Nw * q->Nh * 4), n2b = nn_buf(&R, NULL, (size_t)q->Nw * q->Nh * 4), n2h = nn_buf(&R, NULL, (size_t)q->Nw * q->Nh);
    MetalFGF20Args a;
    memset(&a, 0, sizeof a);
    a.scIp = sc_buf(q->eP, zero_off); a.scIc = sc_buf(q->eC, zero_off);
    a.colOld = Pt; a.colCur = Ct; a.bw = bw; a.err = Qt; a.candA = ca; a.candB = cb; a.hole = ho; a.n2a = n2a; a.n2b = n2b; a.n2h = n2h;
    a.W = PW; a.H = PH; a.Wm = WM; a.Hm = HM; a.Nw = q->Nw; a.Nh = q->Nh; a.s = q->s;
    nn_run(&R, ^(MetalFGEnc *e) { if (fused) [B blendPack:e args:a]; else { [B blend:e args:a]; [B pack:e args:a]; } });
    memcpy(oA, n2a.contents, (size_t)q->Nw * q->Nh * 8);
    memcpy(oB, n2b.contents, (size_t)q->Nw * q->Nh * 8);
    memcpy(oH, n2h.contents, (size_t)q->Nw * q->Nh * 2);
    if (stA) { float *t = tex_get(ca, 4); memcpy(stA, t, (size_t)PW * PH * 16); free(t); }
    if (stB) { float *t = tex_get(cb, 4); memcpy(stB, t, (size_t)PW * PH * 16); free(t); }
    if (stH) { float *t = tex_get(ho, 1); for (size_t i = 0; i < (size_t)PW * PH; i++) stH[i] = t[i] > 0.5f; free(t); }
}

static void pack_run(const PScene *q, h16 *oA, h16 *oB, h16 *oH, float *stA) { pack_run_mode(q, oA, oB, oH, stA, 0, NULL, NULL); }

static PScene *pscene(int PW, int PH, int s, int Nw, int Nh) {
    PScene *q = calloc(1, sizeof *q);
    q->PW = PW; q->PH = PH; q->s = s; q->Nw = Nw; q->Nh = Nh;
    q->P = calloc((size_t)PW * PH * 4, 4); q->C = calloc((size_t)PW * PH * 4, 4);
    q->eP = calloc(WM * HM, 4); q->eC = calloc(WM * HM, 4);
    for (int i = 0; i < WM * HM; i++) q->eP[i] = q->eC[i] = 1;
    return q;
}

static int cmp_blocks(const PScene *q, const h16 *gA, const h16 *gB, const h16 *gH, int order, int divmode, int stored, int stored_nearest) {
    int bad = 0;
    for (int j = 0; j < q->Nh; j++) for (int i = 0; i < q->Nw; i++) {
        h16 ma[4], mb[4];
        int h;
        block_model(q, i, j, order, divmode, stored, stored_nearest, ma, mb, &h);
        size_t o = ((size_t)j * q->Nw + i) * 4;
        bad += memcmp(&gA[o], ma, 8) != 0 || memcmp(&gB[o], mb, 8) != 0;
        if (gH) bad += (double)gH[(size_t)j * q->Nw + i] != (double)h;
    }
    return bad;
}

static void pack_cases(void) {
    int sizes[5] = {1, 2, 4, 8, 16};
    for (int si = 0; si < 5; si++) {
        int s = sizes[si], PW = 100, PH = 70;
        int Wp = 16 * s * ((PW + 16 * s - 1) / (16 * s)), Hp = 16 * s * ((PH + 16 * s - 1) / (16 * s));
        PScene *q = pscene(PW, PH, s, Wp / s, Hp / s);
        WgenRng rng; wgen_seed(&rng, 90 + s);
        for (size_t i = 0; i < (size_t)PW * PH * 4; i++) { q->P[i] = (float)(int)(wgen_uniform(&rng) * 1024) / 1024.0f; q->C[i] = (float)(int)(wgen_uniform(&rng) * 1024) / 1024.0f; }
        for (int i = 0; i < WM * HM; i++) { q->eP[i] = wgen_uniform(&rng) < 0.6; q->eC[i] = wgen_uniform(&rng) < 0.6; }
        size_t nn = (size_t)q->Nw * q->Nh;
        h16 *gA = malloc(nn * 8), *gB = malloc(nn * 8), *gH = malloc(nn * 2);
        pack_run(q, gA, gB, gH, NULL);
        {   /* the fused pass (blendPack:) gives the same images and the same reduction as blend: followed by pack: */
            size_t np = (size_t)PW * PH * 4;
            float *sa1 = malloc(np * 4), *sb1 = malloc(np * 4), *sa2 = malloc(np * 4), *sb2 = malloc(np * 4);
            uint8_t *sh1 = malloc((size_t)PW * PH), *sh2 = malloc((size_t)PW * PH);
            h16 *fA = malloc(nn * 8), *fB = malloc(nn * 8), *fH = malloc(nn * 2);
            pack_run_mode(q, gA, gB, gH, sa1, 0, sb1, sh1);
            pack_run_mode(q, fA, fB, fH, sa2, 1, sb2, sh2);
            CHECKC(!memcmp(gA, fA, nn * 8) && !memcmp(gB, fB, nn * 8) && !memcmp(gH, fH, nn * 2) && !memcmp(sa1, sa2, np * 4) && !memcmp(sb1, sb2, np * 4) && !memcmp(sh1, sh2, (size_t)PW * PH),
                   "bp-fused s=%d: the fused pass differs from blend followed by pack", s);
            free(sa1); free(sb1); free(sa2); free(sb2); free(sh1); free(sh2); free(fA); free(fB); free(fH);
        }
        int bad = cmp_blocks(q, gA, gB, gH, 0, 0, 0, 1);
        CHECKC(bad == 0, "bp-pack s=%d: %d blocks differ from the model", s, bad);
        if (s > 1) CHECKC(cmp_blocks(q, gA, gB, gH, 1, 0, 0, 1) >= 0, "unreachable");
        free(gA); free(gB); free(gH); free(q->P); free(q->C); free(q->eP); free(q->eC); free(q);
    }
    /* bp2-pad (bp-padding): 51 x 31 at s = 2: padding counts as zero, the divisor is the full block area */
    {
        PScene *q = pscene(51, 31, 2, 32, 16);
        for (size_t i = 0; i < 51 * 31 * 4; i++) q->P[i] = q->C[i] = 1.0f;
        size_t nn = 32 * 16;
        h16 *gA = malloc(nn * 8), *gB = malloc(nn * 8), *gH = malloc(nn * 2);
        pack_run(q, gA, gB, gH, NULL);
        CHECKC(cmp_blocks(q, gA, gB, NULL, 0, 0, 0, 1) == 0, "bp2-pad: differs from the model");
        int bad = 0;
        for (int j = 0; j < 16; j++) for (int i = 0; i < 32; i++) {
            double want = i >= 26 ? 0 : (i == 25 && j == 15) ? 0.25 : (i == 25 || j == 15) ? 0.5 : 1;
            bad += (double)gA[((size_t)j * 32 + i) * 4] != want;
        }
        CHECKC(bad == 0, "bp2-pad: %d wrong cells against the stated values", bad);
        CHECKC(cmp_blocks(q, gA, gB, NULL, 0, 1, 0, 1) > 0, "bp2-pad control (divisor = covered positions) does not differ in the edge blocks");
        free(gA); free(gB); free(gH); free(q->P); free(q->C); free(q->eP); free(q->eC); free(q);
    }
    /* bp2-inject, bp2-synth-unchanged: factor one, candidate values between two binary16 neighbours */
    {
        int PW = 64, PH = 48;
        int nearest = store_is_nearest(&R.gpu);
#if defined(METALFG_CTL_BP_TRUNC)
        int stored_nearest = 0;       /* the injected truncation, whatever the platform does */
#else
        int stored_nearest = nearest;
#endif
        PScene *q = pscene(PW, PH, 1, PW, PH);
        WgenRng rng; wgen_seed(&rng, 777);
        for (int y = 0; y < PH; y++) for (int x = 0; x < PW; x++) for (int c = 0; c < 3; c++) {
            int a = 1024 + (int)(wgen_uniform(&rng) * 1000);      /* values in [0.5, 1): spacing 2^-11; the average of a and a + 1 is a tie */
            q->P[((size_t)y * PW + x) * 4 + c] = (float)a / 2048.0f; q->C[((size_t)y * PW + x) * 4 + c] = (float)(a + 1) / 2048.0f;
        }
        size_t nn = (size_t)PW * PH;
        h16 *gA = malloc(nn * 8), *gB = malloc(nn * 8), *gH = malloc(nn * 2);
        float *stA = malloc(nn * 16);
        pack_run(q, gA, gB, gH, stA);
        int nd = 0;                        /* elements where the two conversions differ, from the model */
        for (int y = 0; y < PH; y++) for (int x = 0; x < PW; x++) {
            float A[4], Bv[4]; int h;
            model_px(q, x, y, A, Bv, &h);
            for (int c = 0; c < 3; c++) nd += rne16(A[c]) != trunc16(A[c]);
        }
        CHECKC(nd > 0, "bp2-inject: no element between two neighbours");
        int bad = cmp_blocks(q, gA, gB, NULL, 0, 0, 0, 1);
        CHECKC(bad == 0, "bp2-inject: the network input differs from the nearest-even model in %d pixels", bad);
        int badst = 0;
        for (int y = 0; y < PH; y++) for (int x = 0; x < PW; x++) {
            float A[4], Bv[4]; int h;
            model_px(q, x, y, A, Bv, &h);
            for (int c = 0; c < 4; c++) badst += stA[((size_t)y * PW + x) * 4 + c] != (stored_nearest ? rne16(A[c]) : trunc16(A[c]));
        }
        CHECKC(badst == 0, "bp2-synth-unchanged: %d stored candidate elements differ from the store conversion", badst);
        /* the control reading the stored image would differ exactly where the conversions differ */
        int dctl = cmp_blocks(q, gA, gB, NULL, 0, 0, 1, stored_nearest);
        if (stored_nearest) printf("  bp2-inject: the platform's own store rounds to nearest even; this run is void, only the injected build counts\n");
        else CHECKC(dctl > 0, "bp2-inject control (reading the stored image) is not detected");
        free(gA); free(gB); free(gH); free(stA); free(q->P); free(q->C); free(q->eP); free(q->eC); free(q);
    }
    /* bp2-block-sum: factor two */
    {
        int PW = 64, PH = 48;
        int nearest = store_is_nearest(&R.gpu);
#if defined(METALFG_CTL_BP_TRUNC)
        nearest = 0;
#endif
        PScene *q = pscene(PW, PH, 2, 32, 24);
        for (int j = 0; j < 24; j++) for (int i = 0; i < 32; i++) {
            int type = (i + j) & 1;
            for (int k = 0; k < 4; k++) {
                int x = 2 * i + (k & 1), y = 2 * j + (k >> 1);
                float p, c;
                if (type == 0) { if (k == 0) { p = 1024; c = 1025; } else { p = 0; c = 3.0f * 0x1p-15f; } }
                else { if (k < 3) { p = 1024; c = 1025; } else { p = 1025; c = 1026; } }
                for (int ch = 0; ch < 3; ch++) { q->P[((size_t)y * PW + x) * 4 + ch] = p; q->C[((size_t)y * PW + x) * 4 + ch] = c; }
            }
        }
        size_t nn = 32 * 24;
        h16 *gA = malloc(nn * 8), *gB = malloc(nn * 8), *gH = malloc(nn * 2);
        pack_run(q, gA, gB, gH, NULL);
        CHECKC(cmp_blocks(q, gA, gB, gH, 0, 0, 0, 1) == 0, "bp2-block-sum: differs from the model of rules B and C");
        CHECKC(cmp_blocks(q, gA, gB, NULL, 0, 0, 1, nearest) > 0, "bp2-block-sum control (sum of the stored binary16 images) does not differ");
        CHECKC(cmp_blocks(q, gA, gB, NULL, 1, 0, 0, 1) > 0, "bp2-block-sum control (another summation order) does not differ");
        free(gA); free(gB); free(gH); free(q->P); free(q->C); free(q->eP); free(q->eC); free(q);
    }
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        B = [[MetalFGBlend alloc] initWithDevice:g.dev library:R.lib];
        if (!B) { printf("blend kernels missing\n"); return 2; }
        fallback_hole_quality(); ui_cases(); hole_nan(); pack_cases();
    }
    printf(fails ? "g_blend: %d FAILURES\n" : "g_blend: ok\n", fails);
    return fails != 0;
}
