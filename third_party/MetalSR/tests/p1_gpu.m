// GPU-vs-CPU tests of pass P1 (sr.input_features_r2): p1-static, p1-pan, p1-dilate, p1-reset, p1-memory (ownership at seven
// render sizes and the retention check), a sanitising check, and the controls that must fail.
#import "testutil.h"
#include "ref_p1.h"
#include "testparams.h"
#import "passes.h"
#include "../src/frame.h"

static int SUB_BITS_V(void) { return getenv("SR_SUB") ? atoi(getenv("SR_SUB")) : 8; }
#define SUB_BITS SUB_BITS_V()
static SrParams g_mp;

/* ---- case data ---- */
typedef struct {
    int Wo, Ho, Wr, Hr, mw, mh;        /* output, render and motion-image sizes */
    float *col, *depth;                /* rgba Wr x Hr, Wr x Hr */
    f16 *mot;                          /* motion rg, mw x mh */
    f16 *hist, *memr;                  /* Wo x Ho x 4, Wo x Ho x 2 */
    float e_now, e_prev;
} Case;

static Case *case_new(int Wo, int Ho, int Wr, int Hr, int mv_render) {
    Case *c = calloc(1, sizeof *c);
    c->Wo = Wo; c->Ho = Ho; c->Wr = Wr; c->Hr = Hr; c->mw = mv_render ? Wr : Wo; c->mh = mv_render ? Hr : Ho;
    c->col = calloc((size_t)Wr * Hr * 4, 4); c->depth = calloc((size_t)Wr * Hr, 4); c->mot = calloc((size_t)c->mw * c->mh * 2, 2);
    c->hist = calloc((size_t)Wo * Ho * 4, 2); c->memr = calloc((size_t)Wo * Ho * 2, 2);
    c->e_now = c->e_prev = 1;
    return c;
}
static void case_free(Case *c) { free(c->col); free(c->depth); free(c->mot); free(c->hist); free(c->memr); free(c); }

static FeatParams make_fp(const Case *c, float jx, float jy, float pre, int hdr, int depth_inv, int mv_render, int reset) {
    SrLevels L; sr_levels(c->Wo, c->Ho, &L);
    SrCall call = {(u32)c->Wo, (u32)c->Ho, (u32)c->Wr, (u32)c->Hr, jx, jy, 1, 1, pre, (u32)hdr, (u32)depth_inv, (u32)mv_render, (u32)reset, 0, 0, 0, (u32)L.W[0]};
    FeatParams fp; sr_make_feat(&call, &g_mp, &fp);
    return fp;
}

typedef struct { f16 *R, *memw, *feat; int W0, H0; } Out;
static Out *out_new(const Case *c) {
    SrLevels L; sr_levels(c->Wo, c->Ho, &L);
    Out *o = calloc(1, sizeof *o);
    o->W0 = L.W[0]; o->H0 = L.H[0];
    o->R = calloc((size_t)c->Wo * c->Ho * 4, 2); o->memw = calloc((size_t)c->Wo * c->Ho * 2, 2); o->feat = calloc((size_t)o->W0 * o->H0 * 16, 2);
    return o;
}
static void out_free(Out *o) { free(o->R); free(o->memw); free(o->feat); free(o); }

static id<MTLTexture> tex_put(MTLPixelFormat f, int w, int h, const void *p, int bpp) {
    id<MTLTexture> t = [g_gpu texture:f w:w h:h];
    if (p) [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:p bytesPerRow:(size_t)w * bpp];
    return t;
}
/* the GPU side: memory textures persist across calls in the retention check, so the caller owns them */
typedef struct { id<MTLTexture> mem[2]; } Mems;
static Mems mems_new(int Wo, int Ho) {
    Mems m; f16 *z = calloc((size_t)Wo * Ho * 2, 2);
    for (int i = 0; i < 2; i++) m.mem[i] = tex_put(MTLPixelFormatRG16Float, Wo, Ho, z, 4);
    free(z);
    return m;
}
static void gpu_p1(const FeatParams *fp, const Case *c, id<MTLTexture> mem_read, id<MTLTexture> mem_write, Out *o) {
    id<MTLTexture> col = tex_put(MTLPixelFormatRGBA32Float, c->Wr, c->Hr, c->col, 16), dep = tex_put(MTLPixelFormatR32Float, c->Wr, c->Hr, c->depth, 4),
                   mot = tex_put(MTLPixelFormatRG16Float, c->mw, c->mh, c->mot, 4), hist = tex_put(MTLPixelFormatRGBA16Float, c->Wo, c->Ho, c->hist, 8),
                   warped = tex_put(MTLPixelFormatRGBA16Float, c->Wo, c->Ho, NULL, 8);
    f16 en = (f16)c->e_now, ep = (f16)c->e_prev;
    id<MTLTexture> te = tex_put(MTLPixelFormatR16Float, 1, 1, &en, 2), tp = tex_put(MTLPixelFormatR16Float, 1, 1, &ep, 2);
    if (mem_read == nil) mem_read = tex_put(MTLPixelFormatRG16Float, c->Wo, c->Ho, c->memr, 4);
    id<MTLBuffer> feat = [g_gpu buffer:(size_t)o->W0 * o->H0 * 32];
    SrFeatIo io = {col, dep, mot, hist, mem_read, te, tp, warped, mem_write, feat};
    run1(^(SrEnc *e) { sr_features(e, g_gpu, fp, &io); });
    [warped getBytes:o->R bytesPerRow:(size_t)c->Wo * 8 fromRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0];
    [mem_write getBytes:o->memw bytesPerRow:(size_t)c->Wo * 4 fromRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0];
    memcpy(o->feat, feat.contents, (size_t)o->W0 * o->H0 * 32);
}
/* one call with fresh memory textures (the memory written is in o->memw) */
static void gpu_p1_fresh(const FeatParams *fp, const Case *c, Out *o) {
    id<MTLTexture> mw = tex_put(MTLPixelFormatRG16Float, c->Wo, c->Ho, o->memw, 4);   /* o->memw is zero on entry */
    gpu_p1(fp, c, nil, mw, o);
}

static RefP1In ref_in(const Case *c, float **motf) {
    size_t n = (size_t)c->mw * c->mh * 2;
    *motf = malloc(n * 4);
    for (size_t i = 0; i < n; i++) (*motf)[i] = (float)c->mot[i];
    RefP1In in = {c->col, c->depth, *motf, c->hist, c->memr, c->e_now, c->e_prev};
    return in;
}

/* ---- comparisons ---- */
/* fraction of bit-equal values >= frac_eq and all within `ulps`; returns 1 when the criterion holds */
static int cmp_ok(const char *name, const f16 *got, const f16 *ref, size_t n, double frac_eq, int ulps, int verbose) {
    size_t eq = 0, bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &ref[i], 2) == 0) { eq++; continue; }
        if (!isfinite((float)got[i]) || !isfinite((float)ref[i]) || ulp_dist16(got[i], ref[i]) > ulps) {
            if (verbose && bad < 3 && getenv("SR_DEBUG")) printf("    mismatch @%zu got %g ref %g\n", i, (double)got[i], (double)ref[i]);
            bad++;
        }
    }
    double f = n ? (double)eq / n : 1;
    if (verbose) printf("  %s: %.5f bit-equal, %zu beyond %d ulp\n", name, f, bad, ulps);
    return f >= frac_eq && bad == 0;
}
/* R over the pixels of the L0 rows selected by row_stride */
static int cmp_R(const char *name, const Case *c, const f16 *got, const f16 *ref, int row_stride, int ulps, int verbose) {
    size_t eq = 0, bad = 0, n = 0;
    for (int y = 0; y < c->Ho; y++) {
        if (((y >> 2) % row_stride) != 0) continue;
        for (int x = 0; x < c->Wo; x++) for (int k = 0; k < 4; k++) {
            size_t i = ((size_t)y * c->Wo + x) * 4 + k; n++;
            if (!memcmp(&got[i], &ref[i], 2)) { eq++; continue; }
            if (ulp_dist16(got[i], ref[i]) > ulps) { if (verbose && bad < 3 && getenv("SR_DEBUG")) printf("    R mismatch (%d,%d,%d) got %g ref %g\n", x, y, k, (double)got[i], (double)ref[i]); bad++; }
        }
    }
    if (verbose) printf("  %s: %.5f bit-equal, %zu beyond %d ulp\n", name, n ? (double)eq / n : 1, bad, ulps);
    return bad == 0;
}
static int cmp_feat(const char *name, const Out *go, const f16 *ref, int row_stride, int verbose) {
    size_t eq = 0, bad = 0, n = 0;
    for (int Y = 0; Y < go->H0; Y++) {
        if (Y % row_stride) continue;
        for (size_t i = (size_t)Y * go->W0 * 16; i < (size_t)(Y + 1) * go->W0 * 16; i++) {
            n++;
            if (!memcmp(&go->feat[i], &ref[i], 2)) { eq++; continue; }
            if (ulp_dist16(go->feat[i], ref[i]) > 1) { if (verbose && bad < 3 && getenv("SR_DEBUG")) printf("    feat mismatch @%zu got %g ref %g\n", i, (double)go->feat[i], (double)ref[i]); bad++; }
        }
    }
    double f = n ? (double)eq / n : 1;
    if (verbose) printf("  %s: %.5f bit-equal, %zu beyond 1 ulp\n", name, f, bad);
    return f >= 0.995 && bad == 0;
}

/* run the CPU model with options; fills R and feat for the selected rows (zero elsewhere) */
static void model_run(const FeatParams *fp, const RefP1Opts *opt, const Case *c, int row_stride, f16 *R, f16 *feat) {
    float *motf; RefP1In in = ref_in(c, &motf);
    ref_p1(fp, opt, &in, 0, row_stride, feat, R, NULL);
    free(motf);
}

/* ---- scene fills ---- */
static void fill_colour(Case *c, int variant) {
    for (int j = 0; j < c->Hr; j++) for (int i = 0; i < c->Wr; i++) {
        float *p = c->col + ((size_t)j * c->Wr + i) * 4;
        if (variant == 0) { p[0] = (float)(0.6 + 0.5 * sin(0.21 * i + 0.1 * j)); p[1] = (float)(0.4 + 0.3 * cos(0.13 * j)); p[2] = (float)(1.5 + sin(0.07 * (i + j))); }
        else { p[0] = (float)(4 * urand()); p[1] = (float)(2 * urand()); p[2] = (float)(3 * urand()); }
        p[3] = 1;
    }
}
static void fill_hist(Case *c, double g, double amp, double fq) {
    for (int y = 0; y < c->Ho; y++) for (int x = 0; x < c->Wo; x++) {
        f16 *p = c->hist + ((size_t)y * c->Wo + x) * 4;
        p[0] = (f16)(g * (1.5 * x / c->Wo + 1.0 * y / c->Ho) + amp * sin(fq * (1.0 * x + 0.3 * y)));
        p[1] = (f16)(g * 1.2 * y / c->Ho + 0.5 + amp * cos(fq * 0.9 * y));
        p[2] = (f16)(0.3 + g * 2.5 * x / c->Wo + amp * sin(fq * 0.7 * (x - y)));
        p[3] = (f16)(0.5 + 0.4 * sin(0.37 * x + 0.21 * y));
    }
}
/* constant history: the sampler returns it exactly whatever the sub-texel weights */
static void fill_hist_const(Case *c) {
    for (size_t i = 0; i < (size_t)c->Wo * c->Ho; i++) { f16 *p = c->hist + 4 * i; p[0] = (f16)0.25; p[1] = (f16)0.5; p[2] = (f16)0.125; p[3] = (f16)0.5; }
}
static void fill_motion_const(Case *c, double mx, double my) {
    for (size_t i = 0; i < (size_t)c->mw * c->mh; i++) { c->mot[2 * i] = (f16)mx; c->mot[2 * i + 1] = (f16)my; }
}

/* ---- tests ---- */
/* p1-static (sr.input_features_r2): constant colour, static scene, history equal to the tonemapped scaled colour */
static void test_static(int Wo, int Ho, int Wr, int Hr, int hdr, int mvr, float jx, float jy) {
    Case *c = case_new(Wo, Ho, Wr, Hr, mvr);
    /* the history is chosen first (exactly representable); the colour is then the exposure-scaled colour that equals the
       history's inverse tonemap, so the warped history reproduces it up to the rounding of the inverse tonemap */
    const double ty[3] = {0.1875, 0.0625, 0.296875};
    float pre = 2.0f; c->e_now = c->e_prev = 1.5f;
    FeatParams fp = make_fp(c, jx, jy, pre, hdr, 0, mvr, 0);
    double k = (double)fp.invp * c->e_now, sc[3], C0[3];
    { double y = lumaY((float)ty[0], (float)ty[1], (float)ty[2]), d = 1 - y; if (d < fp.phi_hi) d = fp.phi_hi; for (int q = 0; q < 3; q++) sc[q] = hdr ? ty[q] / d : ty[q]; }
    for (int q = 0; q < 3; q++) C0[q] = hdr ? (double)(float)(sc[q] / k) : sc[q];
    for (int q = 0; q < 3; q++) if (hdr) sc[q] = (double)(float)(k * (double)(float)C0[q]);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) { for (int q = 0; q < 3; q++) c->col[4 * i + q] = (float)C0[q]; c->col[4 * i + 3] = 1; c->depth[i] = 0.5f; }
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) { for (int q = 0; q < 3; q++) c->hist[4 * i + q] = (f16)ty[q]; c->hist[4 * i + 3] = (f16)0.4; }
    Out *o = out_new(c), *m = out_new(c);
    gpu_p1_fresh(&fp, c, o);
    RefP1Opts opt = {0, 0, 0, 0, 0, SUB_BITS, 1};
    model_run(&fp, &opt, c, 1, m->R, m->feat);
    printf("p1-static %dx%d render %dx%d %s motion %s jitter (%g,%g)\n", Wo, Ho, Wr, Hr, hdr ? "hdr" : "display", mvr ? "render-res" : "output-res", jx, jy);
    f16 *want = calloc((size_t)Wo * Ho * 4, 2);
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) for (int q = 0; q < 3; q++) want[4 * i + q] = (f16)sc[q];
    EXPECT(cmp_ok("R vs scaled colour", o->R, want, (size_t)Wo * Ho * 4, 0, 1, 1), "p1-static R");
    EXPECT(cmp_R("R vs model", c, o->R, m->R, 1, 1, 1), "p1-static R model");
    size_t f3 = 0;
    for (size_t pos = 0; pos < (size_t)o->W0 * o->H0; pos++) for (int h = 0; h < 4; h++) if (bits16(o->feat[pos * 16 + 4 * h + 3]) != 0) f3++;
    printf("  F3 nonzero records: %zu\n", f3);
    EXPECT(f3 == 0, "p1-static F3 must be exactly 0");
    EXPECT(cmp_feat("feat vs model", o, m->feat, 1, 1), "p1-static feat");
    /* identical records at L0 positions with the same set of in-output pixels */
    size_t groups = 0, differ = 0;
    f16 *first[4096] = {0}; int key_of[4096]; int nk = 0;
    for (int Y = 0; Y < (Ho + 3) / 4; Y++) for (int X = 0; X < (Wo + 3) / 4; X++) {
        int key = 0;
        for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) if (4 * X + a < Wo && 4 * Y + b < Ho) key |= 1 << (4 * b + a);
        f16 *rec = o->feat + ((size_t)Y * o->W0 + X) * 16;
        int s = -1; for (int i = 0; i < nk; i++) if (key_of[i] == key) s = i;
        if (s < 0) { s = nk++; key_of[s] = key; first[s] = rec; groups++; }
        else if (memcmp(rec, first[s], 32)) differ++;
    }
    printf("  %zu distinct pixel sets, %zu records differ from their group's first\n", groups, differ);
    EXPECT(differ == 0, "p1-static position dependence");
    free(want); out_free(o); out_free(m); case_free(c);
}

/* p1-pan: uniform motion (+3.25, -1.5) over a smooth gradient history; returns the model variant results for controls */
static void test_pan(void) {
    const int Wo = 256, Ho = 144, Wr = 171, Hr = 96;
    Case *c = case_new(Wo, Ho, Wr, Hr, 0);
    fill_colour(c, 0); fill_hist(c, 1.0, 0.5, 0.45); fill_motion_const(c, 3.25, -1.5);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) c->depth[i] = 0.5f;
    FeatParams fp = make_fp(c, 0.2f, -0.35f, 1.0f, 1, 0, 0, 0);
    Out *o = out_new(c), *m = out_new(c);
    gpu_p1_fresh(&fp, c, o);
    RefP1Opts opt = {0, 0, 0, 0, 0, SUB_BITS, 1};
    model_run(&fp, &opt, c, 1, m->R, m->feat);
    printf("p1-pan %dx%d render %dx%d, motion (+3.25, -1.5)\n", Wo, Ho, Wr, Hr);
    EXPECT(cmp_R("R vs model (2 ulp)", c, o->R, m->R, 1, 2, 1), "p1-pan R");
    /* invalid flags: pixels the model marks invalid (reprojected outside) carry the current colour, exactly */
    size_t inv = 0, inv_bad = 0;
    float *motf; RefP1In in = ref_in(c, &motf);
    for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) {
        double F[4], mm[2]; int v; f16 Rm[4];
        ref_p1_pixel(&fp, &opt, &in, x, y, Rm, F, mm, &v);
        if (!v) { inv++; if (memcmp(Rm, o->R + ((size_t)y * Wo + x) * 4, 6)) inv_bad++; }
    }
    free(motf);
    printf("  invalid pixels %zu, differing from the current colour %zu\n", inv, inv_bad);
    EXPECT(inv > 0 && inv_bad == 0, "p1-pan invalid flags");
    EXPECT(cmp_feat("feat vs model", o, m->feat, 1, 1), "p1-pan feat");
    /* controls: each wrong model must fail the 2 ulp criterion */
    RefP1Opts ctl[2] = {{1, 0, 0, 0, 0, SUB_BITS, 1}, {0, 1, 0, 0, 0, SUB_BITS, 1}};
    const char *names[2] = {"control: no luma limiting", "control: all 16 cubic taps"};
    for (int k = 0; k < 2; k++) {
        Out *w = out_new(c);
        model_run(&fp, &ctl[k], c, 1, w->R, w->feat);
        int ok = cmp_R(names[k], c, o->R, w->R, 1, 2, 0);
        printf("  %s: GPU %s the wrong model\n", names[k], ok ? "MATCHES" : "differs from");
        EXPECT(!ok, "%s must fail", names[k]);
        out_free(w);
    }
    out_free(o); out_free(m); case_free(c);
}

/* p1-dilate: near square on a far background, opposite motion vectors, both depth conventions; stored vectors exact */
static void test_dilate(int depth_inv) {
    const int W = 96, H = 64;
    Case *c = case_new(W, H, W, H, 1);
    for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
        int in_sq = i >= 30 && i < 60 && j >= 20 && j < 44;
        double d = in_sq ? 0.1 : 0.9;
        if (i < 12) d = 0.9;       /* flat region: every candidate ties */
        c->depth[(size_t)j * W + i] = (float)(depth_inv ? 1 - d : d);
        c->mot[((size_t)j * W + i) * 2] = (f16)(in_sq ? -3.0 : (i < 12 ? 0.25 * i - 0.125 * j : 2.0));
        c->mot[((size_t)j * W + i) * 2 + 1] = (f16)(in_sq ? 1.0 : (i < 12 ? -0.0625 * j : 0.5));
        c->col[((size_t)j * W + i) * 4 + 3] = 1;
    }
    FeatParams fp = make_fp(c, 0, 0, 1.0f, 1, depth_inv, 1, 0);
    Out *o = out_new(c);
    gpu_p1_fresh(&fp, c, o);
    f16 *want = calloc((size_t)W * H * 2, 2), *wrong = calloc((size_t)W * H * 2, 2);
    float *motf; RefP1In in = ref_in(c, &motf);
    RefP1Opts opt = {0, 0, 0, 0, 0, SUB_BITS, 1}, ctl = {0, 0, 1, 0, 0, SUB_BITS, 1};
    ref_p1_memory(&fp, &opt, &in, want); ref_p1_memory(&fp, &ctl, &in, wrong);
    free(motf);
    size_t bad = 0, bad_ctl = 0;
    for (size_t i = 0; i < (size_t)W * H * 2; i++) { bad += memcmp(&o->memw[i], &want[i], 2) != 0; bad_ctl += memcmp(&wrong[i], &want[i], 2) != 0; }
    printf("p1-dilate %s depth: %zu of %d stored vectors differ from the rule; control (motion at n) differs in %zu\n", depth_inv ? "inverted" : "standard", bad, W * H * 2, bad_ctl);
    EXPECT(bad == 0, "p1-dilate exact");
    EXPECT(bad_ctl > 0, "p1-dilate control must fail");
    free(want); free(wrong); out_free(o); case_free(c);
}

/* p1-reset: history invalid everywhere: R is the current colour, F3 = 0, F2 = F0 */
static void test_reset(int hdr) {
    const int Wo = 200, Ho = 120, Wr = 200, Hr = 120;
    Case *c = case_new(Wo, Ho, Wr, Hr, 0);
    fill_colour(c, 1); fill_hist(c, 1.0, 0.5, 0.45); fill_motion_const(c, 2.5, -1.25);
    for (size_t i = 0; i < (size_t)Wo * Ho * 2; i++) c->memr[i] = (f16)(0.5 + 0.001 * (double)(i % 977));
    c->e_now = 1.25f; c->e_prev = 0.75f;
    const float jx = 0.49999997f, jy = 0.49999997f;      /* the largest float below 1/2: floor(v + 1/2) rounds up, round() does not */
    FeatParams fp = make_fp(c, jx, jy, 1.0f, hdr, 0, 0, 1);
    Out *o = out_new(c), *m = out_new(c);
    gpu_p1_fresh(&fp, c, o);
    RefP1Opts opt = {0, 0, 0, 0, 0, SUB_BITS, 1};
    model_run(&fp, &opt, c, 1, m->R, m->feat);
    printf("p1-reset %s\n", hdr ? "hdr" : "display");
    EXPECT(cmp_R("R vs current colour (model)", c, o->R, m->R, 1, 1, 1), "p1-reset R");
    size_t f3 = 0;
    for (size_t pos = 0; pos < (size_t)o->W0 * o->H0; pos++) for (int h = 0; h < 4; h++) if (bits16(o->feat[pos * 16 + 4 * h + 3]) != 0) f3++;
    printf("  F3 nonzero records: %zu\n", f3);
    EXPECT(f3 == 0, "p1-reset F3 must be 0");
    EXPECT(cmp_feat("feat vs model", o, m->feat, 1, 1), "p1-reset feat");
    RefP1Opts ctl = {0, 0, 0, 1, 0, SUB_BITS, 1};
    Out *w = out_new(c);
    model_run(&fp, &ctl, c, 1, w->R, w->feat);
    int ok = cmp_R("control: floor(v + 1/2)", c, o->R, w->R, 1, 1, 0);
    printf("  control: floor(v + 1/2): GPU %s the wrong model\n", ok ? "MATCHES" : "differs from");
    EXPECT(!ok, "floor(v + 1/2) control must fail");
    out_free(w); out_free(o); out_free(m); case_free(c);
}

/* motion-memory owners: exact rational check of every cell, and the left-edge control */
static int check_cells(const FeatParams *fp, const Case *c, const f16 *got, const f16 *base_other, int own_left, int Wlim, int Hlim, size_t *bad_out) {
    /* the vector of the owner pixel; V = (x/64, y/64) is exact in fp16, S = 1 */
    size_t bad = 0, n = 0;
    for (int j = 0; j < Hlim; j++) for (int i = 0; i < Wlim; i++) {
        int ox = r1_owner(fp->sx, i, c->Wo, own_left), oy = r1_owner(fp->sy, j, c->Ho, own_left);
        f16 want[2];
        if (ox < 0 || oy < 0) { want[0] = want[1] = 0; } else { want[0] = c->mot[((size_t)oy * c->Wo + ox) * 2]; want[1] = c->mot[((size_t)oy * c->Wo + ox) * 2 + 1]; }
        n++;
        if (memcmp(&got[((size_t)j * c->Wo + i) * 2], want, 4)) bad++;
    }
    (void)base_other; *bad_out = bad;
    return n > 0 && bad == 0;
}
static void fill_vxy(Case *c, int n) {
    for (int y = 0; y < c->Ho; y++) for (int x = 0; x < c->Wo; x++) {
        c->mot[((size_t)y * c->Wo + x) * 2] = (f16)((x + n) / 64.0); c->mot[((size_t)y * c->Wo + x) * 2 + 1] = (f16)((y + 2 * n) / 64.0);
    }
}
static void test_memory(void) {
    static const int sizes[7][2] = {{1920, 1080}, {1440, 810}, {1600, 900}, {1280, 720}, {960, 540}, {640, 360}, {1279, 719}};
    printf("p1-memory ownership, output 1920x1080\n");
    for (int s = 0; s < 7; s++) {
        int Wr = sizes[s][0], Hr = sizes[s][1];
        Case *c = case_new(1920, 1080, Wr, Hr, 0);
        fill_colour(c, 0); fill_vxy(c, 0); fill_hist_const(c);
        for (size_t i = 0; i < (size_t)Wr * Hr; i++) c->depth[i] = 0.5f;
        FeatParams fp = make_fp(c, 0, 0, 1.0f, 1, 0, 0, 0);
        Out *o = out_new(c), *m = out_new(c);
        gpu_p1_fresh(&fp, c, o);
        size_t bad, bad_ctl;
        int ok = check_cells(&fp, c, o->memw, NULL, 0, Wr, Hr, &bad);
        int ok_ctl = check_cells(&fp, c, o->memw, NULL, 1, Wr, Hr, &bad_ctl);
        /* nothing outside the render rectangle is written in a call on a fresh memory */
        size_t stray = 0;
        for (int j = 0; j < c->Ho; j++) for (int i = 0; i < c->Wo; i++) if ((i >= Wr || j >= Hr) && (o->memw[((size_t)j * c->Wo + i) * 2] || o->memw[((size_t)j * c->Wo + i) * 2 + 1])) stray++;
        printf("  render %dx%d (ratio %.4f): %zu of %d cells differ from the owner's vector; %zu stray writes; left-edge control differs in %zu cells\n", Wr, Hr, 1920.0 / Wr, bad, Wr * Hr, stray, bad_ctl);
        EXPECT(ok && stray == 0, "p1-memory ownership at %dx%d", Wr, Hr);
        EXPECT(!ok_ctl, "p1-memory control (ownership by the left edge) must fail at %dx%d", Wr, Hr);
        /* R and feat against the model on every 16th L0 row (1 ulp for R, the feat criterion) */
        RefP1Opts opt = {0, 0, 0, 0, 0, SUB_BITS, 1};
        model_run(&fp, &opt, c, 16, m->R, m->feat);
        EXPECT(cmp_R("R vs model (rows sampled)", c, o->R, m->R, 16, 1, 1), "p1-memory R at %dx%d", Wr, Hr);
        EXPECT(cmp_feat("feat vs model (rows sampled)", o, m->feat, 16, 1), "p1-memory feat at %dx%d", Wr, Hr);
        out_free(o); out_free(m); case_free(c);
    }
    /* retention: three calls, 1920x1080, 1920x1080, 960x540, different V per call */
    for (int control = 0; control < 2; control++) {
        Mems mm = mems_new(1920, 1080);
        static const int rs[3][2] = {{1920, 1080}, {1920, 1080}, {960, 540}};
        Out *o = NULL;
        FeatParams last; Case *lastc = NULL;
        for (int n = 0; n < 3; n++) {
            Case *c = case_new(1920, 1080, rs[n][0], rs[n][1], 0);
            fill_colour(c, 0); fill_vxy(c, n);
            for (size_t i = 0; i < (size_t)c->Wr * c->Hr; i++) c->depth[i] = 0.5f;
            FeatParams fp = make_fp(c, 0, 0, 1.0f, 1, 0, 0, 0);
            o = out_new(c);
            id<MTLTexture> mw = mm.mem[(n + 1) % 2];
            if (control && n == 2) { f16 *z = calloc((size_t)1920 * 1080 * 2, 2); [mw replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:z bytesPerRow:1920 * 4]; free(z); }
            gpu_p1(&fp, c, mm.mem[n % 2], mw, o);
            if (n < 2) out_free(o);
            if (n == 2) { last = fp; lastc = c; } else case_free(c);
        }
        /* judged after call 2: inside 960x540 call 2's owner vectors, outside call 0's vector (owner (i, j), V_0 = (i/64, j/64)) */
        size_t bad_in = 0, bad_out = 0;
        for (int j = 0; j < 1080; j++) for (int i = 0; i < 1920; i++) {
            const f16 *g = &o->memw[((size_t)j * 1920 + i) * 2];
            f16 want[2];
            if (i < 960 && j < 540) {
                int ox = r1_owner(last.sx, i, 1920, 0), oy = r1_owner(last.sy, j, 1080, 0);
                want[0] = lastc->mot[((size_t)oy * 1920 + ox) * 2]; want[1] = lastc->mot[((size_t)oy * 1920 + ox) * 2 + 1];
                bad_in += memcmp(g, want, 4) != 0;
            } else {
                want[0] = (f16)(i / 64.0); want[1] = (f16)(j / 64.0);
                bad_out += memcmp(g, want, 4) != 0;
            }
        }
        printf("p1-memory retention%s: %zu cells inside 960x540 and %zu cells outside differ from the expected vectors\n", control ? " (control: M_write cleared before the pass)" : "", bad_in, bad_out);
        if (!control) EXPECT(bad_in == 0 && bad_out == 0, "p1-memory retention");
        else EXPECT(bad_out > 0, "p1-memory retention control must fail");
        out_free(o); case_free(lastc);
    }
}

/* non-finite colour, depth and motion read as 0: bit-identical to the twin with zeros stored */
static void test_sanitise(void) {
    const int Wo = 128, Ho = 72, Wr = 85, Hr = 48;
    Case *a = case_new(Wo, Ho, Wr, Hr, 1), *b = case_new(Wo, Ho, Wr, Hr, 1);
    fill_colour(a, 1); fill_hist(a, 1.0, 0.3, 0.45); for (size_t i = 0; i < (size_t)Wr * Hr; i++) { a->depth[i] = (float)urand(); a->mot[2 * i] = (f16)(urand() * 4 - 2); a->mot[2 * i + 1] = (f16)(urand() * 2 - 1); }
    memcpy(b->col, a->col, (size_t)Wr * Hr * 16); memcpy(b->depth, a->depth, (size_t)Wr * Hr * 4); memcpy(b->mot, a->mot, (size_t)Wr * Hr * 4); memcpy(b->hist, a->hist, (size_t)Wo * Ho * 8);
    for (int k = 0; k < 400; k++) {
        size_t i = (size_t)(urand() * Wr * Hr);
        switch (k % 4) {
        case 0: a->col[4 * i + k % 3] = NAN; b->col[4 * i + k % 3] = 0; break;
        case 1: a->col[4 * i + k % 3] = INFINITY; b->col[4 * i + k % 3] = 0; break;
        case 2: a->depth[i] = (k % 8 == 2) ? NAN : -INFINITY; b->depth[i] = 0; break;
        default: a->mot[2 * i + (k / 4) % 2] = (f16)NAN; b->mot[2 * i + (k / 4) % 2] = (f16)0; break;
        }
    }
    FeatParams fp = make_fp(a, 0.1f, -0.2f, 1.0f, 1, 0, 1, 0);
    Out *oa = out_new(a), *ob = out_new(b);
    gpu_p1_fresh(&fp, a, oa); gpu_p1_fresh(&fp, b, ob);
    int same = !memcmp(oa->R, ob->R, (size_t)Wo * Ho * 8) && !memcmp(oa->memw, ob->memw, (size_t)Wo * Ho * 4) && !memcmp(oa->feat, ob->feat, (size_t)oa->W0 * oa->H0 * 32);
    int fin = 1; for (size_t i = 0; i < (size_t)Wo * Ho * 4; i++) fin &= isfinite((float)oa->R[i]);
    printf("p1-sanitise: non-finite inputs give %s outputs to the zero twin; R finite: %d\n", same ? "bit-identical" : "DIFFERENT", fin);
    EXPECT(same && fin, "p1-sanitise");
    out_free(oa); out_free(ob); case_free(a); case_free(b);
}

int main(void) {
    @autoreleasepool {
        gpu_init();
        g_mp = test_params_r3();
        g_mp.phi_hi = 0.9;   /* a high floor keeps the inverse tonemap from amplifying the 1-ulp rounding differences of the sampler (tests only) */
        test_static(250, 130, 250, 130, 1, 0, 0.0f, 0.0f);
        test_static(250, 130, 167, 87, 1, 1, 0.3f, -0.2f);
        test_static(250, 130, 167, 87, 0, 0, -0.4f, 0.45f);
        test_static(257, 131, 257, 131, 0, 1, 0.25f, 0.1f);
        test_pan();
        test_dilate(0);
        test_dilate(1);
        test_reset(1);
        test_reset(0);
        test_sanitise();
        test_memory();
        printf(g_fails ? "p1_gpu: %d FAILURES\n" : "p1_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
