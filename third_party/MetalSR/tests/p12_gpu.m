// GPU-vs-CPU tests of pass P12 (sr.head_filter, sr.head_filter_r2/r3/r4): head-network, head-quads, head-identity-net,
// head-filter, head-coverage-ratio3 (and the fast-math guard), head-tile-bound, head-exceptional, and the controls.
#import "testutil.h"
#include "ref_p12.h"
#include "testparams.h"
#import "passes.h"
#include "../src/frame.h"

static SrParams g_mp;

/* ---- P12 case: colour, warped image R, head output o ---- */
typedef struct {
    int Wo, Ho, Wr, Hr, W0, H0;
    float *col; f16 *R; f16 *o;     /* rgba Wr x Hr; rgba Wo x Ho; [H0][W0][40] */
    float e_now;
} Case;

static Case *case_new(int Wo, int Ho, int Wr, int Hr) {
    SrLevels L; sr_levels(Wo, Ho, &L);
    Case *c = calloc(1, sizeof *c);
    c->Wo = Wo; c->Ho = Ho; c->Wr = Wr; c->Hr = Hr; c->W0 = L.W[0]; c->H0 = L.H[0]; c->e_now = 1;
    c->col = calloc((size_t)Wr * Hr * 4, 4); c->R = calloc((size_t)Wo * Ho * 4, 2); c->o = calloc((size_t)c->W0 * c->H0 * 40, 2);
    return c;
}
static void case_free(Case *c) { free(c->col); free(c->R); free(c->o); free(c); }

static FiltParams make_fp(const Case *c, float jx, float jy, float pre, int hdr, int wide, int ox, int oy) {
    SrCall call = {(u32)c->Wo, (u32)c->Ho, (u32)c->Wr, (u32)c->Hr, jx, jy, 1, 1, pre, (u32)hdr, 0, 1, 0, (u32)wide, (u32)ox, (u32)oy, (u32)c->W0};
    FiltParams fp; sr_make_filt(&call, &g_mp, &fp);
    return fp;
}

/* set parameter p of quad t of every L0 pixel (or of one) */
static void set_v_all(Case *c, int t, int p, f16 v) {
    for (size_t i = 0; i < (size_t)c->W0 * c->H0; i++) c->o[i * 40 + 8 * (p / 2) + 2 * t + (p % 2)] = v;
}

/* run the filter alone on the GPU; the output texture has the origin (ox, oy) */
static void gpu_filter(SrGpu *gpu, const FiltParams *fp, const Case *c, f16 *out, f16 *hist) {
    id<MTLTexture> col = [g_gpu texture:MTLPixelFormatRGBA32Float w:c->Wr h:c->Hr], R = [g_gpu texture:MTLPixelFormatRGBA16Float w:c->Wo h:c->Ho];
    [col replaceRegion:MTLRegionMake2D(0, 0, c->Wr, c->Hr) mipmapLevel:0 withBytes:c->col bytesPerRow:(size_t)c->Wr * 16];
    [R replaceRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0 withBytes:c->R bytesPerRow:(size_t)c->Wo * 8];
    f16 en = (f16)c->e_now;
    id<MTLTexture> te = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
    [te replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&en bytesPerRow:2];
    int ox = (int)fp->ox, oy = (int)fp->oy;
    id<MTLTexture> ot = [g_gpu texture:MTLPixelFormatRGBA16Float w:c->Wo + ox h:c->Ho + oy], ht = [g_gpu texture:MTLPixelFormatRGBA16Float w:c->Wo h:c->Ho];
    id<MTLBuffer> ob = bufcopy(c->o, (size_t)c->W0 * c->H0 * 80);
    SrFiltIo io = {col, R, te, ot, ht, ob};
    run1(^(SrEnc *e) { sr_filter(e, gpu, fp, &io, (unsigned)c->W0); });
    f16 *tmp = malloc((size_t)(c->Wo + ox) * (c->Ho + oy) * 8);
    [ot getBytes:tmp bytesPerRow:(size_t)(c->Wo + ox) * 8 fromRegion:MTLRegionMake2D(0, 0, c->Wo + ox, c->Ho + oy) mipmapLevel:0];
    for (int y = 0; y < c->Ho; y++) memcpy(out + (size_t)y * c->Wo * 4, tmp + ((size_t)(y + oy) * (c->Wo + ox) + ox) * 4, (size_t)c->Wo * 8);
    free(tmp);
    [ht getBytes:hist bytesPerRow:(size_t)c->Wo * 8 fromRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0];
}

/* fraction of the pixels of the selected rows with all eight stored components within `ulps` (NaN differences count) */
static double frac_within(const Case *c, const f16 *go, const f16 *gh, const f16 *ro, const f16 *rh, int row_stride, int ulps, size_t *n_out) {
    size_t n = 0, ok = 0;
    for (int y = 0; y < c->Ho; y++) {
        if (y % row_stride) continue;
        for (int x = 0; x < c->Wo; x++) {
            size_t i = ((size_t)y * c->Wo + x) * 4; int good = 1;
            for (int k = 0; k < 4; k++) {
                const f16 *a = go + i + k, *b = ro + i + k, *a2 = gh + i + k, *b2 = rh + i + k;
                if (memcmp(a, b, 2) && (!isfinite((float)*a) || !isfinite((float)*b) || ulp_dist16(*a, *b) > ulps)) good = 0;
                if (memcmp(a2, b2, 2) && (!isfinite((float)*a2) || !isfinite((float)*b2) || ulp_dist16(*a2, *b2) > ulps)) good = 0;
            }
            n++; ok += good;
        }
    }
    if (n_out) *n_out = n;
    return n ? (double)ok / n : 1;
}

static void fill_random_o(Case *c, double vmax) {
    for (size_t i = 0; i < (size_t)c->W0 * c->H0; i++) for (int t = 0; t < 4; t++) for (int p = 0; p < 10; p++) {
        double v = (urand() * 2 - 1) * vmax;
        if (p == 8) v = -2 + urand() * 6;
        c->o[i * 40 + 8 * (p / 2) + 2 * t + (p % 2)] = (f16)v;
    }
}

/* head-filter, head-tile-bound: random parameters and colour against the CPU model */
static void test_filter(const char *name, int Wo, int Ho, int Wr, int Hr, int wide, int hdr, float jx, float jy, int row_stride, int controls) {
    Case *c = case_new(Wo, Ho, Wr, Hr);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) { for (int k = 0; k < 3; k++) c->col[4 * i + k] = (float)(hdr ? 3 * urand() * urand() : urand()); c->col[4 * i + 3] = (float)urand(); }
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) { for (int k = 0; k < 3; k++) c->R[4 * i + k] = (f16)(hdr ? 0.8 * urand() : urand()); c->R[4 * i + 3] = (f16)0; }
    fill_random_o(c, 3.0);
    float pre = hdr ? 1.5f : 1.0f; c->e_now = hdr ? 0.75f : 1.0f;
    FiltParams fp = make_fp(c, jx, jy, pre, hdr, wide, 0, 0);
    f16 *go = calloc((size_t)Wo * Ho * 4, 2), *gh = calloc((size_t)Wo * Ho * 4, 2), *ro = calloc((size_t)Wo * Ho * 4, 2), *rh = calloc((size_t)Wo * Ho * 4, 2);
    gpu_filter(g_gpu, &fp, c, go, gh);
    RefP12Opts opt = {0};
    ref_p12(&fp, &opt, c->W0, c->o, c->col, c->R, c->e_now, row_stride, ro, rh);
    size_t n; double f = frac_within(c, go, gh, ro, rh, row_stride, 2, &n);
    printf("%s (%dx%d render %dx%d, %s, %s, jitter %.4f %.4f): %.5f of %zu pixels within 2 ulp\n", name, Wo, Ho, Wr, Hr, wide ? "path W" : "path N", hdr ? "hdr" : "display", jx, jy, f, n);
    EXPECT(f >= 0.999, "%s", name);
    if (controls && wide) {
        RefP12Opts ctl[2] = {{.taps_n = 1}, {.form_swap = 1}};
        const char *cn[2] = {"3 x 3 taps on path W", "form A coordinates on path W"};
        for (int k = 0; k < 2; k++) {
            f16 *co = calloc((size_t)Wo * Ho * 4, 2), *ch = calloc((size_t)Wo * Ho * 4, 2);
            ref_p12(&fp, &ctl[k], c->W0, c->o, c->col, c->R, c->e_now, row_stride, co, ch);
            double fc = frac_within(c, go, gh, co, ch, row_stride, 2, NULL);
            printf("  control: %s: %.5f within 2 ulp\n", cn[k], fc);
            EXPECT(fc < 0.999, "control '%s' must fail", cn[k]);
            free(co); free(ch);
        }
    }
    free(go); free(gh); free(ro); free(rh); case_free(c);
}

/* output origin: writes land at (x + ox, y + oy) and nothing else changes */
static void test_origin(void) {
    Case *c = case_new(120, 70, 80, 47);
    for (size_t i = 0; i < (size_t)c->Wr * c->Hr; i++) { for (int k = 0; k < 4; k++) c->col[4 * i + k] = (float)urand(); }
    for (size_t i = 0; i < (size_t)c->Wo * c->Ho; i++) { for (int k = 0; k < 3; k++) c->R[4 * i + k] = (f16)urand(); }
    fill_random_o(c, 2.0);
    FiltParams a = make_fp(c, 0.1f, -0.2f, 1, 1, 0, 0, 0), b = make_fp(c, 0.1f, -0.2f, 1, 1, 0, 3, 5);
    size_t n = (size_t)c->Wo * c->Ho * 4;
    f16 *o0 = calloc(n, 2), *h0 = calloc(n, 2), *o1 = calloc(n, 2), *h1 = calloc(n, 2);
    gpu_filter(g_gpu, &a, c, o0, h0); gpu_filter(g_gpu, &b, c, o1, h1);
    int same = !memcmp(o0, o1, n * 2) && !memcmp(h0, h1, n * 2);
    printf("output origin (3, 5): output and history %s the zero-origin run\n", same ? "equal" : "DIFFER from");
    EXPECT(same, "origin");
    free(o0); free(h0); free(o1); free(h1); case_free(c);
}

/* head-coverage-ratio3: the covered mask against strict fp32 evaluation; also built in fast math (must fail) */
static int coverage_run(SrGpu *gpu, float jx, float jy, int verbose) {
    const int Wo = 480, Ho = 270, Wr = 160, Hr = 90;
    Case *c = case_new(Wo, Ho, Wr, Hr);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) { for (int k = 0; k < 4; k++) c->col[4 * i + k] = 1.0f; }
    for (int t = 0; t < 4; t++) { set_v_all(c, t, 8, (f16)20); set_v_all(c, t, 9, (f16)-20); }   /* sample weight A, blend weight 0 */
    FiltParams fp = make_fp(c, jx, jy, 1, 0, 1, 0, 0);
    f16 *go = calloc((size_t)Wo * Ho * 4, 2), *gh = calloc((size_t)Wo * Ho * 4, 2);
    gpu_filter(gpu, &fp, c, go, gh);
    size_t bad = 0, covered_n = 0;
    for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) {
        float u = coord_formB(fp.sx, fp.jx, x), v = coord_formB(fp.sy, fp.jy, y);
        float dx = roundf(u) - u, dy = roundf(v) - v;
        volatile float px = fp.cov_x * fabsf(dx), py = fp.cov_y * fabsf(dy);
        int cov = (py < 0.5f) && (px < 0.5f);
        int gpu_cov = (float)go[((size_t)y * Wo + x) * 4] != 0.0f;
        covered_n += cov; bad += cov != gpu_cov;
    }
    if (verbose) printf("  jitter (%+.4f, %+.4f): %zu of %d pixels covered, %zu mask mismatches\n", jx, jy, covered_n, Wo * Ho, bad);
    free(go); free(gh); case_free(c);
    return bad == 0;
}
static void test_coverage(void) {
    printf("head-coverage-ratio3: Wr = Wo / 3, path W\n");
    const float j = 1.0f / 6;
    const float js[4][2] = {{j, j}, {j, -j}, {-j, j}, {-j, -j}};
    int all = 1;
    for (int k = 0; k < 4; k++) all &= coverage_run(g_gpu, js[k][0], js[k][1], 1);
    EXPECT(all, "head-coverage-ratio3");
    /* fastmath-guard: the helpers without their strict-math pragmas keep the explicit fma and precise::round, so a fast-math
       build is immune and passes (reported, see QUESTIONS.md 1); the sensitivity of the test is shown by a build that
       rounds the coordinate product separately, which must fail */
    static const char *files[2] = {"build/filter_fast.metallib", "build/filter_unfused.metallib"};
    static const char *what[2] = {"helpers built in fast math, no strict pragmas", "coordinate product rounded separately (control, must fail)"};
    for (int b = 0; b < 2; b++) {
        NSData *lib = [NSData dataWithContentsOfFile:@(files[b])];
        NSString *err = nil;
        SrGpu *alt = lib ? [[SrGpu alloc] initWithDevice:g_gpu.device libraryData:lib error:&err] : nil;
        if (!alt) { printf("  FAIL: %s is missing or invalid (%s)\n", files[b], err.UTF8String); g_fails++; continue; }
        [g_gpu track:[alt tanhTable]]; [g_gpu commitResidency];   /* the alternative library has its own table */
        int ok = 1;
        printf("  %s:\n", what[b]);
        for (int k = 0; k < 4; k++) ok &= coverage_run(alt, js[k][0], js[k][1], 1);
        if (b == 1) EXPECT(!ok, "head-coverage-ratio3 must fail when the coordinate is evaluated with an unfused product");
    }
}

/* ---- weights with only the head_b bias set ---- */
static uint8_t *bias_weights(const f16 *bias40) {
    uint8_t *img = make_weights(WMODE_ZERO, 1.0);
    memcpy((void *)wt_of(img, 33), bias40, 40 * 2);
    return img;
}
static SrWeights upload(const uint8_t *img) {
    SrWeights w; w.buf = bufcopy(img, SR_WEIGHT_FILE_BYTES);
    sr_weights_prepare(g_gpu, &w);
    return w;
}
/* the head network on the GPU: m8 buffers d1, s0 -> o */
static void gpu_head(const uint8_t *img, int variant, const SrLevels *L, const uint8_t *d1, const uint8_t *s0, f16 *o) {
    SrWeights w = upload(img);
    size_t n0 = (size_t)L->W[0] * L->H[0], n1 = (size_t)L->W[1] * L->H[1];
    id<MTLBuffer> bd = bufcopy(d1, n1 * 32), bs = bufcopy(s0, n0 * 32), bo = [g_gpu buffer:n0 * 80];
    run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, L, variant, bd, bs, bo); });
    memcpy(o, bo.contents, n0 * 80);
}

/* head-network: random d1 / s0 and weights against the CPU model; control: one-rounding epilogue */
static void test_head_network(int Wo, int Ho, int variant) {
    RefLevels RL; ref_levels(Wo, Ho, &RL);
    SrLevels L; sr_levels(Wo, Ho, &L); L.wide = variant;
    size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    uint8_t *img = make_weights(WMODE_RANDOM, 1.0), *d1 = malloc(n1 * 32), *s0 = malloc(n0 * 32);
    for (size_t i = 0; i < n1 * 32; i++) d1[i] = (uint8_t)(0x30 + urand() * 0x50);
    for (size_t i = 0; i < n0 * 32; i++) s0[i] = (uint8_t)(0x30 + urand() * 0x50);
    f16 *go = malloc(n0 * 80), *ref = malloc(n0 * 80), *ctl = malloc(n0 * 80);
    gpu_head(img, variant, &L, d1, s0, go);
    ref_head(img, &RL, variant, d1, s0, ref, 1);
    ref_head(img, &RL, variant, d1, s0, ctl, 0);
    size_t eq = 0, bad = 0, eqc = 0, badc = 0, n = n0 * 40;
    for (size_t i = 0; i < n; i++) {
        eq += !memcmp(&go[i], &ref[i], 2); bad += ulp_dist16(go[i], ref[i]) > 2 && fabs((double)go[i] - (double)ref[i]) > 1e-3;   /* cancellation in the bias add inflates the ulp distance of a tiny result */
        eqc += !memcmp(&go[i], &ctl[i], 2); badc += ulp_dist16(go[i], ctl[i]) > 2 && fabs((double)go[i] - (double)ctl[i]) > 1e-3;
    }
    printf("head-network %dx%d variant %c: %.5f bit-equal to the two-rounding model, %zu beyond 2 ulp; one-rounding control: %.5f bit-equal, %zu beyond 2 ulp\n",
           Wo, Ho, "NW"[variant], (double)eq / n, bad, (double)eqc / n, badc);
    EXPECT(bad == 0 && (double)eq / n >= 0.995, "head-network");
    EXPECT((double)eqc / n < 0.99 * (double)eq / n || badc > 0, "head-network control (one-rounding epilogue) must fail");
    free(img); free(d1); free(s0); free(go); free(ref); free(ctl);
}

/* p12-fused: the fused pass (sr_p12: head network in threadgroup memory, tile-staged filter, P13) against the separate head (dec_head) and filter
   (p12_filter_{n,w}{0,1}) entries, bit for bit in the output and the history, with E_prev checked. A huge jitter takes the group-footprint fallback. */
static void test_fused(const char *name, int Wo, int Ho, int Wr, int Hr, int wide, int hdr, float jx, float jy) {
    SrLevels L; sr_levels(Wo, Ho, &L); L.wide = wide;
    size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    uint8_t *img = make_weights(WMODE_RANDOM, 1.0), *d1 = malloc(n1 * 32), *s0 = malloc(n0 * 32);
    for (size_t i = 0; i < n1 * 32; i++) d1[i] = (uint8_t)(0x30 + urand() * 0x50);
    for (size_t i = 0; i < n0 * 32; i++) s0[i] = (uint8_t)(0x30 + urand() * 0x50);
    Case *c = case_new(Wo, Ho, Wr, Hr);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) { for (int k = 0; k < 3; k++) c->col[i * 4 + k] = (float)(urand() * urand() * 3); c->col[i * 4 + 3] = (float)urand(); }
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) { for (int k = 0; k < 3; k++) c->R[i * 4 + k] = (f16)(urand() * 2); }
    c->col[5] = INFINITY; c->col[9] = NAN;                                 /* sanitised taps */
    FiltParams fp = make_fp(c, jx, jy, 1.0f, hdr, wide, 0, 0);
    SrWeights w = upload(img);
    id<MTLTexture> col = [g_gpu texture:MTLPixelFormatRGBA32Float w:Wr h:Hr], R = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho];
    [col replaceRegion:MTLRegionMake2D(0, 0, Wr, Hr) mipmapLevel:0 withBytes:c->col bytesPerRow:(size_t)Wr * 16];
    [R replaceRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0 withBytes:c->R bytesPerRow:(size_t)Wo * 8];
    id<MTLTexture> te = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1], tp = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
    f16 en = (f16)0.75f, ep0 = (f16)0.125f;
    [te replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&en bytesPerRow:2];
    [tp replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&ep0 bytesPerRow:2];
    id<MTLBuffer> bd = bufcopy(d1, n1 * 32), bs = bufcopy(s0, n0 * 32), bo = [g_gpu buffer:n0 * 80];
    id<MTLTexture> o1 = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], h1 = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho];
    id<MTLTexture> o2 = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], h2 = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho];
    SrFiltIo io = {col, R, te, o1, h1, bo};
    run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, &L, wide, bd, bs, bo); [e barrier]; sr_filter(e, g_gpu, &fp, &io, (unsigned)L.W[0]); });
    SrP12Io io2 = {col, R, te, o2, h2, tp, bd, bs};
    run1(^(SrEnc *e) { sr_p12(e, g_gpu, &w, &L, &fp, &io2); });
    size_t nb = (size_t)Wo * Ho * 8;
    uint8_t *a = malloc(nb), *b = malloc(nb), *a2 = malloc(nb), *b2 = malloc(nb);
    [o1 getBytes:a bytesPerRow:(size_t)Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    [o2 getBytes:b bytesPerRow:(size_t)Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    [h1 getBytes:a2 bytesPerRow:(size_t)Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    [h2 getBytes:b2 bytesPerRow:(size_t)Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    f16 epn = 0; [tp getBytes:&epn bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    int same = !memcmp(a, b, nb), same_h = !memcmp(a2, b2, nb), prev_ok = (epn == en);
    printf("%s %dx%d <- %dx%d path %c jitter (%g, %g): output %s, history %s, E_prev %s\n", name, Wo, Ho, Wr, Hr, "NW"[wide], jx, jy,
           same ? "bit-identical" : "DIFFERS", same_h ? "bit-identical" : "DIFFERS", prev_ok ? "= E_now" : "WRONG");
    EXPECT(same && same_h && prev_ok, "p12-fused %s", name);
    free(a); free(b); free(a2); free(b2); free(img); free(d1); free(s0); case_free(c);
}

/* head-quads: a distinct bias in each of the 40 head_b channels; the derived quad parameters of every quad against the model */
static void test_head_quads(void) {
    const int Wo = 64, Ho = 48;
    SrLevels L; sr_levels(Wo, Ho, &L);
    RefLevels RL; ref_levels(Wo, Ho, &RL);
    f16 bias[40]; for (int i = 0; i < 40; i++) bias[i] = (f16)(-2.0 + 0.1 * i + 0.0013 * i * i);
    uint8_t *img = bias_weights(bias);
    size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    uint8_t *z1 = calloc(n1 * 32, 1), *z0 = calloc(n0 * 32, 1);
    SrWeights w = upload(img);
    id<MTLBuffer> bd = bufcopy(z1, n1 * 32), bs = bufcopy(z0, n0 * 32), bo = [g_gpu buffer:n0 * 80], bg = [g_gpu buffer:(size_t)4 * n0 * 32];
    Case *c = case_new(Wo, Ho, Wo, Ho);
    FiltParams fp = make_fp(c, 0, 0, 1, 0, 0, 0, 0);
    run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, &L, 0, bd, bs, bo); [e barrier]; sr_quad_debug(e, g_gpu, &fp, bo, (unsigned)L.W[0], (unsigned)L.H[0], bg); });
    const f16 *o = bo.contents; const float *dbg = bg.contents;
    int bad = 0, distinct = 1;
    for (size_t pix = 0; pix < n0; pix += 7) {
        int X = (int)(pix % L.W[0]), Y = (int)(pix / L.W[0]);
        for (int t = 0; t < 4; t++) {
            for (int ch = 0; ch < 40; ch++) if (memcmp(&o[pix * 40 + ch], &bias[ch], 2)) bad++;
            f16 v[10]; ref_quad_v(bias, t, v);
            RefP12Opts opt = {0};
            RefQuad q = ref_quad(v, &fp, &opt);
            const float *d = dbg + (((size_t)(2 * Y + t / 2)) * (2 * L.W[0]) + (2 * X + t % 2)) * 8;
            double want[6] = {q.sigma_x, q.sigma_y, q.rho, q.alpha_s, q.beta, (double)q.a16};
            for (int k = 0; k < 6; k++) if (fabs(d[k] - want[k]) > 2e-5 * fabs(want[k]) + 2e-6 + (k == 5 ? 1e-3 : 0)) { bad++; if (getenv("SR_DEBUG")) printf("    quad %d value %d: got %g want %g\n", t, k, (double)d[k], want[k]); }
            if (d[6] != 1 || d[7] != 1) bad++;
        }
    }
    for (int t = 1; t < 4; t++) { const float *a = dbg, *b = dbg + (size_t)t * 8; /* first L0 pixel's quads are consecutive columns / rows */
        const float *q0 = dbg, *qt = dbg + (((size_t)(t / 2)) * (2 * L.W[0]) + (t % 2)) * 8; (void)a; (void)b; if (q0[0] == qt[0] && q0[1] == qt[1] && q0[3] == qt[3]) distinct = 0; }
    printf("head-quads: %d discrepancies between the 40 biases, the head output and the derived quad parameters; quads distinct: %d\n", bad, distinct);
    EXPECT(bad == 0 && distinct, "head-quads");
    free(img); free(z0); free(z1); case_free(c);
}

/* head-identity-net: known parameters through the whole pass, constant colour */
static void test_identity(int hdr) {
    const int Wo = 128, Ho = 64;
    Case *c = case_new(Wo, Ho, Wo, Ho);
    FiltParams fp = make_fp(c, 0, 0, 1, hdr, 0, 0, 0);
    f16 bias[40];
    for (int t = 0; t < 4; t++) {
        double v[10] = {0, 0, 0, 0, 0, 0, 0, 0, -20, 20};
        v[0] = v[1] = log((0.5 - g_mp.sigma_min) / g_mp.sigma_span / (1 - (0.5 - g_mp.sigma_min) / g_mp.sigma_span));   /* sigma = 1/2 */
        for (int p = 0; p < 10; p++) bias[8 * (p / 2) + 2 * t + (p % 2)] = (f16)v[p];
    }
    uint8_t *img = bias_weights(bias);
    SrLevels L; sr_levels(Wo, Ho, &L); RefLevels RL; ref_levels(Wo, Ho, &RL);
    size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    uint8_t *z1 = calloc(n1 * 32, 1), *z0 = calloc(n0 * 32, 1);
    const float C0[3] = {0.6f, 0.3f, 0.9f};
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) { for (int k = 0; k < 3; k++) { c->col[4 * i + k] = C0[k]; c->R[4 * i + k] = (f16)urand(); } c->col[4 * i + 3] = 1; }
    SrWeights w = upload(img);
    id<MTLBuffer> bd = bufcopy(z1, n1 * 32), bs = bufcopy(z0, n0 * 32), bo = [g_gpu buffer:n0 * 80];
    id<MTLTexture> col = [g_gpu texture:MTLPixelFormatRGBA32Float w:Wo h:Ho], R = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], te = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
    id<MTLTexture> ot = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], ht = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho];
    [col replaceRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0 withBytes:c->col bytesPerRow:Wo * 16];
    [R replaceRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0 withBytes:c->R bytesPerRow:Wo * 8];
    f16 one = (f16)1; [te replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:2];
    SrFiltIo io = {col, R, te, ot, ht, bo};
    run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, &L, 0, bd, bs, bo); [e barrier]; sr_filter(e, g_gpu, &fp, &io, (unsigned)L.W[0]); });
    f16 *go = malloc((size_t)Wo * Ho * 8), *gh = malloc((size_t)Wo * Ho * 8), *ro = malloc((size_t)Wo * Ho * 8), *rh = malloc((size_t)Wo * Ho * 8);
    [ot getBytes:go bytesPerRow:Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    [ht getBytes:gh bytesPerRow:Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
    RefP12Opts opt = {0};
    ref_p12(&fp, &opt, L.W[0], (const f16 *)bo.contents, c->col, c->R, 1.0f, 1, ro, rh);
    size_t n;
    double f = frac_within(c, go, gh, ro, rh, 1, 2, &n);
    int rec_ok = 1, col_bad = 0;
    for (size_t i = 0; i < (size_t)Wo * Ho; i++) {
        rec_ok &= bits16(gh[4 * i + 3]) == 0;
        for (int k = 0; k < 3; k++) col_bad += ulp_dist16(go[4 * i + k], (f16)C0[k]) > 4;
    }
    printf("head-identity-net %s: %.5f within 2 ulp of the model; output within 4 ulp of the colour: %s; recurrent scalar exactly 0: %s\n", hdr ? "hdr" : "display", f, col_bad ? "NO" : "yes", rec_ok ? "yes" : "NO");
    EXPECT(f >= 0.999 && col_bad == 0 && rec_ok, "head-identity-net");
    free(img); free(z0); free(z1); free(go); free(gh); free(ro); free(rh); case_free(c);
}

/* ---- head-exceptional (sr.head_filter_r4 section 7) ---- */
static void ex_class(int run_hdr, int cls, float *C, f16 *R) {
    static const float D[4][7] = {{0.25f, 0.5f, 0.75f, 1.0f, 0.5f, 0.5f, 0.5f}, {3, 3, 3, 1, 1, 1, 1}, {1024, 1024, 1024, 1, 0, 0, 0}, {1e5f, 0.5f, 0.5f, 1e5f, 0.5f, 0.5f, 0.5f}};
    static const float H[4][7] = {{0.25f, 0.5f, 0.75f, 1.0f, 0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f, 1, -4, 0, 0}, {8, 8, 8, 1, 0, 0, 0}, {-32, 0, 0, 1, 0.5f, 0.5f, 0.5f}};
    const float *s = run_hdr ? H[cls] : D[cls];
    for (int k = 0; k < 4; k++) C[k] = s[k];
    for (int k = 0; k < 3; k++) R[k] = (f16)s[4 + k];
    R[3] = (f16)0;
}
typedef struct { f16 *go, *gh; } ExOut;
static int ex_check(const char *tag, const f16 *go, const f16 *gh, const f16 *ro, const f16 *rh, size_t npix, int verbose) {
    size_t nonfinite = 0, rec_bad = 0, far = 0;
    for (size_t i = 0; i < npix; i++) {
        for (int k = 0; k < 4; k++) nonfinite += !isfinite((float)go[4 * i + k]) + !isfinite((float)gh[4 * i + k]);
        rec_bad += memcmp(&gh[4 * i + 3], &rh[4 * i + 3], 2) != 0;
        for (int k = 0; k < 4; k++) {
            far += ulp_dist16(go[4 * i + k], ro[4 * i + k]) > 2;
            if (k < 3) far += ulp_dist16(gh[4 * i + k], rh[4 * i + k]) > 2;
        }
    }
    if (verbose) printf("  %s: non-finite stored components %zu, recurrent channel differing %zu, other components beyond 2 ulp %zu\n", tag, nonfinite, rec_bad, far);
    return nonfinite == 0 && rec_bad == 0 && far == 0;
}
static void test_exceptional(void) {
    const int Wo = 256, Ho = 128;
    SrParams saved = g_mp;
    g_mp = test_params_r3(); g_mp.alpha_0 = 64;
    f16 bias[40];
    for (int t = 0; t < 4; t++) {
        double v[10] = {0, 0, 0, 0, 0, 0, 0, 0, 16, 0};
        if (t == 1) { v[0] = INFINITY; v[8] = -INFINITY; }
        if (t == 2) v[3] = NAN;
        if (t == 3) v[9] = NAN;
        for (int p = 0; p < 10; p++) bias[8 * (p / 2) + 2 * t + (p % 2)] = (f16)v[p];
    }
    uint8_t *img = bias_weights(bias);
    SrLevels L; sr_levels(Wo, Ho, &L);
    size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    uint8_t *z1 = calloc(n1 * 32, 1), *z0 = calloc(n0 * 32, 1);
    SrWeights w = upload(img);
    printf("head-exceptional: %dx%d native, alpha_0 = 64\n", Wo, Ho);
    for (int run = 0; run < 2; run++) {
        int hdr = run;
        Case *c = case_new(Wo, Ho, Wo, Ho);
        for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) ex_class(hdr, ((x / 8) + (y / 8)) % 4, c->col + ((size_t)y * Wo + x) * 4, c->R + ((size_t)y * Wo + x) * 4);
        FiltParams fp = make_fp(c, 0, 0, hdr ? 8.0f : 1.0f, hdr, 0, 0, 0);
        id<MTLBuffer> bd = bufcopy(z1, n1 * 32), bs = bufcopy(z0, n0 * 32), bo = [g_gpu buffer:n0 * 80];
        id<MTLTexture> col = [g_gpu texture:MTLPixelFormatRGBA32Float w:Wo h:Ho], R = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], te = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
        id<MTLTexture> ot = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho], ht = [g_gpu texture:MTLPixelFormatRGBA16Float w:Wo h:Ho];
        [col replaceRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0 withBytes:c->col bytesPerRow:Wo * 16];
        [R replaceRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0 withBytes:c->R bytesPerRow:Wo * 8];
        f16 one = (f16)1; [te replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:2];
        SrFiltIo io = {col, R, te, ot, ht, bo};
        run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, &L, 0, bd, bs, bo); [e barrier]; sr_filter(e, g_gpu, &fp, &io, (unsigned)L.W[0]); });
        size_t np = (size_t)Wo * Ho;
        f16 *go = malloc(np * 8), *gh = malloc(np * 8), *ro = calloc(np * 4, 2), *rh = calloc(np * 4, 2);
        [ot getBytes:go bytesPerRow:Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
        [ht getBytes:gh bytesPerRow:Wo * 8 fromRegion:MTLRegionMake2D(0, 0, Wo, Ho) mipmapLevel:0];
        memcpy(c->o, bo.contents, n0 * 80);
        int kinds[5] = {0};
        RefP12Opts opt = {.kinds = kinds};
        ref_p12(&fp, &opt, L.W[0], c->o, c->col, c->R, 1.0f, 1, ro, rh);
        printf(" run %c: pixels by kind: Q kept with a component >= 64: %d, Q' = H: %d, Q' = 0: %d, saturated store: %d, rec = 0 from a NaN parameter: %d\n", hdr ? 'H' : 'D', kinds[0], kinds[1], kinds[2], kinds[3], kinds[4]);
        /* non-void check */
        static const int need_d[5] = {1, 1, 0, 1, 1}, need_h[5] = {0, 1, 1, 1, 1};
        for (int k = 0; k < 5; k++) if ((hdr ? need_h : need_d)[k] && kinds[k] < 256) { printf("  FAIL: head-exceptional is misconfigured: kind %d has %d pixels (< 256)\n", k, kinds[k]); g_fails++; }
        int ok = ex_check("GPU vs model", go, gh, ro, rh, np, 1);
        EXPECT(ok, "head-exceptional run %c", hdr ? 'H' : 'D');
        /* controls: the model of the defective rule must fail at least one threshold in this run (collected over both runs below) */
        static int ctl_failed[4];
        RefP12Opts ctl[4] = {{.old_overflow = 1}, {.rn16_store = 1}, {.rec_clamp = 1}, {.no_unusable = 1}};
        static const char *cn[4] = {"the superseded overflow rule", "rn16 instead of sat16 at the stores", "a NaN recurrent scalar clamped with suppressing min and max", "no unusable-colour rule"};
        for (int k = 0; k < 4; k++) {
            f16 *co = calloc(np * 4, 2), *ch = calloc(np * 4, 2);
            ref_p12(&fp, &ctl[k], L.W[0], c->o, c->col, c->R, 1.0f, 1, co, ch);
            int cok = ex_check(cn[k], co, ch, ro, rh, np, 0);
            printf("  control (%s): %s in run %c\n", cn[k], cok ? "passes" : "fails", hdr ? 'H' : 'D');
            ctl_failed[k] |= !cok;
            free(co); free(ch);
            if (run == 1) EXPECT(ctl_failed[k], "head-exceptional control '%s' must fail", cn[k]);
        }
        free(go); free(gh); free(ro); free(rh); case_free(c);
    }
    free(img); free(z0); free(z1);
    g_mp = saved;
}

int main(void) {
    @autoreleasepool {
        gpu_init();
        g_mp = test_params_r3();
        (void)[g_gpu tanhTable];     /* every allocation must be resident before the first encode that uses it */
        [g_gpu commitResidency];
        test_head_network(300, 170, 0);
        test_head_network(300, 170, 1);
        test_head_quads();
        test_fused("p12-fused", 203, 133, 135, 89, 0, 1, 0.2f, -0.3f);
        test_fused("p12-fused", 203, 133, 135, 89, 0, 0, -0.4f, 0.1f);
        test_fused("p12-fused", 321, 181, 107, 61, 1, 1, 1.0f / 6, -1.0f / 6);
        test_fused("p12-fused", 321, 181, 107, 61, 1, 0, 0.3f, 0.2f);
        test_fused("p12-fused", 160, 96, 160, 96, 0, 1, 0.5f, -0.5f);
        test_fused("p12-fused footprint fallback", 160, 96, 80, 48, 0, 1, 3.0e7f, -2.0e7f);
        test_identity(1);
        test_identity(0);
        test_filter("head-filter", 200, 120, 200, 120, 0, 1, 0.2f, -0.3f, 1, 0);
        test_filter("head-filter", 200, 120, 200, 120, 0, 0, 0.0f, 0.0f, 1, 0);
        test_filter("head-filter", 240, 135, 171, 96, 0, 1, -0.4f, 0.1f, 1, 0);
        test_filter("head-filter", 300, 180, 100, 60, 1, 1, 1.0f / 6, -1.0f / 6, 1, 1);
        test_filter("head-filter", 300, 180, 100, 60, 1, 0, -1.0f / 6, 1.0f / 6, 1, 1);
        test_filter("head-filter", 300, 180, 200, 120, 1, 1, 0.3f, 0.2f, 1, 0);
        test_origin();
        test_coverage();
        /* head-tile-bound: path W at scales 1/3, 38/63 (largest width not above) and 2/3, and native on path N, output 1920x1080 */
        test_filter("head-tile-bound s=1/3", 1920, 1080, 640, 360, 1, 1, 0.1f, -0.2f, 8, 0);
        test_filter("head-tile-bound s<=38/63", 1920, 1080, 1158, 651, 1, 1, -0.3f, 0.4f, 8, 0);
        test_filter("head-tile-bound s=2/3", 1920, 1080, 1280, 720, 1, 1, 0.25f, -0.25f, 8, 0);
        test_filter("head-tile-bound native", 1920, 1080, 1920, 1080, 0, 1, 0.5f, -0.5f, 8, 0);
        test_exceptional();
        printf(g_fails ? "p12_gpu: %d FAILURES\n" : "p12_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
