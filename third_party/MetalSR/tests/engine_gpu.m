// End-to-end tests of the public C interface against the CPU model of the whole call, on the synthetic scenes.
#include "enghost.h"

/* comparison domain of sr.test_scenes_r2: D(c) = c' / (1 + max channel of c'), c' = max(c, 0) */
static void domain_D(const float *c, double *d) {
    double m = 0, v[3];
    for (int k = 0; k < 3; k++) { v[k] = c[k] > 0 ? c[k] : 0; if (v[k] > m) m = v[k]; }
    for (int k = 0; k < 3; k++) d[k] = v[k] / (1 + m);
}
static double psnr_tm(const f16 *a, const f16 *b, size_t npix, int *nonfinite) {
    double se = 0; *nonfinite = 0;
    for (size_t i = 0; i < npix; i++) {
        float ca[3], cb_[3];
        for (int k = 0; k < 3; k++) { ca[k] = (float)a[4 * i + k]; cb_[k] = (float)b[4 * i + k]; if (!isfinite(ca[k])) { (*nonfinite)++; ca[k] = 0; } }
        double da[3], db[3];
        domain_D(ca, da); domain_D(cb_, db);
        for (int k = 0; k < 3; k++) { double d = da[k] - db[k]; se += d * d; }
    }
    double mse = se / (npix * 3);
    return mse <= 0 ? 200 : 10 * log10(1.0 / mse);
}

typedef struct {
    const char *name; int scene, inner, Wo, Ho, Wmax, Hmax, frames;
    int hdr, mvr, dinv, autoexp;
    int sizes[4][2], nsizes, period;      /* render sizes cycled every `period` frames (0: fixed max) */
    int resets[4];                        /* frames with reset (-1 terminated) */
    double min_psnr;
} Cfg;

static const char *g_wfile = "build/e2e_w.bin";

static void run_cfg(const Cfg *c) {
    printf("%s (%dx%d, max render %dx%d, %d frames)\n", c->name, c->Wo, c->Ho, c->Wmax, c->Hmax, c->frames);
    MetalSRCreateInfo ci = {METALSR_INTERFACE_VERSION, (uint32_t)c->Wmax, (uint32_t)c->Hmax, (uint32_t)c->Wo, (uint32_t)c->Ho,
                        (uint32_t)((c->hdr ? METALSR_FLAG_HDR : 0) | (c->dinv ? METALSR_FLAG_DEPTH_INVERTED : 0) | (c->mvr ? METALSR_FLAG_MV_RENDER_RES : 0) | (c->autoexp ? METALSR_FLAG_AUTO_EXPOSURE : 0)),
                        g_wfile, NULL, NULL, 0, 0};
    char err[512] = "";
    MetalSREngine *e = metalsr_create((__bridge void *)dev, &ci, err, sizeof err);
    if (!e) { printf("  create failed: %s\n", err); g_fails++; return; }
    EXPECT(metalsr_debug_path(e) == (((double)c->Wo / (double)c->Wmax) > g_mp.r_wide), "path selection (W = %d)", metalsr_debug_path(e));
    RefState *st = ref_state_new(c->Wo, c->Ho);
    id<MTLTexture> otex = tex2d(MTLPixelFormatRGBA16Float, c->Wo, c->Ho, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    f16 *og = malloc((size_t)c->Wo * c->Ho * 8), *or_ = malloc((size_t)c->Wo * c->Ho * 8);
    double worst = 1e9; int nf_total = 0;
    for (int f = 0; f < c->frames; f++) {
        int Wr = c->Wmax, Hr = c->Hmax;
        if (c->nsizes) { int k = (f / c->period) % c->nsizes; Wr = c->sizes[k][0]; Hr = c->sizes[k][1]; }
        float jx = (float)halton(f % 16, 2) - 0.5f, jy = (float)halton(f % 16, 3) - 0.5f;
        int Wm = c->mvr ? Wr : c->Wo, Hm = c->mvr ? Hr : c->Ho;
        float *col = malloc((size_t)Wr * Hr * 16), *dep = malloc((size_t)Wr * Hr * 4), *mot = malloc((size_t)Wm * Hm * 8);
        scene_render(c->scene, c->inner, f, c->Wo, c->Ho, Wr, Hr, c->mvr, c->dinv, jx, jy, col, dep, mot);
        f16 *col16 = malloc((size_t)Wr * Hr * 8), *mot16 = malloc((size_t)Wm * Hm * 4);
        for (size_t i = 0; i < (size_t)Wr * Hr * 4; i++) { col16[i] = (f16)col[i]; col[i] = (float)col16[i]; }
        for (size_t i = 0; i < (size_t)Wm * Hm * 2; i++) { mot16[i] = (f16)mot[i]; mot[i] = (float)mot16[i]; }
        id<MTLTexture> tc = tex2d(MTLPixelFormatRGBA16Float, Wr, Hr, MTLTextureUsageShaderRead), td = tex2d(MTLPixelFormatR32Float, Wr, Hr, MTLTextureUsageShaderRead),
                       tmv = tex2d(MTLPixelFormatRG16Float, Wm, Hm, MTLTextureUsageShaderRead);
        put(tc, 0, 0, Wr, Hr, col16, 8); put(td, 0, 0, Wr, Hr, dep, 4); put(tmv, 0, 0, Wm, Hm, mot16, 4);
        int reset = 0; for (int i = 0; i < 4 && c->resets[i] >= 0; i++) if (c->resets[i] == f) reset = 1;
        MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
        fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tmv; fi.output = (__bridge void *)otex;
        fi.rw = Wr; fi.rh = Hr; fi.jitter_x = jx; fi.jitter_y = jy; fi.mv_scale_x = 1; fi.mv_scale_y = 1; fi.pre_exposure = 1; fi.reset = reset;
        int rc = host_call(e, &fi);
        EXPECT(rc == 0, "encode returned %d (%s)", rc, metalsr_last_error(e));
        [otex getBytes:og bytesPerRow:c->Wo * 8 fromRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0];
        SrCall call = {(u32)c->Wo, (u32)c->Ho, (u32)Wr, (u32)Hr, jx, jy, 1, 1, 1, (u32)c->hdr, (u32)c->dinv, (u32)c->mvr, (u32)reset,
                       (u32)(((double)c->Wo / (double)c->Wmax) > g_mp.r_wide), 0, 0, 0};
        ref_frame(st, g_img, &g_mp, &call, c->autoexp, 1.0f, col, dep, mot, or_);
        int nf; double ps = psnr_tm(og, or_, (size_t)c->Wo * c->Ho, &nf);
        nf_total += nf;
        if (ps < 50 && getenv("SR_DEBUG")) { double bd = 0; size_t bi = 0; for (size_t i = 0; i < (size_t)c->Wo * c->Ho; i++) { double d = fabs((double)og[4*i] - (double)or_[4*i]) / (fabs((double)or_[4*i]) + 1); if (d > bd) { bd = d; bi = i; } } printf("    worst pixel (%zu,%zu): gpu %g %g %g cpu %g %g %g\n", bi % c->Wo, bi / c->Wo, (double)og[4*bi], (double)og[4*bi+1], (double)og[4*bi+2], (double)or_[4*bi], (double)or_[4*bi+1], (double)or_[4*bi+2]); }
        if (ps < worst) worst = ps;
        if (f < 3 || f % 10 == 9 || f == c->frames - 1) printf("  frame %2d (%dx%d): PSNR vs CPU model %.1f dB\n", f, Wr, Hr, ps);
        free(col); free(dep); free(mot); free(col16); free(mot16);
    }
    printf("  worst %.1f dB, non-finite values %d\n", worst, nf_total);
    EXPECT(worst >= c->min_psnr && nf_total == 0, "%s: PSNR %.1f below %.1f or non-finite output", c->name, worst, c->min_psnr);
    metalsr_destroy(e);
    ref_state_free(st); free(og); free(or_);
}

int main(void) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newMTL4CommandQueue]; alloc = [dev newCommandAllocator]; evt = [dev newSharedEvent];
        g_mp = test_params();
        write_weights("build/e2e_w.bin", WMODE_RANDOM, 1.0);
        Cfg cfgs[] = {
            {"e2e path N hdr", SC_PAN, SC_MIX, 256, 144, 171, 96, 5, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e path W hdr low-res motion", SC_DISOCC, 0, 256, 144, 85, 48, 4, 1, 0, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e display-referred native", SC_PAN, SC_STRIPES, 200, 120, 200, 120, 4, 0, 0, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e auto exposure mix", SC_PAN, SC_MIX, 256, 144, 171, 96, 6, 1, 1, 0, 1, {{0}}, 0, 0, {-1}, 40},
            {"e2e auto exposure + resets", SC_PAN, SC_DISCS, 256, 144, 171, 96, 6, 1, 1, 0, 1, {{0}}, 0, 0, {2, -1}, 40},
            {"e2e resets", SC_PAN, SC_MIX, 256, 144, 171, 96, 6, 1, 1, 0, 0, {{0}}, 0, 0, {2, -1}, 40},
            {"e2e discs all reset (bright pixels, ill-conditioned inverse tonemap)", SC_PAN, SC_DISCS, 256, 144, 171, 96, 5, 1, 1, 0, 0, {{0}}, 0, 0, {1, 2, 3, 4}, 30},
            {"e2e dynamic resolution", SC_PAN, SC_MIX, 256, 144, 171, 96, 6, 1, 1, 0, 0, {{171, 96}, {128, 72}, {149, 84}}, 3, 2, {-1}, 40},
            {"e2e inverted depth", SC_PARALLAX, 0, 256, 144, 171, 96, 4, 1, 1, 1, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e odd size", SC_PAN, SC_MIX, 211, 133, 141, 89, 4, 0, 0, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e long stripes ratio 3 (60 frames)", SC_PAN, SC_STRIPES, 480, 270, 160, 90, 60, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"e2e long stripes ratio 1.5 (60 frames)", SC_PAN, SC_STRIPES, 480, 270, 320, 180, 60, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
        };
        for (size_t i = 0; i < sizeof cfgs / sizeof cfgs[0]; i++) run_cfg(&cfgs[i]);
        /* head-tile-bound (sr.head_filter_r3): the harness writes its own parameter file; scales 1/3, 38/63 (largest width not above), 2/3
           on path W and a native run on path N, at a reduced output size (the CPU model of the whole call is slow) */
        g_mp = test_params_r3();
        write_weights("build/e2e_r3.bin", WMODE_RANDOM, 1.0);
        g_wfile = "build/e2e_r3.bin";
        Cfg tile[] = {
            {"head-tile-bound s=1/3 (path W)", SC_PAN, SC_MIX, 480, 270, 160, 90, 4, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"head-tile-bound s<=38/63 (path W)", SC_PAN, SC_MIX, 480, 270, 289, 162, 4, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"head-tile-bound s=2/3 (path W)", SC_PAN, SC_MIX, 480, 270, 320, 180, 4, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
            {"head-tile-bound native (path N)", SC_PAN, SC_MIX, 480, 270, 480, 270, 4, 1, 1, 0, 0, {{0}}, 0, 0, {-1}, 40},
        };
        for (size_t i = 0; i < sizeof tile / sizeof tile[0]; i++) run_cfg(&tile[i]);
        printf(g_fails ? "engine_gpu: %d FAILURES\n" : "engine_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
