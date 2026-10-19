// Finiteness of the engine's output for finite inputs: tiny renders, extreme aspect ratios, size changes between calls on one
// engine instance, and the two failures reported by the oracle (output 2175x2751 with 28x821 then 1x511; output 663x2354 with
// 3x632 then 3x677). Every output pixel (all four channels) must be finite.
// The reported cases and the sweep run for the three output paths: RGBA16Float (written in place by the filter) and RG11B10Float and
// RGBA32Float (through the conversion pass). Controls (build.sh gputest): the shaders built with SR_INJECT_NONFINITE corrupt some
// stores of the filter and of the conversion pass; with the output guard this test must pass, without it (SR_NO_OUTPUT_GUARD) it must fail.
// SR_FUZZ=n, SR_FUZZ_ORACLE=n, SR_BRIGHT=1 and SR_FMT=rg11b10 select extra exploratory runs.
#include "enghost.h"

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (g_rng >> 11) * (1.0 / 9007199254740992.0); }

typedef struct { const char *name; int Wo, Ho, Wmax, Hmax, hdr, mvr, scene, inner; int sizes[8][2], n, resets; int fuzz; float pre; } Case;   /* resets: bit mask of calls with the reset flag */

static const char *g_wfile = "build/fin_w.bin";


// random finite inputs of widely different magnitudes (fp16-representable)
static float rmag(void) {
    static const float v[] = {0.0f, 1e-7f, 1e-4f, 0.01f, 0.5f, 1.0f, 2.0f, 30.0f, 1000.0f, 60000.0f, 65504.0f};
    float x = v[(size_t)(urand() * (sizeof v / sizeof *v))];
    return urand() < 0.5 ? x : (float)(x * urand());
}
static void fuzz_inputs(const Case *c, int Wr, int Hr, int Wm, int Hm, float *col, float *dep, float *mot) {
    int style = (int)(urand() * 3);                                  // 0 smooth random, 1 random per pixel, 2 sparse extremes
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) {
        for (int k = 0; k < 3; k++) col[4 * i + k] = style == 0 ? (float)(0.5 + 0.5 * sin(0.01 * i + k)) : (style == 1 ? rmag() : (urand() < 0.02 ? rmag() : 0.3f));
        col[4 * i + 3] = urand() < 0.5 ? 1.0f : rmag();
        dep[i] = urand() < 0.3 ? 0.5f : (float)urand();
    }
    for (size_t i = 0; i < (size_t)Wm * Hm * 2; i++) mot[i] = style == 0 ? (float)(3 * (urand() - 0.5)) : (urand() < 0.5 ? 0.0f : (urand() < 0.5 ? (float)(40 * (urand() - 0.5)) : (urand() < 0.5 ? rmag() : -rmag())));
}
static float g_maxv;
static MTLPixelFormat g_fmt = MTLPixelFormatRGBA16Float;

/* runs the calls of one case on one engine; returns the number of non-finite output values (and prints where) */
static long run_case(const Case *c, MTLPixelFormat ofmt) {
    const int ae = c->fuzz && c->hdr && urand() < 0.3;   // fuzz runs use the exposure meter or an exposure texture of extreme value
    MetalSRCreateInfo ci = {METALSR_INTERFACE_VERSION, (uint32_t)c->Wmax, (uint32_t)c->Hmax, (uint32_t)c->Wo, (uint32_t)c->Ho,
                        (uint32_t)((c->hdr ? METALSR_FLAG_HDR : 0) | (c->mvr ? METALSR_FLAG_MV_RENDER_RES : 0) | (ae ? METALSR_FLAG_AUTO_EXPOSURE : 0)), g_wfile, NULL, NULL, 0, 0};
    char err[512] = "";
    MetalSREngine *e = metalsr_create((__bridge void *)dev, &ci, err, sizeof err);
    if (!e) { printf("  %s: create failed: %s\n", c->name, err); g_fails++; return 1; }
    id<MTLTexture> otex = tex2d(ofmt, c->Wo, c->Ho, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    const int bpp = ofmt == MTLPixelFormatRG11B10Float ? 4 : (ofmt == MTLPixelFormatRGBA32Float ? 16 : 8);
    f16 *og = malloc((size_t)c->Wo * c->Ho * 16);
    long bad = 0;
    for (int f = 0; f < c->n; f++) {
        int Wr = c->sizes[f][0], Hr = c->sizes[f][1];
        float jx = (float)halton(f % 16 + 1, 2) - 0.5f, jy = (float)halton(f % 16 + 1, 3) - 0.5f;
        int Wm = c->mvr ? Wr : c->Wo, Hm = c->mvr ? Hr : c->Ho;
        float *col = malloc((size_t)Wr * Hr * 16), *dep = malloc((size_t)Wr * Hr * 4), *mot = malloc((size_t)Wm * Hm * 8);
        if (c->fuzz) fuzz_inputs(c, Wr, Hr, Wm, Hm, col, dep, mot); else scene_render(c->scene, c->inner, f, c->Wo, c->Ho, Wr, Hr, c->mvr, 0, jx, jy, col, dep, mot);
        f16 *col16 = malloc((size_t)Wr * Hr * 8), *mot16 = malloc((size_t)Wm * Hm * 4);
        for (size_t i = 0; i < (size_t)Wr * Hr * 4; i++) col16[i] = (f16)col[i];
        for (size_t i = 0; i < (size_t)Wm * Hm * 2; i++) mot16[i] = (f16)mot[i];
        id<MTLTexture> tc = tex2d(MTLPixelFormatRGBA16Float, Wr, Hr, MTLTextureUsageShaderRead), td = tex2d(MTLPixelFormatR32Float, Wr, Hr, MTLTextureUsageShaderRead),
                       tmv = tex2d(MTLPixelFormatRG16Float, Wm, Hm, MTLTextureUsageShaderRead);
        put(tc, 0, 0, Wr, Hr, col16, 8); put(td, 0, 0, Wr, Hr, dep, 4); put(tmv, 0, 0, Wm, Hm, mot16, 4);
        MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
        fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tmv; fi.output = (__bridge void *)otex;
        id<MTLTexture> texp = nil;
        if (c->fuzz && !ae && urand() < 0.6) {
            static const float ev[] = {0.0f, 1e-30f, 1e-8f, 1e-3f, 1.0f, 1e3f, 1e8f, 3e38f};
            float v = ev[(size_t)(urand() * 8)];
            texp = tex2d(MTLPixelFormatR32Float, 1, 1, MTLTextureUsageShaderRead); put(texp, 0, 0, 1, 1, &v, 4);
            fi.exposure = (__bridge void *)texp;
        }
        fi.rw = Wr; fi.rh = Hr; fi.jitter_x = jx; fi.jitter_y = jy; fi.mv_scale_x = 1; fi.mv_scale_y = 1; fi.pre_exposure = c->pre > 0 ? c->pre : 1; fi.reset = (c->resets >> f) & 1;
        int rc = host_call(e, &fi);
        if (rc != 0) { printf("  %s call %d (%dx%d): encode returned %d (%s)\n", c->name, f, Wr, Hr, rc, metalsr_last_error(e)); g_fails++; }
        [otex getBytes:og bytesPerRow:c->Wo * bpp fromRegion:MTLRegionMake2D(0, 0, c->Wo, c->Ho) mipmapLevel:0];
        long nb = 0; int fx = -1, fy = -1; float fv = 0;
        if (bpp == 8) {
            for (size_t i = 0; i < (size_t)c->Wo * c->Ho * 4; i++) {
                if (isfinite((float)og[i]) && fabsf((float)og[i]) > g_maxv) g_maxv = fabsf((float)og[i]);
                if (!isfinite((float)og[i])) { if (!nb) { fx = (int)((i / 4) % c->Wo); fy = (int)((i / 4) / c->Wo); fv = (float)og[i]; } nb++; }
            }
        } else if (bpp == 16) {
            const float *pf = (const float *)og;
            for (size_t i = 0; i < (size_t)c->Wo * c->Ho * 4; i++)
                if (!isfinite(pf[i])) { if (!nb) { fx = (int)((i / 4) % c->Wo); fy = (int)((i / 4) / c->Wo); fv = pf[i]; } nb++; }
        } else {   // RG11B10Float: an all-ones exponent field is an infinity or a NaN
            const uint32_t *pk = (const uint32_t *)og;
            for (size_t i = 0; i < (size_t)c->Wo * c->Ho; i++) {
                uint32_t v = pk[i];
                int bad = ((v >> 6) & 31) == 31 || ((v >> 17) & 31) == 31 || ((v >> 27) & 31) == 31;
                if (bad) { if (!nb) { fx = (int)(i % c->Wo); fy = (int)(i / c->Wo); fv = (float)(v & 0x7FF); } nb++; }
            }
        }
        if (nb) printf("  %s call %d (%dx%d): %ld non-finite values, first at (%d,%d) = %g\n", c->name, f, Wr, Hr, nb, fx, fy, (double)fv);
        bad += nb;
        free(col); free(dep); free(mot); free(col16); free(mot16);
    }
    free(og);
    metalsr_destroy(e);
    return bad;
}

int main(void) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newMTL4CommandQueue]; alloc = [dev newCommandAllocator]; evt = [dev newSharedEvent];
        g_mp = test_params();
        write_weights("build/fin_w.bin", WMODE_RANDOM, 1.0);
        static const MTLPixelFormat fmts[3] = {MTLPixelFormatRGBA16Float, MTLPixelFormatRG11B10Float, MTLPixelFormatRGBA32Float};
        long total = 0, swept = 0; int ncases = 0;
        for (int fi_ = 0; fi_ < 3; fi_++) {
        g_fmt = fmts[fi_];
        printf("output format %d\n", (int)g_fmt);
        // the two reported failures (rows r-u-42 and r-u-27 of the oracle's T1 tier: HDR, render-resolution motion, resets on calls 0 and 2),
        // each also without resets and in other scenes
        for (int hdr = 1; hdr >= 0; hdr--) for (int rs_ = 0; rs_ < 2; rs_++) for (int sc = 0; sc < 3; sc++) {
            static const int scenes[3][2] = {{SC_PAN, SC_MIX}, {SC_PAN, SC_DISCS}, {SC_PAN, SC_STRIPES}};
            char n1[96], n2[96];
            snprintf(n1, sizeof n1, "oracle-1 hdr%d resets%d scene%d", hdr, rs_, sc);
            snprintf(n2, sizeof n2, "oracle-2 hdr%d resets%d scene%d", hdr, rs_, sc);
            Case a = {n1, 2175, 2751, 109, 1941, hdr, 1, scenes[sc][0], scenes[sc][1], {{28, 821}, {1, 511}, {69, 986}, {96, 960}}, 4, rs_ ? 5 : 1, 0, 0};
            Case b = {n2, 663, 2354, 7, 1181, hdr, 1, scenes[sc][0], scenes[sc][1], {{3, 632}, {3, 677}, {4, 622}, {7, 316}}, 4, rs_ ? 5 : 1, 0, 0};
            total += run_case(&a, g_fmt) + run_case(&b, g_fmt);
        }
        for (int hdr = 1; hdr >= 0; hdr--) for (int sc = 0; sc < 2; sc++) {   // tiny renders into large outputs (the largest sample weights)
            char nm[64]; snprintf(nm, sizeof nm, "tiny-into-large hdr%d scene%d", hdr, sc);
            Case t = {nm, 2175, 2751, 3, 3, hdr, 1, SC_PAN, sc ? SC_DISCS : SC_MIX, {{1, 1}, {1, 2}, {2, 1}, {3, 3}, {1, 1}}, 5, 1, 0, 0};
            total += run_case(&t, g_fmt);
        }
        printf("reported cases: %ld non-finite values\n", total);
        EXPECT(total == 0, "the reported failures still produce non-finite output");
        // sweep: tiny sizes, extreme aspect ratios, size changes between calls (output sizes small enough to run quickly)
        static const int rs[][2] = {{1, 1}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, {2, 2}, {3, 3}, {1, 64}, {64, 1}, {3, 200}, {200, 3}, {1, 511}, {2, 97}, {97, 3}, {5, 5}};
        static const int outs[][2] = {{1, 1}, {2, 3}, {64, 48}, {257, 33}, {33, 257}, {400, 300}};
        /* a render size never exceeds the output size (sr.test_scenes_r6: create rejects a larger maximum) */
        #define RW(i) (rs[i][0] < Wo ? rs[i][0] : Wo)
        #define RH(i) (rs[i][1] < Ho ? rs[i][1] : Ho)
        for (size_t o = 0; o < sizeof outs / sizeof *outs; o++) {
            const int Wo = outs[o][0], Ho = outs[o][1];
            for (int hdr = 0; hdr < 2; hdr++) {
                // one engine per (output, dynamic range): every pair of render sizes that fits, in sequence
                int Wm = 0, Hm = 0;
                for (size_t i = 0; i < sizeof rs / sizeof *rs; i++) { if (RW(i) > Wm) Wm = RW(i); if (RH(i) > Hm) Hm = RH(i); }
                for (int mvr = 0; mvr < 2; mvr++) {
                    char nm[96]; snprintf(nm, sizeof nm, "sweep %dx%d hdr%d mvr%d", Wo, Ho, hdr, mvr);
                    Case c = {nm, Wo, Ho, Wm, Hm, hdr, mvr, SC_PAN, hdr ? SC_DISCS : SC_MIX, {{0}}, 0, 0, 0, 0};
                    // a shuffled walk through all render sizes on one engine, so that every size follows several others
                    int order[sizeof rs / sizeof *rs]; int n = (int)(sizeof rs / sizeof *rs);
                    for (int i = 0; i < n; i++) order[i] = (int)((i * 7 + (int)o * 3 + hdr + mvr) % n);
                    for (int k = 0; k < 8; k++) { c.sizes[k][0] = RW(order[k]); c.sizes[k][1] = RH(order[k]); }
                    c.n = 8; c.resets = 1;
                    swept += run_case(&c, g_fmt);
                    // and the reverse order
                    for (int k = 0; k < 8; k++) { c.sizes[k][0] = RW(order[n - 1 - k]); c.sizes[k][1] = RH(order[n - 1 - k]); }
                    swept += run_case(&c, g_fmt);
                    ncases += 2;
                }
            }
        }
        }
        printf("sweep: %d engines, %ld non-finite values\n", ncases, swept);
        EXPECT(swept == 0, "the sweep produced non-finite output");
        g_fmt = getenv("SR_FMT") && !strcmp(getenv("SR_FMT"), "rg11b10") ? MTLPixelFormatRG11B10Float : MTLPixelFormatRGBA16Float;
        if (getenv("SR_BRIGHT")) {   // bright HDR content with a large pre-exposure: output values far above the converted formats' range
            Case c = {"bright", 64, 48, 64, 48, 1, 1, SC_PAN, SC_DISCS, {{64, 48}, {64, 48}, {40, 30}}, 3, 1, 0, 50.0f};
            long x = run_case(&c, g_fmt);
            printf("bright: %ld non-finite values, largest finite value %g\n", x, (double)g_maxv);
        }
        if (getenv("SR_FUZZ_ORACLE")) {   // extreme random inputs on the two reported geometries
            int n = atoi(getenv("SR_FUZZ_ORACLE")); long fz = 0;
            for (int it = 0; it < n; it++) {
                static const double scales[] = {0.25, 1.0, 4.0, 16.0, 64.0, 300.0};
                write_weights("build/fin_w.bin", WMODE_RANDOM, scales[(int)(urand() * 6)]);
                int hdr = urand() < 0.7, mvr = urand() < 0.8;
                char nm[64]; snprintf(nm, sizeof nm, "fuzz-oracle %d", it);
                Case a = {nm, 2175, 2751, 109, 1941, hdr, mvr, 0, 0, {{28, 821}, {1, 511}, {69, 986}, {96, 960}}, 4, urand() < 0.5 ? 5 : 1, 1, urand() < 0.3 ? (float)pow(10, 6 * (urand() - 0.5)) : 1.0f};
                Case b = {nm, 663, 2354, 7, 1181, hdr, mvr, 0, 0, {{3, 632}, {3, 677}, {4, 622}, {7, 316}}, 4, a.resets, 1, a.pre};
                fz += run_case(&a, g_fmt) + run_case(&b, g_fmt);
            }
            printf("fuzz-oracle: %d iterations, %ld non-finite values\n", n, fz);
            EXPECT(fz == 0, "the oracle-geometry fuzz produced non-finite output");
        }
        if (getenv("SR_FUZZ")) {
            int n = atoi(getenv("SR_FUZZ")); long fz = 0; int fails_before = 0;
            for (int it = 0; it < n; it++) {
                static const double scales[] = {0.25, 1.0, 4.0, 16.0, 64.0, 300.0};
                write_weights("build/fin_w.bin", WMODE_RANDOM, scales[(int)(urand() * 6)]);
                int Wo = 1 + (int)(urand() * urand() * 700), Ho = 1 + (int)(urand() * urand() * 700);
                Case c = {"fuzz", Wo, Ho, 0, 0, urand() < 0.7, urand() < 0.7, 0, 0, {{0}}, 4, urand() < 0.5 ? 5 : 1, 1, urand() < 0.3 ? (float)pow(10, 6 * (urand() - 0.5)) : 1.0f};
                int mw = 1, mh = 1;
                for (int k = 0; k < 4; k++) {
                    int w = urand() < 0.3 ? 1 + (int)(urand() * 3) : 1 + (int)(urand() * (Wo < 400 ? Wo : 400)), h = urand() < 0.3 ? 1 + (int)(urand() * 3) : 1 + (int)(urand() * (Ho < 400 ? Ho : 400));
                    c.sizes[k][0] = w; c.sizes[k][1] = h; if (w > mw) mw = w; if (h > mh) mh = h;
                }
                c.Wmax = mw; c.Hmax = mh;
                char nm[64]; snprintf(nm, sizeof nm, "fuzz %d (%dx%d)", it, Wo, Ho); c.name = nm;
                long x = run_case(&c, g_fmt);
                if (x) { printf("  case: out %dx%d max %dx%d hdr %d mvr %d resets %d pre %g sizes", Wo, Ho, mw, mh, c.hdr, c.mvr, c.resets, (double)c.pre); for (int k = 0; k < 4; k++) printf(" %dx%d", c.sizes[k][0], c.sizes[k][1]); printf("\n"); }
                fz += x; (void)fails_before;
            }
            printf("fuzz: %d cases, %ld non-finite values\n", n, fz);
            EXPECT(fz == 0, "the fuzz produced non-finite output");
        }
        printf(g_fails ? "finite_gpu: %d FAILURES\n" : "finite_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
