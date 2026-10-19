// Acceptance of sr.engine_api(+_r2) and the end-to-end properties of sr.frame_orchestration:
// formats, offsets, concurrency, errors, env-path, weights-tail, reset, determinism, nan-poison.
#include "enghost.h"
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    MTLPixelFormat colF, depF, outF;
    int oc[2], od[2], om[2], oo[2];
    int Wo, Ho, Wr, Hr, hdr, mvr, nan_frames_from, nan_frames_to, auto_exp;
    const char *wfile;
    int wide_hint;
} Setup;

static MetalSREngine *make_engine(const Setup *s, char *err, size_t cap) {
    MetalSRCreateInfo ci = {METALSR_INTERFACE_VERSION, (uint32_t)s->Wr, (uint32_t)s->Hr, (uint32_t)s->Wo, (uint32_t)s->Ho,
                        (uint32_t)((s->hdr ? METALSR_FLAG_HDR : 0) | (s->mvr ? METALSR_FLAG_MV_RENDER_RES : 0) | (s->auto_exp ? METALSR_FLAG_AUTO_EXPOSURE : 0)),
                        s->wfile ? s->wfile : "build/e2e_w.bin", NULL, NULL, 0, 0};
    return metalsr_create((__bridge void *)dev, &ci, err, cap);
}

static uint32_t pack11(f16 h, int mant) { return mant == 6 ? ((bits16(h) >> 4) & 0x7FF) : ((bits16(h) >> 5) & 0x3FF); }
static f16 unpack_small(uint32_t v, int mant) { return from_bits16((uint16_t)(mant == 6 ? (v << 4) : (v << 5))); }

/* quantise to what every colour format under test represents exactly (5 mantissa bits, positive) */
static float q5(float v) { f16 h = (f16)v; uint16_t b = bits16(h) & 0x7FE0; return (float)from_bits16(b); }

typedef struct { int unorm_levels; } ColOpts;

/* build a colour texture of the given format from float rgba (alpha forced to 1) */
static id<MTLTexture> make_colour(MTLPixelFormat F, int W, int H, int ox, int oy, const float *rgba, int rw, int rh) {
    int TW = rw + ox + 7, TH = rh + oy + 5;
    id<MTLTexture> t = tex2d(F, TW, TH, MTLTextureUsageShaderRead);
    size_t bpp = F == MTLPixelFormatRGBA16Float ? 8 : 4;
    uint8_t *buf = calloc((size_t)TW * TH, bpp);
    for (int j = 0; j < rh; j++) for (int i = 0; i < rw; i++) {
        const float *c = rgba + ((size_t)j * rw + i) * 4;
        uint8_t *d = buf + ((size_t)(j + oy) * TW + (i + ox)) * bpp;
        if (F == MTLPixelFormatRGBA16Float) { f16 h[4] = {(f16)c[0], (f16)c[1], (f16)c[2], (f16)1.0f}; memcpy(d, h, 8); }
        else if (F == MTLPixelFormatRG11B10Float) { uint32_t v = pack11((f16)c[0], 6) | (pack11((f16)c[1], 6) << 11) | (pack11((f16)c[2], 5) << 22); memcpy(d, &v, 4); }
        else if (F == MTLPixelFormatBGRA8Unorm) { uint8_t b4[4] = {(uint8_t)lrintf(c[2] * 255), (uint8_t)lrintf(c[1] * 255), (uint8_t)lrintf(c[0] * 255), 255}; memcpy(d, b4, 4); }
        else if (F == MTLPixelFormatRGB10A2Unorm) { uint32_t v = (uint32_t)lrintf(c[0] * 1023) | ((uint32_t)lrintf(c[1] * 1023) << 10) | ((uint32_t)lrintf(c[2] * 1023) << 20) | (3u << 30); memcpy(d, &v, 4); }
    }
    put(t, 0, 0, TW, TH, buf, (int)bpp);
    free(buf); (void)W; (void)H;
    return t;
}
static id<MTLTexture> make_depth(MTLPixelFormat F, int ox, int oy, const float *d, int rw, int rh) {
    int TW = rw + ox + 6, TH = rh + oy + 4;
    id<MTLTexture> t = tex2d(F, TW, TH, MTLTextureUsageShaderRead);
    if (F == MTLPixelFormatR32Float || F == MTLPixelFormatDepth32Float) {
        float *buf = calloc((size_t)TW * TH, 4);
        for (int j = 0; j < rh; j++) for (int i = 0; i < rw; i++) buf[(size_t)(j + oy) * TW + i + ox] = d[(size_t)j * rw + i];
        put(t, 0, 0, TW, TH, buf, 4); free(buf);
    } else {   /* Depth16Unorm */
        uint16_t *buf = calloc((size_t)TW * TH, 2);
        for (int j = 0; j < rh; j++) for (int i = 0; i < rw; i++) buf[(size_t)(j + oy) * TW + i + ox] = (uint16_t)lrintf(d[(size_t)j * rw + i] * 65535);
        put(t, 0, 0, TW, TH, buf, 2); free(buf);
    }
    return t;
}
static id<MTLTexture> make_motion(int ox, int oy, const float *m, int mw, int mh) {
    int TW = mw + ox + 9, TH = mh + oy + 3;
    id<MTLTexture> t = tex2d(MTLPixelFormatRG16Float, TW, TH, MTLTextureUsageShaderRead);
    f16 *buf = calloc((size_t)TW * TH, 4);
    for (int j = 0; j < mh; j++) for (int i = 0; i < mw; i++) { buf[((size_t)(j + oy) * TW + i + ox) * 2] = (f16)m[((size_t)j * mw + i) * 2]; buf[((size_t)(j + oy) * TW + i + ox) * 2 + 1] = (f16)m[((size_t)j * mw + i) * 2 + 1]; }
    put(t, 0, 0, TW, TH, buf, 4); free(buf);
    return t;
}

/* decode the rect of an output texture to f16 rgba */
static void read_output(id<MTLTexture> t, int ox, int oy, int W, int H, f16 *out) {
    MTLPixelFormat F = t.pixelFormat;
    size_t bpp = F == MTLPixelFormatRGBA16Float ? 8 : (F == MTLPixelFormatRGBA32Float ? 16 : 4);
    int TW = (int)t.width, TH = (int)t.height;
    uint8_t *buf = malloc((size_t)TW * TH * bpp);
    [t getBytes:buf bytesPerRow:TW * bpp fromRegion:MTLRegionMake2D(0, 0, TW, TH) mipmapLevel:0];
    for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
        const uint8_t *s = buf + ((size_t)(j + oy) * TW + i + ox) * bpp;
        f16 *d = out + ((size_t)j * W + i) * 4;
        if (F == MTLPixelFormatRGBA16Float) memcpy(d, s, 8);
        else if (F == MTLPixelFormatRGBA32Float) { const float *f = (const float *)s; for (int k = 0; k < 4; k++) d[k] = (f16)f[k]; }
        else if (F == MTLPixelFormatRGBA8Unorm) { for (int k = 0; k < 4; k++) d[k] = (f16)(s[k] / 255.0f); }
        else if (F == MTLPixelFormatBGRA8Unorm) { d[0] = (f16)(s[2] / 255.0f); d[1] = (f16)(s[1] / 255.0f); d[2] = (f16)(s[0] / 255.0f); d[3] = (f16)(s[3] / 255.0f); }
        else if (F == MTLPixelFormatRGB10A2Unorm) { uint32_t v; memcpy(&v, s, 4); d[0] = (f16)((v & 1023) / 1023.0f); d[1] = (f16)(((v >> 10) & 1023) / 1023.0f); d[2] = (f16)(((v >> 20) & 1023) / 1023.0f); d[3] = (f16)((v >> 30) / 3.0f); }
        else if (F == MTLPixelFormatRG11B10Float) { uint32_t v; memcpy(&v, s, 4); d[0] = unpack_small(v & 0x7FF, 6); d[1] = unpack_small((v >> 11) & 0x7FF, 6); d[2] = unpack_small((v >> 22) & 0x3FF, 5); d[3] = (f16)1.0f; }
    }
    free(buf);
}

/* run `nframes` frames of pan(mix) through an engine; fills the last output. `engine` may be provided (else created/destroyed). */
static int run_frames(const Setup *s, MetalSREngine *engine, int nframes, f16 *last_out, int reset_frame, int poison) {
    MetalSREngine *e = engine;
    char err[512];
    if (!e) { e = make_engine(s, err, sizeof err); if (!e) { printf("  create failed: %s\n", err); return 1; } }
    id<MTLTexture> otex = tex2d(s->outF, s->Wo + s->oo[0] + 11, s->Ho + s->oo[1] + 7, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    int Wm = s->mvr ? s->Wr : s->Wo, Hm = s->mvr ? s->Hr : s->Ho;
    int rc = 0;
    for (int f = 0; f < nframes; f++) {
        float jx = (float)halton(f % 16, 2) - 0.5f, jy = (float)halton(f % 16, 3) - 0.5f;
        float *col = malloc((size_t)s->Wr * s->Hr * 16), *dep = malloc((size_t)s->Wr * s->Hr * 4), *mot = malloc((size_t)Wm * Hm * 8);
        scene_render(SC_PAN, SC_MIX, f, s->Wo, s->Ho, s->Wr, s->Hr, s->mvr, 0, jx, jy, col, dep, mot);
        for (size_t i = 0; i < (size_t)s->Wr * s->Hr * 4; i++) {
            if (s->colF == MTLPixelFormatBGRA8Unorm || s->colF == MTLPixelFormatRGB10A2Unorm) col[i] = fminf(col[i], 1.0f);
            else col[i] = q5(col[i]);
        }
        for (size_t i = 0; i < (size_t)s->Wr * s->Hr; i++) { dep[i] = s->depF == MTLPixelFormatDepth16Unorm ? (float)lrintf(dep[i] * 65535) / 65535.0f : dep[i]; }
        if (poison && f >= s->nan_frames_from && f <= s->nan_frames_to) {
            for (int k = 0; k < 64; k++) { size_t i = ((size_t)(37 * k) % s->Wr + (size_t)((23 * k) % s->Hr) * s->Wr) * 4; for (int a = 0; a < 3; a++) col[i + a] = NAN;
                                           size_t j = ((size_t)(53 * k) % Wm + (size_t)((29 * k) % Hm) * Wm) * 2; mot[j] = mot[j + 1] = NAN; }
        }
        id<MTLTexture> tc = make_colour(s->colF, 0, 0, s->oc[0], s->oc[1], col, s->Wr, s->Hr);
        id<MTLTexture> td = make_depth(s->depF, s->od[0], s->od[1], dep, s->Wr, s->Hr);
        id<MTLTexture> tm_ = make_motion(s->om[0], s->om[1], mot, Wm, Hm);
        MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
        fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tm_; fi.output = (__bridge void *)otex;
        fi.rw = s->Wr; fi.rh = s->Hr; fi.jitter_x = jx; fi.jitter_y = jy; fi.mv_scale_x = 1; fi.mv_scale_y = 1; fi.pre_exposure = 1; fi.reset = f == reset_frame;
        memcpy(fi.color_origin, s->oc, 8); memcpy(fi.depth_origin, s->od, 8); memcpy(fi.motion_origin, s->om, 8); memcpy(fi.output_origin, s->oo, 8);
        rc = host_call(e, &fi);
        if (rc) { printf("  encode failed: %s\n", metalsr_last_error(e)); free(col); free(dep); free(mot); break; }
        free(col); free(dep); free(mot);
    }
    if (!rc) read_output(otex, s->oo[0], s->oo[1], s->Wo, s->Ho, last_out);
    if (!engine) metalsr_destroy(e);
    return rc;
}

static Setup base_setup(void) {
    Setup s; memset(&s, 0, sizeof s);
    s.colF = MTLPixelFormatRGBA16Float; s.depF = MTLPixelFormatR32Float; s.outF = MTLPixelFormatRGBA16Float;
    s.Wo = 200; s.Ho = 120; s.Wr = 134; s.Hr = 80; s.hdr = 1; s.mvr = 1;
    return s;
}
/* PSNR in the comparison domain of sr.test_scenes_r2: D(c) = c' / (1 + max channel of c'); peak 1 */
static double psnr16(const f16 *a, const f16 *b, size_t npix) {
    double se = 0;
    for (size_t i = 0; i < npix; i++) {
        float ca[3], cb[3];
        for (int k = 0; k < 3; k++) { ca[k] = (float)a[4 * i + k]; cb[k] = (float)b[4 * i + k]; if (!isfinite(ca[k])) ca[k] = 1e9f; }
        double ma = 0, mb = 0;
        for (int k = 0; k < 3; k++) { if (ca[k] > ma) ma = ca[k]; if (cb[k] > mb) mb = cb[k]; }
        for (int k = 0; k < 3; k++) { double d = (ca[k] > 0 ? ca[k] : 0) / (1 + ma) - (cb[k] > 0 ? cb[k] : 0) / (1 + mb); se += d * d; }
    }
    double mse = se / (npix * 3);
    return mse <= 0 ? 999 : 10 * log10(1.0 / mse);
}
static int bit_equal(const f16 *a, const f16 *b, size_t n) { return memcmp(a, b, n * 2) == 0; }

static void test_formats(void) {
    printf("formats\n");
    Setup s = base_setup();
    size_t n = (size_t)s.Wo * s.Ho;
    f16 *ref = malloc(n * 8), *got = malloc(n * 8);
    if (run_frames(&s, NULL, 3, ref, -1, 0)) { g_fails++; return; }
    /* colour: RG11B10Float is exact; unorm formats are compared to an RGBA16F upload of the same unorm values */
    s.colF = MTLPixelFormatRG11B10Float;
    if (run_frames(&s, NULL, 3, got, -1, 0)) { g_fails++; return; }
    EXPECT(bit_equal(ref, got, n * 4), "RG11B10Float colour must be bit-identical to RGBA16Float");
    printf("  RG11B10Float colour: %s\n", bit_equal(ref, got, n * 4) ? "bit-identical" : "DIFFERS");
    /* depth */
    s = base_setup(); s.depF = MTLPixelFormatDepth32Float;
    if (run_frames(&s, NULL, 3, got, -1, 0)) { g_fails++; return; }
    EXPECT(bit_equal(ref, got, n * 4), "Depth32Float must be bit-identical to R32Float");
    printf("  Depth32Float depth: %s\n", bit_equal(ref, got, n * 4) ? "bit-identical" : "DIFFERS");
    Setup s16 = base_setup();
    s16.depF = MTLPixelFormatDepth16Unorm;
    Setup s16ref = base_setup();
    f16 *r16 = malloc(n * 8), *g16 = malloc(n * 8);
    /* reference for Depth16Unorm: R32Float holding the same quantised values (run_frames quantises the data) */
    if (run_frames(&s16ref, NULL, 3, r16, -1, 0) || run_frames(&s16, NULL, 3, g16, -1, 0)) { g_fails++; return; }
    printf("  Depth16Unorm depth: PSNR vs R32Float upload of the full-precision data %.1f dB\n", psnr16(g16, r16, n));
    EXPECT(psnr16(g16, r16, n) > 40, "Depth16Unorm");
    /* unorm colour */
    for (int k = 0; k < 2; k++) {
        Setup su = base_setup(); su.colF = k ? MTLPixelFormatRGB10A2Unorm : MTLPixelFormatBGRA8Unorm;
        if (run_frames(&su, NULL, 3, got, -1, 0)) { g_fails++; return; }
        double ps = psnr16(got, ref, n);
        printf("  %s colour: PSNR vs RGBA16Float upload %.1f dB\n", k ? "RGB10A2Unorm" : "BGRA8Unorm", ps);
        EXPECT(ps > 30, "unorm colour (%.1f dB)", ps);
    }
    /* output formats through the conversion pass */
    MTLPixelFormat of[] = {MTLPixelFormatRGBA32Float, MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGB10A2Unorm, MTLPixelFormatRG11B10Float};
    for (int k = 0; k < 5; k++) {
        Setup so = base_setup(); so.outF = of[k];
        if (run_frames(&so, NULL, 3, got, -1, 0)) { g_fails++; return; }
        double ps = psnr16(got, ref, n);
        printf("  output format %d: PSNR vs RGBA16Float output %.1f dB\n", (int)of[k], ps);
        EXPECT(ps > (k == 0 ? 90 : 35), "output format %d (%.1f dB)", (int)of[k], ps);
    }
    free(ref); free(got); free(r16); free(g16);
}

static void test_offsets(void) {
    printf("offsets\n");
    Setup s = base_setup();
    size_t n = (size_t)s.Wo * s.Ho;
    f16 *ref = malloc(n * 8), *got = malloc(n * 8);
    if (run_frames(&s, NULL, 3, ref, -1, 0)) { g_fails++; return; }
    Setup so = s; so.oc[0] = 3; so.oc[1] = 5; so.om[0] = 3; so.om[1] = 5; so.od[0] = 3; so.od[1] = 5; so.oo[0] = 3; so.oo[1] = 5;
    if (run_frames(&so, NULL, 3, got, -1, 0)) { g_fails++; return; }
    printf("  colour/depth/motion/output at (3,5): %s\n", bit_equal(ref, got, n * 4) ? "bit-identical" : "DIFFERS");
    EXPECT(bit_equal(ref, got, n * 4), "offsets");
    Setup sd = s; sd.depF = MTLPixelFormatDepth32Float; sd.od[0] = 3; sd.od[1] = 5;
    if (run_frames(&sd, NULL, 3, got, -1, 0)) { g_fails++; return; }
    printf("  Depth32Float depth at (3,5): %s\n", bit_equal(ref, got, n * 4) ? "bit-identical" : "DIFFERS");
    EXPECT(bit_equal(ref, got, n * 4), "depth origin");
    Setup sm = s; sm.mvr = 0; Setup smo = sm; smo.om[0] = 3; smo.om[1] = 5;
    f16 *r2 = malloc(n * 8);
    if (run_frames(&sm, NULL, 3, r2, -1, 0) || run_frames(&smo, NULL, 3, got, -1, 0)) { g_fails++; return; }
    printf("  output-resolution motion at (3,5): %s\n", bit_equal(r2, got, n * 4) ? "bit-identical" : "DIFFERS");
    EXPECT(bit_equal(r2, got, n * 4), "motion origin (output resolution)");
    free(ref); free(got); free(r2);
}

static void test_reset_determinism(void) {
    printf("reset / determinism / nan-poison\n");
    Setup s = base_setup();
    size_t n = (size_t)s.Wo * s.Ho;
    f16 *a = malloc(n * 8), *b = malloc(n * 8), *c = malloc(n * 8);
    if (run_frames(&s, NULL, 5, a, -1, 0) || run_frames(&s, NULL, 5, b, -1, 0)) { g_fails++; return; }
    printf("  determinism (two engines, 5 frames): %s\n", bit_equal(a, b, n * 4) ? "bit-identical" : "DIFFERS");
    EXPECT(bit_equal(a, b, n * 4), "determinism");
    /* frame 4 with reset after 4 frames of history == fresh engine whose first call is frame 4 of the sequence: emulate with a
       sequence of identical length where only the last frame is compared: a reset on the last call must not depend on history */
    if (run_frames(&s, NULL, 5, a, 4, 0)) { g_fails++; return; }
    {   /* fresh engine seeing frame 4 first: run frames 0..4 but with reset at every frame -> last output independent of history */
        Setup s2 = s;
        MetalSREngine *e1 = make_engine(&s2, (char[512]){0}, 512), *e2 = make_engine(&s2, (char[512]){0}, 512);
        f16 *o1 = malloc(n * 8), *o2 = malloc(n * 8);
        run_frames(&s2, e1, 4, o1, -1, 0);     /* history built up on e1 */
        run_frames(&s2, e1, 1, o1, 0, 0);      /* then a reset call on frame index 0 of the second run */
        run_frames(&s2, e2, 1, o2, 0, 0);      /* fresh engine, same frame */
        printf("  reset call vs fresh engine: %s\n", bit_equal(o1, o2, n * 4) ? "bit-identical" : "DIFFERS");
        EXPECT(bit_equal(o1, o2, n * 4), "reset equals fresh engine");
        metalsr_destroy(e1); metalsr_destroy(e2); free(o1); free(o2);
    }
    /* nan-poison: no non-finite output pixel on frames 5-7 */
    Setup sp = base_setup(); sp.nan_frames_from = 5; sp.nan_frames_to = 7;
    MetalSREngine *e = make_engine(&sp, (char[512]){0}, 512);
    int bad = 0;
    for (int f = 0; f < 3; f++) { run_frames(&sp, e, 1, c, -1, 1); for (size_t i = 0; i < n * 4; i++) if (!isfinite((float)c[i])) bad++; }
    /* poison applies when the frame index lies in the window; run_frames restarts the scene index, so force it */
    Setup sq = base_setup(); sq.nan_frames_from = 0; sq.nan_frames_to = 0;
    for (int f = 0; f < 3; f++) { run_frames(&sq, e, 1, c, -1, 1); for (size_t i = 0; i < n * 4; i++) if (!isfinite((float)c[i])) bad++; }
    printf("  nan-poison: %d non-finite output values\n", bad);
    EXPECT(bad == 0, "nan-poison");
    metalsr_destroy(e);
    free(a); free(b); free(c);
}

/* ae-schedule: E_now changes only on calls 0, 2, 4; E_prev of call n equals E_now of call n - 1 */
static void test_ae_schedule(void) {
    printf("ae-schedule\n");
    Setup s = base_setup(); s.auto_exp = 1;
    MetalSREngine *e = make_engine(&s, (char[512]){0}, 512);
    id<MTLTexture> otex = tex2d(MTLPixelFormatRGBA16Float, s.Wo, s.Ho, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    static const float gain[6] = {1, 4, 2, 8, 0.5f, 3};
    float prev_now = 0, now = 0, prev = 0;
    int bad = 0;
    for (int f = 0; f < 6; f++) {
        float *col = malloc((size_t)s.Wr * s.Hr * 16), *dep = malloc((size_t)s.Wr * s.Hr * 4), *mot = malloc((size_t)s.Wr * s.Hr * 8);
        scene_render(SC_PAN, SC_MIX, f, s.Wo, s.Ho, s.Wr, s.Hr, 1, 0, 0, 0, col, dep, mot);
        for (size_t i = 0; i < (size_t)s.Wr * s.Hr; i++) for (int a = 0; a < 3; a++) col[4 * i + a] *= gain[f];
        id<MTLTexture> tc = make_colour(MTLPixelFormatRGBA16Float, 0, 0, 0, 0, col, s.Wr, s.Hr), td = make_depth(MTLPixelFormatR32Float, 0, 0, dep, s.Wr, s.Hr), tm_ = make_motion(0, 0, mot, s.Wr, s.Hr);
        MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
        fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tm_; fi.output = (__bridge void *)otex;
        fi.rw = s.Wr; fi.rh = s.Hr; fi.mv_scale_x = fi.mv_scale_y = 1; fi.pre_exposure = 1;
        if (host_call(e, &fi)) { printf("  encode failed\n"); g_fails++; break; }
        metalsr_debug_exposure(e, &now, &prev);
        int changed = f == 0 || now != prev_now;
        printf("  call %d: E_now %.5f E_prev %.5f%s\n", f, now, prev, changed ? " (changed)" : "");
        if (prev != now) bad++;                                  // P13: E_prev = E_now at the end of every call
        if (f % 2 == 1 && now != prev_now) bad++;                // odd calls keep E_now
        if (f % 2 == 0 && f > 0 && now == prev_now) bad++;       // even calls recompute from a different colour
        if (f == 0 && !(now > 0)) bad++;
        prev_now = now;
        free(col); free(dep); free(mot);
    }
    EXPECT(bad == 0, "ae-schedule (%d violations)", bad);
    metalsr_destroy(e);
}

/* ---- errors ---- */
static void test_errors(void) {
    printf("errors / env-path / weights-tail\n");
    Setup s = base_setup();
    char err[512];
    MetalSRCreateInfo ci = {METALSR_INTERFACE_VERSION, 134, 80, 200, 120, METALSR_FLAG_HDR | METALSR_FLAG_MV_RENDER_RES, "build/e2e_w.bin", NULL, NULL, 0, 0};
    MetalSREngine *e;
    #define EXPECT_FAIL(cond_info, what) do { err[0] = 0; e = metalsr_create((__bridge void *)dev, &(cond_info), err, sizeof err); \
        printf("  %-28s -> %s\n", what, e ? "CREATED (unexpected)" : err); EXPECT(!e && err[0], "%s must fail with a message", what); if (e) metalsr_destroy(e); } while (0)
    MetalSRCreateInfo b = ci; b.version = 1; EXPECT_FAIL(b, "wrong version");
    b = ci; b.max_rw = 0; EXPECT_FAIL(b, "zero size");
    /* size probes of sr.test_scenes_r6 through the real create call: null and a message naming the field */
    #define EXPECT_FIELD(cond_info, what, field) do { EXPECT_FAIL(cond_info, what); EXPECT(strstr(err, field) != NULL, "%s names %s: '%s'", what, field, err); } while (0)
    b = ci; b.out_w = 16385; b.max_rw = 16385; b.out_h = 1024; b.max_rh = 1024; EXPECT_FIELD(b, "x-w-over", "out_w");
    b = ci; b.out_w = 1024; b.max_rw = 1024; b.out_h = 16385; b.max_rh = 16385; EXPECT_FIELD(b, "x-h-over", "out_h");
    b = ci; b.out_w = 1920; b.out_h = 1080; b.max_rw = 1921; b.max_rh = 1080; EXPECT_FIELD(b, "x-rw-over", "max_rw");
    b = ci; b.out_w = 1920; b.out_h = 1080; b.max_rw = 1920; b.max_rh = 1081; EXPECT_FIELD(b, "x-rh-over", "max_rh");
    b = ci; b.out_w = 0; b.out_h = 1080; EXPECT_FIELD(b, "x-zero", "out_w");
    b = ci; b.weight_path = "build/no_such_file.bin"; EXPECT_FAIL(b, "missing weight file");
    b = ci; b.test_hooks = "x"; EXPECT_FAIL(b, "testing hook string");
    err[0] = 0; e = metalsr_create(NULL, &ci, err, sizeof err); EXPECT(!e && err[0], "null device"); printf("  %-28s -> %s\n", "null device", err);
    /* wrong size / tail */
    { FILE *f = fopen("build/e2e_w.bin", "rb"); uint8_t *img = malloc(SR_WEIGHT_FILE_BYTES); fread(img, 1, SR_WEIGHT_FILE_BYTES, f); fclose(f);
      f = fopen("build/e2e_short.bin", "wb"); fwrite(img, 1, SR_WEIGHT_FILE_BYTES - 1, f); fclose(f);
      FILE *p = fopen("build/e2e_w.bin.params", "rb"); uint8_t pb[1024]; size_t pn = fread(pb, 1, sizeof pb, p); fclose(p);
      p = fopen("build/e2e_short.bin.params", "wb"); fwrite(pb, 1, pn, p); fclose(p);
      b = ci; b.weight_path = "build/e2e_short.bin"; EXPECT_FAIL(b, "weight file one byte short");
      for (size_t i = 1903696; i < SR_WEIGHT_FILE_BYTES; i++) img[i] = 0xA5;   /* padding bytes are ignored */
      f = fopen("build/e2e_pad.bin", "wb"); fwrite(img, 1, SR_WEIGHT_FILE_BYTES, f); fclose(f);
      p = fopen("build/e2e_pad.bin.params", "wb"); fwrite(pb, 1, pn, p); fclose(p);
      size_t n = (size_t)s.Wo * s.Ho; f16 *o1 = malloc(n * 8), *o2 = malloc(n * 8);
      run_frames(&s, NULL, 2, o1, -1, 0);
      Setup sp = s; sp.wfile = "build/e2e_pad.bin"; run_frames(&sp, NULL, 2, o2, -1, 0);
      printf("  weights-tail (non-zero padding): %s\n", bit_equal(o1, o2, n * 4) ? "same output" : "DIFFERS");
      EXPECT(bit_equal(o1, o2, n * 4), "weights-tail");
      free(o1); free(o2); free(img); }
    /* env-path / env-ignored */
    b = ci; b.weight_path = NULL;
    unsetenv("METALSR_WEIGHTS"); EXPECT_FAIL(b, "path null, variable unset");
    setenv("METALSR_WEIGHTS", "", 1); EXPECT_FAIL(b, "path null, variable empty");
    setenv("METALSR_WEIGHTS", "build/no_such_file.bin", 1); EXPECT_FAIL(b, "path null, variable -> missing file");
    setenv("METALSR_WEIGHTS", "build/e2e_w.bin", 1);
    e = metalsr_create((__bridge void *)dev, &b, err, sizeof err); EXPECT(e != NULL, "env-path: variable names a valid file (%s)", err); printf("  env-path valid: %s\n", e ? "created" : err); if (e) metalsr_destroy(e);
    setenv("METALSR_WEIGHTS", "build/no_such_file.bin", 1);
    e = metalsr_create((__bridge void *)dev, &ci, err, sizeof err); EXPECT(e != NULL, "env-ignored: explicit path wins (%s)", err); printf("  env-ignored: %s\n", e ? "created" : err);
    unsetenv("METALSR_WEIGHTS");
    /* encode errors */
    e = metalsr_create((__bridge void *)dev, &ci, err, sizeof err);
    id<MTLTexture> tc = tex2d(MTLPixelFormatRGBA16Float, 134, 80, MTLTextureUsageShaderRead), td = tex2d(MTLPixelFormatR32Float, 134, 80, MTLTextureUsageShaderRead),
                   tmv = tex2d(MTLPixelFormatRG16Float, 134, 80, MTLTextureUsageShaderRead);
    id<MTLTexture> to = tex2d(MTLPixelFormatRGBA16Float, 200, 120, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
    fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tmv; fi.output = (__bridge void *)to; fi.rw = 134; fi.rh = 80;
    fi.mv_scale_x = fi.mv_scale_y = 1; fi.pre_exposure = 1;
    #define EXPECT_ENC_FAIL(what) do { int rc = host_call(e, &fi); printf("  encode: %-30s -> rc %d: %s\n", what, rc, metalsr_last_error(e)); EXPECT(rc != 0, "%s must fail", what); } while (0)
    { id<MTLTexture> bad = tex2d(MTLPixelFormatR16Float, 134, 80, MTLTextureUsageShaderRead); fi.depth = (__bridge void *)bad; EXPECT_ENC_FAIL("unsupported depth format"); fi.depth = (__bridge void *)td; }
    { id<MTLTexture> bad = tex2d(MTLPixelFormatRG16Float, 200, 120, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite); fi.output = (__bridge void *)bad; EXPECT_ENC_FAIL("unsupported output format"); fi.output = (__bridge void *)to; }
    { id<MTLTexture> bad = tex2d(MTLPixelFormatRGBA16Float, 200, 120, MTLTextureUsageShaderRead); fi.output = (__bridge void *)bad; EXPECT_ENC_FAIL("output without shader-write"); fi.output = (__bridge void *)to; }
    { fi.rw = 200; EXPECT_ENC_FAIL("render size above the maximum"); fi.rw = 134; }
    { fi.color_origin[0] = 5; EXPECT_ENC_FAIL("colour rect outside its texture"); fi.color_origin[0] = 0; }
    { fi.output_origin[1] = 3; EXPECT_ENC_FAIL("output too small"); fi.output_origin[1] = 0; }
    { fi.color = NULL; EXPECT_ENC_FAIL("null colour texture"); fi.color = (__bridge void *)tc; }
    { int rc = host_call(e, &fi); EXPECT(rc == 0, "a valid call still works after the failures (%s)", metalsr_last_error(e)); }
    metalsr_destroy(e);
}

/* ---- concurrency ---- */
typedef struct { Setup s; MetalSREngine *engine; f16 *out; id<MTL4CommandQueue> q; int nframes; int rc; } Job;
static void *job_main(void *arg) {
    @autoreleasepool {
        Job *j = arg;
        /* own allocator and event per thread; the queue is shared or private */
        id<MTL4CommandAllocator> al = [dev newCommandAllocator];
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        uint64_t val = 0;
        Setup *s = &j->s;
        id<MTLTexture> otex = tex2d(s->outF, s->Wo, s->Ho, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
        int Wm = s->mvr ? s->Wr : s->Wo, Hm = s->mvr ? s->Hr : s->Ho;
        for (int f = 0; f < j->nframes; f++) {
            float jx = (float)halton(f % 16, 2) - 0.5f, jy = (float)halton(f % 16, 3) - 0.5f;
            float *col = malloc((size_t)s->Wr * s->Hr * 16), *dep = malloc((size_t)s->Wr * s->Hr * 4), *mot = malloc((size_t)Wm * Hm * 8);
            scene_render(SC_PAN, SC_MIX, f, s->Wo, s->Ho, s->Wr, s->Hr, s->mvr, 0, jx, jy, col, dep, mot);
            id<MTLTexture> tc = make_colour(MTLPixelFormatRGBA16Float, 0, 0, 0, 0, col, s->Wr, s->Hr), td = make_depth(MTLPixelFormatR32Float, 0, 0, dep, s->Wr, s->Hr), tm_ = make_motion(0, 0, mot, Wm, Hm);
            MetalSRFrameInfo fi; memset(&fi, 0, sizeof fi);
            fi.color = (__bridge void *)tc; fi.depth = (__bridge void *)td; fi.motion = (__bridge void *)tm_; fi.output = (__bridge void *)otex;
            fi.rw = s->Wr; fi.rh = s->Hr; fi.jitter_x = jx; fi.jitter_y = jy; fi.mv_scale_x = fi.mv_scale_y = 1; fi.pre_exposure = 1;
            j->rc |= host_call_on(j->q, al, ev, &val, j->engine, &fi);
            free(col); free(dep); free(mot);
        }
        read_output(otex, 0, 0, s->Wo, s->Ho, j->out);
    }
    return NULL;
}
static void *create_main(void *arg) { Job *j = arg; char err[256]; j->engine = make_engine(&j->s, err, sizeof err); return NULL; }

static void test_concurrency(int shared_queue) {
    printf("concurrency (%s queue)\n", shared_queue ? "one shared" : "separate");
    Setup s = base_setup(); s.Wo = 160; s.Ho = 96; s.Wr = 107; s.Hr = 64;
    size_t n = (size_t)s.Wo * s.Ho;
    f16 *seq[4]; for (int i = 0; i < 4; i++) seq[i] = malloc(n * 8);
    for (int i = 0; i < 4; i++) {   /* sequential references: engine i sees a shifted scene by using i extra frames of warm-up? keep identical input, compare per engine */
        Job j = {s, NULL, seq[i], queue, 4, 0};
        j.engine = make_engine(&s, (char[256]){0}, 256);
        job_main(&j); metalsr_destroy(j.engine);
    }
    Job jobs[4]; pthread_t th[4], cth;
    for (int i = 0; i < 4; i++) { jobs[i] = (Job){s, NULL, malloc(n * 8), shared_queue ? queue : [dev newMTL4CommandQueue], 4, 0}; }
    for (int i = 0; i < 3; i++) jobs[i].engine = make_engine(&s, (char[256]){0}, 256);
    pthread_create(&cth, NULL, create_main, &jobs[3]);          /* fourth engine created on a background thread meanwhile */
    for (int i = 0; i < 3; i++) pthread_create(&th[i], NULL, job_main, &jobs[i]);
    pthread_join(cth, NULL);
    pthread_create(&th[3], NULL, job_main, &jobs[3]);
    for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < 4; i++) {
        printf("  engine %d: %s\n", i, bit_equal(jobs[i].out, seq[0], n * 4) ? "bit-identical to the sequential run" : "DIFFERS");
        EXPECT(jobs[i].rc == 0 && bit_equal(jobs[i].out, seq[0], n * 4), "concurrency engine %d", i);
        metalsr_destroy(jobs[i].engine); free(jobs[i].out); free(seq[i]);
    }
}

int main(void) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newMTL4CommandQueue]; alloc = [dev newCommandAllocator]; evt = [dev newSharedEvent];
        g_mp = test_params();
        write_weights("build/e2e_w.bin", WMODE_RANDOM, 1.0);
        test_formats();
        test_offsets();
        test_reset_determinism();
        test_ae_schedule();
        test_errors();
        test_concurrency(1);
        test_concurrency(0);
        printf(g_fails ? "api_gpu: %d FAILURES\n" : "api_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
