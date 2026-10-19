/* Menu detection (spec fg.menu_detection section 7): md-count-edge, -thr-edge, -grid, -rect, -uicolour, -hold, -sticky, -reset,
   -hud-count, -hud-onechan, -hud-extreme, -hud-nonfinite; pass level with the detector state kept between calls. */
#import <Foundation/Foundation.h>
#include "texutil.h"
#include "menu.h"

static int fails;
static MetalFGParams P;
static NnRig R;
static MetalFGMenu *M;

typedef struct { id<MTLBuffer> state, flag; } Det;

static Det det_new(void) {
    Det d;
    d.state = [R.gpu.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    d.flag = [R.gpu.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(d.state.contents, 0, 16);
    memset(d.flag.contents, 0, 16);
    return d;
}

static uint32_t w0(Det d) { return ((uint32_t *)d.state.contents)[0]; }
static uint32_t wd(Det d, int i) { return ((uint32_t *)d.state.contents)[i]; }
static uint8_t byte(Det d) { return ((uint8_t *)d.flag.contents)[0]; }

static MetalFGDetectArgs dargs(uint32_t cw, uint32_t ch, uint32_t ox, uint32_t oy, float frac) {
    return (MetalFGDetectArgs){(uint32_t)P.gx, (uint32_t)P.gy, cw, ch, ox, oy, P.mt, frac};
}

static void run_alpha(Det d, id<MTLTexture> t, uint32_t channel, MetalFGDetectArgs a) {
    nn_run(&R, ^(MetalFGEnc *e) { [M detectAlpha:e state:d.state flag:d.flag coverage:t channel:channel args:a]; });
}

static uint32_t sx(uint32_t W, uint32_t i) { return (uint32_t)(((uint64_t)(W - 1) * i) / (P.gx - 1)); }
static uint32_t sy(uint32_t H, uint32_t j) { return (uint32_t)(((uint64_t)(H - 1) * j) / (P.gy - 1)); }

/* plane: 255 at the sample pixels with raster index m < K (K = NS: all), 0 elsewhere; origin (ox, oy) in a texture of size TW x TH */
static id<MTLTexture> plane(uint32_t TW, uint32_t TH, uint32_t cw, uint32_t ch, uint32_t ox, uint32_t oy, int K, int invert, MTLPixelFormat pf) {
    id<MTLTexture> t = gpu_tex(&R.gpu, pf, TW, TH);
    tex_set(t, ^(uint32_t x, uint32_t y, float v[4]) {
        int hit = 0;
        for (int j = 0; j < P.gy && !hit; j++)
            for (int i = 0; i < P.gx; i++)
                if (ox + sx(cw, i) == x && oy + sy(ch, j) == y && (int)(P.gx * j + i) < K) { hit = 1; break; }
        float val = invert ? (hit ? 0.0f : 1.0f) : (hit ? 1.0f : 0.0f);
        v[0] = v[1] = v[2] = v[3] = val;
    });
    return t;
}

static void count_edge(void) {
    int NS = P.gx * P.gy;
    int Khi = (int)floor(NS * (double)P.mfa) + 1, Klo = (int)floor(NS * (double)P.mfa);
    id<MTLTexture> zero = plane(640, 360, 640, 360, 0, 0, 0, 0, MTLPixelFormatR8Unorm);
    for (int which = 0; which < 2; which++) {
        Det d = det_new();
        id<MTLTexture> t = plane(640, 360, 640, 360, 0, 0, which ? Klo : Khi, 0, MTLPixelFormatR8Unorm);
        run_alpha(d, zero, 0, dargs(640, 360, 0, 0, P.mfa)); /* call 0 */
        run_alpha(d, t, 0, dargs(640, 360, 0, 0, P.mfa));
        uint32_t a = w0(d);
        run_alpha(d, zero, 0, dargs(640, 360, 0, 0, P.mfa));
        uint32_t b = w0(d);
        if (which == 0) CHECKC(a && b, "md-count-edge K_hi: W0 %u %u", a, b);
        else CHECKC(!a && !b, "md-count-edge K_lo: W0 %u %u (a rule cnt >= K_lo would differ)", a, b);
    }
}

static void thr_edge(void) {
    int lo = (int)floor(255 * (double)P.mt);
    for (int which = 0; which < 2; which++) {
        Det d = det_new();
        id<MTLTexture> t = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, 640, 360);
        float val = (which ? lo + 1 : lo) / 255.0f;
        tex_set(t, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = val; });
        run_alpha(d, t, 0, dargs(640, 360, 0, 0, P.mfa));
        CHECKC((wd(d, 1) != 0) == (which == 1), "md-thr-edge %s: D=%u", which ? "v_hi" : "v_lo", wd(d, 1));
    }
}

static void grid_rect_uicolour(void) {
    Det d = det_new();
    run_alpha(d, plane(640, 360, 640, 360, 0, 0, 1 << 30, 0, MTLPixelFormatR8Unorm), 0, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(wd(d, 1) == 1, "md-grid run H: D=%u", wd(d, 1));
    d = det_new();
    run_alpha(d, plane(640, 360, 640, 360, 0, 0, 1 << 30, 1, MTLPixelFormatR8Unorm), 0, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(wd(d, 1) == 0, "md-grid run M: D=%u", wd(d, 1));
    d = det_new();
    run_alpha(d, plane(704, 424, 640, 360, 5, 7, 1 << 30, 0, MTLPixelFormatR8Unorm), 0, dargs(640, 360, 5, 7, P.mfa));
    CHECKC(wd(d, 1) == 1, "md-rect: D=%u", wd(d, 1));
    d = det_new();
    run_alpha(d, plane(704, 424, 640, 360, 5, 7, 1 << 30, 0, MTLPixelFormatR8Unorm), 0, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(wd(d, 1) == 0, "md-rect control: ignoring the origin must fail to detect");
    d = det_new();
    run_alpha(d, plane(640, 360, 640, 360, 0, 0, 1 << 30, 0, MTLPixelFormatRGBA8Unorm), 3, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(wd(d, 1) == 1, "md-uicolour: D=%u", wd(d, 1));
}

static void hold_sticky_reset(void) {
    id<MTLTexture> zero = plane(640, 360, 640, 360, 0, 0, 0, 0, MTLPixelFormatR8Unorm), full = plane(640, 360, 640, 360, 0, 0, 1 << 30, 0, MTLPixelFormatR8Unorm);
    id<MTLTexture> seq[6] = {zero, full, zero, zero, full, zero};
    int want[6] = {0, 1, 1, 0, 1, 1};
    Det d = det_new();
    run_alpha(d, zero, 0, dargs(640, 360, 0, 0, P.mfa));
    for (int i = 0; i < 6; i++) {
        ((uint8_t *)d.flag.contents)[0] = 0;
        run_alpha(d, seq[i], 0, dargs(640, 360, 0, 0, P.mfa));
        CHECKC((w0(d) != 0) == want[i], "md-hold call %d: W0 %u", i + 1, w0(d));
        CHECKC(byte(d) == want[i], "md-hold call %d: byte %u", i + 1, byte(d));
        CHECKC(wd(d, 2) == 0 && wd(d, 3) == 0, "md-hold: W2, W3 zero");
    }
    d = det_new();
    run_alpha(d, zero, 0, dargs(640, 360, 0, 0, P.mfa));
    int sticky[6] = {0, 1, 1, 1, 1, 1};
    for (int i = 0; i < 6; i++) {
        run_alpha(d, seq[i], 0, dargs(640, 360, 0, 0, P.mfa));
        CHECKC(byte(d) == sticky[i], "md-sticky call %d: byte %u", i + 1, byte(d));
    }
    /* md-reset: call 0 carries the full plane; the engine's reset sequence is detector, then F01 */
    d = det_new();
    ((uint8_t *)d.flag.contents)[0] = 255;
    run_alpha(d, full, 0, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(byte(d) == 1 && wd(d, 1) == 1, "md-reset: detector sets the byte on its own");
    nn_run(&R, ^(MetalFGEnc *e) { [M clearFlag:e flag:d.flag]; });
    CHECKC(byte(d) == 0 && wd(d, 1) == 1, "md-reset: F01 clears the byte, W1 stays one");
    run_alpha(d, zero, 0, dargs(640, 360, 0, 0, P.mfa));
    CHECKC(w0(d) != 0 && byte(d) == 1, "md-reset: call 1 is held by the reset call's detection");
    /* F01 writes exactly one byte */
    ((uint8_t *)d.flag.contents)[1] = 0x7f;
    nn_run(&R, ^(MetalFGEnc *e) { [M clearFlag:e flag:d.flag]; });
    CHECKC(((uint8_t *)d.flag.contents)[1] == 0x7f, "F01 writes one byte, not a word");
}

static void hudless(void) {
    int NS = P.gx * P.gy;
    float al = (1 + P.mt) / 2;
    CHECKC(al > P.mt + 1.0 / 64 && al / 3 < P.mt, "table requirement for the hudless cases");
    int Khi = (int)floor(NS * (double)P.mfh) + 1, Klo = (int)floor(NS * (double)P.mfh);
    CHECKC(P.mfh < P.mfa, "control requirement mfh < mfa");
    id<MTLTexture> H = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, 640, 360), C = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, 640, 360);
    tex_set(H, ^(uint32_t x, uint32_t y, float v[4]) { float q = (float)((7 * x + 13 * y) % 64) / 63; v[0] = v[1] = v[2] = q; v[3] = 1; });
    /* C blended with U at samples m < K, for the channels in mask */
    void (^build)(int, int) = ^(int K, int mask) {
        tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) {
            float q = (float)((7 * x + 13 * y) % 64) / 63;
            v[0] = v[1] = v[2] = q; v[3] = 1;
            for (int j = 0; j < P.gy; j++) for (int i = 0; i < P.gx; i++)
                if (sx(640, i) == x && sy(360, j) == y && (int)(P.gx * j + i) < K)
                    for (int c = 0; c < 3; c++) if (mask & (1 << c)) { float U = q > 0.5f ? 0.0f : 1.0f; v[c] = (1 - al) * q + al * U; }
        });
    };
    void (^run)(Det) = ^(Det d) {
        nn_run(&R, ^(MetalFGEnc *e) { [M detectHudless:e state:d.state flag:d.flag colour:C hudless:H args:dargs(640, 360, 0, 0, P.mfh)]; });
    };
    Det d = det_new();
    build(Khi, 7); run(d);
    CHECKC(wd(d, 1) == 1, "md-hud-count K_hi: D=%u", wd(d, 1));
    d = det_new();
    build(Klo, 7); run(d);
    CHECKC(wd(d, 1) == 0, "md-hud-count K_lo: D=%u", wd(d, 1));
    d = det_new();
    build(NS, 1); run(d);
    CHECKC(wd(d, 1) == 1, "md-hud-onechan: D=%u (an averaging detector would see al/3 < mt)", wd(d, 1));
    /* extremes */
    d = det_new();
    tex_set(H, ^(uint32_t x, uint32_t y, float v[4]) { v[3] = 1; });
    tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = al; v[3] = 1; });
    run(d);
    CHECKC(wd(d, 1) == 1, "md-hud-extreme run L: D=%u", wd(d, 1));
    d = det_new();
    tex_set(H, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = 1; v[3] = 1; });
    tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = 1 - al; v[3] = 1; });
    run(d);
    CHECKC(wd(d, 1) == 1, "md-hud-extreme run U: D=%u", wd(d, 1));
    /* non-finite */
    d = det_new();
    tex_set(H, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = 0.5f; v[3] = 1; });
    tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = INFINITY; v[1] = v[2] = 0.5f; v[3] = 1; });
    run(d);
    CHECKC(wd(d, 1) == 1, "md-hud-nonfinite: D=%u", wd(d, 1));
    d = det_new();
    tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = NAN; v[1] = v[2] = 0.5f; v[3] = 1; });
    run(d);
    CHECKC(wd(d, 1) == 0, "NaN colour never counts");
}

static void override_pass(void) {
    Det d = det_new();
    id<MTLTexture> col = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, 64, 48), out = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, 80, 60);
    tex_set(col, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = (float)x / 64; v[1] = (float)y / 48; v[2] = 0.5f; v[3] = 1; });
    tex_set(out, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.25f; v[3] = 1; });
    static uint32_t so[2] = {0, 0}, dof[2] = {8, 6}, sz[2] = {64, 48};
    nn_run(&R, ^(MetalFGEnc *e) { [M override:e state:d.state colour:col out:out srcOrigin:so dstOrigin:dof size:sz]; });
    float *o = tex_get(out, 4);
    CHECKC(o[((6) * 80 + 8) * 4] == 0.25f, "F27 does nothing while W0 is zero");
    free(o);
    ((uint32_t *)d.state.contents)[0] = 1;
    nn_run(&R, ^(MetalFGEnc *e) { [M override:e state:d.state colour:col out:out srcOrigin:so dstOrigin:dof size:sz]; });
    o = tex_get(out, 4);
    int bad = 0;
    for (int y = 0; y < 48; y++) for (int x = 0; x < 64; x++) if (fabsf(o[((y + 6) * 80 + x + 8) * 4] - (float)x / 64) > 1e-3 || o[((y + 6) * 80 + x + 8) * 4 + 1] != (h16)((float)y / 48)) bad++;
    CHECKC(bad == 0 && o[0] == 0.25f, "F27 copies the rectangle when W0 is set (%d wrong)", bad);
    free(o);
}

int main(void) {
    @autoreleasepool {
        P = wgen_params();
        Gpu g = gpu_open();
        R = plain_rig(g);
        M = [[MetalFGMenu alloc] initWithDevice:g.dev library:R.lib];
        if (!M) { printf("menu kernels missing\n"); return 2; }
        count_edge(); thr_edge(); grid_rect_uicolour(); hold_sticky_reset(); hudless(); override_pass();
    }
    printf(fails ? "g_menu: %d FAILURES\n" : "g_menu: ok\n", fails);
    return fails != 0;
}
