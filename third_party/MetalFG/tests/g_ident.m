/* Identity scene (fg-ident of fg.test_scenes): two identical frames, zero motion, constant depth. Every pass before the networks has
   a value that follows from the specs by hand, and with networks that add nothing the output is the frame bit for bit:

     F04 dY.feat           0     the luma change against an identical history (fg.colour_disocclusion)
     F05/F07 mvo, mvf      0     zero host motion, zero camera: nothing is dynamic (|mv - cam|^2 = 0 <= max(1, .)), v = cam' = 0 (fg.prev2curr)
     F08 nsc.mot           0     every cell receives a record (fg.prev2curr)
     F13 dis.feat          0     forward motion lands everywhere (fg.colour_disocclusion)
     F14/F15 err.feat      0     Pw = P, Cw = C, so the warp errors are zero (fg.net1_io)
     F17 st.feat           1/2   zero logits: 1 / (1 + exp(0))  (fg.net1_io)
     F18 sc.ip, sc.ic      every cell holds its own record with offset -h / (Wm, Hm) = -0: payload 0x80008000 (fg.interm_motion)
     F19 bw.mot            sig(0 - 0) = 1/2 with zero logits; 1 when the logit pairs are (24, 0)
     F20 candA, candB      the frame (alpha = quality = 0), hole.full 0  (fg.blend_pack)
     F21 n2out             0 (zero weights)
     F22/F23 output        the frame

   Two weight sets: all zero, and WGEN_ONES (head biases make both blend weights one). The case guards against bias terms and
   uninitialised reads (fg.test_scenes). Also checks a translating plane with the WGEN_ONES weights (the networks neutral, the blend weights
   one): the generated frame is the exact motion-compensated candidate, so against the analytic truth it must reach the binary16 limit. */
#import <Foundation/Foundation.h>
#include "gputil.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double tex_c(double x, double y) { return (((int)floor(x / 48) + (int)floor(y / 48)) & 1) ? 0.15 : 0; }
static void tex(double x, double y, double *o) {
    double c = tex_c(x, y);
    o[0] = 0.5 + 0.2 * sin(2 * M_PI * x / 59) * cos(2 * M_PI * y / 67) + c;
    o[1] = 0.5 + 0.2 * sin(2 * M_PI * (x + y) / 101) - c / 2;
    o[2] = 0.5 + 0.15 * cos(2 * M_PI * x / 37 + 0.7) + 0.3 * c;
}
static double h2d(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return (double)h; }

static uint8_t buf[64 << 20];

/* count of 16-bit elements (of `ch` per texel) of the probed resource that differ from `want` (channel c) */
static size_t count_h16(MetalFGEngineRef e, const char *nm, int ch, const double *want, size_t *total) {
    size_t n = metalfg_probe(e, nm, 0, buf, sizeof buf), bad = 0;
    const uint16_t *p = (const uint16_t *)buf;
    *total = n / 2;
    for (size_t i = 0; i < n / 2; i++) if (h2d(p[i]) != want[i % ch]) bad++;
    return bad;
}

static size_t count_u8(MetalFGEngineRef e, const char *nm, int want, size_t *total) {
    size_t n = metalfg_probe(e, nm, 0, buf, sizeof buf), bad = 0;
    *total = n;
    for (size_t i = 0; i < n; i++) bad += buf[i] != want;
    return bad;
}

static void run_ident(Gpu *g, const char *wpath, const char *wname, double wbw) {
    const uint32_t W = 640, H = 360;
    float dc[3];
    metalfg_depth_config(0.1f, 1000.0f, 0, dc);
    MetalFGCreateInfo ci = rig_info(W, H, 0, wpath);
    ci.depth_scale = dc[0]; ci.depth_partition = dc[1]; ci.depth_min_sep = dc[2];
    char err[256];
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    if (!e) { CHECK(0, "create failed: %s", err); return; }
    id<MTLTexture> col = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H), dep = gpu_tex(g, MTLPixelFormatR32Float, W, H);
    id<MTLTexture> mot = gpu_tex(g, MTLPixelFormatRG16Float, W, H), out = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H);
    MetalFGFrameInfo f;
    memset(&f, 0, sizeof f);
    f.colour = (__bridge void *)col; f.depth = (__bridge void *)dep; f.motion = (__bridge void *)mot; f.out_interp = (__bridge void *)out;
    f.content_w = W; f.content_h = H; f.motion_scale_x = f.motion_scale_y = 1;
    f.near_plane = 0.1f; f.far_plane = 1000; f.fov_deg = 60; f.aspect = (float)W / H;
    uint16_t *c = malloc((size_t)W * H * 8), *m = calloc((size_t)W * H, 4), *o = malloc((size_t)W * H * 8);
    float *d = malloc((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            double v[3]; tex(x + 0.5, y + 0.5, v);
            size_t i = (size_t)y * W + x;
            for (int k = 0; k < 3; k++) c[i * 4 + k] = f16bits(v[k]);
            c[i * 4 + 3] = f16bits(1); d[i] = 0.9f;
        }
    gpu_upload(col, c, 8); gpu_upload(dep, d, 4); gpu_upload(mot, m, 4);
    for (int fr = 0; fr < 4; fr++) {
        CHECK(gpu_run(g, e, &f, fr & 1) == 0, "%s call %d: %s", wname, fr, metalfg_last_error(e));
        gpu_download(out, o, 8);
        if (fr == 0) { CHECK(memcmp(o, c, (size_t)W * H * 8) == 0, "%s: reset call", wname); continue; }
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)W * H; i++) {
            for (int k = 0; k < 3; k++) bad += o[i * 4 + k] != c[i * 4 + k];
            bad += h2d(o[i * 4 + 3]) != 1.0;
        }
        CHECK(bad == 0, "%s call %d: %zu output elements differ from the frame", wname, fr, bad);
        if (fr != 3) continue;
        size_t tot, nb;
        const double zero2[2] = {0, 0}, bw[2] = {wbw, wbw}, half1[1] = {0.5}, zero1[1] = {0}, zero8[8] = {0};
        nb = count_h16(e, "mvf.mot", 2, zero2, &tot); CHECK(nb == 0, "%s: mvf.mot not zero (%zu of %zu)", wname, nb, tot);
        nb = count_h16(e, "err.feat", 2, zero2, &tot); CHECK(nb == 0, "%s: err.feat.cur not zero (%zu of %zu)", wname, nb, tot);
        nb = count_h16(e, "dY.feat", 1, zero1, &tot); CHECK(nb == 0, "%s: dY.feat not zero (%zu of %zu)", wname, nb, tot);
        nb = count_h16(e, "bw.mot", 2, bw, &tot); CHECK(nb == 0, "%s: bw.mot is not (%g, %g) (%zu of %zu)", wname, wbw, wbw, nb, tot);
        nb = count_h16(e, "n2out", 8, zero8, &tot); CHECK(nb == 0, "%s: n2out not zero (%zu of %zu)", wname, nb, tot);
        if (wbw == 0.5) { nb = count_h16(e, "st.feat", 1, half1, &tot); CHECK(nb == 0, "%s: st.feat is not 1/2 (%zu of %zu)", wname, nb, tot); }
        nb = count_u8(e, "nsc.mot", 0, &tot); CHECK(nb == 0, "%s: nsc.mot not zero (%zu of %zu)", wname, nb, tot);
        nb = count_u8(e, "dis.feat", 0, &tot); CHECK(nb == 0, "%s: dis.feat not zero (%zu of %zu)", wname, nb, tot);
        nb = count_u8(e, "dyn.mot", 0, &tot); CHECK(nb == 0, "%s: dyn.mot not zero (%zu of %zu)", wname, nb, tot);
        nb = count_u8(e, "hole.full", 0, &tot); CHECK(nb == 0, "%s: hole.full not zero (%zu of %zu)", wname, nb, tot);
        const char *sn[] = {"sc.ip", "sc.ic"};
        for (int s = 0; s < 2; s++) {
            size_t n = metalfg_probe(e, sn[s], 0, buf, sizeof buf), badc = 0;
            const uint32_t *w = (const uint32_t *)buf;
            for (size_t k = 0; k + 1 < n / 4; k += 2) badc += w[k] != 0x80008000u;
            CHECK(badc == 0, "%s: %s holds %zu cells that are not (offset -0, -0) of the cell's own record", wname, sn[s], badc);
        }
        const char *cn[] = {"candA.full", "candB.full"};
        for (int s = 0; s < 2; s++) {
            size_t n = metalfg_probe(e, cn[s], 0, buf, sizeof buf), badc = 0;
            const uint16_t *p = (const uint16_t *)buf;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                for (int k = 0; k < 3; k++) badc += p[i * 4 + k] != c[i * 4 + k];
                badc += h2d(p[i * 4 + 3]) != 0.0;
            }
            CHECK(n >= (size_t)W * H * 8 && badc == 0, "%s: %s differs from the frame (alpha 0) at %zu elements", wname, cn[s], badc);
        }
    }
    free(c); free(m); free(o); free(d);
    metalfg_destroy(e);
}

/* translating plane with the neutral networks: the generated frame must equal the exact candidate */
static void run_pan(Gpu *g, const char *wpath, uint32_t W, uint32_t H, double vx, double vy, double min_db) {
    float dc[3];
    metalfg_depth_config(0.1f, 1000.0f, 0, dc);
    MetalFGCreateInfo ci = rig_info(W, H, 0, wpath);
    ci.depth_scale = dc[0]; ci.depth_partition = dc[1]; ci.depth_min_sep = dc[2];
    char err[256];
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    if (!e) { CHECK(0, "create failed: %s", err); return; }
    id<MTLTexture> col = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H), dep = gpu_tex(g, MTLPixelFormatR32Float, W, H);
    id<MTLTexture> mot = gpu_tex(g, MTLPixelFormatRG16Float, W, H), out = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H);
    MetalFGFrameInfo f;
    memset(&f, 0, sizeof f);
    f.colour = (__bridge void *)col; f.depth = (__bridge void *)dep; f.motion = (__bridge void *)mot; f.out_interp = (__bridge void *)out;
    f.content_w = W; f.content_h = H; f.motion_scale_x = f.motion_scale_y = 1;
    f.near_plane = 0.1f; f.far_plane = 1000; f.fov_deg = 60; f.aspect = (float)W / H;
    uint16_t *c = malloc((size_t)W * H * 8), *m = malloc((size_t)W * H * 4), *o = malloc((size_t)W * H * 8);
    float *d = malloc((size_t)W * H * 4);
    double se = 0, npx = 0;
    for (int fr = 0; fr < 4; fr++) {
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                double v[3]; tex(x + 0.5 - vx * fr, y + 0.5 - vy * fr, v);
                size_t i = (size_t)y * W + x;
                for (int k = 0; k < 3; k++) c[i * 4 + k] = f16bits(v[k]);
                c[i * 4 + 3] = f16bits(1); d[i] = 0.9f;
                m[i * 2] = f16bits(-vx); m[i * 2 + 1] = f16bits(-vy);
            }
        gpu_upload(col, c, 8); gpu_upload(dep, d, 4); gpu_upload(mot, m, 4);
        CHECK(gpu_run(g, e, &f, 0) == 0, "pan call %d: %s", fr, metalfg_last_error(e));
        if (fr == 0) continue;
        gpu_download(out, o, 8);
        const int M = 24;
        for (uint32_t y = M; y < H - M; y++)
            for (uint32_t x = M; x < W - M; x++) {
                double t[3]; tex(x + 0.5 - vx * (fr - 0.5), y + 0.5 - vy * (fr - 0.5), t);
                for (int k = 0; k < 3; k++) { double a = h2d(o[(y * W + x) * 4 + k]) - t[k]; se += a * a; npx++; }
            }
    }
    double db = -10 * log10(se / npx);
    printf("  pan (%g, %g) at %ux%u with neutral networks: %.2f dB against the truth\n", vx, vy, W, H, db);
    CHECK(db >= min_db, "pan (%g, %g): %.2f dB, expected at least %.1f", vx, vy, db, min_db);
    free(c); free(m); free(o); free(d);
    metalfg_destroy(e);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        MetalFGParams P = wgen_params();
        wgen_write("build/tests/w_ident_zero.bin", WGEN_ZERO, 1, &P);
        wgen_write("build/tests/w_ident_ones.bin", WGEN_ONES, 1, &P);
        run_ident(&g, "build/tests/w_ident_zero.bin", "zero", 0.5);
        run_ident(&g, "build/tests/w_ident_ones.bin", "ones", 1.0);
        /* whole-pixel motions (1 and 4 Mot pixels per frame) and the 2-pixel plane, whose Mot motion of one pixel is not above the dynamic
           threshold max(1, .) of fg.prev2curr and therefore stays static (v = cam' = 0): the previous candidate is then unwarped, so only the
           current one is exact and the pair is not judged here */
        run_pan(&g, "build/tests/w_ident_ones.bin", 640, 360, 8, 0, 70.0);
        run_pan(&g, "build/tests/w_ident_ones.bin", 640, 360, 4, 0, 70.0);
        run_pan(&g, "build/tests/w_ident_ones.bin", 1280, 720, 8, 0, 70.0);   /* s = 2 */
        run_pan(&g, "build/tests/w_ident_ones.bin", 3840, 2160, 8, 0, 70.0);  /* s = 8, Feat quarter of the display */
        run_pan(&g, "build/tests/w_ident_ones.bin", 331, 187, 8, 0, 70.0);    /* odd size */
        run_pan(&g, "build/tests/w_ident_ones.bin", 640, 360, 0, 8, 70.0);
    }
    printf(fails ? "g_ident: %d FAILURES\n" : "g_ident: ok\n", fails);
    return fails != 0;
}
