/* End to end on the synthetic scenes of fg.test_scenes (plane, disc, depth-edge layers) at 640x360: the engine's generated
   frames are checked to be plausible images (finite, in range, not black, different from the reset frames and from the plain
   blend) and are scored against the analytic truth with the metrics of fg.test_scenes (pooled PSNR on D_q, yardstick, Q2).
   Two weight sets: all-zero (the networks add nothing, so the output is the motion-compensated candidate blend) and random
   (the output must still be finite and in range). */
#import <Foundation/Foundation.h>
#include "gputil.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { C_IDENT, C_PAN_H, C_PAN_D, C_DISOC, C_DEPTH_EDGE };

static double tex_r(double x, double y) {
    double c = (((int)floor(x / 48) + (int)floor(y / 48)) & 1) ? 0.15 : 0;
    return 0.5 + 0.2 * sin(2 * M_PI * x / 59) * cos(2 * M_PI * y / 67) + c;
}
static double tex_g(double x, double y) {
    double c = (((int)floor(x / 48) + (int)floor(y / 48)) & 1) ? 0.15 : 0;
    return 0.5 + 0.2 * sin(2 * M_PI * (x + y) / 101) - c / 2;
}
static double tex_b(double x, double y) {
    double c = (((int)floor(x / 48) + (int)floor(y / 48)) & 1) ? 0.15 : 0;
    return 0.5 + 0.15 * cos(2 * M_PI * x / 37 + 0.7) + 0.3 * c;
}
static void tex(double x, double y, double *o) { o[0] = tex_r(x, y); o[1] = tex_g(x, y); o[2] = tex_b(x, y); }

typedef struct { double rgb[3], depth, vx, vy; } Px;

static void plane(double vx, double vy, double tau, double x, double y, Px *p) {
    tex(x - vx * tau, y - vy * tau, p->rgb);
    p->depth = 0.9; p->vx = vx; p->vy = vy;
}

static void scene(int cs, double tau, double x, double y, Px *p) {
    switch (cs) {
    case C_IDENT: plane(0, 0, tau, x, y, p); break;
    case C_PAN_H: plane(8, 0, tau, x, y, p); break;
    case C_PAN_D: plane(5.5, -3.25, tau, x, y, p); break;
    case C_DISOC: {
        plane(0, 0, tau, x, y, p);
        double cx = 110 + 12 * tau, cy = 180;
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < 48.0 * 48) {
            double t2[3];
            tex(x - 12 * tau + 301, y + 177, t2);
            for (int k = 0; k < 3; k++) p->rgb[k] = 1 - t2[k];
            p->depth = 0.3; p->vx = 12; p->vy = 0;
        }
        break;
    }
    case C_DEPTH_EDGE: {
        plane(2, 0, tau, x, y, p);
        double t2[3];
        if (x >= 440 - 4 * tau && x < 560 - 4 * tau && y >= 200 && y < 300) {
            tex(x + 4 * tau + 523, y + 31, t2);
            for (int k = 0; k < 3; k++) p->rgb[k] = t2[k];
            p->depth = 0.6; p->vx = -4; p->vy = 0;
        }
        if (x >= 60 + 6 * tau && x < 230 + 6 * tau && y >= 40 && y < 160) {
            tex(x - 6 * tau + 101, y + 59, t2);
            for (int k = 0; k < 3; k++) p->rgb[k] = 1 - t2[k];
            p->depth = 0.4; p->vx = 6; p->vy = 0;
        }
        if (x >= 330 + 6 * tau && x < 336 + 6 * tau && y >= 80 && y < 280) {
            p->rgb[0] = 0.95; p->rgb[1] = 0.9; p->rgb[2] = 0.15;
            p->depth = 0.2; p->vx = 6; p->vy = 0;
        }
        break;
    }
    }
}

static const char *NAMES[] = {"fg-ident", "fg-pan-h", "fg-pan-d", "fg-disoc", "fg-depth-edge"};
static const double SMAX[] = {0, 8, 5.5, 12, 6};

static void upload_frame(id<MTLTexture> col, id<MTLTexture> dep, id<MTLTexture> mot, int cs, int frame, uint32_t W, uint32_t H) {
    uint16_t *c = malloc((size_t)W * H * 8), *m = malloc((size_t)W * H * 4);
    float *d = malloc((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            Px p;
            scene(cs, frame, x + 0.5, y + 0.5, &p);
            size_t i = (size_t)y * W + x;
            for (int k = 0; k < 3; k++) c[i * 4 + k] = f16bits(p.rgb[k]);
            c[i * 4 + 3] = f16bits(1);
            d[i] = (float)p.depth;
            m[i * 2] = f16bits(-p.vx); m[i * 2 + 1] = f16bits(-p.vy);
        }
    gpu_upload(col, c, 8); gpu_upload(dep, d, 4); gpu_upload(mot, m, 4);
    free(c); free(d); free(m);
}

static double h2d(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return (double)h; }

typedef struct { double mse_engine, mse_prev, mse_cur, mse_mean; double nonblack, diff_blend, diff_prev, diff_cur; int nonfinite, outrange; } Score;

static double psnr(double mse) { return mse <= 0 ? 999 : 10 * log10(1 / mse); }

static void run_case(Gpu *g, int cs, const char *wpath, const char *wname, int classic, int verbose) {
    const uint32_t W = 640, H = 360;
    float dc[3];
    metalfg_depth_config(0.1f, 1000.0f, 0, dc);
    MetalFGCreateInfo ci = rig_info(W, H, 0, wpath);
    ci.depth_scale = dc[0]; ci.depth_partition = dc[1]; ci.depth_min_sep = dc[2];
    char err[256];
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    if (!e) { printf("create failed: %s\n", err); fails++; return; }
    id<MTLTexture> col = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H), dep = gpu_tex(g, MTLPixelFormatR32Float, W, H);
    id<MTLTexture> mot = gpu_tex(g, MTLPixelFormatRG16Float, W, H), out = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H);
    MetalFGFrameInfo f;
    memset(&f, 0, sizeof f);
    f.colour = (__bridge void *)col; f.depth = (__bridge void *)dep; f.motion = (__bridge void *)mot; f.out_interp = (__bridge void *)out;
    f.content_w = W; f.content_h = H; f.motion_scale_x = f.motion_scale_y = 1;
    f.near_plane = 0.1f; f.far_plane = 1000; f.fov_deg = 60; f.aspect = (float)W / H;
    int nframes = cs == C_DISOC ? 6 : cs == C_IDENT ? 4 : 5;
    int M = 16 + 2 * (int)ceil(SMAX[cs]);
    double se = 0, sp = 0, sc = 0, sm = 0, npx = 0;
    uint16_t *o = malloc((size_t)W * H * 8), *pc = malloc((size_t)W * H * 8), *cc = malloc((size_t)W * H * 8);
    int bad_range = 0, nonfinite = 0;
    size_t diff_blend = 0, diff_prev = 0, diff_cur = 0, moving = 0;
    for (int fr = 0; fr < nframes; fr++) {
        upload_frame(col, dep, mot, cs, fr, W, H);
        gpu_download(col, cc, 8);
        CHECK(gpu_run(g, e, &f, classic) == 0, "%s %s frame %d: %s", NAMES[cs], wname, fr, metalfg_last_error(e));
        gpu_download(out, o, 8);
        if (fr == 0) {
            CHECK(memcmp(o, cc, (size_t)W * H * 8) == 0, "%s %s: the reset call outputs the real frame", NAMES[cs], wname);
        } else {
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++) {
                    size_t i = (size_t)y * W + x;
                    double v[3], pv[3], cv[3];
                    for (int k = 0; k < 3; k++) { v[k] = h2d(o[i * 4 + k]); pv[k] = h2d(pc[i * 4 + k]); cv[k] = h2d(cc[i * 4 + k]); }
                    int finite = 1, range = 1;
                    for (int k = 0; k < 3; k++) { finite &= isfinite(v[k]); range &= v[k] >= 0 && v[k] <= 1; }
                    nonfinite += !finite; bad_range += !range;
                    if (!finite) continue;
                    int inside = x >= M && x < W - M && y >= M && y < H - M;
                    Px t;
                    scene(cs, fr - 0.5, x + 0.5, y + 0.5, &t);
                    double dmax_blend = 0, dmax_prev = 0, dmax_cur = 0;
                    for (int k = 0; k < 3; k++) {
                        double tr = fmin(fmax(t.rgb[k], 0), 1);
                        if (inside) {
                            se += (v[k] - tr) * (v[k] - tr);
                            sp += (pv[k] - tr) * (pv[k] - tr);
                            sc += (cv[k] - tr) * (cv[k] - tr);
                            double mn = (pv[k] + cv[k]) / 2;
                            sm += (mn - tr) * (mn - tr);
                        }
                        dmax_blend = fmax(dmax_blend, fabs(v[k] - (pv[k] + cv[k]) / 2));
                        dmax_prev = fmax(dmax_prev, fabs(v[k] - pv[k]));
                        dmax_cur = fmax(dmax_cur, fabs(v[k] - cv[k]));
                    }
                    if (inside) npx += 3;
                    if (fabs(pv[0] - cv[0]) + fabs(pv[1] - cv[1]) + fabs(pv[2] - cv[2]) > 0.05) {
                        moving++;
                        diff_blend += dmax_blend > 1.0 / 64; diff_prev += dmax_prev > 1.0 / 64; diff_cur += dmax_cur > 1.0 / 64;
                    }
                }
        }
        memcpy(pc, cc, (size_t)W * H * 8);
    }
    double Pe = psnr(se / npx), Y = fmax(psnr(sp / npx), fmax(psnr(sc / npx), psnr(sm / npx)));
    if (verbose)
        printf("  %-14s %-6s %s: P(engine,truth)=%6.2f dB  yardstick Y=%6.2f dB  margin %+6.2f dB  moving px %zu: differ from blend %.1f%%, prev %.1f%%, cur %.1f%%\n",
               NAMES[cs], wname, classic ? "classic" : "metal4 ", Pe, Y, Pe - Y, moving, 100.0 * diff_blend / (moving ? moving : 1),
               100.0 * diff_prev / (moving ? moving : 1), 100.0 * diff_cur / (moving ? moving : 1));
    CHECK(nonfinite == 0, "%s %s: %d non-finite output texels", NAMES[cs], wname, nonfinite);
    CHECK(bad_range == 0, "%s %s: %d output texels outside 0..1", NAMES[cs], wname, bad_range);
    CHECK(Pe > 12.0, "%s %s: the output is far from the truth (%.2f dB): near-black or garbage", NAMES[cs], wname, Pe);
    if (cs != C_IDENT && moving > 0) {
        CHECK(diff_blend * 100 > moving, "%s %s: generated frames equal the plain blend at %zu of %zu moving texels", NAMES[cs], wname, moving - diff_blend, moving);
        CHECK(diff_prev * 100 > moving * 20 && diff_cur * 100 > moving * 20, "%s %s: generated frames equal a reset frame", NAMES[cs], wname);
    }
    /* with all-zero weights the blend weights are one half and the networks add nothing, so the 6 dB margin of Q2 is not
       expected; the motion-compensated candidates must still beat the best trivial frame */
    if (cs != C_IDENT) CHECK(Pe >= Y + 1.0, "%s %s: %.2f dB is not above the yardstick %.2f dB by 1 dB", NAMES[cs], wname, Pe, Y);
    free(o); free(pc); free(cc);
    metalfg_destroy(e);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        Gpu g = gpu_open();
        MetalFGParams P = wgen_params();
        char err[256];
        (void)err;
        wgen_write("build/tests/w_scene_zero.bin", WGEN_ZERO, 1, &P);
        wgen_write("build/tests/w_scene_rand.bin", WGEN_RANDOM, 7, &P);
        wgen_write("build/tests/w_scene_ones.bin", WGEN_ONES, 1, &P);
        int only = argc > 1 ? atoi(argv[1]) : -1;
        for (int cs = C_IDENT; cs <= C_DEPTH_EDGE; cs++) {
            if (only >= 0 && cs != only) continue;
            run_case(&g, cs, "build/tests/w_scene_zero.bin", "zero", 0, 1);
            run_case(&g, cs, "build/tests/w_scene_rand.bin", "random", 0, 1);
            run_case(&g, cs, "build/tests/w_scene_ones.bin", "ones", 0, 1);
        }
        run_case(&g, C_PAN_H, "build/tests/w_scene_zero.bin", "zero", 1, 1);
    }
    printf(fails ? "g_scenes: %d FAILURES\n" : "g_scenes: ok\n", fails);
    return fails != 0;
}
