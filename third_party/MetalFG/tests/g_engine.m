/* Engine skeleton on the GPU: create/encode errors, weights from the environment, outside-rectangle rule, reset output,
   classic/Metal 4 parity, clears and sentinels (specs fg.engine_api, fg.resources, fg.orchestration). */
#import <Foundation/Foundation.h>
#include "gputil.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *WPATH = "build/tests/w_engine.bin";

static void fill_out(id<MTLTexture> t, double v) {
    size_t n = t.width * t.height * 4;
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = f16bits(v);
    gpu_upload(t, p, 8);
    free(p);
}

static void create_errors(Gpu *g) {
    char err[256];
    MetalFGCreateInfo ci = rig_info(640, 360, 0, WPATH);
    void *dev = (__bridge void *)g->dev;
    MetalFGEngineRef e;
    ci.version = 3;
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "version"), "bad version: %s", err);
    ci = rig_info(640, 360, 0, WPATH);
    CHECK(!metalfg_create(NULL, &ci, err, sizeof err) && strstr(err, "device"), "null device: %s", err);
    ci.display_w = 3;
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "at least"), "tiny size: %s", err);
    ci = rig_info(640, 360, 0, "build/tests/missing.bin");
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "cannot open"), "missing weights: %s", err);
    CHECK(strstr(metalfg_last_error(NULL), "cannot open"), "last_error(NULL): %s", metalfg_last_error(NULL));
    ci = rig_info(640, 360, 0, WPATH);
    ci.depth_scale = -1;
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "depth_scale"), "negative depth_scale: %s", err);
    ci = rig_info(640, 360, 0, WPATH);
    ci.metallib_dir = "build/tests/nowhere";
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "shader library"), "missing metallib: %s", err);

    /* api-weights-env */
    ci = rig_info(640, 360, 0, NULL);
    unsetenv("METALFG_WEIGHTS");
    /* num-grid-const (fg.numerics_r3): the entries gx and gy must hold the fixed constants */
    {
        MetalFGParams pp = wgen_params();
        CHECK(pp.gx == 64 && pp.gy == 32, "the test parameter set carries the constants");
        for (int which = 0; which < 3; which++) {
            MetalFGParams q = pp;
            if (which == 1) q.gx += 2;
            if (which == 2) q.gy += 2;
            wgen_write("build/tests/w_grid.bin", WGEN_ZERO, 1, &q);
            MetalFGCreateInfo c2 = rig_info(640, 360, 0, "build/tests/w_grid.bin");
            MetalFGEngineRef e2 = metalfg_create(dev, &c2, err, sizeof err);
            if (which == 0) CHECK(e2 != NULL, "num-grid-const: a file with the constants is accepted: %s", err);
            else CHECK(e2 == NULL && strstr(err, which == 1 ? "gx" : "gy"), "num-grid-const: changed %s is rejected, entry named (%s)", which == 1 ? "gx" : "gy", err);
            if (e2) metalfg_destroy(e2);
        }
    }
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "no weight file"), "unset env: %s", err);
    setenv("METALFG_WEIGHTS", "", 1);
    CHECK(!metalfg_create(dev, &ci, err, sizeof err) && strstr(err, "no weight file"), "empty env: %s", err);
    setenv("METALFG_WEIGHTS", "build/tests/missing.bin", 1);
    CHECK(!metalfg_create(dev, &ci, err, sizeof err), "env names a missing file");
    setenv("METALFG_WEIGHTS", WPATH, 1);
    e = metalfg_create(dev, &ci, err, sizeof err);
    CHECK(e != NULL, "env names a valid file: %s", err);
    metalfg_destroy(e);
    ci.weights_file = WPATH;
    setenv("METALFG_WEIGHTS", "build/tests/missing.bin", 1);
    e = metalfg_create(dev, &ci, err, sizeof err);
    CHECK(e != NULL, "a non-null path ignores the variable: %s", err);
    metalfg_destroy(e);
    unsetenv("METALFG_WEIGHTS");

    /* api-height-cap: cap = min(8192, 16 hb) */
    MetalFGParams P = wgen_params();
    uint32_t cap = 16u * (uint32_t)P.hb < 8192u ? 16u * (uint32_t)P.hb : 8192u;
    ci = rig_info(512, cap, METALFG_CREATE_DIRECT_OUTPUT, WPATH);
    e = metalfg_create(dev, &ci, err, sizeof err);
    CHECK(e != NULL, "height == cap succeeds: %s", err);
    metalfg_destroy(e);
    ci.display_h = cap + 1;
    e = metalfg_create(dev, &ci, err, sizeof err);
    char want[32];
    snprintf(want, sizeof want, "%u", cap);
    CHECK(!e && strstr(err, want), "height cap + 1 fails naming the cap: %s", err);
    metalfg_destroy(e);
}

static int rect_equals_colour(Rig *r, int frame, uint32_t ox, uint32_t oy) {
    uint32_t ow = (uint32_t)r->out.width;
    uint16_t *o = malloc((size_t)ow * r->out.height * 8), *c = malloc((size_t)r->W * r->H * 8);
    gpu_download(r->out, o, 8);
    pan_colour(c, r->W, r->H, frame);
    int ok = 1;
    for (uint32_t y = 0; y < r->H && ok; y++)
        for (uint32_t x = 0; x < r->W && ok; x++)
            if (memcmp(&o[((size_t)(y + oy) * ow + x + ox) * 4], &c[((size_t)y * r->W + x) * 4], 8)) ok = 0;
    free(o); free(c);
    return ok;
}

static void skeleton(Gpu *g) {
    /* api-outside, reset output and parity on both encoder paths; output texture larger than the display */
    uint32_t W = 128, H = 96, OW = 160, OH = 120, ox = 8, oy = 6;
    uint16_t *snap[2] = {0, 0};
    for (int classic = 0; classic < 2; classic++) {
        Rig r;
        if (rig_open(&r, *g, W, H, METALFG_CREATE_DIRECT_OUTPUT, WPATH, OW, OH)) { fails++; return; }
        r.f.out_off[0] = ox; r.f.out_off[1] = oy;
        r.skip = [g->dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        r.f.skip_flag = (__bridge void *)r.skip;
        for (int frame = 0; frame < 4; frame++) {
            fill_out(r.out, 0.123);
            rig_frame(&r, frame);
            ((uint8_t *)r.skip.contents)[0] = 255;
            int rc = gpu_run(g, r.eng, &r.f, classic);
            CHECK(rc == 0, "encode frame %d: %s", frame, metalfg_last_error(r.eng));
            if (frame == 0) {
                CHECK(rect_equals_colour(&r, frame, ox, oy), "classic=%d reset call: rectangle differs from the colour frame", classic);
            } else {
                /* generating calls: a valid frame (finite, in range, alpha one, no debugging marker). Its content depends on
                   passes whose specs are not approved yet, so it is not compared with the real frame here. */
                uint16_t *gp = malloc((size_t)OW * OH * 8);
                gpu_download(r.out, gp, 8);
                int invalid = 0;
                for (uint32_t yy = 0; yy < H; yy++)
                    for (uint32_t xx = 0; xx < W; xx++) {
                        uint16_t *px = &gp[((size_t)(yy + oy) * OW + xx + ox) * 4];
                        _Float16 c0, c1, c2, a;
                        memcpy(&c0, &px[0], 2); memcpy(&c1, &px[1], 2); memcpy(&c2, &px[2], 2); memcpy(&a, &px[3], 2);
                        if (!(c0 >= 0 && c0 <= 1 && c1 >= 0 && c1 <= 1 && c2 >= 0 && c2 <= 1) || a != 1 || (c0 == 0 && c1 == 1 && c2 == 1)) invalid++;
                    }
                CHECK(invalid == 0, "classic=%d frame %d: %d invalid pixels", classic, frame, invalid);
                free(gp);
            }
            /* outside the rectangle: colour texel at the same coordinates where the colour texture has one, untouched otherwise */
            uint16_t *o = malloc((size_t)OW * OH * 8), *c = malloc((size_t)W * H * 8);
            gpu_download(r.out, o, 8);
            pan_colour(c, W, H, frame);
            int bad = 0;
            for (uint32_t y = 0; y < OH; y++)
                for (uint32_t x = 0; x < OW; x++) {
                    int inside = x >= ox && x < ox + W && y >= oy && y < oy + H;
                    if (inside) continue;
                    uint16_t *px = &o[((size_t)y * OW + x) * 4];
                    if (x < W && y < H) { if (memcmp(px, &c[((size_t)y * W + x) * 4], 8)) bad++; }
                    else if (px[0] != f16bits(0.123)) bad++;
                }
            CHECK(bad == 0, "classic=%d frame %d: %d outside texels wrong", classic, frame, bad);
            free(o); free(c);
            if (frame == 3) {
                size_t n = (size_t)OW * OH * 8;
                snap[classic] = malloc(n);
                gpu_download(r.out, snap[classic], 8);
            }
        }
        /* sentinels of the scatter buffers are in place after a generating call */
#ifdef METALFG_TEST
        {
            uint32_t *sc = malloc((size_t)W / 2 * (H / 2) * 8);
            size_t got = metalfg_probe(r.eng, "sc.pv", 0, sc, (size_t)W / 2 * (H / 2) * 8);
            int bad = 0;
            for (size_t i = 0; i < got / 4; i++) bad += sc[i] != 0x7FFFFFFFu;
            CHECK(got > 0 && bad > 0, "sc.pv holds scatter winners after a generating call (%d non-sentinel words of %zu)", bad, got / 4);
            free(sc);
        }
#endif
        metalfg_destroy(r.eng);
    }
    { size_t nd = 0, first = (size_t)-1; for (size_t i = 0; i < (size_t)OW * OH; i++) if (memcmp((char *)snap[0] + i * 8, (char *)snap[1] + i * 8, 8)) { nd++; if (first == (size_t)-1) first = i; } if (nd) printf("  parity: %zu of %zu texels differ, first at (%zu,%zu)\n", nd, (size_t)OW * OH, first % OW, first / OW); }
    CHECK(!memcmp(snap[0], snap[1], (size_t)OW * OH * 8), "api-classic-parity: outputs differ");
    free(snap[0]); free(snap[1]);
}

static void encode_errors(Gpu *g) {
    Rig r;
    if (rig_open(&r, *g, 64, 64, METALFG_CREATE_DIRECT_OUTPUT, WPATH, 64, 64)) { fails++; return; }
    rig_frame(&r, 0);
    MetalFGFrameInfo f = r.f;
    f.colour = NULL;
    CHECK(gpu_run(g, r.eng, &f, 1) != 0 && strstr(metalfg_last_error(r.eng), "colour"), "null colour: %s", metalfg_last_error(r.eng));
    f = r.f;
    f.gen_index = 2; f.gen_count = 1;
    CHECK(gpu_run(g, r.eng, &f, 0) != 0 && strstr(metalfg_last_error(r.eng), "gen_index"), "gen_index > gen_count: %s", metalfg_last_error(r.eng));
    f = r.f;
    f.content_w = 128;
    CHECK(gpu_run(g, r.eng, &f, 1) != 0 && strstr(metalfg_last_error(r.eng), "colour"), "rectangle larger than texture: %s", metalfg_last_error(r.eng));
    f = r.f;
    id<MTLTexture> bad = gpu_tex(&r.gpu, MTLPixelFormatR8Unorm, 64, 64);
    f.colour = (__bridge void *)bad;
    CHECK(gpu_run(g, r.eng, &f, 1) != 0 && strstr(metalfg_last_error(r.eng), "pixel format"), "bad colour format: %s", metalfg_last_error(r.eng));
    f = r.f;
    f.out_interp = (__bridge void *)gpu_tex(&r.gpu, MTLPixelFormatRGBA16Float, 32, 64);
    CHECK(gpu_run(g, r.eng, &f, 1) != 0 && strstr(metalfg_last_error(r.eng), "out_interp"), "output too small: %s", metalfg_last_error(r.eng));
    f = r.f;
    f.gen_index = 1; f.gen_count = 2;
    CHECK(gpu_run(g, r.eng, &f, 0) != 0 && strstr(metalfg_last_error(r.eng), "max_generated"), "n > max_generated: %s", metalfg_last_error(r.eng));
    /* a good call still works after the failures */
    CHECK(gpu_run(g, r.eng, &r.f, 0) == 0, "good call after failures: %s", metalfg_last_error(r.eng));
    metalfg_destroy(r.eng);
}

/* Every option at once on both encoder APIs, including the push-pull output chain: no GPU error, valid output. */
static void options_smoke(Gpu *g) {
    uint32_t flags = METALFG_CREATE_UI_RECOMPOSE | METALFG_CREATE_MENU_DETECT | METALFG_CREATE_DISTORTION; /* chain (bit 4 clear) */
    for (int classic = 0; classic < 2; classic++) {
        Rig r;
        if (rig_open(&r, *g, 128, 96, flags, WPATH, 128, 96)) { fails++; return; }
        r.skip = [g->dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLTexture> hud = gpu_tex(g, MTLPixelFormatRGBA16Float, 128, 96), alpha = gpu_tex(g, MTLPixelFormatR8Unorm, 128, 96);
        id<MTLTexture> ui = gpu_tex(g, MTLPixelFormatRGBA8Unorm, 128, 96), field = gpu_tex(g, MTLPixelFormatRG16Float, 128, 96);
        r.out_real = gpu_tex(g, MTLPixelFormatBGRA8Unorm, 128, 96);
        uint8_t *z = calloc(128 * 96, 8);
        gpu_upload(hud, z, 8); gpu_upload(alpha, z, 1); gpu_upload(ui, z, 4); gpu_upload(field, z, 4);
        free(z);
        r.f.skip_flag = (__bridge void *)r.skip; r.f.hudless = (__bridge void *)hud; r.f.ui_alpha = (__bridge void *)alpha;
        r.f.ui_colour = (__bridge void *)ui; r.f.distortion = (__bridge void *)field; r.f.out_real = (__bridge void *)r.out_real;
        for (int frame = 0; frame < 5; frame++) {
            rig_frame(&r, frame);
            r.f.flags = METALFG_FRAME_DEPTH_REVERSED | (frame == 3 ? METALFG_FRAME_RESET : 0);
            CHECK(gpu_run(g, r.eng, &r.f, classic) == 0, "options frame %d: %s", frame, metalfg_last_error(r.eng));
        }
        uint16_t *o = malloc(128 * 96 * 8);
        gpu_download(r.out, o, 8);
        int invalid = 0;
        for (int i = 0; i < 128 * 96; i++) {
            _Float16 c[3];
            for (int k = 0; k < 3; k++) memcpy(&c[k], &o[4 * i + k], 2);
            invalid += !(c[0] >= 0 && c[0] <= 1 && c[1] >= 0 && c[1] <= 1 && c[2] >= 0 && c[2] <= 1);
        }
        CHECK(invalid == 0, "options (classic=%d): %d invalid pixels", classic, invalid);
        free(o);
        metalfg_destroy(r.eng);
    }
}

#ifdef METALFG_TEST
static int st_feat_is_half(MetalFGEngineRef e) {
    uint16_t buf[64 * 48];
    size_t n = metalfg_probe(e, "st.feat", 0, buf, sizeof buf);
    int ok = n > 0;
    for (size_t i = 0; i < n / 2; i++) ok &= buf[i] == 0x3800;      /* binary16 one half */
    return ok;
}

static int rect_equals(Rig *r, int frame) { return rect_equals_colour(r, frame, 0, 0); }

/* orch-reset-out, the st.feat start value, orch-size-reset, orch-prefill-order (X) and md-off */
/* mg-grid at the engine (fg.motion_grid_r1): the motion grid factor the engine reports, paths N and O */
uint32_t metalfg_probe_dm(MetalFGEngineRef e);
static void mg_grid_engine(Gpu *g) {
    static const uint32_t Hs[4] = {1080, 1439, 1440, 2160};
    MetalFGParams P = wgen_params();
    for (int i = 0; i < 4; i++)
        for (int path = 0; path < 2; path++) {
            const uint32_t H = Hs[i], W = H * 16 / 9, rule = (int64_t)H <= P.hm ? 2 : 4;
            MetalFGCreateInfo ci = rig_info(W, H, path ? METALFG_CREATE_FORMER_MOTION_GRID : 0, WPATH);
            char err[256];
            MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
            CHECK(e != NULL, "mg-grid: create %ux%u failed: %s", W, H, err);
            if (!e) continue;
            const uint32_t want = path ? rule : (H >= METALFG_MOTION_GRID_HQ ? 4u : rule);
            CHECK(metalfg_probe_dm(e) == want, "mg-grid: path %c at H = %u reports dm %u, expected %u", path ? 'O' : 'N', H, metalfg_probe_dm(e), want);
            metalfg_destroy(e);
        }
}

static void orch_cases(Gpu *g) {
    for (int classic = 0; classic < 2; classic++) {
        Rig r;
        if (rig_open(&r, *g, 128, 96, METALFG_CREATE_DIRECT_OUTPUT, WPATH, 128, 96)) { fails++; return; }
        r.skip = [g->dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        r.out_real = gpu_tex(g, MTLPixelFormatRGBA16Float, 128, 96);
        r.f.skip_flag = (__bridge void *)r.skip;
        r.f.out_real = (__bridge void *)r.out_real;
        int non_half_after_generate = 0;
        for (int call = 0; call < 6; call++) {
            rig_frame(&r, call);
            r.f.flags = call == 4 ? METALFG_FRAME_RESET : 0;
            ((uint8_t *)r.skip.contents)[0] = 255;
            CHECK(gpu_run(g, r.eng, &r.f, classic) == 0, "orch-reset-out call %d: %s", call, metalfg_last_error(r.eng));
            int reset = call == 0 || call == 4;
            if (reset) {
                CHECK(rect_equals(&r, call), "orch-reset-out: call %d output must equal the colour frame", call);
                CHECK(((uint8_t *)r.skip.contents)[0] == 0, "orch-reset-out: the byte after a reset call is %u", ((uint8_t *)r.skip.contents)[0]);
                CHECK(st_feat_is_half(r.eng), "call %d: st.feat holds one half after a reset call", call);
            } else {
                CHECK(((uint8_t *)r.skip.contents)[0] == 0, "orch2 flag byte: cleared by F01 at the first call of a non-reset group, no detector (%u)", ((uint8_t *)r.skip.contents)[0]);
                if (call == 3) non_half_after_generate = !st_feat_is_half(r.eng);
            }
            /* out_real holds the real frame in every call */
            uint16_t *a = malloc(128 * 96 * 8), *c = malloc(128 * 96 * 8);
            gpu_download(r.out_real, a, 8);
            pan_colour(c, 128, 96, call);
            CHECK(memcmp(a, c, 128 * 96 * 8) == 0, "out_real equals the colour frame at call %d", call);
            free(a); free(c);
        }
        CHECK(non_half_after_generate, "a generating call advances st.feat away from one half");
        /* orch-size-reset: content 64x48 after calls at 128x96 */
        r.f.content_w = 64; r.f.content_h = 48;
        rig_frame(&r, 6);
        CHECK(gpu_run(g, r.eng, &r.f, classic) == 0, "size change call: %s", metalfg_last_error(r.eng));
        CHECK(st_feat_is_half(r.eng), "orch-size-reset: a content size change resets st.feat");
        metalfg_destroy(r.eng);
    }
    /* orch-prefill-order X: menu detection with an all-255 alpha plane from call 2 on; F27 runs after the synthesis */
    Rig r;
    if (rig_open(&r, *g, 128, 96, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_MENU_DETECT, WPATH, 128, 96)) { fails++; return; }
    id<MTLTexture> alpha = gpu_tex(g, MTLPixelFormatR8Unorm, 128, 96);
    uint8_t *plane = malloc(128 * 96);
    r.f.ui_alpha = (__bridge void *)alpha;
    for (int call = 0; call < 6; call++) {
        memset(plane, call >= 2 ? 255 : 0, 128 * 96);
        gpu_upload(alpha, plane, 1);
        rig_frame(&r, call);
        CHECK(gpu_run(g, r.eng, &r.f, call & 1) == 0, "orch-prefill-order call %d: %s", call, metalfg_last_error(r.eng));
        if (call >= 2) CHECK(rect_equals(&r, call), "orch-prefill-order X: call %d must output the colour frame (override after synthesis)", call);
    }
    free(plane);
    metalfg_destroy(r.eng);
}

/* res-bdf-clear / mp-history-rule: the reduced distortion field is history only while calls carry a field */
static float bdf_max(MetalFGEngineRef e, int old_side) {
    uint16_t buf[64 * 48 * 4];
    size_t n = metalfg_probe(e, "bdf.mot", old_side, buf, sizeof buf);
    float m = 0;
    for (size_t i = 0; i < n / 2; i++) { _Float16 h; memcpy(&h, &buf[i], 2); m = fmaxf(m, fabsf((float)h)); }
    return n ? m : -1;
}

static void bdf_history(Gpu *g) {
    Rig r;
    if (rig_open(&r, *g, 128, 96, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_DISTORTION, WPATH, 128, 96)) { fails++; return; }
    id<MTLTexture> field = gpu_tex(g, MTLPixelFormatRGBA16Float, 128, 96);
    uint16_t *px = calloc(128 * 96 * 4, 2);
    for (int i = 0; i < 128 * 96; i++) px[4 * i] = f16bits(1.0 / 64);
    gpu_upload(field, px, 8);
    free(px);
    int with_field[6] = {0, 1, 1, 1, 0, 1};
    for (int call = 0; call < 6; call++) {
        rig_frame(&r, call);
        r.f.distortion = with_field[call] ? (__bridge void *)field : NULL;
        CHECK(gpu_run(g, r.eng, &r.f, call & 1) == 0, "encode call %d: %s", call, metalfg_last_error(r.eng));
        float cur_after = bdf_max(r.eng, 1); /* after the exchange the history side holds this call's reduced field */
        if (with_field[call]) CHECK(fabsf(cur_after - 1.0f / 64) < 1e-3f, "call %d: reduced field %g", call, cur_after);
        else CHECK(cur_after == 0, "call %d without a field: reduced field must be zero, is %g", call, cur_after);
    }
    metalfg_destroy(r.eng);
}
#endif

int main(void) {
    @autoreleasepool {
        MetalFGParams p = wgen_params();
        if (wgen_write(WPATH, WGEN_RANDOM, 5, &p)) { printf("cannot write weights\n"); return 2; }
        Gpu g = gpu_open();
        create_errors(&g);
        skeleton(&g);
        encode_errors(&g);
        options_smoke(&g);
#ifdef METALFG_TEST
        bdf_history(&g);
        orch_cases(&g);
        mg_grid_engine(&g);
#endif
    }
    printf(fails ? "g_engine: %d FAILURES\n" : "g_engine: ok\n", fails);
    return fails != 0;
}
