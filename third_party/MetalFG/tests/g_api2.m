/* Interface version 2 of the engine API (spec fg.engine_api_r2 section 7): api-size-motion, api-size-depth, api-size-reset, api-size-game, api-v1-length,
   api-version, api-size-bounds. */
#import <Foundation/Foundation.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>
#include "gputil.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *WPATH = "build/tests/w_api2.bin";
enum { NF = 6 };                 /* frames 0 to 5; frame 0 is a reset */

/* depth texture of the given format holding the constant v (depth formats through a blit from a buffer) */
static id<MTLTexture> depth_tex(Gpu *g, MTLPixelFormat fmt, uint32_t w, uint32_t h, float v) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    const BOOL isDepth = fmt != MTLPixelFormatR32Float;
    d.storageMode = isDepth ? MTLStorageModePrivate : MTLStorageModeShared;
    id<MTLTexture> t = [g->dev newTextureWithDescriptor:d];
    float *p = malloc((size_t)w * h * 4);
    for (size_t i = 0; i < (size_t)w * h; i++) p[i] = v;
    if (!isDepth) { gpu_upload(t, p, 4); free(p); return t; }
    id<MTLBuffer> b = [g->dev newBufferWithBytes:p length:(size_t)w * h * 4 options:MTLResourceStorageModeShared];
    free(p);
    id<MTLCommandBuffer> cb = [g->q commandBuffer];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:b sourceOffset:0 sourceBytesPerRow:w * 4 sourceBytesPerImage:(size_t)w * h * 4 sourceSize:MTLSizeMake(w, h, 1) toTexture:t
      destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)
               options:(fmt == MTLPixelFormatDepth32Float_Stencil8 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone)];
    [be endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return t;
}

static id<MTLTexture> motion_tex(Gpu *g, uint32_t w, uint32_t h, double mx, double my) {
    id<MTLTexture> t = gpu_tex(g, MTLPixelFormatRG16Float, w, h);
    uint16_t *m = malloc((size_t)w * h * 4);
    for (size_t i = 0; i < (size_t)w * h; i++) { m[2 * i] = f16bits(mx); m[2 * i + 1] = f16bits(my); }
    gpu_upload(t, m, 4);
    free(m);
    return t;
}

/* flat-pan: c(x, y) = ((x - D t) mod W) / W in the three colour channels, alpha 1 */
static void ramp(uint16_t *out, uint32_t W, uint32_t H, double D, int t) {
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            double u = fmod((double)x - D * t, (double)W);
            if (u < 0) u += W;
            const uint16_t v = f16bits(u / W);
            uint16_t *p = out + ((size_t)y * W + x) * 4;
            p[0] = p[1] = p[2] = v; p[3] = f16bits(1.0);
        }
}

typedef struct {
    uint32_t version;
    id<MTLTexture> depth, motion, depth2, motion2;   /* second pair from frame `switch_at` on (api-size-reset) */
    uint32_t dsz[2], msz[2], msz2[2];
    int switch_at;
    int v1_page;                                     /* api-v1-length: the 336-byte record at the end of a readable page */
} Case;

/* runs frames 0..5 at 1536x864; outputs (RGBA16Float) into out[frame], return codes into rc[frame]; message of the last failure into msg */
static void run_case(Gpu *g, const Case *c, uint16_t **out, int *rc, char *msg) {
    const uint32_t W = 1536, H = 864;
    MetalFGCreateInfo ci = rig_info(W, H, 0, WPATH);
    ci.version = c->version;
    char err[256];
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    if (!e) { printf("create failed: %s\n", err); fails++; return; }
    id<MTLTexture> col = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H), o = gpu_tex(g, MTLPixelFormatRGBA16Float, W, H);
    uint16_t *cbuf = malloc((size_t)W * H * 8);
    long pg = sysconf(_SC_PAGESIZE);
    uint8_t *pages = NULL;
    if (c->v1_page) {
        pages = mmap(NULL, (size_t)pg * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        mprotect(pages + pg, (size_t)pg, PROT_NONE);
    }
    msg[0] = 0;
    for (int t = 0; t < NF; t++) {
        ramp(cbuf, W, H, 6.0, t);
        gpu_upload(col, cbuf, 8);
        const int second = c->switch_at > 0 && t >= c->switch_at;
        MetalFGFrameInfo f;
        memset(&f, 0, sizeof f);
        f.colour = (__bridge void *)col;
        f.depth = (__bridge void *)(second && c->depth2 ? c->depth2 : c->depth);
        f.motion = (__bridge void *)(second && c->motion2 ? c->motion2 : c->motion);
        f.out_interp = (__bridge void *)o;
        f.content_w = W; f.content_h = H;
        f.motion_scale_x = f.motion_scale_y = 1.0f;
        f.near_plane = 0.5f; f.far_plane = 64.0f; f.fov_deg = 60.0f; f.aspect = (float)W / H;
        f.frame_index = (uint64_t)t;
        f.depth_size[0] = c->dsz[0]; f.depth_size[1] = c->dsz[1];
        f.motion_size[0] = second ? c->msz2[0] : c->msz[0]; f.motion_size[1] = second ? c->msz2[1] : c->msz[1];
        const MetalFGFrameInfo *fp = &f;
        if (c->v1_page) {
            uint8_t *dst = pages + pg - METALFG_FRAME_RECORD_V1;          /* last byte of the record = last byte of the readable page */
            memcpy(dst, &f, METALFG_FRAME_RECORD_V1);
            fp = (const MetalFGFrameInfo *)dst;
        }
        rc[t] = gpu_run(g, e, fp, 0);
        if (rc[t]) snprintf(msg, 256, "%s", metalfg_last_error(e));
        out[t] = malloc((size_t)W * H * 8);
        gpu_download(o, out[t], 8);
    }
    if (pages) munmap(pages, (size_t)pg * 2);
    free(cbuf);
    metalfg_destroy(e);
}

static void free_frames(uint16_t **o) { for (int t = 0; t < NF; t++) { free(o[t]); o[t] = NULL; } }

static int all_zero(const int *rc) { for (int t = 0; t < NF; t++) if (rc[t]) return 0; return 1; }

static int finite_alpha1(const uint16_t *o, size_t n) {
    for (size_t i = 0; i < n; i++) {
        _Float16 v[4]; memcpy(v, o + 4 * i, 8);
        if (!isfinite((float)v[0]) || !isfinite((float)v[1]) || !isfinite((float)v[2]) || (float)v[3] != 1.0f) return 0;
    }
    return 1;
}

static int equals_colour(const uint16_t *o, int t) {
    const uint32_t W = 1536, H = 864;
    uint16_t *c = malloc((size_t)W * H * 8);
    ramp(c, W, H, 6.0, t);
    const int eq = !memcmp(c, o, (size_t)W * H * 8);
    free(c);
    return eq;
}

static void size_cases(Gpu *g) {
    const uint32_t W = 1536, H = 864;
    id<MTLTexture> dR = depth_tex(g, MTLPixelFormatR32Float, W, H, 0.5f), mR = motion_tex(g, W, H, 6, 0);
    id<MTLTexture> mS = motion_tex(g, 768, 432, 3, 0), dS = depth_tex(g, MTLPixelFormatR32Float, 768, 432, 0.5f);
    uint16_t *R[NF], *X[NF];
    int rcR[NF], rc[NF];
    char msg[256];

    Case r = {.version = 2, .depth = dR, .motion = mR};
    run_case(g, &r, R, rcR, msg);
    CHECK(all_zero(rcR), "run R: an encode failed: %s", msg);

    /* api-size-motion */
    Case cm = {.version = 2, .depth = dR, .motion = mS, .msz = {768, 432}};
    run_case(g, &cm, X, rc, msg);
    CHECK(all_zero(rc), "api-size-motion: an encode failed: %s", msg);
    for (int t = 0; t < NF; t++) CHECK(X[t] && !memcmp(X[t], R[t], (size_t)W * H * 8), "api-size-motion: frame %d differs from run R", t);
    free_frames(X);

    /* api-size-depth */
    Case cd = {.version = 2, .depth = dS, .motion = mR, .dsz = {768, 432}};
    run_case(g, &cd, X, rc, msg);
    CHECK(all_zero(rc), "api-size-depth: an encode failed: %s", msg);
    for (int t = 0; t < NF; t++) CHECK(X[t] && finite_alpha1(X[t], (size_t)W * H), "api-size-depth: frame %d not finite / alpha 1", t);
    for (int t = 1; t < NF; t++) CHECK(X[t] && !equals_colour(X[t], t), "api-size-depth: frame %d equals its real frame", t);
    free_frames(X);

    /* api-size-reset */
    Case cr = {.version = 2, .depth = dR, .motion = mS, .msz = {768, 432}, .motion2 = mR, .msz2 = {1536, 864}, .switch_at = 3};
    run_case(g, &cr, X, rc, msg);
    CHECK(all_zero(rc), "api-size-reset: an encode failed: %s", msg);
    CHECK(X[3] && equals_colour(X[3], 3), "api-size-reset: frame 3 (size change) is not the real frame");
    for (int t = 1; t < NF; t++) if (t != 3) CHECK(X[t] && !equals_colour(X[t], t), "api-size-reset: frame %d equals its real frame", t);
    free_frames(X);

    /* api-v1-length */
    Case cv = {.version = 1, .depth = dR, .motion = mR, .v1_page = 1};
    run_case(g, &cv, X, rc, msg);
    CHECK(all_zero(rc), "api-v1-length: an encode failed: %s", msg);
    for (int t = 0; t < NF; t++) CHECK(X[t] && !memcmp(X[t], R[t], (size_t)W * H * 8), "api-v1-length: frame %d differs from run R", t);
    free_frames(X);

    /* api-size-bounds */
    Case b1 = {.version = 2, .depth = dR, .motion = mS, .msz = {768, 433}};
    run_case(g, &b1, X, rc, msg);
    CHECK(rc[1] != 0 && strstr(msg, "motion_size"), "api-size-bounds: motion_size (768, 433) rc %d, message '%s'", rc[1], msg);
    free_frames(X);
    Case b2 = {.version = 2, .depth = dS, .motion = mR, .dsz = {769, 432}};
    run_case(g, &b2, X, rc, msg);
    CHECK(rc[1] != 0 && strstr(msg, "depth_size"), "api-size-bounds: depth_size (769, 432) rc %d, message '%s'", rc[1], msg);
    free_frames(X);
    free_frames(R);
}

static void version_case(Gpu *g) {
    char err[256];
    MetalFGCreateInfo ci = rig_info(640, 360, 0, WPATH);
    ci.version = 3;
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    CHECK(!e && strstr(err, "version"), "api-version: version 3 accepted or message without 'version': %s", err);
    for (uint32_t v = 1; v <= 2; v++) {
        ci.version = v;
        e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
        CHECK(e != NULL, "api-version: version %u rejected: %s", v, err);
        if (e) metalfg_destroy(e);
    }
}

/* api-size-game: 5120x2160 RGB10A2, depth Depth32Float_Stencil8 1708x724 (rectangle 1705x720), motion RG16Float 1705x722 (rectangle 1705x720) */
static void game_case(Gpu *g) {
    const uint32_t W = 5120, H = 2160;
    const double D = 5.0 * 5120.0 / 1705.0;
    MetalFGCreateInfo ci = rig_info(W, H, 0, WPATH);
    ci.version = 2;
    char err[256];
    MetalFGEngineRef e = metalfg_create((__bridge void *)g->dev, &ci, err, sizeof err);
    if (!e) { printf("create failed: %s\n", err); fails++; return; }
    id<MTLTexture> col = gpu_tex(g, MTLPixelFormatRGB10A2Unorm, W, H), o = gpu_tex(g, MTLPixelFormatRGB10A2Unorm, W, H);
    id<MTLTexture> dz = depth_tex(g, MTLPixelFormatDepth32Float_Stencil8, 1708, 724, 0.5f), mv = motion_tex(g, 1705, 722, 5, 0);
    uint16_t *c16 = malloc((size_t)W * H * 8);
    uint32_t *c10 = malloc((size_t)W * H * 4), *o10 = malloc((size_t)W * H * 4);
    for (int t = 0; t < NF; t++) {
        ramp(c16, W, H, D, t);
        for (size_t i = 0; i < (size_t)W * H; i++) {
            _Float16 v; memcpy(&v, c16 + 4 * i, 2);
            const uint32_t q = (uint32_t)lrint((double)v * 1023.0);
            c10[i] = q | (q << 10) | (q << 20) | (3u << 30);
        }
        [col replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:c10 bytesPerRow:W * 4];
        MetalFGFrameInfo f;
        memset(&f, 0, sizeof f);
        f.colour = (__bridge void *)col; f.depth = (__bridge void *)dz; f.motion = (__bridge void *)mv; f.out_interp = (__bridge void *)o;
        f.content_w = W; f.content_h = H;
        f.depth_size[0] = f.motion_size[0] = 1705; f.depth_size[1] = f.motion_size[1] = 720;
        f.motion_scale_x = f.motion_scale_y = 1.0f;
        f.near_plane = 0.5f; f.far_plane = 64.0f; f.fov_deg = 60.0f; f.aspect = (float)W / H;
        f.frame_index = (uint64_t)t;
        const int rc = gpu_run(g, e, &f, 0);
        CHECK(rc == 0, "api-size-game: frame %d encode failed: %s", t, metalfg_last_error(e));
        [o getBytes:o10 bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
        int alpha_ok = 1;
        double dmax = 0;
        for (size_t i = 0; i < (size_t)W * H; i++) {
            if ((o10[i] >> 30) != 3u) alpha_ok = 0;
            for (int k = 0; k < 3; k++) {
                const double a = ((o10[i] >> (10 * k)) & 1023u) / 1023.0, b = ((c10[i] >> (10 * k)) & 1023u) / 1023.0;
                if (fabs(a - b) > dmax) dmax = fabs(a - b);
            }
        }
        CHECK(alpha_ok, "api-size-game: frame %d alpha not 1", t);      /* RGB10A2 holds finite values only */
        if (t >= 1) CHECK(dmax > 1.0 / 64, "api-size-game: frame %d differs from the real frame by at most %.4f", t, dmax);
    }
    free(c16); free(c10); free(o10);
    metalfg_destroy(e);
}

int main(void) {
    @autoreleasepool {
        MetalFGParams p = wgen_params();
        if (wgen_write(WPATH, WGEN_RANDOM, 9, &p)) { printf("cannot write weights\n"); return 2; }
        Gpu g = gpu_open();
        version_case(&g);
        size_cases(&g);
        game_case(&g);
    }
    printf(fails ? "g_api2: %d FAILURES\n" : "g_api2: ok\n", fails);
    return fails != 0;
}
