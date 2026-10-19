/* GPU test helpers: device, queues, textures, running one engine call on either encoder API. */
#pragma once
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "metalfg.h"
#include "wgen.h"

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTL4CommandQueue> q4;
    id<MTL4CommandAllocator> alloc;
    id<MTLSharedEvent> ev;
    uint64_t evv;
} Gpu;

static inline Gpu gpu_open(void) {
    Gpu g;
    memset((void *)&g, 0, sizeof g);
    g.dev = MTLCreateSystemDefaultDevice();
    if (!g.dev) { printf("no Metal device\n"); exit(2); }
    g.q = [g.dev newCommandQueue];
    g.q4 = [g.dev newMTL4CommandQueue];
    g.alloc = [g.dev newCommandAllocator];
    g.ev = [g.dev newSharedEvent];
    return g;
}

static inline id<MTLTexture> gpu_tex(Gpu *g, MTLPixelFormat f, uint32_t w, uint32_t h) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [g->dev newTextureWithDescriptor:d];
}

static inline void gpu_upload(id<MTLTexture> t, const void *p, uint32_t bpp) {
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0 withBytes:p bytesPerRow:t.width * bpp];
}

static inline void gpu_download(id<MTLTexture> t, void *p, uint32_t bpp) {
    [t getBytes:p bytesPerRow:t.width * bpp fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
}

/* Encode one call into a fresh command buffer, run it and wait. classic = 1: metalfg_encode_classic. Returns the encode status. */
static inline int gpu_run(Gpu *g, MetalFGEngineRef e, const MetalFGFrameInfo *f, int classic) {
    @autoreleasepool {
        if (classic) {
            id<MTLCommandBuffer> cb = [g->q commandBuffer];
            int rc = metalfg_encode_classic(e, (__bridge void *)cb, f);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) { printf("command buffer error: %s\n", cb.error.localizedDescription.UTF8String); exit(3); }
            return rc;
        }
        id<MTL4CommandBuffer> cb = [g->dev newCommandBuffer];
        [cb beginCommandBufferWithAllocator:g->alloc];
        int rc = metalfg_encode(e, (__bridge void *)cb, f);
        [cb endCommandBuffer];
        id<MTL4CommandBuffer> list[1] = {cb};
        [g->q4 commit:list count:1];
        [g->q4 signalEvent:g->ev value:++g->evv];
        if (![g->ev waitUntilSignaledValue:g->evv timeoutMS:60000]) { printf("Metal 4 call timed out\n"); exit(3); }
        [g->alloc reset];
        return rc;
    }
}

/* The `pan` scene of fg.resources section 7. */
static inline uint16_t f16bits(double v) {
    _Float16 h = (_Float16)v;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

static inline void pan_colour(uint16_t *out, uint32_t w, uint32_t h, int frame) {
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++)
            for (int c = 0; c < 4; c++) {
                int u = (int)x - 3 * frame, v = (int)y - 2 * frame;
                int q = ((7 * u + 13 * v + 29 * c) % 64 + 64) % 64;
                out[(y * w + x) * 4 + c] = c == 3 ? f16bits(1.0) : f16bits(q / 63.0);
            }
}

typedef struct {
    Gpu gpu;
    MetalFGEngineRef eng;
    id<MTLTexture> colour, depth, motion, out, out_real;
    id<MTLBuffer> skip;
    MetalFGFrameInfo f;
    uint32_t W, H;
} Rig;

static inline MetalFGCreateInfo rig_info(uint32_t W, uint32_t H, uint32_t flags, const char *wpath) {
    MetalFGCreateInfo ci;
    memset(&ci, 0, sizeof ci);
    ci.version = 1; ci.display_w = W; ci.display_h = H; ci.flags = flags; ci.max_generated = 1;
    ci.weights_file = wpath;
    ci.metallib_dir = getenv("METALFG_METALLIB_DIR") ? getenv("METALFG_METALLIB_DIR") : "build";
    return ci;
}

/* Rig with the pan scene: textures of the display size, frame record filled. */
static inline int rig_open(Rig *r, Gpu g, uint32_t W, uint32_t H, uint32_t flags, const char *wpath, uint32_t outW, uint32_t outH) {
    memset((void *)r, 0, sizeof *r);
    r->gpu = g; r->W = W; r->H = H;
    MetalFGCreateInfo ci = rig_info(W, H, flags, wpath);
    char err[256];
    r->eng = metalfg_create((__bridge void *)g.dev, &ci, err, sizeof err);
    if (!r->eng) { printf("create failed: %s\n", err); return 1; }
    r->colour = gpu_tex(&r->gpu, MTLPixelFormatRGBA16Float, W, H);
    r->depth = gpu_tex(&r->gpu, MTLPixelFormatR32Float, W, H);
    r->motion = gpu_tex(&r->gpu, MTLPixelFormatRG16Float, W, H);
    r->out = gpu_tex(&r->gpu, MTLPixelFormatRGBA16Float, outW, outH);
    float *d = malloc((size_t)W * H * 4);
    for (size_t i = 0; i < (size_t)W * H; i++) d[i] = 0.5f;
    gpu_upload(r->depth, d, 4);
    free(d);
    uint16_t *m = malloc((size_t)W * H * 4);
    for (size_t i = 0; i < (size_t)W * H; i++) { m[2 * i] = f16bits(-3); m[2 * i + 1] = f16bits(-2); }
    gpu_upload(r->motion, m, 4);
    free(m);
    MetalFGFrameInfo *f = &r->f;
    memset(f, 0, sizeof *f);
    f->colour = (__bridge void *)r->colour; f->depth = (__bridge void *)r->depth; f->motion = (__bridge void *)r->motion;
    f->out_interp = (__bridge void *)r->out;
    f->content_w = W; f->content_h = H;
    f->motion_scale_x = f->motion_scale_y = 1.0f;
    f->near_plane = 0.5f; f->far_plane = 64.0f; f->fov_deg = 60.0f; f->aspect = (float)W / H;
    return 0;
}

static inline void rig_frame(Rig *r, int frame) {
    uint16_t *c = malloc((size_t)r->W * r->H * 8);
    pan_colour(c, r->W, r->H, frame);
    gpu_upload(r->colour, c, 8);
    free(c);
}
