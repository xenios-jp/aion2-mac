/* The built library through its public C interface only (dlopen of build/libmetalfg.dylib): create with the default shader
   library lookup (next to the dylib), reset and generating calls on both encoder APIs, error query, destroy. */
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "metalfg.h"
#include "wgen.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint16_t h16(double v) { _Float16 h = (_Float16)v; uint16_t b; memcpy(&b, &h, 2); return b; }

int main(void) {
    @autoreleasepool {
        const char *wpath = "build/tests/w_dylib.bin";
        MetalFGParams P = wgen_params();
        if (wgen_write(wpath, WGEN_RANDOM, 3, &P)) return 2;
        void *lib = dlopen("build/libmetalfg.dylib", RTLD_NOW);
        if (!lib) { printf("dlopen: %s\n", dlerror()); return 2; }
        MetalFGEngineRef (*create)(void *, const MetalFGCreateInfo *, char *, size_t) = dlsym(lib, "metalfg_create");
        int (*enc4)(MetalFGEngineRef, void *, const MetalFGFrameInfo *) = dlsym(lib, "metalfg_encode");
        int (*enc1)(MetalFGEngineRef, void *, const MetalFGFrameInfo *) = dlsym(lib, "metalfg_encode_classic");
        const char *(*last)(MetalFGEngineRef) = dlsym(lib, "metalfg_last_error");
        void (*destroy)(MetalFGEngineRef) = dlsym(lib, "metalfg_destroy");
        uint32_t (*timings)(MetalFGEngineRef, MetalFGTiming *, uint32_t) = dlsym(lib, "metalfg_timings");
        int (*depthcfg)(float, float, uint32_t, float *) = dlsym(lib, "metalfg_depth_config");
        CHECK(create && enc4 && enc1 && last && destroy && timings && depthcfg, "all exported symbols resolve");
        CHECK(dlsym(lib, "metalfg_probe") == NULL && dlsym(lib, "metalfg_invert4_d") == NULL, "test hooks and helpers are not exported");
        if (fails) return 1;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTL4CommandQueue> q4 = [dev newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        uint64_t evv = 0;

        float dc[3];
        depthcfg(0.5f, 64.0f, 0, dc);
        MetalFGCreateInfo ci;
        memset(&ci, 0, sizeof ci);
        ci.version = 1; ci.display_w = 192; ci.display_h = 108; ci.max_generated = 1; ci.weights_file = wpath;
        ci.depth_scale = dc[0]; ci.depth_partition = dc[1]; ci.depth_min_sep = dc[2];
        char err[256] = "";
        MetalFGEngineRef e = create((__bridge void *)dev, &ci, err, sizeof err);   /* metallib_dir NULL: next to the dylib */
        CHECK(e != NULL, "create: %s", err);
        if (!e) return 1;

        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:192 height:108 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModeShared;
        id<MTLTexture> col = [dev newTextureWithDescriptor:d], out = [dev newTextureWithDescriptor:d];
        d.pixelFormat = MTLPixelFormatR32Float;
        id<MTLTexture> dep = [dev newTextureWithDescriptor:d];
        d.pixelFormat = MTLPixelFormatRG16Float;
        id<MTLTexture> mot = [dev newTextureWithDescriptor:d];
        float *z = malloc(192 * 108 * 4);
        for (int i = 0; i < 192 * 108; i++) z[i] = 0.5f;
        [dep replaceRegion:MTLRegionMake2D(0, 0, 192, 108) mipmapLevel:0 withBytes:z bytesPerRow:192 * 4];
        free(z);
        uint16_t *mv = malloc(192 * 108 * 4);
        for (int i = 0; i < 192 * 108; i++) { mv[2 * i] = h16(-3); mv[2 * i + 1] = h16(-2); }
        [mot replaceRegion:MTLRegionMake2D(0, 0, 192, 108) mipmapLevel:0 withBytes:mv bytesPerRow:192 * 4];
        free(mv);

        MetalFGFrameInfo f;
        memset(&f, 0, sizeof f);
        f.colour = (__bridge void *)col; f.depth = (__bridge void *)dep; f.motion = (__bridge void *)mot; f.out_interp = (__bridge void *)out;
        f.content_w = 192; f.content_h = 108; f.motion_scale_x = f.motion_scale_y = 1; f.near_plane = 0.5f; f.far_plane = 64;
        f.fov_deg = 60; f.aspect = 192.0f / 108;
        uint16_t *px = malloc(192 * 108 * 8);
        for (int frame = 0; frame < 5; frame++) {
            for (int y = 0; y < 108; y++)
                for (int x = 0; x < 192; x++)
                    for (int c = 0; c < 4; c++)
                        px[(y * 192 + x) * 4 + c] = c == 3 ? h16(1) : h16((((7 * (x - 3 * frame) + 13 * (y - 2 * frame) + 29 * c) % 64 + 64) % 64) / 63.0);
            [col replaceRegion:MTLRegionMake2D(0, 0, 192, 108) mipmapLevel:0 withBytes:px bytesPerRow:192 * 8];
            int classic = frame & 1, rc;
            if (classic) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                rc = enc1(e, (__bridge void *)cb, &f);
                [cb commit];
                [cb waitUntilCompleted];
                CHECK(cb.status != MTLCommandBufferStatusError, "classic command buffer error");
            } else {
                id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
                [cb beginCommandBufferWithAllocator:alloc];
                rc = enc4(e, (__bridge void *)cb, &f);
                [cb endCommandBuffer];
                id<MTL4CommandBuffer> l[1] = {cb};
                [q4 commit:l count:1];
                [q4 signalEvent:ev value:++evv];
                CHECK([ev waitUntilSignaledValue:evv timeoutMS:60000], "Metal 4 call timed out");
                [alloc reset];
            }
            CHECK(rc == 0, "frame %d: %s", frame, last(e));
            uint16_t *o = malloc(192 * 108 * 8);
            [out getBytes:o bytesPerRow:192 * 8 fromRegion:MTLRegionMake2D(0, 0, 192, 108) mipmapLevel:0];
            if (frame == 0) CHECK(memcmp(o, px, 192 * 108 * 8) == 0, "the reset call outputs the real frame");
            int invalid = 0;
            for (int i = 0; i < 192 * 108; i++) {
                _Float16 a, b2, c2;
                memcpy(&a, &o[4 * i], 2); memcpy(&b2, &o[4 * i + 1], 2); memcpy(&c2, &o[4 * i + 2], 2);
                invalid += !(a >= 0 && a <= 1 && b2 >= 0 && b2 <= 1 && c2 >= 0 && c2 <= 1);
            }
            CHECK(invalid == 0, "frame %d: %d invalid pixels", frame, invalid);
            free(o);
        }
        free(px);
        /* a failed call reports through the error query and encodes nothing */
        f.gen_index = 2; f.gen_count = 1;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        CHECK(enc1(e, (__bridge void *)cb, &f) != 0 && strstr(last(e), "gen_index"), "error query: %s", last(e));
        [cb commit];
        MetalFGTiming tm[4];
        CHECK(timings(e, tm, 4) <= 4, "timings");
        destroy(e);
        dlclose(lib);
    }
    printf(fails ? "d_dylib: %d FAILURES\n" : "d_dylib: ok\n", fails);
    return fails != 0;
}
