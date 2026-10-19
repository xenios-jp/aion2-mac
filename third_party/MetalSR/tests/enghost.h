/* Host-side helpers of the end-to-end tests: device, queue, one-call command buffers, textures. */
#ifndef ENGHOST_H
#define ENGHOST_H
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include "metalsr.h"
#include "scenes.h"
#include "ref_net.h"
#include "testparams.h"

static int g_fails;
#define EXPECT(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fails++; } } while (0)

static id<MTLDevice> dev;
static id<MTL4CommandQueue> queue;
static id<MTL4CommandAllocator> alloc;
static id<MTLSharedEvent> evt;
static uint64_t evt_val;
static uint8_t *g_img;
static SrParams g_mp;

static void write_weights(const char *path, int mode, double scale) {
    free(g_img);
    g_img = make_weights(mode, scale);
    FILE *f = fopen(path, "wb"); fwrite(g_img, 1, SR_WEIGHT_FILE_BYTES, f); fclose(f);
    uint8_t pb[1024]; size_t n = sr_write_params(&g_mp, pb);
    char pp[512]; snprintf(pp, sizeof pp, "%s.params", path);
    f = fopen(pp, "wb"); fwrite(pb, 1, n, f); fclose(f);
}

static id<MTLTexture> tex2d(MTLPixelFormat f, int w, int h, MTLTextureUsage u) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.usage = u; d.storageMode = MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}
static void put(id<MTLTexture> t, int x0, int y0, int w, int h, const void *p, int bpp) {
    [t replaceRegion:MTLRegionMake2D(x0, y0, w, h) mipmapLevel:0 withBytes:p bytesPerRow:w * bpp];
}

/* one engine call on its own command buffer; returns the encode status */
static int host_call_on(id<MTL4CommandQueue> q, id<MTL4CommandAllocator> al, id<MTLSharedEvent> ev, uint64_t *val, MetalSREngine *e, const MetalSRFrameInfo *fi) {
    id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
    [al reset];
    [cb beginCommandBufferWithAllocator:al];
    int rc = metalsr_encode(e, (__bridge void *)cb, fi);
    [cb endCommandBuffer];
    id<MTL4CommandBuffer> l[1] = {cb};
    [q commit:l count:1];
    [q signalEvent:ev value:++*val];
    [ev waitUntilSignaledValue:*val timeoutMS:60000];
    return rc;
}
static int host_call(MetalSREngine *e, const MetalSRFrameInfo *fi) { return host_call_on(queue, alloc, evt, &evt_val, e, fi); }

#endif
