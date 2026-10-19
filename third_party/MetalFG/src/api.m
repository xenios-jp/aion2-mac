/* Public C entry points (spec fg.engine_api). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include "engine.h"
#include "metalfg.h"

static _Thread_local char g_last_create_error[256];

MetalFGEngineRef metalfg_create(void *dev, const MetalFGCreateInfo *info, char *err, size_t cap) {
    @autoreleasepool {
        char msg[256] = {0};
        MetalFGEngineImpl *e = [[MetalFGEngineImpl alloc] initWithDevice:(__bridge id<MTLDevice>)dev info:info err:msg cap:sizeof msg];
        if (!e) {
            if (!msg[0]) snprintf(msg, sizeof msg, "engine creation failed");
            snprintf(g_last_create_error, sizeof g_last_create_error, "%s", msg);
            if (err && cap) snprintf(err, cap, "%s", msg);
            return NULL;
        }
        g_last_create_error[0] = 0;
        return (__bridge_retained MetalFGEngineRef)e;
    }
}

int metalfg_encode(MetalFGEngineRef h, void *cb, const MetalFGFrameInfo *f) {
    if (!h) return 1;
    @autoreleasepool {
        MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
        return [e encodeMetal4:(__bridge id<MTL4CommandBuffer>)cb frame:f];
    }
}

int metalfg_encode_classic(MetalFGEngineRef h, void *cb, const MetalFGFrameInfo *f) {
    if (!h) return 1;
    @autoreleasepool {
        MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
        return [e encodeClassic:(__bridge id<MTLCommandBuffer>)cb frame:f];
    }
}

const char *metalfg_last_error(MetalFGEngineRef h) {
    if (!h) return g_last_create_error;
    MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
    return e.lastError ? e.lastError.UTF8String : "";
}

/* Per-pass GPU times of the newest call encoded with profiling (create field profile = 1, Metal 4 command buffers): one record per stamp, the label naming the
   work between the previous stamp and this one. The caller reads them after that call's command buffer completed. */
uint32_t metalfg_timings(MetalFGEngineRef h, MetalFGTiming *out, uint32_t cap) {
    if (!h) return 0;
    MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
    if (e->profStamps < 2 || !e.profHeaps.count) return 0;
    NSData *d = [e.profHeaps[e->profHeap] resolveCounterRange:NSMakeRange(0, e->profStamps)];
    if (!d || d.length < e->profStamps * sizeof(MTL4TimestampHeapEntry)) return 0;
    const MTL4TimestampHeapEntry *t = d.bytes;
    const double us = 1e6 / (double)[e.device queryTimestampFrequency];
    uint32_t n = 0;
    for (uint32_t i = 1; i < e->profStamps && n < cap; i++, n++) {
        out[n].label = e->profLabels[i];
        out[n].microseconds = (double)(int64_t)(t[i].timestamp - t[i - 1].timestamp) * us;
    }
    return n;
}

void metalfg_destroy(MetalFGEngineRef h) {
    if (h) CFBridgingRelease(h);
}

#ifdef METALFG_TEST
/* diagnostics (fg.motion_grid_r1): the motion grid factor the engine chose */
uint32_t metalfg_probe_dm(MetalFGEngineRef h) {
    MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
    return (uint32_t)e->grids.dm;
}

size_t metalfg_probe(MetalFGEngineRef h, const char *name, int old_side, void *dst, size_t cap) {
    MetalFGEngineImpl *e = (__bridge MetalFGEngineImpl *)h;
    MetalFGSlot *s = e.slots[@(name)];
    if (!s) return 0;
    id<MTLTexture> t = [e tex:@(name) old:old_side != 0];
    id<MTLBuffer> b = [e buf:@(name) old:old_side != 0];
    if (t) {
        size_t bpr = t.width * s.desc.bytes, n = bpr * t.height;
        if (n > cap) return 0;
        [t getBytes:dst bytesPerRow:bpr fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
        return n;
    }
    if (b) {
        size_t n = b.length < cap ? b.length : cap;
        memcpy(dst, b.contents, n);
        return n;
    }
    return 0;
}
#endif
