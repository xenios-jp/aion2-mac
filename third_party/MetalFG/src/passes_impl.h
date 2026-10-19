#pragma once
/* Pass encoders shared between engine.m and the pass_*.m files. */
#import <Metal/Metal.h>
#include "engine.h"

/* Names of the compute kernels the pass files need, collected at engine creation. */
NSArray<NSString *> *metalfg_pass_kernel_names(void);

/* ---- dispatch helpers (a barrier precedes every dispatch: conservative and correct) ---- */
static inline MTLSize metalfg_tg2d(id<MTLComputePipelineState> p) {
    return p.maxTotalThreadsPerThreadgroup >= 256 ? MTLSizeMake(16, 16, 1) : MTLSizeMake(8, 8, 1);
}

static inline void metalfg_go2d(MetalFGEnc *enc, id<MTLComputePipelineState> p, NSUInteger w, NSUInteger h) {
    if (!w || !h) return;
    [enc setPipeline:p];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerGroup:metalfg_tg2d(p)];
}

static inline void metalfg_go1d(MetalFGEnc *enc, id<MTLComputePipelineState> p, NSUInteger n) {
    if (!n) return;
    [enc setPipeline:p];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
}

/* FH2-style fills */
typedef struct {
    float value[4];
    uint32_t count;
    uint32_t ivalue;
} MetalFGFillParams;

void metalfg_fill_tex(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> t, float r, float g, float b, float a);
void metalfg_fill_tex_u32(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> t, uint32_t v);
void metalfg_fill_buf_u32(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLBuffer> buf, uint32_t count32, uint32_t v);
/* F28 */
void metalfg_rect_copy(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> src, id<MTLTexture> dst, uint32_t sx, uint32_t sy,
                   uint32_t dx, uint32_t dy, uint32_t w, uint32_t h);
