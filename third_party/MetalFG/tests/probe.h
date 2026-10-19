/* Probe kernels compiled from source at run time (device facts the specs ask the harness to determine on the target) and the store-conversion probe. */
#pragma once
#include "gputil.h"

static inline id<MTLComputePipelineState> src_pso(Gpu *g, NSString *src, NSString *name) {
    NSError *e = nil;
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.fastMathEnabled = NO;
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:src options:o error:&e];
    if (!lib) { printf("probe compile: %s\n", e.localizedDescription.UTF8String); exit(2); }
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> p = [g->dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { printf("probe pipeline: %s\n", e.localizedDescription.UTF8String); exit(2); }
    return p;
}

/* Runs pso over n threads with the buffers bound in order, waits. */
static inline void src_run(Gpu *g, id<MTLComputePipelineState> pso, NSArray<id<MTLBuffer>> *bufs, NSUInteger n) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        for (NSUInteger i = 0; i < bufs.count; i++) [ce setBuffer:bufs[i] offset:0 atIndex:i];
        NSUInteger tg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)64);
        [ce dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { printf("probe command buffer error\n"); exit(3); }
    }
}

static inline id<MTLBuffer> src_buf(Gpu *g, const void *p, size_t bytes) {
    id<MTLBuffer> b = [g->dev newBufferWithLength:bytes + 16 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, b.length);
    if (p) memcpy(b.contents, p, bytes);
    return b;
}

/* Store probe (fg.arithmetic_pin_r4): write one plus three quarters of the binary16 spacing at one to a binary16 image. Returns 1 when the platform's store
   conversion rounds to nearest even (result one plus one spacing), 0 when it truncates (result one). */
static inline int store_is_nearest(Gpu *g) {
    NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void sp(texture2d<float, access::write> t [[texture(0)]], uint i [[thread_position_in_grid]]) { t.write(float4(1.0f + 0.75f * 0x1p-10f, 0, 0, 0), uint2(0, 0)); }\n";
    id<MTLComputePipelineState> p = src_pso(g, src, @"sp");
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1 height:1 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [g->dev newTextureWithDescriptor:d];
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:p];
        [ce setTexture:t atIndex:0];
        [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    __fp16 px[4];
    [t getBytes:px bytesPerRow:8 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    return (float)px[0] > 1.0f;
}
