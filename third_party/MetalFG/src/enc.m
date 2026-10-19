#import "enc.h"

@implementation MetalFGEnc {
    id<MTLComputeCommandEncoder> _c;
    id<MTL4ComputeCommandEncoder> _c4;
    id<MTL4CommandBuffer> _cb4;
    id<MTL4ArgumentTable> _table;
    id<MTLBuffer> _ring;
    NSUInteger _ringBase, _ringLen, _ringUsed;
    uint32_t _nd, _nb;
    BOOL _skipBarrier;
    id<MTL4CounterHeap> _heap;
    uint32_t _hcap, _ns;
    const char *_labels[512];
}

- (uint32_t)dispatchCount { return _nd; }
- (uint32_t)barrierCount { return _nb; }
- (uint32_t)stampCount { return _ns; }
- (const char *)stampLabel:(uint32_t)i { return i < _ns ? _labels[i] : ""; }

- (void)profileWithHeap:(id<MTL4CounterHeap>)heap capacity:(uint32_t)cap {
    if (!_c4 || !heap) return;
    _heap = heap;
    _hcap = cap < 512 ? cap : 512;
    _ns = 0;
    [heap invalidateCounterRange:NSMakeRange(0, _hcap)];
    [self stamp:"start"];
}

- (void)stamp:(const char *)label {
    if (!_heap || _ns >= _hcap) return;
    [_c4 writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:_heap atIndex:_ns];
    _labels[_ns++] = label;
}

+ (instancetype)classicWith:(id<MTLCommandBuffer>)cb {
    MetalFGEnc *e = [MetalFGEnc new];
    e->_c = [cb computeCommandEncoder];
    return e->_c ? e : nil;
}

+ (instancetype)metal4With:(id<MTL4CommandBuffer>)cb table:(id<MTL4ArgumentTable>)table ring:(id<MTLBuffer>)ring
                  ringBase:(NSUInteger)base ringLen:(NSUInteger)len {
    MetalFGEnc *e = [MetalFGEnc new];
    e->_cb4 = cb;
    e->_c4 = [cb computeCommandEncoder];
    if (!e->_c4) return nil;
    e->_table = table;
    e->_ring = ring;
    e->_ringBase = base;
    e->_ringLen = len;
    [e->_c4 setArgumentTable:table];
    /* fg.engine_api (encode): the call begins with a barrier. Metal 4 does not order work of different command buffers by itself, so a call waits for all
       earlier work on the queue (the previous call's history writes included) when the host keeps calls in flight without a fence. */
    [e->_c4 barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    return e;
}

- (BOOL)isMetal4 { return _c4 != nil; }

- (void)useResidencySet:(id<MTLResidencySet>)set {
    if (_cb4 && set) [_cb4 useResidencySet:set];
}

- (void)setPipeline:(id<MTLComputePipelineState>)pso {
    if (_c) [_c setComputePipelineState:pso]; else [_c4 setComputePipelineState:pso];
}

- (void)bindBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)offset at:(NSUInteger)index {
    if (_c) [_c setBuffer:buf offset:offset atIndex:index];
    else [_table setAddress:buf.gpuAddress + offset atIndex:index];
}

- (void)bindTexture:(id<MTLTexture>)tex at:(NSUInteger)index {
    if (_c) [_c setTexture:tex atIndex:index];
    else [_table setTexture:tex.gpuResourceID atIndex:index];
}

- (void)setBytes:(const void *)p length:(NSUInteger)n at:(NSUInteger)index {
    if (_c) { [_c setBytes:p length:n atIndex:index]; return; }
    NSUInteger off = (_ringUsed + 63) & ~(NSUInteger)63;
    NSAssert(off + n <= _ringLen, @"constant ring exhausted");
    memcpy((uint8_t *)_ring.contents + _ringBase + off, p, n);
    _ringUsed = off + n;
    [_table setAddress:_ring.gpuAddress + _ringBase + off atIndex:index];
}

- (void)dispatchThreads:(MTLSize)grid threadsPerGroup:(MTLSize)tg {
    _nd++;
    if (_c) [_c dispatchThreads:grid threadsPerThreadgroup:tg];
    else [_c4 dispatchThreads:grid threadsPerThreadgroup:tg];
    [self fill];
}

- (void)dispatchGroups:(MTLSize)groups threadsPerGroup:(MTLSize)tg {
    _nd++;
    if (_c) [_c dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    else [_c4 dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    [self fill];
}

- (void)dispatchGroupsIndirect:(id<MTLBuffer>)buf offset:(NSUInteger)offset threadsPerGroup:(MTLSize)tg {
    _nd++;
    if (_c) [_c dispatchThreadgroupsWithIndirectBuffer:buf indirectBufferOffset:offset threadsPerThreadgroup:tg];
    else [_c4 dispatchThreadgroupsWithIndirectBuffer:buf.gpuAddress + offset threadsPerThreadgroup:tg];
    [self fill];
}

- (void)skipNextBarrier { _skipBarrier = YES; }

- (void)fill {
    void (^f)(void) = _filler;
    if (!f) return;
    _filler = nil;
    f();
    _filler = f;
}

- (void)barrier {
    if (_skipBarrier) { _skipBarrier = NO; return; }   /* the dispatch that follows is independent of the one before */
    _nb++;
    if (_c) return; /* the classic encoder is created serial: dispatches execute in order */
    [_c4 barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                 visibilityOptions:MTL4VisibilityOptionDevice];
}

- (void)waitFence:(id<MTLFence>)f {
    if (!f) return;
    if (_c) [_c waitForFence:f];
    else [_c4 waitForFence:f beforeEncoderStages:MTLStageDispatch];
}

- (void)updateFence:(id<MTLFence>)f {
    if (!f) return;
    if (_c) [_c updateFence:f];
    else [_c4 updateFence:f afterEncoderStages:MTLStageDispatch];
}

- (void)end {
    if (_c) [_c endEncoding]; else [_c4 endEncoding];
}
@end
