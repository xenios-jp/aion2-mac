#pragma once
/* Thin encoder abstraction over classic and Metal 4 compute encoders (spec fg.engine_api, "Choice of command buffer API"):
   bind buffer, bind texture, set bytes, dispatch, barrier. Both back ends emit the same dispatches. */
#import <Metal/Metal.h>

@interface MetalFGEnc : NSObject
+ (instancetype)classicWith:(id<MTLCommandBuffer>)cb;
/* ring: shared buffer used for set-bytes data on Metal 4; the slice [ringBase, ringBase + ringLen) belongs to this call */
+ (instancetype)metal4With:(id<MTL4CommandBuffer>)cb
                     table:(id<MTL4ArgumentTable>)table
                      ring:(id<MTLBuffer>)ring
                  ringBase:(NSUInteger)base
                   ringLen:(NSUInteger)len;
- (void)useResidencySet:(id<MTLResidencySet>)set;
- (void)setPipeline:(id<MTLComputePipelineState>)pso;
- (void)bindBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)offset at:(NSUInteger)index;
- (void)bindTexture:(id<MTLTexture>)tex at:(NSUInteger)index;
- (void)setBytes:(const void *)p length:(NSUInteger)n at:(NSUInteger)index;
/* dispatch with a grid in threads (non-uniform threadgroups allowed) */
- (void)dispatchThreads:(MTLSize)grid threadsPerGroup:(MTLSize)tg;
- (void)dispatchGroups:(MTLSize)groups threadsPerGroup:(MTLSize)tg;
/* threadgroup counts (three uint32) read by the GPU from buf at offset */
- (void)dispatchGroupsIndirect:(id<MTLBuffer>)buf offset:(NSUInteger)offset threadsPerGroup:(MTLSize)tg;
/* dependency barrier between dispatches */
- (void)barrier;
/* the next barrier call is dropped: the dispatch it would precede does not depend on the one just encoded (Metal 4 runs such dispatches concurrently) */
- (void)skipNextBarrier;
/* independent work woven into a chain of dependent dispatches: called after each dispatch while set (not re-entered); it encodes its own dispatch behind
   skipNextBarrier, so that dispatch runs beside the one just encoded and the next barrier waits for both */
@property(nonatomic, copy) void (^filler)(void);
- (void)waitFence:(id<MTLFence>)f;
- (void)updateFence:(id<MTLFence>)f;
- (void)end;
@property(readonly) BOOL isMetal4;
/* Counts of the encoding of one call (fg.performance_r1: Ncd, Nbar). */
@property(readonly) uint32_t dispatchCount, barrierCount;
/* Profiling (create record field profile = 1, Metal 4 only): a precise timestamp is written after the work encoded so far; the label names the work
   since the previous stamp. Without a heap the call does nothing. */
- (void)profileWithHeap:(id<MTL4CounterHeap>)heap capacity:(uint32_t)cap;
- (void)stamp:(const char *)label;
@property(readonly) uint32_t stampCount;
- (const char *)stampLabel:(uint32_t)i;
@end
