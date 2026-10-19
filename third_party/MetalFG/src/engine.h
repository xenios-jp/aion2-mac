#pragma once
/* Internal engine object (specs fg.engine_api, fg.resources, fg.orchestration). */
#import <Metal/Metal.h>
#include "call.h"
#include "enc.h"
#include "metalfg.h"
#include "passes.h"
#include "plan.h"
#include "weights.h"
#include "nn.h"
#include "net1io.h"
#include "menu.h"
#include "motionprep.h"
#include "blend.h"
#include "chain.h"
#include "output.h"

/* One named resource of fg.resources: a texture or buffer, with a second side for history pairs. */
#define METALFG_RING_SLOTS 64
#define METALFG_RING_SLOT 65536

@interface MetalFGSlot : NSObject
@property(nonatomic) MetalFGResDesc desc;
@property(nonatomic, strong) id<MTLTexture> tex0, tex1;
@property(nonatomic, strong) id<MTLBuffer> buf0, buf1;
@property(nonatomic) int cur; /* which side is "cur" (pairs only) */
@end

@interface MetalFGEngineImpl : NSObject {
  @public
    MetalFGCreateInfo info;
    MetalFGParams params;
    MetalFGGrids grids;
    MetalFGCallState state;
    int prevHadField;
    uint64_t callCounter;
    float camMats[32];      /* current-to-previous and previous-to-current clip matrices of this call (row-vector form) */
    float prevV[16], prevP[16];
    /* counts of the newest encoded call (fg.performance_r1 Ncd, Nbar) and the profile of it (create field profile = 1, Metal 4) */
    uint32_t lastDispatches, lastBarriers;
    uint32_t profStamps;
    const char *profLabels[512];
    int profHeap;
}
@property(nonatomic, strong) NSArray<id<MTL4CounterHeap>> *profHeaps;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pso;
@property(nonatomic, strong) NSMutableDictionary<NSString *, MetalFGSlot *> *slots;
@property(nonatomic, strong) id<MTLBuffer> weights;
@property(nonatomic, strong) MetalFGNN *nn;
@property(nonatomic, strong) MetalFGNet1 *net1;
@property(nonatomic, strong) MetalFGNet2 *net2;
@property(nonatomic, strong) MetalFGIO *io;
@property(nonatomic, strong) MetalFGMenu *menu;
@property(nonatomic, strong) MetalFGMotionPrep *motion;
@property(nonatomic, strong) MetalFGBlend *blend;
@property(nonatomic, strong) MetalFGChain *chain;
#if defined(METALFG_CTL_FEAT_WEIGHTS)
@property(nonatomic, strong) id<MTLTexture> ctlBwFeat;   // control: blend weights on the Feat grid
#endif
@property(nonatomic, strong) MetalFGOutput *output;
@property(nonatomic, strong) id<MTL4ArgumentTable> argTable;
@property(nonatomic, strong) id<MTLBuffer> ring;
@property(nonatomic, strong) id<MTLResidencySet> residency;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic, strong) NSMutableArray<id<MTLResidencySet>> *callerSets;
@property(nonatomic, strong) NSMutableArray<NSData *> *callerKeys;   /* the caller objects of each set in callerSets */

- (instancetype)initWithDevice:(id<MTLDevice>)dev info:(const MetalFGCreateInfo *)info err:(char *)err cap:(size_t)cap;
- (int)encodeClassic:(id<MTLCommandBuffer>)cb frame:(const MetalFGFrameInfo *)f;
- (int)encodeMetal4:(id<MTL4CommandBuffer>)cb frame:(const MetalFGFrameInfo *)f;
- (id<MTLTexture>)tex:(NSString *)name old:(BOOL)old;
- (id<MTLBuffer>)buf:(NSString *)name old:(BOOL)old;
@end

@interface MetalFGEngineImpl (Run)
- (void)runCall:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p;
- (void)rotate;
@end

#ifdef METALFG_TEST
/* Test hook: copies the contents of a resource (cur or old side) after the caller's command buffer completed. */
size_t metalfg_probe(MetalFGEngineRef e, const char *name, int old_side, void *dst, size_t cap);
#endif
