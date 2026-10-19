/* Thin Metal 4 plumbing shared by the engine and the tests: pipelines, tracked allocations,
   a per-call constants ring, and a compute-encoder wrapper with MTL4 argument tables. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct { int kind; id obj; size_t offset; const void *bytes; size_t len; } SrArg; /* kind: 0 buffer 1 texture 2 bytes */
#define ARG_BUF(b)        ((SrArg){0, (b), 0, NULL, 0})
#define ARG_BUF_OFF(b, o) ((SrArg){0, (b), (o), NULL, 0})
#define ARG_TEX(t)        ((SrArg){1, (t), 0, NULL, 0})
#define ARG_VAL(v)        ((SrArg){2, nil, 0, &(v), sizeof(v)})

@interface SrGpu : NSObject
@property(readonly) id<MTLDevice> device;
@property(readonly) id<MTL4CommandQueue> queue;
@property(readonly) id<MTL4CommandAllocator> allocator;
@property(readonly) id<MTLResidencySet> residency;
- (instancetype)initWithDevice:(id<MTLDevice>)dev libraryData:(NSData *)data error:(NSString **)err;
- (id<MTLComputePipelineState>)pipeline:(NSString *)name;
- (id<MTLBuffer>)buffer:(size_t)bytes;                       /* shared, zeroed, tracked */
- (id<MTLTexture>)texture:(MTLPixelFormat)fmt w:(int)w h:(int)h;
- (id<MTLTexture>)privateTexture:(MTLPixelFormat)fmt w:(int)w h:(int)h;   /* GPU-only layout */  /* shader read/write, private-ish shared, tracked */
- (void)track:(id<MTLAllocation>)a;                          /* add a caller-owned allocation */
- (void)commitResidency;
- (void)enableArchiveForLibrary:(NSData *)lib fallbackDirectory:(NSString *)fallbackDir;   /* persistent pipeline cache (call before creating pipelines) */
- (id<MTLBuffer>)tanhTable;                       /* tanh of every fp16 bit pattern as correctly rounded fp32 (65536 entries) */
- (void)useLibrary:(id<MTLLibrary>)lib;           /* tests: replace the shader library (drops cached pipelines) */
- (void)saveArchive;                              /* write new pipelines to the cache file */
@end

@interface SrEnc : NSObject
- (instancetype)initWithGpu:(SrGpu *)gpu encoder:(id<MTL4ComputeCommandEncoder>)enc ring:(id<MTLBuffer>)ring;
/* reuse the argument table across calls: point the wrapper at a new encoder and constants buffer */
- (void)attachEncoder:(id<MTL4ComputeCommandEncoder>)enc ring:(id<MTLBuffer>)ring;
@property(readonly) id<MTL4ComputeCommandEncoder> enc;
@property(readonly) int dispatches, barriers;      /* counted since attachEncoder (diagnostics: the structure counts of sr.performance) */
- (void)resetRing;
- (void)dispatch:(id<MTLComputePipelineState>)pso args:(const SrArg *)args count:(int)n
         threadgroups:(MTLSize)tgs threads:(MTLSize)tpt;
/* dispatch covering at least `grid` threads with the given threadgroup size */
- (void)dispatchGrid:(MTLSize)grid pso:(id<MTLComputePipelineState>)pso args:(const SrArg *)args count:(int)n threads:(MTLSize)tpt;
- (void)barrier;
@end

/* Blocking helper for tests: runs `block` inside one encoder on a fresh command buffer and waits. Returns nil or error text. */
NSString *sr_run_sync(SrGpu *gpu, void (^block)(SrEnc *e));
