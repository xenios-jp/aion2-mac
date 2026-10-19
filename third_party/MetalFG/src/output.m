#import <Foundation/Foundation.h>
#include "output.h"
#include "passes_impl.h"

typedef struct { uint32_t W, H, Nw, Nh, s, ox, oy; } SynthP;
typedef struct { uint32_t Wf, Hf, Wc, Hc; } LevelP;

@implementation MetalFGOutput {
    id<MTLComputePipelineState> _f22, _fused, _pull, _push, _fin, _finList, _listReset, _sig;
    id<MTLBuffer> _tiles, _sig4;
    id<MTLBuffer> _lv;                 /* every level in one buffer */
    NSUInteger _off[16];               /* byte offset of each level */
    id<MTLComputePipelineState> _tail;
    uint32_t _W, _H;
    id<MTLTexture> _holeOut;
    NSMutableArray<id<MTLBuffer>> *_levels;
    NSMutableArray<NSValue *> *_sizes;
}

- (id<MTLTexture>)holeOut { return _holeOut; }
- (NSArray<id<MTLBuffer>> *)levels { return _levels; }
- (NSArray<NSValue *> *)levelSizes { return _sizes; }
- (id<MTLBuffer>)fillTiles { return _tiles; }
- (id<MTLBuffer>)cellSig { return _sig4; }
- (void)prepareCells:(uint32_t)n {
#ifdef METALFG_TEST
    _sig4 = [_fused.device newBufferWithLength:(NSUInteger)n * 4 + 16 options:MTLResourceStorageModeShared];
#else
    _sig4 = [_fused.device newBufferWithLength:(NSUInteger)n * 4 + 16 options:MTLResourceStorageModePrivate];
#endif
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib W:(uint32_t)W H:(uint32_t)H chain:(BOOL)chain levels:(uint32_t)cap {
    self = [super init];
    _W = W; _H = H;
    NSDictionary *names = @{@"f22_synth" : @"_f22", @"f23_fused" : @"_fused",
                            @"f24_pull" : @"_pull", @"f25_push" : @"_push", @"f26_final" : @"_fin", @"f26_final_tiles" : @"_finList",
                            @"f23_tiles_reset" : @"_listReset", @"f23_sig" : @"_sig", @"f24_tail" : @"_tail"};
    NSError *e = nil;
    for (NSString *n in names) {
        id<MTLFunction> fn = [lib newFunctionWithName:n];
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!p) return nil;
        [self setValue:p forKey:names[n]];
    }
    if (!chain) return self;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:W height:H mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
#ifdef METALFG_TEST
    td.storageMode = MTLStorageModeShared;
#else
    td.storageMode = MTLStorageModePrivate;
#endif
    _holeOut = [dev newTextureWithDescriptor:td];
    /* tiles of 16 x 16 pixels holding a pixel to fill: threadgroup counts of the final step (x, 1, 1), then one index per tile */
    _tiles = [dev newBufferWithLength:16 + (NSUInteger)((W + 15) / 16) * ((H + 15) / 16) * 4 options:MTLResourceStorageModePrivate];
    /* levels until one cell remains; deeper levels hold coverage below the binary16 range, so cells are fp32 */
    _sizes = [NSMutableArray array];
    uint32_t w = W, h = H;
    NSUInteger total = 0;
    do {
        w = (w + 1) / 2; h = (h + 1) / 2;
        _off[_sizes.count] = total;
        total += (NSUInteger)w * h * 16;
        [_sizes addObject:[NSValue valueWithSize:NSMakeSize(w, h)]];
    } while ((w > 1 || h > 1) && _sizes.count < cap && _sizes.count < 16);   // the planner's count caps the pyramid (only the control builds lower it)
#ifdef METALFG_TEST
    _lv = [dev newBufferWithLength:total options:MTLResourceStorageModeShared];
#else
    _lv = [dev newBufferWithLength:total options:MTLResourceStorageModePrivate];
#endif
    _levels = [NSMutableArray array];
    for (NSUInteger k = 0; k < _sizes.count; k++) [_levels addObject:_lv];   /* one entry per level (the buffer holding it) */
    return self;
}

- (void)synth:(MetalFGEnc *)enc candA:(id<MTLTexture>)a candB:(id<MTLTexture>)b hole:(id<MTLTexture>)h n2out:(id<MTLBuffer>)n
          out:(id<MTLTexture>)out origin:(const uint32_t *)o Nw:(uint32_t)Nw Nh:(uint32_t)Nh s:(uint32_t)s {
    SynthP p = {_W, _H, Nw, Nh, s, o[0], o[1]};
    [enc setPipeline:_f22];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:n offset:0 at:1];
    [enc bindTexture:a at:0]; [enc bindTexture:b at:1]; [enc bindTexture:h at:2]; [enc bindTexture:out at:3];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(_W, _H, 1) threadsPerGroup:metalfg_tg2d(_f22)];
}

- (void)direct:(MetalFGEnc *)enc n2out:(id<MTLBuffer>)n candA:(id<MTLTexture>)a candB:(id<MTLTexture>)b hole:(id<MTLTexture>)h out:(id<MTLTexture>)out
        origin:(const uint32_t *)o Nw:(uint32_t)Nw Nh:(uint32_t)Nh s:(uint32_t)s {
    [self synth:enc candA:a candB:b hole:h n2out:n out:out origin:o Nw:Nw Nh:Nh s:s];
}

- (void)chain:(MetalFGEnc *)enc n2out:(id<MTLBuffer>)n candA:(id<MTLTexture>)a candB:(id<MTLTexture>)b hole:(id<MTLTexture>)h out:(id<MTLTexture>)out
       origin:(const uint32_t *)o Nw:(uint32_t)Nw Nh:(uint32_t)Nh s:(uint32_t)s {
    SynthP p = {_W, _H, Nw, Nh, s, o[0], o[1]};
    NSUInteger L = _levels.count;
    const uint32_t have2 = L >= 2;
    NSAssert(_sig4 && _sig4.length >= (NSUInteger)Nw * Nh * 4, @"prepareCells: before the chain");
    [enc setPipeline:_sig];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:n offset:0 at:1];
    [enc bindBuffer:_sig4 offset:0 at:2];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)Nw * Nh, 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc setPipeline:_listReset];                                          /* no barrier: nothing before reads or writes the tile list in this call */
    [enc bindBuffer:_tiles offset:0 at:0];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerGroup:MTLSizeMake(1, 1, 1)];
    [enc setPipeline:_fused];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:n offset:0 at:1];
    [enc bindBuffer:_lv offset:_off[0] at:2];
    [enc bindBuffer:_lv offset:_off[have2 ? 1 : 0] at:3];
    [enc setBytes:&have2 length:sizeof have2 at:4];
    [enc bindTexture:a at:0]; [enc bindTexture:b at:1]; [enc bindTexture:h at:2]; [enc bindTexture:out at:3]; [enc bindTexture:_holeOut at:4];
    [enc bindBuffer:_tiles offset:0 at:5];
    [enc bindBuffer:_sig4 offset:0 at:6];
    [enc barrier];
    [enc dispatchGroups:MTLSizeMake((_W + 15) / 16, (_H + 15) / 16, 1) threadsPerGroup:MTLSizeMake(16, 16, 1)];
    /* array index k holds level k + 1. Pull targets k = 2 .. L - 1, push targets k = L - 2 .. 1. Targets from kt on (each at most 1024 cells, at most 8 of them)
       run in one threadgroup (f24_tail: the pulls up, then the pushes back down to kt); the larger ones keep a dispatch each */
    NSUInteger kt = L;
    while (kt > 2 && L - (kt - 1) <= 8) {
        NSSize c = _sizes[kt - 1].sizeValue;
        if ((NSUInteger)c.width * (NSUInteger)c.height > 1024) break;
        kt--;
    }
    for (NSUInteger k = 2; k < kt; k++) {                                  /* pull: level k + 1 from level k (levels 1 and 2 come from the fused kernel) */
        NSSize f = _sizes[k - 1].sizeValue, c = _sizes[k].sizeValue;
        LevelP lp = {(uint32_t)f.width, (uint32_t)f.height, (uint32_t)c.width, (uint32_t)c.height};
        [enc setPipeline:_pull];
        [enc setBytes:&lp length:sizeof lp at:0];
        [enc bindBuffer:_lv offset:_off[k - 1] at:1];
        [enc bindBuffer:_lv offset:_off[k] at:2];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(lp.Wc, lp.Hc, 1) threadsPerGroup:metalfg_tg2d(_pull)];
    }
    if (kt < L) {
        struct { struct { LevelP p; uint32_t offF, offC; } pull[8], push[8]; uint32_t npull, npush; } T;
        memset(&T, 0, sizeof T);
        for (NSUInteger k = kt; k < L; k++) {
            NSSize f = _sizes[k - 1].sizeValue, c = _sizes[k].sizeValue;
            T.pull[T.npull].p = (LevelP){(uint32_t)f.width, (uint32_t)f.height, (uint32_t)c.width, (uint32_t)c.height};
            T.pull[T.npull].offF = (uint32_t)(_off[k - 1] / 16); T.pull[T.npull].offC = (uint32_t)(_off[k] / 16);
            T.npull++;
        }
        for (NSInteger k = (NSInteger)L - 2; k >= (NSInteger)kt; k--) {
            NSSize f = _sizes[k].sizeValue, c = _sizes[k + 1].sizeValue;
            T.push[T.npush].p = (LevelP){(uint32_t)f.width, (uint32_t)f.height, (uint32_t)c.width, (uint32_t)c.height};
            T.push[T.npush].offF = (uint32_t)(_off[k] / 16); T.push[T.npush].offC = (uint32_t)(_off[k + 1] / 16);
            T.npush++;
        }
        [enc setPipeline:_tail];
        [enc setBytes:&T length:sizeof T at:0];
        [enc bindBuffer:_lv offset:0 at:1];
        [enc barrier];
        const NSUInteger nth = MIN((NSUInteger)1024, _tail.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(nth, 1, 1) threadsPerGroup:MTLSizeMake(nth, 1, 1)];
    }
    for (NSInteger k = MIN((NSInteger)kt - 1, (NSInteger)L - 2); k >= 1; k--) {   /* push: level k + 1 into level k, coarse to fine; level 1 is pushed inside the final step */
        NSSize f = _sizes[k].sizeValue, c = _sizes[k + 1].sizeValue;
        LevelP lp = {(uint32_t)f.width, (uint32_t)f.height, (uint32_t)c.width, (uint32_t)c.height};
        [enc setPipeline:_push];
        [enc setBytes:&lp length:sizeof lp at:0];
        [enc bindBuffer:_lv offset:_off[k] at:1];
        [enc bindBuffer:_lv offset:_off[k + 1] at:2];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(lp.Wf, lp.Hf, 1) threadsPerGroup:metalfg_tg2d(_push)];
    }
    /* final step on the listed tiles only (the pixels of a tile are written by its own threadgroup, so the order of the list does not matter) */
    [enc setPipeline:_finList];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:_lv offset:_off[0] at:1];
    [enc bindBuffer:_lv offset:_off[have2 ? 1 : 0] at:2];
    [enc setBytes:&have2 length:sizeof have2 at:3];
    [enc bindBuffer:_tiles offset:0 at:4];
    [enc bindTexture:_holeOut at:0]; [enc bindTexture:out at:1];
    [enc barrier];
    [enc dispatchGroupsIndirect:_tiles offset:0 threadsPerGroup:MTLSizeMake(16, 16, 1)];
}
@end
