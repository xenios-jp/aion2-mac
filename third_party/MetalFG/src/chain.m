#import <Foundation/Foundation.h>
#include "chain.h"
#include "passes_impl.h"

/* parameter blocks: layouts match the structs of the shaders */
typedef struct {
    uint32_t Wm, Hm, mode;
    float Wz, Hz, H, d_dyn, d_sim, th_cam, eps_c;
    MetalFGDepthConst D;
    float c2p[16], p2c[16];
} P2CBlock;

typedef struct {
    uint32_t srcW, srcH, dstW, dstH, srcOff, dstOff, upOff, upW, upH, last, level;
    float eps_c;
} PyrBlock;

typedef struct {
    uint32_t Wm, Hm, mode, use_zn;
    float hscale[2];
    float H, d_sim, th_zn, Z_sep;
    int32_t sp_min, sp_max;
    MetalFGDepthConst D;
    uint32_t y0;
} F18Block;

static const float EPS_C = 0x1p-34f;

@implementation MetalFGChain {
    id<MTLComputePipelineState> _f04full, _f04reduce, _f04fused, _f04tiled, _f13, _f14, _f07, _f08v, _f08h, _pyr1, _pyrd, _pyrf, _pyrt, _pyrfin, _f18, _f19, _f19feat;
    id<MTLBuffer> _pyrS, _pyrF;
    uint32_t _levels, _lw[16], _lh[16], _loff[16];
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib Wm:(uint32_t)Wm Hm:(uint32_t)Hm {
    return [self initWithDevice:dev library:lib Wm:Wm Hm:Hm levels:13];
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib Wm:(uint32_t)Wm Hm:(uint32_t)Hm levels:(uint32_t)cap {
    self = [super init];
    NSDictionary *names = @{
        @"f04_full" : @"_f04full", @"f04_reduce" : @"_f04reduce", @"f04_fused" : @"_f04fused", @"f04_tiled" : @"_f04tiled", @"f13_scatter" : @"_f13", @"f14_seed" : @"_f14",
        @"f07_scatter" : @"_f07", @"f08_values" : @"_f08v", @"f08_hole" : @"_f08h", @"pyr_level1" : @"_pyr1", @"pyr_down" : @"_pyrd",
        @"pyr_fill" : @"_pyrf", @"pyr_tail" : @"_pyrt", @"pyr_final" : @"_pyrfin", @"f18_scatter" : @"_f18", @"f19_weights" : @"_f19"
    };
    for (NSString *n in names) {
        id<MTLFunction> fn = [lib newFunctionWithName:n];
        NSError *e = nil;
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!p) return nil;
        [self setValue:p forKey:names[n]];
    }
#if defined(METALFG_CTL_FEAT_WEIGHTS)
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"f19_weights_feat"];
        NSError *e2 = nil;
        _f19feat = fn ? [dev newComputePipelineStateWithFunction:fn error:&e2] : nil;
        if (!_f19feat) return nil;
    }
#endif
    /* pyramid levels 1..K of the coverage fill, kept in fp32 */
    _lw[0] = Wm; _lh[0] = Hm; _loff[0] = 0;
    uint32_t k = 0, total = 0;
    while (_lw[k] > 1 || _lh[k] > 1) {
        k++;
        _lw[k] = (_lw[k - 1] + 1) / 2; _lh[k] = (_lh[k - 1] + 1) / 2;
        _loff[k] = total;
        total += _lw[k] * _lh[k];
    }
    _levels = k < cap ? k : cap;      // the planner's level count caps the pyramid (only the control builds lower it)
    _pyrS = [dev newBufferWithLength:(total + 1) * 16 options:MTLResourceStorageModePrivate];
    _pyrF = [dev newBufferWithLength:(total + 1) * 16 options:MTLResourceStorageModePrivate];
    return self;
}

- (NSArray<id<MTLAllocation>> *)allocations { return @[ _pyrS, _pyrF ]; }

- (void)f04:(MetalFGEnc *)enc args:(MetalFGF04Args)a colour:(id<MTLTexture>)c hudless:(id<MTLTexture>)h coverage:(id<MTLTexture>)cov
     colOld:(id<MTLTexture>)old colCur:(id<MTLTexture>)cur uiCur:(id<MTLTexture>)ui colFeat:(id<MTLTexture>)cf uim:(id<MTLTexture>)uim
         dY:(id<MTLTexture>)dY {
    /* tiled and fused (colour read once, adjacent lanes on adjacent texels) when the blocks cover the display; with the former one-thread-per-block mapping it was slower with a hudless or coverage input at df = 4 */
    if ((NSUInteger)cf.width * a.df >= a.W && (NSUInteger)cf.height * a.df >= a.H) {
        [enc setPipeline:_f04tiled];
        [enc setBytes:&a length:sizeof a at:0];
        [enc bindTexture:c at:0]; [enc bindTexture:h at:1]; [enc bindTexture:cov at:2]; [enc bindTexture:old at:3];
        [enc bindTexture:cf at:4]; [enc bindTexture:uim at:5]; [enc bindTexture:dY at:6];
        [enc bindTexture:cur at:7]; [enc bindTexture:(ui ? ui : cur) at:8];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(cf.width * a.df, cf.height * a.df, 1) threadsPerGroup:MTLSizeMake(16, 16, 1)];
        return;
    }
    [enc setPipeline:_f04full];
    [enc setBytes:&a length:sizeof a at:0];
    [enc bindTexture:c at:0]; [enc bindTexture:h at:1]; [enc bindTexture:cov at:2];
    [enc bindTexture:cur at:3]; [enc bindTexture:(ui ? ui : cur) at:4];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.W, a.H, 1) threadsPerGroup:metalfg_tg2d(_f04full)];
    [enc setPipeline:_f04reduce];
    [enc setBytes:&a length:sizeof a at:0];
    [enc bindTexture:c at:0]; [enc bindTexture:h at:1]; [enc bindTexture:cov at:2]; [enc bindTexture:old at:3];
    [enc bindTexture:cf at:4]; [enc bindTexture:uim at:5]; [enc bindTexture:dY at:6];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(cf.width, cf.height, 1) threadsPerGroup:metalfg_tg2d(_f04reduce)];
}

- (void)f13:(MetalFGEnc *)enc field:(id<MTLTexture>)g out:(id<MTLTexture>)o normalised:(BOOL)norm Wm:(uint32_t)Wm Hm:(uint32_t)Hm {
    uint32_t p[5] = {(uint32_t)o.width, (uint32_t)o.height, Wm, Hm, norm ? 1u : 0u};
    [enc setPipeline:_f13];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindTexture:g at:0]; [enc bindTexture:o at:1];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(o.width, o.height, 1) threadsPerGroup:metalfg_tg2d(_f13)];
}

- (void)f14:(MetalFGEnc *)enc colFeat:(id<MTLTexture>)cf err:(id<MTLTexture>)err {
    [enc setPipeline:_f14];
    [enc bindTexture:cf at:0]; [enc bindTexture:err at:1];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(err.width, err.height, 1) threadsPerGroup:metalfg_tg2d(_f14)];
}

- (void)p2c:(MetalFGEnc *)enc args:(MetalFGP2CArgs)a sc:(id<MTLBuffer>)sc hb:(id<MTLBuffer>)hb depCur:(id<MTLTexture>)depCur
     depOld:(id<MTLTexture>)depOld dynCur:(id<MTLTexture>)dynCur dynOld:(id<MTLTexture>)dynOld mvo:(id<MTLTexture>)mvo
        mvf:(id<MTLTexture>)mvf nsc:(id<MTLTexture>)nsc {
    P2CBlock b;
    memset(&b, 0, sizeof b);
    b.Wm = a.Wm; b.Hm = a.Hm; b.Wz = a.depthW; b.Hz = a.depthH; b.H = a.H;
    b.d_dyn = a.d_dyn; b.d_sim = a.d_sim; b.th_cam = a.th_cam; b.eps_c = EPS_C; b.D = a.D;
    memcpy(b.c2p, a.c2p, sizeof b.c2p); memcpy(b.p2c, a.p2c, sizeof b.p2c);
    {                                                /* F07: one pass (64-bit atomic minimum of key and payload) */
        [enc setPipeline:_f07];
        [enc setBytes:&b length:sizeof b at:0];
        [enc bindBuffer:sc offset:0 at:1];
        [enc bindTexture:depCur at:0]; [enc bindTexture:mvo at:1]; [enc bindTexture:dynCur at:2];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:metalfg_tg2d(_f07)];
    }
    uint32_t wn = a.writeNsc;
    [enc setPipeline:_f08v];                         /* F08: values */
    [enc setBytes:&b length:sizeof b at:0];
    [enc bindBuffer:sc offset:0 at:1]; [enc bindBuffer:hb offset:0 at:2];
    [enc setBytes:&wn length:sizeof wn at:3];
    [enc bindTexture:depOld at:0]; [enc bindTexture:dynOld at:1]; [enc bindTexture:mvo at:2];
    [enc bindTexture:mvf at:3]; [enc bindTexture:(nsc ? nsc : mvf) at:4];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:metalfg_tg2d(_f08v)];
    [enc setPipeline:_f08h];                         /* closing, weights (tiles of 16 x 16 cells) */
    [enc setBytes:&b length:sizeof b at:0];
    [enc bindBuffer:hb offset:0 at:2];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:MTLSizeMake(16, 16, 1)];

    /* pyramid of block sums, levels 1..K */
    /* the small top levels (first..K, at most 8, each of at most 1024 elements) run in one threadgroup */
    uint32_t first = _levels + 1;
    while (first > 2 && first - 1 >= 2 && _levels - (first - 1) < 8 && _lw[first - 1] * _lh[first - 1] <= 1024) first--;
    for (uint32_t k = 1; k < first; k++) {
        PyrBlock q = {_lw[k - 1], _lh[k - 1], _lw[k], _lh[k], _loff[k - 1], _loff[k], 0, 0, 0, 0, k, EPS_C};
        id<MTLComputePipelineState> pso = k == 1 ? _pyr1 : _pyrd;
        [enc setPipeline:pso];
        [enc setBytes:&q length:sizeof q at:0];
        [enc bindBuffer:hb offset:0 at:2]; [enc bindBuffer:_pyrS offset:0 at:4];
        if (k == 1) [enc bindTexture:mvf at:0];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(q.dstW, q.dstH, 1) threadsPerGroup:metalfg_tg2d(pso)];
    }
    if (first <= _levels) {
        struct { PyrBlock down[8], fill[8]; uint32_t n; } T;
        memset(&T, 0, sizeof T);
        T.n = _levels - first + 1;
        for (uint32_t k = first; k <= _levels; k++) {
            T.down[k - first] = (PyrBlock){_lw[k - 1], _lh[k - 1], _lw[k], _lh[k], _loff[k - 1], _loff[k], 0, 0, 0, 0, k, EPS_C};
            T.fill[k - first] = (PyrBlock){0, 0, _lw[k], _lh[k], 0, _loff[k], k < _levels ? _loff[k + 1] : 0, k < _levels ? _lw[k + 1] : 0,
                                           k < _levels ? _lh[k + 1] : 0, k == _levels ? 1u : 0u, k, EPS_C};
        }
        [enc setPipeline:_pyrt];
        [enc setBytes:&T length:sizeof T at:0];
        [enc bindBuffer:_pyrS offset:0 at:4]; [enc bindBuffer:_pyrF offset:0 at:5];
        [enc barrier];
        { const NSUInteger nth = MIN((NSUInteger)1024, _pyrt.maxTotalThreadsPerThreadgroup); [enc dispatchThreads:MTLSizeMake(nth, 1, 1) threadsPerGroup:MTLSizeMake(nth, 1, 1)]; }
    }
    /* filled levels below the fused ones, down to 1 */
    for (uint32_t k = first - 1; k >= 1; k--) {
        PyrBlock q = {0, 0, _lw[k], _lh[k], 0, _loff[k], k < _levels ? _loff[k + 1] : 0, k < _levels ? _lw[k + 1] : 0,
                      k < _levels ? _lh[k + 1] : 0, k == _levels ? 1u : 0u, k, EPS_C};
        [enc setPipeline:_pyrf];
        [enc setBytes:&q length:sizeof q at:0];
        [enc bindBuffer:_pyrS offset:0 at:4]; [enc bindBuffer:_pyrF offset:0 at:5];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(q.dstW, q.dstH, 1) threadsPerGroup:metalfg_tg2d(_pyrf)];
    }
    /* level 0: holes take the upsampled pair of level 1 */
    PyrBlock q = {a.Wm, a.Hm, 0, 0, 0, 0, _levels >= 1 ? _loff[1] : 0, _levels >= 1 ? _lw[1] : 0, _levels >= 1 ? _lh[1] : 0, 0, 0, EPS_C};
    uint32_t have = _levels >= 1 ? 1u : 0u;
    [enc setPipeline:_pyrfin];
    [enc setBytes:&q length:sizeof q at:0];
    [enc bindBuffer:hb offset:0 at:2]; [enc bindBuffer:_pyrF offset:0 at:5];
    [enc setBytes:&have length:sizeof have at:6];
    [enc bindTexture:mvf at:0];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:metalfg_tg2d(_pyrfin)];
}

- (void)f18:(MetalFGEnc *)enc args:(MetalFGF18Args)a field:(id<MTLTexture>)field hscale:(const float *)hscale dep:(id<MTLTexture>)dep
         sc:(id<MTLBuffer>)sc useZn:(BOOL)useZn {
    [self f18:enc args:a field:field hscale:hscale dep:dep sc:sc useZn:useZn rows:0 to:a.Hm];
}

- (void)f18:(MetalFGEnc *)enc args:(MetalFGF18Args)a field:(id<MTLTexture>)field hscale:(const float *)hscale dep:(id<MTLTexture>)dep
         sc:(id<MTLBuffer>)sc useZn:(BOOL)useZn rows:(uint32_t)y0 to:(uint32_t)y1 {
    if (y1 <= y0) return;
    F18Block b = {a.Wm, a.Hm, 0, useZn ? 1u : 0u, {hscale[0], hscale[1]}, a.H, a.d_sim, a.th_zn, a.Z_sep, a.sp_min, a.sp_max, a.D, y0};
#if defined(METALFG_CTL_KEY_RECOMPUTE)
    for (uint32_t mode = 0; mode < 2; mode++) {      /* control: the two-pass form (key pass, payload pass) */
#else
    for (uint32_t mode = 1; mode < 2; mode++) {      /* one pass: 64-bit atomic minimum of (key, payload) */
#endif
        b.mode = mode;
        [enc setPipeline:_f18];
        [enc setBytes:&b length:sizeof b at:0];
        [enc bindBuffer:sc offset:0 at:1];
        [enc bindTexture:field at:0]; [enc bindTexture:dep at:1];
        [enc barrier];
        [enc dispatchThreads:MTLSizeMake(a.Wm, y1 - y0, 1) threadsPerGroup:metalfg_tg2d(_f18)];
    }
}

- (void)f19:(MetalFGEnc *)enc args:(MetalFGF18Args)a scP:(id<MTLBuffer>)scp scC:(id<MTLBuffer>)scc lgP:(id<MTLTexture>)lgP
       lgC:(id<MTLTexture>)lgC bw:(id<MTLTexture>)bw {
    uint32_t p[4] = {a.Wm, a.Hm, a.Wf, a.Hf};
    [enc setPipeline:_f19];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:scp offset:0 at:1]; [enc bindBuffer:scc offset:0 at:2];
    [enc bindTexture:lgP at:0]; [enc bindTexture:lgC at:1]; [enc bindTexture:bw at:2];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:metalfg_tg2d(_f19)];
}

#if defined(METALFG_CTL_FEAT_WEIGHTS)
- (void)f19feat:(MetalFGEnc *)enc args:(MetalFGF18Args)a Wm:(uint32_t)Wm Hm:(uint32_t)Hm scP:(id<MTLBuffer>)scp scC:(id<MTLBuffer>)scc lgP:(id<MTLTexture>)lgP lgC:(id<MTLTexture>)lgC bw:(id<MTLTexture>)bw {
    uint32_t p[4] = {a.Wm, a.Hm, a.Wf, a.Hf};
    [enc setPipeline:_f19feat];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:scp offset:0 at:1]; [enc bindBuffer:scc offset:0 at:2];
    [enc bindTexture:lgP at:0]; [enc bindTexture:lgC at:1]; [enc bindTexture:bw at:2];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(bw.width, bw.height, 1) threadsPerGroup:metalfg_tg2d(_f19feat)];
}
#endif
@end
