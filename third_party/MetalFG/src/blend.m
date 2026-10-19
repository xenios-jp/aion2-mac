#import <Foundation/Foundation.h>
#include "blend.h"
#include "passes_impl.h"

@implementation MetalFGBlend {
    id<MTLComputePipelineState> _blend, _pack, _fused, _fusedSpec;
    id<MTLLibrary> _lib;
    uint32_t _specS, _specUi;
}

/* f20_blend_pack with the engine's net2 factor and UI flag as function constants (the generic pipeline serves other values, for example the tests) */
- (void)specialiseForS:(uint32_t)s ui:(BOOL)ui {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    const bool u = ui;
    [cv setConstantValue:&s type:MTLDataTypeUInt atIndex:10];
    [cv setConstantValue:&u type:MTLDataTypeBool atIndex:11];
    NSError *e = nil;
    id<MTLFunction> f = [_lib newFunctionWithName:@"f20_blend_pack" constantValues:cv error:&e];
    _fusedSpec = f ? [_lib.device newComputePipelineStateWithFunction:f error:&e] : nil;
    _specS = s; _specUi = ui ? 1 : 0;
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib {
    self = [super init];
    NSError *e = nil;
    _lib = lib;
    MTLFunctionConstantValues *none = [MTLFunctionConstantValues new];       /* the kernels declare function constants; none set: they read the parameters */
    id<MTLFunction> a = [lib newFunctionWithName:@"f20_blend" constantValues:none error:&e];
    id<MTLFunction> b = [lib newFunctionWithName:@"f20_pack_n2" constantValues:none error:&e];
    if (!a || !b) return nil;
    _blend = [dev newComputePipelineStateWithFunction:a error:&e];
    _pack = [dev newComputePipelineStateWithFunction:b error:&e];
    id<MTLFunction> c = [lib newFunctionWithName:@"f20_blend_pack" constantValues:none error:&e];
    if (!c) return nil;
    _fused = [dev newComputePipelineStateWithFunction:c error:&e];
    return (_blend && _pack && _fused) ? self : nil;
}

- (void)blend:(MetalFGEnc *)enc args:(MetalFGF20Args)a {
    struct { uint32_t W, H, Wm, Hm, ui; float tau; } p = {a.W, a.H, a.Wm, a.Hm, a.ui, 0.0f};
    [enc setPipeline:_blend];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.scIp offset:0 at:1];
    [enc bindBuffer:a.scIc offset:0 at:2];
    id<MTLTexture> ui0 = a.uiOld ? a.uiOld : a.colOld, ui1 = a.uiCur ? a.uiCur : a.colCur;
    id<MTLTexture> t[9] = {a.colOld, a.colCur, a.bw, a.err, ui0, ui1, a.candA, a.candB, a.hole};
    for (int i = 0; i < 9; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.W, a.H, 1) threadsPerGroup:metalfg_tg2d(_blend)];
}

- (void)pack:(MetalFGEnc *)enc args:(MetalFGF20Args)a {
    /* fg.blend_pack_pin_r3: the reduction starts from the binary32 blend values, which the kernel computes again from the blend's inputs */
    struct { uint32_t W, H, Wm, Hm, ui; float tau; uint32_t Nw, Nh, s; } p = {a.W, a.H, a.Wm, a.Hm, a.ui, 0.0f, a.Nw, a.Nh, a.s};
    [enc setPipeline:_pack];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.n2a offset:0 at:1];
    [enc bindBuffer:a.n2b offset:0 at:2];
    [enc bindBuffer:a.n2h offset:0 at:3];
    [enc bindBuffer:a.scIp offset:0 at:4];
    [enc bindBuffer:a.scIc offset:0 at:5];
    id<MTLTexture> ui0 = a.uiOld ? a.uiOld : a.colOld, ui1 = a.uiCur ? a.uiCur : a.colCur;
    id<MTLTexture> t[6] = {a.colOld, a.colCur, a.bw, a.err, ui0, ui1};
    for (int i = 0; i < 6; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Nw, a.Nh, 1) threadsPerGroup:metalfg_tg2d(_pack)];
}

/* F20 as one pass: the candidate images, the hole mask and the reduction to the net2 grid (the same values as blend: followed by pack:) */
- (void)blendPack:(MetalFGEnc *)enc args:(MetalFGF20Args)a {
    struct { uint32_t W, H, Wm, Hm, ui; float tau; uint32_t Nw, Nh, s; } p = {a.W, a.H, a.Wm, a.Hm, a.ui, 0.0f, a.Nw, a.Nh, a.s};
    [enc setPipeline:(_fusedSpec && _specS == a.s && _specUi == a.ui) ? _fusedSpec : _fused];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.n2a offset:0 at:1];
    [enc bindBuffer:a.n2b offset:0 at:2];
    [enc bindBuffer:a.n2h offset:0 at:3];
    [enc bindBuffer:a.scIp offset:0 at:4];
    [enc bindBuffer:a.scIc offset:0 at:5];
    id<MTLTexture> ui0 = a.uiOld ? a.uiOld : a.colOld, ui1 = a.uiCur ? a.uiCur : a.colCur;
    id<MTLTexture> t[9] = {a.colOld, a.colCur, a.bw, a.err, ui0, ui1, a.candA, a.candB, a.hole};
    for (int i = 0; i < 9; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    const NSUInteger T = a.s > 16 ? a.s : 16;      /* square tiles of T positions per side; (T / s)^2 blocks of the net2 grid each */
    [enc dispatchGroups:MTLSizeMake(((NSUInteger)a.Nw * a.s + T - 1) / T, ((NSUInteger)a.Nh * a.s + T - 1) / T, 1) threadsPerGroup:MTLSizeMake(T, T, 1)];
}
@end
