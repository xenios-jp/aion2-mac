#import <Foundation/Foundation.h>
#include "net1io.h"
#include "passes_impl.h"

@implementation MetalFGIO {
    id<MTLComputePipelineState> _f15, _f17;
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib {
    self = [super init];
    NSError *e = nil;
    id<MTLFunction> a = [lib newFunctionWithName:@"f15_pack"];
    id<MTLFunction> b = [lib newFunctionWithName:@"f17_unpack"];
    if (!a || !b) return nil;
    _f15 = [dev newComputePipelineStateWithFunction:a error:&e];
    _f17 = [dev newComputePipelineStateWithFunction:b error:&e];
    return (_f15 && _f17) ? self : nil;
}

- (void)f15:(MetalFGEnc *)enc args:(MetalFGF15Args)a {
    struct { uint32_t Wf, Hf, w1, h1, ox, oy, Wm, Hm; } p = {a.Wf, a.Hf, a.w1, a.h1, a.ox, a.oy, a.Wm, a.Hm};
    [enc setPipeline:_f15];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.n1in offset:0 at:1];
    id<MTLTexture> t[12] = {a.colOld, a.colCur, a.dY, a.errOld, a.ns, a.dis, a.disOld, a.st, a.mcp, a.mpc, a.errOut, a.msOut};
    for (int i = 0; i < 12; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(2 * a.w1, 2 * a.h1, 1) threadsPerGroup:metalfg_tg2d(_f15)];
}

- (void)f17:(MetalFGEnc *)enc args:(MetalFGF17Args)a {
    struct { uint32_t Wf, Hf, w1, h1, ox, oy, steering; float b, th; int drift; } p = {a.Wf, a.Hf, a.w1, a.h1, a.ox, a.oy, a.steering, a.b_steer, a.th_mask, 0};
#if defined(METALFG_CTL_DRIFT)
    /* control of fg.arithmetic_tolerance_r2 tol-drift-ctl: METALFG_DRIFT_K = k ladder units 2^-11 on every call (ladder), METALFG_DRIFT_RAMP = 1 adds c units at the c-th call of the process (ramp) */
    static int ncall;
    const char *vk = getenv("METALFG_DRIFT_K"), *vr = getenv("METALFG_DRIFT_RAMP");
    p.drift = vr && atoi(vr) ? ncall : (vk ? atoi(vk) : 0);
    ncall++;
#endif
    [enc setPipeline:_f17];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.n1out offset:0 at:1];
    id<MTLTexture> t[6] = {a.ms, a.uimOld, a.uimCur, a.lgP, a.lgC, a.st};
    for (int i = 0; i < 6; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wf, a.Hf, 1) threadsPerGroup:metalfg_tg2d(_f17)];
}
@end
