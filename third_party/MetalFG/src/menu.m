#import <Foundation/Foundation.h>
#include "menu.h"
#include "passes_impl.h"

typedef struct {
    uint32_t gx, gy, cw, ch, ox, oy, channel;
    float mt, frac;
    uint32_t has_flag;
} DetParamsGPU;

@implementation MetalFGMenu {
    id<MTLComputePipelineState> _alpha, _hud, _clear, _over;
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib {
    self = [super init];
    NSError *e = nil;
    NSDictionary *names = @{@"f02_menu_alpha" : @"_alpha", @"f03_menu_hudless" : @"_hud", @"f01_flag_clear" : @"_clear", @"f27_menu_override" : @"_over"};
    for (NSString *n in names) {
        id<MTLFunction> fn = [lib newFunctionWithName:n];
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!p) return nil;
        [self setValue:p forKey:names[n]];
    }
    return self;
}

- (void)runDetect:(MetalFGEnc *)enc pso:(id<MTLComputePipelineState>)pso state:(id<MTLBuffer>)state flag:(id<MTLBuffer>)flag args:(MetalFGDetectArgs)a channel:(uint32_t)ch
          tex0:(id<MTLTexture>)t0 tex1:(id<MTLTexture>)t1 {
    DetParamsGPU p = {a.gx, a.gy, a.cw, a.ch, a.ox, a.oy, ch, a.mt, a.frac, flag ? 1u : 0u};
    [enc setPipeline:pso];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:state offset:0 at:1];
    [enc bindBuffer:(flag ? flag : state) offset:0 at:2];
    [enc bindTexture:t0 at:0];
    if (t1) [enc bindTexture:t1 at:1];
    [enc barrier];
    NSUInteger n = MIN((NSUInteger)1024, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchGroups:MTLSizeMake(1, 1, 1) threadsPerGroup:MTLSizeMake(n, 1, 1)];
}

- (void)detectAlpha:(MetalFGEnc *)enc state:(id<MTLBuffer>)state flag:(id<MTLBuffer>)flag coverage:(id<MTLTexture>)cov channel:(uint32_t)ch args:(MetalFGDetectArgs)a {
    [self runDetect:enc pso:_alpha state:state flag:flag args:a channel:ch tex0:cov tex1:nil];
}

- (void)detectHudless:(MetalFGEnc *)enc state:(id<MTLBuffer>)state flag:(id<MTLBuffer>)flag colour:(id<MTLTexture>)col hudless:(id<MTLTexture>)hud args:(MetalFGDetectArgs)a {
    [self runDetect:enc pso:_hud state:state flag:flag args:a channel:0 tex0:col tex1:hud];
}

- (void)clearFlag:(MetalFGEnc *)enc flag:(id<MTLBuffer>)flag {
    [enc setPipeline:_clear];
    [enc bindBuffer:flag offset:0 at:0];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerGroup:MTLSizeMake(1, 1, 1)];
}

- (void)override:(MetalFGEnc *)enc state:(id<MTLBuffer>)state colour:(id<MTLTexture>)col out:(id<MTLTexture>)out srcOrigin:(const uint32_t *)so dstOrigin:(const uint32_t *)dof size:(const uint32_t *)sz {
    uint32_t p[6] = {so[0], so[1], dof[0], dof[1], sz[0], sz[1]};
    [enc setPipeline:_over];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:state offset:0 at:1];
    [enc bindTexture:col at:0];
    [enc bindTexture:out at:1];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(sz[0], sz[1], 1) threadsPerGroup:metalfg_tg2d(_over)];
}
@end
