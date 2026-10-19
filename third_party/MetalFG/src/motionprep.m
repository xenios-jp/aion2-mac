#import <Foundation/Foundation.h>
#include "motionprep.h"
#include "passes_impl.h"

typedef struct {
    uint32_t Wm, Hm, dox, doy, dw, dh, mox, moy, mw, mh;
    float sx, sy, mtw, mth;
    uint32_t reversed;
    float eps_n;
} F05P;

@implementation MetalFGMotionPrep {
    id<MTLComputePipelineState> _f05, _f05d, _f06, _f12;
}

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib {
    self = [super init];
    NSError *e = nil;
    NSDictionary *names = @{@"f05_motion_prep" : @"_f05", @"f05_motion_prep_depth" : @"_f05d", @"f06_distortion" : @"_f06", @"f12_distortion_correction" : @"_f12"};
    for (NSString *n in names) {
        id<MTLFunction> fn = [lib newFunctionWithName:n];
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!p) return nil;
        [self setValue:p forKey:names[n]];
    }
    return self;
}

static BOOL is_depth_format(MTLPixelFormat f) {
    return f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth16Unorm;
}

- (void)f05:(MetalFGEnc *)enc depth:(id<MTLTexture>)depth motion:(id<MTLTexture>)motion dep:(id<MTLTexture>)depOut mvo:(id<MTLTexture>)mvoOut args:(MetalFGF05Args)a {
    F05P p = {a.Wm, a.Hm, a.depth_origin[0], a.depth_origin[1], a.depth_size[0], a.depth_size[1],
              a.motion_origin[0], a.motion_origin[1], a.motion_size[0], a.motion_size[1], a.scale_x, a.scale_y,
              (float)motion.width, (float)motion.height, a.reversed, a.eps_n};
    id<MTLComputePipelineState> pso = is_depth_format(depth.pixelFormat) ? _f05d : _f05;
    [enc setPipeline:pso];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindTexture:depth at:0];
    [enc bindTexture:motion at:1];
    [enc bindTexture:depOut at:2];
    [enc bindTexture:mvoOut at:3];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(a.Wm, a.Hm, 1) threadsPerGroup:metalfg_tg2d(pso)];
}

- (void)f06:(MetalFGEnc *)enc field:(id<MTLTexture>)field origin:(const uint32_t *)o size:(const uint32_t *)s out:(id<MTLTexture>)out Wm:(uint32_t)Wm Hm:(uint32_t)Hm {
    MTLPixelFormat f = field.pixelFormat;
    uint32_t chans = (f == MTLPixelFormatRG16Float || f == MTLPixelFormatRG32Float) ? 2 : 4;
    struct { uint32_t Wm, Hm, ox, oy, w, h; float tw, th; uint32_t chans; } p = {Wm, Hm, o[0], o[1], s[0], s[1], (float)field.width, (float)field.height, chans};
    [enc setPipeline:_f06];
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindTexture:field at:0];
    [enc bindTexture:out at:1];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(Wm, Hm, 1) threadsPerGroup:metalfg_tg2d(_f06)];
}

- (void)f12:(MetalFGEnc *)enc bdfCur:(id<MTLTexture>)b1 bdfOld:(id<MTLTexture>)b0 mvo:(id<MTLTexture>)mvo mvf:(id<MTLTexture>)mvf dep:(id<MTLTexture>)dep
      mvod:(id<MTLTexture>)mvod mvfd:(id<MTLTexture>)mvfd dep2:(id<MTLTexture>)dep2 Wm:(uint32_t)Wm Hm:(uint32_t)Hm {
    uint32_t p[2] = {Wm, Hm};
    [enc setPipeline:_f12];
    [enc setBytes:p length:sizeof p at:0];
    id<MTLTexture> t[8] = {b1, b0, mvo, mvf, dep, mvod, mvfd, dep2};
    for (int i = 0; i < 8; i++) [enc bindTexture:t[i] at:i];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(Wm, Hm, 1) threadsPerGroup:metalfg_tg2d(_f12)];
}
@end
