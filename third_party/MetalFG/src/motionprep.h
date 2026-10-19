#pragma once
/* F05 motion preparation, F06 distortion field downsample, F12 distortion correction (spec fg.motion_prep). */
#import <Metal/Metal.h>
#include "enc.h"

typedef struct {
    uint32_t Wm, Hm;
    uint32_t depth_origin[2], depth_size[2];
    uint32_t motion_origin[2], motion_size[2];
    float scale_x, scale_y;
    uint32_t reversed;
    float eps_n;
} MetalFGF05Args;

@interface MetalFGMotionPrep : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib;
/* depth: a colour-format or a depth-format texture */
- (void)f05:(MetalFGEnc *)enc depth:(id<MTLTexture>)depth motion:(id<MTLTexture>)motion dep:(id<MTLTexture>)depOut mvo:(id<MTLTexture>)mvoOut args:(MetalFGF05Args)a;
- (void)f06:(MetalFGEnc *)enc field:(id<MTLTexture>)field origin:(const uint32_t *)o size:(const uint32_t *)s out:(id<MTLTexture>)out Wm:(uint32_t)Wm Hm:(uint32_t)Hm;
- (void)f12:(MetalFGEnc *)enc bdfCur:(id<MTLTexture>)b1 bdfOld:(id<MTLTexture>)b0 mvo:(id<MTLTexture>)mvo mvf:(id<MTLTexture>)mvf dep:(id<MTLTexture>)dep
      mvod:(id<MTLTexture>)mvod mvfd:(id<MTLTexture>)mvfd dep2:(id<MTLTexture>)dep2 Wm:(uint32_t)Wm Hm:(uint32_t)Hm;
@end
