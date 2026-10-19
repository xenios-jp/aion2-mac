#pragma once
/* F01 flag clear, F02/F03 menu detection, F27 override (spec fg.menu_detection). */
#import <Metal/Metal.h>
#include "enc.h"

typedef struct {
    uint32_t gx, gy, cw, ch, ox, oy;
    float mt, frac;
} MetalFGDetectArgs;

@interface MetalFGMenu : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib;
/* state: 16-byte menu.state buffer; flag: optional host buffer (first byte) */
- (void)detectAlpha:(MetalFGEnc *)enc state:(id<MTLBuffer>)state flag:(id<MTLBuffer>)flag coverage:(id<MTLTexture>)cov channel:(uint32_t)ch args:(MetalFGDetectArgs)a;
- (void)detectHudless:(MetalFGEnc *)enc state:(id<MTLBuffer>)state flag:(id<MTLBuffer>)flag colour:(id<MTLTexture>)col hudless:(id<MTLTexture>)hud args:(MetalFGDetectArgs)a;
- (void)clearFlag:(MetalFGEnc *)enc flag:(id<MTLBuffer>)flag;
- (void)override:(MetalFGEnc *)enc state:(id<MTLBuffer>)state colour:(id<MTLTexture>)col out:(id<MTLTexture>)out srcOrigin:(const uint32_t *)so dstOrigin:(const uint32_t *)dof size:(const uint32_t *)sz;
@end
