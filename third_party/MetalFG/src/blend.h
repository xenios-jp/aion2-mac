#pragma once
/* F20 candidate blend and pack (spec fg.blend_pack). */
#import <Metal/Metal.h>
#include "enc.h"

typedef struct {
    __unsafe_unretained id<MTLBuffer> scIp, scIc, n2a, n2b, n2h;
    __unsafe_unretained id<MTLTexture> colOld, colCur, bw, err, uiOld, uiCur, candA, candB, hole;
    uint32_t W, H, Wm, Hm, Nw, Nh, s;
    uint32_t ui;
} MetalFGF20Args;

@interface MetalFGBlend : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib;
- (void)blend:(MetalFGEnc *)enc args:(MetalFGF20Args)a;    /* candidates and hole mask */
- (void)pack:(MetalFGEnc *)enc args:(MetalFGF20Args)a;     /* reduction to the net2 grid, from the binary32 blend values (same inputs as blend:) */
- (void)blendPack:(MetalFGEnc *)enc args:(MetalFGF20Args)a;  /* blend: and pack: as one pass (identical results) */
/* builds the specialised blend-and-pack pipeline for a net2 factor and UI flag (used when a call has them) */
- (void)specialiseForS:(uint32_t)s ui:(BOOL)ui;
@end
