#pragma once
/* F15 net1 input packing and F17 net1 output unpacking (spec fg.net1_io). */
#import <Metal/Metal.h>
#include "enc.h"
#include "plan.h"

typedef struct {
    __unsafe_unretained id<MTLTexture> colOld, colCur, dY, errOld, ns, dis, disOld, st, mcp, mpc, errOut, msOut;
    __unsafe_unretained id<MTLBuffer> n1in;
    uint32_t Wf, Hf, w1, h1, ox, oy, Wm, Hm;
} MetalFGF15Args;

typedef struct {
    __unsafe_unretained id<MTLTexture> ms, uimOld, uimCur, lgP, lgC, st;
    __unsafe_unretained id<MTLBuffer> n1out;
    uint32_t Wf, Hf, w1, h1, ox, oy;
    uint32_t steering;
    float b_steer, th_mask;
} MetalFGF17Args;

@interface MetalFGIO : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib;
- (void)f15:(MetalFGEnc *)enc args:(MetalFGF15Args)a;
- (void)f17:(MetalFGEnc *)enc args:(MetalFGF17Args)a;
@end
