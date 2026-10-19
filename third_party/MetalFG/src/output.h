#pragma once
/* Output synthesis F22 and the hole-fill chain F23..F26 (spec fg.output_synthesis). */
#import <Metal/Metal.h>
#include "enc.h"

@interface MetalFGOutput : NSObject
@property(readonly) id<MTLTexture> holeOut;                 /* F: 1 where the pixel needs filling (chain) */
@property(readonly) NSArray<id<MTLBuffer>> *levels;         /* levels 1..L as (colour, coverage) float4 cells */
@property(readonly) id<MTLBuffer> fillTiles;
@property(readonly) id<MTLBuffer> cellSig;                  /* F23: the logistic of channel 4 of every net2 cell (f23_sig) */
/* allocates cellSig for n net2 cells; required before chain: */
- (void)prepareCells:(uint32_t)n;                /* F23 to F26: indirect threadgroup counts, then the indices of the tiles holding a pixel to fill */
@property(readonly) NSArray<NSValue *> *levelSizes;         /* NSValue of (w, h) as NSSize */
/* chain = NO allocates none of the pyramid storage (direct output pass only) */
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib W:(uint32_t)W H:(uint32_t)H chain:(BOOL)chain levels:(uint32_t)cap;
/* F22 */
- (void)direct:(MetalFGEnc *)enc n2out:(id<MTLBuffer>)n candA:(id<MTLTexture>)a candB:(id<MTLTexture>)b hole:(id<MTLTexture>)h out:(id<MTLTexture>)out
        origin:(const uint32_t *)o Nw:(uint32_t)Nw Nh:(uint32_t)Nh s:(uint32_t)s;
/* F23 to F26 */
- (void)chain:(MetalFGEnc *)enc n2out:(id<MTLBuffer>)n candA:(id<MTLTexture>)a candB:(id<MTLTexture>)b hole:(id<MTLTexture>)h out:(id<MTLTexture>)out
       origin:(const uint32_t *)o Nw:(uint32_t)Nw Nh:(uint32_t)Nh s:(uint32_t)s;
@end
