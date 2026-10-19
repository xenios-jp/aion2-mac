#pragma once
/* Shared network machinery: convolution/pool/upsample encoders (specs fg.net1_forward, fg.net2_forward). */
#import <Metal/Metal.h>
#include "enc.h"
#include "weights.h"

enum { METALFG_CONV_ACT = 1, METALFG_CONV_BOUND = 2, METALFG_CONV_POOL = 4, METALFG_CONV_RES = 8, METALFG_CONV_ZADD = 16, METALFG_CONV_MAXPOOL = 128, METALFG_CONV_FINAL = 256 };

typedef struct {
    __unsafe_unretained id<MTLBuffer> in, out, res;
    __unsafe_unretained id<MTLBuffer> pout;   /* METALFG_CONV_MAXPOOL: the 2 x 2 maximum of the output (matrix path only) */
    __unsafe_unretained id<MTLBuffer> usrc;   /* up_ws != 0 (1 x 1 layers, matrix path only): input = up_corner(usrc) + in, usrc of up_ws x up_hs pixels, cin channels */
    uint32_t up_ws, up_hs;
    uint32_t W, H;               /* spatial size of the input */
    uint32_t cin, in_stride, in_base;
    uint32_t cout, out_stride, out_base;
    uint32_t row0;               /* first weight row (grouped layers) */
    uint32_t flags;
    uint32_t res_stride, res_base;
    uint32_t bound_from;         /* the bound applies to output channels >= bound_from */
    uint32_t nest;               /* order of the (tap, channel group) pairs of a 3x3 layer (fg.arithmetic_pin_r4 rule B): 1, 2 or 3; 0 = 3 (net2) */
    int widx;                    /* weight tensor index in fg.weights_format; the bias follows it */
} MetalFGConvArgs;

@interface MetalFGNN : NSObject
@property(readonly) id<MTLDevice> device;
@property(readonly) id<MTLBuffer> weights;
@property(readonly) float leak, satN2;
+ (NSArray<NSString *> *)kernelNames;
/* weights: buffer holding the whole weight file image plus at least 4096 bytes of zero padding */
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib weights:(id<MTLBuffer>)w leak:(float)leak satN2:(float)sat error:(NSString **)err;
- (id<MTLBuffer>)scratch:(NSUInteger)halves;
- (void)conv:(MetalFGEnc *)enc args:(MetalFGConvArgs)a;
/* specialised pipelines: between beginCollecting and buildCollected the layers encoded record their shapes; buildCollected builds them all */
- (void)beginCollecting;
- (void)buildCollected;
/* two 1 x 1 layers in one pass (a: 32 outputs, b: 32 inputs and at most 16 outputs; or a: 64 outputs, b: 64 inputs and at most 32 outputs; the intermediate
   is not stored), matrix path only */
- (void)convPair:(MetalFGEnc *)enc first:(MetalFGConvArgs)a second:(MetalFGConvArgs)b;
/* YES when conv: serves the layer on the matrix unit (the path that knows METALFG_CONV_MAXPOOL) */
- (BOOL)matrixPath:(const char *)wname cin:(uint32_t)cin;
- (void)poolMax:(MetalFGEnc *)enc in:(id<MTLBuffer>)in out:(id<MTLBuffer>)out W:(uint32_t)W H:(uint32_t)H C:(uint32_t)C;
- (void)upCornerAdd:(MetalFGEnc *)enc src:(id<MTLBuffer>)src skip:(id<MTLBuffer>)skip out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C;
- (void)up2:(MetalFGEnc *)enc src:(id<MTLBuffer>)src out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C;
/* the enlargement feeding the two final heads (rule E): extrapolating at the left and top edge */
- (void)up2x:(MetalFGEnc *)enc src:(id<MTLBuffer>)src out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C;
- (id<MTLComputePipelineState>)pipeline:(NSString *)n;
@property(readonly) NSArray<id<MTLBuffer>> *extraBuffers;
/* net2 helper kernels */
- (void)packX0:(MetalFGEnc *)enc a:(id<MTLBuffer>)a b:(id<MTLBuffer>)b h:(id<MTLBuffer>)h out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh t:(float)t;
- (void)join:(MetalFGEnc *)enc a:(id<MTLBuffer>)a b:(id<MTLBuffer>)b h:(id<MTLBuffer>)h u:(id<MTLBuffer>)u out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh t:(float)t;
- (void)final:(MetalFGEnc *)enc k:(id<MTLBuffer>)k u:(id<MTLBuffer>)u out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh;
@end

/* Net1 (F16): n1in (w1 x h1 x 48) to n1out (w1 x h1 x 16). */
@interface MetalFGNet1 : NSObject
@property(readonly) NSDictionary<NSString *, id<MTLBuffer>> *tensors; /* intermediates by name, for tests */
- (instancetype)initWithNN:(MetalFGNN *)nn w1:(uint32_t)w1 h1:(uint32_t)h1;
- (void)encode:(MetalFGEnc *)enc in:(id<MTLBuffer>)n1in out:(id<MTLBuffer>)n1out;
@end

/* Net2 (F21): n2a, n2b (Nw x Nh x 4), n2h (Nw x Nh x 1) and t to n2out (Nw x Nh x 8). */
@interface MetalFGNet2 : NSObject
@property(readonly) NSDictionary<NSString *, id<MTLBuffer>> *tensors;
- (instancetype)initWithNN:(MetalFGNN *)nn Nw:(uint32_t)Nw Nh:(uint32_t)Nh;
- (void)encode:(MetalFGEnc *)enc a:(id<MTLBuffer>)a b:(id<MTLBuffer>)b h:(id<MTLBuffer>)h t:(float)t out:(id<MTLBuffer>)out;
@end
