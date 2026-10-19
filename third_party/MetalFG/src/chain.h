#pragma once
/* F04 colour reduction and UI estimate, F13 disocclusion scatter, F14 warp-error seed (spec fg.colour_disocclusion),
   F07 to F11 previous-to-current field (spec fg.prev2curr), F18 and F19 intermediate motion and blend weights
   (spec fg.interm_motion). */
#import <Metal/Metal.h>
#include "enc.h"

typedef struct {
    float Zs, Zp, Zsep, eps_n;
    uint32_t reversed;
} MetalFGDepthConst;

typedef struct {
    uint32_t W, H;
    uint32_t origin[2];
    uint32_t mode;            /* 0 P, 1 D, 2 A, 3 R */
    uint32_t ui;              /* create bit 0 */
    uint32_t covFromColour;
    uint32_t df;
    float th_ui, th_uia;
} MetalFGF04Args;

typedef struct {
    uint32_t Wm, Hm;
    float depthW, depthH;     /* depth rectangle size (tap offsets) */
    float H;                  /* display height */
    float d_dyn, d_sim, th_cam;
    MetalFGDepthConst D;
    float c2p[16], p2c[16];
    uint32_t writeNsc;
} MetalFGP2CArgs;

typedef struct {
    uint32_t Wm, Hm, Wf, Hf;
    float H, d_sim, th_zn, Z_sep;
    int32_t sp_min, sp_max;
    MetalFGDepthConst D;
} MetalFGF18Args;

@interface MetalFGChain : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib Wm:(uint32_t)Wm Hm:(uint32_t)Hm;
- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib Wm:(uint32_t)Wm Hm:(uint32_t)Hm levels:(uint32_t)cap;
/* Textures that a mode does not read may be passed as any readable texture (hud, cov), uiCur as nil when bit 0 is clear. */
/* Buffers the engine must keep resident for the Metal 4 encoder. */
@property(readonly) NSArray<id<MTLAllocation>> *allocations;
- (void)f04:(MetalFGEnc *)enc args:(MetalFGF04Args)a colour:(id<MTLTexture>)c hudless:(id<MTLTexture>)h coverage:(id<MTLTexture>)cov
     colOld:(id<MTLTexture>)old colCur:(id<MTLTexture>)cur uiCur:(id<MTLTexture>)ui colFeat:(id<MTLTexture>)cf uim:(id<MTLTexture>)uim
         dY:(id<MTLTexture>)dY;
/* Writes zeros into the prefilled (value one) output O. Wm, Hm: the Mot grid (for the Mot-pixel input). */
- (void)f13:(MetalFGEnc *)enc field:(id<MTLTexture>)g out:(id<MTLTexture>)o normalised:(BOOL)norm Wm:(uint32_t)Wm Hm:(uint32_t)Hm;
- (void)f14:(MetalFGEnc *)enc colFeat:(id<MTLTexture>)cf err:(id<MTLTexture>)err;
/* F07 to F11. sc is prefilled with 0x7FFFFFFF in every 32-bit word. */
- (void)p2c:(MetalFGEnc *)enc args:(MetalFGP2CArgs)a sc:(id<MTLBuffer>)sc hb:(id<MTLBuffer>)hb depCur:(id<MTLTexture>)depCur
     depOld:(id<MTLTexture>)depOld dynCur:(id<MTLTexture>)dynCur dynOld:(id<MTLTexture>)dynOld mvo:(id<MTLTexture>)mvo
        mvf:(id<MTLTexture>)mvf nsc:(id<MTLTexture>)nsc;
/* F18 for one phase: field in Mot pixels after scaling by hscale; sc prefilled. */
- (void)f18:(MetalFGEnc *)enc args:(MetalFGF18Args)a field:(id<MTLTexture>)field hscale:(const float *)hscale dep:(id<MTLTexture>)dep
         sc:(id<MTLBuffer>)sc useZn:(BOOL)useZn;   /* useZn: scale-distance admission (the projection of the current-call field only) */
/* the same for rows y0 .. y1 - 1 only */
- (void)f18:(MetalFGEnc *)enc args:(MetalFGF18Args)a field:(id<MTLTexture>)field hscale:(const float *)hscale dep:(id<MTLTexture>)dep
         sc:(id<MTLBuffer>)sc useZn:(BOOL)useZn rows:(uint32_t)y0 to:(uint32_t)y1;
- (void)f19:(MetalFGEnc *)enc args:(MetalFGF18Args)a scP:(id<MTLBuffer>)scp scC:(id<MTLBuffer>)scc lgP:(id<MTLTexture>)lgP
       lgC:(id<MTLTexture>)lgC bw:(id<MTLTexture>)bw;
@end

#if defined(METALFG_CTL_FEAT_WEIGHTS)
@interface MetalFGChain (CtlFeatWeights)
- (void)f19feat:(MetalFGEnc *)enc args:(MetalFGF18Args)a Wm:(uint32_t)Wm Hm:(uint32_t)Hm scP:(id<MTLBuffer>)scp scC:(id<MTLBuffer>)scc lgP:(id<MTLTexture>)lgP lgC:(id<MTLTexture>)lgC bw:(id<MTLTexture>)bw;
@end
#endif
