/* Engine entry points and per-call frame orchestration (sr.engine_api, sr.frame_orchestration). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "metalsr.h"
#include "gpu.h"
#include "passes.h"
#include "frame.h"

extern unsigned char metalsr_metallib[];
extern unsigned int metalsr_metallib_len;

#define RING_SLOTS 3
#define RING_BYTES (256 * 1024)
#define MAX_STAMPS 40

@interface SrEngine : NSObject {
@public
    SrGpu *gpu;
    MetalSRCreateInfo info;
    SrParams params;
    int wide, hdr, autoexp;
    SrLevels L;
    SrWeights w;
    SrScratch scratch;
    id<MTLBuffer> feat, s0, s1, q2, s2, q3, s3, q4, s4, q5, z5a, z5b, d4, d3, d2, d1, ohead, ae0, ae1, done;
    id<MTLTexture> hist, mem[2], warped, e_now, e_prev, one, internal_out, depth_r32;
    NSMutableDictionary<NSNumber *, id<MTLTexture>> *copies;   /* pixel format -> engine copy at maximum size (colour, motion) */
    id<MTLBuffer> ring[RING_SLOTS];
    SrEnc *enc;
    uint64_t call;
    int last_dispatches, last_barriers;
    id<MTLResidencySet> callerSet;
    NSMapTable<id, NSNumber *> *callerSeen;
    BOOL callerDirty;
    id<MTL4CounterHeap> heap[RING_SLOTS];
    id<MTLBuffer> stampBuf[RING_SLOTS];
    int nstamps[RING_SLOTS];
    const char *stamp_label[RING_SLOTS][MAX_STAMPS];
    uint64_t slot_call[RING_SLOTS];
    double tick_us;
    char err[512];
}
@end
@implementation SrEngine
@end

static const char *kPipelines[] = {"conv3x3_relu", "conv3x3_mm", "conv_tile_p2", "conv_tile_p3", "conv_tile_p4", "conv_tile_p5", "conv_tile_p2f", "conv_tile_w3", "conv_tile_w4", "conv_tile_w5", "conv1x1_splitk", "pool_m8", "maxpool2x2", "conv1x1", "conv1x1_mm", "conv1x1_mm48", "dec_p8", "dec_p9", "dec_p10", "dec_p11", "dec_head", "m8_encode_buf", "up2_add",
                                   "ae_first", "ae_last", "ae_copy", "clear_tex", "depth_convert", "out_convert"};

static void set_err(char *err, size_t cap, const char *msg) {
    if (!err || !cap) return;
    strncpy(err, msg, cap - 1);
    err[cap - 1] = 0;
}

MetalSREngine *metalsr_create(void *device, const MetalSRCreateInfo *info, char *err, size_t err_cap) {
    @autoreleasepool {
        if (!device) { set_err(err, err_cap, "null device"); return NULL; }
        if (!info) { set_err(err, err_cap, "null create record"); return NULL; }
        char msg[600];
        if (sr_check_create(info, msg, sizeof msg)) { set_err(err, err_cap, msg); return NULL; }
        if (info->test_hooks && info->test_hooks[0]) { set_err(err, err_cap, "testing hooks are not supported"); return NULL; }
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
        if (![dev supportsFamily:MTLGPUFamilyMetal4]) { set_err(err, err_cap, "device lacks Metal 4 support"); return NULL; }
        const char *wp = info->weight_path;
        if (!wp) { wp = getenv("METALSR_WEIGHTS"); if (wp && !wp[0]) wp = NULL; }
        if (!wp) { set_err(err, err_cap, "no weight file was given"); return NULL; }
        uint8_t *wimg = sr_load_weights(wp, msg, sizeof msg);
        if (!wimg) { set_err(err, err_cap, msg); return NULL; }
        SrEngine *e = [SrEngine new];
        if (sr_load_params(wp, &e->params, msg, sizeof msg) != 0) { free(wimg); set_err(err, err_cap, msg); return NULL; }
        e->info = *info;
        e->info.weight_path = e->info.shader_dir = e->info.test_hooks = NULL;
        e->hdr = (info->flags & METALSR_FLAG_HDR) != 0;
        e->autoexp = (info->flags & METALSR_FLAG_AUTO_EXPOSURE) != 0;
        MetalSRSizes zs;
        sr_derive_sizes(info, info->max_rw, info->max_rh, e->params.r_wide, (unsigned)e->params.n_luma_min, &zs);
        e->wide = (int)zs.path;
        NSString *gerr = nil;
        e->gpu = [[SrGpu alloc] initWithDevice:dev libraryData:[NSData dataWithBytes:metalsr_metallib length:metalsr_metallib_len] error:&gerr];
        if (!e->gpu) { free(wimg); set_err(err, err_cap, gerr.UTF8String); return NULL; }
        SrGpu *g = e->gpu;
        [g enableArchiveForLibrary:[NSData dataWithBytes:metalsr_metallib length:metalsr_metallib_len] fallbackDirectory:[@(wp) stringByDeletingLastPathComponent]];
        for (size_t i = 0; i < sizeof kPipelines / sizeof *kPipelines; i++)
            if (![g pipeline:@(kPipelines[i])]) { free(wimg); set_err(err, err_cap, "pipeline creation failed"); return NULL; }
        {   /* the configurations need up to ~19 KB of threadgroup memory and these group sizes: fail cleanly if the device cannot */
            static const struct { const char *name; int threads; } need[] = {
                {"conv_tile_p2", 256}, {"conv_tile_p2f", 128}, {"conv_tile_p3", 256}, {"conv_tile_p4", 256}, {"conv_tile_p5", 128}, {"dec_p8", 128}, {"dec_p9", 128},
                {"dec_p10", 128}, {"dec_p11", 128}, {"dec_head", 128}, {"ae_first", 512}, {"ae_last", 512}};
            if (dev.maxThreadgroupMemoryLength < 20 * 1024) { free(wimg); set_err(err, err_cap, "device threadgroup memory is too small (needs 20 KB)"); return NULL; }
            for (size_t i = 0; i < sizeof need / sizeof *need; i++)
                if ([g pipeline:@(need[i].name)].maxTotalThreadsPerThreadgroup < (NSUInteger)need[i].threads) {
                    free(wimg); snprintf(msg, sizeof msg, "kernel %s cannot run %d threads per group on this device", need[i].name, need[i].threads);
                    set_err(err, err_cap, msg); return NULL;
                }
        }

        {   /* the kernels of this configuration (T1): P1 by colour mode, motion-vector resolution and depth direction, P12 by path and colour mode */
            FeatParams fz; memset(&fz, 0, sizeof fz); fz.hdr = e->hdr; fz.mv_render = (info->flags & METALSR_FLAG_MV_RENDER_RES) != 0; fz.depth_inv = (info->flags & METALSR_FLAG_DEPTH_INVERTED) != 0;
            FiltParams pz; memset(&pz, 0, sizeof pz); pz.wide = (u32)e->wide; pz.hdr = e->hdr;
            NSString *names[2] = {sr_p1_kernel(&fz), sr_p12_kernel(&pz)};
            for (int i = 0; i < 2; i++) {
                id<MTLComputePipelineState> ps = [g pipeline:names[i]];
                if (!ps || ps.maxTotalThreadsPerThreadgroup < 128) { free(wimg); snprintf(msg, sizeof msg, "kernel %s cannot run 128 threads per group on this device", names[i].UTF8String); set_err(err, err_cap, msg); return NULL; }
            }
        }
        [g saveArchive];
        sr_levels(info->out_w, info->out_h, &e->L);
        e->L.wide = e->wide;
        SrLevels *L = &e->L;
        e->w.buf = [g buffer:SR_WEIGHT_FILE_BYTES];
        memcpy(e->w.buf.contents, wimg, SR_WEIGHT_FILE_BYTES);
        sr_weights_prepare(g, &e->w);
        free(wimg);
        size_t px[6];
        for (int k = 0; k < 6; k++) px[k] = (size_t)L->W[k] * L->H[k];
        e->feat = [g buffer:px[0] * 32];
        e->s0 = [g buffer:px[0] * 32]; e->s1 = [g buffer:px[1] * 64]; e->q2 = [g buffer:px[2] * 64];
        e->s2 = [g buffer:px[2] * 256]; e->q3 = [g buffer:px[3] * 256]; e->s3 = [g buffer:px[3] * 256]; e->q4 = [g buffer:px[4] * 256];
        e->s4 = [g buffer:px[4] * 512]; e->q5 = [g buffer:px[5] * 512]; e->z5a = [g buffer:px[5] * 1024]; e->z5b = [g buffer:px[5] * 512];
        e->d4 = [g buffer:px[4] * 256]; e->d3 = [g buffer:px[3] * 256]; e->d2 = [g buffer:px[2] * 64]; e->d1 = [g buffer:px[1] * 32];
        e->ohead = [g buffer:px[0] * 80];
        e->scratch = (SrScratch){[g buffer:sr_scratch_bytes(L, 0)], [g buffer:sr_scratch_bytes(L, 1)], [g buffer:sr_scratch_bytes(L, 2)]};
        size_t aef = sr_ae_scratch_floats(zs.gw, zs.gh);
        e->ae0 = [g buffer:aef * 4]; e->ae1 = [g buffer:aef * 4];
        e->done = [g buffer:16];
        e->hist = [g privateTexture:MTLPixelFormatRGBA16Float w:info->out_w h:info->out_h];
        e->mem[0] = [g privateTexture:MTLPixelFormatRG16Float w:info->out_w h:info->out_h];
        e->mem[1] = [g privateTexture:MTLPixelFormatRG16Float w:info->out_w h:info->out_h];
        e->warped = [g privateTexture:MTLPixelFormatRGBA16Float w:info->out_w h:info->out_h];
        e->e_now = [g texture:MTLPixelFormatR16Float w:1 h:1];
        e->e_prev = [g texture:MTLPixelFormatR16Float w:1 h:1];
        e->one = [g texture:MTLPixelFormatR16Float w:1 h:1];
        { uint16_t one = 0x3C00; [e->one replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:2]; }
        e->internal_out = [g privateTexture:MTLPixelFormatRGBA16Float w:info->out_w h:info->out_h];
        e->depth_r32 = [g privateTexture:MTLPixelFormatR32Float w:info->max_rw h:info->max_rh];
        e->copies = [NSMutableDictionary new];
        for (int i = 0; i < RING_SLOTS; i++) e->ring[i] = [g buffer:RING_BYTES];
        e->enc = [[SrEnc alloc] initWithGpu:g encoder:nil ring:nil];
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        NSError *ne = nil;
        e->callerSet = [dev newResidencySetWithDescriptor:rd error:&ne];
        e->callerSeen = [NSMapTable strongToStrongObjectsMapTable];
        if (info->profile == 1) {
            MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new];
            hd.type = MTL4CounterHeapTypeTimestamp; hd.count = MAX_STAMPS;
            for (int i = 0; i < RING_SLOTS; i++) {
                e->heap[i] = [dev newCounterHeapWithDescriptor:hd error:&ne];
                e->stampBuf[i] = [g buffer:MAX_STAMPS * 8];
            }
            e->tick_us = 1e6 / (double)[dev queryTimestampFrequency];
        }
        (void)[g tanhTable];          /* every allocation must exist and be committed to the residency set before the first encode */
        [g commitResidency];
        e->err[0] = 0;
        return (__bridge_retained MetalSREngine *)e;
    }
}

/* ---- per-call helpers ---- */
static int fail(SrEngine *e, int code, const char *fmt, long a) {
    snprintf(e->err, sizeof e->err, fmt, a);
    fprintf(stderr, "metalsr: %s\n", e->err);
    return code;
}

static void touch_caller(SrEngine *e, id<MTLTexture> t) {
    if (!t) return;
    if (![e->callerSeen objectForKey:t]) { [e->callerSet addAllocation:t]; e->callerDirty = YES; }
    [e->callerSeen setObject:@(e->call) forKey:t];
}
static void expire_caller(SrEngine *e) {
    NSMutableArray *drop = [NSMutableArray new];
    for (id t in e->callerSeen) if (e->call - [[e->callerSeen objectForKey:t] unsignedLongLongValue] >= 16) [drop addObject:t];
    for (id t in drop) { [e->callerSet removeAllocation:t]; [e->callerSeen removeObjectForKey:t]; e->callerDirty = YES; }
}

static id<MTLTexture> engine_copy(SrEngine *e, MTLPixelFormat f, int w, int h) {
    id<MTLTexture> t = e->copies[@(f)];
    if (!t || t.width < (NSUInteger)w || t.height < (NSUInteger)h) {
        t = [e->gpu privateTexture:f w:w h:h];
        e->copies[@(f)] = t;
    }
    return t;
}

static void clear_tex(SrEnc *en, SrGpu *g, id<MTLTexture> t) {
    struct { uint32_t w, h; } p = {(uint32_t)t.width, (uint32_t)t.height};
    SrArg a[2] = {ARG_TEX(t), ARG_VAL(p)};
    [en dispatchGrid:MTLSizeMake(p.w, p.h, 1) pso:[g pipeline:@"clear_tex"] args:a count:2 threads:MTLSizeMake(16, 16, 1)];
}

static BOOL depth_format(MTLPixelFormat f) {
    return f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth16Unorm;
}
static BOOL output_convertible(MTLPixelFormat f) {
    return f == MTLPixelFormatRG11B10Float || f == MTLPixelFormatRGBA32Float || f == MTLPixelFormatRGB10A2Unorm || f == MTLPixelFormatRGBA8Unorm || f == MTLPixelFormatBGRA8Unorm;
}

int metalsr_encode(MetalSREngine *eh, void *cmdbuf, const MetalSRFrameInfo *f) {
    @autoreleasepool {
        SrEngine *e = (__bridge SrEngine *)eh;
        if (!e) return 1;
        if (!cmdbuf || !f) return fail(e, 1, "null command buffer or frame record", 0);
        if (!f->color || !f->depth || !f->motion || !f->output) return fail(e, 1, "null required texture", 0);
        id<MTL4CommandBuffer> cb = (__bridge id<MTL4CommandBuffer>)cmdbuf;
        id<MTLTexture> colour = (__bridge id<MTLTexture>)f->color, depth = (__bridge id<MTLTexture>)f->depth, motion = (__bridge id<MTLTexture>)f->motion,
                       output = (__bridge id<MTLTexture>)f->output, exposure = f->exposure ? (__bridge id<MTLTexture>)f->exposure : nil;
        SrGpu *g = e->gpu;
        uint32_t Wo = e->info.out_w, Ho = e->info.out_h, rw = f->rw, rh = f->rh;
        if (sr_check_call(&e->info, rw, rh, e->err, sizeof e->err)) { fprintf(stderr, "metalsr: %s\n", e->err); return 3; }
        int mvr = (e->info.flags & METALSR_FLAG_MV_RENDER_RES) != 0;
        uint32_t mw = mvr ? rw : Wo, mh = mvr ? rh : Ho;
        /* formats */
        if (depth.pixelFormat != MTLPixelFormatR32Float && !depth_format(depth.pixelFormat)) return fail(e, 2, "unsupported depth format %ld", (long)depth.pixelFormat);
        BOOL out_direct = output.pixelFormat == MTLPixelFormatRGBA16Float;
        if (!out_direct && !output_convertible(output.pixelFormat)) return fail(e, 2, "unsupported output format %ld", (long)output.pixelFormat);
        if (!(output.usage & MTLTextureUsageShaderWrite)) return fail(e, 2, "output texture lacks shader-write usage (format %ld)", (long)output.pixelFormat);
        if (exposure && !(exposure.width == 1 && exposure.height == 1 && (exposure.pixelFormat == MTLPixelFormatR16Float || exposure.pixelFormat == MTLPixelFormatR32Float)))
            return fail(e, 2, "exposure texture must be 1x1 R16Float or R32Float (format %ld)", (long)exposure.pixelFormat);
        /* rects */
        #define RECT_OK(tex, org, w_, h_) ((uint64_t)(org)[0] + (w_) <= (tex).width && (uint64_t)(org)[1] + (h_) <= (tex).height)
        if (!RECT_OK(colour, f->color_origin, rw, rh)) return fail(e, 3, "colour rect outside the texture", 0);
        if (!RECT_OK(depth, f->depth_origin, rw, rh)) return fail(e, 3, "depth rect outside the texture", 0);
        if (!RECT_OK(motion, f->motion_origin, mw, mh)) return fail(e, 3, "motion rect outside the texture", 0);
        if (!RECT_OK(output, f->output_origin, Wo, Ho)) return fail(e, 3, "output texture too small for the output rect", 0);

        /* ring slot */
        int slot = (int)(e->call % RING_SLOTS);
        if (e->call >= RING_SLOTS) {
            uint64_t need = e->call - RING_SLOTS + 1;
            double t0 = 0; useconds_t waited = 0;
            while (*(volatile uint64_t *)e->done.contents < need) {
                usleep(50); waited += 50; t0 += 50;
                if (waited > 2000000) return fail(e, 4, "timed out waiting for the GPU to finish an earlier call", 0);
            }
            (void)t0;
        }

        touch_caller(e, colour); touch_caller(e, depth); touch_caller(e, motion); touch_caller(e, output); touch_caller(e, exposure);
        expire_caller(e);
        if (e->callerDirty) { [e->callerSet commit]; e->callerDirty = NO; }

        int first = e->call == 0;
        int reset = first || f->reset != 0;
        SrCall c = {Wo, Ho, rw, rh, f->jitter_x, f->jitter_y, f->mv_scale_x, f->mv_scale_y, f->pre_exposure, (u32)e->hdr,
                    (u32)((e->info.flags & METALSR_FLAG_DEPTH_INVERTED) != 0), (u32)mvr, (u32)reset, (u32)e->wide, f->output_origin[0], f->output_origin[1], (u32)e->L.W[0]};

        /* engine copies for non-zero origins */
        id<MTLTexture> colour_t = colour, motion_t = motion, depth_t = depth;
        int need_cc = f->color_origin[0] || f->color_origin[1], need_mc = f->motion_origin[0] || f->motion_origin[1];
        if (need_cc) colour_t = engine_copy(e, colour.pixelFormat, e->info.max_rw, e->info.max_rh);
        if (need_mc) motion_t = engine_copy(e, motion.pixelFormat, mvr ? e->info.max_rw : Wo, mvr ? e->info.max_rh : Ho);
        int depth_conv = depth.pixelFormat != MTLPixelFormatR32Float;
        int depth_blit = !depth_conv && (f->depth_origin[0] || f->depth_origin[1]);
        if (depth_conv || depth_blit) depth_t = e->depth_r32;
        [g commitResidency];

        id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];
        id<MTLResidencySet> sets[2] = {g.residency, e->callerSet};
        [cb useResidencySets:sets count:2];
        SrEnc *s = e->enc;
        [s attachEncoder:en ring:e->ring[slot]];
        int ns = 0;
        BOOL prof = e->info.profile == 1;
        #define STAMP(label) do { if (prof && ns < MAX_STAMPS) { [en writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:e->heap[slot] atIndex:ns]; e->stamp_label[slot][ns] = label; ns++; } } while (0)

        if (f->fence) [en waitForFence:(__bridge id<MTLFence>)f->fence beforeEncoderStages:MTLStageDispatch];
        [en barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
        STAMP("start");
        if (first) {   /* line 1: clear every engine buffer and image */
            id<MTLBuffer> bufs[] = {e->feat, e->s0, e->s1, e->q2, e->s2, e->q3, e->s3, e->q4, e->s4, e->q5, e->z5a, e->z5b, e->d4, e->d3, e->d2, e->d1, e->ohead,
                                    e->scratch.a, e->scratch.b, e->scratch.c, e->ae0, e->ae1};
            for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) [en fillBuffer:bufs[i] range:NSMakeRange(0, bufs[i].length) value:0];
            id<MTLTexture> texs[] = {e->hist, e->mem[0], e->mem[1], e->warped, e->e_now, e->e_prev, e->internal_out, e->depth_r32};
            for (size_t i = 0; i < sizeof texs / sizeof *texs; i++) clear_tex(s, g, texs[i]);
            [s barrier];
        } else if (reset) {   /* line 2 */
            clear_tex(s, g, e->hist); clear_tex(s, g, e->mem[0]); clear_tex(s, g, e->mem[1]);
            [s barrier];
        }
        if (first) STAMP("clear"); else if (reset) STAMP("reset");
        /* line 3: input preparation */
        BOOL prep = NO;
        if (need_cc) { [en copyFromTexture:colour sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(f->color_origin[0], f->color_origin[1], 0) sourceSize:MTLSizeMake(rw, rh, 1)
                                 toTexture:colour_t destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)]; prep = YES; }
        if (need_mc) { [en copyFromTexture:motion sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(f->motion_origin[0], f->motion_origin[1], 0) sourceSize:MTLSizeMake(mw, mh, 1)
                                 toTexture:motion_t destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)]; prep = YES; }
        if (depth_blit) { [en copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(f->depth_origin[0], f->depth_origin[1], 0) sourceSize:MTLSizeMake(rw, rh, 1)
                                  toTexture:depth_t destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)]; prep = YES; }
        if (depth_conv) {
            struct { uint32_t w, h, ox, oy; } dp = {rw, rh, f->depth_origin[0], f->depth_origin[1]};
            SrArg a[3] = {ARG_TEX(depth), ARG_TEX(depth_t), ARG_VAL(dp)};
            [s dispatchGrid:MTLSizeMake(rw, rh, 1) pso:[g pipeline:@"depth_convert"] args:a count:3 threads:MTLSizeMake(16, 16, 1)];
            prep = YES;
        }
        if (prep) { [s barrier]; STAMP("prepare"); }

        /* exposure source for this call */
        id<MTLTexture> e_now_t = (e->hdr && e->autoexp) ? e->e_now : (exposure ? exposure : e->one);
        if (e->hdr && e->autoexp && e->call % 2 == 0) {   /* line 4 */
            AeParams ap; sr_make_ae(&c, &e->params, &ap);
            sr_ae(s, g, &ap, colour_t, e->ae0, e->ae1, e->e_now);
            [s barrier]; STAMP("A1-A3");
        }
        /* line 5: P1 */
        FeatParams p1; sr_make_feat(&c, &e->params, &p1);
        SrFeatIo io1 = {colour_t, depth_t, motion_t, e->hist, e->mem[e->call % 2], e_now_t, e->e_prev, e->warped, e->mem[(e->call + 1) % 2], e->feat};
        sr_features(s, g, &p1, &io1);
        [s barrier]; STAMP("P1");
        SrWeights *w = &e->w; SrLevels *L = &e->L; SrScratch sc = e->scratch; int var = e->wide;
        sr_p2(s, g, w, L, e->feat, e->s0, e->s1, e->q2, sc); [s barrier]; STAMP("P2");
        sr_p3to5(s, g, w, L, 3, e->q2, e->s2, e->q3, sc); [s barrier]; STAMP("P3");
        sr_p3to5(s, g, w, L, 4, e->q3, e->s3, e->q4, sc); [s barrier]; STAMP("P4");
        sr_p3to5(s, g, w, L, 5, e->q4, e->s4, e->q5, sc); [s barrier]; STAMP("P5");
        sr_p6(s, g, w, L, e->q5, e->z5a); [s barrier]; STAMP("P6");
        sr_p7(s, g, w, L, e->z5a, e->z5b); [s barrier]; STAMP("P7");
        sr_dec(s, g, w, L, 8, var, e->z5b, e->s4, e->d4, sc); [s barrier]; STAMP("P8");
        sr_dec(s, g, w, L, 9, var, e->d4, e->s3, e->d3, sc); [s barrier]; STAMP("P9");
        sr_dec(s, g, w, L, 10, var, e->d3, e->s2, e->d2, sc); [s barrier]; STAMP("P10");
        sr_dec(s, g, w, L, 11, var, e->d2, e->s1, e->d1, sc); [s barrier]; STAMP("P11");
        /* line 16: head network and P12 */
        FiltParams p12; sr_make_filt(&c, &e->params, &p12);
        id<MTLTexture> target = out_direct ? output : e->internal_out;
        if (!out_direct) { p12.ox = 0; p12.oy = 0; }
        SrP12Io io12 = {colour_t, e->warped, e_now_t, target, e->hist, e->e_prev, e->d1, e->s0};
        sr_p12(s, g, w, L, &p12, &io12);
        [s barrier]; STAMP("P12");
        if (!out_direct) {   /* line 17 */
            /* largest finite magnitude of the destination: RG11B10Float stops at 65024 (an fp16 65504 would round to infinity), the others hold the fp16 range */
            float maxv = output.pixelFormat == MTLPixelFormatRG11B10Float ? 65024.0f : (output.pixelFormat == MTLPixelFormatRGBA32Float ? 3.4028234e38f : 65504.0f);
            struct { uint32_t w, h, ox, oy; float maxv; } op = {Wo, Ho, f->output_origin[0], f->output_origin[1], maxv};
            SrArg a[3] = {ARG_TEX(e->internal_out), ARG_TEX(output), ARG_VAL(op)};
            [s dispatchGrid:MTLSizeMake(Wo, Ho, 1) pso:[g pipeline:@"out_convert"] args:a count:3 threads:MTLSizeMake(16, 16, 1)];
            [s barrier]; STAMP("convert");
        }
        /* line 18, P13 (E_prev := E_now): done by one thread of P12 (no dispatch of its own) */
        {   /* completion marker: a copy of the call number (written by the CPU into the last word of the ring slot) after the last dispatch, not a dispatch */
            *(uint64_t *)((char *)e->ring[slot].contents + RING_BYTES - 8) = e->call + 1;
            [en copyFromBuffer:e->ring[slot] sourceOffset:RING_BYTES - 8 toBuffer:e->done destinationOffset:0 size:8];
        }
        [en barrierAfterStages:MTLStageDispatch | MTLStageBlit beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        if (f->fence) [en updateFence:(__bridge id<MTLFence>)f->fence afterEncoderStages:MTLStageDispatch];
        e->last_dispatches = s.dispatches; e->last_barriers = s.barriers + 2;   /* the two queue-stage barriers of the call */
        [en endEncoding];
        if (prof) {
            [cb resolveCounterHeap:e->heap[slot] withRange:NSMakeRange(0, ns) intoBuffer:(MTL4BufferRange){e->stampBuf[slot].gpuAddress, (uint64_t)ns * 8} waitFence:nil updateFence:nil];
            e->nstamps[slot] = ns; e->slot_call[slot] = e->call;
        }
        e->call++;
        return 0;
    }
}

uint32_t metalsr_timings(MetalSREngine *eh, MetalSRTiming *out, uint32_t cap) {
    SrEngine *e = (__bridge SrEngine *)eh;
    if (!e || e->info.profile != 1 || !out || !e->call) return 0;
    uint64_t completed = *(volatile uint64_t *)e->done.contents;   /* calls fully finished */
    if (completed < 2) return 0;
    /* the resolve of call n sits after its encoder; call n - 1 is certainly resolved once call n has completed */
    uint64_t want = completed - 2;
    int slot = (int)(want % RING_SLOTS);
    if (e->slot_call[slot] != want || e->nstamps[slot] < 2) return 0;
    const uint64_t *t = e->stampBuf[slot].contents;
    uint32_t n = 0;
    for (int i = 1; i < e->nstamps[slot] && n < cap; i++) {
        out[n].label = e->stamp_label[slot][i];
        out[n].microseconds = (double)(t[i] - t[i - 1]) * e->tick_us;
        n++;
    }
    return n;
}

void metalsr_destroy(MetalSREngine *eh) {
    if (!eh) return;
    SrEngine *e = (__bridge_transfer SrEngine *)eh;
    (void)e;
}

const char *metalsr_last_error(MetalSREngine *eh) {
    SrEngine *e = (__bridge SrEngine *)eh;
    return e ? e->err : "";
}

int metalsr_debug_path(MetalSREngine *eh) {
    SrEngine *e = (__bridge SrEngine *)eh;
    return e ? e->wide : 0;
}

/* test support: exposure state after the newest completed call (values of the 1x1 R16F textures) */
void metalsr_debug_exposure(MetalSREngine *eh, float *now, float *prev) {
    SrEngine *e = (__bridge SrEngine *)eh;
    uint16_t a = 0, b = 0;
    if (e) {
        [e->e_now getBytes:&a bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
        [e->e_prev getBytes:&b bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    }
    _Float16 ha, hb; memcpy(&ha, &a, 2); memcpy(&hb, &b, 2);
    if (now) *now = (float)ha;
    if (prev) *prev = (float)hb;
}

void metalsr_debug_counts(MetalSREngine *eh, int *dispatches, int *barriers) {
    SrEngine *e = (__bridge SrEngine *)eh;
    if (dispatches) *dispatches = e ? e->last_dispatches : 0;
    if (barriers) *barriers = e ? e->last_barriers : 0;
}
