/* Engine object: create, resources, validation, per-call orchestration (specs fg.engine_api, fg.resources,
   fg.orchestration). Passes whose specs are approved are encoded by pass_*.m; the others are seams. */
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <math.h>
#include <stdlib.h>
#include "engine.h"
#include "camera.h"
#include "metalfgconst.h"
#include "passes_impl.h"

@implementation MetalFGSlot
@end

static MTLPixelFormat fmt_from_name(const char *n) {
    if (!strcmp(n, "RGBA16Float")) return MTLPixelFormatRGBA16Float;
    if (!strcmp(n, "RG16Float")) return MTLPixelFormatRG16Float;
    if (!strcmp(n, "R16Float")) return MTLPixelFormatR16Float;
    if (!strcmp(n, "R8Unorm")) return MTLPixelFormatR8Unorm;
    if (!strcmp(n, "R32Uint")) return MTLPixelFormatR32Uint;
    if (!strcmp(n, "RG32Float")) return MTLPixelFormatRG32Float;
    return MTLPixelFormatInvalid;
}

/* Kernels every engine needs; pass files add theirs through metalfg_kernel_names(). */
static NSArray<NSString *> *kernel_names(void) {
    NSMutableArray *a = [NSMutableArray arrayWithArray:@[ @"rect_copy", @"fill_tex", @"fill_tex_u32", @"fill_buf_u32" ]];
#if defined(METALFG_CTL_BLEND)
    [a addObject:@"ctl_blend"];
#endif
#if defined(METALFG_CTL_PERF_SPIN)
    [a addObject:@"ctl_spin"];
#endif
    [a addObjectsFromArray:metalfg_pass_kernel_names()];
    return a;
}

@implementation MetalFGEngineImpl

- (id<MTLTexture>)tex:(NSString *)name old:(BOOL)old {
    MetalFGSlot *s = self.slots[name];
    if (!s) return nil;
    int side = s.desc.pair ? (old ? 1 - s.cur : s.cur) : 0;
    return side ? s.tex1 : s.tex0;
}

- (id<MTLBuffer>)buf:(NSString *)name old:(BOOL)old {
    MetalFGSlot *s = self.slots[name];
    if (!s) return nil;
    int side = s.desc.pair ? (old ? 1 - s.cur : s.cur) : 0;
    return side ? s.buf1 : s.buf0;
}

#define FAIL(...) do { snprintf(err, cap, __VA_ARGS__); return nil; } while (0)

- (instancetype)initWithDevice:(id<MTLDevice>)dev info:(const MetalFGCreateInfo *)ci err:(char *)err cap:(size_t)cap {
    self = [super init];
    if (!self) return nil;
    if (!ci) FAIL("create record is null");
    if (ci->version != 1u && ci->version != 2u) FAIL("unsupported interface version %u (supported: 1 and 2)", ci->version);
    if (!dev) FAIL("Metal device is null");
    if (ci->display_w < METALFG_EFF_MIN_SIDE || ci->display_h < METALFG_EFF_MIN_SIDE)
        FAIL("display size %ux%u: each side must be at least %u", ci->display_w, ci->display_h, METALFG_EFF_MIN_SIDE);
    if (ci->display_w > METALFG_EFF_MAX_WIDTH) FAIL("display width %u exceeds the width cap of %u", ci->display_w, METALFG_EFF_MAX_WIDTH);
    /* the height cap depends on the parameter table (min(8192, 16 hb)) and is checked with the grid plan below */
    if (ci->max_generated > 5) FAIL("max_generated %u exceeds 5", ci->max_generated);
    if (ci->reserved != 0) FAIL("reserved field must be zero");
    const char *dn[3] = {"depth_scale", "depth_partition", "depth_min_sep"};
    float dv[3] = {ci->depth_scale, ci->depth_partition, ci->depth_min_sep};
    for (int i = 0; i < 3; i++)
        if (!isfinite(dv[i]) || dv[i] < 0) FAIL("create field %s must be finite and not negative", dn[i]);
    info = *ci;
    if (info.max_generated == 0) info.max_generated = 1;
    if (info.depth_scale == 0) info.depth_scale = 1.0f;
    if (info.depth_partition == 0) info.depth_partition = 1024.0f * info.depth_scale;
    if (info.depth_min_sep == 0) info.depth_min_sep = info.depth_partition / 1024.0f;
    self.device = dev;

    /* weights and parameters */
    const char *wpath = ci->weights_file;
    if (!wpath) {
        const char *env = getenv("METALFG_WEIGHTS");
        if (env && *env) wpath = env;
    }
    if (!wpath) FAIL("no weight file was given (weights_file is null and METALFG_WEIGHTS is unset or empty)");
    MetalFGWeightFile wf;
    char msg[256];
    if (metalfg_weights_load(wpath, &wf, msg, sizeof msg)) FAIL("%s", msg);
    char ppath[1100];
    snprintf(ppath, sizeof ppath, "%s.params", wpath);
    if (metalfg_params_load(ppath, &params, msg, sizeof msg)) { metalfg_weights_free(&wf); FAIL("%s", msg); }
    if (params.gx != METALFG_GX) { metalfg_weights_free(&wf); FAIL("parameter file entry gx is %d; the menu-detection grid is fixed at %d columns (fg.numerics_r3)", params.gx, METALFG_GX); }
    if (params.gy != METALFG_GY) { metalfg_weights_free(&wf); FAIL("parameter file entry gy is %d; the menu-detection grid is fixed at %d rows (fg.numerics_r3)", params.gy, METALFG_GY); }
    if (metalfg_plan_grids_ex(ci->display_w, ci->display_h, &params, ci->flags, &grids)) {
        metalfg_weights_free(&wf);
        if (ci->display_h > metalfg_height_cap(&params))
            FAIL("display height %u exceeds the height cap of %u (min(8192, 16 hb))", ci->display_h, metalfg_height_cap(&params));
        FAIL("display size %ux%u is not supported", ci->display_w, ci->display_h);
    }
    _weights = [dev newBufferWithLength:wf.size + 4096 options:MTLResourceStorageModeShared];
    memset(_weights.contents, 0, wf.size + 4096);
    memcpy(_weights.contents, wf.data, wf.size);
    metalfg_weights_free(&wf);

    /* shader library */
    NSString *dir = ci->metallib_dir ? @(ci->metallib_dir) : nil;
    if (!dir) {
        Dl_info di;
        if (dladdr((void *)&metalfg_depth_config, &di) && di.dli_fname) dir = [@(di.dli_fname) stringByDeletingLastPathComponent];
    }
    NSString *lib = [dir stringByAppendingPathComponent:@"metalfg.metallib"];
    NSError *nserr = nil;
    _library = lib ? [dev newLibraryWithURL:[NSURL fileURLWithPath:lib] error:&nserr] : nil;
    if (!_library) FAIL("shader library not found: %s", lib ? lib.UTF8String : "(no directory)");
    _pso = [NSMutableDictionary dictionary];
    for (NSString *k in kernel_names()) {
        id<MTLFunction> fn = [_library newFunctionWithName:k];
        if (!fn) FAIL("shader library lacks the function '%s'", k.UTF8String);
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&nserr];
        if (!p) FAIL("pipeline '%s' failed: %s", k.UTF8String, nserr.localizedDescription.UTF8String);
        _pso[k] = p;
    }

    /* networks and their pass helpers */
    NSString *nnerr = nil;
    _nn = [[MetalFGNN alloc] initWithDevice:dev library:_library weights:_weights leak:params.leak satN2:params.sat_n2 error:&nnerr];
    if (!_nn) FAIL("%s", nnerr ? nnerr.UTF8String : "network kernels unavailable");
    _output = [[MetalFGOutput alloc] initWithDevice:dev library:_library W:grids.W H:grids.H chain:!(info.flags & METALFG_CREATE_DIRECT_OUTPUT) levels:grids.pyr_out_levels];
    [_output prepareCells:grids.Nw * grids.Nh];
    if (!_output) FAIL("shader library lacks the output synthesis kernels");
    _blend = [[MetalFGBlend alloc] initWithDevice:dev library:_library];
    if (!_blend) FAIL("shader library lacks the blend kernels");
    [_blend specialiseForS:grids.s ui:(info.flags & METALFG_CREATE_UI_RECOMPOSE) != 0];
    _motion = [[MetalFGMotionPrep alloc] initWithDevice:dev library:_library];
    if (!_motion) FAIL("shader library lacks the motion preparation kernels");
    _chain = [[MetalFGChain alloc] initWithDevice:dev library:_library Wm:grids.Wm Hm:grids.Hm levels:grids.pyr_mot_levels];
    if (!_chain) FAIL("shader library lacks the motion chain kernels");
    _menu = [[MetalFGMenu alloc] initWithDevice:dev library:_library];
    if (!_menu) FAIL("shader library lacks the menu detection kernels");
    _io = [[MetalFGIO alloc] initWithDevice:dev library:_library];
    if (!_io) FAIL("shader library lacks the net1 packing kernels");
    _net1 = [[MetalFGNet1 alloc] initWithNN:_nn w1:grids.w1 h1:grids.h1];
    _net2 = [[MetalFGNet2 alloc] initWithNN:_nn Nw:grids.Nw Nh:grids.Nh];

    /* resources */
    MetalFGResDesc d[METALFG_MAX_RES];
    int nd = metalfg_plan_resources(&grids, info.flags, d);
    _slots = [NSMutableDictionary dictionary];
    NSMutableArray<id<MTLAllocation>> *allocs = [NSMutableArray array];
#ifdef METALFG_TEST
    MTLStorageMode sm = MTLStorageModeShared;
#else
    MTLStorageMode sm = MTLStorageModePrivate;
#endif
    for (int i = 0; i < nd; i++) {
        MetalFGSlot *s = [MetalFGSlot new];
        s.desc = d[i];
        NSString *key = d[i].level ? [NSString stringWithFormat:@"%s.%u", d[i].name, d[i].level] : @(d[i].name);
        int sides = d[i].pair ? 2 : 1;
        for (int sd = 0; sd < sides; sd++) {
            if (strcmp(d[i].format, "buffer") == 0) {
                NSUInteger len = (NSUInteger)d[i].w * d[i].h * d[i].bytes;
                if (len < 16) len = 16;
                MTLStorageMode bsm = strcmp(d[i].name, "menu.state") == 0 ? MTLStorageModeShared : sm;
                id<MTLBuffer> b = [dev newBufferWithLength:len options:(bsm == MTLStorageModeShared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate)];
                if (!b) FAIL("cannot allocate %s", d[i].name);
                if (bsm == MTLStorageModeShared) memset(b.contents, 0, len);
                if (sd) s.buf1 = b; else s.buf0 = b;
                [allocs addObject:b];
            } else {
                MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt_from_name(d[i].format)
                                                                                              width:d[i].w height:d[i].h mipmapped:NO];
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                td.storageMode = sm;
                id<MTLTexture> t = [dev newTextureWithDescriptor:td];
                if (!t) FAIL("cannot allocate %s", d[i].name);
                if (sd) s.tex1 = t; else s.tex0 = t;
                [allocs addObject:t];
            }
        }
        _slots[key] = s;
    }

    /* Metal 4 support objects */
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 16;
    atd.maxTextureBindCount = 24;
    _argTable = [dev newArgumentTableWithDescriptor:atd error:&nserr];
    _ring = [dev newBufferWithLength:METALFG_RING_SLOTS * METALFG_RING_SLOT options:MTLResourceStorageModeShared];
    [allocs addObject:_ring];
    [allocs addObject:_weights];
    [allocs addObjectsFromArray:_chain.allocations];
#if defined(METALFG_CTL_FEAT_WEIGHTS)
    {   // control: blend weights on the Feat grid
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:grids.Wf height:grids.Hf mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModePrivate;
        _ctlBwFeat = [dev newTextureWithDescriptor:d];
        [allocs addObject:_ctlBwFeat];
    }
#endif
    [allocs addObjectsFromArray:_output.levels];
    if (_output.holeOut) [allocs addObject:_output.holeOut];
    if (_output.fillTiles) [allocs addObject:_output.fillTiles];
    if (_output.cellSig) [allocs addObject:_output.cellSig];
    /* every specialised network pipeline is built now (both networks encoded once without an encoder: the host logic runs, nothing is recorded) */
    [_nn beginCollecting];
    [_net1 encode:nil in:[self buf:@"n1in" old:NO] out:[self buf:@"n1out" old:NO]];
    [_net2 encode:nil a:[self buf:@"n2a" old:NO] b:[self buf:@"n2b" old:NO] h:[self buf:@"n2h" old:NO] t:0.5f out:[self buf:@"n2out" old:NO]];
    [_nn buildCollected];
    [allocs addObjectsFromArray:_net1.tensors.allValues];
    [allocs addObjectsFromArray:_net2.tensors.allValues];
    [allocs addObjectsFromArray:_nn.extraBuffers];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = allocs.count + 8;
    _residency = [dev newResidencySetWithDescriptor:rd error:&nserr];
    if (_residency) {
        for (id<MTLAllocation> a in allocs) [_residency addAllocation:a];
        [_residency commit];
        [_residency requestResidency];
    }
    memset(&state, 0, sizeof state);
    return self;
}

/* ---- validation of the frame record against the textures ---- */
typedef struct { NSUInteger f[10]; } FmtList;

static int in_list(NSUInteger fmt, const NSUInteger *l, int n) {
    for (int i = 0; i < n; i++) if (l[i] == fmt) return 1;
    return 0;
}

static NSString *check_tex(void *obj, const char *what, const NSUInteger *fmts, int nf, int need_write, uint32_t ox, uint32_t oy,
                           uint32_t w, uint32_t h) {
    id<MTLTexture> t = (__bridge id<MTLTexture>)obj;
    if (t.textureType != MTLTextureType2D) return [NSString stringWithFormat:@"%s: texture is not 2D", what];
    if (!in_list(t.pixelFormat, fmts, nf)) return [NSString stringWithFormat:@"%s: pixel format %lu is not accepted", what, (unsigned long)t.pixelFormat];
    if (need_write && !(t.usage & MTLTextureUsageShaderWrite)) return [NSString stringWithFormat:@"%s: texture lacks shader-write usage", what];
    if (!need_write && !(t.usage & MTLTextureUsageShaderRead)) return [NSString stringWithFormat:@"%s: texture lacks shader-read usage", what];
    if ((uint64_t)ox + w > t.width || (uint64_t)oy + h > t.height)
        return [NSString stringWithFormat:@"%s: rectangle %ux%u at (%u, %u) leaves the %lux%lu texture", what, w, h, ox, oy, (unsigned long)t.width, (unsigned long)t.height];
    return nil;
}

- (NSString *)validate:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p {
    static const NSUInteger colour[] = {MTLPixelFormatRGBA16Float, MTLPixelFormatRG11B10Float, MTLPixelFormatRGB10A2Unorm,
                                        MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA8Unorm};
    static const NSUInteger depth[] = {MTLPixelFormatR32Float, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8,
                                       MTLPixelFormatDepth16Unorm};
    static const NSUInteger motion[] = {MTLPixelFormatRG16Float, MTLPixelFormatRG32Float};
    static const NSUInteger uic[] = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float};
    static const NSUInteger uia[] = {MTLPixelFormatR8Unorm, MTLPixelFormatR16Float};
    static const NSUInteger dist[] = {MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA32Float, MTLPixelFormatRG16Float, MTLPixelFormatRG32Float};
    static const NSUInteger outf[] = {MTLPixelFormatRGBA16Float, MTLPixelFormatRGB10A2Unorm, MTLPixelFormatRGBA8Unorm,
                                      MTLPixelFormatBGRA8Unorm, MTLPixelFormatRG11B10Float};
    NSString *e;
    uint32_t cw = f->content_w, ch = f->content_h;
    if ((e = check_tex(f->colour, "colour", colour, 5, 0, f->colour_off[0], f->colour_off[1], cw, ch))) return e;
    /* fg.engine_api_r2: the depth and motion rectangles have their own (effective) sizes */
    const uint32_t dzw = METALFG_DEPTH_W(f), dzh = METALFG_DEPTH_H(f), mvw = METALFG_MOTION_W(f), mvh = METALFG_MOTION_H(f);
    if (dzw < 1 || dzh < 1 || dzw > METALFG_MAX_WIDTH || dzh > METALFG_MAX_HEIGHT)
        return [NSString stringWithFormat:@"depth_size: effective size %ux%u outside 1..%u", dzw, dzh, METALFG_MAX_WIDTH];
    if (mvw < 1 || mvh < 1 || mvw > METALFG_MAX_WIDTH || mvh > METALFG_MAX_HEIGHT)
        return [NSString stringWithFormat:@"motion_size: effective size %ux%u outside 1..%u", mvw, mvh, METALFG_MAX_WIDTH];
    if ((e = check_tex(f->depth, (f->depth_size[0] || f->depth_size[1]) ? "depth (depth_size)" : "depth", depth, 4, 0, f->depth_off[0], f->depth_off[1], dzw, dzh))) return e;
    if ((e = check_tex(f->motion, (f->motion_size[0] || f->motion_size[1]) ? "motion (motion_size)" : "motion", motion, 2, 0, f->motion_off[0], f->motion_off[1], mvw, mvh))) return e;
    if (f->prev_colour && (e = check_tex(f->prev_colour, "prev_colour", colour, 5, 0, f->colour_off[0], f->colour_off[1], cw, ch))) return e;
    if (f->hudless && (e = check_tex(f->hudless, "hudless", colour, 5, 0, f->colour_off[0], f->colour_off[1], cw, ch))) return e;
    if (f->ui_colour && (e = check_tex(f->ui_colour, "ui_colour", uic, 3, 0, f->colour_off[0], f->colour_off[1], cw, ch))) return e;
    if (f->ui_alpha && (e = check_tex(f->ui_alpha, "ui_alpha", uia, 2, 0, f->colour_off[0], f->colour_off[1], cw, ch))) return e;
    if (f->distortion) {
        uint32_t dw = f->distortion_size[0] ? f->distortion_size[0] : cw, dh = f->distortion_size[1] ? f->distortion_size[1] : ch;
        if ((e = check_tex(f->distortion, "distortion", dist, 4, 0, f->distortion_off[0], f->distortion_off[1], dw, dh))) return e;
    }
    uint32_t W = grids.W, H = grids.H;
    if ((e = check_tex(f->out_interp, "out_interp", outf, 5, 1, f->out_off[0], f->out_off[1], W, H))) return e;
    if (f->out_real && (e = check_tex(f->out_real, "out_real", outf, 5, 1, f->out_off[0], f->out_off[1], W, H))) return e;
    if (f->skip_flag) {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)f->skip_flag;
        if (b.length < 1) return @"skip_flag: buffer is empty";
    }
    if (!isfinite(f->motion_scale_x) || !isfinite(f->motion_scale_y)) return @"motion scale is not finite";
    return nil;
}

/* ---- encoding ---- */


- (int)fail:(NSString *)m {
    self.lastError = m;
    return 1;
}

/* fg.engine_api_r2: the record length is fixed by the interface version given at create; under version 1 nothing past byte 335 is read (the two size fields
   are then zero, the content size) */
static const MetalFGFrameInfo *frame_copy(const MetalFGFrameInfo *f, uint32_t version, MetalFGFrameInfo *dst) {
    if (!f) return NULL;
    memset(dst, 0, sizeof *dst);
    memcpy(dst, f, version >= 2 ? METALFG_FRAME_RECORD_V2 : METALFG_FRAME_RECORD_V1);
    return dst;
}

- (int)encodeClassic:(id<MTLCommandBuffer>)cb frame:(const MetalFGFrameInfo *)fin {
    MetalFGFrameInfo fbuf;
    const MetalFGFrameInfo *f = frame_copy(fin, info.version, &fbuf);
    if (!cb) return [self fail:@"command buffer is null"];
    MetalFGCallPlan p;
    char err[200];
    if (metalfg_call_plan(&state, info.flags, info.max_generated, f, &p, err, sizeof err)) return [self fail:@(err)];
    NSString *v = [self validate:f plan:&p];
    if (v) return [self fail:v];
    MetalFGEnc *enc = [MetalFGEnc classicWith:cb];
    if (!enc) return [self fail:@"cannot create a compute encoder"];
    [self runCall:enc frame:f plan:&p];
    profStamps = 0;
    return 0;
}

- (int)encodeMetal4:(id<MTL4CommandBuffer>)cb frame:(const MetalFGFrameInfo *)fin {
    MetalFGFrameInfo fbuf;
    const MetalFGFrameInfo *f = frame_copy(fin, info.version, &fbuf);
    if (!cb) return [self fail:@"command buffer is null"];
    MetalFGCallPlan p;
    char err[200];
    if (metalfg_call_plan(&state, info.flags, info.max_generated, f, &p, err, sizeof err)) return [self fail:@(err)];
    NSString *v = [self validate:f plan:&p];
    if (v) return [self fail:v];
    if (!_argTable) return [self fail:@"Metal 4 argument tables are unavailable on this device"];
    NSUInteger slot = callCounter % METALFG_RING_SLOTS;
    MetalFGEnc *enc = [MetalFGEnc metal4With:cb table:_argTable ring:_ring ringBase:slot * METALFG_RING_SLOT ringLen:METALFG_RING_SLOT];
    if (!enc) return [self fail:@"cannot create a Metal 4 compute encoder"];
    [enc useResidencySet:_residency];
    /* caller textures are made resident for this call. A set is built per combination of caller objects and reused while the host passes the same objects again
       (the usual case: a few rotating images); the 16 most recent combinations are kept alive */
    {
        void *objs[11] = {f->colour, f->prev_colour, f->hudless, f->depth, f->motion, f->ui_colour, f->ui_alpha,
                          f->distortion, f->out_interp, f->out_real, f->skip_flag};
        NSData *key = [NSData dataWithBytes:objs length:sizeof objs];
        if (!_callerSets) _callerSets = [NSMutableArray array];
        if (!_callerKeys) _callerKeys = [NSMutableArray array];
        NSUInteger hit = [_callerKeys indexOfObject:key];
        id<MTLResidencySet> cs = nil;
        if (hit != NSNotFound) {
            cs = _callerSets[hit];
        } else {
            cs = [_device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
            if (cs) {
                for (size_t i = 0; i < 11; i++)
                    if (objs[i]) [cs addAllocation:(__bridge id<MTLAllocation>)objs[i]];
                [cs commit];
                [cs requestResidency];
                [_callerSets addObject:cs];
                [_callerKeys addObject:key];
                if (_callerSets.count > 16) { [_callerSets removeObjectAtIndex:0]; [_callerKeys removeObjectAtIndex:0]; }
            }
        }
        if (cs) [enc useResidencySet:cs];
    }
    int heap = -1;
    if (info.profile) {
        if (!self.profHeaps) {
            NSMutableArray *hs = [NSMutableArray array];
            MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new];
            hd.type = MTL4CounterHeapTypeTimestamp;
            hd.count = 512;
            for (int i = 0; i < 4; i++) {
                id<MTL4CounterHeap> h = [_device newCounterHeapWithDescriptor:hd error:nil];
                if (h) [hs addObject:h];
            }
            self.profHeaps = hs;
        }
        if (self.profHeaps.count) {
            heap = (int)(callCounter % self.profHeaps.count);
            [enc profileWithHeap:self.profHeaps[heap] capacity:512];
        }
    }
    [self runCall:enc frame:f plan:&p];
    if (heap >= 0) {
        profHeap = heap;
        profStamps = enc.stampCount;
        for (uint32_t i = 0; i < profStamps; i++) profLabels[i] = [enc stampLabel:i];
    }
    return 0;
}

@end
