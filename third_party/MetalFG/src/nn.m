#import <Foundation/Foundation.h>
#include <string.h>
#include "nn.h"
#include "passes_impl.h"

typedef struct {
    uint32_t W, H, cin, in_stride, in_base, cout, out_stride, out_base, k, row0, cin_w, flags;
    float leak, bound;
    uint32_t res_stride, res_base, bound_from, nest;
    uint32_t up_ws, up_hs;
} ConvParamsGPU;

/* layers served by the tensor-operation kernel by default (measured per layer, docs/STATUS.md) */
/* every layer with cin a multiple of 16; the narrow heads use the 8-channel tile (conv_mx8). Lane diagnostics: METALFG_NN_ONLY=a,b serves only the named weight tensors on the
   matrix unit, METALFG_NN_SKIP=a,b serves all but those (the others run on the pinned register-blocked kernel) */
static BOOL metalfg_listed(const char *v, const char *name) {
    if (!v) return NO;
    const size_t n = strlen(name);
    for (const char *p = v; (p = strstr(p, name)); p++) if ((p == v || p[-1] == ',') && (p[n] == ',' || p[n] == 0)) return YES;
    return NO;
}
static BOOL metalfg_mx_layer(const char *name) {
    static const char *only, *skip;
    static BOOL init;
    if (!init) { only = getenv("METALFG_NN_ONLY"); skip = getenv("METALFG_NN_SKIP"); init = YES; }
    return only ? metalfg_listed(only, name) : !metalfg_listed(skip, name);
}

/* threadgroup of an elementwise kernel on a (C / 4, x, y) grid: all channel groups of a row of pixels, about 256 threads */
static MTLSize metalfg_tg_c4(NSUInteger c4) {
    const NSUInteger cx = c4 < 256 ? c4 : 256;
    return MTLSizeMake(cx, MAX((NSUInteger)1, 256 / cx), 1);
}

@implementation MetalFGNN {
    id<MTLDevice> _dev;
    id<MTLLibrary> _lib;
    NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *_spec;   /* conv_mx* specialised per layer shape (function constants of nn_mx.metal) */
    NSMutableSet<NSNumber *> *_wanted;                                       /* shapes requested while collecting (prepareSpecialised) */
    id<MTLBuffer> _w, _wpad;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *_pso;
    float _leak, _sat;
}

/* channels per thread of the elementwise kernels (four); the up_half_add control kernel keeps one */
#if defined(METALFG_CTL_UP_HALF)
#define UP_ADD_LANES 1u
#else
#define UP_ADD_LANES 4u   /* applies when C is a multiple of four; other counts use the scalar kernels */
#endif
+ (NSArray<NSString *> *)kernelNames {
    NSMutableArray *a = [NSMutableArray arrayWithArray:@[ @"conv_pin", @"conv_blk4x8", @"conv_blk4x4", @"conv_mx", @"conv_mx8", @"conv_mxr", @"conv_mxr8", @"conv_mxr16", @"conv_pair1_16", @"conv_pair2_32", @"pool_max2", @"up_corner_add", @"up2_half", @"up2_extrap", @"pool_max2_s", @"up_corner_add_s", @"up2_half_s", @"up2_extrap_s", @"n2_pack_x0", @"n2_join", @"n2_final" ]];
#if defined(METALFG_CTL_UP_HALF)
    [a addObject:@"up_half_add"];
#endif
    return a;
}

- (id<MTLDevice>)device { return _dev; }
- (id<MTLBuffer>)weights { return _w; }
- (NSArray<id<MTLBuffer>> *)extraBuffers { return @[ _wpad ]; }
- (float)leak { return _leak; }
- (float)satN2 { return _sat; }

- (instancetype)initWithDevice:(id<MTLDevice>)dev library:(id<MTLLibrary>)lib weights:(id<MTLBuffer>)w leak:(float)leak satN2:(float)sat error:(NSString **)err {
    self = [super init];
    _dev = dev; _w = w; _leak = leak; _sat = sat; _lib = lib;
    _spec = [NSMutableDictionary dictionary];
    _pso = [NSMutableDictionary dictionary];
    for (NSString *k in [MetalFGNN kernelNames]) {
        NSError *fe = nil;      /* through constantValues: the matrix kernels declare function constants (none set here: they read the parameters) */
        id<MTLFunction> fn = [lib newFunctionWithName:k constantValues:[MTLFunctionConstantValues new] error:&fe];
        NSError *e = nil;
        id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!p) { if (err) *err = [NSString stringWithFormat:@"shader library lacks a working '%@'", k]; return nil; }
        _pso[k] = p;
    }
    /* n2.b.d0 stores 18 input columns per tap; the kernels read multiples of 8, so a copy padded to 32 columns is built */
    int bi = metalfg_weight_index("n2.b.d0.w");
    const MetalFGTensorInfo *t = &metalfg_weight_table()[bi];
    _wpad = [dev newBufferWithLength:(NSUInteger)t->O * 9 * 32 * 2 + 4096 options:MTLResourceStorageModeShared];
    memset(_wpad.contents, 0, _wpad.length);
    const uint16_t *src = (const uint16_t *)((const uint8_t *)w.contents + t->offset);
    uint16_t *dst = _wpad.contents;
    for (uint32_t o = 0; o < t->O; o++)
        for (uint32_t tap = 0; tap < 9; tap++)
            memcpy(dst + ((size_t)o * 9 + tap) * 32, src + ((size_t)o * 9 + tap) * 18, 18 * 2);
    return self;
}

- (id<MTLBuffer>)scratch:(NSUInteger)halves {
    id<MTLBuffer> b = [_dev newBufferWithLength:halves * 2 + 64 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, b.length);
    return b;
}

- (BOOL)matrixPath:(const char *)wname cin:(uint32_t)cin {
    const char *v = getenv("METALFG_NN_PATH");
    const int mode = (v && !strcmp(v, "scalar")) ? 1 : (v && !strcmp(v, "mx")) ? 2 : (v && !strcmp(v, "blk")) ? 3 : 0;
#if defined(METALFG_CTL_ACC3)
    if (!strncmp(wname, "n2.", 3)) return NO;
#endif
    return (cin % 16) == 0 && (mode == 0 || mode == 2) && metalfg_mx_layer(wname);
}

/* specialised matrix kernels: key = kernel index (bits 0..3), flags (4..15), kernel size (16..19), channel groups (20..27), decoder fold (28) */
static NSString *const MXNAMES[8] = {@"conv_mx", @"conv_mx8", @"conv_mxr", @"conv_mxr8", @"conv_mxr16", @"conv_mxr_s2", @"conv_mxr16_s2", @"conv_mxr8_s2"};

- (id<MTLComputePipelineState>)buildMX:(uint64_t)key {
    uint32_t flags = (uint32_t)(key >> 4) & 0xFFFu, k = (uint32_t)(key >> 16) & 0xFu, nch = (uint32_t)(key >> 20) & 0xFFu;
    const bool upb = ((key >> 28) & 1u) != 0;
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&flags type:MTLDataTypeUInt atIndex:0];
    [cv setConstantValue:&k type:MTLDataTypeUInt atIndex:1];
    [cv setConstantValue:&nch type:MTLDataTypeUInt atIndex:2];
    [cv setConstantValue:&upb type:MTLDataTypeBool atIndex:3];
    NSError *e = nil;
    id<MTLFunction> fn = [_lib newFunctionWithName:MXNAMES[key & 0xFu] constantValues:cv error:&e];
    id<MTLComputePipelineState> p = fn ? [_dev newComputePipelineStateWithFunction:fn error:&e] : nil;
    if (p) return p;
    /* not expected; the unspecialised pipeline computes the same values (built here for the kernels that init does not build: the two-simdgroup forms) */
    if (!_pso[MXNAMES[key & 0xFu]]) {
        id<MTLFunction> f0 = [_lib newFunctionWithName:MXNAMES[key & 0xFu] constantValues:[MTLFunctionConstantValues new] error:&e];
        id<MTLComputePipelineState> p0 = f0 ? [_dev newComputePipelineStateWithFunction:f0 error:&e] : nil;
        if (p0) _pso[MXNAMES[key & 0xFu]] = p0;
    }
    return _pso[MXNAMES[key & 0xFu]];
}

/* the pipeline of a matrix kernel with the layer's flags, kernel size, channel groups and decoder fold as function constants. While collecting, the shape is
   recorded and the unspecialised pipeline returned; otherwise a missing shape is built here (the engine prepares all shapes at create). */
- (id<MTLComputePipelineState>)psoMX:(uint32_t)base flags:(uint32_t)flags k:(uint32_t)k nch:(uint32_t)nch up:(BOOL)up {
    const uint64_t key = (uint64_t)base | ((uint64_t)(flags & 0xFFFu) << 4) | ((uint64_t)(k & 0xFu) << 16) | ((uint64_t)(nch & 0xFFu) << 20) | ((uint64_t)(up ? 1 : 0) << 28);
    NSNumber *nk = @(key);
    id<MTLComputePipelineState> p = _spec[nk];
    if (p) return p;
    if (_wanted) { [_wanted addObject:nk]; return _pso[MXNAMES[base]] ? _pso[MXNAMES[base]] : _pso[MXNAMES[2]]; }   /* the dry run only records the shape */
    p = [self buildMX:key];
    _spec[nk] = p;
    return p;
}

- (void)beginCollecting { _wanted = [NSMutableSet set]; }

/* builds every collected shape, in parallel */
- (void)buildCollected {
    NSArray<NSNumber *> *keys = _wanted.allObjects;
    _wanted = nil;
    __block id<MTLComputePipelineState> __strong *out = (id<MTLComputePipelineState> __strong *)calloc(keys.count, sizeof(id));
    dispatch_apply(keys.count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) { out[i] = [self buildMX:keys[i].unsignedLongLongValue]; });
    for (NSUInteger i = 0; i < keys.count; i++) { _spec[keys[i]] = out[i]; out[i] = nil; }
    free(out);
}

- (void)conv:(MetalFGEnc *)enc args:(MetalFGConvArgs)a {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    const MetalFGTensorInfo *wt = &T[a.widx], *bt = &T[a.widx + 1];
    ConvParamsGPU p;
    p.W = a.W; p.H = a.H; p.cin = a.cin; p.in_stride = a.in_stride; p.in_base = a.in_base;
    p.cout = a.cout; p.out_stride = a.out_stride; p.out_base = a.out_base;
    p.k = wt->kh; p.row0 = a.row0; p.cin_w = wt->I; p.flags = a.flags;
    p.leak = _leak; p.bound = _sat; p.res_stride = a.res_stride; p.res_base = a.res_base; p.bound_from = a.bound_from; p.nest = a.nest ? a.nest : 3;
    p.up_ws = a.up_ws; p.up_hs = a.up_hs;
#if defined(METALFG_CTL_ORDER_ROT)
    { static int rc = -1; if (rc < 0) { const char *v = getenv("METALFG_RUN_COUNTER"); rc = v ? atoi(v) : 0; } if (rc & 1) p.flags |= 64u; }   /* control of tol-det */
#endif
    id<MTLBuffer> wb = _w;
    NSUInteger woff = wt->offset;
    if (a.widx == metalfg_weight_index("n2.b.d0.w")) { wb = _wpad; woff = 0; p.cin_w = 32; }
#if defined(METALFG_CTL_ACC_ORDER)
    if (a.widx == metalfg_weight_index("n2.a.res3.w")) p.flags |= 32u;      /* control 3: this layer sums its channel groups in the opposite order */
#endif
    /* the matrix path serves layers whose channel count is a multiple of 16 and whose output count is a multiple of 8 (fg.metal_techniques_r4 T1) */
    static int mode = -1;     /* lane diagnostic METALFG_NN_PATH=scalar forces the generic kernel for the ablation of fg.metal_techniques_r4 T12 */
    if (mode < 0) { const char *v = getenv("METALFG_NN_PATH"); mode = (v && !strcmp(v, "scalar")) ? 1 : (v && !strcmp(v, "mx")) ? 2 : (v && !strcmp(v, "blk")) ? 3 : 0; }
    /* the register-blocked kernel serves layers whose channel count is a multiple of 16 and whose output count is a multiple of 8 (the heads of 4, 1 and 3 outputs
       keep the generic kernel) */
    /* the tensor-operation kernel (fg.arithmetic_tolerance_r1): layers named in metalfg_mx_layer, or every eligible layer with METALFG_NN_PATH=mx */
    BOOL mx = (p.cin % 16) == 0 && (mode == 0 || mode == 2) && metalfg_mx_layer(wt->name);
#if defined(METALFG_CTL_ACC3)
    if (!strncmp(wt->name, "n2.", 3)) mx = NO;          /* control of tol-budget: the second network runs on the generic kernel with a three-bit accumulator */
#endif
    BOOL blk = mode != 1 && !mx && (p.cin % 16) == 0;
#if defined(METALFG_CTL_ACC3)
    if (!strncmp(wt->name, "n2.", 3)) blk = NO;
#endif
    static int tcForce = -1, tpForce = -1;
    if (tcForce < 0) { const char *v = getenv("METALFG_BLK_TC"); tcForce = v ? atoi(v) : 0; v = getenv("METALFG_BLK_TP"); tpForce = v ? atoi(v) : 0; }
    /* measured per layer (docs/STATUS.md): positions and output channels per thread */
    NSUInteger tp = tpForce ? (NSUInteger)tpForce : 4;
    NSUInteger tc = tcForce ? (NSUInteger)tcForce : (!strcmp(wt->name, "n1.dec0a.w") || !strcmp(wt->name, "n1.head.w") ? 8 : 4);
    const NSUInteger per = (a.flags & METALFG_CONV_POOL) ? tp / 4 : tp;
    const NSUInteger npb = ((((a.flags & METALFG_CONV_POOL) ? a.W / 2 : a.W) + per - 1) / per) * ((a.flags & METALFG_CONV_POOL) ? a.H / 2 : a.H);
    NSString *kn = [NSString stringWithFormat:@"conv_blk%lux%lu", (unsigned long)tp, (unsigned long)tc];
    /* default: the matrix unit with the pin's rounding after every pair (conv_mxr*); METALFG_NN_PATH=mx selects the unrounded sum of the unit (conv_mx*) */
    /* tile width (output channels): 8 for the narrow heads, 16 for sixteen-output layers on the pinned-rounding path, else 32 */
    const BOOL blocks0 = (a.flags & (METALFG_CONV_POOL | METALFG_CONV_MAXPOOL)) != 0;
    const NSUInteger NTw = (a.cout <= 8 && !blocks0) ? 8 : (a.cout == 16 && mode == 0) ? 16 : 32;
    /* simdgroups per threadgroup: two for the second network's layers on the pinned-rounding path (measured; four elsewhere) */
    const NSUInteger SGc = (mode == 0 && !strncmp(wt->name, "n2.", 3)) ? 2 : 4;
    if (mx) {
        uint32_t base = NTw == 8 ? (mode == 2 ? 1u : 3u) : NTw == 16 ? 4u : (mode == 2 ? 0u : 2u);
        if (SGc == 2) base = NTw == 32 ? 5u : NTw == 16 ? 6u : 7u;
        [enc setPipeline:[self psoMX:base flags:p.flags k:p.k nch:p.cin / 16 up:p.up_ws != 0]];
    } else {
        [enc setPipeline:_pso[blk ? kn : @"conv_pin"]];
    }
    [enc setBytes:&p length:sizeof p at:0];
    [enc bindBuffer:a.in offset:0 at:1];
    [enc bindBuffer:a.out offset:0 at:2];
    [enc bindBuffer:wb offset:woff at:3];
    [enc bindBuffer:_w offset:bt->offset at:4];
    [enc bindBuffer:(a.res ? a.res : a.in) offset:0 at:5];
    [enc bindBuffer:(a.pout ? a.pout : a.out) offset:0 at:6];
    [enc bindBuffer:(a.usrc ? a.usrc : a.in) offset:0 at:7];
    if (mx) {
        /* the order of the (tap, group) pairs (fg.arithmetic_pin_r4 rule B for the pinned-rounding kernels, tap-major for the unrounded ones): entry s = group | (column
           offset + 1) << 8 | (row offset + 1) << 10 | (first weight column of the tap / 8) << 12 */
        uint32_t tab[9 * 64];
        const uint32_t taps = p.k == 3 ? 9 : 1, nch = p.cin / 16, np = taps * nch;
        const BOOL rnd = mode != 2;
        NSAssert(np <= 9 * 64, @"pair table");
        for (uint32_t s = 0; s < np; s++) {
            uint32_t tap, g;
            if (taps == 1) { tap = 0; g = s; }
            else if (rnd && p.nest == 1) { g = s / 9; const uint32_t r = s % 9; tap = (r % 3) * 3 + r / 3; }
            else if (rnd && p.nest == 2) { const uint32_t kx = s / (3 * nch), r = s % (3 * nch); g = r / 3; tap = (r % 3) * 3 + kx; }
            else { tap = s / nch; g = s % nch; }
            const uint32_t dxp = taps == 9 ? tap % 3 : 1, dyp = taps == 9 ? tap / 3 : 1;
            NSAssert((tap * p.cin_w) % 8 == 0 && tap * p.cin_w / 8 < (1u << 20), @"pair table weight offset");
            tab[s] = g | (dxp << 8) | (dyp << 10) | ((tap * p.cin_w / 8) << 12);
        }
        [enc setBytes:tab length:np * 4 at:9];
    }
    NSAssert(mx || !(a.flags & METALFG_CONV_MAXPOOL), @"the fused maximum exists on the matrix path only");
    NSAssert((mx && wt->kh == 1 && (a.cin % 16) == 0) || !a.up_ws, @"the folded enlargement exists for 1 x 1 layers on the matrix path only");
    [enc barrier];
    if (mx) {
        const BOOL blocks = (a.flags & (METALFG_CONV_POOL | METALFG_CONV_MAXPOOL)) != 0;
        const NSUInteger npos = blocks ? (NSUInteger)(a.W / 2) * (a.H / 2) : (NSUInteger)a.W * a.H;
        const NSUInteger ntiles = ((npos + (blocks ? 8 : 32) - 1) / (blocks ? 8 : 32)) * ((a.cout + NTw - 1) / NTw);
        [enc dispatchThreads:MTLSizeMake(((ntiles + SGc - 1) / SGc) * 32 * SGc, 1, 1) threadsPerGroup:MTLSizeMake(32 * SGc, 1, 1)];
    } else if (blk) {
        [enc dispatchThreads:MTLSizeMake(npb * ((a.cout + tc - 1) / tc), 1, 1) threadsPerGroup:MTLSizeMake(64, 1, 1)];
    } else {
        const NSUInteger pix = (a.flags & METALFG_CONV_POOL) ? (NSUInteger)(a.W / 2) * (a.H / 2) : (NSUInteger)a.W * a.H;
        [enc dispatchThreads:MTLSizeMake(pix * a.cout, 1, 1) threadsPerGroup:MTLSizeMake(64, 1, 1)];
    }
    [enc stamp:T[a.widx].name];
}

static ConvParamsGPU conv_params(MetalFGConvArgs a, float leak, float sat) {
    const MetalFGTensorInfo *wt = &metalfg_weight_table()[a.widx];
    ConvParamsGPU p;
    memset(&p, 0, sizeof p);
    p.W = a.W; p.H = a.H; p.cin = a.cin; p.in_stride = a.in_stride; p.in_base = a.in_base;
    p.cout = a.cout; p.out_stride = a.out_stride; p.out_base = a.out_base;
    p.k = wt->kh; p.row0 = a.row0; p.cin_w = wt->I; p.flags = a.flags;
    p.leak = leak; p.bound = sat; p.res_stride = a.res_stride; p.res_base = a.res_base; p.bound_from = a.bound_from; p.nest = a.nest ? a.nest : 3;
    p.up_ws = a.up_ws; p.up_hs = a.up_hs;
    return p;
}

- (void)convPair:(MetalFGEnc *)enc first:(MetalFGConvArgs)a second:(MetalFGConvArgs)b {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    const BOOL two = a.cout == 64;
    NSAssert((two ? (b.cin == 64 && b.cout <= 32) : (a.cout == 32 && b.cin == 32 && b.cout <= 16)) && T[a.widx].kh == 1 && T[b.widx].kh == 1 &&
             !(a.flags & ~METALFG_CONV_ACT) && !(b.flags & ~METALFG_CONV_ACT), @"conv_pair: two plain 1 x 1 layers of 32 and at most 16, or 64 and at most 32 outputs");
    ConvParamsGPU p1 = conv_params(a, _leak, _sat), p2 = conv_params(b, _leak, _sat);
    [enc setPipeline:_pso[two ? @"conv_pair2_32" : @"conv_pair1_16"]];
    [enc setBytes:&p1 length:sizeof p1 at:0];
    [enc bindBuffer:a.in offset:0 at:1];
    [enc bindBuffer:b.out offset:0 at:2];
    [enc bindBuffer:_w offset:T[a.widx].offset at:3];
    [enc bindBuffer:_w offset:T[a.widx + 1].offset at:4];
    [enc setBytes:&p2 length:sizeof p2 at:5];
    [enc bindBuffer:_w offset:T[b.widx].offset at:6];
    [enc bindBuffer:(a.usrc ? a.usrc : a.in) offset:0 at:7];
    [enc bindBuffer:_w offset:T[b.widx + 1].offset at:8];
    [enc barrier];
    const NSUInteger ntiles = ((NSUInteger)a.W * a.H + 31) / 32;
    [enc dispatchThreads:MTLSizeMake(((ntiles + 3) / 4) * 128, 1, 1) threadsPerGroup:MTLSizeMake(128, 1, 1)];
    [enc stamp:T[a.widx].name];
}

- (void)poolMax:(MetalFGEnc *)enc in:(id<MTLBuffer>)in out:(id<MTLBuffer>)out W:(uint32_t)W H:(uint32_t)H C:(uint32_t)C {
    uint32_t p[3] = {W, H, C};
    [enc setPipeline:_pso[C % 4 == 0 ? @"pool_max2" : @"pool_max2_s"]];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:in offset:0 at:1];
    [enc bindBuffer:out offset:0 at:2];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(W / 2) * (H / 2) * (C % 4 == 0 ? C / 4 : C), 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc stamp:"n1.pool"];
}

- (void)upCornerAdd:(MetalFGEnc *)enc src:(id<MTLBuffer>)src skip:(id<MTLBuffer>)skip out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C {
    uint32_t p[3] = {Ws, Hs, C};
#if defined(METALFG_CTL_UP_HALF)
    [enc setPipeline:_pso[@"up_half_add"]];                 // control: half-pixel-centre rule in the first network's decoder
#else
    [enc setPipeline:_pso[C % 4 == 0 ? @"up_corner_add" : @"up_corner_add_s"]];
#endif
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:src offset:0 at:1];
    [enc bindBuffer:skip offset:0 at:2];
    [enc bindBuffer:out offset:0 at:3];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)Ws * 2 * Hs * 2 * (C % 4 == 0 ? C / UP_ADD_LANES : C), 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc stamp:"n1.up"];
}

- (void)up2:(MetalFGEnc *)enc src:(id<MTLBuffer>)src out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C {
    uint32_t p[3] = {Ws, Hs, C};
    [enc setPipeline:_pso[C % 4 == 0 ? @"up2_half" : @"up2_half_s"]];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:src offset:0 at:1];
    [enc bindBuffer:out offset:0 at:2];
    [enc barrier];
    if (C % 4 == 0) [enc dispatchThreads:MTLSizeMake(C / 4, (NSUInteger)Ws * 2, (NSUInteger)Hs * 2) threadsPerGroup:metalfg_tg_c4(C / 4)];
    else [enc dispatchThreads:MTLSizeMake((NSUInteger)Ws * 2 * Hs * 2 * C, 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc stamp:"n2.up2"];
}

- (void)up2x:(MetalFGEnc *)enc src:(id<MTLBuffer>)src out:(id<MTLBuffer>)out Ws:(uint32_t)Ws Hs:(uint32_t)Hs C:(uint32_t)C {
    uint32_t p[3] = {Ws, Hs, C};
    [enc setPipeline:_pso[C % 4 == 0 ? @"up2_extrap" : @"up2_extrap_s"]];
    [enc setBytes:p length:sizeof p at:0];
    [enc bindBuffer:src offset:0 at:1];
    [enc bindBuffer:out offset:0 at:2];
    [enc barrier];
    if (C % 4 == 0) [enc dispatchThreads:MTLSizeMake(C / 4, (NSUInteger)Ws * 2, (NSUInteger)Hs * 2) threadsPerGroup:metalfg_tg_c4(C / 4)];
    else [enc dispatchThreads:MTLSizeMake((NSUInteger)Ws * 2 * Hs * 2 * C, 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc stamp:"n2.up2x"];
}

- (id<MTLComputePipelineState>)pipeline:(NSString *)n { return _pso[n]; }

typedef struct { uint32_t Nw, Nh; float t; } N2P;

- (void)packX0:(MetalFGEnc *)enc a:(id<MTLBuffer>)a b:(id<MTLBuffer>)b h:(id<MTLBuffer>)h out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh t:(float)t {
    N2P P = {Nw, Nh, t};
    id<MTLComputePipelineState> pk = _pso[@"n2_pack_x0"];
    [enc setPipeline:pk];
    [enc setBytes:&P length:sizeof P at:0];
    [enc bindBuffer:a offset:0 at:1]; [enc bindBuffer:b offset:0 at:2]; [enc bindBuffer:h offset:0 at:3]; [enc bindBuffer:out offset:0 at:4];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(Nw, Nh, 1) threadsPerGroup:metalfg_tg2d(pk)];
    [enc stamp:"n2.pack_x0"];
}

- (void)join:(MetalFGEnc *)enc a:(id<MTLBuffer>)a b:(id<MTLBuffer>)b h:(id<MTLBuffer>)h u:(id<MTLBuffer>)u out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh t:(float)t {
    N2P P = {Nw, Nh, t};
    id<MTLComputePipelineState> jn = _pso[@"n2_join"];
    [enc setPipeline:jn];
    [enc setBytes:&P length:sizeof P at:0];
    [enc bindBuffer:a offset:0 at:1]; [enc bindBuffer:b offset:0 at:2]; [enc bindBuffer:h offset:0 at:3];
    [enc bindBuffer:u offset:0 at:4]; [enc bindBuffer:out offset:0 at:5];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake(Nw, Nh, 1) threadsPerGroup:metalfg_tg2d(jn)];
    [enc stamp:"n2.join"];
}

- (void)final:(MetalFGEnc *)enc k:(id<MTLBuffer>)k u:(id<MTLBuffer>)u out:(id<MTLBuffer>)out Nw:(uint32_t)Nw Nh:(uint32_t)Nh {
    N2P P = {Nw, Nh, 0};
    [enc setPipeline:_pso[@"n2_final"]];
    [enc setBytes:&P length:sizeof P at:0];
    [enc bindBuffer:k offset:0 at:1]; [enc bindBuffer:u offset:0 at:2]; [enc bindBuffer:out offset:0 at:3];
    [enc barrier];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)Nw * Nh * 8, 1, 1) threadsPerGroup:MTLSizeMake(256, 1, 1)];
    [enc stamp:"n2.final"];
}
@end

/* ---------------- net1 ---------------- */

static MetalFGConvArgs conv1(MetalFGNN *nn, const char *stem, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t W, uint32_t H, uint32_t flags) {
    char name[64];
    snprintf(name, sizeof name, "%s.w", stem);
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H; a.cin = t->I; a.in_stride = t->I; a.cout = t->O; a.out_stride = t->O;
    a.flags = flags; a.widx = wi;
    /* fg.arithmetic_pin_r4 rule B: the first, third and fourth encoder convolutions take groups outermost, the fifth takes the horizontal tap outermost */
    if (strstr(stem, "enc0") || strstr(stem, "enc2") || strstr(stem, "enc3")) a.nest = 1;
    else if (strstr(stem, "enc4")) a.nest = 2;
    return a;
}

@implementation MetalFGNet1 {
    MetalFGNN *_nn;
    uint32_t _w1, _h1;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *_t;
}

- (NSDictionary<NSString *, id<MTLBuffer>> *)tensors { return _t; }

- (instancetype)initWithNN:(MetalFGNN *)nn w1:(uint32_t)w1 h1:(uint32_t)h1 {
    self = [super init];
    _nn = nn; _w1 = w1; _h1 = h1;
    _t = [NSMutableDictionary dictionary];
    static const struct { const char *name; int level; int ch; } T[] = {
        {"s0", 0, 32}, {"p0", 1, 32}, {"s1", 1, 64}, {"p1", 2, 64}, {"s2", 2, 128}, {"p2", 3, 128}, {"s3", 3, 128},
        {"p3", 4, 128}, {"s4", 4, 256}, {"p4", 5, 256}, {"mid", 5, 512}, {"z", 5, 256},
        {"q4", 4, 256}, {"t4", 4, 256}, {"u4", 4, 128}, {"q3", 3, 128}, {"t3", 3, 128}, {"u3", 3, 128},
        {"q2", 2, 128}, {"t2", 2, 128}, {"u2", 2, 64}, {"q1", 1, 64}, {"t1", 1, 64}, {"u1", 1, 32},
        {"q0", 0, 32}, {"v", 0, 32},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        _t[@(T[i].name)] = [nn scratch:(NSUInteger)(w1 >> T[i].level) * (h1 >> T[i].level) * T[i].ch];
    return self;
}

- (void)encode:(MetalFGEnc *)enc in:(id<MTLBuffer>)n1in out:(id<MTLBuffer>)n1out {
    MetalFGNN *nn = _nn;
    uint32_t w[6], h[6];
    for (int k = 0; k < 6; k++) { w[k] = _w1 >> k; h[k] = _h1 >> k; }
    id<MTLBuffer> (^B)(NSString *) = ^id<MTLBuffer>(NSString *n) { return self->_t[n]; };
    const uint32_t A = METALFG_CONV_ACT;
    /* encoder level k: s_k = conv(p_{k-1}), p_k = 2 x 2 maximum of s_k (on the matrix path in the convolution's epilogue, else a separate pass) */
    static const char *E[5] = {"n1.enc0", "n1.enc1", "n1.enc2", "n1.enc3", "n1.enc4"};
    static const char *SN[5] = {"s0", "s1", "s2", "s3", "s4"}, *PN[5] = {"p0", "p1", "p2", "p3", "p4"};
    static const uint32_t EC[5] = {32, 64, 128, 128, 256};
    for (int k = 0; k < 5; k++) {
        MetalFGConvArgs a = conv1(nn, E[k], k ? B(@(PN[k - 1])) : n1in, B(@(SN[k])), w[k], h[k], A);
        char wn[32];
        snprintf(wn, sizeof wn, "%s.w", E[k]);
        const BOOL fuse = [nn matrixPath:wn cin:a.cin] && (w[k] % 2) == 0 && (h[k] % 2) == 0;
        if (fuse) { a.flags |= METALFG_CONV_MAXPOOL; a.pout = B(@(PN[k])); }
        [nn conv:enc args:a];
        if (!fuse) [nn poolMax:enc in:B(@(SN[k])) out:B(@(PN[k])) W:w[k] H:h[k] C:EC[k]];
    }
    [nn conv:enc args:conv1(nn, "n1.mida", B(@"p4"), B(@"mid"), w[5], h[5], A)];
    [nn conv:enc args:conv1(nn, "n1.midb", B(@"mid"), B(@"z"), w[5], h[5], A)];
    /* decoder levels 4..1: q = up(prev) + s; u = act(conv(act(conv(q)))) */
    static const char *U[] = {"u4", "u3", "u2", "u1"};
    static const char *S[] = {"s4", "s3", "s2", "s1"};
    static const char *Q[] = {"q4", "q3", "q2", "q1"};
    static const char *Tm[] = {"t4", "t3", "t2", "t1"};
    static const char *LA[] = {"n1.dec4a", "n1.dec3a", "n1.dec2a", "n1.dec1a"};
    static const char *LB[] = {"n1.dec4b", "n1.dec3b", "n1.dec2b", "n1.dec1b"};
    id<MTLBuffer> prev = B(@"z");
    uint32_t prevC = 256;
    /* q = up(prev) + s feeds a 1 x 1 layer: on the matrix path the layer forms q while loading its input (up_corner_add's values), else q is stored first */
    BOOL (^decA)(const char *, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) =
        ^BOOL(const char *stem, id<MTLBuffer> src, id<MTLBuffer> skip, id<MTLBuffer> q, id<MTLBuffer> outb, uint32_t C, uint32_t W, uint32_t H, uint32_t Ws, uint32_t Hs,
              uint32_t flags) {
        char wn[32];
        snprintf(wn, sizeof wn, "%s.w", stem);
        MetalFGConvArgs a = conv1(nn, stem, q, outb, W, H, flags);
        if ([nn matrixPath:wn cin:a.cin] && a.cin == C && (C % 16) == 0) {
            a.in = skip; a.usrc = src; a.up_ws = Ws; a.up_hs = Hs;
            [nn conv:enc args:a];
            return YES;
        }
        [nn upCornerAdd:enc src:src skip:skip out:q Ws:Ws Hs:Hs C:C];
        [nn conv:enc args:a];
        return NO;
    };
    for (int i = 0; i < 4; i++) {
        int lv = 4 - i;
        /* level 1 (64 -> 64 -> 32): both layers in one pass on the matrix path (t1 is not stored) */
        MetalFGConvArgs la = conv1(nn, LA[i], B(@(S[i])), B(@(Tm[i])), w[lv], h[lv], A), lb = conv1(nn, LB[i], B(@(Tm[i])), B(@(U[i])), w[lv], h[lv], A);
        char wa[32], wb[32];
        snprintf(wa, sizeof wa, "%s.w", LA[i]); snprintf(wb, sizeof wb, "%s.w", LB[i]);
        if (lv == 1 && [nn matrixPath:wa cin:la.cin] && [nn matrixPath:wb cin:lb.cin] && la.cin == prevC && la.cout == 64 && lb.cin == 64 && lb.cout <= 32) {
            la.usrc = prev; la.up_ws = w[lv + 1]; la.up_hs = h[lv + 1];
            [nn convPair:enc first:la second:lb];
        } else {
            decA(LA[i], prev, B(@(S[i])), B(@(Q[i])), B(@(Tm[i])), prevC, w[lv], h[lv], w[lv + 1], h[lv + 1], A);
            [nn conv:enc args:lb];
        }
        prev = B(@(U[i]));
        static const uint32_t OUTC[4] = {128, 128, 64, 32};
        prevC = OUTC[i];
    }
    /* the last decoder layer and the head: one pass on the matrix path (v is not stored), else two */
    MetalFGConvArgs d0 = conv1(nn, "n1.dec0a", B(@"s0"), B(@"v"), w[0], h[0], A), hd = conv1(nn, "n1.head", B(@"v"), n1out, w[0], h[0], 0);
    if ([nn matrixPath:"n1.dec0a.w" cin:d0.cin] && [nn matrixPath:"n1.head.w" cin:hd.cin] && d0.cin == 32 && d0.cout == 32 && hd.cin == 32 && hd.cout <= 16) {
        d0.usrc = prev; d0.up_ws = w[1]; d0.up_hs = h[1];
        [nn convPair:enc first:d0 second:hd];
    } else {
        decA("n1.dec0a", prev, B(@"s0"), B(@"q0"), B(@"v"), 32, w[0], h[0], w[1], h[1], A);
        [nn conv:enc args:hd];
    }
}
@end

/* ---------------- net2 ---------------- */

@implementation MetalFGNet2 {
    MetalFGNN *_nn;
    uint32_t _Nw, _Nh;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *_t;
    id<MTLComputePipelineState> _pack, _join, _final;
}

- (NSDictionary<NSString *, id<MTLBuffer>> *)tensors { return _t; }

- (instancetype)initWithNN:(MetalFGNN *)nn Nw:(uint32_t)Nw Nh:(uint32_t)Nh {
    self = [super init];
    _nn = nn; _Nw = Nw; _Nh = Nh;
    _t = [NSMutableDictionary dictionary];
    static const struct { const char *name; int level; int ch; } T[] = {
        {"x0", 0, 16}, {"p1", 1, 32}, {"p2", 2, 32}, {"a0", 3, 64}, {"ah", 3, 64}, {"a1", 3, 64},
        {"hA", 2, 64}, {"cA", 2, 96}, {"vB", 1, 96}, {"mB", 1, 8}, {"u", 0, 8}, {"z", 0, 32},
        {"f1", 1, 16}, {"f2", 2, 32}, {"fh", 2, 32}, {"f3", 2, 32}, {"gA", 1, 32}, {"eB", 1, 48}, {"zB", 0, 48}, {"k", 0, 8},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        _t[@(T[i].name)] = [nn scratch:(NSUInteger)(Nw >> T[i].level) * (Nh >> T[i].level) * T[i].ch];
    return self;
}

static MetalFGConvArgs convn(const char *name, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t W, uint32_t H, uint32_t flags) {
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H;
    a.cin = t->I == 18 ? 32 : t->I; a.in_stride = a.cin; a.cout = t->O; a.out_stride = t->O;
    a.flags = flags; a.widx = wi;
    return a;
}

- (void)encode:(MetalFGEnc *)enc a:(id<MTLBuffer>)ca b:(id<MTLBuffer>)cb h:(id<MTLBuffer>)hm t:(float)t out:(id<MTLBuffer>)out {
    MetalFGNN *nn = _nn;
    uint32_t w[4], h[4];
    for (int g = 0; g < 4; g++) { w[g] = _Nw >> g; h[g] = _Nh >> g; }
    id<MTLBuffer> (^B)(NSString *) = ^id<MTLBuffer>(NSString *n) { return self->_t[n]; };
    const uint32_t AB = METALFG_CONV_ACT | METALFG_CONV_BOUND, DN = AB | METALFG_CONV_POOL;

    /* coarse stage */
    [nn packX0:enc a:ca b:cb h:hm out:B(@"x0") Nw:_Nw Nh:_Nh t:t];
    [nn conv:enc args:convn("n2.a.d0.w", B(@"x0"), B(@"p1"), w[0], h[0], DN)];
    [nn conv:enc args:convn("n2.a.d1.w", B(@"p1"), B(@"p2"), w[1], h[1], DN)];
    [nn conv:enc args:convn("n2.a.d2.w", B(@"p2"), B(@"a0"), w[2], h[2], DN)];
    id<MTLBuffer> cur = B(@"a0"), nxt = B(@"a1"), mid = B(@"ah");
    for (int j = 0; j < 4; j++) {
        char n1[32], n2[32];
        snprintf(n1, sizeof n1, "n2.a.res%d.w", 2 * j);
        snprintf(n2, sizeof n2, "n2.a.res%d.w", 2 * j + 1);
        [nn conv:enc args:convn(n1, cur, mid, w[3], h[3], AB | METALFG_CONV_ZADD)];     /* fg.net2_forward_r2: the first convolution of a pair has no skip, only the second adds a_j */
        MetalFGConvArgs a = convn(n2, mid, nxt, w[3], h[3], AB | METALFG_CONV_RES);
        a.res = cur; a.res_stride = 64;
        [nn conv:enc args:a];
        id<MTLBuffer> tmp = cur; cur = nxt; nxt = tmp;
    }
    /* heads at g = 2 and g = 1 (y is `cur`) */
    [nn up2:enc src:cur out:B(@"hA") Ws:w[3] Hs:h[3] C:64];
    MetalFGConvArgs ha = convn("n2.a.headA.w", B(@"hA"), B(@"cA"), w[2], h[2], AB);
    ha.bound_from = 32; /* group 0 is activated only, groups 1 and 2 are bounded */
    [nn conv:enc args:ha];
    [nn up2x:enc src:B(@"cA") out:B(@"vB") Ws:w[2] Hs:h[2] C:96];
    static const uint32_t rows0[3] = {0, 4, 5}, nrows[3] = {4, 1, 3};
    for (int q = 0; q < 3; q++) {
        MetalFGConvArgs hb = convn("n2.a.headB.w", B(@"vB"), B(@"mB"), w[1], h[1], q ? METALFG_CONV_BOUND : 0);
        hb.cin = 32; hb.in_stride = 96; hb.in_base = 32 * q;
        hb.row0 = rows0[q]; hb.cout = nrows[q]; hb.out_stride = 8; hb.out_base = rows0[q];
        if (q) [enc skipNextBarrier];   /* the three head groups read the same input and write distinct rows */
        [nn conv:enc args:hb];
    }
    [nn up2:enc src:B(@"mB") out:B(@"u") Ws:w[1] Hs:h[1] C:8];
    /* stage join */
    [nn join:enc a:ca b:cb h:hm u:B(@"u") out:B(@"z") Nw:_Nw Nh:_Nh t:t];
    /* fine stage */
    [nn conv:enc args:convn("n2.b.d0.w", B(@"z"), B(@"f1"), w[0], h[0], DN)];
    [nn conv:enc args:convn("n2.b.d1.w", B(@"f1"), B(@"f2"), w[1], h[1], DN)];
    cur = B(@"f2"); nxt = B(@"f3"); mid = B(@"fh");
    for (int j = 0; j < 4; j++) {
        char n1[32], n2[32];
        snprintf(n1, sizeof n1, "n2.b.res%d.w", 2 * j);
        snprintf(n2, sizeof n2, "n2.b.res%d.w", 2 * j + 1);
        [nn conv:enc args:convn(n1, cur, mid, w[2], h[2], AB | METALFG_CONV_ZADD)];
        MetalFGConvArgs a = convn(n2, mid, nxt, w[2], h[2], AB | METALFG_CONV_RES);
        a.res = cur; a.res_stride = 32;
        [nn conv:enc args:a];
        id<MTLBuffer> tmp = cur; cur = nxt; nxt = tmp;
    }
    [nn up2:enc src:cur out:B(@"gA") Ws:w[2] Hs:h[2] C:32];
    MetalFGConvArgs hA2 = convn("n2.b.headA.w", B(@"gA"), B(@"eB"), w[1], h[1], AB);
    hA2.bound_from = 16;
    [nn conv:enc args:hA2];
    [nn up2x:enc src:B(@"eB") out:B(@"zB") Ws:w[1] Hs:h[1] C:48];
    /* on the matrix path each head group forms its rows of N = k + m u in the epilogue (n2_final's expression), so k is not stored */
    const BOOL fold = [nn matrixPath:"n2.b.headB.w" cin:16];
    for (int q = 0; q < 3; q++) {
        MetalFGConvArgs hb = convn("n2.b.headB.w", B(@"zB"), B(@"k"), w[0], h[0], q ? METALFG_CONV_BOUND : 0);
        hb.cin = 16; hb.in_stride = 48; hb.in_base = 16 * q;
        hb.row0 = rows0[q]; hb.cout = nrows[q]; hb.out_stride = 8; hb.out_base = rows0[q];
        if (fold) { hb.flags |= METALFG_CONV_FINAL; hb.out = out; hb.res = B(@"u"); hb.res_stride = 8; hb.res_base = rows0[q]; }
        if (q) [enc skipNextBarrier];   /* the three head groups read the same input and write distinct rows */
        [nn conv:enc args:hb];
    }
    if (!fold) [nn final:enc k:B(@"k") u:B(@"u") out:out Nw:_Nw Nh:_Nh];
}
@end
