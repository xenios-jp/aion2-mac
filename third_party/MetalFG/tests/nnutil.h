/* Helpers for the network tests: weights in memory, running an encoder closure, buffer upload/download. */
#pragma once
#include "gputil.h"
#include "nn.h"
#include "ref_nn.h"

typedef struct {
    Gpu gpu;
    id<MTLLibrary> lib;
    id<MTLBuffer> wbuf;
    uint8_t *img;       /* weight file image (host copy) */
    MetalFGNN *nn;
    MetalFGParams params;
} NnRig;

static inline h16 *wt(const NnRig *r, const char *name) {
    return (h16 *)(void *)(r->img + metalfg_weight_table()[metalfg_weight_index(name)].offset);
}

/* tensors: 84 malloc'd arrays (ownership passes to the rig, which frees them) */
static inline int nn_open(NnRig *r, Gpu g, uint16_t *tensors[METALFG_TENSOR_COUNT]) {
    memset((void *)r, 0, sizeof *r);
    r->gpu = g;
    r->params = wgen_params();
    size_t n;
    r->img = metalfg_weights_image((const uint16_t *const *)tensors, &n);
    for (int i = 0; i < METALFG_TENSOR_COUNT; i++) free(tensors[i]);
    r->wbuf = [g.dev newBufferWithLength:n + 4096 options:MTLResourceStorageModeShared];
    memset(r->wbuf.contents, 0, n + 4096);
    memcpy(r->wbuf.contents, r->img, n);
    NSError *e = nil;
    const char *dir = getenv("METALFG_METALLIB_DIR") ? getenv("METALFG_METALLIB_DIR") : "build";
    NSString *path = [NSString stringWithFormat:@"%s/metalfg.metallib", dir];
    r->lib = [g.dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
    if (!r->lib) { printf("no metallib at %s\n", path.UTF8String); return 1; }
    NSString *err = nil;
    r->nn = [[MetalFGNN alloc] initWithDevice:g.dev library:r->lib weights:r->wbuf leak:r->params.leak satN2:r->params.sat_n2 error:&err];
    if (!r->nn) { printf("MetalFGNN: %s\n", err.UTF8String); return 1; }
    return 0;
}

static inline void nn_run(NnRig *r, void (^body)(MetalFGEnc *)) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [r->gpu.q commandBuffer];
        MetalFGEnc *enc = [MetalFGEnc classicWith:cb];
        body(enc);
        [enc end];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { printf("command buffer error: %s\n", cb.error.localizedDescription.UTF8String); exit(3); }
    }
}

static inline id<MTLBuffer> nn_buf(NnRig *r, const h16 *data, size_t n) {
    id<MTLBuffer> b = [r->gpu.dev newBufferWithLength:n * 2 + 64 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, b.length);
    if (data) memcpy(b.contents, data, n * 2);
    return b;
}

static inline void nn_fill_nan(id<MTLBuffer> b, size_t n) {
    h16 *p = b.contents;
    for (size_t i = 0; i < n; i++) p[i] = (h16)NAN;
}

static inline h16 *rand_h16(WgenRng *rng, size_t n, double lo, double hi) {
    h16 *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = (h16)(lo + (hi - lo) * wgen_uniform(rng));
    return p;
}

#define CHECKC(cond, ...) do { if (!(cond)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* per-layer criterion of the specs: at least 99 % bit-equal and all within one ulp */
/* The pinned convolution paths (METALFG_NN_PATH=blk or scalar) reproduce the model of fg.arithmetic_pin_r4 bit for bit; the default path (tensor operations,
   fg.arithmetic_tolerance_r1) sums in the unit's own order, and its per-layer accuracy is judged by g_tol. For it these per-layer comparisons with the model keep only the
   end-to-end bound as a gross check. */
static inline int nn_pinned(void) { const char *v = getenv("METALFG_NN_PATH"); return v && (!strcmp(v, "blk") || !strcmp(v, "scalar")); }
static inline int e2e_ok(Cmp c);
static inline int layer_ok(Cmp c) { return nn_pinned() ? (c.bad == 0 && c.within1 == c.n && c.exact * 100 >= c.n * 99) : e2e_ok(c); }
/* end-to-end criterion */
static inline int e2e_ok(Cmp c) { return c.bad == 0 && c.rms_err <= 0.003 * c.rms_ref && c.max_abs <= 0.02 * c.max_ref; }
