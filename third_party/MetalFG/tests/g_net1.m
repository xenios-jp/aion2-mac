/* net1 forward (spec fg.net1_forward section 7): n1f-conv3-random, -impulse, -border, -wiring, -head-linear, -random-full,
   -coverage, with controls that must be detected. */
#import <Foundation/Foundation.h>
#include "nnutil.h"

static int fails;

static MetalFGConvArgs conv_args(const char *name, id<MTLBuffer> in, id<MTLBuffer> out, int W, int H, int flags) {
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H; a.cin = t->I; a.in_stride = t->I; a.cout = t->O; a.out_stride = t->O;
    a.flags = flags; a.widx = wi; a.nest = nest_for_name(name);
    return a;
}

/* reference net1 on the rig's weights */
static void ref_layer(NnRig *r, const char *stem, const h16 *in, int W, int H, h16 *out, int act) {
    char wn[48], bn[48];
    snprintf(wn, sizeof wn, "%s.w", stem);
    snprintf(bn, sizeof bn, "%s.b", stem);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[metalfg_weight_index(wn)];
    g_ref_nest = nest_for_name(stem);
    ref_conv(in, W, H, t->I, 0, t->I, wt(r, wn), wt(r, bn), t->kh, t->I, 0, t->O, act ? 1 : 0, r->params.leak, 0, 0, out, t->O, 0, NULL, 0, 0);
}

static void (^g_hook)(const char *, const h16 *, size_t);
#define HOOK(name, p, n) do { if (g_hook) g_hook(name, p, n); } while (0)
static void ref_net1(NnRig *r, const h16 *T, int w1, int h1, h16 *O) {
    int w[6], h[6];
    for (int k = 0; k < 6; k++) { w[k] = w1 >> k; h[k] = h1 >> k; }
    static const int ch[5] = {32, 64, 128, 128, 256};
    static const char *enc[5] = {"n1.enc0", "n1.enc1", "n1.enc2", "n1.enc3", "n1.enc4"};
    h16 *s[5], *pooled = NULL, *cur = NULL;
    for (int k = 0; k < 5; k++) {
        const h16 *src = T;
        if (k > 0) {
            pooled = malloc((size_t)w[k] * h[k] * ch[k - 1] * 2);
            ref_pool_max(s[k - 1], w[k - 1], h[k - 1], ch[k - 1], pooled);
            src = pooled;
        }
        s[k] = malloc((size_t)w[k] * h[k] * ch[k] * 2);
        ref_layer(r, enc[k], src, w[k], h[k], s[k], 1);
        { char nm[8]; snprintf(nm, sizeof nm, "s%d", k); HOOK(nm, s[k], (size_t)w[k] * h[k] * ch[k]); }
        if (pooled) { free(pooled); pooled = NULL; }
    }
    h16 *p4 = malloc((size_t)w[5] * h[5] * 256 * 2), *mid = malloc((size_t)w[5] * h[5] * 512 * 2), *z = malloc((size_t)w[5] * h[5] * 256 * 2);
    ref_pool_max(s[4], w[4], h[4], 256, p4);
    ref_layer(r, "n1.mida", p4, w[5], h[5], mid, 1);
    ref_layer(r, "n1.midb", mid, w[5], h[5], z, 1);
    HOOK("z", z, (size_t)w[5] * h[5] * 256);
    cur = z;
    int cprev = 256;
    static const char *A[4] = {"n1.dec4a", "n1.dec3a", "n1.dec2a", "n1.dec1a"}, *Bn[4] = {"n1.dec4b", "n1.dec3b", "n1.dec2b", "n1.dec1b"};
    static const int cmid[4] = {256, 128, 128, 64}, cout[4] = {128, 128, 64, 32};
    for (int i = 0; i < 4; i++) {
        int lv = 4 - i;
        h16 *q = malloc((size_t)w[lv] * h[lv] * cprev * 2), *t = malloc((size_t)w[lv] * h[lv] * cmid[i] * 2), *u = malloc((size_t)w[lv] * h[lv] * cout[i] * 2);
        ref_up_corner(cur, w[lv + 1], h[lv + 1], cprev, s[lv], q);
        ref_layer(r, A[i], q, w[lv], h[lv], t, 1);
        ref_layer(r, Bn[i], t, w[lv], h[lv], u, 1);
        { char nm[8]; snprintf(nm, sizeof nm, "u%d", lv); HOOK(nm, u, (size_t)w[lv] * h[lv] * cout[i]); }
        free(q); free(t);
        if (cur != z) free(cur);
        cur = u;
        cprev = cout[i];
    }
    h16 *q0 = malloc((size_t)w[0] * h[0] * 32 * 2), *v = malloc((size_t)w[0] * h[0] * 32 * 2);
    ref_up_corner(cur, w[1], h[1], 32, s[0], q0);
    ref_layer(r, "n1.dec0a", q0, w[0], h[0], v, 1);
    ref_layer(r, "n1.head", v, w[0], h[0], O, 0);
    free(q0); free(v); free(cur); free(z); free(mid); free(p4);
    for (int k = 0; k < 5; k++) free(s[k]);
}

static NnRig open_with(Gpu g, WgenMode mode, uint64_t seed, void (^edit)(uint16_t **)) {
    uint16_t *t[METALFG_TENSOR_COUNT];
    wgen_tensors(mode, seed, t);
    if (edit) edit(t);
    NnRig r;
    if (nn_open(&r, g, t)) exit(2);
    return r;
}

static uint16_t *tensor(uint16_t **t, const char *name) { return t[metalfg_weight_index(name)]; }

static void conv3_random(Gpu g) {
    NnRig r = open_with(g, WGEN_RANDOM, 11, nil);
    WgenRng rng; wgen_seed(&rng, 12);
    int W = 64, H = 32;
    h16 *T = rand_h16(&rng, (size_t)W * H * 48, -1, 1);
    id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:conv_args("n1.enc0.w", in, out, W, H, METALFG_CONV_ACT)]; });
    h16 *ref = malloc((size_t)W * H * 32 * 2);
    ref_layer(&r, "n1.enc0", T, W, H, ref, 1);
    Cmp c = cmp_tensors(out.contents, ref, (size_t)W * H * 32);
    CHECKC(layer_ok(c), "n1f-conv3-random: exact %zu / %zu, within1 %zu, bad %zu", c.exact, c.n, c.within1, c.bad);
    /* control: the plain rectifier must be distinguishable (leak >= 2^-8) */
    h16 *plain = malloc((size_t)W * H * 32 * 2);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[metalfg_weight_index("n1.enc0.w")];
    g_ref_nest = 1;
    ref_conv(T, W, H, 48, 0, 48, wt(&r, "n1.enc0.w"), wt(&r, "n1.enc0.b"), 3, 48, 0, t->O, 1, 0.0, 0, 0, plain, 32, 0, NULL, 0, 0);
    CHECKC(!layer_ok(cmp_tensors(out.contents, plain, (size_t)W * H * 32)), "control: plain rectifier must fail");
    free(T); free(ref); free(plain);
}

static void conv3_impulse(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        const MetalFGTensorInfo *ti = &metalfg_weight_table()[metalfg_weight_index("n1.enc0.w")];
        uint16_t *w = tensor(t, "n1.enc0.w");
        for (uint32_t o = 0; o < 32; o++) for (uint32_t ky = 0; ky < 3; ky++) for (uint32_t kx = 0; kx < 3; kx++) for (uint32_t i = 0; i < 48; i++)
            w[(((size_t)o * 3 + ky) * 3 + kx) * ti->I + i] = f16bits((1 + (o + 3 * ky + 11 * kx + 7 * i) % 64) / 64.0);
    });
    int W = 64, H = 32;
    int cs[3] = {33, 1, 47};
    for (int ci = 0; ci < 3; ci++) {
        int c = cs[ci];
        h16 *T = calloc((size_t)W * H * 48, 2);
        T[((size_t)10 * W + 20) * 48 + c] = (h16)1;
        id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 32);
        nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:conv_args("n1.enc0.w", in, out, W, H, METALFG_CONV_ACT)]; });
        const h16 *o = out.contents;
        int bad = 0;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int oc = 0; oc < 32; oc++) {
            double want = 0;
            for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                if (x == 20 - (kx - 1) && y == 10 - (ky - 1)) want = (1 + (oc + 3 * ky + 11 * kx + 7 * c) % 64) / 64.0;
            if ((double)o[((size_t)y * W + x) * 32 + oc] != want) bad++;
        }
        CHECKC(bad == 0, "n1f-conv3-impulse c=%d: %d wrong elements", c, bad);
        free(T);
    }
}

static void conv3_border(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n1.enc0.w");
        for (size_t i = 0; i < 32 * 9 * 48; i++) w[i] = f16bits(1.0 / 64);
    });
    int W = 32, H = 32;
    h16 *T = malloc((size_t)W * H * 48 * 2);
    for (size_t i = 0; i < (size_t)W * H * 48; i++) T[i] = (h16)1;
    id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:conv_args("n1.enc0.w", in, out, W, H, METALFG_CONV_ACT)]; });
    const h16 *o = out.contents;
    int bad = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int nx = (x == 0 || x == W - 1) ? 2 : 3, ny = (y == 0 || y == H - 1) ? 2 : 3;
        double want = nx * ny * 48 / 64.0;
        for (int c = 0; c < 32; c++) if ((double)o[((size_t)y * W + x) * 32 + c] != want) bad++;
    }
    CHECKC(bad == 0, "n1f-conv3-border: %d wrong", bad);
    free(T);
}

/* one-hot weights: (o, centre tap, o) = 1, biases zero */
static void one_hot(uint16_t **t) {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    for (int i = 0; i < METALFG_TENSOR_COUNT; i += 2) {
        if (strncmp(T[i].name, "n1.", 3)) continue;
        uint32_t m = T[i].O < T[i].I ? T[i].O : T[i].I;
        for (uint32_t o = 0; o < m; o++)
            t[i][(((size_t)o * T[i].kh + T[i].kh / 2) * T[i].kw + T[i].kw / 2) * T[i].I + o] = f16bits(1.0);
    }
}

static void wiring_case(Gpu g, double lo, double hi, double head_bias, const char *label) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        one_hot(t);
        if (head_bias != 0) for (int i = 0; i < 16; i++) tensor(t, "n1.head.b")[i] = f16bits(head_bias);
    });
    int W = 64, H = 64;
    WgenRng rng; wgen_seed(&rng, 21);
    h16 *T = calloc((size_t)W * H * 48, 2);
    for (size_t p = 0; p < (size_t)W * H; p++) for (int c = 0; c < 32; c++) T[p * 48 + c] = (h16)(lo + (hi - lo) * wgen_uniform(&rng));
    id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 16);
    MetalFGNet1 *net = [[MetalFGNet1 alloc] initWithNN:r.nn w1:W h1:H];
    nn_run(&r, ^(MetalFGEnc *e) { [net encode:e in:in out:out]; });
    const h16 *o = out.contents;
    double worst = 0;
    long over_alt = 0, n_alt = 0;
    for (int c = 0; c < 16; c++) {
        int w[6], h[6];
        for (int k = 0; k < 6; k++) { w[k] = W >> k; h[k] = H >> k; }
        h16 *P[6];
        h16 *a = malloc((size_t)W * H * 2);
        for (size_t p = 0; p < (size_t)W * H; p++) a[p] = T[p * 48 + c];
        P[0] = a;
        for (int k = 1; k < 6; k++) { P[k] = malloc((size_t)w[k] * h[k] * 2); ref_pool_max(P[k - 1], w[k - 1], h[k - 1], 1, P[k]); }
        h16 *x = P[5];
        h16 *xs[6];
        for (int k = 4; k >= 0; k--) {
            xs[k] = malloc((size_t)w[k] * h[k] * 2);
            ref_up_corner(x, w[k + 1], h[k + 1], 1, P[k], xs[k]);
            x = xs[k];
        }
        for (size_t p = 0; p < (size_t)W * H; p++) {
            double e = fabs((double)o[p * 16 + c] - ((double)x[p] + head_bias));
            if (e > worst) worst = e;
        }
        /* control: dropping the skip at the finest level */
        h16 *noskip = malloc((size_t)W * H * 2);
        ref_up_corner(xs[1], w[1], h[1], 1, NULL, noskip);
        for (size_t p = 0; p < (size_t)W * H; p++) { n_alt++; if (fabs((double)o[p * 16 + c] - ((double)noskip[p] + head_bias)) > 5 * 0x1p-8) over_alt++; }
        free(noskip);
        for (int k = 0; k < 6; k++) free(P[k]);
        for (int k = 0; k < 5; k++) free(xs[k]);
    }
    CHECKC(worst <= 5 * 0x1p-8, "%s: worst abs error %g", label, worst);
    CHECKC(over_alt * 2 > n_alt, "%s control: dropping the finest skip must fail (%ld of %ld)", label, over_alt, n_alt);
    free(T);
}

static void random_full(Gpu g, int W, int H) {
    NnRig r = open_with(g, WGEN_RANDOM, 31 + W, nil);
    WgenRng rng; wgen_seed(&rng, 32);
    h16 *T = rand_h16(&rng, (size_t)W * H * 48, -1, 1);
    id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 16);
    MetalFGNet1 *net = [[MetalFGNet1 alloc] initWithNN:r.nn w1:W h1:H];
    nn_run(&r, ^(MetalFGEnc *e) { [net encode:e in:in out:out]; });
    h16 *ref = malloc((size_t)W * H * 16 * 2);
    g_hook = ^(const char *name, const h16 *p, size_t n) {
        Cmp k = cmp_tensors(net.tensors[@(name)].contents, p, n);
        if (getenv("METALFG_VERBOSE")) printf("  stage %s: rms err %g (ref %g) max %g\n", name, k.rms_err, k.rms_ref, k.max_abs);
    };
    ref_net1(&r, T, W, H, ref);
    g_hook = nil;
    Cmp c = cmp_tensors(out.contents, ref, (size_t)W * H * 16);
    CHECKC(e2e_ok(c), "n1f-random-full %dx%d: rms %g (ref %g), max %g (ref %g), bad %zu", W, H, c.rms_err, c.rms_ref, c.max_abs, c.max_ref, c.bad);
    /* control: dec2a and dec3a swapped (same shape) must fail */
    NnRig r2 = r;
    h16 *a = wt(&r, "n1.dec2a.w"), *b = wt(&r, "n1.dec3a.w");
    size_t nb = 128 * 128;
    h16 *save = malloc(nb * 2);
    memcpy(save, a, nb * 2); memcpy(a, b, nb * 2); memcpy(b, save, nb * 2);
    h16 *ref2 = malloc((size_t)W * H * 16 * 2);
    ref_net1(&r2, T, W, H, ref2);
    CHECKC(!e2e_ok(cmp_tensors(out.contents, ref2, (size_t)W * H * 16)), "control: swapped layers must fail");
    free(T); free(ref); free(ref2); free(save);
}

static void coverage(Gpu g) {
    NnRig r = open_with(g, WGEN_RANDOM, 41, nil);
    int sizes[3][2] = {{32, 32}, {64, 32}, {480, 288}};
    for (int i = 0; i < 3; i++) {
        int W = sizes[i][0], H = sizes[i][1];
        WgenRng rng; wgen_seed(&rng, 42 + i);
        h16 *T = rand_h16(&rng, (size_t)W * H * 48, -1, 1);
        id<MTLBuffer> in = nn_buf(&r, T, (size_t)W * H * 48), out = nn_buf(&r, NULL, (size_t)W * H * 16);
        nn_fill_nan(out, (size_t)W * H * 16);
        MetalFGNet1 *net = [[MetalFGNet1 alloc] initWithNN:r.nn w1:W h1:H];
        nn_run(&r, ^(MetalFGEnc *e) { [net encode:e in:in out:out]; });
        const h16 *o = out.contents;
        size_t nan = 0;
        for (size_t k = 0; k < (size_t)W * H * 16; k++) nan += isnan((double)o[k]);
        CHECKC(nan == 0, "n1f-coverage %dx%d: %zu NaN left", W, H, nan);
        free(T);
    }
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        conv3_random(g);
        conv3_impulse(g);
        conv3_border(g);
        wiring_case(g, 0, 1, 0, "n1f-wiring");
        wiring_case(g, 0, 0.05, -0.5, "n1f-head-linear");
        random_full(g, 64, 64);
        random_full(g, 96, 64);
        coverage(g);
    }
    printf(fails ? "g_net1: %d FAILURES\n" : "g_net1: ok\n", fails);
    return fails != 0;
}
