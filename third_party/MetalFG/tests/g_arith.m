/* Network arithmetic (spec fg.arithmetic_pin_r4 section 7): ar-pair-round, ar-sum-order, ar-nest, ar-act-res, ar-pool-order, ar-up2-edge, ar-join-quant,
   ar-head-sat, ar-up-scale, ar-logistic, ar-mask-fma, ar-luma. The CPU model is tests/ref_nn.h; every case has controls that must fail (a control that passes
   voids the case and is reported as a failure here). The store probe selects the store mode of the model for image stores. */
#import <Foundation/Foundation.h>
#include "nnutil.h"
#include "probe.h"
#include "net1io.h"

static int fails;

static MetalFGConvArgs cargs(const char *name, id<MTLBuffer> in, id<MTLBuffer> out, int W, int H, int flags) {
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H; a.cin = t->I; a.in_stride = t->I; a.cout = t->O; a.out_stride = t->O;
    a.flags = flags; a.widx = wi; a.nest = nest_for_name(name);
    return a;
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

/* a slope other than the rig's */
static void set_leak(NnRig *r, float leak) {
    NSString *err = nil;
    r->params.leak = leak;
    r->nn = [[MetalFGNN alloc] initWithDevice:r->gpu.dev library:r->lib weights:r->wbuf leak:leak satN2:r->params.sat_n2 error:&err];
    if (!r->nn) { printf("MetalFGNN: %s\n", err.UTF8String); exit(2); }
}

/* the model of one layer on the rig's named weights, order of pairs `nest` */
static void model_layer(NnRig *r, const char *stem, const h16 *in, int W, int H, int cin_use, int flags, int nest, h16 *out) {
    char wn[64], bn[64];
    snprintf(wn, sizeof wn, "%s.w", stem);
    snprintf(bn, sizeof bn, "%s.b", stem);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[metalfg_weight_index(wn)];
    g_ref_nest = nest;
    ref_conv(in, W, H, t->I, 0, cin_use, wt(r, wn), wt(r, bn), t->kh, t->I, 0, t->O, flags, r->params.leak, r->params.sat_n2, 0, out, t->O, 0, NULL, 0, 0);
}

static int count_diff(const h16 *a, const h16 *b, size_t n) {
    int d = 0;
    for (size_t i = 0; i < n; i++) d += memcmp(&a[i], &b[i], 2) != 0;
    return d;
}

static uint16_t hb(double v) { return f16bits(v); }

/* ---- ar-pair-round ---- */
static void ar_pair_round(Gpu g) {
    const float sp = 0x1p-10f;
    h16 wv = (h16)(0.6f * sp);
    uint16_t wb; memcpy(&wb, &wv, 2);
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n1.dec0a.w"), *b = tensor(t, "n1.dec0a.b");
        w[0] = wb; w[16] = wb; b[0] = hb(1.0);
    });
    int W = 4, H = 2;
    size_t n = (size_t)W * H * 32;
    h16 *in = malloc(n * 2);
    for (size_t i = 0; i < n; i++) in[i] = (h16)1;
    id<MTLBuffer> bi = nn_buf(&r, in, n), bo = nn_buf(&r, NULL, n);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n1.dec0a.w", bi, bo, W, H, 0)]; });
    h16 *m = malloc(n * 2);
    model_layer(&r, "n1.dec0a", in, W, H, 32, 0, 1, m);
    CHECKC(count_diff(bo.contents, m, n) == 0, "ar-pair-round: engine differs from the model");
    h16 want2 = (h16)(1.0f + 2 * sp), want1 = (h16)(1.0f + sp);
    CHECKC(memcmp(&((h16 *)bo.contents)[0], &want2, 2) == 0, "ar-pair-round: channel 0 is %g, want one plus two spacings", (double)((h16 *)bo.contents)[0]);
    h16 ctl = (h16)(1.0f + (float)wv + (float)wv);          /* binary32 accumulation, one rounding at the end */
    CHECKC(memcmp(&ctl, &want1, 2) == 0 && memcmp(&ctl, &((h16 *)bo.contents)[0], 2) != 0, "ar-pair-round control: single rounding must differ");
    free(in); free(m);
}

/* ---- ar-sum-order ---- */
static void ar_sum_order(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n1.dec0a.w");
        w[0] = hb(4096); w[1] = hb(1); w[2] = hb(-4096);
    });
    int W = 2, H = 1;
    size_t n = (size_t)W * H * 32;
    h16 *in = calloc(n, 2);
    for (int p = 0; p < W * H; p++) { in[p * 32 + 0] = (h16)4096; in[p * 32 + 1] = (h16)1; in[p * 32 + 2] = (h16)4096; }
    id<MTLBuffer> bi = nn_buf(&r, in, n), bo = nn_buf(&r, NULL, n);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n1.dec0a.w", bi, bo, W, H, 0)]; });
    const h16 *o = bo.contents;
    CHECKC((double)o[0] == 0, "ar-sum-order: ascending order gives %g, want 0", (double)o[0]);
    float t = 0; float pr[3] = {16777216.0f, 1.0f, -16777216.0f};
    for (int i = 2; i >= 0; i--) t += pr[i];                  /* descending: control */
    CHECKC(t == 1.0f, "ar-sum-order control: the descending sum must give one (%g)", t);
    free(in);
}

/* ---- ar-nest ---- */
static void wide_fill(h16 *p, size_t n, WgenRng *rng) {
    for (size_t i = 0; i < n; i++) {
        double mag = ldexp(1.0 + wgen_uniform(rng), -(int)(wgen_uniform(rng) * 9));
        p[i] = (h16)((wgen_uniform(rng) < 0.5 ? -1 : 1) * mag);
    }
}

static void ar_nest(Gpu g) {
    struct { const char *stem; int cin; int nest; } L[3] = {{"n1.enc2", 64, 1}, {"n1.enc4", 128, 2}, {"n2.a.res0", 64, 3}};
    for (int li = 0; li < 3; li++) {
        char wn_[64];
        snprintf(wn_, sizeof wn_, "%s.w", L[li].stem);
        const char *wn = wn_;
        const MetalFGTensorInfo *ti = &metalfg_weight_table()[metalfg_weight_index(wn)];
        size_t nw = (size_t)ti->O * 9 * ti->I;
        WgenRng rng; wgen_seed(&rng, 300 + li);
        h16 *wide = malloc(nw * 2);
        wide_fill(wide, nw, &rng);
        NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) { memcpy(tensor(t, wn), wide, nw * 2); });
        int W = 6, H = 5;
        size_t nin = (size_t)W * H * ti->I, nout = (size_t)W * H * ti->O;
        h16 *in = malloc(nin * 2);
        wide_fill(in, nin, &rng);
        id<MTLBuffer> bi = nn_buf(&r, in, nin), bo = nn_buf(&r, NULL, nout);
        nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs(wn, bi, bo, W, H, 0)]; });
        h16 *m[4];
        for (int k = 1; k <= 3; k++) { m[k] = malloc(nout * 2); model_layer(&r, L[li].stem, in, W, H, ti->I, 0, k, m[k]); }
        CHECKC(count_diff(bo.contents, m[L[li].nest], nout) == 0, "ar-nest %s: engine differs from the specified nesting (%d of %zu)", L[li].stem,
               count_diff(bo.contents, m[L[li].nest], nout), nout);
        for (int k = 1; k <= 3; k++) {
            if (k == L[li].nest) continue;
            int d = count_diff(m[L[li].nest], m[k], nout);
            CHECKC(d * 10 >= (int)nout, "ar-nest %s: alternative %d disagrees in only %d of %zu outputs (case void)", L[li].stem, k, d, nout);
            CHECKC(count_diff(bo.contents, m[k], nout) > 0, "ar-nest %s control %d: the engine equals the alternative nesting", L[li].stem, k);
        }
        for (int k = 1; k <= 3; k++) free(m[k]);
        free(wide); free(in);
    }
}

/* ---- ar-act-res ---- */
static void ar_act_res(Gpu g) {
    /* leaky rule with a slope that is not a power of two (0.3): the binary16 product before the maximum; the explicit zero of a pair's first convolution */
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n1.dec0a.w");
        for (int o = 0; o < 32; o++) w[o * 32 + o] = hb(1.0);
        uint16_t *b = tensor(t, "n2.a.res0.b");
        b[0] = hb(-0x1p-24);                       /* the leaky product of this value underflows to negative zero */
    });
    set_leak(&r, 0.3f);
    int W = 8, H = 8;
    size_t n = (size_t)W * H * 32;
    WgenRng rng; wgen_seed(&rng, 41);
    h16 *in = malloc(n * 2);
    for (size_t i = 0; i < n; i++) in[i] = (h16)(-(1.0 + wgen_uniform(&rng) * 100.0));      /* negative: the slope applies */
    id<MTLBuffer> bi = nn_buf(&r, in, n), bo = nn_buf(&r, NULL, n);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n1.dec0a.w", bi, bo, W, H, METALFG_CONV_ACT)]; });
    h16 *m = malloc(n * 2);
    model_layer(&r, "n1.dec0a", in, W, H, 32, 1, 1, m);
    CHECKC(count_diff(bo.contents, m, n) == 0, "ar-act-res: engine differs from the model");
    const h16 *o = bo.contents;
    int dctl = 0;
    for (size_t i = 0; i < n; i++) {
        float v = (float)in[i];
        h16 p32 = (h16)(v * 0.3f);                              /* control: the binary32 slope */
        h16 pw = (h16)(v * (float)(h16)0.3f);
        h16 c = (float)p32 > (float)in[i] ? p32 : in[i];
        (void)pw;
        dctl += memcmp(&c, &o[i], 2) != 0;
    }
    CHECKC(dctl > 0, "ar-act-res control (binary32 product) does not differ in any element");
    /* the first convolution of a pair: exactly negative zero before the explicit add */
    int Wn = 3, Hn = 3;
    size_t n2 = (size_t)Wn * Hn * 64;
    h16 *z = calloc(n2, 2);
    id<MTLBuffer> bz = nn_buf(&r, z, n2), bp = nn_buf(&r, NULL, n2), bq = nn_buf(&r, NULL, n2);
    nn_run(&r, ^(MetalFGEnc *e) {
        [r.nn conv:e args:cargs("n2.a.res0.w", bz, bp, Wn, Hn, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_ZADD)];
        [r.nn conv:e args:cargs("n2.a.res0.w", bz, bq, Wn, Hn, METALFG_CONV_ACT | METALFG_CONV_BOUND)];
    });
    uint16_t with = ((uint16_t *)bp.contents)[0], without = ((uint16_t *)bq.contents)[0];
    CHECKC(with == 0x0000, "ar-act-res: the explicit zero add leaves bits %04x, want +0", with);
    CHECKC(without == 0x8000, "ar-act-res control: skipping the add must leave negative zero (bits %04x; otherwise the case is void)", without);
    free(in); free(m); free(z);
}

/* ---- ar-pool-order ---- */
static void ar_pool_order(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n2.a.d1.w");
        w[((0 * 3 + 1) * 3 + 1) * 32 + 0] = hb(1.0);       /* output 0 = input 0 through the centre tap */
    });
    int W = 2, H = 2;
    size_t n = (size_t)W * H * 32;
    h16 *in = calloc(n, 2);
    in[0 * 32] = (h16)2048; in[1 * 32] = (h16)1; in[2 * 32] = (h16)1; in[3 * 32] = (h16)1;       /* raster order within the block */
    id<MTLBuffer> bi = nn_buf(&r, in, n), bo = nn_buf(&r, NULL, 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n2.a.d1.w", bi, bo, W, H, METALFG_CONV_ACT | METALFG_CONV_POOL)]; });
    h16 m[32];
    model_layer(&r, "n2.a.d1", in, W, H, 32, METALFG_CONV_ACT | METALFG_CONV_POOL, 3, m);
    const h16 *o = bo.contents;
    CHECKC(memcmp(&o[0], &m[0], 2) == 0 && (double)o[0] == 512, "ar-pool-order: %g, want 512", (double)o[0]);
    h16 pair = (h16)((float)(h16)(2048.0f + 1.0f) + (float)(h16)(1.0f + 1.0f));                  /* pairwise: (2048 + 1) + (1 + 1) */
    CHECKC((double)pair == 2050 && (double)pair * 0.25 != 512, "ar-pool-order control: pairwise addition must give 2050 before the quarter (%g)", (double)pair);
    free(in);
}

/* ---- ar-up2-edge ---- */
static void ar_up2_edge(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    int Ws = 6, Hs = 1;
    h16 *src = calloc((size_t)Ws * Hs, 2);
    for (int x = 0; x < Ws; x++) src[x] = (h16)(x == 0 ? 0.7f : 0.0f);
    id<MTLBuffer> bi = nn_buf(&r, src, Ws * Hs), bo = nn_buf(&r, NULL, 4 * Ws * Hs);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn up2x:e src:bi out:bo Ws:Ws Hs:Hs C:1]; });
    h16 m[4 * 6];
    ref_up2x(src, Ws, Hs, 1, m);
    CHECKC(count_diff(bo.contents, m, 4 * Ws * Hs) == 0, "ar-up2-edge: engine differs from the model");
    double v = (double)src[0], got = (double)((h16 *)bo.contents)[0];
    CHECKC(fabs(got - v) <= v * 0x1p-9, "ar-up2-edge: output column 0 is %g, want %g", got, v);
    CHECKC(fabs(0.75 * v - got) > v * 0x1p-9, "ar-up2-edge control (the left neighbour counted as zero) does not differ");
    free(src);
}

/* ---- ar-join-quant ---- */
static void ar_join_quant(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    int Nw = 640, Nh = 2;
    size_t n0 = (size_t)Nw * Nh;
    h16 *CA = calloc(n0 * 4, 2), *CB = calloc(n0 * 4, 2), *HM = calloc(n0, 2), *U = calloc(n0 * 8, 2);
    CA[(size_t)600 * 4] = (h16)0; CA[(size_t)601 * 4] = (h16)1;          /* channel 0: zero at 600, one at 601 */
    for (size_t p = 0; p < n0; p++) U[p * 8 + 0] = (h16)0.3f;           /* offset 0.3 in x for candidate A */
    id<MTLBuffer> bA = nn_buf(&r, CA, n0 * 4), bB = nn_buf(&r, CB, n0 * 4), bH = nn_buf(&r, HM, n0), bU = nn_buf(&r, U, n0 * 8), bZ = nn_buf(&r, NULL, n0 * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn join:e a:bA b:bB h:bH u:bU out:bZ Nw:Nw Nh:Nh t:0.5f]; });
    const h16 *z = bZ.contents;
    h16 want[4];
    ref_gather4(CA, Nw, Nh, 600, 0, U[0], U[1], want);
    CHECKC(memcmp(&z[(size_t)600 * 32], want, 8) == 0, "ar-join-quant: engine differs from the model");
    CHECKC((double)z[(size_t)600 * 32] == 0.5, "ar-join-quant: the sample at 600 is %g, want one half (fraction one half)", (double)z[(size_t)600 * 32]);
    float pos32 = 600.0f + 2.0f * (float)U[0], fr32 = pos32 - floorf(pos32);
    CHECKC(fabsf(fr32 - 0.5f) > 0.05f, "ar-join-quant control: the binary32 fraction %g must be near six tenths", fr32);
    free(CA); free(CB); free(HM); free(U);
}

/* ---- ar-head-sat ---- */
static void ar_head_sat(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n2.b.headB.w");
        for (int row = 0; row < 8; row++) w[(((size_t)row * 3 + 1) * 3 + 1) * 16 + 0] = hb(1.0);
    });
    int W = 8, H = 8;
    size_t n0 = (size_t)W * H;
    h16 *x = calloc(n0 * 48, 2), *u = calloc(n0 * 8, 2);
    for (size_t p = 0; p < n0; p++) {
        for (int q = 0; q < 3; q++) x[p * 48 + 16 * q] = (h16)20;
        for (int c = 0; c < 8; c++) u[p * 8 + c] = (h16)(c < 4 ? 1.5 : -3.0);
    }
    id<MTLBuffer> bx = nn_buf(&r, x, n0 * 48), bk = nn_buf(&r, NULL, n0 * 8), bu = nn_buf(&r, u, n0 * 8), bo = nn_buf(&r, NULL, n0 * 8);
    static const uint32_t rows0[3] = {0, 4, 5}, nrows[3] = {4, 1, 3};
    nn_run(&r, ^(MetalFGEnc *e) {
        for (int q = 0; q < 3; q++) {
            MetalFGConvArgs hb2 = cargs("n2.b.headB.w", bx, bk, W, H, q ? METALFG_CONV_BOUND : 0);
            hb2.cin = 16; hb2.in_stride = 48; hb2.in_base = 16 * q; hb2.row0 = rows0[q]; hb2.cout = nrows[q]; hb2.out_stride = 8; hb2.out_base = rows0[q];
            [r.nn conv:e args:hb2];
        }
        [r.nn final:e k:bk u:bu out:bo Nw:W Nh:H];
    });
    const h16 *o = bo.contents;
    /* model: heads 1 and 2 saturated first (8), then the sum with the joined sample; head 0 is not saturated */
    int bad = 0, dctl = 0;
    for (size_t p = 0; p < n0; p++) for (int c = 0; c < 8; c++) {
        float head = c < 4 ? 20.0f : 8.0f, uu = c < 4 ? 1.5f : -3.0f;
        h16 want = (h16)(head + (c < 4 ? 2.0f : 1.0f) * uu);
        bad += memcmp(&o[p * 8 + c], &want, 2) != 0;
        if (c >= 4) { float s = 20.0f + uu; s = s > 8 ? 8 : s < -8 ? -8 : s; dctl += (float)o[p * 8 + c] != s; }     /* control: add, then saturate */
    }
    CHECKC(bad == 0, "ar-head-sat: %d elements differ from saturate-then-add", bad);
    CHECKC(dctl > 0, "ar-head-sat control (add, then saturate) does not differ");
    free(x); free(u);
}

/* ---- ar-up-scale ---- */
static void ar_up_scale(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    const int ext[][2] = {{1, 1}, {2, 2}, {3, 2}, {15, 9}, {30, 18}, {32, 32}, {45, 27}, {60, 34}};
    int ctl_differs = 0;
    for (size_t e = 0; e < sizeof ext / sizeof ext[0]; e++) {
        int Ws = ext[e][0], Hs = ext[e][1], C = 3, Wd = 2 * Ws, Hd = 2 * Hs;
        WgenRng rng; wgen_seed(&rng, 500 + (uint64_t)e);
        h16 *src = rand_h16(&rng, (size_t)Ws * Hs * C, -300, 300), *skip = rand_h16(&rng, (size_t)Wd * Hd * C, -300, 300);
        size_t no = (size_t)Wd * Hd * C;
        id<MTLBuffer> bs = nn_buf(&r, src, (size_t)Ws * Hs * C), bk = nn_buf(&r, skip, no), bo = nn_buf(&r, NULL, no);
        nn_run(&r, ^(MetalFGEnc *en) { [r.nn upCornerAdd:en src:bs skip:bk out:bo Ws:Ws Hs:Hs C:C]; });
        /* the admissible scales: every binary32 number within 2^-21 relative of the exact quotient */
        float sx[64], sy[64]; int nx = 0, ny = 0;
        for (int ax = 0; ax < 2; ax++) {
            int num = ax ? Hs - 1 : Ws - 1, den = ax ? Hd - 1 : Wd - 1;
            long double q = den > 0 ? (long double)num / (long double)den : 0.0L;
            float f0 = (float)q, set[64]; int cnt = 0;
            if (q == 0) set[cnt++] = 0.0f;
            else {
                float lo = f0;
                while (fabsl((long double)nextafterf(lo, 0.0f) - q) <= q * 0x1p-21L) lo = nextafterf(lo, 0.0f);
                for (float f = lo; cnt < 64 && fabsl((long double)f - q) <= q * 0x1p-21L; f = nextafterf(f, 1e9f)) set[cnt++] = f;
            }
            if (ax) { memcpy(sy, set, sizeof set); ny = cnt; } else { memcpy(sx, set, sizeof set); nx = cnt; }
        }
        h16 *m = malloc(no * 2);
        int matched = 0;
        for (int i = 0; i < nx && !matched; i++) for (int j = 0; j < ny && !matched; j++) {
            ref_up_corner_scaled(src, Ws, Hs, C, skip, m, sx[i], sy[j]);
            matched = count_diff(bo.contents, m, no) == 0;
        }
        CHECKC(matched, "ar-up-scale %dx%d: the engine output equals the model for none of the %d admissible scales", Ws, Hs, nx * ny);
        /* control: a scale moved by eight times the bound */
        float cx = (Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.0f) * (1.0f + 8 * 0x1p-21f), cy = (Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.0f) * (1.0f + 8 * 0x1p-21f);
        ref_up_corner_scaled(src, Ws, Hs, C, skip, m, cx, cy);
        if (count_diff(bo.contents, m, no) > 0) ctl_differs++;
        free(src); free(skip); free(m);
    }
    CHECKC(ctl_differs > 0, "ar-up-scale control (scale moved by eight bounds) differs on no extent: case void");
}

/* ---- ar-logistic, ar-mask-fma (pass level on F17, outputs read in binary32) ---- */
static void ar_f17(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    MetalFGIO *io = [[MetalFGIO alloc] initWithDevice:g.dev library:r.lib];
    MetalFGParams P = wgen_params();
    const uint32_t Wf = 64, Hf = 64, w1 = 32, h1 = 32;
    static const double S1[] = {0, 0.25, -0.25, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 3, -3, 4.5, -4.5, 6, -6, 8, -8, 10, -10, 12.5, -12.5, 16, -16, 20, -20, 30, -30,
                               40, -40, 60, -60, 64, -64, 88, -88, 100, -100, 120, -120, 150, -150, 200, -200, 256, -256, 0.1, -0.1, 0.7, -0.7, 2.3, -2.3, 7.7, -7.7,
                               13.3, -13.3, 25.1, -25.1, 33.3, -33.3, 77, -77, 99.9};
    const int NS = sizeof S1 / sizeof S1[0];
    id<MTLBuffer> out = nn_buf(&r, NULL, (size_t)w1 * h1 * 16);
    h16 *o = out.contents;
    id<MTLTexture> ms = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf), uo = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf), uc = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf);
    id<MTLTexture> lgP = gpu_tex(&g, MTLPixelFormatRG32Float, Wf, Hf), lgC = gpu_tex(&g, MTLPixelFormatRG32Float, Wf, Hf), st = gpu_tex(&g, MTLPixelFormatR32Float, Wf, Hf);
    for (id<MTLTexture> t in @[ uo, uc ]) { uint8_t *z = calloc(Wf * Hf, 1); gpu_upload(t, z, 1); free(z); }
    /* k-values per Feat pixel: pixel (x, y) holds the pair (k2, k3) = (S1[x], S1[y]) and k0 = S1[y], k1 = 0 */
    for (uint32_t Y = 0; Y < Hf; Y++) for (uint32_t X = 0; X < Wf; X++) {
        h16 *px = o + ((size_t)(Y / 2) * w1 + X / 2) * 16;
        int sl = 2 * (Y & 1) + (X & 1);
        px[sl] = (h16)S1[Y % NS]; px[4 + sl] = (h16)0; px[8 + sl] = (h16)S1[X % NS]; px[12 + sl] = (h16)S1[Y % NS];
    }
    void (^run)(uint32_t) = ^(uint32_t mask_code) {
        uint8_t *mm = malloc(Wf * Hf);
        memset(mm, mask_code, Wf * Hf);
        gpu_upload(ms, mm, 1);
        free(mm);
        MetalFGF17Args a = {ms, uo, uc, lgP, lgC, st, out, Wf, Hf, w1, h1, 0, 0, 0, P.b_steer, P.th_mask};
        nn_run(&r, ^(MetalFGEnc *e) { [io f17:e args:a]; });
    };
    /* ar-logistic: mask zero */
    run(0);
    uint8_t *dummy = NULL; (void)dummy;
    float *stv = malloc(Wf * Hf * 4);
    gpu_download(st, stv, 4);
    const double bound = 0x1p-20 + 0x1p-21 + 0x1p-22;
    int bad = 0, ctl_leaves = 0;
    for (uint32_t Y = 0; Y < Hf; Y++) for (uint32_t X = 0; X < Wf; X++) {
        float k2 = (float)(h16)S1[X % NS], k3 = (float)(h16)S1[Y % NS];
        float arg = fabsf(k2 - k3) * -1.44269502f;                    /* the binary32 argument, formed as the engine is specified to */
        double e = exp2((double)arg), num = k2 > k3 ? 1.0 : e, model = num / (e + 1.0);
        double got = stv[Y * Wf + X];
        if (fabs(got - model) > bound * model + 0x1p-126) bad++;
        double c16 = (double)(h16)got;                                 /* control: the state rounded to binary16 */
        if (fabs(c16 - model) > bound * model + 0x1p-126) ctl_leaves++;
    }
    CHECKC(bad == 0, "ar-logistic: %d state values outside the bound of the model", bad);
    CHECKC(ctl_leaves > 0, "ar-logistic control (state rounded to binary16) stays inside the bound everywhere: case void");
    {   /* equal logits: one half within the bound */
        int badeq = 0;
        for (int i = 0; i < NS; i++) badeq += fabs(stv[i * Wf + i] - 0.5) > bound * 0.5;
        CHECKC(badeq == 0, "ar-logistic: equal logits do not give one half (%d)", badeq);
    }
    free(stv);
    /* ar-mask-fma: mask code m (as m / 255), override level -b: k0' = fma(m, -b - k0, k0) in one rounding */
    const uint32_t code = 77;
    run(code);
    float *lp = malloc(Wf * Hf * 8);
    gpu_download(lgP, lp, 8);
    const float m = (float)code / 255.0f, b = P.b_steer;
    int badm = 0, dsep = 0;
    for (uint32_t Y = 0; Y < Hf; Y++) for (uint32_t X = 0; X < Wf; X++) {
        float k0 = (float)(h16)S1[Y % NS];
        float fused = fmaf(m, -b - k0, k0), sep = k0 + m * (-b - k0);
        float got = lp[(Y * Wf + X) * 2];
        badm += got != fused;
        dsep += sep != fused;
    }
    CHECKC(badm == 0, "ar-mask-fma: %d overridden logits differ from the single-rounded fused value", badm);
    CHECKC(dsep > 0, "ar-mask-fma control (separate multiply and add) equals the fused value everywhere: case void");
    free(lp);
}

/* ---- ar-luma ---- */
static void ar_luma(Gpu g) {
    NSString *hdr = [NSString stringWithContentsOfFile:@"shaders/fgcommon.h" encoding:NSUTF8StringEncoding error:nil];
    if (!hdr) { printf("shaders/fgcommon.h not found (run from the repository root)\n"); exit(2); }
    NSString *src = [hdr stringByAppendingString:@"\nkernel void lm(device const float *c [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]]) {"
                                                  " o[i] = lum3(float3(c[3 * i], c[3 * i + 1], c[3 * i + 2])); }\n"];
    id<MTLComputePipelineState> pso = src_pso(&g, src, @"lm");
    enum { N = 4096 };
    float *c = malloc(N * 12);
    WgenRng rng; wgen_seed(&rng, 61);
    for (int i = 0; i < N * 3; i++) c[i] = (float)wgen_uniform(&rng);
    id<MTLBuffer> bc = src_buf(&g, c, N * 12), bo = src_buf(&g, NULL, N * 4);
    src_run(&g, pso, @[ bc, bo ], N);
    const float *o = bo.contents;
    int bad = 0, dplain = 0;
    for (int i = 0; i < N; i++) {
        float R = c[3 * i], G = c[3 * i + 1], B = c[3 * i + 2];
        float model = fmaf(0.114f, B, fmaf(0.299f, R, 0.587f * G));
        float plain = 0.299f * R + 0.587f * G + 0.114f * B;
        bad += o[i] != model;
        dplain += plain != model;
    }
    CHECKC(bad == 0, "ar-luma: %d colours differ from the fused model", bad);
    CHECKC(dplain > 0, "ar-luma control (plain sum of products) agrees everywhere: case void");
    free(c);
}

/* ---- ar-conv-layers: every convolution layer shape that the register-blocked path serves,
   on a grid with interior tiles, partial tiles and border tiles, wide-range random weights and inputs, each epilogue variant of the networks ---- */
static void ar_mma_layers(Gpu g) {
    const int W = 26, H = 14;           /* not multiples of the tile: partial tiles; interior tiles exist */
    int tested = 0;
    for (int ti = 0; ti < METALFG_TENSOR_COUNT; ti++) {
        const MetalFGTensorInfo *t = &metalfg_weight_table()[ti];
        if (t->layout != 0 || (t->kh != 1 && t->kh != 3)) continue;
        if (t->I % 16) continue;                                 /* shapes of the register-blocked path only */
        const char *wn = t->name;
        int net2 = !strncmp(wn, "n2.", 3);
        int flagsets[3], nf = 0;
        if (!net2) { flagsets[nf++] = METALFG_CONV_ACT; if (strstr(wn, "head")) flagsets[nf++] = 0; }
        else if (strstr(wn, ".d")) { flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_POOL; }
        else if (strstr(wn, ".res")) { flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_ZADD; flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_RES; }
        else flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND;
        size_t nw = (size_t)t->O * t->kh * t->kw * t->I;
        WgenRng rng; wgen_seed(&rng, 900 + (uint64_t)ti);
        h16 *wide = malloc(nw * 2);
        wide_fill(wide, nw, &rng);
        NnRig r = open_with(g, WGEN_RANDOM, 2, ^(uint16_t **tt) { memcpy(tt[ti], wide, nw * 2); });
        size_t nin = (size_t)W * H * t->I, nout = (size_t)W * H * t->O;
        h16 *in = malloc(nin * 2), *res = malloc(nout * 2);
        wide_fill(in, nin, &rng);
        wide_fill(res, nout, &rng);
        char stem[64];
        snprintf(stem, sizeof stem, "%.*s", (int)(strlen(wn) - 2), wn);
        /* output ranges: the whole tensor; for the head tensors also the owned row ranges of the networks (predicated channels) */
        int ranges[4][2] = {{0, (int)t->O}, {0, 0}, {0, 0}, {0, 0}};
        int nr = 1;
        if (strstr(wn, "headB")) { ranges[1][0] = 0; ranges[1][1] = 4; ranges[2][0] = 4; ranges[2][1] = 1; ranges[3][0] = 5; ranges[3][1] = 3; nr = 4; }
        for (int ri = 0; ri < nr; ri++)
        for (int fi = 0; fi < nf; fi++) {
            int fl = flagsets[fi];
            int row0 = ranges[ri][0], cnt = ranges[ri][1];
            id<MTLBuffer> bi = nn_buf(&r, in, nin), br = nn_buf(&r, res, nout), bo = nn_buf(&r, NULL, nout);
            nn_run(&r, ^(MetalFGEnc *e) {
                MetalFGConvArgs a = cargs(wn, bi, bo, W, H, fl);
                a.row0 = row0; a.cout = cnt; a.out_stride = t->O; a.out_base = row0;
                if (fl & METALFG_CONV_RES) { a.res = br; a.res_stride = t->O; }
                if (strstr(wn, "headA")) a.bound_from = net2 && strstr(wn, "n2.a") ? 32 : 16;
                [r.nn conv:e args:a];
            });
            size_t no = (fl & METALFG_CONV_POOL) ? (size_t)(W / 2) * (H / 2) * t->O : nout;
            h16 *m = malloc(nout * 2);
            char wn2[64], bn2[64];
            snprintf(wn2, sizeof wn2, "%s.w", stem);
            snprintf(bn2, sizeof bn2, "%s.b", stem);
            g_ref_nest = nest_for_name(wn);
            int bf = strstr(wn, "headA") ? (net2 && strstr(wn, "n2.a") ? 32 : 16) : 0;
            ref_conv(in, W, H, t->I, 0, t->I, wt(&r, wn2), wt(&r, bn2), t->kh, t->I, row0, cnt, fl, r.params.leak, r.params.sat_n2, bf, m, t->O, row0,
                     (fl & METALFG_CONV_RES) ? res : NULL, t->O, 0);
            int d = 0;
            int ow = (fl & METALFG_CONV_POOL) ? W / 2 : W, oh = (fl & METALFG_CONV_POOL) ? H / 2 : H;
            for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) for (int c = row0; c < row0 + cnt; c++) { size_t o = ((size_t)y * ow + x) * t->O + c; d += memcmp((h16 *)bo.contents + o, m + o, 2) != 0; }
            if (nn_pinned()) CHECKC(d == 0, "ar-conv-layers %s (flags %d): %d outputs differ from the model", wn, fl, d);
            else printf("  FIG ar-conv-layers %s flags %d: %d of %zu outputs differ from the model\n", wn, fl, d, (size_t)cnt * ow * oh);
            tested++;
            free(m);
        }
        free(wide); free(in); free(res);
    }
    CHECKC(tested >= 30, "ar-conv-layers: only %d layer variants tested", tested);
    printf("  ar-conv-layers: %d layer variants\n", tested);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        printf("store probe: image stores %s\n", store_is_nearest(&g) ? "round to nearest even" : "truncate toward zero");
        if (nn_pinned()) { ar_pair_round(g); ar_sum_order(g); ar_nest(g); }   /* void under fg.arithmetic_tolerance_r1: they test the order of the sum */
        ar_act_res(g); ar_pool_order(g); ar_up2_edge(g); ar_join_quant(g); ar_head_sat(g); ar_up_scale(g);
        ar_f17(g); ar_luma(g); ar_mma_layers(g);   /* off the pinned paths the layers report the count of outputs that differ from the model */
    }
    printf(fails ? "g_arith: %d FAILURES\n" : "g_arith: ok\n", fails);
    return fails != 0;
}
