/* net2 forward (spec fg.net2_forward section 7): n2f-dn-clamp, -dn-impulse, -res-pair, -pair-chain, -up2-impulse, -join-shift, -head-rows,
   -final-residual, -d0-padding, -random-full, -coverage, with controls. */
#import <Foundation/Foundation.h>
#include "nnutil.h"

static int fails;

static MetalFGConvArgs cargs(const char *name, id<MTLBuffer> in, id<MTLBuffer> out, int W, int H, int flags) {
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H; a.cin = t->I; a.in_stride = t->I; a.cout = t->O; a.out_stride = t->O;
    a.flags = flags; a.widx = wi;
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

/* reference conv with the rig's named weights; cin/cin_w from the table */
static void rconv(NnRig *r, const char *stem, const h16 *in, int W, int H, h16 *out, int flags, int row0, int cout, int in_stride, int in_base, int cin,
                  int out_stride, int out_base, const h16 *res, int res_stride, int bound_from) {
    char wn[64], bn[64];
    snprintf(wn, sizeof wn, "%s.w", stem);
    snprintf(bn, sizeof bn, "%s.b", stem);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[metalfg_weight_index(wn)];
    ref_conv(in, W, H, in_stride, in_base, cin, wt(r, wn), wt(r, bn), 3, t->I, row0, cout, flags, r->params.leak, r->params.sat_n2, bound_from,
             out, out_stride, out_base, res, res_stride, 0);
}

#define ACT 1
#define BND 2
#define POOL 4
#define RES 8
#define ZADD 16

static void rdn(NnRig *r, const char *stem, const h16 *in, int W, int H, int cin, int cout, h16 *out) {
    rconv(r, stem, in, W, H, out, ACT | BND | POOL, 0, cout, cin, 0, cin, cout, 0, NULL, 0, 0);
}

/* control: the former reading of fg.net2_forward, a skip from the pair's input on the first convolution as well */
static int ref_twoskip;

static void rres_stage(NnRig *r, const char *pfx, h16 *a, int W, int H, int C, h16 **y_out) {
    char n1[48], n2[48];
    h16 *cur = a;
    for (int j = 0; j < 4; j++) {
        snprintf(n1, sizeof n1, "%s.res%d", pfx, 2 * j);
        snprintf(n2, sizeof n2, "%s.res%d", pfx, 2 * j + 1);
        h16 *hh = malloc((size_t)W * H * C * 2), *nx = malloc((size_t)W * H * C * 2);
        if (ref_twoskip) rconv(r, n1, cur, W, H, hh, ACT | BND | RES, 0, C, C, 0, C, C, 0, cur, C, 0);
        else rconv(r, n1, cur, W, H, hh, ACT | BND | 16, 0, C, C, 0, C, C, 0, NULL, 0, 0);
        rconv(r, n2, hh, W, H, nx, ACT | BND | RES, 0, C, C, 0, C, C, 0, cur, C, 0);
        free(hh);
        if (cur != a) free(cur);
        cur = nx;
    }
    *y_out = cur;
}

static const int ROWS0[3] = {0, 4, 5}, NROWS[3] = {4, 1, 3};

/* full reference. CA, CB: Nw x Nh x 4, HM: Nw x Nh. N: Nw x Nh x 8 */
static void ref_net2(NnRig *r, const h16 *CA, const h16 *CB, const h16 *HM, double t, int Nw, int Nh, h16 *N) {
    int w[4], h[4];
    for (int g = 0; g < 4; g++) { w[g] = Nw >> g; h[g] = Nh >> g; }
    size_t n0 = (size_t)Nw * Nh;
    h16 *x0 = calloc(n0 * 16, 2);
    for (size_t p = 0; p < n0; p++) {
        for (int c = 0; c < 4; c++) { x0[p * 16 + c] = CA[p * 4 + c]; x0[p * 16 + 4 + c] = CB[p * 4 + c]; }
        x0[p * 16 + 8] = HM[p];
        x0[p * 16 + 9] = (h16)t;
    }
    h16 *p1 = malloc((size_t)w[1] * h[1] * 32 * 2), *p2 = malloc((size_t)w[2] * h[2] * 32 * 2), *a0 = malloc((size_t)w[3] * h[3] * 64 * 2);
    rdn(r, "n2.a.d0", x0, w[0], h[0], 16, 32, p1);
    rdn(r, "n2.a.d1", p1, w[1], h[1], 32, 32, p2);
    rdn(r, "n2.a.d2", p2, w[2], h[2], 32, 64, a0);
    h16 *y;
    rres_stage(r, "n2.a", a0, w[3], h[3], 64, &y);
    h16 *wA = malloc((size_t)w[2] * h[2] * 64 * 2);
    ref_up2(y, w[3], h[3], 64, wA);
    h16 *c[3], *v[3], *m[3], *u[3];
    for (int q = 0; q < 3; q++) {
        c[q] = malloc((size_t)w[2] * h[2] * 32 * 2);
        rconv(r, "n2.a.headA", wA, w[2], h[2], c[q], ACT | (q ? BND : 0), 32 * q, 32, 64, 0, 64, 32, 0, NULL, 0, 0);
        v[q] = malloc((size_t)w[1] * h[1] * 32 * 2);
        ref_up2x(c[q], w[2], h[2], 32, v[q]);   /* rule E: the enlargement feeding the final head */
        m[q] = malloc((size_t)w[1] * h[1] * NROWS[q] * 2);
        rconv(r, "n2.a.headB", v[q], w[1], h[1], m[q], q ? BND : 0, ROWS0[q], NROWS[q], 32, 0, 32, NROWS[q], 0, NULL, 0, 0);
        u[q] = malloc(n0 * NROWS[q] * 2);
        ref_up2(m[q], w[1], h[1], NROWS[q], u[q]);
    }
    /* stage join */
    h16 *Z = calloc(n0 * 18, 2);
    for (int Y = 0; Y < Nh; Y++)
        for (int X = 0; X < Nw; X++) {
            size_t p = (size_t)Y * Nw + X;
            const h16 *img[2] = {CA, CB};
            for (int s = 0; s < 2; s++) {
                ref_gather4(img[s], Nw, Nh, X, Y, u[0][p * 4 + 2 * s], u[0][p * 4 + 2 * s + 1], &Z[p * 18 + 4 * s]);   /* rule F */
            }
            for (int ch = 0; ch < 4; ch++) Z[p * 18 + 8 + ch] = u[0][p * 4 + ch];
            Z[p * 18 + 12] = u[1][p];
            Z[p * 18 + 13] = HM[p];
            for (int ch = 0; ch < 3; ch++) Z[p * 18 + 14 + ch] = u[2][p * 3 + ch];
            Z[p * 18 + 17] = (h16)t;
        }
    /* fine stage */
    h16 *f1 = malloc((size_t)w[1] * h[1] * 16 * 2), *f2 = malloc((size_t)w[2] * h[2] * 32 * 2);
    rdn(r, "n2.b.d0", Z, w[0], h[0], 18, 16, f1);
    rdn(r, "n2.b.d1", f1, w[1], h[1], 16, 32, f2);
    h16 *f3;
    rres_stage(r, "n2.b", f2, w[2], h[2], 32, &f3);
    h16 *wB = malloc((size_t)w[1] * h[1] * 32 * 2);
    ref_up2(f3, w[2], h[2], 32, wB);
    h16 *k[3];
    for (int q = 0; q < 3; q++) {
        h16 *e = malloc((size_t)w[1] * h[1] * 16 * 2), *z = malloc(n0 * 16 * 2);
        rconv(r, "n2.b.headA", wB, w[1], h[1], e, ACT | (q ? BND : 0), 16 * q, 16, 32, 0, 32, 16, 0, NULL, 0, 0);
        ref_up2x(e, w[1], h[1], 16, z);
        k[q] = malloc(n0 * NROWS[q] * 2);
        rconv(r, "n2.b.headB", z, w[0], h[0], k[q], q ? BND : 0, ROWS0[q], NROWS[q], 16, 0, 16, NROWS[q], 0, NULL, 0, 0);
        free(e); free(z);
    }
    for (size_t p = 0; p < n0; p++) {
        for (int ch = 0; ch < 4; ch++) N[p * 8 + ch] = (h16)((double)k[0][p * 4 + ch] + 2.0 * (double)u[0][p * 4 + ch]);
        N[p * 8 + 4] = (h16)((double)k[1][p] + (double)u[1][p]);
        for (int ch = 0; ch < 3; ch++) N[p * 8 + 5 + ch] = (h16)((double)k[2][p * 3 + ch] + (double)u[2][p * 3 + ch]);
    }
    free(x0); free(p1); free(p2); free(a0); free(y); free(wA); free(Z); free(f1); free(f2); free(f3); free(wB);
    for (int q = 0; q < 3; q++) { free(c[q]); free(v[q]); free(m[q]); free(u[q]); free(k[q]); }
}

/* ---- single-layer tests ---- */

static void dn_clamp(Gpu g) {
    MetalFGParams P = wgen_params();
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        WgenRng rng; wgen_seed(&rng, 5);
        uint16_t *w = tensor(t, "n2.a.d1.w");
        double amp = P.sat_n2 / (2 * sqrt(9 * 32.0));
        for (size_t i = 0; i < 32 * 9 * 32; i++) w[i] = f16bits((wgen_uniform(&rng) * 2 - 1) * amp);
    });
    int W = 32, H = 24;
    WgenRng rng; wgen_seed(&rng, 6);
    h16 *in = rand_h16(&rng, (size_t)W * H * 32, -4, 4);
    id<MTLBuffer> bi = nn_buf(&r, in, (size_t)W * H * 32), bo = nn_buf(&r, NULL, (size_t)W / 2 * H / 2 * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n2.a.d1.w", bi, bo, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_POOL)]; });
    h16 *ref = malloc((size_t)W / 2 * H / 2 * 32 * 2);
    rdn(&r, "n2.a.d1", in, W, H, 32, 32, ref);
    Cmp c = cmp_tensors(bo.contents, ref, (size_t)W / 2 * H / 2 * 32);
    CHECKC(layer_ok(c), "n2f-dn-clamp: exact %zu / %zu within1 %zu", c.exact, c.n, c.within1);
    /* controls: no bound; max pooling; pooling the input first */
    h16 *nb = malloc((size_t)W / 2 * H / 2 * 32 * 2);
    rconv(&r, "n2.a.d1", in, W, H, nb, ACT | POOL, 0, 32, 32, 0, 32, 32, 0, NULL, 0, 0);
    CHECKC(!layer_ok(cmp_tensors(bo.contents, nb, (size_t)W / 2 * H / 2 * 32)), "control: omitted bound must fail");
    free(in); free(ref); free(nb);
}

static void dn_impulse(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n2.a.d1.w");
        for (uint32_t o = 0; o < 32; o++) for (uint32_t ky = 0; ky < 3; ky++) for (uint32_t kx = 0; kx < 3; kx++) for (uint32_t i = 0; i < 32; i++)
            w[(((size_t)o * 3 + ky) * 3 + kx) * 32 + i] = f16bits((1 + (o + 3 * ky + 11 * kx + 7 * i) % 64) / 64.0 / 16);
    });
    int W = 32, H = 24;
    h16 *in = calloc((size_t)W * H * 32, 2);
    in[((size_t)6 * W + 10) * 32 + 5] = (h16)1;
    id<MTLBuffer> bi = nn_buf(&r, in, (size_t)W * H * 32), bo = nn_buf(&r, NULL, (size_t)W / 2 * H / 2 * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn conv:e args:cargs("n2.a.d1.w", bi, bo, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_POOL)]; });
    const h16 *o = bo.contents;
    int bad = 0;
    for (int y = 0; y < H / 2; y++) for (int x = 0; x < W / 2; x++) for (int oc = 0; oc < 32; oc++) {
        double s = 0;
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int px = 2 * x + dx, py = 2 * y + dy;
            for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++)
                if (px == 10 - (kx - 1) && py == 6 - (ky - 1)) s += rn16((1 + (oc + 3 * ky + 11 * kx + 7 * 5) % 64) / 64.0 / 16);
        }
        if (fabs((double)o[((size_t)y * (W / 2) + x) * 32 + oc] - rn16(s / 4)) > 1e-6) bad++;
    }
    CHECKC(bad == 0, "n2f-dn-impulse: %d wrong", bad);
    free(in);
}

static void res_pair(Gpu g) {
    NnRig r = open_with(g, WGEN_RANDOM, 7, nil);
    int W = 16, H = 12;
    WgenRng rng; wgen_seed(&rng, 8);
    h16 *a = rand_h16(&rng, (size_t)W * H * 64, -1, 1);
    id<MTLBuffer> ba = nn_buf(&r, a, (size_t)W * H * 64), bh = nn_buf(&r, NULL, (size_t)W * H * 64), bn = nn_buf(&r, NULL, (size_t)W * H * 64);
    nn_run(&r, ^(MetalFGEnc *e) {
        MetalFGConvArgs x = cargs("n2.a.res0.w", ba, bh, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_ZADD);
        [r.nn conv:e args:x];
        x = cargs("n2.a.res1.w", bh, bn, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_RES);
        x.res = ba; x.res_stride = 64;
        [r.nn conv:e args:x];
    });
    h16 *hh = malloc((size_t)W * H * 64 * 2), *nx = malloc((size_t)W * H * 64 * 2), *wrong = malloc((size_t)W * H * 64 * 2);
    rconv(&r, "n2.a.res0", a, W, H, hh, ACT | BND | ZADD, 0, 64, 64, 0, 64, 64, 0, NULL, 0, 0);
    Cmp c1 = cmp_tensors(bh.contents, hh, (size_t)W * H * 64);
    CHECKC(layer_ok(c1), "n2f-res-pair: first convolution (no skip) exact %zu / %zu", c1.exact, c1.n);
    rconv(&r, "n2.a.res1", bh.contents, W, H, nx, ACT | BND | RES, 0, 64, 64, 0, 64, 64, 0, a, 64, 0); /* from the GPU h: per-layer criterion */
    /* a sum that cancels against its skip can sit several ulps from the reference: one ulp of the operands, 2^-10, is allowed */
    Cmp c = cmp_tensors_tol(bn.contents, nx, (size_t)W * H * 64, 0x1p-10);
    CHECKC(layer_ok(c), "n2f-res-pair: exact %zu / %zu", c.exact, c.n);
    rconv(&r, "n2.a.res1", hh, W, H, wrong, ACT | BND | RES, 0, 64, 64, 0, 64, 64, 0, hh, 64, 0); /* second skip from h: control */
    CHECKC(!layer_ok(cmp_tensors(bn.contents, wrong, (size_t)W * H * 64)), "control: skip from h must fail");
    rconv(&r, "n2.a.res0", a, W, H, wrong, ACT | BND | RES, 0, 64, 64, 0, 64, 64, 0, a, 64, 0); /* skip on the first convolution as well: control */
    CHECKC(!layer_ok(cmp_tensors(bh.contents, wrong, (size_t)W * H * 64)), "control: skip on the first convolution must fail");
    rconv(&r, "n2.a.res1", bh.contents, W, H, wrong, ACT | BND, 0, 64, 64, 0, 64, 64, 0, NULL, 0, 0); /* no skip at all: control */
    CHECKC(!layer_ok(cmp_tensors(bn.contents, wrong, (size_t)W * H * 64)), "control: no skip must fail");
    free(a); free(hh); free(nx); free(wrong);
}

/* n2f-pair-chain: the four pairs of the coarse stage on random weights and a random a_0, the engine's launch sequence against the recurrence */
static double rms_h(const h16 *v, size_t n) {
    double s = 0;
    for (size_t k = 0; k < n; k++) s += (double)v[k] * (double)v[k];
    return sqrt(s / n);
}

static void pair_chain(Gpu g) {
    NnRig r = open_with(g, WGEN_RANDOM, 21, nil);
    int W = 16, H = 12;
    size_t n = (size_t)W * H * 64;
    WgenRng rng; wgen_seed(&rng, 22);
    h16 *a0 = rand_h16(&rng, n, -1, 1);
    id<MTLBuffer> b0 = nn_buf(&r, a0, n), b1 = nn_buf(&r, NULL, n), bh = nn_buf(&r, NULL, n);
    nn_run(&r, ^(MetalFGEnc *e) {
        id<MTLBuffer> c = b0, x = b1;
        for (int j = 0; j < 4; j++) {
            char n1[32], n2[32];
            snprintf(n1, sizeof n1, "n2.a.res%d.w", 2 * j);
            snprintf(n2, sizeof n2, "n2.a.res%d.w", 2 * j + 1);
            [r.nn conv:e args:cargs(n1, c, bh, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_ZADD)];
            MetalFGConvArgs s = cargs(n2, bh, x, W, H, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_RES);
            s.res = c; s.res_stride = 64;
            [r.nn conv:e args:s];
            id<MTLBuffer> t = c; c = x; x = t;
        }
    });
    id<MTLBuffer> fin = b0;   /* four swaps: the result is back in the first buffer */
    h16 *ref, *ref2;
    ref_twoskip = 0; rres_stage(&r, "n2.a", a0, W, H, 64, &ref);
    ref_twoskip = 1; rres_stage(&r, "n2.a", a0, W, H, 64, &ref2);
    ref_twoskip = 0;
    double ratio = rms_h(fin.contents, n) / rms_h(ref, n), ratio2 = rms_h(fin.contents, n) / rms_h(ref2, n);
    CHECKC(fabs(ratio - 1) <= 0x1p-8, "n2f-pair-chain: rms ratio %g", ratio);
    CHECKC(ratio2 <= 1 / 1.4, "control: two-skip recurrence must be far out (ratio %g)", ratio2);
    free(a0); free(ref); free(ref2);
}

static void up2_impulse(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    int Ws = 8, Hs = 6;
    int pts[3][2] = {{0, 0}, {3, 2}, {7, 5}};
    for (int k = 0; k < 3; k++) {
        h16 *in = calloc((size_t)Ws * Hs, 2);
        in[(size_t)pts[k][1] * Ws + pts[k][0]] = (h16)1;
        id<MTLBuffer> bi = nn_buf(&r, in, (size_t)Ws * Hs), bo = nn_buf(&r, NULL, (size_t)Ws * Hs * 4);
        nn_run(&r, ^(MetalFGEnc *e) { [r.nn up2:e src:bi out:bo Ws:Ws Hs:Hs C:1]; });
        const h16 *o = bo.contents;
        int bad = 0;
        for (int Y = 0; Y < 2 * Hs; Y++) for (int X = 0; X < 2 * Ws; X++) {
            /* per-axis weight of source index j at destination D: interior 0.25 0.75 0.75 0.25 at 2j-1..2j+2; edges repeat */
            double wx = 0, wy = 0;
            int j = pts[k][0], i = pts[k][1];
            int dxs[4] = {2 * j - 1, 2 * j, 2 * j + 1, 2 * j + 2};
            double ws[4] = {0.25, 0.75, 0.75, 0.25};
            for (int q = 0; q < 4; q++) if (X == dxs[q]) wx += ws[q];
            if (j == 0 && X == 0) wx += 0.25; /* the repeated edge column gets the clamped far sample */
            if (j == Ws - 1 && X == 2 * Ws - 1) wx += 0.25;
            int dys[4] = {2 * i - 1, 2 * i, 2 * i + 1, 2 * i + 2};
            for (int q = 0; q < 4; q++) if (Y == dys[q]) wy += ws[q];
            if (i == 0 && Y == 0) wy += 0.25;
            if (i == Hs - 1 && Y == 2 * Hs - 1) wy += 0.25;
            if (fabs((double)o[Y * 2 * Ws + X] - wx * wy) > 1e-3) bad++;
        }
        CHECKC(bad == 0, "n2f-up2-impulse (%d,%d): %d wrong", pts[k][0], pts[k][1], bad);
        free(in);
    }
}

static void join_shift(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, nil);
    int Nw = 32, Nh = 16;
    WgenRng rng; wgen_seed(&rng, 9);
    size_t n0 = (size_t)Nw * Nh;
    h16 *CA = rand_h16(&rng, n0 * 4, -1, 1), *CB = rand_h16(&rng, n0 * 4, -1, 1), *HM = rand_h16(&rng, n0, 0, 1), *U = malloc(n0 * 8 * 2);
    for (size_t p = 0; p < n0; p++) {
        double u[8] = {1, -1, 0.5, 0, 0.125, 0.25, -0.5, 0.75};
        for (int c = 0; c < 8; c++) U[p * 8 + c] = (h16)u[c];
    }
    id<MTLBuffer> bA = nn_buf(&r, CA, n0 * 4), bB = nn_buf(&r, CB, n0 * 4), bH = nn_buf(&r, HM, n0), bU = nn_buf(&r, U, n0 * 8), bZ = nn_buf(&r, NULL, n0 * 32);
    nn_run(&r, ^(MetalFGEnc *e) { [r.nn join:e a:bA b:bB h:bH u:bU out:bZ Nw:Nw Nh:Nh t:0.5f]; });
    const h16 *z = bZ.contents;
    int bad = 0;
    for (int y = 0; y < Nh; y++) for (int x = 0; x < Nw; x++) {
        size_t p = (size_t)y * Nw + x;
        int ax = x + 2, ay = y - 2, bx = x + 1, by = y;
        ax = ax > Nw - 1 ? Nw - 1 : ax; ay = ay < 0 ? 0 : ay;
        bx = bx > Nw - 1 ? Nw - 1 : bx;
        for (int c = 0; c < 4; c++) {
            if (z[p * 32 + c] != CA[((size_t)ay * Nw + ax) * 4 + c]) bad++;
            if (z[p * 32 + 4 + c] != CB[((size_t)by * Nw + bx) * 4 + c]) bad++;
        }
        for (int c = 0; c < 4; c++) if (z[p * 32 + 8 + c] != U[p * 8 + c]) bad++;
        if (z[p * 32 + 12] != U[p * 8 + 4] || z[p * 32 + 13] != HM[p] || z[p * 32 + 17] != (h16)0.5) bad++;
        for (int c = 0; c < 3; c++) if (z[p * 32 + 14 + c] != U[p * 8 + 5 + c]) bad++;
        for (int c = 18; c < 32; c++) if ((double)z[p * 32 + c] != 0) bad++;
    }
    CHECKC(bad == 0, "n2f-join-shift: %d wrong", bad);
    free(CA); free(CB); free(HM); free(U);
}

static void head_rows(Gpu g) {
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n2.a.headB.w");
        for (int row = 0; row < 8; row++) w[(((size_t)row * 3 + 1) * 3 + 1) * 32 + 0] = f16bits((row + 1) / 8.0);
    });
    int W = 32, H = 16;
    h16 *v = calloc((size_t)W * H * 96, 2);
    for (size_t p = 0; p < (size_t)W * H; p++) for (int q = 0; q < 3; q++) v[p * 96 + 32 * q] = (h16)(q + 1);
    id<MTLBuffer> bv = nn_buf(&r, v, (size_t)W * H * 96), bm = nn_buf(&r, NULL, (size_t)W * H * 8);
    nn_run(&r, ^(MetalFGEnc *e) {
        for (int q = 0; q < 3; q++) {
            MetalFGConvArgs hb = cargs("n2.a.headB.w", bv, bm, W, H, q ? METALFG_CONV_BOUND : 0);
            hb.cin = 32; hb.in_stride = 96; hb.in_base = 32 * q; hb.row0 = ROWS0[q]; hb.cout = NROWS[q]; hb.out_stride = 8; hb.out_base = ROWS0[q];
            [r.nn conv:e args:hb];
        }
    });
    const h16 *m = bm.contents;
    double want[8] = {1 / 8.0, 2 / 8.0, 3 / 8.0, 4 / 8.0, 2 * 5 / 8.0, 3 * 6 / 8.0, 3 * 7 / 8.0, 3 * 8 / 8.0};
    int bad = 0;
    for (int y = 2; y < H - 2; y++) for (int x = 2; x < W - 2; x++) for (int c = 0; c < 8; c++)
        if (fabs((double)m[((size_t)y * W + x) * 8 + c] - want[c]) > 1e-3) bad++;
    CHECKC(bad == 0, "n2f-head-rows: %d wrong", bad);
    free(v);
}

static void final_residual(Gpu g) {
    MetalFGParams P = wgen_params();
    double beta = P.sat_n2 + 1.5;
    NnRig r = open_with(g, WGEN_ZERO, 1, ^(uint16_t **t) {
        uint16_t *b = tensor(t, "n2.b.headB.b");
        for (int i = 0; i < 8; i++) b[i] = f16bits(beta);
    });
    int W = 16, H = 16;
    size_t n0 = (size_t)W * H;
    h16 *z = calloc(n0 * 48, 2), *u = malloc(n0 * 8 * 2);
    double uu[8] = {0.25, -0.25, 0.5, 0, 0.125, 0.0625, 0, -0.5};
    for (size_t p = 0; p < n0; p++) for (int c = 0; c < 8; c++) u[p * 8 + c] = (h16)uu[c];
    id<MTLBuffer> bz = nn_buf(&r, z, n0 * 48), bk = nn_buf(&r, NULL, n0 * 8), bu = nn_buf(&r, u, n0 * 8), bo = nn_buf(&r, NULL, n0 * 8);
    nn_run(&r, ^(MetalFGEnc *e) {
        for (int q = 0; q < 3; q++) {
            MetalFGConvArgs hb = cargs("n2.b.headB.w", bz, bk, W, H, q ? METALFG_CONV_BOUND : 0);
            hb.cin = 16; hb.in_stride = 48; hb.in_base = 16 * q; hb.row0 = ROWS0[q]; hb.cout = NROWS[q]; hb.out_stride = 8; hb.out_base = ROWS0[q];
            [r.nn conv:e args:hb];
        }
        [r.nn final:e k:bk u:bu out:bo Nw:W Nh:H];
    });
    const h16 *o = bo.contents;
    double want[8];
    for (int c = 0; c < 4; c++) want[c] = rn16(beta) + 2 * uu[c];
    for (int c = 4; c < 8; c++) want[c] = P.sat_n2 + uu[c];
    int bad = 0;
    for (size_t p = 0; p < n0; p++) for (int c = 0; c < 8; c++) {
        double ulp = fabs(want[c]) * 0x1p-10;
        if (fabs((double)o[p * 8 + c] - want[c]) > ulp) bad++;
    }
    CHECKC(bad == 0, "n2f-final-residual: %d wrong", bad);
    free(z); free(u);
}

static void d0_padding(Gpu g) {
    /* n2.a.d0 with non-zero columns 10..15: the staged input has zeros there, so the output equals a 10-column reference */
    NnRig r = open_with(g, WGEN_RANDOM, 13, ^(uint16_t **t) {
        uint16_t *w = tensor(t, "n2.a.d0.w");
        WgenRng rng; wgen_seed(&rng, 14);
        for (size_t k = 0; k < 32 * 9 * 16; k++) if (k % 16 >= 10) w[k] = f16bits(wgen_uniform(&rng) - 0.5);
    });
    int Nw = 16, Nh = 16;
    size_t n0 = (size_t)Nw * Nh;
    WgenRng rng; wgen_seed(&rng, 15);
    h16 *CA = rand_h16(&rng, n0 * 4, 0, 1), *CB = rand_h16(&rng, n0 * 4, 0, 1), *HM = rand_h16(&rng, n0, 0, 1);
    id<MTLBuffer> bA = nn_buf(&r, CA, n0 * 4), bB = nn_buf(&r, CB, n0 * 4), bH = nn_buf(&r, HM, n0), bx = nn_buf(&r, NULL, n0 * 16), bo = nn_buf(&r, NULL, n0 / 4 * 32);
    nn_run(&r, ^(MetalFGEnc *e) {
        [r.nn packX0:e a:bA b:bB h:bH out:bx Nw:Nw Nh:Nh t:0.5f];
        [r.nn conv:e args:cargs("n2.a.d0.w", bx, bo, Nw, Nh, METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_POOL)];
    });
    const h16 *x = bx.contents;
    int bad = 0;
    for (size_t p = 0; p < n0; p++) for (int c = 10; c < 16; c++) if ((double)x[p * 16 + c] != 0) bad++;
    CHECKC(bad == 0, "n2f-d0-padding: staged padded channels not zero");
    h16 *ref = malloc(n0 / 4 * 32 * 2);
    rconv(&r, "n2.a.d0", x, Nw, Nh, ref, ACT | BND | POOL, 0, 32, 16, 0, 10, 32, 0, NULL, 0, 0); /* 10 real columns only */
    Cmp c = cmp_tensors(bo.contents, ref, n0 / 4 * 32);
    CHECKC(layer_ok(c), "n2f-d0-padding: exact %zu / %zu", c.exact, c.n);
    free(CA); free(CB); free(HM); free(ref);
}

static void random_full(Gpu g, int Nw, int Nh) {
    NnRig r = open_with(g, WGEN_RANDOM, 51 + Nw, nil);
    size_t n0 = (size_t)Nw * Nh;
    WgenRng rng; wgen_seed(&rng, 52);
    h16 *CA = rand_h16(&rng, n0 * 4, 0, 1), *CB = rand_h16(&rng, n0 * 4, 0, 1), *HM = malloc(n0 * 2);
    for (size_t p = 0; p < n0; p++) HM[p] = (h16)(wgen_uniform(&rng) < 0.5 ? 0 : 1);
    id<MTLBuffer> bA = nn_buf(&r, CA, n0 * 4), bB = nn_buf(&r, CB, n0 * 4), bH = nn_buf(&r, HM, n0), bo = nn_buf(&r, NULL, n0 * 8);
    MetalFGNet2 *net = [[MetalFGNet2 alloc] initWithNN:r.nn Nw:Nw Nh:Nh];
    nn_run(&r, ^(MetalFGEnc *e) { [net encode:e a:bA b:bB h:bH t:0.5f out:bo]; });
    h16 *ref = malloc(n0 * 8 * 2);
    ref_net2(&r, CA, CB, HM, 0.5, Nw, Nh, ref);
    Cmp c = cmp_tensors(bo.contents, ref, n0 * 8);
    CHECKC(e2e_ok(c), "n2f-random-full %dx%d: rms %g (ref %g), max %g (ref %g), bad %zu", Nw, Nh, c.rms_err, c.rms_ref, c.max_abs, c.max_ref, c.bad);
    CHECKC(!nn_pinned() || c.exact == c.n, "n2f-random-full %dx%d: not bit exact against the model (%zu of %zu elements equal)", Nw, Nh, c.exact, c.n);
    /* control: the former two-skip recurrence */
    h16 *ref3 = malloc(n0 * 8 * 2);
    ref_twoskip = 1;
    ref_net2(&r, CA, CB, HM, 0.5, Nw, Nh, ref3);
    ref_twoskip = 0;
    CHECKC(!e2e_ok(cmp_tensors(bo.contents, ref3, n0 * 8)), "control: two-skip recurrence must fail");
    free(ref3);
    /* control: res2 and res3 exchanged */
    h16 *a = wt(&r, "n2.a.res2.w"), *b = wt(&r, "n2.a.res3.w");
    size_t nb = 64 * 9 * 64;
    h16 *save = malloc(nb * 2);
    memcpy(save, a, nb * 2); memcpy(a, b, nb * 2); memcpy(b, save, nb * 2);
    h16 *ref2 = malloc(n0 * 8 * 2);
    ref_net2(&r, CA, CB, HM, 0.5, Nw, Nh, ref2);
    CHECKC(!e2e_ok(cmp_tensors(bo.contents, ref2, n0 * 8)), "control: swapped res2/res3 must fail");
    /* coverage */
    nn_fill_nan(bo, n0 * 8);
    nn_run(&r, ^(MetalFGEnc *e) { [net encode:e a:bA b:bB h:bH t:0.5f out:bo]; });
    size_t nan = 0;
    for (size_t k = 0; k < n0 * 8; k++) nan += isnan((double)((h16 *)bo.contents)[k]);
    CHECKC(nan == 0, "n2f-coverage %dx%d: %zu NaN", Nw, Nh, nan);
    free(CA); free(CB); free(HM); free(ref); free(ref2); free(save);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        dn_clamp(g); dn_impulse(g); res_pair(g); pair_chain(g); up2_impulse(g); join_shift(g); head_rows(g); final_residual(g); d0_padding(g);
        random_full(g, 64, 48);
        random_full(g, 32, 16);
        random_full(g, 96, 64);
    }
    printf(fails ? "g_net2: %d FAILURES\n" : "g_net2: ok\n", fails);
    return fails != 0;
}
