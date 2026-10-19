/* Layer accuracy of fg.arithmetic_tolerance_r2 (section 7): tol-layer-acc and tol-layer-pos with their two controls.
   Every convolution layer shape of both networks (all epilogue variants, partial tiles, the owned row ranges of the heads) runs on the engine's convolution path;
   its stored binary16 output z is compared with the float64 evaluator y (every sum in float64, the epilogue in float64, no rounding to binary16) as
   e = (z - y) / S, S = |bias| + sum |products| + |residual| + smin. Limits: Erms 2^-8, Emax 2^-5, |Ebias| 2^-12, smin 2^-14.
   Two data sets: "random" (weights uniform within +-1/sqrt(fan-in), biases within +-0.1, inputs within +-1: fg.net1_forward section 7) and "wide" (binary16
   values of magnitude 2^-9 .. 2, random signs, as ar-conv-layers). tol-layer-pos: the largest fan-in layer of each network with non-negative weights and inputs and zero
   bias, so that S equals the magnitude of the result. Controls on those layers (evaluated on the CPU through the same error statistics): binary16 accumulation with
   truncation toward zero after every addition, and an accumulator rounded to three mantissa bits after every addition. A control that does not exceed twice its
   limit is reported as void and fails the harness. A line "FIG" per layer is the figure record that goes into docs/STATUS.md. */
#import <Foundation/Foundation.h>
#include "nnutil.h"

static int fails;
static const double ERMS = 1.0 / 256, EMAX = 1.0 / 32, EBIAS = 1.0 / 4096, SMIN = 0x1p-14;

typedef struct { double sq, mx, sum; long n; } Stat;
static void stat_add(Stat *s, double e) { s->sq += e * e; s->sum += e; if (fabs(e) > s->mx) s->mx = fabs(e); s->n++; }
static double stat_rms(const Stat *s) { return s->n ? sqrt(s->sq / s->n) : 0; }
static double stat_bias(const Stat *s) { return s->n ? s->sum / s->n : 0; }

static MetalFGConvArgs cargs(const char *name, id<MTLBuffer> in, id<MTLBuffer> out, int W, int H, int flags, int cin) {
    int wi = metalfg_weight_index(name);
    const MetalFGTensorInfo *t = &metalfg_weight_table()[wi];
    MetalFGConvArgs a;
    memset(&a, 0, sizeof a);
    a.in = in; a.out = out; a.W = W; a.H = H; a.cin = cin; a.in_stride = cin; a.cout = t->O; a.out_stride = t->O;
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

static void wide_fill(h16 *p, size_t n, WgenRng *rng, int positive) {
    for (size_t i = 0; i < n; i++) {
        double mag = ldexp(1.0 + wgen_uniform(rng), -(int)(wgen_uniform(rng) * 9));
        p[i] = (h16)((positive || wgen_uniform(rng) < 0.5 ? 1 : -1) * mag);
    }
}

/* the float64 evaluator: value y and scale S of output channel ch (relative to row0) at (x, y) of one convolution (before pooling) */
typedef struct {
    const h16 *in, *wgt, *bias, *res;
    int W, H, in_stride, cin, k, cin_w, row0, flags, bound_from, res_stride;
    double leak, bound;
} Eval;

static void eval64(const Eval *E, int x, int y, int ch, double *yv, double *S) {
    const h16 *wrow = E->wgt + (size_t)(E->row0 + ch) * E->k * E->k * E->cin_w;
    double acc = (double)E->bias[E->row0 + ch], mag = fabs(acc);
    for (int ky = 0; ky < E->k; ky++)
        for (int kx = 0; kx < E->k; kx++) {
            int px = E->k == 3 ? x + kx - 1 : x, py = E->k == 3 ? y + ky - 1 : y;
            if (px < 0 || py < 0 || px >= E->W || py >= E->H) continue;
            const h16 *ip = E->in + ((size_t)py * E->W + px) * E->in_stride;
            const h16 *wp = wrow + (E->k == 3 ? ky * 3 + kx : 0) * E->cin_w;
            for (int c = 0; c < E->cin; c++) { double p = (double)wp[c] * (double)ip[c]; acc += p; mag += fabs(p); }
        }
    double v = acc;
    if (E->flags & 1) v = fmax(v, v * E->leak);
    if ((E->flags & 2) && ch >= E->bound_from) v = fmin(fmax(v, -E->bound), E->bound);
    if (E->flags & 8) { double r = (double)E->res[((size_t)y * E->W + x) * E->res_stride + ch]; v += r; mag += fabs(r); }
    *yv = v; *S = mag + SMIN;
}

/* the two controls: a running total in a lossy accumulator, products exact */
static double ctl_acc(const Eval *E, int x, int y, int ch, int mode) {
    const h16 *wrow = E->wgt + (size_t)(E->row0 + ch) * E->k * E->k * E->cin_w;
    double acc = (double)E->bias[E->row0 + ch];
    for (int ky = 0; ky < E->k; ky++)
        for (int kx = 0; kx < E->k; kx++) {
            int px = E->k == 3 ? x + kx - 1 : x, py = E->k == 3 ? y + ky - 1 : y;
            if (px < 0 || py < 0 || px >= E->W || py >= E->H) continue;
            const h16 *ip = E->in + ((size_t)py * E->W + px) * E->in_stride;
            const h16 *wp = wrow + (E->k == 3 ? ky * 3 + kx : 0) * E->cin_w;
            for (int c = 0; c < E->cin; c++) {
                double s = acc + (double)wp[c] * (double)ip[c];
                if (mode == 1) {                       /* binary16, truncation toward zero */
                    h16 r = (h16)s;
                    if (fabs((double)r) > fabs(s)) { h16 d = (h16)nextafterf((float)r, 0.0f); r = d; }
                    acc = (double)r;
                } else {                               /* three mantissa bits, nearest */
                    int e; double m = frexp(s, &e);     /* s = m 2^e, 0.5 <= |m| < 1: four significant bits */
                    acc = ldexp(nearbyint(ldexp(m, 4)), e - 4);
                }
            }
        }
    double v = acc;
    if (E->flags & 1) v = fmax(v, v * E->leak);
    return v;
}

typedef struct { Stat all; } LayerFig;

/* one layer on the engine against the evaluator. `pos` selects non-negative weights and inputs and zero bias; data: 0 random, 1 wide */
static void run_layer(Gpu g, int ti, int fl, int data, int pos, int row0, int cnt, Stat *st, int ctl) {
    const int W = 26, H = 14;
    const MetalFGTensorInfo *t = &metalfg_weight_table()[ti];
    const char *wn = t->name;
    char stem[64], bnn[64];
    snprintf(stem, sizeof stem, "%.*s", (int)(strlen(wn) - 2), wn);
    snprintf(bnn, sizeof bnn, "%s.b", stem);
    const int net2 = !strncmp(wn, "n2.", 3);
    const int cin = t->I == 18 ? 32 : (int)t->I;
    size_t nw = (size_t)t->O * t->kh * t->kw * t->I;
    WgenRng rng;
    wgen_seed(&rng, 5000 + 7 * (uint64_t)ti + (uint64_t)data + 2 * (uint64_t)pos);
    h16 *wsrc = malloc(nw * 2);
    if (data == 1) wide_fill(wsrc, nw, &rng, pos);
    size_t nb = t->O;
    NnRig r = open_with(g, WGEN_RANDOM, 3, ^(uint16_t **tt) {
        if (data == 1) memcpy(tt[ti], wsrc, nw * 2);
        else if (pos) for (size_t i = 0; i < nw; i++) tt[ti][i] &= 0x7FFF;
        if (pos) memset(tt[ti + 1], 0, nb * 2);
    });
    size_t nin = (size_t)W * H * cin, nout = (size_t)W * H * t->O;
    h16 *in = calloc(nin, 2), *res = malloc(nout * 2);
    for (size_t i = 0; i < (size_t)W * H; i++)
        for (int c = 0; c < (int)t->I; c++) {
            double v;
            if (data == 1) { WgenRng *q = &rng; double mag = ldexp(1.0 + wgen_uniform(q), -(int)(wgen_uniform(q) * 9)); v = (pos || wgen_uniform(q) < 0.5 ? 1 : -1) * mag; }
            else v = pos ? wgen_uniform(&rng) : 2 * wgen_uniform(&rng) - 1;
            in[i * cin + c] = (h16)v;
        }
    for (size_t i = 0; i < nout; i++) { double mag = ldexp(1.0 + wgen_uniform(&rng), -(int)(wgen_uniform(&rng) * 9)); res[i] = (h16)((pos || wgen_uniform(&rng) < 0.5 ? 1 : -1) * mag); }
    id<MTLBuffer> bi = nn_buf(&r, in, nin), br = nn_buf(&r, res, nout), bo = nn_buf(&r, NULL, nout);
    int bf = strstr(wn, "headA") ? (net2 && strstr(wn, "n2.a") ? 32 : 16) : 0;
    nn_run(&r, ^(MetalFGEnc *e) {
        MetalFGConvArgs a = cargs(wn, bi, bo, W, H, fl, cin);
        a.row0 = row0; a.cout = cnt; a.out_stride = t->O; a.out_base = row0;
        if (fl & METALFG_CONV_RES) { a.res = br; a.res_stride = t->O; }
        a.bound_from = bf;
        [r.nn conv:e args:a];
    });
    Eval E = {in, wt(&r, [NSString stringWithFormat:@"%s.w", stem].UTF8String), wt(&r, bnn), res, W, H, cin, (int)t->I, (int)t->kh, (int)t->I, row0, fl, bf, (int)t->O,
              (double)(h16)r.params.leak, r.params.sat_n2};
    const int pool = (fl & METALFG_CONV_POOL) != 0, ow = pool ? W / 2 : W, oh = pool ? H / 2 : H;
    const h16 *z = bo.contents;
    long step = ctl ? 7 : 1;                                 /* the controls evaluate a subset of the outputs */
    long idx = 0;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            for (int c = 0; c < cnt; c++) {
                if (ctl && (idx++ % step)) continue;
                double yv = 0, S = 0, cv = 0;
                int n = pool ? 4 : 1;
                for (int j = 0; j < n; j++) {
                    double v, s;
                    int xx = pool ? 2 * x + (j & 1) : x, yy = pool ? 2 * y + (j >> 1) : y;
                    eval64(&E, xx, yy, c, &v, &s);
                    if (ctl) {
                        double q = ctl_acc(&E, xx, yy, c, ctl);
                        if ((fl & 2) && c >= bf) q = fmin(fmax(q, -E.bound), E.bound);
                        cv += q / n;
                    }
                    yv += v / n; S += s / n;
                }
                double zv = ctl ? cv : (double)z[((size_t)y * ow + x) * t->O + row0 + c];
                if (!ctl && (fl & 16) && ((const uint16_t *)z)[((size_t)y * ow + x) * t->O + row0 + c] == 0x8000) { fails++; printf("FAIL %s: negative zero in the output of a first convolution of a pair\n", wn); }
                stat_add(st, (zv - yv) / S);
            }
    free(wsrc); free(in); free(res);
}

static void fig(const char *what, const char *wn, int fl, const char *data, const Stat *s, int limited) {
    int ok = stat_rms(s) <= ERMS && s->mx <= EMAX && fabs(stat_bias(s)) <= EBIAS;
    printf("FIG %s %s flags=%d data=%s n=%ld Erms=%.3e Emax=%.3e Ebias=%+.3e%s\n", what, wn, fl, data, s->n, stat_rms(s), s->mx, stat_bias(s), ok ? "" : "  OVER LIMIT");
    if (limited && !ok) fails++;
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        int tested = 0;
        double wr = 0, wm = 0, wb = 0;
        long maxfan = 0, maxfan2 = 0;
        int ti1 = -1, ti2 = -1;
        for (int ti = 0; ti < METALFG_TENSOR_COUNT; ti++) {
            const MetalFGTensorInfo *t = &metalfg_weight_table()[ti];
            if (t->layout != 0 || (t->kh != 1 && t->kh != 3)) continue;
            const char *wn = t->name;
            int net2 = !strncmp(wn, "n2.", 3);
            long fan = (long)t->kh * t->kw * t->I;
            if (!net2 && fan > maxfan) { maxfan = fan; ti1 = ti; }
            if (net2 && fan > maxfan2) { maxfan2 = fan; ti2 = ti; }
            int flagsets[3], nf = 0;
            if (!net2) { flagsets[nf++] = METALFG_CONV_ACT; if (strstr(wn, "head")) flagsets[nf++] = 0; }
            else if (strstr(wn, ".d")) flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_POOL;
            else if (strstr(wn, ".res")) { flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_ZADD; flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND | METALFG_CONV_RES; }
            else flagsets[nf++] = METALFG_CONV_ACT | METALFG_CONV_BOUND;
            int ranges[4][2] = {{0, (int)t->O}, {0, 0}, {0, 0}, {0, 0}};
            int nr = 1;
            if (strstr(wn, "headB")) { ranges[1][0] = 0; ranges[1][1] = 4; ranges[2][0] = 4; ranges[2][1] = 1; ranges[3][0] = 5; ranges[3][1] = 3; nr = 4; }
            for (int data = 0; data < 2; data++)
                for (int ri = 0; ri < nr; ri++)
                    for (int fi = 0; fi < nf; fi++) {
                        Stat s = {0};
                        run_layer(g, ti, flagsets[fi], data, 0, ranges[ri][0], ranges[ri][1], &s, 0);
                        char nm[96];
                        snprintf(nm, sizeof nm, "%s[%d+%d]", wn, ranges[ri][0], ranges[ri][1]);
                        fig("tol-layer-acc", nm, flagsets[fi], data ? "wide" : "random", &s, 1);
                        if (stat_rms(&s) > wr) wr = stat_rms(&s);
                        if (s.mx > wm) wm = s.mx;
                        if (fabs(stat_bias(&s)) > wb) wb = fabs(stat_bias(&s));
                        tested++;
                    }
        }
        printf("FIG tol-layer-acc worst over %d variants: Erms=%.3e Emax=%.3e |Ebias|=%.3e (limits %.3e %.3e %.3e)\n", tested, wr, wm, wb, ERMS, EMAX, EBIAS);
        /* tol-layer-pos and the controls on the largest-fan-in layer of each network */
        int tis[2] = {ti1, ti2};
        for (int k = 0; k < 2; k++) {
            const MetalFGTensorInfo *t = &metalfg_weight_table()[tis[k]];
            int fl = METALFG_CONV_ACT;                      /* no saturation: it would hide the error of a large positive sum */
            for (int data = 0; data < 2; data++) {
                Stat s = {0};
                run_layer(g, tis[k], fl, data, 1, 0, (int)t->O, &s, 0);
                fig("tol-layer-pos", t->name, fl, data ? "wide" : "random", &s, 1);
            }
            for (int ctl = 1; ctl <= 2; ctl++) {
                int ok = 0;
                for (int data = 0; data < 2; data++) {     /* detected if either data set shows it; the data set of the figure is reported */
                    Stat s = {0};
                    run_layer(g, tis[k], fl, data, 1, 0, (int)t->O, &s, ctl);
                    double lim = ctl == 1 ? EBIAS : EMAX, got = ctl == 1 ? fabs(stat_bias(&s)) : s.mx;
                    ok |= got > 2 * lim;
                    printf("FIG tol-layer-ctl%d %s data=%s: Erms=%.3e Emax=%.3e Ebias=%+.3e, %s %.3e against twice the limit %.3e\n", ctl, t->name, data ? "wide" : "random",
                           stat_rms(&s), s.mx, stat_bias(&s), ctl == 1 ? "|Ebias|" : "Emax", got, 2 * lim);
                }
                printf("FIG tol-layer-ctl%d %s: %s\n", ctl, t->name, ok ? "detected" : "NOT DETECTED: void");
                if (!ok) fails++;
            }
        }
    }
    printf(fails ? "g_tol: %d FAILURES\n" : "g_tol: ok\n", fails);
    return fails != 0;
}
