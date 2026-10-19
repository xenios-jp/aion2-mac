/* Output synthesis (spec fg.output_synthesis section 7): os-mix, -sample-position, -offset, -fill, -sanitise, -needfill,
   -pyramid, -trusted-equal, -all-hole. */
#import <Foundation/Foundation.h>
#include "texutil.h"
#include "output.h"

static int fails;
static NnRig R;

static double sigd(double v) { return 1 / (1 + exp(-v)); }

typedef struct {
    int W, H, Nw, Nh, s;
    float *A, *B;       /* W x H x 4, half-rounded */
    int *hole;          /* W x H */
    h16 *N;             /* Nw x Nh x 8 */
} OScene;

static float q16(double v) { return (float)(h16)v; }

static OScene scene(int W, int H, int s) {
    OScene o;
    o.W = W; o.H = H; o.s = s;
    int Wp = 16 * s * ((W + 16 * s - 1) / (16 * s)), Hp = 16 * s * ((H + 16 * s - 1) / (16 * s));
    o.Nw = Wp / s; o.Nh = Hp / s;
    o.A = malloc((size_t)W * H * 16); o.B = malloc((size_t)W * H * 16);
    o.hole = calloc((size_t)W * H, sizeof(int));
    o.N = calloc((size_t)o.Nw * o.Nh * 8, 2);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float *a = &o.A[((size_t)y * W + x) * 4], *b = &o.B[((size_t)y * W + x) * 4];
        a[0] = q16(x / 64.0); a[1] = q16(y / 64.0); a[2] = q16(0.5); a[3] = 1;
        b[0] = q16(0.25); b[1] = q16(0.75); b[2] = q16(x / 64.0); b[3] = 1;
    }
    return o;
}

static void set_cells(OScene *o, void (^f)(int i, int j, double n[8])) {
    for (int j = 0; j < o->Nh; j++) for (int i = 0; i < o->Nw; i++) {
        double n[8] = {0};
        f(i, j, n);
        for (int c = 0; c < 8; c++) o->N[((size_t)j * o->Nw + i) * 8 + c] = (h16)n[c];
    }
}

static void gather(const OScene *o, const float *img, double qx, double qy, double out[3]) {
    double fx = floor(qx), fy = floor(qy), ax = qx - fx, ay = qy - fy;
    int x0 = (int)fx, y0 = (int)fy;
#define CL(v, n) ((v) < 0 ? 0 : (v) > (n) - 1 ? (n) - 1 : (v))
    for (int c = 0; c < 3; c++) {
#define T(xx, yy) (double)img[((size_t)CL(yy, o->H) * o->W + CL(xx, o->W)) * 4 + c]
        double a = (1 - ax) * T(x0, y0) + ax * T(x0 + 1, y0), b = (1 - ax) * T(x0, y0 + 1) + ax * T(x0 + 1, y0 + 1);
#undef T
        out[c] = (1 - ay) * a + ay * b;
    }
}

/* spec section 3, steps 1 to 5 */
static void ref_synth(const OScene *o, int x, int y, double out[3]) {
    double gx = fmin((double)x / o->s, o->Nw - 1), gy = fmin((double)y / o->s, o->Nh - 1);
    int i0 = (int)floor(gx), j0 = (int)floor(gy), i1 = i0 + 1 < o->Nw - 1 ? i0 + 1 : o->Nw - 1, j1 = j0 + 1 < o->Nh - 1 ? j0 + 1 : o->Nh - 1;
    double fx = gx - i0, fy = gy - j0;
    double wts[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
    int ci[4] = {j0 * o->Nw + i0, j0 * o->Nw + i1, j1 * o->Nw + i0, j1 * o->Nw + i1};
    double v[8] = {0};
    for (int k = 0; k < 4; k++) for (int c = 0; c < 8; c++) {
        double n = (double)o->N[(size_t)ci[k] * 8 + c];
        v[c] += wts[k] * (c == 4 ? sigd(n) : n);
    }
    double col[3];
    if (o->hole[(size_t)y * o->W + x]) {
        for (int c = 0; c < 3; c++) col[c] = sigd(o->s * v[5 + c]);
    } else {
        double sa[3], sb[3];
        gather(o, o->A, x + o->s * v[0], y + o->s * v[1], sa);
        gather(o, o->B, x + o->s * v[2], y + o->s * v[3], sb);
        for (int c = 0; c < 3; c++) col[c] = v[4] * sa[c] + (1 - v[4]) * sb[c];
    }
    double prod = col[0] * col[1] * col[2];
    if (isnan(prod)) { col[0] = 0; col[1] = 1; col[2] = 1; }
    for (int c = 0; c < 3; c++) out[c] = fmin(fmax(col[c], 0), 1);
}

typedef struct {
    OScene sc;
    id<MTLTexture> A, B, hole, out;
    id<MTLBuffer> N;
    MetalFGOutput *O;
} ORig;

static ORig open_case(OScene sc) {
    ORig r;
    memset((void *)&r, 0, sizeof r);
    r.sc = sc;
    r.A = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, sc.W, sc.H);
    r.B = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, sc.W, sc.H);
    r.hole = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, sc.W, sc.H);
    r.out = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, sc.W, sc.H);
    r.N = nn_buf(&R, sc.N, (size_t)sc.Nw * sc.Nh * 8);
    r.O = [[MetalFGOutput alloc] initWithDevice:R.gpu.dev library:R.lib W:sc.W H:sc.H chain:YES levels:13];
    [r.O prepareCells:(uint32_t)(sc.Nw * sc.Nh)];
    OScene *s = &r.sc;
    tex_set(r.A, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->A[((size_t)y * s->W + x) * 4], 16); });
    tex_set(r.B, ^(uint32_t x, uint32_t y, float v[4]) { memcpy(v, &s->B[((size_t)y * s->W + x) * 4], 16); });
    tex_set(r.hole, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->hole[(size_t)y * s->W + x]; });
    return r;
}

static void run(ORig *r, int chain) {
    static const uint32_t origin[2] = {0, 0};
    OScene *s = &r->sc;
    nn_run(&R, ^(MetalFGEnc *e) {
        if (chain) [r->O chain:e n2out:r->N candA:r->A candB:r->B hole:r->hole out:r->out origin:origin Nw:s->Nw Nh:s->Nh s:s->s];
        else [r->O direct:e n2out:r->N candA:r->A candB:r->B hole:r->hole out:r->out origin:origin Nw:s->Nw Nh:s->Nh s:s->s];
    });
}

static void compare_direct(ORig *r, const char *name, int skip_hole_check) {
    float *g = tex_get(r->out, 4);
    int bad = 0;
    for (int y = 0; y < r->sc.H; y++) for (int x = 0; x < r->sc.W; x++) {
        double want[3];
        ref_synth(&r->sc, x, y, want);
        for (int c = 0; c < 3; c++)
            if (ulp_dist((h16)g[((size_t)y * r->sc.W + x) * 4 + c], (h16)want[c]) > 2 && fabs(g[((size_t)y * r->sc.W + x) * 4 + c] - want[c]) > 1e-3) bad++;
        if (g[((size_t)y * r->sc.W + x) * 4 + 3] != 1) bad++;
    }
    (void)skip_hole_check;
    CHECKC(bad == 0, "%s: %d wrong components", name, bad);
    free(g);
}

static void free_scene(OScene *s) { free(s->A); free(s->B); free(s->hole); free(s->N); }

static void simple_cases(void) {
    OScene sc = scene(64, 48, 2);
    set_cells(&sc, ^(int i, int j, double n[8]) { n[4] = (i % 2) ? 4 : 0; });
    ORig r = open_case(sc);
    run(&r, 0);
    compare_direct(&r, "os-mix", 0);
    free_scene(&sc);

    sc = scene(64, 48, 2);
    set_cells(&sc, ^(int i, int j, double n[8]) { n[0] = i; n[4] = 30; });
    r = open_case(sc);
    run(&r, 0);
    compare_direct(&r, "os-sample-position", 0);
    float *g = tex_get(r.out, 4);
    int bad = 0;
    for (int x = 0; x < 64; x++) {
        double want = sc.A[((size_t)10 * 64 + (2 * x < 63 ? 2 * x : 63)) * 4];
        if (fabs(g[((size_t)10 * 64 + x) * 4] - want) > 1.0 / 512) bad++;
    }
    CHECKC(bad == 0, "os-sample-position: colour(x, y) = CAf(min(2x, 63), y): %d wrong", bad);
    free(g); free_scene(&sc);

    sc = scene(64, 48, 2);
    set_cells(&sc, ^(int i, int j, double n[8]) { n[0] = 1.5; n[1] = -0.5; n[2] = -2; n[3] = 0; });
    r = open_case(sc);
    run(&r, 0);
    compare_direct(&r, "os-offset", 0);
    free_scene(&sc);

    sc = scene(64, 48, 2);
    for (int y = 10; y < 30; y++) for (int x = 20; x < 40; x++) sc.hole[y * 64 + x] = 1;
    set_cells(&sc, ^(int i, int j, double n[8]) { n[5] = 0.5; n[6] = -0.5; n[7] = 2; });
    r = open_case(sc);
    run(&r, 0);
    compare_direct(&r, "os-fill", 0);
    g = tex_get(r.out, 4);
    const float *px = &g[((size_t)20 * 64 + 30) * 4];
    CHECKC(fabs(px[0] - 0.731) < 2e-3 && fabs(px[1] - 0.269) < 2e-3 && fabs(px[2] - 0.982) < 2e-3, "os-fill: (%g, %g, %g) vs (0.731, 0.269, 0.982) (no factor s would give 0.622, 0.378, 0.881)", px[0], px[1], px[2]);
    free(g); free_scene(&sc);

    /* os-sanitise */
    sc = scene(64, 48, 2);
    set_cells(&sc, ^(int i, int j, double n[8]) { n[4] = 30; });
    sc.A[((size_t)20 * 64 + 30) * 4] = NAN;
    sc.A[((size_t)20 * 64 + 40) * 4] = INFINITY; sc.A[((size_t)20 * 64 + 40) * 4 + 1] = 0; sc.A[((size_t)20 * 64 + 40) * 4 + 2] = 0.5f;
    sc.A[((size_t)20 * 64 + 50) * 4] = 3; sc.A[((size_t)20 * 64 + 50) * 4 + 1] = -1; sc.A[((size_t)20 * 64 + 50) * 4 + 2] = 0.5f;
    sc.A[((size_t)20 * 64 + 12) * 4] = INFINITY; sc.A[((size_t)20 * 64 + 12) * 4 + 1] = 0.25f; sc.A[((size_t)20 * 64 + 12) * 4 + 2] = 0.5f;
    r = open_case(sc);
    run(&r, 0);
    g = tex_get(r.out, 4);
    struct { int x; float e[3]; } ex[] = {{30, {0, 1, 1}}, {40, {0, 1, 1}}, {50, {1, 0, 0.5f}}, {12, {1, 0.25f, 0.5f}}};
    for (int k = 0; k < 4; k++) {
        const float *q = &g[((size_t)20 * 64 + ex[k].x) * 4];
        CHECKC(q[0] == ex[k].e[0] && q[1] == ex[k].e[1] && q[2] == ex[k].e[2] && q[3] == 1, "os-sanitise pixel %d: (%g, %g, %g, %g)", ex[k].x, q[0], q[1], q[2], q[3]);
    }
    free(g); free_scene(&sc);
}

/* ---- chain ---- */

static void ref_F(const OScene *o, int *F) {
    for (int y = 0; y < o->H; y++) for (int x = 0; x < o->W; x++) {
        int trusted = 0;
        for (int ty = y - 2; ty <= y && !trusted; ty++) for (int tx = x - 2; tx <= x && !trusted; tx++) {
            int good = 1;
            for (int j = 0; j < 3 && good; j++) for (int i = 0; i < 3; i++) {
                int px = tx + i, py = ty + j;
                if (px < 0 || py < 0 || px >= o->W || py >= o->H || o->hole[py * o->W + px]) { good = 0; break; }
            }
            trusted = good;
        }
        F[y * o->W + x] = !trusted;
    }
}

static void needfill(void) {
    for (int cs = 0; cs < 5; cs++) {
        OScene sc = scene(64, 48, 2);
        for (int y = 0; y < 48; y++) for (int x = 0; x < 64; x++) {
            int h = 0;
            if (cs == 0) h = (x == 30 && y == 20);
            if (cs == 1) h = !(x >= 10 && x <= 12);
            if (cs == 2) h = (x == 2 && y == 2);
            if (cs == 4) h = 1;
            sc.hole[y * 64 + x] = h;
        }
        set_cells(&sc, ^(int i, int j, double n[8]) { n[4] = 30; });
        ORig r = open_case(sc);
        run(&r, 1);
        float *f = tex_get(r.O.holeOut, 1);
        int *want = malloc(64 * 48 * sizeof(int));
        ref_F(&sc, want);
        int bad = 0;
        for (int i = 0; i < 64 * 48; i++) if ((f[i] > 0.5f) != want[i]) bad++;
        CHECKC(bad == 0, "os-needfill case %c: %d wrong", 'a' + cs, bad);
        /* the cases of the spec text */
        if (cs == 0) CHECKC(f[20 * 64 + 30] > 0.5f && f[20 * 64 + 31] < 0.5f, "(a) only the hole pixel");
        if (cs == 1) CHECKC(f[5 * 64 + 11] < 0.5f && f[0 * 64 + 11] < 0.5f && f[47 * 64 + 11] < 0.5f && f[5 * 64 + 9] > 0.5f && f[5 * 64 + 13] > 0.5f, "(b) strip");
        if (cs == 2) CHECKC(f[0] > 0.5f && f[2 * 64 + 2] > 0.5f && f[3 * 64 + 0] < 0.5f, "(c) corner block");
        free(f); free(want); free_scene(&sc);
    }
}

/* brute-force pyramid reference following spec steps 7 to 9 */
static int g_lw[16], g_lh[16];
static double *g_col[16], *g_cov[16];
#define lw g_lw
#define lh g_lh
#define col g_col
#define cov g_cov
static void ref_chain(const OScene *o, const double *syn, const int *F, double *final_rgb /* W x H x 3, only F = 1 pixels set */) {
    int W = o->W, H = o->H;
    int L = 0;
    int w = W, h = H;
    do { w = (w + 1) / 2; h = (h + 1) / 2; lw[L] = w; lh[L] = h; L++; } while (w > 1 || h > 1);
    for (int k = 0; k < L; k++) { col[k] = calloc((size_t)lw[k] * lh[k] * 3, sizeof(double)); cov[k] = calloc((size_t)lw[k] * lh[k], sizeof(double)); }
    const double tau = 0x1p-34;
    for (int cy = 0; cy < lh[0]; cy++) for (int cx = 0; cx < lw[0]; cx++) {
        double sc = 0, sr[3] = {0};
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int x = 2 * cx + dx, y = 2 * cy + dy;
            if (x >= W || y >= H) continue;
            double c = 1 - F[y * W + x];
            sc += c;
            for (int k = 0; k < 3; k++) sr[k] += c * syn[((size_t)y * W + x) * 3 + k];
        }
        cov[0][cy * lw[0] + cx] = sc / 4;
        for (int k = 0; k < 3; k++) col[0][(cy * lw[0] + cx) * 3 + k] = sr[k] / fmax(sc, tau);
    }
    for (int l = 1; l < L; l++) for (int cy = 0; cy < lh[l]; cy++) for (int cx = 0; cx < lw[l]; cx++) {
        double sc = 0, sr[3] = {0};
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int x = 2 * cx + dx, y = 2 * cy + dy;
            if (x >= lw[l - 1] || y >= lh[l - 1]) continue;
            double c = cov[l - 1][y * lw[l - 1] + x];
            sc += c;
            for (int k = 0; k < 3; k++) sr[k] += c * col[l - 1][(y * lw[l - 1] + x) * 3 + k];
        }
        cov[l][cy * lw[l] + cx] = sc / 4;
        for (int k = 0; k < 3; k++) col[l][(cy * lw[l] + cx) * 3 + k] = sr[k] / fmax(sc, tau);
    }
    void (^push)(int, int, int, double *, double *, double *) = ^(int l, int ax, int ay, double *cv, double *u, double *V) {
        *V = 0; u[0] = u[1] = u[2] = 0;
        int xs[2] = {ax / 2, (ax & 1) ? ax / 2 + 1 : ax / 2 - 1}, ys[2] = {ay / 2, (ay & 1) ? ay / 2 + 1 : ay / 2 - 1};
        for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
            if (xs[i] < 0 || ys[j] < 0 || xs[i] >= lw[l] || ys[j] >= lh[l]) continue;
            double wgt = (i ? 0.25 : 0.75) * (j ? 0.25 : 0.75);
            *V += wgt * cov[l][ys[j] * lw[l] + xs[i]];
            for (int k = 0; k < 3; k++) u[k] += wgt * cov[l][ys[j] * lw[l] + xs[i]] * col[l][(ys[j] * lw[l] + xs[i]) * 3 + k];
        }
        (void)cv;
    };
    for (int l = L - 2; l >= 0; l--) for (int cy = 0; cy < lh[l]; cy++) for (int cx = 0; cx < lw[l]; cx++) {
        if (cov[l][cy * lw[l] + cx] > 0) continue;
        double u[3], V;
        push(l + 1, cx, cy, NULL, u, &V);
        cov[l][cy * lw[l] + cx] = V;
        for (int k = 0; k < 3; k++) col[l][(cy * lw[l] + cx) * 3 + k] = u[k] / fmin(fmax(V, tau), 1);
    }
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        if (!F[y * W + x]) continue;
        double u[3], V;
        push(0, x, y, NULL, u, &V);
        for (int k = 0; k < 3; k++) final_rgb[((size_t)y * W + x) * 3 + k] = u[k] / fmin(fmax(V, tau), 1);
    }
    for (int k = 0; k < L; k++) { free(col[k]); free(cov[k]); }
}

#undef lw
#undef lh
#undef col
#undef cov

static void pyramid_cases(void) {
    OScene sc = scene(40, 24, 2);
    WgenRng rng; wgen_seed(&rng, 123);
    for (int i = 0; i < 40 * 24 * 4; i++) { sc.A[i] = q16(wgen_uniform(&rng)); sc.B[i] = q16(wgen_uniform(&rng)); }
    for (int i = 0; i < 40 * 24; i++) sc.hole[i] = wgen_uniform(&rng) < 1.0 / 25;
    set_cells(&sc, ^(int i, int j, double n[8]) { n[4] = 30; });       /* a = 1, zero offsets: the synthesis equals CAf */
    ORig r = open_case(sc);
    run(&r, 1);
    int *F = malloc(40 * 24 * sizeof(int));
    ref_F(&sc, F);
    double *syn = malloc(40 * 24 * 3 * sizeof(double)), *fin = calloc(40 * 24 * 3, sizeof(double));
    for (int y = 0; y < 24; y++) for (int x = 0; x < 40; x++) {
        double c[3];
        ref_synth(&sc, x, y, c);
        for (int k = 0; k < 3; k++) syn[((size_t)y * 40 + x) * 3 + k] = c[k];
    }
    ref_chain(&sc, syn, F, fin);
    float *g = tex_get(r.out, 4);
    int bad = 0, nF = 0;
    for (int y = 0; y < 24; y++) for (int x = 0; x < 40; x++) {
        if (F[y * 40 + x]) nF++;
        for (int k = 0; k < 3; k++) {
            double want = F[y * 40 + x] ? fin[((size_t)y * 40 + x) * 3 + k] : syn[((size_t)y * 40 + x) * 3 + k];
            if (ulp_dist((h16)g[((size_t)y * 40 + x) * 4 + k], (h16)want) > 8 && fabs(g[((size_t)y * 40 + x) * 4 + k] - want) > 2e-3) bad++;
        }
    }
    CHECKC(bad == 0 && nF > 0, "os-pyramid: %d wrong components (%d filled pixels)", bad, nF);
    CHECKC(r.O.levels.count == 6, "os-pyramid: six levels for 40x24 (%lu)", (unsigned long)r.O.levels.count);
    /* trusted pixels equal the direct variant bit for bit */
    ORig d = open_case(sc);
    run(&d, 0);
    float *gd = tex_get(d.out, 4);
    int diff = 0;
    for (int i = 0; i < 40 * 24; i++) if (!F[i]) for (int k = 0; k < 4; k++) diff += g[i * 4 + k] != gd[i * 4 + k];
    CHECKC(diff == 0, "os-trusted-equal: %d trusted components differ", diff);
    free(g); free(gd); free(F); free(syn); free(fin); free_scene(&sc);

    /* os-all-hole */
    sc = scene(40, 24, 2);
    for (int i = 0; i < 40 * 24; i++) sc.hole[i] = 1;
    set_cells(&sc, ^(int i, int j, double n[8]) { n[4] = 30; });
    r = open_case(sc);
    run(&r, 1);
    g = tex_get(r.out, 4);
    int nonblack = 0;
    for (int i = 0; i < 40 * 24; i++) nonblack += g[i * 4] != 0 || g[i * 4 + 1] != 0 || g[i * 4 + 2] != 0 || g[i * 4 + 3] != 1;
    CHECKC(nonblack == 0, "os-all-hole: %d pixels are not black with alpha 1", nonblack);
    free(g); free_scene(&sc);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        simple_cases(); needfill(); pyramid_cases();
    }
    printf(fails ? "g_output: %d FAILURES\n" : "g_output: ok\n", fails);
    return fails != 0;
}
