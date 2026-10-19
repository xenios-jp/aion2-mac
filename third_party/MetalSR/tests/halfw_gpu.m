// Tests of the half-precision filter weights (sr.metal_techniques_r3 section 7, items 2 and 3): filter-bound and filter-sharp.
// Only meaningful on a library built with -DSR_TESTOUT (the filter then writes the filtered colour C'_g, before the blend, to its output texture)
// and with SR_HW = 1 (V1); build.sh gputest builds that library and the controls: SR_HW = 2 (no scaling by a_max, must fail filter-sharp) and
// SR_HW = 3 (weights with 3 mantissa bits, must fail filter-bound). The status is the number of failures.
//
// P12 runs alone on 256 x 128 output pixels, display-referred colour, pre-exposure 1, jitter (1/4, -1/4); path N at render size 170 x 85, path W at 85 x 42.
// Colour: independent uniform values in [0, 1) per component (alpha 1); warped image: uniform in [0, 1). The head output comes from the real head network
// (the unfused test entry) on d1 and s0 filled with independent uniform activation codes. The reference is the float64 filter of tests/ref_p12.h with the
// exact weights. The bound per pixel and component: |C'_g - C_g| <= E / (1 - E) times the spread of the tap values + one binary16 unit in the last place of C_g,
// with E = 2^-10.
#import "testutil.h"
#include "ref_p12.h"
#include "testparams.h"
#import "passes.h"
#include "../src/frame.h"

extern unsigned char metalsr_metallib[];
extern unsigned int metalsr_metallib_len;
static SrParams g_mp;
enum { WO = 256, HO = 128 };

static SrWeights upload(const uint8_t *img) {
    SrWeights w; w.buf = bufcopy(img, SR_WEIGHT_FILE_BYTES);
    sr_weights_prepare(g_gpu, &w);
    return w;
}

static double ulp16(double v) {
    v = fabs(v);
    int e = v < 6.103515625e-05 ? -14 : (int)floor(log2(v));
    return ldexp(1.0, e - 10);
}

typedef struct { long checked, over_bound, nonfinite, below14, below25; double worst; } Stats;

// one run: path (wide), render size, sigma_min / sigma_span (< 0: the values of the parameter file)
static Stats run_case(int wide, int Wr, int Hr, double sigma_min, double sigma_span) {
    Stats st; memset(&st, 0, sizeof st);
    SrLevels L; sr_levels(WO, HO, &L); L.wide = wide;
    const size_t n0 = (size_t)L.W[0] * L.H[0], n1 = (size_t)L.W[1] * L.H[1];
    g_rng = 0x1234567ull;                                   // the lane's seeded generators (seed 1 for the colour, 2 for the activations: two fixed streams)
    float *col = calloc((size_t)Wr * Hr * 4, 4);
    f16 *R = calloc((size_t)WO * HO * 4, 2), *o = calloc(n0 * 40, 2);
    for (size_t i = 0; i < (size_t)Wr * Hr; i++) { for (int k = 0; k < 3; k++) col[4 * i + k] = (float)urand(); col[4 * i + 3] = 1.0f; }
    for (size_t i = 0; i < (size_t)WO * HO; i++) { for (int k = 0; k < 3; k++) R[4 * i + k] = (f16)urand(); }
    g_rng = 0x2345678ull;
    uint8_t *img = make_weights(WMODE_RANDOM, 1.0), *d1 = malloc(n1 * 32), *s0 = malloc(n0 * 32);
    // finite activation codes only: the codes 0xF8 to 0xFF decode to infinity and NaN, which would make the head output non-finite on nearly every pixel
    // (QUESTIONS.md 31); the codes below 0x80 keep the head output varying from pixel to pixel
    for (size_t i = 0; i < n1 * 32; i++) d1[i] = (uint8_t)(urand() * 0x80);
    for (size_t i = 0; i < n0 * 32; i++) s0[i] = (uint8_t)(urand() * 0x80);
    SrCall call = {(u32)WO, (u32)HO, (u32)Wr, (u32)Hr, 0.25f, -0.25f, 1, 1, 1.0f, 0, 0, 1, 0, (u32)wide, 0, 0, (u32)L.W[0]};
    SrParams mp = g_mp;
    FiltParams fp; sr_make_filt(&call, &mp, &fp);
    if (sigma_min >= 0) { fp.sigma_min = (float)sigma_min; fp.sigma_span = (float)sigma_span; }
    SrWeights w = upload(img);
    id<MTLBuffer> bd = bufcopy(d1, n1 * 32), bs = bufcopy(s0, n0 * 32), bo = [g_gpu buffer:n0 * 80];
    run1(^(SrEnc *e) { sr_head(e, g_gpu, &w, &L, wide, bd, bs, bo); });
    memcpy(o, bo.contents, n0 * 80);
    id<MTLTexture> tcol = [g_gpu texture:MTLPixelFormatRGBA32Float w:Wr h:Hr], tR = [g_gpu texture:MTLPixelFormatRGBA16Float w:WO h:HO];
    [tcol replaceRegion:MTLRegionMake2D(0, 0, Wr, Hr) mipmapLevel:0 withBytes:col bytesPerRow:(size_t)Wr * 16];
    [tR replaceRegion:MTLRegionMake2D(0, 0, WO, HO) mipmapLevel:0 withBytes:R bytesPerRow:(size_t)WO * 8];
    f16 en = (f16)1.0f;
    id<MTLTexture> te = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
    [te replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&en bytesPerRow:2];
    id<MTLTexture> ot = [g_gpu texture:MTLPixelFormatRGBA16Float w:WO h:HO], ht = [g_gpu texture:MTLPixelFormatRGBA16Float w:WO h:HO];
    SrFiltIo io = {tcol, tR, te, ot, ht, bo};
    run1(^(SrEnc *e) { sr_filter(e, g_gpu, &fp, &io, (unsigned)L.W[0]); });
    f16 *go = malloc((size_t)WO * HO * 8);
    [ot getBytes:go bytesPerRow:(size_t)WO * 8 fromRegion:MTLRegionMake2D(0, 0, WO, HO) mipmapLevel:0];

    const double E = ldexp(1.0, -10);
    RefP12Opts opts; memset(&opts, 0, sizeof opts);
    const int Rr = wide ? 2 : 1;
    float k_c = fp.invp * 1.0f; if (k_c == 0) k_c = 1.0f;
    for (int y = 0; y < HO; y++) for (int x = 0; x < WO; x++) {
        int X = x >> 2, Y = y >> 2, t = ((x >> 1) & 1) + 2 * ((y >> 1) & 1);
        f16 v[10]; ref_quad_v(o + ((size_t)Y * L.W[0] + X) * 40, t, v);
        RefQuad q = ref_quad(v, &fp, &opts);
        if (!q.colour_ok) continue;
        float u = wide ? coord_formB(fp.sx, fp.jx, x) : coord_formA(fp.sx, fp.jx, x);
        float vv = wide ? coord_formB(fp.sy, fp.jy, y) : coord_formA(fp.sy, fp.jy, y);
        int nx = (int)roundf(u), ny = (int)roundf(vv);
        float dxc = (float)nx - u, dyc = (float)ny - vv;
        double acc[3] = {0, 0, 0}, Ws = 0, lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}, wmax = 0;
        for (int j = -Rr; j <= Rr; j++) for (int i = -Rr; i <= Rr; i++) {
            int tx = nx + i, ty = ny + j;
            tx = tx < 0 ? 0 : tx >= Wr ? Wr - 1 : tx; ty = ty < 0 ? 0 : ty >= Hr ? Hr - 1 : ty;
            f16 c[4]; r12_tap(col + ((size_t)ty * Wr + tx) * 4, k_c, &fp, c);
            double dx = (double)dxc + i, dy = (double)dyc + j;
            double wgt = exp(q.k_xx * dx * dx + q.k_yy * dy * dy + q.k_xy * dx * dy);
            if (wgt > wmax) wmax = wgt;
            for (int k = 0; k < 3; k++) { acc[k] += wgt * (double)c[k]; if ((double)c[k] < lo[k]) lo[k] = (double)c[k]; if ((double)c[k] > hi[k]) hi[k] = (double)c[k]; }
            Ws += wgt;
        }
        if (wmax < ldexp(1.0, -14)) st.below14++;
        if (wmax < ldexp(1.0, -25)) st.below25++;
        st.checked++;
        const f16 *g = go + ((size_t)y * WO + x) * 4;
        int bad = 0;
        for (int k = 0; k < 3; k++) {
            double cg = acc[k] / Ws, gv = (double)g[k];
            if (!isfinite((float)g[k])) { st.nonfinite++; bad = 1; continue; }
            double bound = E / (1 - E) * (hi[k] - lo[k]) + ulp16(cg), err = fabs(gv - cg);
            if (err > bound) { if (!bad && st.over_bound == 0) printf("  first failure at (%d, %d) component %d: GPU %g, model %g, bound %g, spread %g\n", x, y, k, gv, cg, bound, hi[k] - lo[k]); bad = 1; }
            if (err / bound > st.worst) st.worst = err / bound;
        }
        st.over_bound += bad;
    }
    free(col); free(R); free(o); free(img); free(d1); free(s0); free(go);
    return st;
}

int main(void) {
    @autoreleasepool {
        {   // the library embedded in this binary (the variant under test), not build/metalsr.metallib
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            NSData *lib = [NSData dataWithBytes:metalsr_metallib length:metalsr_metallib_len];
            NSString *err = nil;
            g_gpu = [[SrGpu alloc] initWithDevice:dev libraryData:lib error:&err];
            if (!g_gpu) { printf("gpu init: %s\n", err.UTF8String); return 2; }
        }
        g_mp = test_params_r3();
        (void)[g_gpu tanhTable];
        [g_gpu commitResidency];
        for (int wide = 0; wide < 2; wide++) {
            const int Wr = wide ? 85 : 170, Hr = wide ? 42 : 85;
            Stats b = run_case(wide, Wr, Hr, -1, -1);
            printf("filter-bound path %c: %ld pixels, %ld over the bound, %ld non-finite components, worst error / bound %.3f\n", "NW"[wide], b.checked, b.over_bound, b.nonfinite, b.worst);
            EXPECT(b.checked > 20000, "filter-bound path %c: too few judged pixels (%ld)", "NW"[wide], b.checked);
            EXPECT(b.over_bound == 0, "filter-bound path %c: %ld pixels exceed the bound", "NW"[wide], b.over_bound);
            Stats s = run_case(wide, Wr, Hr, 1.0 / 16, 1.0 / 16);
            printf("filter-sharp path %c: %ld pixels, largest unscaled weight below 2^-14 on %ld, below 2^-25 on %ld; %ld over the bound, %ld non-finite components\n",
                   "NW"[wide], s.checked, s.below14, s.below25, s.over_bound, s.nonfinite);
            EXPECT(s.below14 >= 256 && s.below25 >= 64, "filter-sharp path %c is misconfigured: counts %ld (< 256) and %ld (< 64)", "NW"[wide], s.below14, s.below25);
            EXPECT(s.over_bound == 0 && s.nonfinite == 0, "filter-sharp path %c: %ld pixels exceed the bound, %ld non-finite", "NW"[wide], s.over_bound, s.nonfinite);
        }
        printf(g_fails ? "halfw_gpu: %d FAILURES\n" : "halfw_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
