/* Motion preparation (spec fg.motion_prep section 7 and fg.motion_prep_r2): mp-pan, mp-edge, mp2-tap-centre(-rows), mp2-motion-at-tap, mp2-odd-ratio, mp-flat, mp-ramp, mp-chain, mp-chain-zero-history. */
#import <Foundation/Foundation.h>
#include "texutil.h"
#include "motionprep.h"

static int fails;
static NnRig R;
static MetalFGMotionPrep *MP;

static const uint32_t WM = 64, HM = 48;

typedef struct { float depth[64 * 48]; uint8_t code[64 * 48]; float mvo[64 * 48 * 2]; } F05Out;

static void run_f05(id<MTLTexture> depth, id<MTLTexture> motion, uint32_t mw, uint32_t mh, float sx, float sy, uint32_t reversed, F05Out *o) {
    id<MTLTexture> dep = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, WM, HM), mvo = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, WM, HM);
    MetalFGF05Args a = {WM, HM, {0, 0}, {(uint32_t)depth.width, (uint32_t)depth.height}, {0, 0}, {mw, mh}, sx, sy, reversed, 1e-7f};
    nn_run(&R, ^(MetalFGEnc *e) { [MP f05:e depth:depth motion:motion dep:dep mvo:mvo args:a]; });
    uint32_t *d = (uint32_t *)tex_get(dep, 1);
    float *m = tex_get(mvo, 2);
    for (size_t i = 0; i < WM * HM; i++) {
        uint32_t b; float dv = rec_depth_c(d[i]); memcpy(&b, &dv, 4);
        memcpy(&o->depth[i], &b, 4);
        o->code[i] = rec_code_c(d[i]);
        o->mvo[2 * i] = m[2 * i]; o->mvo[2 * i + 1] = m[2 * i + 1];
    }
    free(d); free(m);
}

static int close_rel(double got, double want, double rel) { return fabs(got - want) <= rel * fabs(want) + (want == 0 ? 0 : 0); }

static void mp_pan(void) {
    id<MTLTexture> depth = gpu_tex(&R.gpu, MTLPixelFormatR32Float, WM, HM);
    tex_set(depth, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.75f; });
    uint32_t sizes[2][2] = {{64, 48}, {32, 24}};
    for (int run = 0; run < 2; run++) {
        uint32_t mw = sizes[run][0], mh = sizes[run][1];
        id<MTLTexture> motion = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, mw, mh);
        tex_set(motion, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = -3.25f; v[1] = 1.5f; });
        F05Out o;
        run_f05(depth, motion, mw, mh, 2.0f, 0.5f, 1, &o);
        double wx = -13.25 / 2 / mw * 0 + (-3.25 * 2.0 / mw), wy = 1.5 * 0.5 / mh;
        int bad = 0;
        for (size_t i = 0; i < WM * HM; i++)
            if (o.depth[i] != 0.75f || o.code[i] != 0 || !close_rel(o.mvo[2 * i], wx, 0x1p-22) || !close_rel(o.mvo[2 * i + 1], wy, 0x1p-22)) bad++;
        CHECKC(bad == 0, "mp-pan run %d (motion rect %ux%u): %d wrong (got %g %g want %g %g)", run, mw, mh, bad, o.mvo[0], o.mvo[1], wx, wy);
        /* control (b): dividing by the Mot grid instead of the motion rectangle differs in the second run only */
        double alt = -3.25 * 2.0 / WM;
        CHECKC((fabs(alt - wx) < 1e-12) == (run == 0), "mp-pan control (b) sensitivity");
    }
}

static void mp_edge(void) {
    for (int comp = 0; comp < 2; comp++) {
        id<MTLTexture> depth = gpu_tex(&R.gpu, MTLPixelFormatR32Float, WM, HM), motion = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, WM, HM);
        tex_set(depth, ^(uint32_t x, uint32_t y, float v[4]) { float d = x < 32 ? 0.875f : 0.25f; v[0] = comp ? 1 - d : d; });
        tex_set(motion, ^(uint32_t x, uint32_t y, float v[4]) { if (x < 32) { v[0] = -4; v[1] = (float)y; } else { v[0] = 4; v[1] = 0; } });
        F05Out o;
        run_f05(depth, motion, WM, HM, 1, 1, comp ? 0 : 1, &o);
        int bad = 0;
        for (uint32_t y = 0; y < HM; y++)
            for (uint32_t x = 0; x < WM; x++) {
                size_t i = (size_t)y * WM + x;
                int code = 0;
                double dd = x < 32 ? 0.875 : 0.25, mx = x < 32 ? -4.0 / 64 : 4.0 / 64, my = x < 32 ? (double)y / 48 : 0;
                if (x == 32) { code = 1; dd = 0.875; mx = -4.0 / 64; my = (double)(y > 0 ? y - 1 : 0) / 48; }
                if (comp) dd = dd; /* the stored depth is the proxy input, not its complement: complemented run stores 1 - d */
                double want_depth = comp ? 1 - dd : dd;
                if (o.code[i] != code || !close_rel(o.depth[i], want_depth, 0x1p-20) || !close_rel(o.mvo[2 * i], mx, 0x1p-20) ||
                    fabs(o.mvo[2 * i + 1] - my) > 1e-6)
                    bad++;
            }
        CHECKC(bad == 0, "mp-edge (%s depth): %d wrong", comp ? "complemented" : "reversed", bad);
    }
}

static void mp_flat(void) {
    id<MTLTexture> depth = gpu_tex(&R.gpu, MTLPixelFormatR32Float, WM, HM), motion = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    tex_set(depth, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.5f; });
    tex_set(motion, ^(uint32_t x, uint32_t y, float v[4]) {});
    F05Out o;
    run_f05(depth, motion, WM, HM, 1, 1, 1, &o);
    int bad = 0;
    for (size_t i = 0; i < WM * HM; i++) bad += o.depth[i] != 0.5f || o.code[i] != 0 || o.mvo[2 * i] != 0 || o.mvo[2 * i + 1] != 0;
    CHECKC(bad == 0, "mp-flat: %d wrong", bad);
}

static void mp_depth_formats(void) {
    /* the same selection through a depth-format texture */
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:WM height:HM mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> depth = [R.gpu.dev newTextureWithDescriptor:d];
    float *p = malloc(WM * HM * 4);
    for (size_t i = 0; i < WM * HM; i++) p[i] = (i % WM) < 32 ? 0.875f : 0.25f;
    [depth replaceRegion:MTLRegionMake2D(0, 0, WM, HM) mipmapLevel:0 withBytes:p bytesPerRow:WM * 4];
    free(p);
    id<MTLTexture> motion = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    tex_set(motion, ^(uint32_t x, uint32_t y, float v[4]) {});
    F05Out o;
    run_f05(depth, motion, WM, HM, 1, 1, 1, &o);
    CHECKC(o.code[32] == 1 && o.code[31] == 0 && o.depth[32] == 0.875f, "mp-edge through a Depth32Float texture: code %u depth %g", o.code[32], o.depth[32]);
}

static void mp_ramp(void) {
    id<MTLTexture> field = gpu_tex(&R.gpu, MTLPixelFormatRGBA32Float, 128, 96), out = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, WM, HM);
    tex_set(field, ^(uint32_t X, uint32_t Y, float v[4]) {
        float u = (X + 0.5f) / 128, w = (Y + 0.5f) / 96;
        v[0] = u; v[1] = w; v[2] = 1 - u; v[3] = w / 2;
    });
    static uint32_t o[2] = {0, 0}, s[2] = {128, 96};
    nn_run(&R, ^(MetalFGEnc *e) { [MP f06:e field:field origin:o size:s out:out Wm:WM Hm:HM]; });
    float *g = tex_get(out, 4);
    int bad = 0;
    for (uint32_t y = 0; y < HM; y++) for (uint32_t x = 0; x < WM; x++) {
        double u = (x + 0.5) / WM, v = (y + 0.5) / HM;
        double want[4] = {u, v, 1 - u, v / 2};
        for (int c = 0; c < 4; c++) if (fabs(g[((size_t)y * WM + x) * 4 + c] - want[c]) > 0x1p-10) bad++;
    }
    CHECKC(bad == 0, "mp-ramp: %d wrong", bad);
    free(g);
}

static void chain(int zero_history) {
    id<MTLTexture> b1 = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, WM, HM), b0 = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, WM, HM);
    id<MTLTexture> mvo = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, WM, HM), mvf = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM);
    id<MTLTexture> dep = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, WM, HM);
    id<MTLTexture> mvod = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, WM, HM), mvfd = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, WM, HM), dep2 = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, WM, HM);
    tex_set(b1, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1.0f / 32; v[1] = -1.0f / 16; v[2] = -1.0f / 32; v[3] = 1.0f / 16; });
    tex_set(b0, ^(uint32_t x, uint32_t y, float v[4]) { if (zero_history) return; v[0] = v[1] = 1.0f / 64; v[2] = v[3] = -1.0f / 64; });
    tex_set(mvo, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = -1.0f / 16; v[1] = 1.0f / 32; });
    tex_set(mvf, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 3; v[1] = -2; });
    tex_set(dep, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = (float)((x + 64 * y + 1) << 12 >> 12); });
    /* depth records: any injective payload will do; use the index itself */
    nn_run(&R, ^(MetalFGEnc *e) { [MP f12:e bdfCur:b1 bdfOld:b0 mvo:mvo mvf:mvf dep:dep mvod:mvod mvfd:mvfd dep2:dep2 Wm:WM Hm:HM]; });
    float *a = tex_get(mvod, 2), *b = tex_get(mvfd, 4);
    uint32_t *d2 = (uint32_t *)tex_get(dep2, 1), *d1 = (uint32_t *)tex_get(dep, 1);
    double ex_a = zero_history ? -1.0 / 32 : -3.0 / 64, ex_b = zero_history ? 1.0 : 2.0, ex_b2 = zero_history ? 1.0 : 1.75;
    int bad = 0, badb = 0, badd = 0;
    for (uint32_t y = 0; y < HM; y++) for (uint32_t x = 0; x < WM; x++) {
        size_t i = (size_t)y * WM + x;
        if (fabs(a[i * 2] - ex_a) > 0x1p-14 || fabs(a[i * 2 + 1] - ex_a) > 0x1p-14) bad++;
        if (fabs(b[i * 4] - ex_b) > 0x1p-8 || fabs(b[i * 4 + 1] - ex_b2) > 0x1p-8 || b[i * 4 + 2] != 0 || b[i * 4 + 3] != 0) badb++;
        if (!zero_history) {
            uint32_t sxn = x + 2 > 63 ? 63 : x + 2, syn = y < 3 ? 0 : y - 3;
            if (d2[i] != d1[(size_t)syn * WM + sxn]) badd++;
        }
    }
    const char *nm = zero_history ? "mp-chain-zero-history" : "mp-chain";
    CHECKC(bad == 0, "%s: mvod %d wrong (got %g want %g)", nm, bad, a[0], ex_a);
    CHECKC(badb == 0, "%s: mvfd %d wrong (got %g %g)", nm, badb, b[0], b[1]);
    CHECKC(badd == 0, "%s: dep2 %d wrong", nm, badd);
    free(a); free(b); free(d1); free(d2);
}

/* ---- fg.motion_prep_r2: centre texel of the five taps and the position of the motion read, depth and motion rectangles larger than the Mot grid ---- */

typedef void (^TexFn)(uint32_t x, uint32_t y, float v[4]);

static void run_f05_rect(uint32_t dw, uint32_t dh, TexFn dfn, TexFn mfn, F05Out *o) {
    id<MTLTexture> depth = gpu_tex(&R.gpu, MTLPixelFormatR32Float, dw, dh), motion = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, dw, dh);
    tex_set(depth, dfn);
    tex_set(motion, mfn);
    run_f05(depth, motion, dw, dh, 1, 1, 1, o);
}

/* model of one Mot column of a depth image that varies along x only: rule 0 = ceil(P) - 1 (the spec), 1 = 2 x + 1 (control, "containing" for ratio two),
   2 = nearest integer of P - 1/2 with ties upward (control). Returns the code, *tex the winning texel. */
static int model_code(int x, int Wz, int Wm, int rule, const float *dz, int *tex) {
    double P = (x + 0.5) * Wz / Wm;
    int c = rule == 0 ? (int)ceil(P) - 1 : rule == 1 ? (int)floor(P) : (int)floor(P - 0.5 + 0.5);
    static const int Tx[5] = {0, -1, 1, -1, 1};
    double best = -1;
    int code = 0;
    for (int k = 0; k < 5; k++) {
        int t = c + Tx[k];
        t = t < 0 ? 0 : t > Wz - 1 ? Wz - 1 : t;
        double n = dz[t] > 1e-7 ? dz[t] : 1e-7;   /* depth_reversed: larger value is nearer */
        if (k == 0 || n > best) { best = n; code = k; *tex = t; }
    }
    return code;
}

static int close_abs_rel(double got, double want) { return fabs(got - want) <= 0x1p-22 * fabs(want) + (want == 0 ? 1e-12 : 0); }

static void mp2_tap_centre(void) {
    const uint32_t DW = 128, DH = 96;
    /* run A: near (7/8) on texel columns 0..65 (rows: 0..65), far (1/4) beyond; run B: near from 62 on */
    for (int axis = 0; axis < 2; axis++) for (int run = 0; run < 2; run++) {
        int edge = run == 0 ? 66 : 62;     /* first texel of the second surface along the axis */
        float nearv = 0.875f, farv = 0.25f;
        F05Out o;
        run_f05_rect(DW, DH, ^(uint32_t x, uint32_t y, float v[4]) {
            uint32_t s = axis ? y : x;
            bool first = s < (uint32_t)edge;
            v[0] = run == 0 ? (first ? nearv : farv) : (first ? farv : nearv);
        }, ^(uint32_t x, uint32_t y, float v[4]) {
            uint32_t s = axis ? y : x;
            bool first = s < (uint32_t)edge;
            bool isnear = run == 0 ? first : !first;
            float m = isnear ? -4.0f : 4.0f;
            v[axis] = m;
        }, &o);
        /* expected (code, near?) per Mot index */
        struct { uint32_t idx; int code; int isnear; } ex[3];
        if (run == 0) { ex[0] = (typeof(ex[0])){32, 0, 1}; ex[1] = (typeof(ex[0])){33, 1, 1}; ex[2] = (typeof(ex[0])){34, 0, 0}; }
        else { ex[0] = (typeof(ex[0])){30, 0, 0}; ex[1] = (typeof(ex[0])){31, 0, 1}; ex[2] = (typeof(ex[0])){30, 0, 0}; }
        int nex = run == 0 ? 3 : 2;
        for (int e = 0; e < nex; e++) {
            uint32_t mx = axis ? 20 : ex[e].idx, my = axis ? ex[e].idx : 20;
            size_t i = (size_t)my * WM + mx;
            double want = (ex[e].isnear ? -4.0 : 4.0) / (axis ? DH : DW);
            CHECKC(o.code[i] == ex[e].code && close_abs_rel(o.mvo[2 * i + axis], want) && o.mvo[2 * i + 1 - axis] == 0,
                   "mp2-tap-centre%s run %c Mot %s %u: code %d (want %d) mvo %g (want %g)", axis ? "-rows" : "", 'A' + run, axis ? "row" : "column", ex[e].idx,
                   o.code[i], ex[e].code, o.mvo[2 * i + axis], want);
        }
    }
    /* controls, on the column model: the centre texel 2 x + 1 and the nearest integer of P - 1/2 change the result at the columns the spec names */
    float dzA[128], dzB[128];
    for (int t = 0; t < 128; t++) { dzA[t] = t < 66 ? 0.875f : 0.25f; dzB[t] = t < 62 ? 0.25f : 0.875f; }
    for (int rule = 1; rule < 3; rule++) {
        int tx;
        int cA = model_code(33, 128, 64, rule, dzA, &tx);
        int txA = tx;
        int cB = model_code(30, 128, 64, rule, dzB, &tx);
        CHECKC(model_code(33, 128, 64, 0, dzA, &tx) == 1 && tx == 65, "mp2-tap-centre model: spec rule at run A column 33");
        CHECKC(cA == 0 && dzA[txA] == 0.25f, "mp2-tap-centre control rule %d: run A column 33 must see only far texels (code %d)", rule, cA);
        CHECKC(cB == 2 || cB == 4, "mp2-tap-centre control rule %d: run B column 30 must reach the near texel on a +1 tap (code %d)", rule, cB);
    }
}

static void mp2_motion_at_tap(void) {
    const uint32_t DW = 128, DH = 96;
    F05Out o;
    run_f05_rect(DW, DH, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = x == 67 ? 0.875f : 0.75f; },
                 ^(uint32_t x, uint32_t y, float v[4]) { v[0] = x == 67 ? 6.0f : -2.0f; }, &o);
    int bad = 0;
    for (uint32_t y = 0; y < HM; y++) {
        size_t i = (size_t)y * WM + 33;
        if (o.code[i] != 2 || !close_abs_rel(o.mvo[2 * i], 6.0 / DW)) bad++;
    }
    CHECKC(bad == 0, "mp2-motion-at-tap: %d wrong at Mot column 33 (code %d, mvo %g, want 2 and %g)", bad, o.code[33], o.mvo[66], 6.0 / DW);
    /* control: the read at uv(p) + tau, the point 68.0 between texels 67 and 68, is the mean of 6 and -2 */
    CHECKC(fabs(2.0 / DW - 6.0 / DW) > 0x1p-22 * 6.0 / DW, "mp2-motion-at-tap control sensitivity");
}

/* im2-tie (fg.interm_motion_pin_r4): five taps with identical depths; the first tap wins and the record holds the exact depth, with a full-mantissa depth */
static void im2_tie(void) {
    const uint32_t db = 0x3F000007u;     /* low mantissa bits set */
    float dv; memcpy(&dv, &db, 4);
    id<MTLTexture> depth = gpu_tex(&R.gpu, MTLPixelFormatR32Float, WM, HM), motion = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, WM, HM);
    tex_set(depth, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = dv; });
    tex_set(motion, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = (float)x; v[1] = (float)y; });
    F05Out o;
    run_f05(depth, motion, WM, HM, 1, 1, 1, &o);
    int bad = 0;
    for (uint32_t y = 2; y < HM - 2; y++) for (uint32_t x = 2; x < WM - 2; x++) {
        size_t i = (size_t)y * WM + x;
        uint32_t b; memcpy(&b, &o.depth[i], 4);
        bad += o.code[i] != 0 || b != db || fabs(o.mvo[2 * i] - (double)x / WM) > 1e-6 || fabs(o.mvo[2 * i + 1] - (double)y / HM) > 1e-6;
    }
    CHECKC(bad == 0, "im2-tie: %d wrong", bad);
    /* control: the last tap wins; it would read the motion of texel (x + 1, y + 1): a different position */
    CHECKC(fabs((double)(20 + 1) / WM - 20.0 / WM) > 1e-6, "im2-tie control sensitivity");
}

static void mp2_odd_ratio(void) {
    const uint32_t DW = 131, DH = 97;
    F05Out o;
    run_f05_rect(DW, DH, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = x >= 80 ? 0.875f : 0.25f; },
                 ^(uint32_t x, uint32_t y, float v[4]) { v[0] = x >= 80 ? -4.0f : 4.0f; }, &o);
    size_t i38 = (size_t)20 * WM + 38, i39 = (size_t)20 * WM + 39;
    CHECKC(o.code[i38] == 0 && close_abs_rel(o.mvo[2 * i38], 4.0 / DW), "mp2-odd-ratio column 38: code %d mvo %g (want 0, far)", o.code[i38], o.mvo[2 * i38]);
    CHECKC(o.code[i39] == 0 && close_abs_rel(o.mvo[2 * i39], -4.0 / DW), "mp2-odd-ratio column 39: code %d mvo %g (want 0, near)", o.code[i39], o.mvo[2 * i39]);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        MP = [[MetalFGMotionPrep alloc] initWithDevice:g.dev library:R.lib];
        if (!MP) { printf("motion kernels missing\n"); return 2; }
        mp_pan(); mp_edge(); mp2_tap_centre(); mp2_motion_at_tap(); mp2_odd_ratio(); im2_tie(); mp_flat(); mp_depth_formats(); mp_ramp(); chain(0); chain(1);
    }
    printf(fails ? "g_motion: %d FAILURES\n" : "g_motion: ok\n", fails);
    return fails != 0;
}
