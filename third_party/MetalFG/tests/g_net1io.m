/* F15 / F17 (spec fg.net1_io section 7): n1io-pack-index, -warp-error, -mot-unit, -ns-point (r3), -motion-point, -equal-grids, -unpack-identity,
   -mask-steer, -nan, -reset-state, driven pass by pass with injected images. */
#import <Foundation/Foundation.h>
#include "nnutil.h"
#include "net1io.h"

static int fails;

/* generic image helpers: set every texel from a callback, read texels back as floats */
static void tex_set(id<MTLTexture> t, void (^f)(uint32_t x, uint32_t y, float v[4])) {
    uint32_t W = (uint32_t)t.width, H = (uint32_t)t.height;
    MTLPixelFormat pf = t.pixelFormat;
    uint32_t bpp = pf == MTLPixelFormatRGBA16Float ? 8 : pf == MTLPixelFormatRG32Float ? 8 : pf == MTLPixelFormatRG16Float ? 4 : pf == MTLPixelFormatR16Float ? 2 : 1;
    uint8_t *buf = calloc((size_t)W * H, bpp);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            float v[4] = {0, 0, 0, 0};
            f(x, y, v);
            uint8_t *p = buf + ((size_t)y * W + x) * bpp;
            switch (pf) {
            case MTLPixelFormatRGBA16Float: for (int c = 0; c < 4; c++) ((h16 *)p)[c] = (h16)v[c]; break;
            case MTLPixelFormatRG16Float: for (int c = 0; c < 2; c++) ((h16 *)p)[c] = (h16)v[c]; break;
            case MTLPixelFormatR16Float: ((h16 *)p)[0] = (h16)v[0]; break;
            case MTLPixelFormatRG32Float: for (int c = 0; c < 2; c++) ((float *)p)[c] = v[c]; break;
            default: { float q = fminf(fmaxf(v[0], 0), 1); p[0] = (uint8_t)lrintf(q * 255); break; }
            }
        }
    gpu_upload(t, buf, bpp);
    free(buf);
}

static float *tex_get(id<MTLTexture> t, int comps) {
    uint32_t W = (uint32_t)t.width, H = (uint32_t)t.height;
    MTLPixelFormat pf = t.pixelFormat;
    uint32_t bpp = pf == MTLPixelFormatRGBA16Float ? 8 : pf == MTLPixelFormatRG16Float ? 4 : pf == MTLPixelFormatR16Float ? 2 : 1;
    uint8_t *buf = malloc((size_t)W * H * bpp);
    gpu_download(t, buf, bpp);
    float *o = calloc((size_t)W * H * comps, sizeof(float));
    for (size_t i = 0; i < (size_t)W * H; i++)
        for (int c = 0; c < comps; c++) {
            const uint8_t *p = buf + i * bpp;
            o[i * comps + c] = pf == MTLPixelFormatR8Unorm ? p[0] / 255.0f : (float)((const h16 *)p)[c];
        }
    free(buf);
    return o;
}

typedef struct {
    NnRig rig;
    MetalFGIO *io;
    uint32_t Wf, Hf, w1, h1, ox, oy, Wm, Hm;
    id<MTLTexture> colOld, colCur, dY, errOld, ns, dis, disOld, st, mcp, mpc, errOut, msOut;
    id<MTLBuffer> n1in;
} Io;

static Io io_open(Gpu g, uint32_t Wf, uint32_t Hf, uint32_t Wm, uint32_t Hm) {
    Io s;
    memset((void *)&s, 0, sizeof s);
    uint16_t *t[METALFG_TENSOR_COUNT];
    wgen_tensors(WGEN_ZERO, 1, t);
    if (nn_open(&s.rig, g, t)) exit(2);
    s.io = [[MetalFGIO alloc] initWithDevice:g.dev library:s.rig.lib];
    MetalFGParams P = wgen_params();
    MetalFGGrids gr;
    // reuse the planner's embedding rule for the N1 grid of this Feat size
    uint32_t cw = (Wf + 1) / 2, ch = (Hf + 1) / 2;
    gr.w1 = (cw + 31) / 32 * 32; gr.h1 = (ch + 31) / 32 * 32;
    (void)P;
    s.Wf = Wf; s.Hf = Hf; s.Wm = Wm; s.Hm = Hm; s.w1 = gr.w1; s.h1 = gr.h1;
    s.ox = 2 * ((s.w1 - cw) / 2); s.oy = 2 * ((s.h1 - ch) / 2);
    s.colOld = gpu_tex(&g, MTLPixelFormatRGBA16Float, Wf, Hf);
    s.colCur = gpu_tex(&g, MTLPixelFormatRGBA16Float, Wf, Hf);
    s.dY = gpu_tex(&g, MTLPixelFormatR16Float, Wf, Hf);
    s.errOld = gpu_tex(&g, MTLPixelFormatRG16Float, Wf, Hf);
    s.ns = gpu_tex(&g, MTLPixelFormatR8Unorm, Wm, Hm);
    s.dis = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf);
    s.disOld = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf);
    s.st = gpu_tex(&g, MTLPixelFormatR16Float, Wf, Hf);
    s.mcp = gpu_tex(&g, MTLPixelFormatRG32Float, Wm, Hm);
    s.mpc = gpu_tex(&g, MTLPixelFormatRG16Float, Wm, Hm);
    s.errOut = gpu_tex(&g, MTLPixelFormatRG16Float, Wf, Hf);
    s.msOut = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf);
    s.n1in = nn_buf(&s.rig, NULL, (size_t)s.w1 * s.h1 * 48);
    return s;
}

static void io_f15(Io *s) {
    MetalFGF15Args a = {s->colOld, s->colCur, s->dY, s->errOld, s->ns, s->dis, s->disOld, s->st, s->mcp, s->mpc, s->errOut, s->msOut, s->n1in,
                    s->Wf, s->Hf, s->w1, s->h1, s->ox, s->oy, s->Wm, s->Hm};
    nn_run(&s->rig, ^(MetalFGEnc *e) { [s->io f15:e args:a]; });
}

static void zero_all(Io *s) {
    id<MTLTexture> all[] = {s->colOld, s->colCur, s->dY, s->errOld, s->ns, s->dis, s->disOld, s->st, s->mcp, s->mpc};
    for (int i = 0; i < 10; i++) tex_set(all[i], ^(uint32_t x, uint32_t y, float v[4]) {});
}

static void pack_index(Gpu g) {
    Io s = io_open(g, 100, 68, 100, 68);
    CHECKC(s.w1 == 64 && s.h1 == 64 && s.ox == 14 && s.oy == 30, "geometry %u %u %u %u", s.w1, s.h1, s.ox, s.oy);
    zero_all(&s);
    void (^grey)(uint32_t, uint32_t, float *) = ^(uint32_t x, uint32_t y, float v[4]) {
        float q = (float)((3 * x + 7 * y) % 16) / 16;
        v[0] = v[1] = v[2] = q; v[3] = 1;
    };
    tex_set(s.colOld, grey); tex_set(s.colCur, grey);
    tex_set(s.dY, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.25f; });
    tex_set(s.errOld, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.5f; });
    tex_set(s.st, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 0.25f; });
    io_f15(&s);
    const h16 *o = s.n1in.contents;
    int bad = 0, nonzero_margin = 0;
    for (uint32_t Y = 0; Y < 2 * s.h1; Y++)
        for (uint32_t X = 0; X < 2 * s.w1; X++) {
            int px = (int)X - (int)s.ox, py = (int)Y - (int)s.oy;
            int inside = px >= 0 && py >= 0 && px < (int)s.Wf && py < (int)s.Hf;
            double g = inside ? (double)((3 * px + 7 * py) % 16) / 16 : 0;
            double feat[10] = {0.25, g, 0 /* Ep */, 0.5, 0, g, 0 /* Ec */, 0, 0.25, 0.75};
            if (!inside) for (int f = 0; f < 10; f++) feat[f] = 0;
            const h16 *pix = o + ((size_t)(Y / 2) * s.w1 + X / 2) * 48;
            int slot = (X & 1) + 2 * (Y & 1);
            for (int f = 0; f < 10; f++) if ((double)pix[4 * f + slot] != feat[f]) bad++;
            if (!inside) for (int f = 0; f < 10; f++) nonzero_margin += (double)pix[4 * f + slot] != 0;
            if (slot == 0) for (int c = 40; c < 48; c++) bad += (double)pix[c] != 0;
        }
    CHECKC(bad == 0 && nonzero_margin == 0, "n1io-pack-index: %d wrong, %d margin non-zero", bad, nonzero_margin);
    /* feature 8 is r, 9 is 1 - r (the reset-state part two with a quarter) */
    float *err = tex_get(s.errOut, 2);
    CHECKC(err[0] == 0 && err[1] == 0, "equal images and zero motion: zero warp error");
    free(err);
}

static void warp_error(Gpu g) {
    Io s = io_open(g, 64, 64, 64, 64);
    CHECKC(s.w1 == 32 && s.h1 == 32 && s.ox == 0 && s.oy == 0, "64x64 has no margin");
    zero_all(&s);
    tex_set(s.colOld, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = (float)x / 64; v[3] = 1; });
    tex_set(s.colCur, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = x < 63 ? (float)(x + 1) / 64 : 0.5f; v[3] = 1; });
    tex_set(s.mcp, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1.0f / 64; });
    tex_set(s.mpc, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = -1; });
    io_f15(&s);
    float *e = tex_get(s.errOut, 2);
    int bad = 0;
    for (uint32_t y = 0; y < 64; y++) for (uint32_t x = 0; x < 64; x++) {
        double ec = x < 63 ? 0 : 31.0 / 64, ep = x >= 1 ? 0 : 1.0 / 64;
        if (fabs(e[(y * 64 + x) * 2] - ec) > 2e-3 || fabs(e[(y * 64 + x) * 2 + 1] - ep) > 2e-3) bad++;
    }
    CHECKC(bad == 0, "n1io-warp-error: %d wrong", bad);
    free(e);
}

static void mot_unit(Gpu g) {
    Io s = io_open(g, 64, 48, 128, 96); /* Feat is half of Mot per axis */
    zero_all(&s);
    void (^grey)(uint32_t, uint32_t, float *) = ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = (float)(x % 16) / 16; v[3] = 1; };
    tex_set(s.colOld, grey); tex_set(s.colCur, grey);
    tex_set(s.mpc, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = -2; });
    io_f15(&s);
    float *e = tex_get(s.errOut, 2);
    int bad = 0, n = 0;
    for (uint32_t y = 0; y < 48; y++) for (uint32_t x = 0; x < 64; x++) {
        if (x % 16 < 3) continue;
        double want = fabs((double)((x - 1) % 16) / 16 - (double)(x % 16) / 16);
        n++;
        if (fabs(e[(y * 64 + x) * 2 + 1] - want) > 1.0 / 1024) bad++;
    }
    CHECKC(bad == 0 && n > 0, "n1io-mot-unit: %d of %d wrong", bad, n);
    free(e);
}

/* feature 4 (NS) and the mask M of every Feat pixel against ns(texel) given by the callback; returns the number of wrong entries */
static int check_ns(Io *s, double (^want)(uint32_t x, uint32_t y)) {
    const h16 *o = s->n1in.contents;
    float *m = tex_get(s->msOut, 1);
    int bad = 0;
    for (uint32_t y = 0; y < s->Hf; y++) for (uint32_t x = 0; x < s->Wf; x++) {
        uint32_t X = x + s->ox, Y = y + s->oy;
        double f4 = (double)o[((size_t)(Y / 2) * s->w1 + X / 2) * 48 + 16 + (X & 1) + 2 * (Y & 1)];
        double w = want(x, y);
        if (fabs(f4 - w) > 0.01 || fabs(m[y * s->Wf + x] - w) > 1.0 / 255 + 1e-6) bad++;
    }
    free(m);
    return bad;
}

/* fg.net1_io_r3: n1io3-ns-point, n1io3-motion-point, n1io3-equal-grids */
static void ns_point(Gpu g) {
    Io s = io_open(g, 32, 24, 64, 48);
    zero_all(&s);
    tex_set(s.dis, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1; });
    tex_set(s.disOld, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1; });
    /* run one: one on Mot row 0 only; run two: one on Mot column 0 only; run three: one on odd Mot columns (texel 2 x + 1 is odd for every x) */
    for (int run = 0; run < 3; run++) {
        tex_set(s.ns, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = run == 0 ? (y == 0) : run == 1 ? (x == 0) : (x & 1); });
        io_f15(&s);
        int bad = check_ns(&s, ^double(uint32_t x, uint32_t y) { return run == 2 ? 1.0 : 0.0; });
        CHECKC(bad == 0, "n1io3-ns-point run %d: %d wrong (the bilinear read gives one half along the first row or column)", run + 1, bad);
    }
}

static void motion_point(Gpu g) {
    Io s = io_open(g, 32, 24, 64, 48);
    zero_all(&s);
    void (^ramp)(uint32_t, uint32_t, float *) = ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = (float)x / 32; v[3] = 1; };
    tex_set(s.colOld, ramp); tex_set(s.colCur, ramp);
    tex_set(s.mcp, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = (x & 1) ? -1.0f / 64 : 1.0f / 64; });
    tex_set(s.mpc, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = (x & 1) ? -3.0f : 3.0f; });
    io_f15(&s);
    float *e = tex_get(s.errOut, 2);
    int bad = 0;
    /* odd texel: the previous colour is read half a Feat pixel to the left, the current one one and a half pixels to the left; ramp slope 1/32 per pixel,
       bilinear reads of a ramp are exact away from the clamped edge. The averaged (zero) displacement gives zero errors. */
    for (uint32_t y = 0; y < 24; y++) for (uint32_t x = 2; x < 32; x++) {
        double ec = 0.5 / 32, ep = 1.5 / 32;
        if (fabs(e[(y * 32 + x) * 2] - ec) > 2e-3 || fabs(e[(y * 32 + x) * 2 + 1] - ep) > 2e-3) bad++;
    }
    CHECKC(bad == 0, "n1io3-motion-point: %d wrong (got %g %g want %g %g)", bad, e[(5 * 32 + 10) * 2], e[(5 * 32 + 10) * 2 + 1], 0.5 / 32, 1.5 / 32);
    free(e);
}

static void equal_grids(Gpu g) {
    Io s = io_open(g, 32, 24, 32, 24);
    zero_all(&s);
    tex_set(s.dis, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1; });
    tex_set(s.disOld, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1; });
    for (int run = 0; run < 2; run++) {
        tex_set(s.ns, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = run == 0 ? (y == 0) : (x % 2 == 0); });
        io_f15(&s);
        int bad = check_ns(&s, ^double(uint32_t x, uint32_t y) { return run == 0 ? (y == 0) : (x % 2 == 0); });
        CHECKC(bad == 0, "n1io3-equal-grids run %d: %d wrong (the texel itself is read)", run + 1, bad);
    }
}

static void nan_case(Gpu g) {
    Io s = io_open(g, 100, 68, 100, 68);
    zero_all(&s);
    void (^grey)(uint32_t, uint32_t, float *) = ^(uint32_t x, uint32_t y, float v[4]) {
        float q = (float)((3 * x + 7 * y) % 16) / 16;
        v[0] = v[1] = v[2] = q; v[3] = 1;
    };
    tex_set(s.colCur, grey);
    tex_set(s.colOld, ^(uint32_t x, uint32_t y, float v[4]) {
        grey(x, y, v);
        if (x == 10 && y == 10) v[0] = v[1] = v[2] = NAN;
    });
    io_f15(&s);
    float *e = tex_get(s.errOut, 2);
    const h16 *o = s.n1in.contents;
    uint32_t X = 10 + s.ox, Y = 10 + s.oy;
    const h16 *pix = o + ((size_t)(Y / 2) * s.w1 + X / 2) * 48;
    int slot = (X & 1) + 2 * (Y & 1);
    CHECKC(e[(10 * 100 + 10) * 2] == 0 && e[(10 * 100 + 10) * 2 + 1] == 0, "n1io-nan: err.feat.cur at the NaN pixel is (%g, %g)", e[(10 * 100 + 10) * 2], e[(10 * 100 + 10) * 2 + 1]);
    CHECKC((double)pix[4 * 1 + slot] == 0 && (double)pix[4 * 2 + slot] == 0 && (double)pix[4 * 6 + slot] == 0, "n1io-nan: features 1, 2, 6 are zero");
    free(e);
}

/* ---- F17 ---- */

static void unpack_cases(Gpu g) {
    MetalFGParams P = wgen_params();
    uint32_t Wf = 100, Hf = 68;
    Io s = io_open(g, Wf, Hf, Wf, Hf);
    id<MTLTexture> ms = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf), uo = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf), uc = gpu_tex(&g, MTLPixelFormatR8Unorm, Wf, Hf);
    id<MTLTexture> lgP = gpu_tex(&g, MTLPixelFormatRG16Float, Wf, Hf), lgC = gpu_tex(&g, MTLPixelFormatRG16Float, Wf, Hf), st = gpu_tex(&g, MTLPixelFormatR16Float, Wf, Hf);
    id<MTLBuffer> out = nn_buf(&s.rig, NULL, (size_t)s.w1 * s.h1 * 16);
    h16 *o = out.contents;
    void (^run)(uint32_t) = ^(uint32_t steer) {
        MetalFGF17Args a = {ms, uo, uc, lgP, lgC, st, out, Wf, Hf, s.w1, s.h1, s.ox, s.oy, steer, P.b_steer, P.th_mask};
        nn_run(&s.rig, ^(MetalFGEnc *e) { [s.io f17:e args:a]; });
    };
    /* n1io-unpack-identity */
    for (uint32_t v = 0; v < s.h1; v++) for (uint32_t u = 0; u < s.w1; u++) for (int c = 0; c < 16; c++) o[((size_t)v * s.w1 + u) * 16 + c] = (h16)(c + (u % 4) / 4.0);
    for (id<MTLTexture> t in @[ ms, uo, uc ]) tex_set(t, ^(uint32_t x, uint32_t y, float v[4]) {});
    run(0);
    float *a = tex_get(lgP, 2), *b = tex_get(lgC, 2), *r = tex_get(st, 1);
    int bad = 0;
    for (uint32_t y = 0; y < Hf; y++) for (uint32_t x = 0; x < Wf; x++) {
        uint32_t X = x + s.ox, Y = y + s.oy;
        const h16 *px = o + ((size_t)(Y / 2) * s.w1 + X / 2) * 16;
        int sl = 2 * (Y & 1) + (X & 1);
        double k[4];
        for (int j = 0; j < 4; j++) k[j] = (double)px[4 * j + sl];
        size_t i = (size_t)y * Wf + x;
        if (a[i * 2] != k[0] || a[i * 2 + 1] != k[1] || b[i * 2] != k[2] || b[i * 2 + 1] != k[3]) bad++;
        double rr = 1 / (1 + exp(k[3] - k[2]));
        if (fabs(r[i] - rr) > fmax(rr * 0x1p-10, 0x1p-14)) bad++;
    }
    CHECKC(bad == 0, "n1io-unpack-identity: %d wrong", bad);
    free(a); free(b); free(r);

    /* n1io-mask-steer */
    for (uint32_t v = 0; v < s.h1; v++) for (uint32_t u = 0; u < s.w1; u++) for (int c = 0; c < 16; c++) o[((size_t)v * s.w1 + u) * 16 + c] = (h16)(floor(c / 4.0) / 2 - 0.75);
    int clo = (int)floor(255 * P.th_mask), chi = clo + 1;
    static float mcodes[3] = {0, 128, 255};
    static int ucodes[3];
    ucodes[0] = clo; ucodes[1] = chi; ucodes[2] = 255;
    tex_set(ms, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = mcodes[x * 3 / Wf] / 255.0f; });
    void (^bands)(uint32_t, uint32_t, float *) = ^(uint32_t x, uint32_t y, float v[4]) { v[0] = ucodes[y * 3 / Hf] / 255.0f; };
    tex_set(uo, bands); tex_set(uc, bands);
    for (uint32_t steer = 0; steer < 2; steer++) {
        run(steer);
        a = tex_get(lgP, 2); b = tex_get(lgC, 2); r = tex_get(st, 1);
        bad = 0;
        for (uint32_t y = 0; y < Hf; y++) for (uint32_t x = 0; x < Wf; x++) {
            double m = mcodes[x * 3 / Wf] / 255.0;
            int uc_code = ucodes[y * 3 / Hf];
            double k0 = -0.75, k1 = -0.25, k2 = 0.25, k3 = 0.75, bs = P.b_steer;
            k0 += m * (-bs - k0); k2 += m * (-bs - k2);
            double rr = 1 / (1 + exp(k3 - k2));
            if (steer && uc_code / 255.0 > P.th_mask) { k0 -= bs; k1 += bs; k2 -= bs; k3 += bs; }
            size_t i = (size_t)y * Wf + x;
            double tol = 0x1p-10;
            if (fabs(a[i * 2] - k0) > fmax(fabs(k0) * tol, 1e-3) || fabs(a[i * 2 + 1] - k1) > fmax(fabs(k1) * tol, 1e-3) ||
                fabs(b[i * 2] - k2) > fmax(fabs(k2) * tol, 1e-3) || fabs(b[i * 2 + 1] - k3) > fmax(fabs(k3) * tol, 1e-3)) bad++;
            if (fabs(r[i] - rr) > fmax(rr * tol, 0x1p-14)) bad++;
        }
        CHECKC(bad == 0, "n1io-mask-steer (steering %u): %d wrong", steer, bad);
        free(a); free(b); free(r);
    }
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        pack_index(g); warp_error(g); mot_unit(g); ns_point(g); motion_point(g); equal_grids(g); nan_case(g); unpack_cases(g);
    }
    printf(fails ? "g_net1io: %d FAILURES\n" : "g_net1io: ok\n", fails);
    return fails != 0;
}
