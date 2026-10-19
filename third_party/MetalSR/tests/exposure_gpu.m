// sr.exposure_meter acceptance: ae-constant, ae-grid, ae-special, ae-determinism, controls.
#import "testutil.h"
#import "passes.h"
#include "../src/frame.h"
#include "testparams.h"

static id<MTLTexture> colour_tex(int W, int H, const float *rgba) {
    id<MTLTexture> t = [g_gpu texture:MTLPixelFormatRGBA32Float w:W h:H];
    [t replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:rgba bytesPerRow:W * 16];
    return t;
}

/* returns E_now as fp16 bits */
static uint16_t run_ae(const AeParams *p, id<MTLTexture> colour) {
    size_t nf = sr_ae_scratch_floats(p->lw, p->lh);
    id<MTLBuffer> s0 = [g_gpu buffer:nf * 4], s1 = [g_gpu buffer:nf * 4];
    id<MTLTexture> en = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
    AeParams pv = *p;
    run1(^(SrEnc *e) { sr_ae(e, g_gpu, &pv, colour, s0, s1, en); });
    uint16_t v; [en getBytes:&v bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    return v;
}

/* CPU model: indexing in fp32 (no contraction), sum in float64 */
static void cpu_pixel(const AeParams *p, unsigned i, unsigned *qx, unsigned *qy) {
    unsigned a = i % p->lw, b = i / p->lw;
    float fx = (float)a / (float)(p->lw - 1), fy = (float)b / (float)(p->lh - 1);
    volatile float px = fx * (float)(p->Wr - 1), py = fy * (float)(p->Hr - 1);
    volatile float qx_ = px + 0.5f, qy_ = py + 0.5f;
    *qx = (unsigned)truncf(qx_); *qy = (unsigned)truncf(qy_);
}
static double cpu_ae(const AeParams *p, const float *rgba, int exclude_nonpositive, int *distinct) {
    double sum = 0; double n = (double)p->lw * p->lh; unsigned cnt = 0;
    for (unsigned i = 0; i < p->lw * p->lh; i++) {
        unsigned qx, qy; cpu_pixel(p, i, &qx, &qy);
        const float *c = rgba + ((size_t)qy * p->Wr + qx) * 4;
        float L = p->invp * lumaY(c[0], c[1], c[2]);
        if (isfinite(L) && L > 0) { sum += log2((double)(L + p->eps_luma)) * M_LN2; cnt++; }
    }
    if (exclude_nonpositive) n = cnt;
    (void)distinct;
    double mu = sum / n;
    return (double)p->key_ae * exp(-mu);
}
static int ulp16(uint16_t a, uint16_t b) { return abs((int)a - (int)b); }

static void make_call(SrCall *c, int Wo, int Ho, int Wr, int Hr, float pre) {
    memset(c, 0, sizeof *c);
    c->Wo = Wo; c->Ho = Ho; c->Wr = Wr; c->Hr = Hr; c->pre = pre; c->hdr = 1;
}

int main(void) {
    @autoreleasepool {
        gpu_init();
        SrParams mp = test_params();
        int sizes[][4] = {{1920, 1080, 1280, 720}, {640, 360, 640, 360}, {3840, 2160, 1920, 1080}, {1366, 768, 911, 512}, {100, 60, 50, 30}, {200, 120, 1000, 600}};
        /* ae-constant */
        for (size_t s = 0; s < 6; s++) {
            int Wo = sizes[s][0], Ho = sizes[s][1], Wr = sizes[s][2], Hr = sizes[s][3];
            SrCall c; make_call(&c, Wo, Ho, Wr, Hr, 1.0f); AeParams p; sr_make_ae(&c, &mp, &p);
            float *img = malloc((size_t)Wr * Hr * 16);
            for (size_t i = 0; i < (size_t)Wr * Hr; i++) { img[4 * i] = img[4 * i + 1] = img[4 * i + 2] = 0.37f; img[4 * i + 3] = 1; }
            uint16_t got = run_ae(&p, colour_tex(Wr, Hr, img));
            float L0 = lumaY(0.37f, 0.37f, 0.37f);
            double ex = (double)p.key_ae * exp(-(double)log((double)(L0 + p.eps_luma)));
            uint16_t want = bits16((f16)ex);
            printf("ae-constant %dx%d <- %dx%d (grid %ux%u): E=%g want %g\n", Wo, Ho, Wr, Hr, p.lw, p.lh, (double)from_bits16(got), (double)from_bits16(want));
            EXPECT(ulp16(got, want) <= 1, "ae-constant");
            free(img);
        }
        /* ae-grid: sampled pixel set identical to the fp32 CPU computation; exhaustive over the listed render sizes */
        {
            int wr[4] = {640, 960, 1280, 1920}, hr[4] = {360, 540, 720, 1080};
            long mism = 0, total = 0;
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
                SrCall c; make_call(&c, 1920, 1080, wr[a], hr[b], 1.0f); AeParams p; sr_make_ae(&c, &mp, &p);
                uint32_t n = p.lw * p.lh;
                id<MTLBuffer> out = [g_gpu buffer:(size_t)n * 8];
                AeParams pv = p;
                run1(^(SrEnc *e) { SrArg ar[2] = {ARG_BUF(out), ARG_VAL(pv)}; [e dispatchGrid:MTLSizeMake(n, 1, 1) pso:[g_gpu pipeline:@"ae_debug_index"] args:ar count:2 threads:MTLSizeMake(256, 1, 1)]; });
                const uint32_t *q = out.contents;
                for (uint32_t i = 0; i < n; i++) { unsigned qx, qy; cpu_pixel(&p, i, &qx, &qy); total++; if (q[2 * i] != qx || q[2 * i + 1] != qy) mism++; }
            }
            printf("ae-grid: %ld sampled pixels compared, %ld differ\n", total, mism);
            EXPECT(mism == 0, "ae-grid index set");
            /* E with a coordinate-encoding colour */
            SrCall c; make_call(&c, 1920, 1080, 1280, 720, 1.0f); AeParams p; sr_make_ae(&c, &mp, &p);
            float *img = malloc((size_t)1280 * 720 * 16);
            for (int y = 0; y < 720; y++) for (int x = 0; x < 1280; x++) { float *t = img + ((size_t)y * 1280 + x) * 4; t[0] = (x % 97) / 50.0f + 0.01f; t[1] = (y % 61) / 40.0f; t[2] = ((x + y) % 31) / 20.0f; t[3] = 1; }
            uint16_t got = run_ae(&p, colour_tex(1280, 720, img));
            uint16_t want = bits16((f16)cpu_ae(&p, img, 0, NULL));
            printf("ae-grid E: %g want %g\n", (double)from_bits16(got), (double)from_bits16(want));
            EXPECT(ulp16(got, want) <= 1, "ae-grid E");
            /* ae-special */
            for (int k = 0; k < 400; k++) {
                size_t i = (size_t)(urand() * 1280 * 720) * 4;
                int kind = k % 4;
                float v = kind == 0 ? NAN : (kind == 1 ? INFINITY : (kind == 2 ? 0.0f : -3.0f));
                img[i] = img[i + 1] = img[i + 2] = v;
            }
            got = run_ae(&p, colour_tex(1280, 720, img));
            want = bits16((f16)cpu_ae(&p, img, 0, NULL));
            printf("ae-special: E=%g want %g\n", (double)from_bits16(got), (double)from_bits16(want));
            EXPECT(ulp16(got, want) <= 1, "ae-special");
            uint16_t ctl = bits16((f16)cpu_ae(&p, img, 1, NULL));
            printf("  control (non-positive luma excluded from the count): %g (must differ)\n", (double)from_bits16(ctl));
            EXPECT(ctl != want, "control must differ");
            /* ae-determinism */
            uint16_t again = run_ae(&p, colour_tex(1280, 720, img));
            EXPECT(again == got, "ae-determinism");
            free(img);
        }
        /* P13 */
        {
            id<MTLTexture> a = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1], b = [g_gpu texture:MTLPixelFormatR16Float w:1 h:1];
            uint16_t v = bits16((f16)0.73f);
            [a replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&v bytesPerRow:2];
            run1(^(SrEnc *e) { sr_p13(e, g_gpu, a, b); });
            uint16_t r; [b getBytes:&r bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
            EXPECT(r == v, "P13 copy");
        }
        printf(g_fails ? "exposure_gpu: %d FAILURES\n" : "exposure_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
