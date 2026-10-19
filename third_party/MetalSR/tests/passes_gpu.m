// GPU-vs-CPU tests of the network passes P2-P11 and the shared helpers (m8, upsampling).
#import "testutil.h"
#include "ref.h"
#import "passes.h"

static SrWeights upload(const uint8_t *img) {
    SrWeights w;
    w.buf = bufcopy(img, SR_WEIGHT_FILE_BYTES);
    sr_weights_prepare(g_gpu, &w);
    return w;
}
static SrScratch scratch_for(const SrLevels *L) {
    return (SrScratch){[g_gpu buffer:sr_scratch_bytes(L, 0)], [g_gpu buffer:sr_scratch_bytes(L, 1)], [g_gpu buffer:sr_scratch_bytes(L, 2)]};
}
static int g_wide;   /* path W kernels (K chunks of 32) when set */
static void to_levels(const RefLevels *r, SrLevels *l) { memcpy(l, r, sizeof *l); l->wide = g_wide; }

/* m8 comparison: fraction of equal codes, rest within one code (two only where small) */
static void compare_m8(const char *name, const uint8_t *got, const uint8_t *ref, size_t n, double frac, int exact) {
    size_t eq = 0, bad = 0, nan_left = 0;
    for (size_t i = 0; i < n; i++) {
        if (got[i] == 0xFF) nan_left++;
        if (got[i] == ref[i]) { eq++; continue; }
        int d = abs((int)got[i] - (int)ref[i]);
        if (!exact && fabsf(m8_decode(ref[i])) < 1.0f / 16 && fabsf(m8_decode(got[i])) < 1.0f / 16 && fabsf(m8_decode(ref[i]) - m8_decode(got[i])) <= 2e-3f) { continue; }
        if (exact || d > 2 || (d == 2 && m8_decode(ref[i]) >= 1.0f / 16)) { if (bad < 3 && getenv("SR_DEBUG")) printf("    m8 mismatch @%zu got %d (%g) ref %d (%g)\n", i, got[i], m8_decode(got[i]), ref[i], m8_decode(ref[i])); bad++; }
    }
    printf("%s: %.5f codes equal, %zu out of tolerance, %zu unwritten\n", name, n ? (double)eq / n : 1, bad, nan_left);
    EXPECT(bad == 0 && nan_left == 0 && (double)eq / n >= frac, "%s", name);
}

static void test_m8_and_upsample(void) {
    printf("m8 / upsample on GPU\n");
    uint16_t bits[32768]; uint8_t out[32768]; (void)out;
    for (int i = 0; i < 32768; i++) bits[i] = (uint16_t)i;
    id<MTLBuffer> bi = bufcopy(bits, sizeof bits), bo = bufcopy(NULL, sizeof out);
    run1(^(SrEnc *e) {
        struct { uint32_t n; } p = {32768};
        SrArg a[3] = {ARG_BUF(bi), ARG_BUF(bo), ARG_VAL(p)};
        [e dispatchGrid:MTLSizeMake(32768, 1, 1) pso:[g_gpu pipeline:@"m8_test_enc"] args:a count:3 threads:MTLSizeMake(256, 1, 1)];
    });
    int bad = 0;
    for (int i = 0; i < 32768; i++) {
        uint8_t want = m8_encode(from_bits16((uint16_t)i));
        uint8_t got = ((uint8_t *)bo.contents)[i];
        if (i > 0x7C00 ? got < 0xF9 : got != want) bad++;
    }
    EXPECT(bad == 0, "GPU m8 encode differs on %d patterns", bad);

    for (int variant = 0; variant < 2; variant++) {
        int wc = 9, hc = 6, C = 24;
        size_t nc = (size_t)wc * hc * C, nf = nc * 4;
        f16 *coarse = malloc(nc * 2), *skip = malloc(nf * 2), *ref = malloc(nf * 2);
        fill16(coarse, nc, -8, 8); fill16(skip, nf, -8, 8);
        ref_up2_add(coarse, wc, hc, C, skip, variant, ref);
        id<MTLBuffer> bc = bufcopy(coarse, nc * 2), bs = bufcopy(skip, nf * 2), bx = bufcopy(NULL, nf * 2);
        run1(^(SrEnc *e) { sr_up2_add(e, g_gpu, bc, 0, bs, 0, bx, wc, hc, C, variant); });
        compare16(variant ? "  upsample variant W" : "  upsample variant N", bx.contents, ref, nf, 1.0, 0, 0);
        free(coarse); free(skip); free(ref);
    }
}

static void test_p2(int Wo, int Ho, int mode, const char *name) {
    RefLevels R; ref_levels(Wo, Ho, &R); SrLevels L; to_levels(&R, &L);
    uint8_t *img = make_weights(mode, 1.0);
    size_t n0 = (size_t)R.W[0] * R.H[0], n1 = (size_t)R.W[1] * R.H[1], n2 = (size_t)R.W[2] * R.H[2];
    f16 *feat = malloc(n0 * 16 * 2);
    fill16(feat, n0 * 16, 0, 4);
    uint8_t *r0 = malloc(n0 * 32), *r1 = malloc(n1 * 64), *r2 = malloc(n2 * 64);
    ref_p2(img, &R, feat, r0, r1, r2);
    SrWeights w = upload(img);
    id<MTLBuffer> bf = bufcopy(feat, n0 * 32), b0 = [g_gpu buffer:n0 * 32], b1 = [g_gpu buffer:n1 * 64], b2 = [g_gpu buffer:n2 * 64];
    memset(b0.contents, 0xFF, n0 * 32); memset(b1.contents, 0xFF, n1 * 64); memset(b2.contents, 0xFF, n2 * 64);
    SrScratch s = scratch_for(&L);
    run1(^(SrEnc *e) { sr_p2(e, g_gpu, &w, &L, bf, b0, b1, b2, s); });
    printf("P2 %s (%dx%d)\n", name, Wo, Ho);
    int exact = mode != WMODE_RANDOM;
    compare_m8("  s0", b0.contents, r0, n0 * 32, exact ? 1.0 : 0.9999, exact);
    compare_m8("  s1", b1.contents, r1, n1 * 64, exact ? 1.0 : 0.9999, exact);
    compare_m8("  q2", b2.contents, r2, n2 * 64, exact ? 1.0 : 0.9999, exact);
    free(img); free(feat); free(r0); free(r1); free(r2);
}

static void test_p345(int Wo, int Ho, int mode, const char *name) {
    RefLevels R; ref_levels(Wo, Ho, &R); SrLevels L; to_levels(&R, &L);
    uint8_t *img = make_weights(mode, 1.0);
    SrWeights w = upload(img);
    SrScratch s = scratch_for(&L);
    static const int cin[6] = {0, 0, 0, 64, 128, 128}, cout[6] = {0, 0, 0, 128, 128, 256};
    for (int pass = 3; pass <= 5; pass++) {
        int k = pass - 1, W = R.W[k], H = R.H[k];
        size_t nin = (size_t)W * H * cin[pass], nout = (size_t)W * H * cout[pass], npool = (size_t)R.W[k + 1] * R.H[k + 1] * cout[pass];
        uint8_t *m8 = malloc(nin); f16 *x = malloc(nin * 2), *ys = malloc(nout * 2), *qs = malloc(npool * 2);
        if (pass == 3) { for (size_t i = 0; i < nin; i++) m8[i] = (uint8_t)(rand() % 0x78); m8_decode_buf(m8, nin, x); }
        else fill16(x, nin, 0, 8);
        ref_conv3x3(x, W, H, cin[pass], wt_of(img, 2 * (pass - 1)), wt_of(img, 2 * (pass - 1) + 1), cout[pass], ys);
        ref_maxpool(ys, W, H, cout[pass], qs);
        id<MTLBuffer> bin = bufcopy(pass == 3 ? (void *)m8 : (void *)x, pass == 3 ? nin : nin * 2), bs = [g_gpu buffer:nout * 2], bq = [g_gpu buffer:npool * 2];
        memset(bs.contents, 0xFF, nout * 2); memset(bq.contents, 0xFF, npool * 2);
        run1(^(SrEnc *e) { sr_p3to5(e, g_gpu, &w, &L, pass, bin, bs, bq, s); });
        printf("P%d %s (%dx%d)\n", pass, name, Wo, Ho);
        if (mode == WMODE_RANDOM) {
            compare16("  skip", bs.contents, ys, nout, 0.995, 32, 2e-3);
            compare16("  pooled", bq.contents, qs, npool, 0.995, 32, 2e-3);
        } else {
            compare16("  skip", bs.contents, ys, nout, 1.0, 0, 0);
            compare16("  pooled", bq.contents, qs, npool, 1.0, 0, 0);
        }
        free(m8); free(x); free(ys); free(qs);
    }
    free(img);
}

static void test_bottleneck(int Wo, int Ho, int mode, const char *name) {
    RefLevels R; ref_levels(Wo, Ho, &R); SrLevels L; to_levels(&R, &L);
    uint8_t *img = make_weights(mode, 1.0);
    SrWeights w = upload(img);
    int n = R.W[5] * R.H[5];
    f16 *q5 = malloc((size_t)n * 256 * 2), *za = malloc((size_t)n * 512 * 2), *zb = malloc((size_t)n * 256 * 2);
    fill16(q5, (size_t)n * 256, mode == WMODE_IDENTITY ? -2 : 0, 8);
    ref_conv1x1(q5, n, 256, wt_of(img, 10), wt_of(img, 11), 512, za, 2);
    ref_conv1x1(za, n, 512, wt_of(img, 12), wt_of(img, 13), 256, zb, 2);
    id<MTLBuffer> bq = bufcopy(q5, (size_t)n * 512), ba = [g_gpu buffer:(size_t)n * 1024], bb = [g_gpu buffer:(size_t)n * 512];
    id<MTLBuffer> bza = bufcopy(za, (size_t)n * 1024);
    run1(^(SrEnc *e) { sr_p6(e, g_gpu, &w, &L, bq, ba); sr_p7(e, g_gpu, &w, &L, bza, bb); });   /* P7 sees the reference z5a */
    printf("P6/P7 %s (L5 %dx%d)\n", name, R.W[5], R.H[5]);
    if (mode == WMODE_RANDOM) {
        compare16("  z5a", ba.contents, za, (size_t)n * 512, 0.995, 4, 1e-3);
        compare16("  z5b", bb.contents, zb, (size_t)n * 256, 0.995, 4, 1e-3);
    } else {
        compare16("  z5a", ba.contents, za, (size_t)n * 512, 1.0, 0, 0);
        compare16("  z5b", bb.contents, zb, (size_t)n * 256, 1.0, 0, 0);
    }
    free(img); free(q5); free(za); free(zb);
}

static void test_dec(int Wo, int Ho, int mode, int variant, const char *name) {
    RefLevels R; ref_levels(Wo, Ho, &R); SrLevels L; to_levels(&R, &L);
    uint8_t *img = make_weights(mode, 1.0);
    SrWeights w = upload(img);
    SrScratch s = scratch_for(&L);
    static const int lvl[12] = {0,0,0,0,0,0,0,0,4,3,2,1}, cc[12] = {0,0,0,0,0,0,0,0,256,128,128,64}, co[12] = {0,0,0,0,0,0,0,0,128,128,64,32};
    for (int pass = 8; pass <= 11; pass++) {
        int k = lvl[pass], cin = cc[pass], cout = co[pass];
        int cch = pass == 8 ? 256 : (pass == 9 ? 128 : (pass == 10 ? 128 : 64));   /* coarse channels */
        size_t nc = (size_t)R.W[k + 1] * R.H[k + 1] * cch, nf = (size_t)R.W[k] * R.H[k] * cin, no = (size_t)R.W[k] * R.H[k] * cout;
        int m8in = pass == 11;
        uint8_t *cm = malloc(nc), *sm = malloc(nf);
        f16 *c = malloc(nc * 2), *sk = malloc(nf * 2), *ref = malloc(no * 2);
        if (m8in) {
            for (size_t i = 0; i < nc; i++) cm[i] = (uint8_t)(rand() % 0x78);
            for (size_t i = 0; i < nf; i++) sm[i] = (uint8_t)(rand() % 0x78);
            m8_decode_buf(cm, nc, c); m8_decode_buf(sm, nf, sk);
        } else { fill16(c, nc, 0, 8); fill16(sk, nf, 0, 8); }
        ref_dec(img, &R, pass, variant, c, sk, ref);
        id<MTLBuffer> bc = bufcopy(m8in ? (void *)cm : (void *)c, m8in ? nc : nc * 2), bs = bufcopy(m8in ? (void *)sm : (void *)sk, m8in ? nf : nf * 2);
        int m8out = pass >= 10;
        id<MTLBuffer> bo = [g_gpu buffer:m8out ? no : no * 2];
        memset(bo.contents, 0xFF, bo.length);
        run1(^(SrEnc *e) { sr_dec(e, g_gpu, &w, &L, pass, variant, bc, bs, bo, s); });
        char nm[80]; snprintf(nm, sizeof nm, "P%d %s variant %c (%dx%d)", pass, name, "NW"[variant], Wo, Ho);
        printf("%s\n", nm);
        if (m8out) {
            uint8_t *rm = malloc(no); m8_encode_buf_cpu(ref, no, rm);
            compare_m8("  out", bo.contents, rm, no, mode == WMODE_RANDOM ? 0.998 : 1.0, mode != WMODE_RANDOM);
            free(rm);
        } else compare16("  out", bo.contents, ref, no, mode == WMODE_RANDOM ? 0.99 : 1.0, mode == WMODE_RANDOM ? 8 : 0, 2e-3);
        free(cm); free(sm); free(c); free(sk); free(ref);
    }
    free(img);
}

/* weights-roundtrip: the engine-side re-layout (head_b padded to 48 rows (head_pad)) read back and inverted equals the file bytes;
   bott-count: multiply-accumulates per bottleneck launch from the layer shapes (sr.bottleneck_1x1_r2) */
static void test_weights_and_counts(void) {
    uint8_t *img = make_weights(WMODE_RANDOM, 1.0);
    SrWeights w = upload(img);
    const uint8_t *padded = w.head_pad.contents, *file = img + w.off[32];
    int bad = memcmp(padded, file, 40 * 32 * 2) != 0;
    for (size_t i = 40 * 32 * 2; i < 48 * 32 * 2; i++) if (padded[i]) bad = 1;
    printf("weights-roundtrip: head_b padded copy %s the file bytes (rows 40..47 zero)\n", bad ? "DIFFERS from" : "equals");
    EXPECT(!bad, "weights-roundtrip");
    EXPECT(!memcmp(w.buf.contents, img, SR_WEIGHT_FILE_BYTES), "the resident file image equals the file");
    static const int npix[3] = {135, 240, 510};
    static const double want[3] = {17.7e6, 31.5e6, 66.8e6};
    for (int i = 0; i < 3; i++) {
        double p6 = (double)npix[i] * 256 * 512, p7 = (double)npix[i] * 512 * 256;
        printf("bott-count: %d L5 pixels: P6 %.1f M, P7 %.1f M multiply-accumulates (spec %.1f M)\n", npix[i], p6 / 1e6, p7 / 1e6, want[i] / 1e6);
        EXPECT(fabs(p6 / 1e6 - want[i] / 1e6) < 0.06 && p6 == p7, "bott-count");
    }
    free(img);
}

int main(void) {
    @autoreleasepool {
        gpu_init();
        test_weights_and_counts();
        test_m8_and_upsample();
        test_p2(256, 128, WMODE_IDENTITY, "enc1-identity");
        test_p2(1920, 1080, WMODE_RANDOM, "enc1-random");
        int sizes[5] = {640, 1024, 1920, 3840, 7680};
        for (int i = 0; i < 5; i++) { test_p2(sizes[i], 128, WMODE_RANDOM, "enc1-sizes wide"); test_p2(128, sizes[i], WMODE_RANDOM, "enc1-sizes tall"); }
        for (g_wide = 0; g_wide < 2; g_wide++) {
        test_p345(256, 128, WMODE_IDENTITY, g_wide ? "identity (path W)" : "identity");
        test_p345(1920, 1080, WMODE_RANDOM, "random");
        test_p345(640, 128, WMODE_RANDOM, "coverage");
        test_p345(1366, 768, WMODE_RANDOM, "odd");
        }
        g_wide = 0;
        test_bottleneck(256, 128, WMODE_IDENTITY, "identity");
        for (int i = 0; i < 3; i++) { int ws[3] = {1920, 640, 3840}, hs[3] = {1080, 128, 2160}; test_bottleneck(ws[i], hs[i], WMODE_RANDOM, "random"); }
        for (int v = 0; v < 2; v++) {
            test_dec(256, 128, WMODE_IDENTITY, v, "identity");
            test_dec(1920, 1080, WMODE_RANDOM, v, "random");
            test_dec(640, 128, WMODE_RANDOM, v, "coverage");
            test_dec(3840, 128, WMODE_RANDOM, v, "coverage");
        }
        printf(g_fails ? "passes_gpu: %d FAILURES\n" : "passes_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
