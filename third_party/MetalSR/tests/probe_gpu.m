// Probes of GPU behaviour that the CPU models of the tests rely on (sr.input_features_r2 section 7: the model emulates the
// fixed-point sub-texel weights of the GPU sampler):
//  1. the sub-texel weight of a bilinear sample is the fractional position rounded to the nearest 1/256;
//  2. for fp16 images the sampler returns an fp16 value: the interpolation with those weights, rounded to fp16, a tie
//     going away from zero. The sampler's internal precision is finite, so about 0.5 % of random samples land one fp16 ulp
//     away from the exact result (the exact value then lies within 1e-2 ulp of a rounding tie).
#import "testutil.h"

int main(void) {
    @autoreleasepool {
        gpu_init();
        const int N = 4096;
        f16 tex[8 * 2 * 4];
        for (int y = 0; y < 2; y++) for (int x = 0; x < 8; x++) for (int c = 0; c < 4; c++)
            tex[((y * 8) + x) * 4 + c] = (f16)(x == 4 ? (c == 0 ? 1.0 : c == 1 ? 1.0009765625 : c == 2 ? 0.25 : 1.0) : (x == 3 ? (c == 1 ? 1.0 : 0.0) : 0.0));
        id<MTLTexture> t = [g_gpu texture:MTLPixelFormatRGBA16Float w:8 h:2];
        [t replaceRegion:MTLRegionMake2D(0, 0, 8, 2) mipmapLevel:0 withBytes:tex bytesPerRow:8 * 8];
        float *uv = malloc(N * 8);
        for (int k = 0; k < N; k++) { uv[2 * k] = (3.5f + (float)k / N) / 8.0f; uv[2 * k + 1] = 0.5f; }
        id<MTLBuffer> bu = bufcopy(uv, N * 8), bo = [g_gpu buffer:N * 16];
        run1(^(SrEnc *e) {
            SrArg a[3] = {ARG_TEX(t), ARG_BUF(bu), ARG_BUF(bo)};
            [e dispatchGrid:MTLSizeMake(N, 1, 1) pso:[g_gpu pipeline:@"test_bilinear"] args:a count:3 threads:MTLSizeMake(64, 1, 1)];
        });
        const float *o = bo.contents;
        int mism_near = 0;
        for (int k = 0; k < N; k++) if (fabs(o[4 * k] - floor((double)k / N * 256 + 0.5) / 256) > 1e-7) mism_near++;
        printf("sub-texel weight vs the fractional position rounded to the nearest 1/256: %d of %d positions differ\n", mism_near, N);
        EXPECT(mism_near == 0, "sub-texel weights are 8-bit, round to nearest");
        printf("mid value of 1 .. 1 + 2^-10: %.10f (the exact value 1.00048828125 is a tie of fp16: the sampler returns an fp16 value, tie away from zero)\n", (double)o[4 * (N / 2) + 1]);
        EXPECT(o[4 * (N / 2) + 1] == 1.0009765625f, "fp16 result of the sampler, tie away from zero");

        /* random texels and positions against the model: quantised weights, exact interpolation, fp16 result with ties away */
        const int M = 20000, S = 16;
        f16 rt[S * S * 4];
        for (int i = 0; i < S * S * 4; i++) rt[i] = (f16)(0.25 + 1.75 * urand());
        id<MTLTexture> t2 = [g_gpu texture:MTLPixelFormatRGBA16Float w:S h:S];
        [t2 replaceRegion:MTLRegionMake2D(0, 0, S, S) mipmapLevel:0 withBytes:rt bytesPerRow:S * 8];
        float *uv2 = malloc(M * 8);
        for (int k = 0; k < M; k++) {
            uv2[2 * k] = (float)((2.0 + floor(urand() * 12 * 4096) / 4096) / S); uv2[2 * k + 1] = (float)((2.0 + floor(urand() * 12 * 4096) / 4096) / S);
        }
        id<MTLBuffer> bu2 = bufcopy(uv2, M * 8), bo2 = [g_gpu buffer:M * 16];
        run1(^(SrEnc *e) {
            SrArg a[3] = {ARG_TEX(t2), ARG_BUF(bu2), ARG_BUF(bo2)};
            [e dispatchGrid:MTLSizeMake(M, 1, 1) pso:[g_gpu pipeline:@"test_bilinear"] args:a count:3 threads:MTLSizeMake(64, 1, 1)];
        });
        const float *o2 = bo2.contents;
        int bad = 0, far = 0, rne_bad = 0;
        for (int k = 0; k < M; k++) {
            double tx = (double)uv2[2 * k] * S - 0.5, ty = (double)uv2[2 * k + 1] * S - 0.5;
            int x0 = (int)floor(tx), y0 = (int)floor(ty);
            double fx = floor((tx - x0) * 256 + 0.5) / 256, fy = floor((ty - y0) * 256 + 0.5) / 256;
            for (int c = 0; c < 3; c++) {
                double a = (double)rt[(y0 * S + x0) * 4 + c], b = (double)rt[(y0 * S + x0 + 1) * 4 + c], d = (double)rt[((y0 + 1) * S + x0) * 4 + c], e = (double)rt[((y0 + 1) * S + x0 + 1) * 4 + c];
                double ex = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
                int ee; frexp(ex, &ee);
                double ulp = ldexp(1.0, ee - 11), want = floor(ex / ulp + 0.5) * ulp, g = o2[4 * k + c];
                if (g != want) { bad++; if (fabs(ex / ulp - floor(ex / ulp) - 0.5) > 0.02 || fabs(g - want) > 1.5 * ulp) far++; }
                if (g != (double)(f16)ex) rne_bad++;
            }
        }
        printf("random samples: %d of %d differ from the model (ties away), %d of them are not a near-tie or are more than one ulp off; ties to even would differ in %d\n", bad, M * 3, far, rne_bad);
        EXPECT(far == 0 && bad < M * 3 / 100, "sampler model");
    }
    printf(g_fails ? "probe_gpu: %d FAILURES\n" : "probe_gpu: all ok\n", g_fails);
    return g_fails != 0;
}
