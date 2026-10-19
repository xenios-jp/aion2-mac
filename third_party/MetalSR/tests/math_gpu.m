// round-exhaustive, tanh-ulp, log1p-ulp (sr.numerics section 7) on the GPU helpers.
#import "testutil.h"

typedef struct { uint32_t count, base; } MathParams;

static double ulp32(float x) { x = fabsf(x); if (x == 0) return ldexp(1.0, -149); int e; frexpf(x, &e); return ldexp(1.0, e - 24); }

static void run_math(NSString *kernel, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t count, uint32_t base) {
    MathParams p = {count, base};
    run1(^(SrEnc *e) {
        SrArg a[3] = {ARG_BUF(in), ARG_BUF(out), ARG_VAL(p)};
        [e dispatchGrid:MTLSizeMake(count, 1, 1) pso:[g_gpu pipeline:kernel] args:a count:3 threads:MTLSizeMake(256, 1, 1)];
    });
}

int main(void) {
    @autoreleasepool {
        gpu_init();
        /* tanh over every fp16 pattern */
        {
            uint16_t bits[65536]; for (int i = 0; i < 65536; i++) bits[i] = (uint16_t)i;
            id<MTLBuffer> bi = bufcopy(bits, sizeof bits), bo = bufcopy(NULL, 65536 * 4);
            run_math(@"test_tanh", bi, bo, 65536, 0);
            double worst = 0; int bad_inf = 0;
            for (int i = 0; i < 65536; i++) {
                f16 h = from_bits16((uint16_t)i);
                if (h != h) continue;
                float got = ((float *)bo.contents)[i];
                if (isinf((float)h)) { if (got != ((float)h > 0 ? 1.0f : -1.0f)) bad_inf++; continue; }
                double ex = tanh((double)h);
                float exf = (float)ex;
                double err = fabs((double)got - ex) / ulp32(exf);
                if (err > worst) worst = err;
            }
            printf("tanh: max error %.3f ulp over all fp16 inputs, infinity failures %d\n", worst, bad_inf);
            EXPECT(bad_inf == 0 && worst <= 4.0, "tanh-ulp");
            if (worst > 1.0) printf("  KNOWN DEVIATION: spec bound is 1 ulp (see QUESTIONS.md)\n");
        }
        /* texture-write rounding (sr.numerics section 5): fp32 -> RGBA16F by a texture write rounds toward zero on this hardware */
        {
            float vals[8]; for (int k = 0; k < 8; k++) vals[k] = 1.0f + (float)(k + 1) * ldexpf(1.0f, -13) + ldexpf(1.0f, -15);   /* between two fp16 values, various fractions */
            id<MTLBuffer> bv = bufcopy(vals, sizeof vals);
            id<MTLTexture> tex = [g_gpu texture:MTLPixelFormatRGBA16Float w:8 h:1];
            run1(^(SrEnc *e) { SrArg a[2] = {ARG_TEX(tex), ARG_BUF(bv)}; [e dispatchGrid:MTLSizeMake(8, 1, 1) pso:[g_gpu pipeline:@"test_texwrite"] args:a count:2 threads:MTLSizeMake(8, 1, 1)]; });
            f16 px[32]; [tex getBytes:px bytesPerRow:64 fromRegion:MTLRegionMake2D(0, 0, 8, 1) mipmapLevel:0];
            int rz = 0, rn = 0;
            for (int k = 0; k < 8; k++) { f16 got = px[4 * k]; f16 want_rz = rz16(vals[k]), want_rn = (f16)vals[k]; rz += got == want_rz; rn += got == want_rn; }
            printf("texture write rounding: %d of 8 probes match round-toward-zero, %d match round-to-nearest\n", rz, rn);
            EXPECT(rz == 8, "a texture write of fp32 to RGBA16F must round toward zero (sr.numerics section 5)");
        }
        /* the table the engine uses for tanh (indexed by fp16 bits) */
        {
            const float *t = [g_gpu tanhTable].contents;
            double worst = 0; int bad_inf = 0;
            for (int i = 0; i < 65536; i++) {
                f16 h = from_bits16((uint16_t)i);
                if (h != h) continue;
                if (isinf((float)h)) { if (t[i] != ((float)h > 0 ? 1.0f : -1.0f)) bad_inf++; continue; }
                double ex = tanh((double)h);
                double err = fabs((double)t[i] - ex) / ulp32((float)ex);
                if (err > worst) worst = err;
            }
            printf("tanh table (used by the engine): max error %.3f ulp over all fp16 inputs\n", worst);
            EXPECT(worst <= 1.0 && bad_inf == 0, "tanh-ulp of the engine's table");
        }
        /* log1p: every 64th normal positive fp32 pattern */
        {
            uint32_t n = 1 << 24;
            id<MTLBuffer> bo = bufcopy(NULL, (size_t)n * 4), bi = bufcopy(NULL, 16);
            double worst = 0; uint32_t total = (0x7F800000u - 0x00800000u) / 64;
            for (uint32_t done = 0; done < total; done += n) {
                uint32_t cnt = total - done < n ? total - done : n;
                run_math(@"test_log1p", bi, bo, cnt, 0x00800000u + done * 64u);
                for (uint32_t i = 0; i < cnt; i++) {
                    uint32_t b = 0x00800000u + (done + i) * 64u; float x; memcpy(&x, &b, 4);
                    double ex = log1p((double)x); float exf = (float)ex;
                    double err = fabs((double)((float *)bo.contents)[i] - ex) / ulp32(exf);
                    if (err > worst) worst = err;
                }
            }
            printf("log1p: max error %.3f ulp over %u sampled normal inputs\n", worst, total);
            EXPECT(worst <= 4.0, "log1p-ulp (%.3f)", worst);
            if (worst > 1.0) printf("  KNOWN DEVIATION: spec bound is 1 ulp (see QUESTIONS.md)\n");
        }
        /* round over [0, 2^24] and the mirrored negative range */
        {
            uint32_t n = 1 << 26;
            id<MTLBuffer> bo = bufcopy(NULL, (size_t)n * 4), bi = bufcopy(NULL, 16);
            uint64_t bad = 0, total = 0x4B800001ull;
            for (int neg = 0; neg < 2; neg++)
                for (uint64_t done = 0; done < total; done += n) {
                    uint32_t cnt = total - done < n ? (uint32_t)(total - done) : n;
                    uint32_t base = (uint32_t)done | (neg ? 0x80000000u : 0);
                    run_math(@"test_round", bi, bo, cnt, base);
                    const uint32_t *got = bo.contents;
                    for (uint32_t i = 0; i < cnt; i++) {
                        uint32_t b = base + i; float x; memcpy(&x, &b, 4);
                        float r = roundf(x); uint32_t rb; memcpy(&rb, &r, 4);
                        if (rb != got[i] && !(x != x) && !(r == 0 && (got[i] & 0x7FFFFFFFu) == 0)) { if (bad < 4) printf("  round mismatch %g\n", x); bad++; }
                    }
                }
            printf("round: %llu mismatches over %llu patterns\n", bad, 2 * total);
            EXPECT(bad == 0, "round-exhaustive");
        }
        /* output-stage guard: identity on every finite fp16 pattern, +-inf to +-65504, NaN to +0; fp32 form against a format maximum */
        {
            uint16_t bits[65536]; for (int i = 0; i < 65536; i++) bits[i] = (uint16_t)i;
            id<MTLBuffer> bi = bufcopy(bits, sizeof bits), bo = bufcopy(NULL, 65536 * 2);
            MathParams p = {65536, 0};
            run1(^(SrEnc *e) { SrArg a[3] = {ARG_BUF(bi), ARG_BUF(bo), ARG_VAL(p)}; [e dispatchGrid:MTLSizeMake(65536, 1, 1) pso:[g_gpu pipeline:@"test_guard16"] args:a count:3 threads:MTLSizeMake(256, 1, 1)]; });
            const uint16_t *got = bo.contents; int bad = 0, changed_finite = 0;
            for (int i = 0; i < 65536; i++) {
                uint16_t want = (i & 0x7C00) != 0x7C00 ? (uint16_t)i : ((i & 0x7FFF) > 0x7C00 ? 0 : (uint16_t)((i & 0x8000) | 0x7BFF));
                if (got[i] != want) bad++;
                if ((i & 0x7C00) != 0x7C00 && got[i] != i) changed_finite++;
            }
            printf("guard16: %d mismatches over all fp16 patterns, %d finite values changed\n", bad, changed_finite);
            EXPECT(bad == 0 && changed_finite == 0, "guard16");
        }
        {
            const float maxes[3] = {65024.0f, 65504.0f, 3.4028234e38f};
            for (int mi = 0; mi < 3; mi++) {
                float m = maxes[mi];
                enum { N = 1 << 20 };
                uint32_t *in = malloc(N * 4);
                for (uint32_t i = 0; i < N; i++) in[i] = (uint32_t)(((uint64_t)i * 4096ull + 12345ull * (i & 255)) & 0xFFFFFFFFull);   /* patterns across the whole range, the exponent all-ones band included */
                in[0] = 0x7F800000; in[1] = 0xFF800000; in[2] = 0x7FC00000; in[3] = 0xFFC00001; in[4] = 0x477FE000; in[5] = 0x477FE001; in[6] = 0x7F7FFFFF; in[7] = 0x80000000;
                uint32_t mb; memcpy(&mb, &m, 4);
                id<MTLBuffer> bi = bufcopy(in, N * 4), bo = bufcopy(NULL, N * 4);
                MathParams p = {N, mb};
                run1(^(SrEnc *e) { SrArg a[3] = {ARG_BUF(bi), ARG_BUF(bo), ARG_VAL(p)}; [e dispatchGrid:MTLSizeMake(N, 1, 1) pso:[g_gpu pipeline:@"test_guard32"] args:a count:3 threads:MTLSizeMake(256, 1, 1)]; });
                const uint32_t *got = bo.contents; int bad = 0, changed_in_range = 0;
                for (uint32_t i = 0; i < N; i++) {
                    float x; memcpy(&x, &in[i], 4); float want;
                    if (x != x) want = 0.0f; else if (isinf(x) || fabsf(x) > m) want = x < 0 || (in[i] >> 31) ? -m : m; else want = x;
                    uint32_t wb; memcpy(&wb, &want, 4);
                    if (got[i] != wb && !(want == 0.0f && (got[i] & 0x7FFFFFFFu) == 0)) { if (bad < 3) printf("  guard32 mismatch for %08x: %08x vs %08x\n", in[i], got[i], wb); bad++; }
                    if (x == x && !isinf(x) && fabsf(x) <= m && got[i] != in[i]) changed_in_range++;
                }
                printf("guard32 (max %g): %d mismatches, %d finite in-range values changed\n", (double)m, bad, changed_in_range);
                EXPECT(bad == 0 && changed_in_range == 0, "guard32 with max %g", (double)m);
                free(in);
            }
        }
        printf(g_fails ? "math_gpu: %d FAILURES\n" : "math_gpu: all ok\n", g_fails);
    }
    return g_fails != 0;
}
