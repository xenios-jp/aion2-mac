/* Shared helpers of the GPU tests. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#import "gpu.h"
#include "numerics.h"

static int g_fails;
static double g_bad_frac = 0;   /* fraction of values allowed outside the tolerance (specs allow e.g. 0.1 %) */
#define EXPECT(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fails++; } } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static inline double urand(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (g_rng >> 11) * (1.0 / 9007199254740992.0); }
static inline double nrand(void) { return sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307179586 * urand()); }

static SrGpu *g_gpu;
static void gpu_init(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSData *lib = [NSData dataWithContentsOfFile:@"build/metalsr.metallib"];
    NSString *err = nil;
    g_gpu = [[SrGpu alloc] initWithDevice:dev libraryData:lib error:&err];
    if (!g_gpu) { printf("gpu init: %s\n", err.UTF8String); exit(2); }
}
static void run1(void (^b)(SrEnc *)) {
    NSString *e = sr_run_sync(g_gpu, b);
    if (e) { printf("GPU error: %s\n", e.UTF8String); exit(2); }
}
static id<MTLBuffer> bufcopy(const void *src, size_t n) {
    id<MTLBuffer> b = [g_gpu buffer:n];
    if (src) memcpy(b.contents, src, n);
    return b;
}

static inline int ulp_dist16(f16 a, f16 b) {
    int16_t ia, ib; memcpy(&ia, &a, 2); memcpy(&ib, &b, 2);
    int ka = ia < 0 ? 0x8000 - ia : ia + 0x8000, kb = ib < 0 ? 0x8000 - ib : ib + 0x8000;
    return abs(ka - kb);
}
/* fraction of bit-equal values must be >= frac_eq; all values within `ulps` or `abs_tol` */
static void compare16(const char *name, const f16 *got, const f16 *ref, size_t n, double frac_eq, int ulps, double abs_tol) {
    size_t eq = 0; int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &ref[i], 2) == 0) { eq++; continue; }
        if (!(got[i] == got[i]) || !(ref[i] == ref[i])) { bad++; continue; }
        if (ulp_dist16(got[i], ref[i]) > ulps && fabs((double)got[i] - (double)ref[i]) > abs_tol) {
            if (bad < 4 && getenv("SR_DEBUG")) printf("    mismatch @%zu got %g ref %g\n", i, (double)got[i], (double)ref[i]);
            bad++;
        }
    }
    double f = n ? (double)eq / n : 1;
    printf("%s: %.5f bit-equal, %d out of tolerance\n", name, f, bad);
    EXPECT(f >= frac_eq && (double)bad <= g_bad_frac * (double)n, "%s", name);
}
static void fill16(f16 *a, size_t n, double lo, double hi) { for (size_t i = 0; i < n; i++) a[i] = (f16)(lo + (hi - lo) * urand()); }
