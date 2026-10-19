/* sr.numerics acceptance, CPU part: m8-exhaustive, controls that must fail. */
#include <stdio.h>
#include <stdlib.h>
#include "numerics.h"

static int fails;
#define EXPECT(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

/* Independent model written from the definition with exact real arithmetic (doubles hold every value involved). */
static uint8_t m8_model(double x, int ties_even) {
    if (x != x) return 0xFF;
    if (isinf(x)) return 0xF8;
    if (x < ldexp(1.0, -14)) {                        /* subnormal or zero: toward zero, multiple of 2^-17 */
        double q = floor(x / ldexp(1.0, -17));
        return (uint8_t)q;
    }
    int e; frexp(x, &e);                              /* x = f * 2^e, f in [0.5, 1) */
    int e2 = e - 1;                                   /* floor(log2 x) */
    double quantum = ldexp(1.0, e2 - 3);
    double t = x / quantum;                           /* in [8, 16) */
    double r = floor(t + 0.5);                        /* ties away (x > 0) */
    if (ties_even) r = rint(t);
    double val = r * quantum;
    if (val >= 65536.0) return 0xF8;
    int ve; double vf = frexp(val, &ve);              /* val = vf * 2^ve */
    int ex = ve - 1 + 15;                             /* biased exponent */
    int m = (int)lrint((vf * 2 - 1) * 8);
    if (m == 8) { m = 0; ex++; }
    if (ex >= 31) return 0xF8;
    return (uint8_t)(ex * 8 + m);
}

int main(void) {
    int bad = 0, bad_even = 0;
    for (uint32_t b = 0; b < 0x8000; b++) {
        f16 x = from_bits16((uint16_t)b);
        uint8_t got = m8_encode(x), want = m8_model((double)x, 0);
        if (b > 0x7C00) { if (got < 0xF9) bad++; continue; }
        if (got != want) { if (bad < 5) printf("  m8 mismatch bits %04x got %02x want %02x\n", b, got, want); bad++; }
        if (m8_model((double)x, 1) != want) bad_even++;
    }
    EXPECT(bad == 0, "m8 encode differs from the model on %d inputs", bad);
    EXPECT(bad_even > 0, "control: ties-to-even model must differ (%d)", bad_even);
    for (int c = 0; c < 256; c++) EXPECT(bits16(from_bits16((uint16_t)(c << 7))) == (c << 7), "decode %d", c);
    EXPECT(m8_encode((f16)65504.0f) == 0xF8 && m8_encode((f16)61440.0f) == 0xF7 && m8_encode((f16)63488.0f) == 0xF8, "overflow codes");
    /* monotonic */
    uint8_t prev = 0;
    for (uint32_t b = 0; b <= 0x7C00; b++) { uint8_t c = m8_encode(from_bits16((uint16_t)b)); EXPECT(c >= prev, "monotonic at %04x", b); prev = c; }
    /* rz16 */
    EXPECT(bits16(rz16(1.0f + 1.0f / 4096)) == 0x3C00 && bits16(rz16(-1.0f - 1.0f / 4096)) == 0xBC00, "rz16");
    /* metric-bound (sr.test_scenes_r2): D(c) = c' / (1 + max channel of c') stays in [0, 1) for channels up to 2^27, and one extreme
       pixel cannot change a frame's squared error by more than a unit error per channel */
    {
        uint64_t s = 12345; int bad = 0; double worst = 0;
        for (int t = 0; t < 200000; t++) {
            double c[3], d[3], m = 0;
            for (int k = 0; k < 3; k++) { s = s * 6364136223846793005ull + 1442695040888963407ull; double u = (double)(s >> 11) / 9007199254740992.0; c[k] = (t % 3 == 0 ? -1 : 1) * u * 134217728.0 * (t % 5 == 0 ? 1.0 : u); }
            for (int k = 0; k < 3; k++) if (c[k] > m) m = c[k];
            for (int k = 0; k < 3; k++) { d[k] = (c[k] > 0 ? c[k] : 0) / (1 + m); if (!(d[k] >= 0 && d[k] < 1)) bad++; }
            double e = 0; for (int k = 0; k < 3; k++) e += d[k] * d[k];
            if (e > worst) worst = e;
        }
        printf("metric-bound: %d values outside [0, 1), largest squared error of one pixel against black %.3f (bound 3)\n", bad, worst);
        EXPECT(bad == 0 && worst <= 3.0, "metric-bound");
    }
    printf(fails ? "numerics_test: %d FAILURES\n" : "numerics_test: all ok\n", fails);
    return fails != 0;
}
