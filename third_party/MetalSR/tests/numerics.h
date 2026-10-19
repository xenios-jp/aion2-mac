/* CPU models of the number formats and shared operators of sr.numerics (test/reference side). */
#ifndef SR_NUMERICS_H
#define SR_NUMERICS_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "../src/model.h"

typedef _Float16 f16;

static inline f16 rn16(double v) { return (f16)v; }
static inline uint16_t bits16(f16 h) { uint16_t b; memcpy(&b, &h, 2); return b; }
static inline f16 from_bits16(uint16_t b) { f16 h; memcpy(&h, &b, 2); return h; }
static inline f16 fma16(double a, double b, double c) { return (f16)fma(a, b, c); }

/* round toward zero to binary16 of an fp32 value */
static inline f16 rz16(float v) {
    f16 r = (f16)v;
    if (!isfinite(v)) return r;
    if (fabsf((float)r) > fabsf(v)) { uint16_t b = bits16(r); b--; r = from_bits16(b); }
    return r;
}

static inline float relu32(float v) { return v > 0 ? v : 0.0f; }
static inline f16 relu16(f16 v) { return v > (f16)0 ? v : (f16)0; }

/* luma of the YCoCg transform, strict fp32 */
static inline float lumaY(float r, float g, float b) { return fmaf(b, 0.25f, fmaf(r, 0.25f, g * 0.5f)); }

/* ---- m8 ---- */
static inline float m8_decode(uint8_t c) { return (float)from_bits16((uint16_t)(c << 7)); }
static inline uint8_t m8_encode(f16 x) {
    uint16_t b = bits16(x), mag = b & 0x7FFF;
    if (mag > 0x7C00) return 0xFF;                 /* NaN */
    if (mag < 0x400) return (uint8_t)(mag >> 7);   /* subnormal / zero: toward zero */
    return (uint8_t)((mag + 0x40) >> 7);           /* ties away; overflow reaches 0xF8 = infinity */
}

/* ---- upsampling (sr.numerics) ---- */
static inline f16 lerp16(int variant, f16 n, f16 f) {
    if (variant == 0) return fma16(0.75, (double)rn16((double)n - (double)f), (double)f);
    return fma16(0.75, (double)n, (double)rn16((double)f * 0.25));
}
/* fetch: coarse value (c, y, x) as fp16 */
typedef f16 (*fetch_fn)(const void *ctx, int c, int y, int x);
static inline f16 up2_at(const void *ctx, fetch_fn fetch, int Wc, int Hc, int c, int X, int Y, int variant) {
    int nx = X >> 1, ny = Y >> 1;
    int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
    if (fx < 0) fx = 0; if (fx > Wc - 1) fx = Wc - 1;
    if (fy < 0) fy = 0; if (fy > Hc - 1) fy = Hc - 1;
    f16 hn = lerp16(variant, fetch(ctx, c, ny, nx), fetch(ctx, c, ny, fx));
    f16 hf = lerp16(variant, fetch(ctx, c, fy, nx), fetch(ctx, c, fy, fx));
    return lerp16(variant, hn, hf);
}

/* ---- coordinates, strict fp32 ---- */
static inline float coord_formA(float s, float J, int x) { return J + fmaf(s, (float)x + 0.5f, -0.5f); }
static inline float coord_formB(float s, float J, int x) { return fmaf(s, (float)x + 0.5f, J - 0.5f); }

#endif
