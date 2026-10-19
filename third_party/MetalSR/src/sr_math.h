// Shared Metal helpers for the numerics of sr.numerics. The library is compiled with fast math; the helpers whose
// result depends on exact IEEE behaviour (coordinates, rounding, coverage, non-finite tests, transcendentals,
// the auto-exposure grid) are compiled with strict math through the function-scope pragma SR_SAFE.
#ifndef SR_MATH_H
#define SR_MATH_H
#include <metal_stdlib>
using namespace metal;

#ifndef SR_NO_SAFE_PRAGMA
#define SR_SAFE _Pragma("METAL fp math_mode(safe)")
#else
#define SR_SAFE
#endif

static inline bool sr_isfinite(float v) { SR_SAFE return isfinite(v); }
static inline bool sr_isfinite_h(half v) { SR_SAFE return isfinite(v); }
static inline float sr_san(float v) { SR_SAFE return isfinite(v) ? v : 0.0f; }
static inline float3 sr_san3(float3 v) { SR_SAFE return float3(isfinite(v.x) ? v.x : 0.0f, isfinite(v.y) ? v.y : 0.0f, isfinite(v.z) ? v.z : 0.0f); }
static inline float4 sr_san4(float4 v) {
    SR_SAFE
    return float4(isfinite(v.x) ? v.x : 0.0f, isfinite(v.y) ? v.y : 0.0f, isfinite(v.z) ? v.z : 0.0f, isfinite(v.w) ? v.w : 0.0f);
}

// no reassociation and no contraction of separate multiplies and adds: for sums whose single roundings are fixed by a spec
#ifndef SR_NO_SAFE_PRAGMA
#define SR_STRICT _Pragma("METAL fp math_mode(safe)") _Pragma("METAL fp contract(off)")
#else
#define SR_STRICT
#endif

// the head's epilogue (sr.numerics, Epilogues): the accumulator is rounded to fp16, then the fp16 bias is added in fp16
static inline half sr_two_round_add(float acc, half bias) { SR_STRICT half r = (half)acc; return r + bias; }

// IEEE classification of an fp16 value by its exponent field (all ones: infinity or NaN)
static inline bool sr_finite_bits(half v) { return (as_type<ushort>(v) & 0x7C00u) != 0x7C00u; }
static inline bool sr_isnan_bits(half v) { return (as_type<ushort>(v) & 0x7FFFu) > 0x7C00u; }

// round toward zero to binary16 (sr.numerics section 5: what a texture write of fp32 to RGBA16F does on the reference hardware)
static inline half sr_rz16(float v) {
    SR_STRICT
    half r = (half)v;
    ushort b = as_type<ushort>(r);
    if (isfinite(v) && fabs((float)r) > fabs(v)) r = as_type<half>((ushort)(b - 1));
    return r;
}

// sat16 of sr.head_filter_r4: rn16 when finite, the largest finite magnitude (with the sign of w) when rn16 overflows, +0 for NaN
static inline half sr_sat16(float w) {
    SR_STRICT
    if (isnan(w)) return (half)0.0f;
    return (half)clamp(w, -65504.0f, 65504.0f);
}

// Output-stage guard, by bit patterns (nothing a fast-math compile can fold away): an infinity becomes the largest finite magnitude
// of its sign, a NaN +0, every finite value is returned unchanged. sat16 above already does this for the filter's values; this is
// the last line before a store that leaves the engine.
static inline half sr_guard16(half v) {
    ushort b = as_type<ushort>(v);
    ushort r = (ushort)(((b & 0x7FFFu) > 0x7C00u) ? 0u : ((b & 0x8000u) | 0x7BFFu));
    return as_type<half>(((b & 0x7C00u) == 0x7C00u) ? r : b);
}
static inline half4 sr_guard16(half4 v) { return half4(sr_guard16(v.x), sr_guard16(v.y), sr_guard16(v.z), sr_guard16(v.w)); }
// fp32 form for the converted output formats: non-finite as above, with m the largest finite magnitude of the destination format;
// a finite value beyond m is clamped to it (the destination would round it to infinity)
static inline float sr_guard32(float v, float m) {
    uint b = as_type<uint>(v);
    bool nonfinite = (b & 0x7F800000u) == 0x7F800000u;
    bool nan = (b & 0x7FFFFFFFu) > 0x7F800000u;
    bool over = !nonfinite && as_type<float>(b & 0x7FFFFFFFu) > m;
    float lim = (b >> 31) ? -m : m;
    return nan ? 0.0f : ((nonfinite || over) ? lim : v);
}

static inline float sr_luma(float3 c) { return fma(c.z, 0.25f, fma(c.x, 0.25f, c.y * 0.5f)); }

// form A: u = J + fma(s, x + 0.5, -0.5); form B: u = fma(s, x + 0.5, J - 0.5)
#ifdef SR_GUARD_UNFUSED
// test build only (the head-coverage-ratio3 guard): the same coordinates with a separately rounded product
static inline float sr_coord_a(float s, float J, int x) { SR_STRICT float t = s * ((float)x + 0.5f); return J + (t - 0.5f); }
static inline float sr_coord_b(float s, float J, int x) { SR_STRICT float t = s * ((float)x + 0.5f); return t + (J - 0.5f); }
#else
static inline float sr_coord_a(float s, float J, int x) { SR_SAFE return J + fma(s, (float)x + 0.5f, -0.5f); }
static inline float sr_coord_b(float s, float J, int x) { SR_SAFE return fma(s, (float)x + 0.5f, J - 0.5f); }
#endif

// round half away from zero, exact for every fp32 input
static inline float sr_round(float v) { SR_SAFE return precise::round(v); }

// the correlation squash q(v), exactly as written (one division)
static inline float sr_qfun(float v, float q_a, float q_b, float q_clip) {
    SR_SAFE
    return fabs(v) <= q_clip ? v * (q_a + v * v) / (q_a + q_b * (v * v)) : copysign(1.0f, v);
}

static inline float sr_sig(float v) { SR_SAFE return 1.0f / (1.0f + precise::exp(-v)); }

// tanh for the recurrent channels (fp16 inputs widened to fp32)
static inline float sr_tanh(float x) {
    SR_SAFE
    float a = fabs(x);
    if (a >= 10.0f) return copysign(1.0f, x);
    float r;
    if (a < 0.5f) {
        float t = 2.0f * a;
        float e = precise::exp(t);
        float em1 = e - 1.0f;
        em1 = fma(t - precise::log(e), e, em1);        // first-order correction of the rounding of e
        r = em1 / (em1 + 2.0f);
    } else {
        float e = precise::exp(2.0f * a);
        r = 1.0f - 2.0f / (e + 1.0f);
    }
    return copysign(r, x);
}

// log1p for the motion-difference feature (fp32), about 1 ulp: 1 + x as an exact (u, c) pair, log(u) = e ln 2 + 2 atanh(s)
// with s = (m - 1) / (m + 1) refined by a residual step, summed in double-float order.
static inline float sr_log1p(float x) {
    SR_SAFE
    if (isnan(x)) return x;
    if (isinf(x)) return x > 0 ? x : NAN;
    if (x == 0.0f) return x;
    float u = 1.0f + x;
    float bb = u - 1.0f;
    float c = (1.0f - (u - bb)) + (x - bb);                       // exact error of the sum (two-sum)
    if (u == 1.0f) return x;
    uint bits = as_type<uint>(u);
    int e = (int)((bits >> 23) & 0xFFu) - 127;
    float m = as_type<float>((bits & 0x007FFFFFu) | 0x3F800000u);   // [1, 2)
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }
    float f = m - 1.0f;                                           // exact
    float d = 2.0f + f;
    float d_lo = f - (d - 2.0f);                                  // exact: 2 + f = d + d_lo
    float s = f / d;
    s += fma(-s, d_lo, fma(-s, d, f)) / d;                         // residual of the division including the rounding of d
    float s2 = s * s;
    float corr = 2.0f * s * s2 * fma(s2, fma(s2, fma(s2, fma(s2, 1.0f / 11.0f, 1.0f / 9.0f), 1.0f / 7.0f), 1.0f / 5.0f), 1.0f / 3.0f);
    const float LN2_HI = 0.693145751953125f, LN2_LO = 1.42860682030941723212e-6f;
    float a = (float)e * LN2_HI;                                  // exact
    float b = 2.0f * s;
    float hi = a + b;
    float t = hi - a;
    float err = (a - (hi - t)) + (b - t);                         // two-sum error of a + b
    float tail = corr + fma((float)e, LN2_LO, c / u) + err;
    return hi + tail;
}

#endif
