// Test entry points for the numeric helpers (exhaustive scans run from tests/math_gpu.m).
#include "sr_math.h"

struct MathParams { uint count; uint base; };

kernel void test_tanh(device const ushort *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = sr_tanh((float)as_type<half>(x[i]));
}
kernel void test_log1p(device float *y [[buffer(1)]], constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = sr_log1p(as_type<float>(p.base + i * 64u));
}
// round() of consecutive fp32 bit patterns starting at p.base (positive range), writes bit patterns
kernel void test_round(device uint *y [[buffer(1)]], constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = as_type<uint>(sr_round(as_type<float>(p.base + i)));
}
kernel void test_ln(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                    constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = precise::log2(x[i]) * 0.6931471805599453f;
}

// bilinear sampler probe (tests/probe_gpu.m): hardware sub-texel weights and filtering precision
constexpr sampler kProbeSampler(coord::normalized, address::clamp_to_edge, filter::linear);
kernel void test_bilinear(texture2d<float> src [[texture(0)]], device const float2 *uv [[buffer(0)]], device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = src.sample(kProbeSampler, uv[i]);
}

// texture-write rounding probe: writes a float to an RGBA16F texel as is
kernel void test_texwrite(texture2d<float, access::write> t [[texture(0)]], device const float *v [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    t.write(float4(v[i], -v[i], v[i], 1.0f), uint2(i, 0));
}

// output-stage guard over every fp16 bit pattern and a list of fp32 patterns (tests/math_gpu.m)
kernel void test_guard16(device const ushort *x [[buffer(0)]], device ushort *y [[buffer(1)]], constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = as_type<ushort>(sr_guard16(as_type<half>(x[i])));
}
kernel void test_guard32(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]], constant MathParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = as_type<uint>(sr_guard32(as_type<float>(x[i]), as_type<float>(p.base)));
}
