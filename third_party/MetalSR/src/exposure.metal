// A1-A3 (automatic exposure) and P13 (exposure state copy), sr.exposure_meter.
#include "sr_math.h"
#include "shared_types.h"

// Grid element i -> render pixel (strict fp32: IEEE quotient, separate product and sum)
static inline uint2 ae_pixel(constant AeParams &p, uint i) {
    SR_SAFE
    uint a = i % p.lw, b = i / p.lw;
    float fx = (float)a / (float)(p.lw - 1);
    float fy = (float)b / (float)(p.lh - 1);
    float px = fx * (float)(p.Wr - 1);
    float py = fy * (float)(p.Hr - 1);
    px = px + 0.5f;
    py = py + 0.5f;
    return uint2((uint)trunc(px), (uint)trunc(py));
}

static inline float ae_value(constant AeParams &p, texture2d<float, access::read> colour, uint i) {
    SR_SAFE
    uint2 q = ae_pixel(p, i);
    float4 c = colour.read(q);
    float L = p.invp * sr_luma(c.xyz);
    if (isfinite(L) && L > 0.0f) return precise::log2(L + p.eps_luma) * 0.6931471805599453f;
    return 0.0f;
}

// first reduction step: luma, log and a threadgroup tree sum over blocks of 512 elements
kernel void ae_first(texture2d<float, access::read> colour [[texture(0)]],
                     device float *partial [[buffer(0)]],
                     constant AeParams &p [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
    SR_SAFE
    threadgroup float sh[512];
    uint i = tg * 512 + tid;
    sh[tid] = i < p.lw * p.lh ? ae_value(p, colour, i) : 0.0f;
    float tot_ = simd_sum(sh[tid]);                       // per-simdgroup sums, then the 16 partials in a fixed order
    threadgroup float part_[16];
    if ((tid & 31) == 0) part_[tid >> 5] = tot_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t_ = 0.0f; for (int k_ = 0; k_ < 16; k_++) t_ += part_[k_]; sh[0] = t_; }
    if (tid == 0) partial[tg] = sh[0];
}

struct AeMid { uint n; };

// sum of the 512 values in sh (a block of the reduction): per-simdgroup sums, then the 16 partials in a fixed order
static inline float ae_block_sum(threadgroup float *sh, threadgroup float *part_, uint tid) {
    SR_SAFE
    float tot_ = simd_sum(sh[tid]);
    if ((tid & 31) == 0) part_[tid >> 5] = tot_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t_ = 0.0f;
    for (int k_ = 0; k_ < 16; k_++) t_ += part_[k_];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t_;
}

// every step after the first, in one group: blocks of 512 partials are summed level by level (a and b alternate as source and destination, as
// the separate steps did) until at most 512 remain; then their sum, the mean, the exposure and the 1x1 R16F texel
kernel void ae_last(device float *a [[buffer(0)]], device float *b [[buffer(1)]], constant AeMid &m [[buffer(2)]], constant AeParams &p [[buffer(3)]],
                    texture2d<float, access::write> e_now [[texture(0)]],
                    uint tid [[thread_index_in_threadgroup]]) {
    SR_SAFE
    threadgroup float sh[512];
    threadgroup float part_[16];
    device float *in = a, *out = b;
    uint n = m.n;
    while (n > 512) {
        const uint nn = (n + 511) / 512;
        for (uint blk = 0; blk < nn; blk++) {
            const uint i = blk * 512 + tid;
            sh[tid] = i < n ? in[i] : 0.0f;
            const float t_ = ae_block_sum(sh, part_, tid);
            if (tid == 0) out[blk] = t_;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        device float *t = in; in = out; out = t;
        n = nn;
    }
    sh[tid] = tid < n ? in[tid] : 0.0f;
    const float tot = ae_block_sum(sh, part_, tid);
    if (tid == 0) {
        float mu = tot / (float)(p.lw * p.lh);
        float E = p.key_ae * precise::exp(-mu);
        e_now.write(float4((float)(half)E, 0.0f, 0.0f, 0.0f), uint2(0, 0));
    }
}

// P13: E_prev = E_now
kernel void ae_copy(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]]) {
    dst.write(src.read(uint2(0, 0)), uint2(0, 0));
}

// debug: sampled render pixel per grid element
kernel void ae_debug_index(device uint2 *out [[buffer(0)]], constant AeParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p.lw * p.lh) out[i] = ae_pixel(p, i);
}
