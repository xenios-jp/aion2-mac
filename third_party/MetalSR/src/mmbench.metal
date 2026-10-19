// mm-peak microbenchmark kernels (sr.metal_techniques): matmul2d building block, fp16 operands, fp32 accumulation.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct MmBenchParams { uint iters; };

template <int K>
static inline void mm_peak_impl(device float *sink, constant MmBenchParams &p, uint tid, uint sg, uint tgid, threadgroup half *smem) {
    // per simdgroup: A tile [32][K], B tile [32][K] (B transposed view), both in threadgroup memory
    threadgroup half *a = smem + sg * 2 * 32 * K, *b = a + 32 * K;
    for (uint i = tid % 32; i < 32 * K; i += 32) { a[i] = (half)((i * 7) % 13) * (half)0.01h; b[i] = (half)((i * 5) % 11) * (half)0.01h; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    using AT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    constexpr auto desc = matmul2d_descriptor(32, 32, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    AT A(a, dextents<int32_t, 2>(K, 32)), B(b, dextents<int32_t, 2>(K, 32));
    auto cT = op.template get_destination_cooperative_tensor<AT, AT, float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint it = 0; it < p.iters; it++) op.run(A, B, cT);
    float s = 0;
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) s += cT[i];
    if (s == 123456.0f) sink[tgid] = s;     // keeps the loop alive
}

kernel void mm_peak32(device float *sink [[buffer(0)]], constant MmBenchParams &p [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup half smem[4 * 2 * 32 * 32];
    mm_peak_impl<32>(sink, p, tid, sg, tgid, smem);
}
kernel void mm_peak16(device float *sink [[buffer(0)]], constant MmBenchParams &p [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup half smem[4 * 2 * 32 * 32];
    mm_peak_impl<16>(sink, p, tid, sg, tgid, smem);
}
