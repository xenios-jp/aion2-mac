// FH2 clears and sentinel fills (spec fg.resources "Interoperability: sentinel and start values").
#include <metal_stdlib>
using namespace metal;

struct FillParams {
    float4 value;
    uint count;
    uint ivalue;
};

kernel void fill_tex(texture2d<float, access::write> dst [[texture(0)]],
                     constant FillParams &prm [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(prm.value, gid);
}

kernel void fill_tex_u32(texture2d<uint, access::write> dst [[texture(0)]],
                         constant FillParams &prm [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(uint4(prm.ivalue, 0, 0, 0), gid);
}

kernel void fill_buf_u32(device uint *dst [[buffer(1)]],
                         constant FillParams &prm [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= prm.count) return;
    dst[gid] = prm.ivalue;
}

#if defined(METALFG_CTL_PERF_SPIN)
// Control 1 of fg.performance_r1: a dependent chain of n fused multiply-adds per thread (one simdgroup), so the dispatch lasts a fixed time.
kernel void ctl_spin(constant uint &n [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float a = float(i) * 1e-6f;
    for (uint k = 0; k < n; k++) a = fma(a, 1.0000001f, 1e-7f);
    o[i] = a;
}
#endif
