// Engine housekeeping kernels: clears, depth-format conversion, output-format conversion, completion marker.
#include <metal_stdlib>
#include "sr_math.h"
using namespace metal;

struct ClearParams { uint w, h; };

kernel void clear_tex(texture2d<float, access::write> t [[texture(0)]], constant ClearParams &p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x < p.w && g.y < p.h) t.write(float4(0.0f), g);
}

struct DepthParams { uint w, h, ox, oy; };

// any depth-format texture (Depth32Float, Depth32Float_Stencil8 depth plane, Depth16Unorm) -> R32Float at (0, 0)
kernel void depth_convert(depth2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                          constant DepthParams &p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x < p.w && g.y < p.h) dst.write(float4(src.read(g + uint2(p.ox, p.oy)), 0.0f, 0.0f, 0.0f), g);
}

struct OutParams { uint w, h, ox, oy; float maxv; };   // maxv: largest finite magnitude of the destination format

// internal RGBA16F result -> caller format (the store converts)
kernel void out_convert(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                        constant OutParams &p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    float4 v = src.read(g);
#ifdef SR_INJECT_NONFINITE
    // test builds only (control of tests/finite_gpu.m)
    { const uint h = g.x * 7u + g.y * 13u; if (h % 61u == 0u) v.x = INFINITY; if (h % 61u == 1u) v.y = NAN; }
#endif
#ifndef SR_NO_OUTPUT_GUARD
    v = float4(sr_guard32(v.x, p.maxv), sr_guard32(v.y, p.maxv), sr_guard32(v.z, p.maxv), sr_guard32(v.w, p.maxv));   // no infinity or NaN leaves the engine
#endif
    dst.write(v, g + uint2(p.ox, p.oy));
}

struct DoneParams { ulong value; };

// completion marker: written by the last dispatch of a call so the CPU can tell when its ring slot is free
kernel void done_write(device volatile ulong *counter [[buffer(0)]], constant DoneParams &p [[buffer(1)]]) {
    *counter = p.value;
}
