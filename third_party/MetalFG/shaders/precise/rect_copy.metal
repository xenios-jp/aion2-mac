// F28 rectangle copy (spec fg.orchestration): dst(d + p) = src(s + p), integer reads, all four components.
#include <metal_stdlib>
using namespace metal;

struct RectCopyParams {
    uint2 src_origin;
    uint2 dst_origin;
    uint2 size;
};

kernel void rect_copy(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      constant RectCopyParams &prm [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= prm.size.x || gid.y >= prm.size.y) return;
    uint2 s = prm.src_origin + gid;
    uint2 d = prm.dst_origin + gid;
    if (s.x >= src.get_width() || s.y >= src.get_height()) return;
    if (d.x >= dst.get_width() || d.y >= dst.get_height()) return;
    dst.write(src.read(s), d);
}

#if defined(METALFG_CTL_BLEND)
// Control (fg.test_scenes_r3): the generated frame is replaced by the per-channel mean of the previous and the current real frame.
kernel void ctl_blend(texture2d<float, access::read> prev [[texture(0)]], texture2d<float, access::read> cur [[texture(1)]],
                      texture2d<float, access::write> dst [[texture(2)]], constant RectCopyParams &prm [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= prm.size.x || gid.y >= prm.size.y) return;
    const uint2 d = prm.dst_origin + gid;
    if (d.x >= dst.get_width() || d.y >= dst.get_height()) return;
    dst.write(float4(0.5f * (prev.read(gid).rgb + cur.read(gid).rgb), 1.0f), d);
}
#endif
