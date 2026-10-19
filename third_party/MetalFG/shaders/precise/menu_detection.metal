// F01 flag buffer clear, F02/F03 menu detection, F27 menu output override (spec fg.menu_detection).
#include <metal_stdlib>
using namespace metal;

// menu.state words: 0 active (W0), 1 detection of the previous real frame (W1), 2 and 3 unused (zero)
struct DetParams {
    uint gx, gy;
    uint cw, ch;         // content rectangle size
    uint ox, oy;         // origin in the colour, hudless and UI textures
    uint channel;        // F02: coverage channel (0 for a plane, 3 for the alpha of ui_colour)
    float mt, frac;      // coverage threshold, sample fraction
    uint has_flag;
};

static inline uint sample_pos(uint extent, uint i, uint n) {
    return (uint)(((ulong)(extent - 1) * i) / (n - 1));
}

constant float ME = 0x1p-30f;

// One threadgroup: each thread handles several samples, simdgroup counts, thread 0 decides and updates the state.
static inline void decide(threadgroup uint *partial, uint cnt_thread, uint tid, uint nthreads, uint lane, uint sg,
                          constant DetParams &P, device uint *state, device uchar *flag) {
    uint c = simd_sum(cnt_thread);
    if (lane == 0) partial[sg] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint total = 0;
        for (uint i = 0; i < (nthreads + 31) / 32; i++) total += partial[i];
        const float ns = float(P.gx * P.gy);
        const uint D = float(total) > ns * P.frac ? 1u : 0u;
        const uint w1_old = state[1];
        state[1] = D;
        state[0] = (D != 0 || w1_old != 0) ? 1u : 0u;
        state[2] = 0;
        state[3] = 0;
        if (state[0] != 0 && P.has_flag) flag[0] = 1;
    }
}

kernel void f02_menu_alpha(constant DetParams &P [[buffer(0)]],
                           device uint *state [[buffer(1)]],
                           device uchar *flag [[buffer(2)]],
                           texture2d<float, access::read> cov [[texture(0)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint nthreads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial[32];
    uint cnt = 0;
    for (uint s = tid; s < P.gx * P.gy; s += nthreads) {
        const uint i = s % P.gx, j = s / P.gx;
        const uint2 q = uint2(P.ox + sample_pos(P.cw, i, P.gx), P.oy + sample_pos(P.ch, j, P.gy));
        const float a = cov.read(q)[P.channel];
        cnt += (a > P.mt) ? 1u : 0u;      // NaN never counts
    }
    decide(partial, cnt, tid, nthreads, lane, sg, P, state, flag);
}

kernel void f03_menu_hudless(constant DetParams &P [[buffer(0)]],
                             device uint *state [[buffer(1)]],
                             device uchar *flag [[buffer(2)]],
                             texture2d<float, access::read> col [[texture(0)]],
                             texture2d<float, access::read> hud [[texture(1)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint nthreads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial[32];
    uint cnt = 0;
    for (uint s = tid; s < P.gx * P.gy; s += nthreads) {
        const uint i = s % P.gx, j = s / P.gx;
        const uint2 q = uint2(P.ox + sample_pos(P.cw, i, P.gx), P.oy + sample_pos(P.ch, j, P.gy));
        const float3 C = col.read(q).rgb, H = hud.read(q).rgb;
        float a = -INFINITY;
        for (uint c = 0; c < 3; c++) {
            const float d = H[c] - C[c];
            const float t1 = d / (H[c] + ME), t2 = -d / (1.0f + ME - H[c]);
            a = fmax(a, fmax(t1, t2));      // a NaN term is ignored by the maximum
        }
        cnt += (a > P.mt) ? 1u : 0u;
    }
    decide(partial, cnt, tid, nthreads, lane, sg, P, state, flag);
}

kernel void f01_flag_clear(device uchar *flag [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) flag[0] = 0;
}

struct OverrideParams { uint2 src_origin, dst_origin, size; };

// F27: when the active word is non-zero, copy the colour content rectangle over the display rectangle.
kernel void f27_menu_override(constant OverrideParams &P [[buffer(0)]],
                              device const uint *state [[buffer(1)]],
                              texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (state[0] == 0) return;
    if (gid.x >= P.size.x || gid.y >= P.size.y) return;
    const uint2 s = P.src_origin + gid, d = P.dst_origin + gid;
    if (s.x >= src.get_width() || s.y >= src.get_height() || d.x >= dst.get_width() || d.y >= dst.get_height()) return;
    dst.write(src.read(s), d);
}
