// Baseline (non-tensor) kernels for the encoder/bottleneck passes. Correctness first; the
// matrix-unit versions replace these per sr.metal_techniques.
#include <metal_stdlib>
using namespace metal;

static inline uchar m8_enc(half x) {
    ushort b = as_type<ushort>(x) & 0x7FFF;
    if (b > 0x7C00) return 0xFF;
    if (b < 0x400) return (uchar)(b >> 7);
    return (uchar)((b + 0x40) >> 7);
}

struct ConvParams {
    uint w, h;      // level size
    uint cin, cout;
    uint m8_in;     // input is m8 codes (fp16 bits = code << 7)
};

static inline half load_in(device const void *p, uint idx, uint m8) {
    if (m8) {
        ushort code = ((device const uchar *)p)[idx];
        return as_type<half>((ushort)(code << 7));
    }
    return ((device const half *)p)[idx];
}

// y = relu(rn16(conv3x3(x, W) + b)); channels-last; weights [cout][3][3][cin].
kernel void conv3x3_relu(device const void *x [[buffer(0)]],
                         device const half *wt [[buffer(1)]],
                         device const half *bias [[buffer(2)]],
                         device half *y [[buffer(3)]],
                         constant ConvParams &p [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.cout) return;
    float acc = 0.0f;
    for (uint ky = 0; ky < 3; ky++) {
        int yy = (int)gid.y + (int)ky - 1;
        if (yy < 0 || yy >= (int)p.h) continue;
        for (uint kx = 0; kx < 3; kx++) {
            int xx = (int)gid.x + (int)kx - 1;
            if (xx < 0 || xx >= (int)p.w) continue;
            uint xb = ((uint)yy * p.w + (uint)xx) * p.cin;
            uint wb = ((gid.z * 3 + ky) * 3 + kx) * p.cin;
            for (uint c = 0; c < p.cin; c++)
                acc = fma((float)load_in(x, xb + c, p.m8_in), (float)wt[wb + c], acc);
        }
    }
    half v = (half)(acc + (float)bias[gid.z]);
    y[(gid.y * p.w + gid.x) * p.cout + gid.z] = max(v, (half)0);
}

// 2x2 max pool (ceil sizes; out-of-range taps ignored), channels-last.
kernel void maxpool2x2(device const half *y [[buffer(0)]],
                       device half *q [[buffer(1)]],
                       constant ConvParams &p [[buffer(2)]],  // w,h,cout of the source level
                       uint3 gid [[thread_position_in_grid]]) {
    uint qw = (p.w + 1) / 2, qh = (p.h + 1) / 2;
    if (gid.x >= qw || gid.y >= qh || gid.z >= p.cout) return;
    half m = -INFINITY;
    for (uint dy = 0; dy < 2; dy++)
        for (uint dx = 0; dx < 2; dx++) {
            uint xx = gid.x * 2 + dx, yy = gid.y * 2 + dy;
            if (xx < p.w && yy < p.h) m = max(m, y[(yy * p.w + xx) * p.cout + gid.z]);
        }
    q[(gid.y * qw + gid.x) * p.cout + gid.z] = m;
}

// Per-pixel 1x1 convolution (scalar fp32 accumulation); weights [cout][cin].
// p.w = pixel count, p.h = 1; flags in p.m8_in: 1 = m8 input, 2 = relu, 4 = head epilogue
// (rn16(acc) then fp16 bias add), 8 = m8 output.
kernel void conv1x1(device const void *x [[buffer(0)]],
                    device const half *wt [[buffer(1)]],
                    device const half *bias [[buffer(2)]],
                    device void *y [[buffer(3)]],
                    constant ConvParams &p [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.cout) return;
    float acc = 0.0f;
    for (uint c = 0; c < p.cin; c++)
        acc = fma((float)load_in(x, gid.x * p.cin + c, p.m8_in & 1u), (float)wt[gid.y * p.cin + c], acc);
    half v;
    if (p.m8_in & 4u) v = (half)((half)acc + bias[gid.y]);
    else v = (half)(acc + (float)bias[gid.y]);
    if (p.m8_in & 2u) v = v > (half)0 ? v : (half)0;
    if (p.m8_in & 8u) ((device uchar *)y)[gid.x * p.cout + gid.y] = m8_enc(v);
    else ((device half *)y)[gid.x * p.cout + gid.y] = v;
}

