// Shared helpers: m8 codec, pooling, bilinear 2x upsample + skip add (sr.numerics).
#include <metal_stdlib>
using namespace metal;

static inline half m8_dec(uchar c) { return as_type<half>((ushort)((ushort)c << 7)); }
static inline uchar m8_enc(half x) {
    ushort b = as_type<ushort>(x) & 0x7FFF;
    if (b > 0x7C00) return 0xFF;
    if (b < 0x400) return (uchar)(b >> 7);
    return (uchar)((b + 0x40) >> 7);
}

struct M8Params { uint count; };

// fp16 -> m8 (storage form of s0, s1, q2, d2, d1)
kernel void m8_encode_buf(device const half *x [[buffer(0)]], device uchar *y [[buffer(1)]],
                          constant M8Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = m8_enc(x[i]);
}

// test hook: decode / encode of arbitrary bit patterns
kernel void m8_test_enc(device const ushort *x [[buffer(0)]], device uchar *y [[buffer(1)]],
                        constant M8Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) y[i] = m8_enc(as_type<half>(x[i]));
}

struct PoolM8Params { uint w, h, c; };
// s1 = E(B) and q2 = E(P(B)) from the fp16 level-1 tensor B (w, h = level 1 size, c a multiple of 8);
// one thread per pooled pixel and 8-channel group: grid (w/2, h/2, c/8)
kernel void pool_m8(device const half *b [[buffer(0)]], device uchar *s1 [[buffer(1)]], device uchar *q2 [[buffer(2)]],
                    constant PoolM8Params &p [[buffer(3)]], uint3 g [[thread_position_in_grid]]) {
    if (g.x >= p.w / 2 || g.y >= p.h / 2 || g.z >= p.c / 8) return;
    half4 mlo = half4(-INFINITY), mhi = half4(-INFINITY);
    for (uint dy = 0; dy < 2; dy++)
        for (uint dx = 0; dx < 2; dx++) {
            uint i = ((g.y * 2 + dy) * p.w + g.x * 2 + dx) * p.c + g.z * 8;
            half4 lo = *(device const half4 *)(b + i), hi = *(device const half4 *)(b + i + 4);
            ulong raw = 0;
            for (int c = 0; c < 4; c++) raw |= ((ulong)m8_enc(lo[c]) << (8 * c)) | ((ulong)m8_enc(hi[c]) << (8 * (c + 4)));
            *(device ulong *)(s1 + i) = raw;
            mlo = max(mlo, lo); mhi = max(mhi, hi);
        }
    ulong raw = 0;
    for (int c = 0; c < 4; c++) raw |= ((ulong)m8_enc(mlo[c]) << (8 * c)) | ((ulong)m8_enc(mhi[c]) << (8 * (c + 4)));
    *(device ulong *)(q2 + (g.y * (p.w / 2) + g.x) * p.c + g.z * 8) = raw;
}

struct UpParams {
    uint wc, hc;     // coarse size (pixels)
    uint c;          // channels
    uint variant;    // 0 = N, 1 = W
    uint coarse_m8, skip_m8;
};

static inline half lerp16(uint variant, half n, half f) {
    if (variant == 0) return fma((half)0.75h, (half)(n - f), f);
    return fma((half)0.75h, n, (half)(f * (half)0.25h));
}

static inline half fetch_h(device const void *p, uint idx, uint m8) {
    return m8 ? m8_dec(((device const uchar *)p)[idx]) : ((device const half *)p)[idx];
}

// x = rn16(up2(coarse) + skip) at the fine level (2*wc x 2*hc), channels-last. Grid: (2*wc, 2*hc, c).
kernel void up2_add(device const void *coarse [[buffer(0)]],
                    device const void *skip [[buffer(1)]],
                    device half *x [[buffer(2)]],
                    constant UpParams &p [[buffer(3)]],
                    uint3 g [[thread_position_in_grid]]) {
    uint W = p.wc * 2, H = p.hc * 2;
    if (g.x >= W || g.y >= H || g.z >= p.c) return;
    int X = (int)g.x, Y = (int)g.y;
    int nx = X >> 1, ny = Y >> 1;
    int fx = (X & 1) ? nx + 1 : nx - 1, fy = (Y & 1) ? ny + 1 : ny - 1;
    fx = clamp(fx, 0, (int)p.wc - 1); fy = clamp(fy, 0, (int)p.hc - 1);
    uint c = g.z;
    half a = fetch_h(coarse, ((uint)ny * p.wc + (uint)nx) * p.c + c, p.coarse_m8);
    half b = fetch_h(coarse, ((uint)ny * p.wc + (uint)fx) * p.c + c, p.coarse_m8);
    half d = fetch_h(coarse, ((uint)fy * p.wc + (uint)nx) * p.c + c, p.coarse_m8);
    half e = fetch_h(coarse, ((uint)fy * p.wc + (uint)fx) * p.c + c, p.coarse_m8);
    half hn = lerp16(p.variant, a, b), hf = lerp16(p.variant, d, e);
    half up = lerp16(p.variant, hn, hf);
    half s = fetch_h(skip, ((uint)Y * W + (uint)X) * p.c + c, p.skip_m8);
    x[((uint)Y * W + (uint)X) * p.c + c] = (half)(up + s);
}
