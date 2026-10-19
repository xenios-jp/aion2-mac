// F15 net1 input packing and F17 net1 output unpacking (spec fg.net1_io). Precise math: sat of NaN must give zero.
#include "fgcommon.h"
using namespace metal;

constexpr sampler lin(filter::linear, address::clamp_to_edge, coord::normalized);

static inline float sat(float v) { return fmin(fmax(v, 0.0f), 1.0f); }   // min(max(v, 0), 1): a NaN gives 0
static inline float luma(float3 c) { return sat(lum3(c)); }   // rule J of fg.arithmetic_pin_r4

struct F15Params {
    uint Wf, Hf, w1, h1, ox, oy, Wm, Hm;
};

// One thread per position of the padded 2 w1 x 2 h1 area. Feature f of position (dx, dy) in N1 pixel (u, v) goes to
// channel 4 f + dx + 2 dy; margin positions and channels 40..47 are zero.
kernel void f15_pack(constant F15Params &P [[buffer(0)]],
                     device half *n1in [[buffer(1)]],
                     texture2d<float> Pc [[texture(0)]],      // col.feat.old
                     texture2d<float> Cc [[texture(1)]],      // col.feat.cur
                     texture2d<float, access::read> lamT [[texture(2)]],    // dY.feat
                     texture2d<float, access::read> qold [[texture(3)]],    // err.feat.old
                     texture2d<float, access::read> nsT [[texture(4)]],     // nsc.mot
                     texture2d<float> dT [[texture(5)]],      // dis.feat.cur
                     texture2d<float> doldT [[texture(6)]],   // dis.feat.old
                     texture2d<float, access::read> rT [[texture(7)]],      // st.feat
                     texture2d<float, access::read> mcpT [[texture(8)]],    // mvo.mot or mvod.mot
                     texture2d<float, access::read> mpcT [[texture(9)]],    // mvf.mot or mvfd.mot
                     texture2d<float, access::write> errOut [[texture(10)]],
                     texture2d<float, access::write> msOut [[texture(11)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 2 * P.w1 || gid.y >= 2 * P.h1) return;
    const uint u = gid.x / 2, v = gid.y / 2, dx = gid.x & 1, dy = gid.y & 1;
    device half *dst = n1in + ((ulong)v * P.w1 + u) * 48;
    const uint slot = dx + 2 * dy;
    if (slot == 0)
        for (uint c = 40; c < 48; c++) dst[c] = half(0);
    const int px = int(gid.x) - int(P.ox), py = int(gid.y) - int(P.oy);
    if (px < 0 || py < 0 || px >= int(P.Wf) || py >= int(P.Hf)) {
        for (uint f = 0; f < 10; f++) dst[4 * f + slot] = half(0);
        return;
    }
    const uint2 p = uint2(px, py);
    const float2 uv = (float2(p) + 0.5f) / float2(P.Wf, P.Hf);
    const float3 prev = Pc.read(p).rgb, cur = Cc.read(p).rgb;
    // fg.net1_io_r3: NS, mcp and mpc live on the Mot grid and are point reads of the texel floor(Q), Q = ((x + 1/2) Wm / Wf, (y + 1/2) Hm / Hf)
    // (a point on the line between two texels belongs to the upper one); the integer form below is floor(Q) exactly. No interpolation.
    const uint2 tm = min(((2u * p + 1u) * uint2(P.Wm, P.Hm)) / (2u * uint2(P.Wf, P.Hf)), uint2(P.Wm - 1u, P.Hm - 1u));
    const float2 mcp = mcpT.read(tm).xy;
    const float2 mpc = mpcT.read(tm).xy;
    const float3 pw = Pc.sample(lin, uv + mcp).rgb;
    const float3 cw = Cc.sample(lin, uv + mpc / float2(P.Wm, P.Hm)).rgb;
    const float3 ec3 = abs(pw - cur), ep3 = abs(cw - prev);
    const float Ec = sat((ec3.x + ec3.y + ec3.z) / 3.0f);
    const float Ep = sat((ep3.x + ep3.y + ep3.z) / 3.0f);
    const float ns = nsT.read(tm).x;
    const float dd = dT.sample(lin, uv).x, dold = doldT.sample(lin, uv).x;
    const float M = sat(ns * dold * dd);
    const float r = rT.read(p).x;
    float feat[10];
    feat[0] = lamT.read(p).x;
    feat[1] = luma(prev);
    feat[2] = Ep;
    feat[3] = qold.read(p).x;
    feat[4] = ns;
    feat[5] = luma(cur);
    feat[6] = Ec;
    feat[7] = dd;
    feat[8] = r;
    feat[9] = 1.0f - r;
    for (uint f = 0; f < 10; f++) dst[4 * f + slot] = half(feat[f]);
    errOut.write(float4(Ec, Ep, 0, 0), p);
    msOut.write(float4(M, 0, 0, 0), p);
}

struct F17Params {
    uint Wf, Hf, w1, h1, ox, oy;
    uint steering;
    float b_steer, th_mask;
#if defined(METALFG_CTL_DRIFT)
    int drift;
#endif
};

// One thread per Feat pixel: unfold the 16 channels of its N1 pixel, steer the logits, derive the recurrent state.
kernel void f17_unpack(constant F17Params &P [[buffer(0)]],
                       device const half *n1out [[buffer(1)]],
                       texture2d<float, access::read> Mt [[texture(0)]],        // ms.feat
                       texture2d<float, access::read> uimOld [[texture(1)]],    // uim.feat.old
                       texture2d<float, access::read> uimCur [[texture(2)]],    // uim.feat.cur
                       texture2d<float, access::write> lgP [[texture(3)]],
                       texture2d<float, access::write> lgC [[texture(4)]],
                       texture2d<float, access::write> stOut [[texture(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= P.Wf || gid.y >= P.Hf) return;
    const uint X = gid.x + P.ox, Y = gid.y + P.oy;
    const uint u = X / 2, v = Y / 2, dx = X & 1, dy = Y & 1;
    device const half *src = n1out + ((ulong)v * P.w1 + u) * 16;
    float k0 = float(src[2 * dy + dx]), k1 = float(src[4 + 2 * dy + dx]);
    float k2 = float(src[8 + 2 * dy + dx]), k3 = float(src[12 + 2 * dy + dx]);
    const float b = P.b_steer;
    const float m = Mt.read(gid).x;
    // 1. mask override
    k0 = fma(m, -b - k0, k0);                    // rule I: one fused multiply-add
    k2 = fma(m, -b - k2, k2);
    // 2. recurrent output, before the UI steering: 1 / (1 + exp(k3 - k2)) in the safe form
    // rule I: exp2(|k2 - k3| * -log2 e) with fast exp2, the numerator one when the first logit exceeds the second and the exponential otherwise, then fast divide
    const float e = fast::exp2(fabs(k2 - k3) * -1.44269502f);
    float r = fast::divide(k2 > k3 ? 1.0f : e, e + 1.0f);
    // 3. UI steering
    if (P.steering != 0) {
        if (uimOld.read(gid).x > P.th_mask) { k0 -= b; k1 += b; }
        if (uimCur.read(gid).x > P.th_mask) { k2 -= b; k3 += b; }
    }
    lgP.write(float4(k0, k1, 0, 0), gid);
    lgC.write(float4(k2, k3, 0, 0), gid);
#if defined(METALFG_CTL_DRIFT)
    // control of fg.arithmetic_tolerance_r2 tol-drift-ctl: after the state is formed, add P.drift ladder units Uk = 2^-11 (an absolute step) to every non-zero element with a fixed pseudo-random sign
    {
        if (r != 0.0f && P.drift != 0) {
            const bool neg = (((gid.x * 73856093u) ^ (gid.y * 19349663u)) >> 7) & 1u;
            r = clamp(r + (neg ? -1.0f : 1.0f) * float(P.drift) * 0x1p-11f, 0x1p-24f, 1.0f);
        }
    }
#endif
    stOut.write(float4(r, 0, 0, 0), gid);
}
