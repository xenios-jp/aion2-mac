/* CPU reference of the network passes P2-P11 (float64 accumulation, fp16 steps emulated), plus test weight files. */
#ifndef SR_REF_H
#define SR_REF_H
#include <stdlib.h>
#include "numerics.h"
static uint64_t rng_ = 0x2545F4914F6CDD1Dull;
static inline double urand_(void) { rng_ ^= rng_ << 13; rng_ ^= rng_ >> 7; rng_ ^= rng_ << 17; return (rng_ >> 11) * (1.0 / 9007199254740992.0); }
static inline double nrand_(void) { return sqrt(-2 * log(urand_() + 1e-300)) * cos(6.283185307179586 * urand_()); }

typedef struct { int Wp, Hp, W[6], H[6]; int wide; } RefLevels;
static inline void ref_levels(int Wo, int Ho, RefLevels *L) {
    L->wide = 0; L->Wp = 128 * ((Wo + 127) / 128); L->Hp = 128 * ((Ho + 127) / 128);
    for (int k = 0; k < 6; k++) { L->W[k] = L->Wp >> (k + 2); L->H[k] = L->Hp >> (k + 2); }
}

/* ---- weight images ---- */
static inline const f16 *wt_of(const uint8_t *img, int idx) { size_t off[SR_NUM_TENSORS]; sr_tensor_offsets(off); return (const f16 *)(img + off[idx]); }
enum { WMODE_ZERO, WMODE_IDENTITY, WMODE_RANDOM };
static inline uint8_t *make_weights(int mode, double std_scale) {
    size_t off[SR_NUM_TENSORS];
    size_t total = sr_tensor_offsets(off);
    uint8_t *img = calloc(1, total);
    for (int t = 0; t < SR_NUM_TENSORS; t += 2) {
        const SrTensorShape *s = &sr_tensor_shapes[t];
        f16 *w = (f16 *)(img + off[t]), *b = (f16 *)(img + off[t + 1]);
        size_t K = (size_t)s->k * s->k * s->i;
        if (mode == WMODE_IDENTITY) {
            for (uint32_t o = 0; o < s->o && o < s->i; o++) w[(size_t)o * K + (size_t)(s->k / 2 * s->k + s->k / 2) * s->i + o] = (f16)1.0f;
        } else if (mode == WMODE_RANDOM) {
            for (size_t i = 0; i < (size_t)s->o * K; i++) w[i] = (f16)(std_scale * 0.7 / sqrt((double)K) * nrand_());
            for (uint32_t o = 0; o < s->o; o++) b[o] = (f16)(-0.05 + 0.1 * urand_());
        }
    }
    return img;
}

/* ---- layers ---- */
/* y = relu(rn16(conv3x3(x, w) + b)); x [H][W][cin] fp16 */
static inline void ref_conv3x3(const f16 *x, int W, int H, int cin, const f16 *w, const f16 *b, int cout, f16 *y) {
    for (int py = 0; py < H; py++) for (int px = 0; px < W; px++) for (int o = 0; o < cout; o++) {
        double acc = 0;
        for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
            int yy = py + ky - 1, xx = px + kx - 1;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            const f16 *xp = x + ((size_t)yy * W + xx) * cin, *wp = w + (((size_t)o * 3 + ky) * 3 + kx) * cin;
            for (int c = 0; c < cin; c++) acc += (double)xp[c] * (double)wp[c];
        }
        f16 v = (f16)(acc + (double)b[o]);
        y[((size_t)py * W + px) * cout + o] = v > (f16)0 ? v : (f16)0;
    }
}
static inline void ref_maxpool(const f16 *y, int W, int H, int C, f16 *q) {
    int qw = (W + 1) / 2, qh = (H + 1) / 2;
    for (int py = 0; py < qh; py++) for (int px = 0; px < qw; px++) for (int c = 0; c < C; c++) {
        f16 m = (f16)-INFINITY;
        for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            int yy = py * 2 + dy, xx = px * 2 + dx;
            if (yy < H && xx < W && y[((size_t)yy * W + xx) * C + c] > m) m = y[((size_t)yy * W + xx) * C + c];
        }
        q[((size_t)py * qw + px) * C + c] = m;
    }
}
/* flags: 2 relu, 4 head epilogue (rn16(acc) first, then fp16 bias add) */
static inline void ref_conv1x1(const f16 *x, int npix, int cin, const f16 *w, const f16 *b, int cout, f16 *y, int flags) {
    for (int p = 0; p < npix; p++) for (int o = 0; o < cout; o++) {
        double acc = 0;
        for (int c = 0; c < cin; c++) acc += (double)x[(size_t)p * cin + c] * (double)w[(size_t)o * cin + c];
        f16 v = (flags & 4) ? (f16)((double)(f16)acc + (double)b[o]) : (f16)(acc + (double)b[o]);
        if ((flags & 2) && !(v > (f16)0)) v = 0;
        y[(size_t)p * cout + o] = v;
    }
}
typedef struct { const f16 *p; int W, C; } RefTensor;
static inline f16 ref_fetch(const void *ctx, int c, int y, int x) { const RefTensor *t = ctx; return t->p[((size_t)y * t->W + x) * t->C + c]; }
/* x = rn16(up2(coarse) + skip) at the fine level */
static inline void ref_up2_add(const f16 *coarse, int wc, int hc, int C, const f16 *skip, int variant, f16 *x) {
    RefTensor t = {coarse, wc, C};
    for (int Y = 0; Y < 2 * hc; Y++) for (int X = 0; X < 2 * wc; X++) for (int c = 0; c < C; c++) {
        f16 up = up2_at(&t, ref_fetch, wc, hc, c, X, Y, variant);
        x[((size_t)Y * 2 * wc + X) * C + c] = (f16)((double)up + (double)skip[((size_t)Y * 2 * wc + X) * C + c]);
    }
}
static inline void m8_decode_buf(const uint8_t *m, size_t n, f16 *o) { for (size_t i = 0; i < n; i++) o[i] = (f16)m8_decode(m[i]); }
static inline void m8_encode_buf_cpu(const f16 *x, size_t n, uint8_t *m) { for (size_t i = 0; i < n; i++) m[i] = m8_encode(x[i]); }

/* ---- passes ---- */
/* P2: feat [H0][W0][16] -> s0 (m8 32@L0), s1 (m8 64@L1), q2 (m8 64@L2) */
static inline void ref_p2(const uint8_t *img, const RefLevels *L, const f16 *feat, uint8_t *s0, uint8_t *s1, uint8_t *q2) {
    int W0 = L->W[0], H0 = L->H[0], W1 = L->W[1], H1 = L->H[1], W2 = L->W[2], H2 = L->H[2];
    f16 *A = malloc((size_t)W0 * H0 * 32 * 2), *PA = malloc((size_t)W1 * H1 * 32 * 2), *B = malloc((size_t)W1 * H1 * 64 * 2), *C = malloc((size_t)W2 * H2 * 64 * 2);
    ref_conv3x3(feat, W0, H0, 16, wt_of(img, 0), wt_of(img, 1), 32, A);
    m8_encode_buf_cpu(A, (size_t)W0 * H0 * 32, s0);
    ref_maxpool(A, W0, H0, 32, PA);
    ref_conv1x1(PA, W1 * H1, 32, wt_of(img, 2), wt_of(img, 3), 64, B, 2);
    m8_encode_buf_cpu(B, (size_t)W1 * H1 * 64, s1);
    ref_maxpool(B, W1, H1, 64, C);
    m8_encode_buf_cpu(C, (size_t)W2 * H2 * 64, q2);
    free(A); free(PA); free(B); free(C);
}
/* decoder pass 8..11: coarse/skip fp16 (already decoded), result fp16 before any m8 encode */
static inline void ref_dec(const uint8_t *img, const RefLevels *L, int pass, int variant, const f16 *coarse, const f16 *skip, f16 *out) {
    static const int lvl[12] = {0,0,0,0,0,0,0,0,4,3,2,1}, c0[12] = {0,0,0,0,0,0,0,0,256,128,128,64},
                     c1[12] = {0,0,0,0,0,0,0,0,256,128,128,64}, c2[12] = {0,0,0,0,0,0,0,0,128,128,64,32}, w0[12] = {0,0,0,0,0,0,0,0,14,18,22,26};
    int k = lvl[pass], npix = L->W[k] * L->H[k];
    f16 *x = malloc((size_t)npix * c0[pass] * 2), *h = malloc((size_t)npix * c1[pass] * 2);
    ref_up2_add(coarse, L->W[k + 1], L->H[k + 1], c0[pass], skip, variant, x);
    ref_conv1x1(x, npix, c0[pass], wt_of(img, w0[pass]), wt_of(img, w0[pass] + 1), c1[pass], h, 2);
    ref_conv1x1(h, npix, c1[pass], wt_of(img, w0[pass] + 2), wt_of(img, w0[pass] + 3), c2[pass], out, 2);
    free(x); free(h);
}
#endif
