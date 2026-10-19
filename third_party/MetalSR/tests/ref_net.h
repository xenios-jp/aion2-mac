/* CPU model of one whole engine call (all passes), with persistent state, for the end-to-end tests. */
#ifndef SR_REF_NET_H
#define SR_REF_NET_H
#include "ref_p1.h"
#include "ref_p12.h"
#ifndef REF_SUB_BITS
#define REF_SUB_BITS 8   /* fractional bits of the sub-texel weights of the GPU samplers */
#endif
#include "../src/frame.h"

typedef struct {
    int Wo, Ho, W0, H0;
    f16 *hist, *mem[2];
    f16 e_now, e_prev;
    uint64_t call;
} RefState;

static inline RefState *ref_state_new(int Wo, int Ho) {
    RefState *s = calloc(1, sizeof *s);
    RefLevels L; ref_levels(Wo, Ho, &L);
    s->Wo = Wo; s->Ho = Ho; s->W0 = L.W[0]; s->H0 = L.H[0];
    s->hist = calloc((size_t)Wo * Ho * 4, 2);
    s->mem[0] = calloc((size_t)Wo * Ho * 2, 2); s->mem[1] = calloc((size_t)Wo * Ho * 2, 2);
    return s;
}
static inline void ref_state_free(RefState *s) { free(s->hist); free(s->mem[0]); free(s->mem[1]); free(s); }

static inline double ref_ae(const AeParams *p, const float *rgba) {
    double sum = 0;
    for (unsigned i = 0; i < p->lw * p->lh; i++) {
        unsigned a = i % p->lw, b = i / p->lw;
        float fx = (float)a / (float)(p->lw - 1), fy = (float)b / (float)(p->lh - 1);
        volatile float px = fx * (float)(p->Wr - 1), py = fy * (float)(p->Hr - 1);
        volatile float qx = px + 0.5f, qy = py + 0.5f;
        const float *c = rgba + ((size_t)(unsigned)truncf(qy) * p->Wr + (unsigned)truncf(qx)) * 4;
        float L = p->invp * lumaY(c[0], c[1], c[2]);
        if (isfinite(L) && L > 0) sum += log2((double)(L + p->eps_luma)) * M_LN2;
    }
    return (double)p->key_ae * exp(-sum / ((double)p->lw * p->lh));
}

static inline void m8_roundtrip(f16 *x, size_t n) { for (size_t i = 0; i < n; i++) x[i] = (f16)m8_decode(m8_encode(x[i])); }

/* exposure_in: caller exposure value (used when auto exposure is off); out: Wo*Ho*4 f16 */
static inline void ref_frame(RefState *st, const uint8_t *img, const SrParams *mp, const SrCall *c, int autoexp, float exposure_in,
                             const float *col, const float *depth, const float *mot, f16 *out) {
    int Wo = c->Wo, Ho = c->Ho, var = c->wide;
    RefLevels L; ref_levels(Wo, Ho, &L);
    int first = st->call == 0, reset = first || c->reset;
    SrCall cc = *c; cc.reset = reset; cc.W0 = L.W[0];
    if (reset) { memset(st->hist, 0, (size_t)Wo * Ho * 8); memset(st->mem[0], 0, (size_t)Wo * Ho * 4); memset(st->mem[1], 0, (size_t)Wo * Ho * 4); }
    if (c->hdr && autoexp) {
        if (st->call % 2 == 0) { AeParams ap; sr_make_ae(&cc, mp, &ap); st->e_now = (f16)ref_ae(&ap, col); }
    } else st->e_now = (f16)exposure_in;
    size_t n[6]; for (int k = 0; k < 6; k++) n[k] = (size_t)L.W[k] * L.H[k];
    FeatParams p1; sr_make_feat(&cc, mp, &p1);
    f16 *feat = calloc(n[0] * 16, 2), *R = calloc((size_t)Wo * Ho * 4, 2);
    f16 *mr = st->mem[st->call % 2], *mw = st->mem[(st->call + 1) % 2];
    RefP1Opts p1o = {0, 0, 0, 0, 0, REF_SUB_BITS, 1};
    RefP1In p1in = {col, depth, mot, st->hist, mr, (float)st->e_now, (float)st->e_prev};
    ref_p1(&p1, &p1o, &p1in, L.H[0], 1, feat, R, NULL);
    ref_p1_memory(&p1, &p1o, &p1in, mw);
    /* network */
    uint8_t *s0 = malloc(n[0] * 32), *s1 = malloc(n[1] * 64), *q2 = malloc(n[2] * 64);
    ref_p2(img, &L, feat, s0, s1, q2);
    f16 *q2d = malloc(n[2] * 128), *s2 = malloc(n[2] * 256), *q3 = malloc(n[3] * 256), *s3 = malloc(n[3] * 256), *q4 = malloc(n[4] * 256), *s4 = malloc(n[4] * 512), *q5 = malloc(n[5] * 512);
    m8_decode_buf(q2, n[2] * 64, q2d);
    ref_conv3x3(q2d, L.W[2], L.H[2], 64, wt_of(img, 4), wt_of(img, 5), 128, s2); ref_maxpool(s2, L.W[2], L.H[2], 128, q3);
    ref_conv3x3(q3, L.W[3], L.H[3], 128, wt_of(img, 6), wt_of(img, 7), 128, s3); ref_maxpool(s3, L.W[3], L.H[3], 128, q4);
    ref_conv3x3(q4, L.W[4], L.H[4], 128, wt_of(img, 8), wt_of(img, 9), 256, s4); ref_maxpool(s4, L.W[4], L.H[4], 256, q5);
    f16 *za = malloc(n[5] * 1024), *zb = malloc(n[5] * 512);
    ref_conv1x1(q5, (int)n[5], 256, wt_of(img, 10), wt_of(img, 11), 512, za, 2);
    ref_conv1x1(za, (int)n[5], 512, wt_of(img, 12), wt_of(img, 13), 256, zb, 2);
    f16 *d4 = malloc(n[4] * 256), *d3 = malloc(n[3] * 256), *d2 = malloc(n[2] * 128), *d1 = malloc(n[1] * 64);
    ref_dec(img, &L, 8, var, zb, s4, d4);
    ref_dec(img, &L, 9, var, d4, s3, d3);
    ref_dec(img, &L, 10, var, d3, s2, d2); m8_roundtrip(d2, n[2] * 64);
    f16 *s1d = malloc(n[1] * 128); m8_decode_buf(s1, n[1] * 64, s1d);
    ref_dec(img, &L, 11, var, d2, s1d, d1); m8_roundtrip(d1, n[1] * 32);
    f16 *s0d = malloc(n[0] * 64), *x = malloc(n[0] * 64), *h = malloc(n[0] * 64), *o = malloc(n[0] * 80);
    m8_decode_buf(s0, n[0] * 32, s0d);
    ref_up2_add(d1, L.W[1], L.H[1], 32, s0d, var, x);
    ref_conv1x1(x, (int)n[0], 32, wt_of(img, 30), wt_of(img, 31), 32, h, 2 | 4);
    ref_conv1x1(h, (int)n[0], 32, wt_of(img, 32), wt_of(img, 33), 40, o, 4);
    FiltParams p12; sr_make_filt(&cc, mp, &p12); p12.ox = p12.oy = 0; RefP12Opts p12o = {0};
    f16 *hn = malloc((size_t)Wo * Ho * 8);
    ref_p12(&p12, &p12o, L.W[0], o, col, R, (float)st->e_now, 1, out, hn);
    memcpy(st->hist, hn, (size_t)Wo * Ho * 8);
    st->e_prev = st->e_now;
    st->call++;
    free(feat); free(R); free(s0); free(s1); free(q2); free(q2d); free(s2); free(q3); free(s3); free(q4); free(s4); free(q5);
    free(za); free(zb); free(d4); free(d3); free(d2); free(d1); free(s1d); free(s0d); free(x); free(h); free(o); free(hn);
}
#endif
