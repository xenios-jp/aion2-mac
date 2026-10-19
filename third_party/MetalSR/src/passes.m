#import "passes.h"

typedef struct { uint32_t w, h, cin, cout, flags; } ConvParams;
typedef struct { uint32_t npix, cin, cout, flags, ncols; } MmParams;
typedef struct { uint32_t count; } M8Params;
typedef struct { uint32_t wc, hc, c, variant, coarse_m8, skip_m8; } UpParams;

void sr_levels(int Wo, int Ho, SrLevels *L) {
    L->wide = 0;
    L->Wp = 128 * ((Wo + 127) / 128);
    L->Hp = 128 * ((Ho + 127) / 128);
    for (int k = 0; k < 6; k++) { L->W[k] = L->Wp >> (k + 2); L->H[k] = L->Hp >> (k + 2); }
}

size_t sr_scratch_bytes(const SrLevels *L, int which) {
    size_t l0 = (size_t)L->W[0] * L->H[0];
    return l0 * (which == 2 ? 32 : 64);
}

static MTLSize T8x4(void) { return MTLSizeMake(8, 4, 1); }

void sr_conv3x3_relu(SrEnc *e, SrGpu *g, id<MTLBuffer> x, int x_m8, const SrWeights *w, int widx, id<MTLBuffer> y, int W, int H, int cin, int cout) {
    ConvParams p = {(uint32_t)W, (uint32_t)H, (uint32_t)cin, (uint32_t)cout, (uint32_t)x_m8};
    SrArg a[5] = {ARG_BUF(x), ARG_BUF_OFF(w->buf, w->off[widx]), ARG_BUF_OFF(w->buf, w->off[widx + 1]), ARG_BUF(y), ARG_VAL(p)};
    if (cin % 32 == 0 && cout % 32 == 0) {
        [e dispatch:[g pipeline:@"conv3x3_mm"] args:a count:5 threadgroups:MTLSizeMake((W + 15) / 16, H, cout / 32) threads:MTLSizeMake(32, 1, 1)];
    } else {
        [e dispatchGrid:MTLSizeMake(W, H, cout) pso:[g pipeline:@"conv3x3_relu"] args:a count:5 threads:T8x4()];
    }
}

void sr_maxpool(SrEnc *e, SrGpu *g, id<MTLBuffer> y, id<MTLBuffer> q, int W, int H, int C) {
    ConvParams p = {(uint32_t)W, (uint32_t)H, 0, (uint32_t)C, 0};
    SrArg a[3] = {ARG_BUF(y), ARG_BUF(q), ARG_VAL(p)};
    [e dispatchGrid:MTLSizeMake((W + 1) / 2, (H + 1) / 2, C) pso:[g pipeline:@"maxpool2x2"] args:a count:3 threads:T8x4()];
}

void sr_conv1x1(SrEnc *e, SrGpu *g, id<MTLBuffer> x, const SrWeights *w, int widx, id<MTLBuffer> y, int npix, int cin, int cout, int flags) {
    SrArg a[5];
    a[0] = ARG_BUF(x); a[1] = ARG_BUF_OFF(w->buf, w->off[widx]); a[2] = ARG_BUF_OFF(w->buf, w->off[widx + 1]); a[3] = ARG_BUF(y);
    if (flags == 2 && cout % 32 == 0 && cin % 16 == 0) {
        MmParams p = {(uint32_t)npix, (uint32_t)cin, (uint32_t)cout, 1, (uint32_t)cout};
        a[4] = ARG_VAL(p);
        [e dispatch:[g pipeline:@"conv1x1_mm"] args:a count:5 threadgroups:MTLSizeMake(cout / 32, (npix + 15) / 16, 1) threads:MTLSizeMake(32, 1, 1)];
    } else {
        ConvParams p = {(uint32_t)npix, 1, (uint32_t)cin, (uint32_t)cout, (uint32_t)flags};
        a[4] = ARG_VAL(p);
        [e dispatchGrid:MTLSizeMake(npix, cout, 1) pso:[g pipeline:@"conv1x1"] args:a count:5 threads:MTLSizeMake(8, 8, 1)];
    }
}

void sr_m8_encode(SrEnc *e, SrGpu *g, id<MTLBuffer> x, id<MTLBuffer> y, int count) {
    M8Params p = {(uint32_t)count};
    SrArg a[3] = {ARG_BUF(x), ARG_BUF(y), ARG_VAL(p)};
    [e dispatchGrid:MTLSizeMake(count, 1, 1) pso:[g pipeline:@"m8_encode_buf"] args:a count:3 threads:MTLSizeMake(256, 1, 1)];
}

void sr_up2_add(SrEnc *e, SrGpu *g, id<MTLBuffer> coarse, int coarse_m8, id<MTLBuffer> skip, int skip_m8, id<MTLBuffer> x,
                int wc, int hc, int c, int variant) {
    UpParams p = {(uint32_t)wc, (uint32_t)hc, (uint32_t)c, (uint32_t)variant, (uint32_t)coarse_m8, (uint32_t)skip_m8};
    SrArg a[4] = {ARG_BUF(coarse), ARG_BUF(skip), ARG_BUF(x), ARG_VAL(p)};
    [e dispatchGrid:MTLSizeMake(2 * wc, 2 * hc, c) pso:[g pipeline:@"up2_add"] args:a count:4 threads:MTLSizeMake(8, 4, 4)];
}

/* fused 3x3 conv + ReLU + pool on the matrix units; kernel selected by pass (2: P2 first layer, 3..5) */
static void conv_tile(SrEnc *e, SrGpu *g, const SrWeights *w, int widx, int pass, int wide, id<MTLBuffer> x, id<MTLBuffer> skip, id<MTLBuffer> pooled, int W, int H, int cin, int cout) {
    static NSString *const namesN[6] = {@"", @"", @"conv_tile_p2", @"conv_tile_p3", @"conv_tile_p4", @"conv_tile_p5"};
    static NSString *const namesW[6] = {@"", @"", @"conv_tile_p2", @"conv_tile_w3", @"conv_tile_w4", @"conv_tile_w5"};
    NSString *const *names = (wide && pass >= 3) ? namesW : namesN;
    static const int TWN[6] = {16, 16, 16, 32, 32, 16}, TWW[6] = {16, 16, 16, 16, 16, 16}, THN[6] = {0, 0, 8, 4, 4, 4}, TNN[6] = {0, 0, 32, 64, 32, 32}, THW[6] = {0, 0, 8, 4, 4, 2}, TNW[6] = {0, 0, 32, 64, 64, 32};
    const int *TH = (wide && pass >= 3) ? THW : THN, *TN = (wide && pass >= 3) ? TNW : TNN, *TW = (wide && pass >= 3) ? TWW : TWN;
    struct { uint32_t w, h, cin, cout; } p = {(uint32_t)W, (uint32_t)H, (uint32_t)cin, (uint32_t)cout};
    SrArg a[6] = {ARG_BUF(x), ARG_BUF_OFF(w->buf, w->off[widx]), ARG_BUF_OFF(w->buf, w->off[widx + 1]), ARG_BUF(skip), ARG_BUF(pooled), ARG_VAL(p)};
    [e dispatch:[g pipeline:names[pass]] args:a count:6 threadgroups:MTLSizeMake((W + TW[pass] - 1) / TW[pass], (H + TH[pass] - 1) / TH[pass], cout / TN[pass]) threads:MTLSizeMake(32 * TH[pass], 1, 1)];
}

void sr_p2(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> feat, id<MTLBuffer> s0, id<MTLBuffer> s1, id<MTLBuffer> q2, SrScratch s) {
    /* one fused kernel: conv 16 -> 32, pool, conv 32 -> 64, pool; the intermediate tensors stay in threadgroup memory */
    struct { uint32_t w, h; } p = {(uint32_t)L->W[0], (uint32_t)L->H[0]};
    SrArg a[9] = {ARG_BUF(feat), ARG_BUF_OFF(w->buf, w->off[0]), ARG_BUF_OFF(w->buf, w->off[1]), ARG_BUF_OFF(w->buf, w->off[2]), ARG_BUF_OFF(w->buf, w->off[3]),
                  ARG_BUF(s0), ARG_BUF(s1), ARG_BUF(q2), ARG_VAL(p)};
    [e dispatch:[g pipeline:@"conv_tile_p2f"] args:a count:9 threadgroups:MTLSizeMake(L->W[0] / 32, L->H[0] / 4, 1) threads:MTLSizeMake(128, 1, 1)];   /* tiles of 32 x 4 (W0 and H0 are multiples of 32) */
}

void sr_p3to5(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int pass, id<MTLBuffer> in, id<MTLBuffer> skip, id<MTLBuffer> pooled, SrScratch s) {
    static const int cin[6] = {0, 0, 0, 64, 128, 128}, cout[6] = {0, 0, 0, 128, 128, 256};
    int k = pass - 1;                                   /* P3 -> level 2 */
    conv_tile(e, g, w, 2 * (pass - 1), pass, L->wide, in, skip, pooled, L->W[k], L->H[k], cin[pass], cout[pass]);
}

/* bottleneck 1x1: two K slices per output tile, partial sums added before the epilogue (sr.bottleneck_1x1) */
static void bott(SrEnc *e, SrGpu *g, const SrWeights *w, int widx, id<MTLBuffer> x, id<MTLBuffer> y, int npix, int cin, int cout) {
    MmParams p = {(uint32_t)npix, (uint32_t)cin, (uint32_t)cout, 1, (uint32_t)cout};
    SrArg a[5] = {ARG_BUF(x), ARG_BUF_OFF(w->buf, w->off[widx]), ARG_BUF_OFF(w->buf, w->off[widx + 1]), ARG_BUF(y), ARG_VAL(p)};
    [e dispatch:[g pipeline:@"conv1x1_splitk"] args:a count:5 threadgroups:MTLSizeMake(cout / 32, (npix + 15) / 16, 1) threads:MTLSizeMake(64, 1, 1)];
}
void sr_p6(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> q5, id<MTLBuffer> z5a) { bott(e, g, w, 10, q5, z5a, L->W[5] * L->H[5], 256, 512); }
void sr_p7(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> z5a, id<MTLBuffer> z5b) { bott(e, g, w, 12, z5a, z5b, L->W[5] * L->H[5], 512, 256); }

/* fused upsample + skip + two 1x1 convolutions; kernel and tile per pass (see dec_fused.metal) */
static void dec_fused(SrEnc *e, SrGpu *g, NSString *kernel, int TP, int nsg, const SrWeights *w, int wa, id<MTLBuffer> wb_buf, size_t wb_off, size_t bb_off_idx_buf,
                      id<MTLBuffer> coarse, id<MTLBuffer> skip, id<MTLBuffer> out, int W, int H, int Wc, int Hc, int variant) {
    struct { uint32_t w, h, wc, hc, npix, variant; } p = {(uint32_t)W, (uint32_t)H, (uint32_t)Wc, (uint32_t)Hc, (uint32_t)(W * H), (uint32_t)variant};
    SrArg a[8] = {ARG_BUF(coarse), ARG_BUF(skip), ARG_BUF_OFF(w->buf, w->off[wa]), ARG_BUF_OFF(w->buf, w->off[wa + 1]),
                  ARG_BUF_OFF(wb_buf, wb_off), ARG_BUF_OFF(w->buf, bb_off_idx_buf), ARG_BUF(out), ARG_VAL(p)};
    [e dispatch:[g pipeline:kernel] args:a count:8 threadgroups:MTLSizeMake((p.npix + TP - 1) / TP, 1, 1) threads:MTLSizeMake(32 * nsg, 1, 1)];
}

void sr_dec(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int pass, int variant, id<MTLBuffer> coarse, id<MTLBuffer> skip, id<MTLBuffer> out, SrScratch s) {
    static NSString *const names[12] = {@"", @"", @"", @"", @"", @"", @"", @"", @"dec_p8", @"dec_p9", @"dec_p10", @"dec_p11"};
    static const int TP[12] = {0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 32}, NSG[12] = {0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4};
    static const int lvl[12] = {0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1}, w0[12] = {0, 0, 0, 0, 0, 0, 0, 0, 14, 18, 22, 26};
    int k = lvl[pass];
    dec_fused(e, g, names[pass], TP[pass], NSG[pass], w, w0[pass], w->buf, w->off[w0[pass] + 2], w->off[w0[pass] + 3], coarse, skip, out,
              L->W[k], L->H[k], L->W[k + 1], L->H[k + 1], variant);
}

size_t sr_ae_scratch_floats(unsigned lw, unsigned lh) { return ((size_t)lw * lh + 511) / 512 + 1; }

void sr_ae(SrEnc *e, SrGpu *g, const AeParams *p, id<MTLTexture> colour, id<MTLBuffer> s0, id<MTLBuffer> s1, id<MTLTexture> e_now) {
    AeParams pv = *p;
    uint32_t n = (p->lw * p->lh + 511) / 512;
    {
        SrArg a[3] = {ARG_TEX(colour), ARG_BUF(s0), ARG_VAL(pv)};
        [e dispatch:[g pipeline:@"ae_first"] args:a count:3 threadgroups:MTLSizeMake(n, 1, 1) threads:MTLSizeMake(512, 1, 1)];
    }
    [e barrier];
    struct { uint32_t n; } m = {n};
    SrArg a[5] = {ARG_BUF(s0), ARG_BUF(s1), ARG_VAL(m), ARG_VAL(pv), ARG_TEX(e_now)};
    [e dispatch:[g pipeline:@"ae_last"] args:a count:5 threadgroups:MTLSizeMake(1, 1, 1) threads:MTLSizeMake(512, 1, 1)];
}

void sr_p13(SrEnc *e, SrGpu *g, id<MTLTexture> e_now, id<MTLTexture> e_prev) {
    SrArg a[2] = {ARG_TEX(e_now), ARG_TEX(e_prev)};
    [e dispatch:[g pipeline:@"ae_copy"] args:a count:2 threadgroups:MTLSizeMake(1, 1, 1) threads:MTLSizeMake(1, 1, 1)];
}


/* P1 (sr.input_features_r2): one thread per output pixel, a 2 x 2 block in the lanes of a quad */
void sr_features(SrEnc *e, SrGpu *g, const FeatParams *p, const SrFeatIo *io) {
    FeatParams pv = *p;
    SrArg a[11] = {ARG_TEX(io->colour), ARG_TEX(io->depth), ARG_TEX(io->motion), ARG_TEX(io->hist), ARG_TEX(io->mem_read), ARG_TEX(io->e_now), ARG_TEX(io->e_prev),
                   ARG_TEX(io->warped), ARG_TEX(io->mem_write), ARG_BUF(io->feat), ARG_VAL(pv)};
    [e dispatch:[g pipeline:sr_p1_kernel(p)] args:a count:11 threadgroups:MTLSizeMake((p->Wo + 15) / 16, (p->Ho + 7) / 8, 1) threads:MTLSizeMake(128, 1, 1)];
}

/* P12, network head: d1 (m8 @L1), s0 (m8 @L0) -> o (fp16, 40 channels per L0 pixel) */
void sr_head(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int variant, id<MTLBuffer> d1, id<MTLBuffer> s0, id<MTLBuffer> o) {
    dec_fused(e, g, @"dec_head", 64, 4, w, 30, w->head_pad, 0, w->off[33], d1, s0, o, L->W[0], L->H[0], L->W[1], L->H[1], variant);
}

/* P12, reconstruction filter and blend: one thread per quad of 2 x 2 output pixels */
void sr_filter(SrEnc *e, SrGpu *g, const FiltParams *p, const SrFiltIo *io, unsigned W0) {
    FiltParams pv = *p;
    SrArg a[9] = {ARG_TEX(io->colour), ARG_TEX(io->warped), ARG_TEX(io->e_now), ARG_TEX(io->out), ARG_TEX(io->hist), ARG_BUF(io->o), ARG_VAL(pv), ARG_VAL(W0), ARG_BUF([g tanhTable])};
    const uint32_t qw = (p->Wo + 1) / 2, qh = (p->Ho + 1) / 2;
    [e dispatch:[g pipeline:[NSString stringWithFormat:@"p12_filter_%c%d", p->wide ? 'w' : 'n', p->hdr != 0]] args:a count:9 threadgroups:MTLSizeMake((qw + 15) / 16, (qh + 7) / 8, 1) threads:MTLSizeMake(16, 8, 1)];
}

NSString *sr_p12_kernel(const FiltParams *p) { return [NSString stringWithFormat:@"p12_fused_%c%d", p->wide ? 'w' : 'n', p->hdr != 0]; }
NSString *sr_p1_kernel(const FeatParams *p) { return [NSString stringWithFormat:@"p1_features_%d%d%d", p->hdr != 0, p->mv_render != 0, p->depth_inv != 0]; }

void sr_p12(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, const FiltParams *p, const SrP12Io *io) {
    FiltParams pv = *p;
    uint32_t W0 = (uint32_t)L->W[0], W1H1 = (uint32_t)L->W[1] | ((uint32_t)L->H[1] << 16);
    SrArg a[16] = {ARG_TEX(io->colour), ARG_TEX(io->warped), ARG_TEX(io->e_now), ARG_TEX(io->out), ARG_TEX(io->hist), ARG_TEX(io->e_prev),
                   ARG_BUF(io->d1), ARG_BUF(io->s0), ARG_BUF_OFF(w->buf, w->off[30]), ARG_BUF_OFF(w->buf, w->off[31]), ARG_BUF_OFF(w->head_pad, 0), ARG_BUF_OFF(w->buf, w->off[33]),
                   ARG_VAL(pv), ARG_VAL(W0), ARG_VAL(W1H1), ARG_BUF([g tanhTable])};
    const uint32_t qw = (p->Wo + 1) / 2, qh = (p->Ho + 1) / 2;
    [e dispatch:[g pipeline:sr_p12_kernel(p)] args:a count:16 threadgroups:MTLSizeMake((qw + 15) / 16, (qh + 7) / 8, 1) threads:MTLSizeMake(16, 8, 1)];
}

void sr_quad_debug(SrEnc *e, SrGpu *g, const FiltParams *p, id<MTLBuffer> o, unsigned W0, unsigned H0, id<MTLBuffer> dbg) {
    FiltParams pv = *p;
    SrArg a[5] = {ARG_BUF(o), ARG_VAL(pv), ARG_VAL(W0), ARG_BUF(dbg), ARG_BUF([g tanhTable])};
    [e dispatchGrid:MTLSizeMake(2 * W0, 2 * H0, 1) pso:[g pipeline:@"p12_quad_debug"] args:a count:5 threads:MTLSizeMake(16, 8, 1)];
}

void sr_weights_prepare(SrGpu *g, SrWeights *w) {
    sr_tensor_offsets(w->off);
    w->head_pad = [g buffer:48 * 32 * 2];    /* head_b padded to 48 rows with zeros: the matmul N dimension is a multiple of 16 */
    memcpy(w->head_pad.contents, (const uint8_t *)w->buf.contents + w->off[32], 40 * 32 * 2);
}

