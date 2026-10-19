/* Encoders for the network passes P2-P11 (channels-last internal layouts; m8 tensors are uint8). */
#import "gpu.h"
#include "model.h"
#include "shared_types.h"

typedef struct {
    id<MTLBuffer> buf;               /* weight file image (or any buffer laid out like it) */
    size_t off[SR_NUM_TENSORS];
    id<MTLBuffer> head_pad;          /* head_b weights padded to 48 rows (zero rows) */
} SrWeights;

typedef struct { int Wp, Hp, W[6], H[6]; int wide; } SrLevels;   /* wide: path W (K chunks of 32, 5x5 head filter) */
void sr_levels(int Wo, int Ho, SrLevels *L);
void sr_weights_prepare(SrGpu *g, SrWeights *w);   /* builds engine-side copies; w->buf must hold the file image */

typedef struct { id<MTLBuffer> b; } SrT; /* a tensor is just a buffer; shapes come from SrLevels */

/* building blocks */
void sr_conv3x3_relu(SrEnc *e, SrGpu *g, id<MTLBuffer> x, int x_m8, const SrWeights *w, int widx, id<MTLBuffer> y, int W, int H, int cin, int cout);
void sr_maxpool(SrEnc *e, SrGpu *g, id<MTLBuffer> y, id<MTLBuffer> q, int W, int H, int C);
/* flags: 1 m8 input, 2 relu, 4 head epilogue (two roundings), 8 m8 output */
void sr_conv1x1(SrEnc *e, SrGpu *g, id<MTLBuffer> x, const SrWeights *w, int widx, id<MTLBuffer> y, int npix, int cin, int cout, int flags);
void sr_m8_encode(SrEnc *e, SrGpu *g, id<MTLBuffer> x, id<MTLBuffer> y, int count);
void sr_up2_add(SrEnc *e, SrGpu *g, id<MTLBuffer> coarse, int coarse_m8, id<MTLBuffer> skip, int skip_m8, id<MTLBuffer> x,
                int wc, int hc, int c, int variant);

/* passes. `scratch` buffers must be large enough (see sr_scratch_bytes). */
typedef struct { id<MTLBuffer> a, b, c; } SrScratch;
size_t sr_scratch_bytes(const SrLevels *L, int which);
void sr_p2(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> feat, id<MTLBuffer> s0, id<MTLBuffer> s1, id<MTLBuffer> q2, SrScratch s);
void sr_p3to5(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int pass /*3..5*/, id<MTLBuffer> in, id<MTLBuffer> skip, id<MTLBuffer> pooled, SrScratch s);
void sr_p6(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> q5, id<MTLBuffer> z5a);
void sr_p7(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, id<MTLBuffer> z5a, id<MTLBuffer> z5b);
/* decoders P8..P11: coarse/skip -> out; out is fp16 for P8, P9 and m8 for P10, P11 */
/* P1 (sr.input_features_r2): feat is [H0][W0][16] fp16 (written for the blocks that touch the output, zeros elsewhere
   from the first-call clear); warped is the image R, mem_write the motion memory of this call */
typedef struct { id<MTLTexture> colour, depth, motion, hist, mem_read, e_now, e_prev, warped, mem_write; id<MTLBuffer> feat; } SrFeatIo;
void sr_features(SrEnc *e, SrGpu *g, const FeatParams *p, const SrFeatIo *io);
/* P12 (sr.head_filter): network head d1 (m8 @L1), s0 (m8 @L0) -> o ([H0][W0][40] fp16), then the filter and blend of o,
   colour and R into the output and the history */
void sr_head(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int variant, id<MTLBuffer> d1, id<MTLBuffer> s0, id<MTLBuffer> o);
typedef struct { id<MTLTexture> colour, warped, e_now, out, hist; id<MTLBuffer> o; } SrFiltIo;
void sr_filter(SrEnc *e, SrGpu *g, const FiltParams *p, const SrFiltIo *io, unsigned W0);
/* P12 as one dispatch: head network (d1, s0 -> parameters in threadgroup memory), filter and blend, and P13 (e_prev := the exposure source e_now) */
typedef struct { id<MTLTexture> colour, warped, e_now, out, hist, e_prev; id<MTLBuffer> d1, s0; } SrP12Io;
/* names of the kernels a configuration uses (T1: one kernel per path, colour mode, motion-vector resolution and depth direction) */
NSString *sr_p12_kernel(const FiltParams *p);
NSString *sr_p1_kernel(const FeatParams *p);
void sr_p12(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, const FiltParams *p, const SrP12Io *io);
/* test hook: derived per-quad parameters, eight floats per quad of the (2 W0) x (2 H0) quad grid */
void sr_quad_debug(SrEnc *e, SrGpu *g, const FiltParams *p, id<MTLBuffer> o, unsigned W0, unsigned H0, id<MTLBuffer> dbg);
/* A1-A3: scratch buffers hold ceil(lw*lh/512) and ceil(that/512) floats at least */
void sr_ae(SrEnc *e, SrGpu *g, const AeParams *p, id<MTLTexture> colour, id<MTLBuffer> scratch0, id<MTLBuffer> scratch1, id<MTLTexture> e_now);
void sr_p13(SrEnc *e, SrGpu *g, id<MTLTexture> e_now, id<MTLTexture> e_prev);
size_t sr_ae_scratch_floats(unsigned lw, unsigned lh);
void sr_dec(SrEnc *e, SrGpu *g, const SrWeights *w, const SrLevels *L, int pass /*8..11*/, int variant, id<MTLBuffer> coarse, id<MTLBuffer> skip,
            id<MTLBuffer> out, SrScratch s);
