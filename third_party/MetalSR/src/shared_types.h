/* Structs shared between host code and shaders. Only 4-byte scalars so both sides lay them out identically. */
#ifndef SR_SHARED_TYPES_H
#define SR_SHARED_TYPES_H
#ifdef __METAL_VERSION__
typedef uint u32;
#else
#include <stdint.h>
typedef uint32_t u32;
#endif

/* per-call scalars of the input-feature pass P1 (sr.input_features_r2); k and the exposure ratio are formed in the shader
   from invp and the exposure texels */
typedef struct {
    u32 Wo, Ho, Wr, Hr, W0;
    u32 hdr, depth_inv, mv_render, reset;
    float sx, sy, jx, jy;
    float Sx, Sy;                      /* motion factor (render-resolution vectors: 1 / scale, times the motion scale) */
    float invp;                        /* rn32(1 / pre-exposure) */
    float g_mv;                        /* rn32(kappa_mv / Ho) */
    float inv_wo, inv_ho;              /* fp32 reciprocals of the output size */
    float phi_lo, phi_hi;
} FeatParams;

/* per-call scalars of the reconstruction filter of P12 (sr.head_filter, sr.head_filter_r4) */
typedef struct {
    u32 Wo, Ho, Wr, Hr, ox, oy;
    u32 hdr, wide;                     /* wide: path W (form B coordinates, 5 x 5 taps) */
    float sx, sy, jx, jy;
    float cov_x, cov_y;                /* coverage factors */
    float A;                           /* sample-weight base */
    float invp, pre;                   /* rn32(1 / pre-exposure), rn32(pre-exposure) */
    float sigma_min, sigma_span, rho_max, q_a, q_b, q_clip, g_beta, phi_lo, phi_hi;
} FiltParams;

/* auto exposure (sr.exposure_meter) */
typedef struct {
    u32 Wr, Hr, lw, lh;
    float invp;
    float key_ae, eps_luma;
} AeParams;

#endif
