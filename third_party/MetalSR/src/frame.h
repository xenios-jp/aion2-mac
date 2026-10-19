/* Per-call scalar derivation (sr.input_features, sr.head_filter, sr.exposure_meter). */
#ifndef SR_FRAME_H
#define SR_FRAME_H
#include <stddef.h>
#include "metalsr.h"
#include "model.h"
#include "shared_types.h"

typedef struct {
    u32 Wo, Ho, Wr, Hr;
    float jx, jy, tx, ty, pre;
    u32 hdr, depth_inv, mv_render, reset, wide, ox, oy, W0;
} SrCall;

void sr_make_feat(const SrCall *c, const SrParams *m, FeatParams *out);   /* P1: sr.input_features_r2 */
void sr_make_filt(const SrCall *c, const SrParams *m, FiltParams *out);   /* P12 filter: sr.head_filter */
/* auto exposure grid (integer halves, lower bound n_luma_min) */
void sr_make_ae(const SrCall *c, const SrParams *m, AeParams *out);


/* Size contract (sr.test_scenes_r6): the one code create, encode and metalsr_sizes use. The checks return 0, or 1 with a message in err. */
#define SR_MAX_SIZE 16384u
int sr_check_create(const MetalSRCreateInfo *info, char *err, size_t cap);                           /* version and sizes */
int sr_check_call(const MetalSRCreateInfo *info, unsigned rw, unsigned rh, char *err, size_t cap);   /* 1 <= rw <= max_rw, 1 <= rh <= max_rh */
void sr_derive_sizes(const MetalSRCreateInfo *info, unsigned rw, unsigned rh, double r_wide, unsigned n_luma_min, MetalSRSizes *out);

#endif
