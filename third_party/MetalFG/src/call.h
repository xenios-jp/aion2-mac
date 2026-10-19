/* Per-call mode decisions (spec fg.orchestration "Call mode", fg.engine_api "Generation position", "Encode"). Pure logic. */
#ifndef METALFG_CALL_H
#define METALFG_CALL_H
#include <stddef.h>
#include <stdint.h>
#include "metalfg.h"

typedef enum { F04_P = 0, F04_D, F04_A, F04_R } F04Mode;
typedef enum { DET_NONE = 0, DET_ALPHA /* F02 */, DET_HUDLESS /* F03 */ } DetMode;

/* Engine state that survives between calls. */
typedef struct {
    int have_prev;        /* a previous call exists */
    uint32_t prev_cw, prev_ch;
    uint32_t prev_dz[2], prev_mv[2];  /* effective depth and motion rectangle sizes of the previous call (fg.engine_api_r2) */
    int group_open;       /* a group started (k = 1) and has not reached k = n */
    int group_reset;      /* reset status of the open group */
    int prev_dist;        /* prevD: variant of the previous call that exchanged history */
    int have_set;         /* prev_set is valid */
    int prev_set;         /* supplied set T of the previous group: bit 0 dist, bit 1 hud, bit 2 cov */
} MetalFGCallState;

typedef struct {
    int reset;            /* R */
    uint32_t k, n;
    float t;
    int advance;          /* k == n: history exchange at the end of this call */
    int new_group;        /* first call of a group: FH2 runs here when reset */
    int dist;             /* variant D */
    int dist_call_without_field; /* create bit 2 set and no field: bdf.mot.cur zero filled */
    F04Mode f04;
    int ui_zero;          /* create bit 0 set but no recomposition: ui.full.cur written as zero */
    DetMode det;
    int steering;         /* net1 UI steering switch (fg.net1_io) */
    int chain;            /* push-pull output chain F23..F26 (create bit 4 clear) */
    int cov_src_alpha;    /* coverage from ui_alpha (1) or ui_colour alpha (0), valid when a source exists */
    int has_cov;
    int set;              /* supplied set T of this call (bit 0 dist, bit 1 hud, bit 2 cov) */
} MetalFGCallPlan;

/* Plans a call. Returns 0 on success; otherwise writes the message and encodes nothing. The state is updated only by
   metalfg_call_commit. */
int metalfg_call_plan(const MetalFGCallState *st, uint32_t create_flags, uint32_t max_generated, const MetalFGFrameInfo *f,
                  MetalFGCallPlan *out, char *err, size_t cap);
/* Records the call in the state after it was encoded. */
void metalfg_call_commit(MetalFGCallState *st, const MetalFGCallPlan *p, const MetalFGFrameInfo *f);

/* fg.engine_api_r2: the effective depth and motion rectangle sizes (a zero component takes the content size) */
static inline uint32_t metalfg_eff(uint32_t v, uint32_t c) { return v ? v : c; }
#define METALFG_DEPTH_W(f) metalfg_eff((f)->depth_size[0], (f)->content_w)
#define METALFG_DEPTH_H(f) metalfg_eff((f)->depth_size[1], (f)->content_h)
#define METALFG_MOTION_W(f) metalfg_eff((f)->motion_size[0], (f)->content_w)
#define METALFG_MOTION_H(f) metalfg_eff((f)->motion_size[1], (f)->content_h)

#endif
