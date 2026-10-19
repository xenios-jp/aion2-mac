#include <stdio.h>
#include <string.h>
#include "call.h"

static int fail(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg);
    return 1;
}

int metalfg_call_plan(const MetalFGCallState *st, uint32_t cf, uint32_t max_gen, const MetalFGFrameInfo *f, MetalFGCallPlan *o,
                  char *err, size_t cap) {
    memset(o, 0, sizeof *o);
    if (!f) return fail(err, cap, "frame record is null");
    if (!f->colour) return fail(err, cap, "colour texture is null");
    if (!f->depth) return fail(err, cap, "depth texture is null");
    if (!f->motion) return fail(err, cap, "motion texture is null");
    if (!f->out_interp) return fail(err, cap, "out_interp texture is null");
    if (f->content_w == 0 || f->content_h == 0) return fail(err, cap, "content_w and content_h must be non-zero");

    uint32_t k = f->gen_index ? f->gen_index : 1, n = f->gen_count ? f->gen_count : 1;
    uint32_t mg = max_gen ? max_gen : 1;
    if (n > mg) return fail(err, cap, "gen_count exceeds max_generated");
    if (k > n) return fail(err, cap, "gen_index exceeds gen_count");
    o->k = k; o->n = n;
    o->t = (float)k / (float)(n + 1); /* fp32 division, once per call */
#if defined(METALFG_CTL_T_HALF)
    o->t = 0.5f;                                     /* control: t = 1/2 whatever k and n are */
#endif
    o->advance = (k == n);

    /* reset: decided by the first call of a group, held for the group */
    o->new_group = (k == 1) || !st->group_open;
    int content_changed = st->have_prev && (f->content_w != st->prev_cw || f->content_h != st->prev_ch);
    /* fg.engine_api_r2: a change of the effective depth or motion rectangle size also makes the group a reset group */
    content_changed = content_changed || (st->have_prev && (METALFG_DEPTH_W(f) != st->prev_dz[0] || METALFG_DEPTH_H(f) != st->prev_dz[1] ||
                                                            METALFG_MOTION_W(f) != st->prev_mv[0] || METALFG_MOTION_H(f) != st->prev_mv[1]));
    int dist_in = (cf & METALFG_CREATE_DISTORTION) && f->distortion;
    o->set = (dist_in ? 1 : 0) | (f->hudless ? 2 : 0) | ((f->ui_alpha || f->ui_colour) ? 4 : 0);
    int set_changed = st->have_set && o->set != st->prev_set;
    int host_reset = (f->flags & METALFG_FRAME_RESET) != 0;
#if defined(METALFG_CTL_NO_RESET)
    host_reset = 0;                                  /* control: the host reset flag is ignored */
#endif
    if (o->new_group)
        o->reset = host_reset || !st->have_prev || content_changed || set_changed;
    else
        o->reset = st->group_reset;

    o->dist = (cf & METALFG_CREATE_DISTORTION) && f->distortion;
    o->dist_call_without_field = (cf & METALFG_CREATE_DISTORTION) && !f->distortion;
    o->chain = !(cf & METALFG_CREATE_DIRECT_OUTPUT);

    int bit0 = (cf & METALFG_CREATE_UI_RECOMPOSE) != 0;
    int has_cov = f->ui_alpha != NULL || f->ui_colour != NULL;
    int hud = f->hudless != NULL;
    o->has_cov = has_cov;
    o->cov_src_alpha = f->ui_alpha != NULL;
    int rc = bit0 && has_cov && hud;
    if (rc) o->f04 = F04_R;
    else if (has_cov) o->f04 = F04_A;
    else o->f04 = hud ? F04_D : F04_P;
    o->ui_zero = bit0 && (o->f04 == F04_A || o->f04 == F04_D || o->f04 == F04_P);
    o->steering = (hud || has_cov) && !rc;

#if !defined(METALFG_CTL_NO_MENU)                         /* control: menu detection never runs */
    if (cf & METALFG_CREATE_MENU_DETECT)
        o->det = has_cov ? DET_ALPHA : (hud ? DET_HUDLESS : DET_NONE);
#endif
    return 0;
}

void metalfg_call_commit(MetalFGCallState *st, const MetalFGCallPlan *p, const MetalFGFrameInfo *f) {
    st->have_prev = 1;
    st->prev_cw = f->content_w;
    st->prev_ch = f->content_h;
    st->prev_dz[0] = METALFG_DEPTH_W(f); st->prev_dz[1] = METALFG_DEPTH_H(f);
    st->prev_mv[0] = METALFG_MOTION_W(f); st->prev_mv[1] = METALFG_MOTION_H(f);
    if (p->new_group) { st->have_set = 1; st->prev_set = p->set; }
    if (p->advance) {
        st->group_open = 0;
        st->prev_dist = p->dist;
    } else {
        st->group_open = 1;
        st->group_reset = p->reset;
    }
}
