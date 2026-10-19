#include "frame.h"
#include <stdio.h>
#include <string.h>

static float rn32(double v) { return (float)v; }

void sr_make_feat(const SrCall *c, const SrParams *m, FeatParams *o) {
    float sx = rn32((double)c->Wr / (double)c->Wo), sy = rn32((double)c->Hr / (double)c->Ho);
    float pre = c->pre <= 0 ? 1.0f : c->pre;
    o->Wo = c->Wo; o->Ho = c->Ho; o->Wr = c->Wr; o->Hr = c->Hr; o->W0 = c->W0;
    o->hdr = c->hdr; o->depth_inv = c->depth_inv; o->mv_render = c->mv_render; o->reset = c->reset;
    o->sx = sx; o->sy = sy; o->jx = c->jx; o->jy = c->jy;
    o->Sx = c->mv_render ? rn32(1.0 / (double)sx) : 1.0f;
    o->Sy = c->mv_render ? rn32(1.0 / (double)sy) : 1.0f;
    o->Sx = o->Sx * c->tx; o->Sy = o->Sy * c->ty;
    o->invp = rn32(1.0 / (double)pre);
    o->g_mv = rn32(m->kappa_mv / (double)c->Ho);
    o->inv_wo = rn32(1.0 / (double)c->Wo); o->inv_ho = rn32(1.0 / (double)c->Ho);
    o->phi_lo = rn32(m->phi_lo); o->phi_hi = rn32(m->phi_hi);
}

void sr_make_filt(const SrCall *c, const SrParams *m, FiltParams *o) {
    float sx = rn32((double)c->Wr / (double)c->Wo), sy = rn32((double)c->Hr / (double)c->Ho);
    float pre = c->pre <= 0 ? 1.0f : c->pre;
    float isx = rn32(1.0 / (double)sx), isy = rn32(1.0 / (double)sy), a0 = rn32(m->alpha_0);
    int wide = c->wide != 0;
    o->Wo = c->Wo; o->Ho = c->Ho; o->Wr = c->Wr; o->Hr = c->Hr; o->ox = c->ox; o->oy = c->oy;
    o->hdr = c->hdr; o->wide = (u32)wide;
    o->sx = sx; o->sy = sy; o->jx = c->jx; o->jy = c->jy;
    double rw = (double)c->Wo / (double)c->Wr, rh = (double)c->Ho / (double)c->Hr;
    o->cov_x = wide ? rn32(rw) : isx;
    o->cov_y = (!wide && (rw < rh ? rw : rh) >= m->r_fast) ? rn32(rh) : isy;
    o->A = wide ? (a0 * isx) * isy : a0 * (isy * isx);
    o->invp = rn32(1.0 / (double)pre); o->pre = rn32(pre);
    o->sigma_min = rn32(m->sigma_min); o->sigma_span = rn32(m->sigma_span); o->rho_max = rn32(m->rho_max);
    o->q_a = rn32(m->q_a); o->q_b = rn32(m->q_b); o->q_clip = rn32(m->q_clip); o->g_beta = rn32(m->g_beta);
    o->phi_lo = rn32(m->phi_lo); o->phi_hi = rn32(m->phi_hi);
}

/* exposure grid size on one axis (integer halves, lower bound n_luma_min) */
static unsigned grid_axis(unsigned wo, unsigned wr, unsigned mn) {
    unsigned l = wo / 2 <= wr ? wo / 2 : wr / 2;
    return l < mn ? mn : l;
}

void sr_make_ae(const SrCall *c, const SrParams *m, AeParams *o) {
    unsigned mn = (unsigned)m->n_luma_min;
    o->Wr = c->Wr; o->Hr = c->Hr; o->lw = grid_axis(c->Wo, c->Wr, mn); o->lh = grid_axis(c->Ho, c->Hr, mn);
    float pre = c->pre <= 0 ? 1.0f : c->pre;
    o->invp = rn32(1.0 / (double)pre);
    o->key_ae = rn32(m->key_ae); o->eps_luma = rn32(m->eps_luma);
}

static int size_fail(char *err, size_t cap, const char *fmt, const char *field, unsigned v, unsigned lim) {
    if (err && cap) snprintf(err, cap, fmt, field, v, lim);
    return 1;
}

int sr_check_create(const MetalSRCreateInfo *i, char *err, size_t cap) {
    if (i->version != METALSR_INTERFACE_VERSION) {
        if (err && cap) snprintf(err, cap, "unsupported interface version %u (expected %u)", i->version, METALSR_INTERFACE_VERSION);
        return 1;
    }
    if (!i->out_w || i->out_w > SR_MAX_SIZE) return size_fail(err, cap, "%s = %u is outside 1 to %u", "out_w", i->out_w, SR_MAX_SIZE);
    if (!i->out_h || i->out_h > SR_MAX_SIZE) return size_fail(err, cap, "%s = %u is outside 1 to %u", "out_h", i->out_h, SR_MAX_SIZE);
    if (!i->max_rw || i->max_rw > i->out_w) return size_fail(err, cap, "%s = %u is outside 1 to out_w = %u", "max_rw", i->max_rw, i->out_w);
    if (!i->max_rh || i->max_rh > i->out_h) return size_fail(err, cap, "%s = %u is outside 1 to out_h = %u", "max_rh", i->max_rh, i->out_h);
    return 0;
}

int sr_check_call(const MetalSRCreateInfo *i, unsigned rw, unsigned rh, char *err, size_t cap) {
    if (!rw || rw > i->max_rw) return size_fail(err, cap, "%s = %u is outside 1 to max_rw = %u", "rw", rw, i->max_rw);
    if (!rh || rh > i->max_rh) return size_fail(err, cap, "%s = %u is outside 1 to max_rh = %u", "rh", rh, i->max_rh);
    return 0;
}

void sr_derive_sizes(const MetalSRCreateInfo *i, unsigned rw, unsigned rh, double r_wide, unsigned mn, MetalSRSizes *o) {
    memset(o, 0, sizeof *o);
    o->wp = 128u * ((i->out_w + 127u) / 128u); o->hp = 128u * ((i->out_h + 127u) / 128u);
    for (int k = 0; k < 6; k++) { o->lw[k] = o->wp >> (k + 2); o->lh[k] = o->hp >> (k + 2); }
    o->path = ((double)i->out_w / (double)i->max_rw) > r_wide;
    o->sx = rn32((double)rw / (double)i->out_w); o->sy = rn32((double)rh / (double)i->out_h);
    o->gw = grid_axis(i->out_w, rw, mn); o->gh = grid_axis(i->out_h, rh, mn);
}

int metalsr_sizes(const MetalSRCreateInfo *info, uint32_t rw, uint32_t rh, double r_wide, uint32_t n_luma_min, MetalSRSizes *out, char *err, size_t err_cap) {
    if (err && err_cap) err[0] = 0;
    if (!info || !out) { if (err && err_cap) snprintf(err, err_cap, "null create record or size record"); return 1; }
    if (sr_check_create(info, err, err_cap)) return 1;
    if (sr_check_call(info, rw, rh, err, err_cap)) return 2;
    sr_derive_sizes(info, rw, rh, r_wide, n_luma_min, out);
    return 0;
}
