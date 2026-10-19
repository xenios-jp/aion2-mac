#include <string.h>
#include "metalfg.h"
#include "plan.h"

static uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static uint32_t up(uint32_t a, uint32_t m) { return cdiv(a, m) * m; }

uint32_t metalfg_height_cap(const MetalFGParams *p) {
    uint64_t c = 16ull * (uint64_t)p->hb;
    return c < METALFG_MAX_HEIGHT ? (uint32_t)c : METALFG_MAX_HEIGHT;
}

int metalfg_size_ok(uint32_t W, uint32_t H, const MetalFGParams *p) {
    if (W < METALFG_EFF_MIN_SIDE || H < METALFG_EFF_MIN_SIDE) return 0;
    if (W > METALFG_EFF_MAX_WIDTH) return 0;
#if defined(METALFG_CTL_NO_HCAP)
    (void)p;
#elif defined(METALFG_CTL_HCAP_OFF1)
    if (H >= metalfg_height_cap(p)) return 0;
#else
    if (H > metalfg_height_cap(p)) return 0;
#endif
    return 1;
}

static uint32_t pyr_levels(uint32_t w, uint32_t h, uint32_t cap) {
    uint32_t m = w > h ? w : h, L = 9;
    while (L < 13 && (1u << L) < m) L++;
    return L < cap ? L : cap;
}

int metalfg_plan_grids(uint32_t W, uint32_t H, const MetalFGParams *p, MetalFGGrids *g) {
    return metalfg_plan_grids_ex(W, H, p, METALFG_CREATE_FORMER_MOTION_GRID, g);
}

int metalfg_plan_grids_ex(uint32_t W, uint32_t H, const MetalFGParams *p, uint32_t cflags, MetalFGGrids *g) {
    memset(g, 0, sizeof *g);
    if (!metalfg_size_ok(W, H, p)) return 1;
    g->W = W; g->H = H;
    /* factors: motion (b) and feature (c) switch above hm and hf; the controls move the switch to H >= hm (hf) */
#if defined(METALFG_CTL_DM_GE)
    g->dm = (int64_t)H < p->hm ? 2 : 4;
#else
    g->dm = (int64_t)H <= p->hm ? 2 : 4;
#endif
    /* fg.motion_grid_r1: path N (create bit 6 clear) uses dm = 4 from hq display rows on; path O keeps the rule above */
    if (!(cflags & METALFG_CREATE_FORMER_MOTION_GRID) && H >= METALFG_MOTION_GRID_HQ) g->dm = 4;
#if defined(METALFG_CTL_DM4)
    if (!(cflags & METALFG_CREATE_FORMER_MOTION_GRID)) g->dm = 4;   /* test-only build of fg.motion_grid_r1 (soak drift): path N with dm = 4 at every height */
#endif
#if defined(METALFG_CTL_DF_GE)
    g->df = (int64_t)H < p->hf ? 2 : 4;
#else
    g->df = (int64_t)H <= p->hf ? 2 : 4;
#endif
#if defined(METALFG_CTL_CEIL)
    g->Wm = cdiv(W, g->dm); g->Hm = cdiv(H, g->dm);      // control: rounded up
    g->Wf = cdiv(W, g->df); g->Hf = cdiv(H, g->df);
#else
    g->Wm = W / g->dm; g->Hm = H / g->dm;
    g->Wf = W / g->df; g->Hf = H / g->df;
#endif
    uint32_t cw = cdiv(g->Wf, 2), ch = cdiv(g->Hf, 2);
#if defined(METALFG_CTL_N1_ROUND16)
    g->w1 = up(cw, 16); g->h1 = up(ch, 16);               // control: multiple of 16 instead of 32
#else
    g->w1 = up(cw, 32); g->h1 = up(ch, 32);
#endif
    g->ox = 2 * ((g->w1 - cw) / 2);
    g->oy = 2 * ((g->h1 - ch) / 2);
#if defined(METALFG_CTL_N1_ORIGIN)
    g->ox = 0; g->oy = 0;                                 // control: embedded at (0, 0)
#endif
#if defined(METALFG_CTL_N1_ORIGIN_X)
    g->ox = 0;                                            // control: horizontal offset only
#endif
    g->s = 1;
#if defined(METALFG_CTL_S_LT)
    while ((int64_t)H >= (int64_t)p->hb * g->s) g->s *= 2; // control: H / s < hb
#else
    while ((int64_t)H > (int64_t)p->hb * g->s) g->s *= 2; /* H / s <= hb in real arithmetic */
#endif
#if defined(METALFG_CTL_N2_PAD16)
    g->Wp = 16 * cdiv(W, 16);                             // control: multiple of 16 instead of 16 s
    g->Hp = 16 * cdiv(H, 16);
#else
    g->Wp = 16 * g->s * cdiv(W, 16 * g->s);
    g->Hp = 16 * g->s * cdiv(H, 16 * g->s);
#endif
    g->Nw = g->Wp / g->s; g->Nh = g->Hp / g->s;
    uint32_t capm = 13, capo = 13;                         // pyramid level caps (the controls lower them)
#if defined(METALFG_CTL_PYR9)
    capm = capo = 9;
#endif
#if defined(METALFG_CTL_PYR12)
    capm = capo = 12;
#endif
#if defined(METALFG_CTL_OPYR12)
    capo = 12;
#endif
#if defined(METALFG_CTL_MPYR11)
    capm = 11;
#endif
    g->pyr_mot_levels = pyr_levels(g->Wm, g->Hm, capm);
    g->pyr_out_levels = pyr_levels(W, H, capo);
    return 0;
}

int metalfg_plan_resources(const MetalFGGrids *g, uint32_t flags, MetalFGResDesc out[METALFG_MAX_RES]) {
    int n = 0;
    MetalFGResDesc all[METALFG_MAX_RES];
    int m = 0;
#define ADD(nm, gk, lv, ww, hh, fmt, by, pr, cd) \
    all[m++] = (MetalFGResDesc){nm, gk, lv, ww, hh, fmt, by, pr, cd}
    uint32_t W = g->W, H = g->H;
    /* history pairs */
    ADD("col.full", METALFG_GRID_FULL, 0, W, H, "RGBA16Float", 8, 1, 0);
    ADD("ui.full", METALFG_GRID_FULL, 0, W, H, "RGBA16Float", 8, 1, METALFG_COND_UI);
    ADD("col.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "RGBA16Float", 8, 1, 0);
    ADD("uim.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "R8Unorm", 1, 1, 0);
    ADD("err.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "RG16Float", 4, 1, 0);
    ADD("dis.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "R8Unorm", 1, 1, 0);
    ADD("dep.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "R32Uint", 4, 1, 0);
    ADD("dyn.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "R8Unorm", 1, 1, 0);
    ADD("dep2.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "R32Uint", 4, 1, METALFG_COND_DIST);
    ADD("bdf.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RGBA16Float", 8, 1, METALFG_COND_DIST);
    /* single persistent */
    ADD("st.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "R16Float", 2, 0, 0);
    ADD("menu.state", METALFG_GRID_FIXED, 0, 4, 1, "buffer", 4, 0, 0);
    /* per call */
    ADD("dY.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "R16Float", 2, 0, 0);
    ADD("mvo.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RG32Float", 8, 0, 0);
    ADD("mvf.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RG16Float", 4, 0, 0);
    ADD("nsc.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "R8Unorm", 1, 0, 0);
    ADD("mvod.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RG16Float", 4, 0, METALFG_COND_DIST);
    ADD("mvfd.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RGBA16Float", 8, 0, METALFG_COND_DIST);
    ADD("sc.pv", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "buffer", 8, 0, 0);
    ADD("hb.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "buffer", 1, 0, 0);
    for (uint32_t k = 1; k <= 7; k++)
        ADD("pyr.mot", METALFG_GRID_PYR_MOT, k, cdiv(g->Wm, 1u << k), cdiv(g->Hm, 1u << k), "buffer", 6, 0, 0);
    ADD("ms.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "R8Unorm", 1, 0, 0);
    ADD("n1in", METALFG_GRID_N1, 0, g->w1, g->h1, "buffer", 96, 0, 0);
    ADD("n1out", METALFG_GRID_N1, 0, g->w1, g->h1, "buffer", 32, 0, 0);
    ADD("lgP.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "RG16Float", 4, 0, 0);
    ADD("lgC.feat", METALFG_GRID_FEAT, 0, g->Wf, g->Hf, "RG16Float", 4, 0, 0);
    ADD("sc.ic", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "buffer", 8, 0, 0);
    ADD("sc.ip", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "buffer", 8, 0, 0);
    ADD("bw.mot", METALFG_GRID_MOT, 0, g->Wm, g->Hm, "RG16Float", 4, 0, 0);
    ADD("candA.full", METALFG_GRID_FULL, 0, W, H, "RGBA16Float", 8, 0, 0);
    ADD("candB.full", METALFG_GRID_FULL, 0, W, H, "RGBA16Float", 8, 0, 0);
    ADD("hole.full", METALFG_GRID_FULL, 0, W, H, "R8Unorm", 1, 0, 0);
    ADD("n2a", METALFG_GRID_N2, 0, g->Nw, g->Nh, "buffer", 8, 0, 0);
    ADD("n2b", METALFG_GRID_N2, 0, g->Nw, g->Nh, "buffer", 8, 0, 0);
    ADD("n2h", METALFG_GRID_N2, 0, g->Nw, g->Nh, "buffer", 2, 0, 0);
    ADD("n2out", METALFG_GRID_N2, 0, g->Nw, g->Nh, "buffer", 16, 0, 0);
    ADD("hole.out", METALFG_GRID_FULL, 0, W, H, "R8Unorm", 1, 0, METALFG_COND_CHAIN);
    for (uint32_t k = 1; k <= 8; k++)
        ADD("pyr.out", METALFG_GRID_PYR_OUT, k, cdiv(W, 1u << k), cdiv(H, 1u << k), "buffer", 8, 0, METALFG_COND_CHAIN);
#undef ADD
    for (int i = 0; i < m; i++) {
        int c = all[i].cond;
        if (c == METALFG_COND_UI && !(flags & METALFG_CREATE_UI_RECOMPOSE)) continue;
        if (c == METALFG_COND_DIST && !(flags & METALFG_CREATE_DISTORTION)) continue;
        if (c == METALFG_COND_CHAIN && (flags & METALFG_CREATE_DIRECT_OUTPUT)) continue;
        out[n++] = all[i];
    }
    return n;
}
