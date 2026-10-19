/* Planner signature for the control-build check (tools/ctl_check.sh): the grid planner of fg.resources, the create limits of fg.engine_api
   and the call planner, evaluated on a fixed list of sizes and call states, folded into one 64-bit value. The release build and the build
   with METALFG_CTL alone must print the same value; every planner control must print a different one (the planner test hook of `res-grids` is
   the planner itself, so a control that changes the planner changes this value). */
#include <stdio.h>
#include <string.h>
#include "call.h"
#include "plan.h"
#include "wgen.h"

static uint64_t acc = 1469598103934665603ull;
static void mix(uint64_t v) { for (int i = 0; i < 8; i++) { acc ^= (v >> (8 * i)) & 0xff; acc *= 1099511628211ull; } }

int main(void) {
    MetalFGParams P = wgen_params();
    uint32_t cap = metalfg_height_cap(&P);
    uint32_t hs[64], nh = 0;
    uint32_t base[] = {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 187, 360, 720, 1080, 2160, 4320, 8192, 8193, 9000};
    for (size_t i = 0; i < sizeof base / 4; i++) hs[nh++] = base[i];
    uint32_t hv[] = {(uint32_t)P.hf, (uint32_t)P.hm, (uint32_t)P.hb, cap};
    for (int i = 0; i < 4; i++) { hs[nh++] = hv[i] - 1; hs[nh++] = hv[i]; hs[nh++] = hv[i] + 1; }
    for (uint32_t k = 1; k <= 4; k++) { hs[nh++] = (uint32_t)P.hb << k; hs[nh++] = ((uint32_t)P.hb << k) + 1; }
    uint32_t ws[] = {3, 4, 7, 8, 331, 640, 1280, 3840, 8192, 8193};
    for (uint32_t i = 0; i < nh; i++)
        for (size_t j = 0; j < sizeof ws / 4; j++) {
            MetalFGGrids g;
            int rc = metalfg_plan_grids(ws[j], hs[i], &P, &g);
            mix(rc);
            if (rc) continue;
            mix(g.dm); mix(g.df); mix(g.Wm); mix(g.Hm); mix(g.Wf); mix(g.Hf); mix(g.w1); mix(g.h1); mix(g.ox); mix(g.oy); mix(g.s);
            mix(g.Wp); mix(g.Hp); mix(g.Nw); mix(g.Nh); mix(g.pyr_mot_levels); mix(g.pyr_out_levels);
            MetalFGResDesc d[METALFG_MAX_RES];
            int n = metalfg_plan_resources(&g, 0x1f, d);
            mix(n);
            for (int k = 0; k < n; k++) { mix(d[k].w); mix(d[k].h); }
        }
    /* the call planner: t, reset and the detector choice */
    MetalFGCallState st;
    memset(&st, 0, sizeof st);
    int dummy;
    MetalFGFrameInfo f;
    memset(&f, 0, sizeof f);
    f.colour = f.depth = f.motion = f.out_interp = &dummy;
    f.content_w = f.content_h = 64;
    for (uint32_t n = 1; n <= 3; n++)
        for (uint32_t k = 1; k <= n; k++) {
            MetalFGCallPlan p;
            char err[64];
            f.gen_count = n; f.gen_index = k; f.flags = ((n + k) % 2 == 1) ? METALFG_FRAME_RESET : 0;
            metalfg_call_plan(&st, METALFG_CREATE_MENU_DETECT, 3, &f, &p, err, sizeof err);
            union { float f; uint32_t u; } t = {p.t};
            mix(t.u); mix(p.reset); mix(p.det);
            f.hudless = &dummy;
            metalfg_call_plan(&st, METALFG_CREATE_MENU_DETECT, 3, &f, &p, err, sizeof err);
            mix(p.det);
            f.hudless = NULL;
            metalfg_call_commit(&st, &p, &f);
        }
    printf("sig=%016llx\n", (unsigned long long)acc);
    return 0;
}
