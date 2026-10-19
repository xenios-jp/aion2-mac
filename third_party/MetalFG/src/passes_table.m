#include <string.h>
#include "metalfg.h"
#include "passes.h"

#define PENDING "(pending)"

static const MetalFGPassInfo TABLE[] = {
    {"F01", "flag buffer clear", 'C', "-", "skip_flag", "fg.menu_detection"},
    {"F02", "menu detection (alpha)", 'C', "ui_alpha|ui_colour, menu.state", "menu.state, skip_flag", "fg.menu_detection"},
    {"F03", "menu detection (hudless)", 'C', "colour, hudless, menu.state", "menu.state, skip_flag", "fg.menu_detection"},
    {"F04", "colour downsample and UI estimate", 'C', "colour, hudless, ui_colour, ui_alpha, col.full.old", "col.full.cur, ui.full.cur, col.feat.cur, dY.feat, uim.feat.cur", "fg.colour_disocclusion"},
    {"F05", "motion preparation", 'C', "depth, motion", "dep.mot.cur, mvo.mot", "fg.motion_prep"},
    {"F06", "distortion field downsample", 'C', "distortion", "bdf.mot.cur", "fg.motion_prep"},
    {"F07", "previous-to-current scatter", 'C', "dep.mot.cur, mvo.mot, camera matrices", "sc.pv, dyn.mot.cur", "fg.prev2curr"},
    {"F08", "motion unpack and pull", 'C', "sc.pv, dep.mot.old, dyn.mot.old, mvo.mot", "mvf.mot, nsc.mot, hb.mot, pyr.mot 1..4", "fg.prev2curr"},
    {"F09", "motion pull (middle levels)", 'C', "pyr.mot 4", "pyr.mot 5..7", "fg.prev2curr"},
    {"F10", "motion push", 'C', "hb.mot, pyr.mot", "mvf.mot", "fg.prev2curr"},
    {"F11", "motion push (finest)", 'C', "mvf.mot, pyr.mot", "mvf.mot", "fg.prev2curr"},
    {"F12", "distortion correction", 'C', "mvo.mot, mvf.mot, dep.mot.cur, bdf.mot.cur, bdf.mot.old", "mvod.mot, mvfd.mot, dep2.mot.cur", "fg.motion_prep"},
    {"F13", "disocclusion scatter", 'C', "mvf.mot | mvfd.mot", "dis.feat.cur, nsc.mot", "fg.colour_disocclusion"},
    {"F14", "history-free warp-error seed", 'C', "col.feat.cur", "err.feat.cur", "fg.colour_disocclusion"},
    {"F15", "net1 input packing", 'C', "col.feat.cur/old, dY.feat, err.feat.old, nsc.mot, dis.feat.cur/old, st.feat, mvo.mot|mvod.mot, mvf.mot|mvfd.mot", "n1in, err.feat.cur, ms.feat", "fg.net1_io"},
    {"F16", "net1 forward", 'N', "n1in, n1.* weights", "n1out", "fg.net1_forward"},
    {"F17", "net1 output unpacking", 'C', "n1out, ms.feat, uim.feat.cur/old", "lgP.feat, lgC.feat, st.feat", "fg.net1_io"},
    {"F18", "intermediate motion scatter", 'C', "dep.mot|dep2.mot cur/old, mvo.mot|mvod.mot, mvf.mot|mvfd.mot, t", "sc.ic, sc.ip", "fg.interm_motion"},
    {"F19", "blend weight computation", 'C', "lgP.feat, lgC.feat, sc.ic, sc.ip", "bw.mot", "fg.interm_motion"},
    {"F20", "candidate blend and pack", 'C', "col.full.cur/old, sc.ic, sc.ip, bw.mot, err.feat.cur, ui.full.cur/old", "candA.full, candB.full, hole.full, n2a, n2b, n2h", "fg.blend_pack"},
    {"F21", "net2 forward", 'N', "n2a, n2b, n2h, t, n2.* weights", "n2out", "fg.net2_forward"},
    {"F22", "direct output synthesis", 'C', "n2out, candA.full, candB.full, hole.full", "out_interp", "fg.output_synthesis"},
    {"F23", "output hole pull", 'C', "n2out, candA.full, candB.full, hole.full", "out_interp, hole.out, pyr.out 1..4", "fg.output_synthesis"},
    {"F24", "output hole pull (middle levels)", 'C', "pyr.out 4", "pyr.out 5..8", "fg.output_synthesis"},
    {"F25", "output hole push", 'C', "hole.out, pyr.out", "out_interp", "fg.output_synthesis"},
    {"F26", "output hole push (finest)", 'C', "hole.out, pyr.out", "out_interp", "fg.output_synthesis"},
    {"F27", "menu output override", 'C', "colour, menu.state", "out_interp", "fg.menu_detection"},
    {"F28", "rect copy", 'C', "colour", "out_interp, out_real", "fg.orchestration"},
    {"FH1", "history rotation", 'H', "-", "-", "fg.resources"},
    {"FH2", "clears", 'H', "-", "history start values, sentinels", "fg.resources"},
};
#define NTABLE ((int)(sizeof TABLE / sizeof TABLE[0]))

const MetalFGPassInfo *metalfg_pass_table(int *count) {
    if (count) *count = NTABLE;
    return TABLE;
}

const MetalFGPassInfo *metalfg_pass_by_tag(const char *tag) {
    for (int i = 0; i < NTABLE; i++)
        if (strcmp(TABLE[i].tag, tag) == 0) return &TABLE[i];
    return NULL;
}

int metalfg_schedule(const MetalFGCallPlan *p, uint32_t cf, const MetalFGGrids *g, const char *t[48]) {
    int n = 0;
    int R = p->reset;
#define ADD(x) t[n++] = (x)
    if (R && p->new_group) ADD("FH2");
    if (p->new_group) {
        /* F01: before the detector in a non-reset group, after it in a reset group (fg.orchestration_r2 "Flag byte") */
        if (!R) ADD("F01");
        if (p->det == DET_ALPHA) ADD("F02");
        else if (p->det == DET_HUDLESS) ADD("F03");
        if (R) ADD("F01");
        ADD("F04");
        ADD("F05");
        if (p->dist) ADD("F06");
        ADD("F07");
        ADD("F08");
        if (g->pyr_mot_levels > 9) ADD("F09");
        ADD("F10");
        if (g->pyr_mot_levels > 9) ADD("F11");
        if (p->dist) { ADD("F12"); ADD("F13"); ADD("F13"); }
        else ADD("F13");
        if (R) ADD("F14");
        else { ADD("F15"); ADD("F16"); ADD("F17"); }
    } else if (R) {
        ADD("F01");
    }
    if (!R) { ADD("F18"); ADD("F19"); ADD("F20"); ADD("F21"); }
    ADD("F28");
    if (!R) {
        if (p->chain) {
            ADD("F23");
            if (g->pyr_out_levels > 9) ADD("F24");
            ADD("F25");
            if (g->pyr_out_levels > 9) ADD("F26");
        } else ADD("F22");
        if (p->det != DET_NONE) ADD("F27");
    }
    if (p->advance) ADD("FH1");
#undef ADD
    return n;
}
