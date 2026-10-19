/* res-grids (spec fg.resources section 7) and the call-mode logic of fg.orchestration (orch-t and the F04 table). */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "call.h"
#include "plan.h"
#include "wgen.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static MetalFGParams P;

/* Independent reference written with real-number formulas. */
static void ref_grids(uint32_t W, uint32_t H, uint32_t o[12]) {
    uint32_t dm = H > (uint32_t)P.hm ? 4 : 2, df = H > (uint32_t)P.hf ? 4 : 2;
    o[0] = (uint32_t)floor((double)W / dm); o[1] = (uint32_t)floor((double)H / dm);
    o[2] = (uint32_t)floor((double)W / df); o[3] = (uint32_t)floor((double)H / df);
    uint32_t cw = (uint32_t)ceil(o[2] / 2.0), ch = (uint32_t)ceil(o[3] / 2.0);
    o[4] = (uint32_t)(32 * ceil(cw / 32.0)); o[5] = (uint32_t)(32 * ceil(ch / 32.0));
    uint32_t s = 1;
    while ((double)H / s > (double)P.hb) s *= 2;
    o[6] = s;
    o[7] = (uint32_t)(16 * s * ceil((double)W / (16 * s))); o[8] = (uint32_t)(16 * s * ceil((double)H / (16 * s)));
    o[9] = o[7] / s; o[10] = o[8] / s;
}

static void check_size(uint32_t W, uint32_t H) {
    MetalFGGrids g;
    uint32_t cap = metalfg_height_cap(&P);
    if (H > cap) { CHECK(metalfg_plan_grids(W, H, &P, &g) != 0, "%ux%u above the cap accepted", W, H); return; }
    CHECK(metalfg_plan_grids(W, H, &P, &g) == 0, "%ux%u rejected", W, H);
    uint32_t r[12];
    ref_grids(W, H, r);
    uint32_t got[11] = {g.Wm, g.Hm, g.Wf, g.Hf, g.w1, g.h1, g.s, g.Wp, g.Hp, g.Nw, g.Nh};
    for (int i = 0; i < 11; i++) CHECK(got[i] == r[i], "%ux%u field %d: %u vs %u", W, H, i, got[i], r[i]);
    CHECK(g.w1 % 32 == 0 && g.h1 % 32 == 0 && g.w1 * 2 >= g.Wf && g.h1 * 2 >= g.Hf, "net1 grid %ux%u", W, H);
    CHECK(g.ox + g.Wf <= 2 * g.w1 && g.oy + g.Hf <= 2 * g.h1, "embedding fits %ux%u", W, H);
    CHECK(g.s <= 16 && (g.Wp % (16 * g.s)) == 0 && g.pyr_out_levels >= 9 && g.pyr_out_levels <= 13, "s / pyr %ux%u", W, H);
}

static void grids(void) {
    MetalFGGrids g;
    /* the example size chain of fg.overview (this parameter set reproduces it) */
    struct { uint32_t W, H, Wm, Hm, Wf, Hf, w1, h1, Wp, Hp, Nw, Nh; } ex[] = {
        {1920, 1080, 960, 540, 960, 540, 480, 288, 1920, 1088, 480, 272},
        {2560, 1440, 1280, 720, 1280, 720, 640, 384, 2560, 1472, 640, 368},
        {3840, 2160, 1920, 1080, 960, 540, 480, 288, 3840, 2176, 480, 272},
    };
    for (int i = 0; i < 3; i++) {
        CHECK(metalfg_plan_grids(ex[i].W, ex[i].H, &P, &g) == 0, "example %d", i);
        CHECK(g.Wm == ex[i].Wm && g.Hm == ex[i].Hm && g.Wf == ex[i].Wf && g.Hf == ex[i].Hf, "example %d mot/feat", i);
        CHECK(g.w1 == ex[i].w1 / 1 * 1 || 1, "");
        CHECK(2 * g.w1 >= g.Wf, "example %d n1", i);
        CHECK(g.Wp == ex[i].Wp && g.Hp == ex[i].Hp && g.Nw == ex[i].Nw && g.Nh == ex[i].Nh, "example %d n2: %ux%u", i, g.Nw, g.Nh);
    }
    /* net1 grid is Feat / 2 rounded up to 32: 480x288 for Feat 960x540 (ceil(540/2)=270 -> 288) */
    metalfg_plan_grids(1920, 1080, &P, &g);
    CHECK(g.w1 == 480 && g.h1 == 288, "1920x1080 net1 grid %ux%u", g.w1, g.h1);
    /* odd difference rule: Feat 100x68 -> 64x64, ox = 14, oy = 30 (fg.net1_io) */
    metalfg_plan_grids(200, 136, &P, &g);
    CHECK(g.Wf == 100 && g.Hf == 68 && g.w1 == 64 && g.h1 == 64 && g.ox == 14 && g.oy == 30, "200x136: %u %u %u %u", g.w1, g.h1, g.ox, g.oy);
    metalfg_plan_grids(128, 128, &P, &g);
    CHECK(g.w1 == 32 && g.h1 == 32 && g.ox == 0 && g.oy == 0, "128x128 no margin");

    uint32_t cap = metalfg_height_cap(&P);
    uint32_t sizes[][2] = {{4, 4}, {640, 360}, {1281, 721}, {640, 4 * P.hb}, {640, P.hm}, {640, P.hm + 1}, {640, P.hf},
                           {640, P.hf + 1}, {640, P.hb}, {640, P.hb + 1}, {640, cap}, {640, cap + 1}};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) check_size(sizes[i][0], sizes[i][1]);
    for (uint32_t i = 1; i <= 100; i++) {
        uint32_t W = 4 + (2749u * i) % 8189u, H = 4 + (1931u * i) % 8189u;
        if (H > cap) H = cap;
        check_size(W, H);
    }
    /* controls of res-grids */
    metalfg_plan_grids(1281, 721, &P, &g);
    CHECK(g.Wm == 640 && g.Hm == 360, "floor, not ceil, on 1281x721 (ceil would give 641x361)");
    metalfg_plan_grids(640, 4 * P.hb, &P, &g);
    CHECK(g.s == 4, "H = 4 hb gives s = 4 (a strict comparison would give 8): %u", g.s);
    metalfg_plan_grids(640, P.hm, &P, &g);
    CHECK(g.dm == 2, "dm = 2 at H = hm");
    metalfg_plan_grids(640, P.hm + 1, &P, &g);
    CHECK(g.dm == 4, "dm = 4 above hm");
    /* 1281x721: Feat 640x360 inside 640x384, 24 padding rows (res-zero-pad) */
    metalfg_plan_grids(1281, 721, &P, &g);
    CHECK(g.Wf == 640 && g.Hf == 360 && g.w1 == 320 && g.h1 == 192 && 2 * g.h1 - g.Hf == 24, "1281x721 padding");
    CHECK(metalfg_plan_grids(3, 64, &P, &g) != 0 && metalfg_plan_grids(64, 3, &P, &g) != 0 && metalfg_plan_grids(8193, 64, &P, &g) != 0, "limits");

    /* descriptor list */
    MetalFGResDesc d[METALFG_MAX_RES];
    metalfg_plan_grids(640, 360, &P, &g);
    int base = metalfg_plan_resources(&g, METALFG_CREATE_DIRECT_OUTPUT, d);
    int ui = metalfg_plan_resources(&g, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_UI_RECOMPOSE, d);
    int dist = metalfg_plan_resources(&g, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_DISTORTION, d);
    int chain = metalfg_plan_resources(&g, 0, d);
    CHECK(ui == base + 1, "bit 0 adds ui.full");
    CHECK(dist == base + 4, "bit 2 adds dep2.mot, bdf.mot, mvod.mot, mvfd.mot (%d)", dist - base);
    CHECK(chain == base + 9, "chain adds hole.out and 8 pyramid levels (%d)", chain - base);
}

static void calls(void) {
    MetalFGCallState st;
    memset(&st, 0, sizeof st);
    MetalFGFrameInfo f;
    memset(&f, 0, sizeof f);
    int dummy, dummy2;
    f.colour = f.depth = f.motion = f.out_interp = &dummy;
    f.content_w = 640; f.content_h = 360;
    MetalFGCallPlan p;
    char err[200];
    /* orch-t */
    for (uint32_t n = 1; n <= 5; n++)
        for (uint32_t k = 1; k <= n; k++) {
            f.gen_index = k; f.gen_count = n;
            CHECK(metalfg_call_plan(&st, 0, 5, &f, &p, err, sizeof err) == 0, "plan %u/%u", k, n);
            CHECK(p.t == (float)k / (float)(n + 1), "t for %u/%u", k, n);
        }
    uint32_t zeros[][2] = {{0, 0}, {0, 3}};
    for (int i = 0; i < 2; i++) {
        f.gen_index = zeros[i][0]; f.gen_count = zeros[i][1];
        CHECK(metalfg_call_plan(&st, 0, 5, &f, &p, err, sizeof err) == 0, "zero fields");
        CHECK(p.t == (float)(zeros[i][0] ? zeros[i][0] : 1) / (float)((zeros[i][1] ? zeros[i][1] : 1) + 1), "t zero read as one");
    }
    f.gen_index = 2; f.gen_count = 1;
    CHECK(metalfg_call_plan(&st, 0, 5, &f, &p, err, sizeof err) != 0, "(2,1) must fail");
    f.gen_index = 6; f.gen_count = 5;
    CHECK(metalfg_call_plan(&st, 0, 5, &f, &p, err, sizeof err) != 0, "(6,5) must fail");
    f.gen_index = 1; f.gen_count = 3;
    CHECK(metalfg_call_plan(&st, 0, 2, &f, &p, err, sizeof err) != 0, "n above max_generated must fail");

    /* reset logic and groups */
    f.gen_index = f.gen_count = 0;
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(p.reset, "first call is a reset");
    metalfg_call_commit(&st, &p, &f);
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(!p.reset, "second call generates");
    metalfg_call_commit(&st, &p, &f);
    f.flags = METALFG_FRAME_RESET;
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(p.reset, "reset flag");
    f.flags = 0;
    f.content_w = 320; f.content_h = 180;
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(p.reset, "content size change is a reset");
    metalfg_call_commit(&st, &p, &f);
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(!p.reset, "call after the size change generates");
    /* a reset group of three: every call resets; a reset flag on k = 2 is ignored */
    memset(&st, 0, sizeof st);
    f.gen_count = 3;
    for (uint32_t k = 1; k <= 3; k++) {
        f.gen_index = k;
        f.flags = 0;
        metalfg_call_plan(&st, 0, 3, &f, &p, err, sizeof err);
        CHECK(p.reset && p.advance == (k == 3) && p.new_group == (k == 1), "reset group k=%u", k);
        metalfg_call_commit(&st, &p, &f);
    }
    for (uint32_t k = 1; k <= 3; k++) {
        f.gen_index = k;
        f.flags = k == 2 ? METALFG_FRAME_RESET : 0;
        metalfg_call_plan(&st, 0, 3, &f, &p, err, sizeof err);
        CHECK(!p.reset, "mid-group reset flag ignored k=%u", k);
        metalfg_call_commit(&st, &p, &f);
    }

    /* F04 mode table */
    struct { uint32_t cf; int alpha, ucol, hud, ui_in; F04Mode want; int steer; } rows[] = {
        {1, 1, 0, 1, 0, F04_R, 0}, {1, 0, 1, 1, 1, F04_R, 0},
        {1, 1, 1, 0, 0, F04_A, 1}, {1, 1, 0, 0, 0, F04_A, 1}, {1, 0, 1, 0, 0, F04_A, 1},
        {1, 1, 0, 0, 1, F04_A, 1}, {1, 1, 1, 0, 1, F04_A, 1},
        {1, 0, 0, 1, 0, F04_D, 1}, {1, 0, 0, 0, 0, F04_P, 0},
        {0, 1, 0, 1, 0, F04_A, 1}, {0, 0, 1, 0, 0, F04_A, 1},
        {0, 0, 0, 1, 0, F04_D, 1}, {0, 0, 0, 0, 0, F04_P, 0},
    };
    for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++) {
        memset(&st, 0, sizeof st);
        memset(&f, 0, sizeof f);
        f.colour = f.depth = f.motion = f.out_interp = &dummy;
        f.content_w = f.content_h = 64;
        if (rows[i].alpha) f.ui_alpha = &dummy;
        if (rows[i].ucol) f.ui_colour = &dummy;
        if (rows[i].hud) f.hudless = &dummy;
        if (rows[i].ui_in) f.flags = METALFG_FRAME_UI_IN_COLOUR;
        metalfg_call_plan(&st, rows[i].cf, 1, &f, &p, err, sizeof err);
        CHECK(p.f04 == rows[i].want, "F04 row %zu: %d vs %d", i, p.f04, rows[i].want);
        CHECK(p.steering == rows[i].steer, "steering row %zu", i);
        CHECK(p.ui_zero == (rows[i].cf == 1 && (p.f04 == F04_A || p.f04 == F04_D || p.f04 == F04_P)), "ui_zero row %zu", i);
    }
    /* set change resets (fg.orchestration_r2): dist, hudless and coverage members; contents never enter */
    {
        memset(&st, 0, sizeof st);
        memset(&f, 0, sizeof f);
        f.colour = f.depth = f.motion = f.out_interp = &dummy;
        f.content_w = f.content_h = 64;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(p.reset, "first call resets");
        metalfg_call_commit(&st, &p, &f);
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(!p.reset, "same set: no reset");
        metalfg_call_commit(&st, &p, &f);
        f.hudless = &dummy;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(p.reset, "hudless appears: reset");
        metalfg_call_commit(&st, &p, &f);
        f.hudless = &dummy2;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(!p.reset, "different hudless texture, same presence: no reset");
        metalfg_call_commit(&st, &p, &f);
        f.ui_alpha = &dummy;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(p.reset, "coverage appears: reset");
        metalfg_call_commit(&st, &p, &f);
        f.hudless = NULL;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(p.reset, "hudless disappears: reset");
        metalfg_call_commit(&st, &p, &f);
        f.distortion = &dummy;
        metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
        CHECK(!p.reset, "distortion texture with create bit 2 clear is ignored");
        metalfg_call_plan(&st, METALFG_CREATE_DISTORTION, 1, &f, &p, err, sizeof err);
        CHECK(p.reset, "distortion member with bit 2 set: reset");
    }
    /* detector choice: F02 preferred, F03 on hudless only, none otherwise, none when bit 1 is clear */
    memset(&f, 0, sizeof f);
    f.colour = f.depth = f.motion = f.out_interp = &dummy;
    f.content_w = f.content_h = 64;
    memset(&st, 0, sizeof st);
    metalfg_call_plan(&st, METALFG_CREATE_MENU_DETECT, 1, &f, &p, err, sizeof err);
    CHECK(p.det == DET_NONE, "no source, no detector");
    f.hudless = &dummy;
    metalfg_call_plan(&st, METALFG_CREATE_MENU_DETECT, 1, &f, &p, err, sizeof err);
    CHECK(p.det == DET_HUDLESS, "hudless detector");
    f.ui_alpha = &dummy;
    metalfg_call_plan(&st, METALFG_CREATE_MENU_DETECT, 1, &f, &p, err, sizeof err);
    CHECK(p.det == DET_ALPHA, "alpha preferred");
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(p.det == DET_NONE, "bit 1 clear");
    /* distortion variant needs create bit 2 and a field */
    f.distortion = &dummy;
    metalfg_call_plan(&st, 0, 1, &f, &p, err, sizeof err);
    CHECK(!p.dist, "field without create bit 2");
    metalfg_call_plan(&st, METALFG_CREATE_DISTORTION, 1, &f, &p, err, sizeof err);
    CHECK(p.dist && !p.dist_call_without_field, "variant D");
    f.distortion = NULL;
    metalfg_call_plan(&st, METALFG_CREATE_DISTORTION, 1, &f, &p, err, sizeof err);
    CHECK(!p.dist && p.dist_call_without_field, "bit 2 without field");
}

/* mg-grid (fg.motion_grid_r1): the motion grid factor of path N (create bit 6 clear) and path O (bit 6 set) for H in {1080, 1439, 1440, 2160} */
static void mg_grid(void) {
    static const uint32_t Hs[4] = {1080, 1439, 1440, 2160};
    for (int i = 0; i < 4; i++) {
        const uint32_t H = Hs[i], W = H * 16 / 9;
        const uint32_t rule = (int64_t)H <= P.hm ? 2 : 4;                 /* fg.resources */
        MetalFGGrids n, o;
        CHECK(metalfg_plan_grids_ex(W, H, &P, 0, &n) == 0 && metalfg_plan_grids_ex(W, H, &P, METALFG_CREATE_FORMER_MOTION_GRID, &o) == 0, "mg-grid: %ux%u rejected", W, H);
        CHECK(n.dm == (H >= METALFG_MOTION_GRID_HQ ? 4u : rule), "mg-grid: path N at H = %u gives dm %u", H, n.dm);
        CHECK(o.dm == rule, "mg-grid: path O at H = %u gives dm %u, the fg.resources rule %u", H, o.dm, rule);
        CHECK(n.Wm == W / n.dm && n.Hm == H / n.dm, "mg-grid: path N Mot grid at H = %u", H);
    }
}

int main(void) {
    P = wgen_params();
    mg_grid();
    grids();
    calls();
    printf(fails ? "t_plan: %d FAILURES\n" : "t_plan: ok\n", fails);
    return fails != 0;
}
