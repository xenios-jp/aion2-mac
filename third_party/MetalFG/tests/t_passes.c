/* Pass table and schedule (fg.overview section 3, fg.orchestration "Reset call", "Normal call", "Option interplay"). */
#include <stdio.h>
#include <string.h>
#include "passes.h"
#include "wgen.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int has(const char **t, int n, const char *tag) { int c = 0; for (int i = 0; i < n; i++) c += !strcmp(t[i], tag); return c; }
static int pos(const char **t, int n, const char *tag) { for (int i = 0; i < n; i++) if (!strcmp(t[i], tag)) return i; return -1; }

int main(void) {
    int nt;
    const MetalFGPassInfo *tab = metalfg_pass_table(&nt);
    CHECK(nt == 28 + 2, "28 passes F01..F28 plus FH1, FH2 (%d)", nt);
    for (int i = 1; i <= 28; i++) {
        char tag[8];
        snprintf(tag, sizeof tag, "F%02d", i);
        CHECK(metalfg_pass_by_tag(tag) != NULL, "pass %s listed", tag);
    }
    (void)tab;
    MetalFGParams P = wgen_params();
    MetalFGGrids g, tall;
    metalfg_plan_grids(640, 360, &P, &g);
    metalfg_plan_grids(640, 1500, &P, &tall);
    const char *t[48];
    MetalFGCallPlan p;
    memset(&p, 0, sizeof p);
    p.k = p.n = 1; p.advance = 1; p.new_group = 1; p.chain = 0;

    /* normal generating call, direct output */
    int n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT, &g, t);
    const char *want[] = {"F01", "F04", "F05", "F07", "F08", "F10", "F13", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F28", "F22", "FH1"};
    CHECK(n == 17, "normal call length %d", n);
    for (int i = 0; i < n && i < 17; i++) CHECK(!strcmp(t[i], want[i]), "normal[%d] = %s, want %s", i, t[i], want[i]);
    CHECK(pos(t, n, "F28") < pos(t, n, "F22"), "prefill before synthesis");
    CHECK(pos(t, n, "F19") > pos(t, n, "F17") && pos(t, n, "F19") > pos(t, n, "F18") && pos(t, n, "F20") > pos(t, n, "F19"), "F19/F20 order");

    /* reset call: only F04..F13, F14 and F28 (+ F02/F03, F01 when enabled) */
    p.reset = 1;
    n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT, &g, t);
    CHECK(has(t, n, "FH2") == 1 && has(t, n, "F14") == 1 && has(t, n, "F28") == 1, "reset call has FH2, F14, F28");
    CHECK(has(t, n, "F01") == 1, "F01 runs on every reset call, also without menu detection");
    for (const char *no[] = {"F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F27"}, **q = no; q < no + 10; q++)
        CHECK(!has(t, n, *q), "reset call must not run %s", *q);
    p.det = DET_ALPHA;
    n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_MENU_DETECT, &g, t);
    CHECK(has(t, n, "F02") && has(t, n, "F01") && pos(t, n, "F01") > pos(t, n, "F02"), "F01 after the detector on reset");
    CHECK(!has(t, n, "F27"), "no override on reset");
    /* later calls of a reset group do not clear again */
    p.new_group = 0;
    n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT, &g, t);
    CHECK(!has(t, n, "FH2"), "FH2 once per group");
    p.new_group = 1; p.reset = 0;

    /* menu detection: F27 after the synthesis, F02 before F28 */
    n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_MENU_DETECT, &g, t);
    CHECK(pos(t, n, "F27") > pos(t, n, "F22") && pos(t, n, "F02") < pos(t, n, "F28") && pos(t, n, "F01") < pos(t, n, "F02"), "F27 order, F01 before the detector in a non-reset group");
    p.det = DET_NONE;

    /* distortion variant: F06, F12 and F13 twice */
    p.dist = 1;
    n = metalfg_schedule(&p, METALFG_CREATE_DIRECT_OUTPUT | METALFG_CREATE_DISTORTION, &g, t);
    CHECK(has(t, n, "F06") == 1 && has(t, n, "F12") == 1 && has(t, n, "F13") == 2, "distortion passes");
    CHECK(pos(t, n, "F12") > pos(t, n, "F10") && pos(t, n, "F13") > pos(t, n, "F12"), "F12 between the fill and F13");
    p.dist = 0;

    /* output chain, and the extra levels for tall grids */
    p.chain = 1;
    n = metalfg_schedule(&p, 0, &g, t);
    CHECK(has(t, n, "F23") && has(t, n, "F25") && !has(t, n, "F22"), "chain replaces F22");
    n = metalfg_schedule(&p, 0, &tall, t);
    CHECK(has(t, n, "F09") && has(t, n, "F11") && has(t, n, "F24") && has(t, n, "F26"), "tall: middle and finest passes");
    n = metalfg_schedule(&p, 0, &g, t);
    CHECK(!has(t, n, "F09") && !has(t, n, "F11"), "short: no middle passes");

    /* no history exchange before the last call of a group */
    p.advance = 0; p.k = 1; p.n = 3;
    n = metalfg_schedule(&p, 0, &g, t);
    CHECK(!has(t, n, "FH1"), "FH1 only at k = n");
    /* groups: calls k > 1 run F18 onwards only (fg.orchestration_r2) */
    p.k = 2; p.new_group = 0; p.advance = 0; p.reset = 0;
    n = metalfg_schedule(&p, METALFG_CREATE_MENU_DETECT, &g, t);
    for (const char *no[] = {"F01", "F02", "F04", "F05", "F07", "F13", "F15", "F16", "F17"}, **q = no; q < no + 9; q++)
        CHECK(!has(t, n, *q), "group call k = 2 must not run %s", *q);
    CHECK(has(t, n, "F18") && has(t, n, "F19") && has(t, n, "F20") && has(t, n, "F21") && has(t, n, "F23") && has(t, n, "F28"), "k = 2 runs F18 onwards");
    p.reset = 1;
    n = metalfg_schedule(&p, 0, &g, t);
    CHECK(n == 2 && !strcmp(t[0], "F01") && !strcmp(t[1], "F28"), "reset group, k = 2: flag clear and rect copy only (%d)", n);
    printf(fails ? "t_passes: %d FAILURES\n" : "t_passes: ok\n", fails);
    return fails != 0;
}
