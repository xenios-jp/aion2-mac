/* Grid sizes and the resource descriptor list as pure functions of the display size (spec fg.resources section 3). */
#ifndef METALFG_PLAN_H
#define METALFG_PLAN_H
#include <stdint.h>
#include "weights.h"

typedef struct {
    uint32_t W, H;
    uint32_t dm, df;          /* reduction factors of the motion and feature grids */
    uint32_t Wm, Hm;          /* Mot grid */
    uint32_t Wf, Hf;         /* Feat grid */
    uint32_t w1, h1;         /* net1 grid (level 0) */
    uint32_t ox, oy;         /* embedding offset of Feat inside 2 w1 x 2 h1, in Feat positions */
    uint32_t s;              /* net2 factor */
    uint32_t Wp, Hp;         /* padded display */
    uint32_t Nw, Nh;         /* net2 grid */
    uint32_t pyr_mot_levels; /* L of the motion pyramid (9..13) */
    uint32_t pyr_out_levels; /* L of the output pyramid */
} MetalFGGrids;

/* Size limits of fg.engine_api (create). The control builds of fg.test_scenes_r3 move them (METALFG_CTL_MIN8, METALFG_CTL_NO_MIN, METALFG_CTL_NO_WCAP). */
#if defined(METALFG_CTL_NO_MIN)
#define METALFG_EFF_MIN_SIDE 0u
#elif defined(METALFG_CTL_MIN8)
#define METALFG_EFF_MIN_SIDE 8u
#else
#define METALFG_EFF_MIN_SIDE METALFG_MIN_SIDE
#endif
#if defined(METALFG_CTL_NO_WCAP)
#define METALFG_EFF_MAX_WIDTH 0x40000000u
#else
#define METALFG_EFF_MAX_WIDTH METALFG_MAX_WIDTH
#endif
/* Non-zero when the display size is accepted. */
int metalfg_size_ok(uint32_t W, uint32_t H, const MetalFGParams *p);

/* Height cap of fg.engine_api: min(8192, 16 hb). */
uint32_t metalfg_height_cap(const MetalFGParams *p);
/* Returns 0 on success; non-zero when a side is below 4, above the width cap or above the height cap. */
int metalfg_plan_grids(uint32_t W, uint32_t H, const MetalFGParams *p, MetalFGGrids *g);
/* The same with the create flags: fg.motion_grid_r1 path N (bit 6 clear: dm = 4 from METALFG_MOTION_GRID_HQ display rows) or path O (bit 6 set: the rule of
   fg.resources). metalfg_plan_grids is path O. */
int metalfg_plan_grids_ex(uint32_t W, uint32_t H, const MetalFGParams *p, uint32_t cflags, MetalFGGrids *g);

typedef enum { METALFG_GRID_FULL, METALFG_GRID_MOT, METALFG_GRID_FEAT, METALFG_GRID_N1, METALFG_GRID_N2, METALFG_GRID_PYR_MOT, METALFG_GRID_PYR_OUT, METALFG_GRID_FIXED } MetalFGGridKind;

/* Descriptor conditions */
#define METALFG_COND_ALWAYS 0
#define METALFG_COND_UI 1      /* create bit 0 */
#define METALFG_COND_DIST 2    /* create bit 2 */
#define METALFG_COND_CHAIN 4   /* create bit 4 clear: push-pull output chain */

typedef struct {
    const char *name;     /* fg.resources name; pair members are listed once, "X" with pair = 1 */
    MetalFGGridKind grid;
    uint32_t level;       /* pyramid level */
    uint32_t w, h;
    const char *format;   /* texture format name, or "buffer" */
    uint32_t bytes;       /* bytes per element */
    int pair;             /* history pair (cur and old exist) */
    int cond;
} MetalFGResDesc;

#define METALFG_MAX_RES 96
/* Fills out[]; returns the number of descriptors that exist for the create flags. */
int metalfg_plan_resources(const MetalFGGrids *g, uint32_t create_flags, MetalFGResDesc out[METALFG_MAX_RES]);

#endif
