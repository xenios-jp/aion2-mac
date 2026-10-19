/* Pass table and per-call schedule (spec fg.overview section 3, fg.orchestration "Order constraints").
   Pure data and logic; the encoders live in engine.m and pass_*.m. */
#ifndef METALFG_PASSES_H
#define METALFG_PASSES_H
#include "call.h"
#include "plan.h"

typedef struct {
    const char *tag;      /* F01..F28, FH1, FH2 */
    const char *name;     /* binding pass name of fg.overview */
    char kind;            /* C compute pass, N network, H host side */
    const char *inputs;   /* resource names of fg.resources read (and host inputs) */
    const char *outputs;  /* resource names written */
    const char *spec;     /* spec that defines the pass; "(pending)" when no approved spec covers it yet */
} MetalFGPassInfo;

const MetalFGPassInfo *metalfg_pass_table(int *count);
const MetalFGPassInfo *metalfg_pass_by_tag(const char *tag);

/* Passes that run in a call, in a valid sequential order. A pass listed twice runs twice (F13 in the distortion
   variant). Returns the number of entries written to tags[] (capacity 48). */
int metalfg_schedule(const MetalFGCallPlan *p, uint32_t create_flags, const MetalFGGrids *g, const char *tags[48]);

#endif
