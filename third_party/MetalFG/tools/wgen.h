/* Generator of synthetic weight and parameter files in the fg.weights_format / fg.extractor formats.
   Used by tests and by tools/genweights; the learned values never appear here. */
#ifndef METALFG_WGEN_H
#define METALFG_WGEN_H
#include <stdint.h>
#include <stdlib.h>
#include "weights.h"

typedef enum {
    WGEN_RANDOM, /* weights uniform in +-1/sqrt(fan-in), biases in +-0.1 (fg.net1_forward acceptance) */
    WGEN_INDEX,  /* element n of each tensor holds binary16(n mod 2048) (weights-index) */
    WGEN_ZERO,
    WGEN_ONES, /* all zero except the head biases of the two candidate logit pairs k_0 = k_2 = +24, k_1 = k_3 = 0: both blend weights are one */
} WgenMode;

typedef struct { uint64_t s; } WgenRng;
void wgen_seed(WgenRng *r, uint64_t seed);
uint64_t wgen_u64(WgenRng *r);
double wgen_uniform(WgenRng *r); /* [0, 1) */
uint16_t wgen_f16(double v);     /* round to nearest even binary16 */
double wgen_f16_to_d(uint16_t h);

/* Fill all 84 tensors (malloc'd arrays, caller frees each). Columns 10..15 of n2.a.d0 stay zero in random mode. */
void wgen_tensors(WgenMode mode, uint64_t seed, uint16_t *tensors[METALFG_TENSOR_COUNT]);
/* Parameter values used by tests: they satisfy every table requirement stated in the pass specs. */
MetalFGParams wgen_params(void);
int wgen_write(const char *weights_path, WgenMode mode, uint64_t seed, const MetalFGParams *params);

#endif
