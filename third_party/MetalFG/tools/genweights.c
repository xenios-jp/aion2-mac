/* genweights: write a synthetic weight file and its .params file for tests.
   usage: genweights [--mode random|index|zero] [--seed N] OUT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wgen.h"

int main(int argc, char **argv) {
    WgenMode mode = WGEN_RANDOM;
    uint64_t seed = 1;
    const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            mode = !strcmp(m, "index") ? WGEN_INDEX : !strcmp(m, "zero") ? WGEN_ZERO : WGEN_RANDOM;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 0);
        } else if (argv[i][0] != '-') {
            out = argv[i];
        } else {
            fprintf(stderr, "usage: genweights [--mode random|index|zero] [--seed N] OUT\n");
            return 1;
        }
    }
    if (!out) { fprintf(stderr, "usage: genweights [--mode random|index|zero] [--seed N] OUT\n"); return 1; }
    MetalFGParams p = wgen_params();
    return wgen_write(out, mode, seed, &p) != 0;
}
