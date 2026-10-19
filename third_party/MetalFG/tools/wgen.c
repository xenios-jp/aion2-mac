#include <math.h>
#include <stdio.h>
#include <string.h>
#include "wgen.h"

void wgen_seed(WgenRng *r, uint64_t seed) { r->s = seed * 0x9E3779B97F4A7C15ull + 0x1234567ull; }

uint64_t wgen_u64(WgenRng *r) { /* splitmix64 */
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double wgen_uniform(WgenRng *r) { return (double)(wgen_u64(r) >> 11) * 0x1p-53; }

uint16_t wgen_f16(double v) {
    _Float16 h = (_Float16)v;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

double wgen_f16_to_d(uint16_t b) {
    _Float16 h;
    memcpy(&h, &b, 2);
    return (double)h;
}

void wgen_tensors(WgenMode mode, uint64_t seed, uint16_t *tensors[METALFG_TENSOR_COUNT]) {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    WgenRng rng;
    wgen_seed(&rng, seed);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) {
        uint64_t n = metalfg_tensor_elems(&T[t]);
        tensors[t] = calloc(n, 2);
        if (mode == WGEN_ZERO) continue;
        if (mode == WGEN_ONES) {
            if (strcmp(T[t].name, "n1.head.b") == 0) for (int k = 0; k < 4; k++) tensors[t][k] = tensors[t][8 + k] = wgen_f16(24.0); /* k_0 and k_2 of the four pixels of a block */
            continue;
        }
        if (mode == WGEN_INDEX) {
            for (uint64_t k = 0; k < n; k++) tensors[t][k] = wgen_f16((double)(k % 2048));
            continue;
        }
        double amp = T[t].layout == 0 ? 1.0 / sqrt((double)T[t].kh * T[t].kw * T[t].I) : 0.1;
        for (uint64_t k = 0; k < n; k++) tensors[t][k] = wgen_f16((wgen_uniform(&rng) * 2 - 1) * amp);
        if (strcmp(T[t].name, "n2.a.d0.w") == 0) /* only the first ten input columns are real */
            for (uint64_t k = 0; k < n; k++)
                if (k % 16 >= 10) tensors[t][k] = 0;
    }
}

MetalFGParams wgen_params(void) {
    MetalFGParams p;
    memset(&p, 0, sizeof p);
    p.hm = 2160; p.hf = 1440; p.hb = 384;
    p.gx = 64; p.gy = 32;
    p.mt = 0.5f; p.mfa = 0.8f; p.mfh = 0.6f;
    p.th_ui = 0.0625f; p.th_uia = 0.5f;
    p.leak = 0.125f; p.th_mask = 0.5f; p.b_steer = 16.0f; p.sat_n2 = 8.0f;
    p.d_dyn = 4; p.d_sim = 4; p.th_cam = 1.0f;
    p.sp_min = 1; p.sp_max = 4; p.th_zn = 0.01f;
    return p;
}

int wgen_write(const char *path, WgenMode mode, uint64_t seed, const MetalFGParams *params) {
    uint16_t *tensors[METALFG_TENSOR_COUNT];
    wgen_tensors(mode, seed, tensors);
    size_t n;
    uint8_t *img = metalfg_weights_image((const uint16_t *const *)tensors, &n);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) free(tensors[t]);
    if (!img) return 1;
    char err[256];
    int rc = metalfg_write_file_atomic(path, img, n, err, sizeof err);
    free(img);
    if (rc) { fprintf(stderr, "%s\n", err); return rc; }
    size_t pn;
    uint8_t *pi = metalfg_params_image(params, &pn);
    char ppath[1100];
    snprintf(ppath, sizeof ppath, "%s.params", path);
    rc = metalfg_write_file_atomic(ppath, pi, pn, err, sizeof err);
    free(pi);
    if (rc) fprintf(stderr, "%s\n", err);
    return rc;
}
