#include "model.h"
#include <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

const SrTensorShape sr_tensor_shapes[SR_NUM_TENSORS] = {
    {"enc1a", 0, 32, 16, 3},  {"enc1a", 1, 32, 0, 0},
    {"enc1b", 0, 64, 32, 1},  {"enc1b", 1, 64, 0, 0},
    {"enc2", 0, 128, 64, 3},  {"enc2", 1, 128, 0, 0},
    {"enc3", 0, 128, 128, 3}, {"enc3", 1, 128, 0, 0},
    {"enc4", 0, 256, 128, 3}, {"enc4", 1, 256, 0, 0},
    {"bott_a", 0, 512, 256, 1}, {"bott_a", 1, 512, 0, 0},
    {"bott_b", 0, 256, 512, 1}, {"bott_b", 1, 256, 0, 0},
    {"dec4a", 0, 256, 256, 1}, {"dec4a", 1, 256, 0, 0},
    {"dec4b", 0, 128, 256, 1}, {"dec4b", 1, 128, 0, 0},
    {"dec3a", 0, 128, 128, 1}, {"dec3a", 1, 128, 0, 0},
    {"dec3b", 0, 128, 128, 1}, {"dec3b", 1, 128, 0, 0},
    {"dec2a", 0, 128, 128, 1}, {"dec2a", 1, 128, 0, 0},
    {"dec2b", 0, 64, 128, 1},  {"dec2b", 1, 64, 0, 0},
    {"dec1a", 0, 64, 64, 1},   {"dec1a", 1, 64, 0, 0},
    {"dec1b", 0, 32, 64, 1},   {"dec1b", 1, 32, 0, 0},
    {"head_a", 0, 32, 32, 1},  {"head_a", 1, 32, 0, 0},
    {"head_b", 0, 40, 32, 1},  {"head_b", 1, 40, 0, 0},
};

size_t sr_tensor_offsets(size_t offsets[SR_NUM_TENSORS]) {
    size_t pos = 0;
    for (int t = 0; t < SR_NUM_TENSORS; t++) {
        const SrTensorShape *s = &sr_tensor_shapes[t];
        pos = (pos + 255) & ~(size_t)255;
        offsets[t] = pos;
        pos += 2 * (s->is_bias ? s->o : (size_t)s->o * s->i * s->k * s->k);
    }
    return (pos + 255) & ~(size_t)255;
}

static void seterr(char *err, size_t cap, const char *fmt, const char *a, unsigned long b, unsigned long c) {
    if (err && cap) snprintf(err, cap, fmt, a, b, c);
}

uint8_t *sr_load_weights(const char *path, char *err, size_t cap) {
    size_t offs[SR_NUM_TENSORS];
    size_t want = sr_tensor_offsets(offs);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) { seterr(err, cap, "unreadable weight file %s", path, 0, 0); return NULL; }
    if ((size_t)st.st_size != want) {
        seterr(err, cap, "weight file %s has %lu bytes, expected %lu", path, (unsigned long)st.st_size, (unsigned long)want);
        return NULL;
    }
    FILE *f = fopen(path, "rb");
    uint8_t *buf = f ? malloc(want) : NULL;
    if (!buf || fread(buf, 1, want, f) != want) { if (f) fclose(f); free(buf); seterr(err, cap, "cannot read weight file %s", path, 0, 0); return NULL; }
    fclose(f);
    return buf;
}

static const char *kNames[16] = {"r_wide", "r_fast", "phi_lo", "phi_hi", "kappa_mv", "sigma_min", "sigma_span", "rho_max",
                                 "q_a", "q_b", "q_clip", "g_beta", "alpha_0", "key_ae", "eps_luma", "n_luma_min"};

static double *param_slot(SrParams *p, int i) {
    double *d[15] = {&p->r_wide, &p->r_fast, &p->phi_lo, &p->phi_hi, &p->kappa_mv, &p->sigma_min, &p->sigma_span, &p->rho_max,
                     &p->q_a, &p->q_b, &p->q_clip, &p->g_beta, &p->alpha_0, &p->key_ae, &p->eps_luma};
    return d[i];
}

#define PARAM_BYTES (16 + 16 * 40 + 32)

/* stored type of every entry (sr.extractor_r2): 0 = fp32, 1 = fp64, 2 = int32 */
static const uint32_t kTypes[16] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};

size_t sr_write_params(const SrParams *p, uint8_t *out) {
    memset(out, 0, PARAM_BYTES);
    memcpy(out, "SRPARAM1", 8);
    uint32_t n = 16; memcpy(out + 8, &n, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t *e = out + 16 + 40 * i;
        strncpy((char *)e, kNames[i], 24);
        uint32_t type = kTypes[i]; memcpy(e + 24, &type, 4);
        if (type == 2) memcpy(e + 32, &p->n_luma_min, 4);
        else if (type == 1) { double v = *param_slot((SrParams *)p, i); memcpy(e + 32, &v, 8); }
        else { float v = (float)*param_slot((SrParams *)p, i); memcpy(e + 32, &v, 4); }
    }
    CC_SHA256(out, 16 + 16 * 40, out + 16 + 16 * 40);
    return PARAM_BYTES;
}

int sr_load_params(const char *weight_path, SrParams *out, char *err, size_t cap) {
    char path[4096];
    snprintf(path, sizeof path, "%s.params", weight_path);
    uint8_t b[PARAM_BYTES + 1];
    FILE *f = fopen(path, "rb");
    size_t got = f ? fread(b, 1, sizeof b, f) : 0;
    if (f) fclose(f);
    if (got != PARAM_BYTES) { seterr(err, cap, "parameter file %s has wrong size or is unreadable", path, 0, 0); return -1; }
    if (memcmp(b, "SRPARAM1", 8)) { seterr(err, cap, "parameter file %s: bad magic", path, 0, 0); return -1; }
    uint32_t n; memcpy(&n, b + 8, 4);
    if (n != 16) { seterr(err, cap, "parameter file %s: entry count ", path, n, 0); return -1; }
    uint8_t dig[32];
    CC_SHA256(b, 16 + 16 * 40, dig);
    if (memcmp(dig, b + 16 + 16 * 40, 32)) { seterr(err, cap, "parameter file %s: checksum mismatch", path, 0, 0); return -1; }
    for (int i = 0; i < 16; i++) {
        const uint8_t *e = b + 16 + 40 * i;
        char name[25] = {0}; memcpy(name, e, 24);
        char m[240];
        if (strcmp(name, kNames[i])) { snprintf(m, sizeof m, "parameter file: entry %d is '%s', expected '%s'", i, name, kNames[i]); seterr(err, cap, m, "", 0, 0); return -1; }
        uint32_t type; memcpy(&type, e + 24, 4);
        if (type != kTypes[i]) { snprintf(m, sizeof m, "parameter file: entry '%s' has type code %u, expected %u", kNames[i], type, kTypes[i]); seterr(err, cap, m, "", 0, 0); return -1; }
        double v;
        if (type == 2) { memcpy(&out->n_luma_min, e + 32, 4); continue; }
        if (type == 1) memcpy(&v, e + 32, 8);
        else { float x; memcpy(&x, e + 32, 4); v = x; }
        if (!isfinite(v)) { snprintf(m, sizeof m, "parameter file: entry '%s' is not finite", kNames[i]); seterr(err, cap, m, "", 0, 0); return -1; }
        *param_slot(out, i) = v;
    }
    return 0;
}