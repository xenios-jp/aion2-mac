#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "sha256.h"
#include "weights.h"

typedef struct { const char *name; uint32_t O, k, I; } LayerDef;

/* Layer chain in file order (fg.weights_format table). */
static const LayerDef LAYERS[] = {
    {"n1.enc0", 32, 3, 48}, {"n1.enc1", 64, 1, 32}, {"n1.enc2", 128, 3, 64}, {"n1.enc3", 128, 3, 128},
    {"n1.enc4", 256, 3, 128}, {"n1.mida", 512, 1, 256}, {"n1.midb", 256, 1, 512}, {"n1.dec4a", 256, 1, 256},
    {"n1.dec4b", 128, 1, 256}, {"n1.dec3a", 128, 1, 128}, {"n1.dec3b", 128, 1, 128}, {"n1.dec2a", 128, 1, 128},
    {"n1.dec2b", 64, 1, 128}, {"n1.dec1a", 64, 1, 64}, {"n1.dec1b", 32, 1, 64}, {"n1.dec0a", 32, 1, 32},
    {"n1.head", 16, 1, 32},
    {"n2.a.d0", 32, 3, 16}, {"n2.a.d1", 32, 3, 32}, {"n2.a.d2", 64, 3, 32},
    {"n2.a.res0", 64, 3, 64}, {"n2.a.res1", 64, 3, 64}, {"n2.a.res2", 64, 3, 64}, {"n2.a.res3", 64, 3, 64},
    {"n2.a.res4", 64, 3, 64}, {"n2.a.res5", 64, 3, 64}, {"n2.a.res6", 64, 3, 64}, {"n2.a.res7", 64, 3, 64},
    {"n2.a.headA", 96, 3, 64}, {"n2.a.headB", 8, 3, 32},
    {"n2.b.d0", 16, 3, 18}, {"n2.b.d1", 32, 3, 16},
    {"n2.b.res0", 32, 3, 32}, {"n2.b.res1", 32, 3, 32}, {"n2.b.res2", 32, 3, 32}, {"n2.b.res3", 32, 3, 32},
    {"n2.b.res4", 32, 3, 32}, {"n2.b.res5", 32, 3, 32}, {"n2.b.res6", 32, 3, 32}, {"n2.b.res7", 32, 3, 32},
    {"n2.b.headA", 48, 3, 32}, {"n2.b.headB", 8, 3, 16},
};
#define NLAYERS (sizeof LAYERS / sizeof LAYERS[0])

static MetalFGTensorInfo g_table[METALFG_TENSOR_COUNT];
static int g_table_ready;

static uint64_t up256(uint64_t v) { return (v + 255) & ~(uint64_t)255; }

const MetalFGTensorInfo *metalfg_weight_table(void) {
    if (g_table_ready) return g_table;
    _Static_assert(NLAYERS * 2 == METALFG_TENSOR_COUNT, "layer count");
    uint64_t off = METALFG_WEIGHT_DATA_OFFSET;
    for (size_t l = 0; l < NLAYERS; l++) {
        MetalFGTensorInfo *w = &g_table[2 * l], *b = &g_table[2 * l + 1];
        snprintf(w->name, sizeof w->name, "%s.w", LAYERS[l].name);
        w->O = LAYERS[l].O; w->kh = w->kw = LAYERS[l].k; w->I = LAYERS[l].I; w->layout = 0;
        w->offset = off;
        w->bytes = 2ull * w->O * w->kh * w->kw * w->I;
        off = up256(off + w->bytes);
        snprintf(b->name, sizeof b->name, "%s.b", LAYERS[l].name);
        b->O = LAYERS[l].O; b->layout = 1;
        b->offset = off;
        b->bytes = 2ull * b->O;
        off = up256(off + b->bytes);
    }
    g_table_ready = 1;
    return g_table;
}

int metalfg_weight_index(const char *name) {
    const MetalFGTensorInfo *t = metalfg_weight_table();
    for (int i = 0; i < METALFG_TENSOR_COUNT; i++)
        if (strcmp(t[i].name, name) == 0) return i;
    return -1;
}

uint64_t metalfg_tensor_elems(const MetalFGTensorInfo *t) { return t->bytes / 2; }

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static void wr32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void wr64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static int fail(char *err, size_t cap, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#include <stdarg.h>
static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return 1;
}

int metalfg_weights_validate(const uint8_t *d, size_t size, char *err, size_t cap) {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    if (size < METALFG_WEIGHT_DATA_OFFSET) return fail(err, cap, "weight file: shorter than the header and table (%zu bytes)", size);
    if (memcmp(d, "FGWEIGHT", 8) != 0) return fail(err, cap, "weight file: header field magic is wrong");
    if (rd32(d + 8) != 1) return fail(err, cap, "weight file: header field version is %u, expected 1", rd32(d + 8));
    if (rd32(d + 12) != METALFG_TENSOR_COUNT) return fail(err, cap, "weight file: header field tensor_count is %u, expected %d", rd32(d + 12), METALFG_TENSOR_COUNT);
    if (rd64(d + 16) != METALFG_WEIGHT_HEADER_BYTES) return fail(err, cap, "weight file: header field table_offset is %llu, expected 128", (unsigned long long)rd64(d + 16));
    if (rd64(d + 24) != METALFG_WEIGHT_DATA_OFFSET) return fail(err, cap, "weight file: header field data_offset is %llu, expected 8192", (unsigned long long)rd64(d + 24));
    uint64_t fs = rd64(d + 32);
    if (fs % 256 != 0 || fs != METALFG_WEIGHT_FILE_BYTES) return fail(err, cap, "weight file: header field file_size is %llu, expected %u", (unsigned long long)fs, METALFG_WEIGHT_FILE_BYTES);
    if (fs != size) return fail(err, cap, "weight file: header field file_size (%llu) differs from the real size (%zu)", (unsigned long long)fs, size);
    for (int i = 72; i < 128; i++)
        if (d[i]) return fail(err, cap, "weight file: header field reserved is not zero (byte %d)", i);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) {
        const uint8_t *e = d + METALFG_WEIGHT_HEADER_BYTES + 96 * t;
        const MetalFGTensorInfo *x = &T[t];
        char nm[49] = {0};
        memcpy(nm, e, 48);
        char want[48] = {0};
        memcpy(want, x->name, strlen(x->name));
        if (memcmp(e, want, 48) != 0) return fail(err, cap, "weight file: table entry %d name is '%s', expected '%s'", t, nm, x->name);
        if (rd32(e + 48) != 0) return fail(err, cap, "weight file: tensor %s: elem_type must be 0", x->name);
        if (rd32(e + 52) != x->layout) return fail(err, cap, "weight file: tensor %s: layout is %u, expected %u", x->name, rd32(e + 52), x->layout);
        uint32_t rank = x->layout == 0 ? 4 : 1;
        if (rd32(e + 56) != rank) return fail(err, cap, "weight file: tensor %s: rank is %u, expected %u", x->name, rd32(e + 56), rank);
        uint32_t dims[4] = {x->layout == 0 ? x->O : x->O, x->kh, x->kw, x->I};
        if (x->layout) dims[1] = dims[2] = dims[3] = 0;
        for (int k = 0; k < 4; k++)
            if (rd32(e + 60 + 4 * k) != dims[k]) return fail(err, cap, "weight file: tensor %s: dims[%d] is %u, expected %u", x->name, k, rd32(e + 60 + 4 * k), dims[k]);
        if (rd32(e + 76) != 0) return fail(err, cap, "weight file: tensor %s: reserved is not zero", x->name);
        if (rd64(e + 80) != x->offset) return fail(err, cap, "weight file: tensor %s: offset is %llu, expected %llu", x->name, (unsigned long long)rd64(e + 80), (unsigned long long)x->offset);
        if (rd64(e + 88) != x->bytes) return fail(err, cap, "weight file: tensor %s: bytes is %llu, expected %llu", x->name, (unsigned long long)rd64(e + 88), (unsigned long long)x->bytes);
    }
    for (int i = 128 + 96 * METALFG_TENSOR_COUNT; i < METALFG_WEIGHT_DATA_OFFSET; i++)
        if (d[i]) return fail(err, cap, "weight file: table padding is not zero (byte %d)", i);
    uint8_t sum[32];
    metalfg_sha256(d + METALFG_WEIGHT_DATA_OFFSET, size - METALFG_WEIGHT_DATA_OFFSET, sum);
    if (memcmp(sum, d + 40, 32) != 0) return fail(err, cap, "weight file: header field payload_sha256 does not match the payload");
    return 0;
}

int metalfg_weights_load(const char *path, MetalFGWeightFile *out, char *err, size_t cap) {
    memset(out, 0, sizeof *out);
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, cap, "weight file: cannot open '%s'", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return fail(err, cap, "weight file: cannot size '%s'", path); }
    uint8_t *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return fail(err, cap, "weight file: cannot read '%s'", path); }
    fclose(f);
    if (metalfg_weights_validate(buf, (size_t)n, err, cap)) { free(buf); return 1; }
    out->data = buf;
    out->size = (size_t)n;
    return 0;
}

void metalfg_weights_free(MetalFGWeightFile *w) { free(w->data); w->data = NULL; w->size = 0; }

const uint16_t *metalfg_weights_tensor(const MetalFGWeightFile *w, int idx) {
    return (const uint16_t *)(const void *)(w->data + metalfg_weight_table()[idx].offset);
}

uint16_t metalfg_weight_elem(const MetalFGWeightFile *w, int idx, uint32_t o, uint32_t ky, uint32_t kx, uint32_t i) {
    const MetalFGTensorInfo *t = &metalfg_weight_table()[idx];
    const uint8_t *p = w->data + t->offset;
    uint64_t n = (((uint64_t)o * t->kh + ky) * t->kw + kx) * t->I + i;
    return (uint16_t)(p[2 * n] | p[2 * n + 1] << 8);
}

uint8_t *metalfg_weights_image(const uint16_t *const tensors[METALFG_TENSOR_COUNT], size_t *size_out) {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    uint8_t *d = calloc(1, METALFG_WEIGHT_FILE_BYTES);
    if (!d) return NULL;
    memcpy(d, "FGWEIGHT", 8);
    wr32(d + 8, 1);
    wr32(d + 12, METALFG_TENSOR_COUNT);
    wr64(d + 16, METALFG_WEIGHT_HEADER_BYTES);
    wr64(d + 24, METALFG_WEIGHT_DATA_OFFSET);
    wr64(d + 32, METALFG_WEIGHT_FILE_BYTES);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) {
        uint8_t *e = d + METALFG_WEIGHT_HEADER_BYTES + 96 * t;
        memcpy(e, T[t].name, strlen(T[t].name));
        wr32(e + 48, 0);
        wr32(e + 52, T[t].layout);
        wr32(e + 56, T[t].layout ? 1 : 4);
        wr32(e + 60, T[t].O);
        if (!T[t].layout) { wr32(e + 64, T[t].kh); wr32(e + 68, T[t].kw); wr32(e + 72, T[t].I); }
        wr64(e + 80, T[t].offset);
        wr64(e + 88, T[t].bytes);
        for (uint64_t k = 0; k < T[t].bytes / 2; k++) {
            d[T[t].offset + 2 * k] = (uint8_t)tensors[t][k];
            d[T[t].offset + 2 * k + 1] = (uint8_t)(tensors[t][k] >> 8);
        }
    }
    metalfg_sha256(d + METALFG_WEIGHT_DATA_OFFSET, METALFG_WEIGHT_FILE_BYTES - METALFG_WEIGHT_DATA_OFFSET, d + 40);
    *size_out = METALFG_WEIGHT_FILE_BYTES;
    return d;
}

int metalfg_write_file_atomic(const char *path, const void *data, size_t size, char *err, size_t cap) {
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f) return fail(err, cap, "cannot create '%s'", tmp);
    if (fwrite(data, 1, size, f) != size || fflush(f) != 0 || fclose(f) != 0) { remove(tmp); return fail(err, cap, "cannot write '%s'", tmp); }
    chmod(tmp, 0644);
    if (rename(tmp, path) != 0) { remove(tmp); return fail(err, cap, "cannot rename onto '%s'", path); }
    return 0;
}

/* ---- parameter file ---- */
typedef struct { const char *name; int type; size_t off; } ParamDef;
#define P(n, t, f) {n, t, offsetof(MetalFGParams, f)}
static const ParamDef PARAMS[METALFG_PARAM_COUNT] = {
    P("hm", 2, hm), P("hf", 2, hf), P("hb", 2, hb), P("gx", 2, gx), P("gy", 2, gy),
    P("mt", 0, mt), P("mfa", 0, mfa), P("mfh", 0, mfh), P("th_ui", 0, th_ui), P("th_uia", 0, th_uia),
    P("leak", 0, leak), P("th_mask", 0, th_mask), P("b_steer", 0, b_steer), P("sat_n2", 0, sat_n2),
    P("d_dyn", 2, d_dyn), P("d_sim", 2, d_sim), P("th_cam", 0, th_cam), P("sp_min", 2, sp_min),
    P("sp_max", 2, sp_max), P("th_zn", 0, th_zn),
};

int metalfg_params_parse(const uint8_t *d, size_t size, MetalFGParams *out, char *err, size_t cap) {
    if (size != METALFG_PARAM_FILE_BYTES) return fail(err, cap, "parameter file: size is %zu, expected %d", size, METALFG_PARAM_FILE_BYTES);
    if (memcmp(d, "FGPARAM1", 8) != 0) return fail(err, cap, "parameter file: magic is wrong");
    if (rd32(d + 8) != METALFG_PARAM_COUNT) return fail(err, cap, "parameter file: entry count is %u, expected %d", rd32(d + 8), METALFG_PARAM_COUNT);
    if (rd32(d + 12) != 0) return fail(err, cap, "parameter file: reserved word is not zero");
    MetalFGParams p;
    memset(&p, 0, sizeof p);
    for (int i = 0; i < METALFG_PARAM_COUNT; i++) {
        const uint8_t *e = d + 16 + 40 * i;
        char nm[25] = {0};
        memcpy(nm, e, 24);
        if (strcmp(nm, PARAMS[i].name) != 0) return fail(err, cap, "parameter file: entry %d is named '%s', expected '%s'", i, nm, PARAMS[i].name);
        if ((int)rd32(e + 24) != PARAMS[i].type) return fail(err, cap, "parameter file: %s has type code %u, expected %d", PARAMS[i].name, rd32(e + 24), PARAMS[i].type);
        if (rd32(e + 28) != 0) return fail(err, cap, "parameter file: %s: reserved word is not zero", PARAMS[i].name);
        if (rd32(e + 36) != 0) return fail(err, cap, "parameter file: %s: upper value word is not zero", PARAMS[i].name);
        uint32_t bits = rd32(e + 32);
        if (PARAMS[i].type == 0) {
            float v;
            memcpy(&v, &bits, 4);
            if (!isfinite(v)) return fail(err, cap, "parameter file: %s is not finite", PARAMS[i].name);
            memcpy((uint8_t *)&p + PARAMS[i].off, &v, 4);
        } else {
            int32_t v = (int32_t)bits;
            memcpy((uint8_t *)&p + PARAMS[i].off, &v, 4);
        }
    }
    uint8_t sum[32];
    metalfg_sha256(d, size - 32, sum);
    if (memcmp(sum, d + size - 32, 32) != 0) return fail(err, cap, "parameter file: trailing checksum does not match");
    *out = p;
    return 0;
}

int metalfg_params_load(const char *path, MetalFGParams *out, char *err, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, cap, "parameter file: cannot open '%s'", path);
    uint8_t buf[METALFG_PARAM_FILE_BYTES + 1];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    return metalfg_params_parse(buf, n, out, err, cap);
}

uint8_t *metalfg_params_image(const MetalFGParams *p, size_t *size_out) {
    uint8_t *d = calloc(1, METALFG_PARAM_FILE_BYTES);
    if (!d) return NULL;
    memcpy(d, "FGPARAM1", 8);
    wr32(d + 8, METALFG_PARAM_COUNT);
    for (int i = 0; i < METALFG_PARAM_COUNT; i++) {
        uint8_t *e = d + 16 + 40 * i;
        memcpy(e, PARAMS[i].name, strlen(PARAMS[i].name));
        wr32(e + 24, (uint32_t)PARAMS[i].type);
        uint32_t bits;
        memcpy(&bits, (const uint8_t *)p + PARAMS[i].off, 4);
        wr32(e + 32, bits);
    }
    metalfg_sha256(d, METALFG_PARAM_FILE_BYTES - 32, d + METALFG_PARAM_FILE_BYTES - 32);
    *size_out = METALFG_PARAM_FILE_BYTES;
    return d;
}
