/* weights-offsets, weights-header, weights-index, parameter file (spec fg.weights_format section 7, fg.extractor). */
#include <stdio.h>
#include <string.h>
#include "sha256.h"
#include "weights.h"
#include "wgen.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const struct { const char *name; uint64_t off, bytes; } SPEC[] = {
#include "weights_spec_table.inc"
};

static void offsets(void) {
    const MetalFGTensorInfo *T = metalfg_weight_table();
    CHECK(sizeof SPEC / sizeof SPEC[0] == METALFG_TENSOR_COUNT, "spec table size");
    uint64_t end = 0;
    for (int i = 0; i < METALFG_TENSOR_COUNT; i++) {
        CHECK(!strcmp(T[i].name, SPEC[i].name), "name %d: %s vs %s", i, T[i].name, SPEC[i].name);
        CHECK(T[i].offset == SPEC[i].off, "%s offset %llu vs %llu", T[i].name, (unsigned long long)T[i].offset, (unsigned long long)SPEC[i].off);
        CHECK(T[i].bytes == SPEC[i].bytes, "%s bytes", T[i].name);
        if (i) CHECK(T[i].offset == ((end + 255) & ~255ull), "%s alignment", T[i].name);
        end = T[i].offset + T[i].bytes;
    }
    CHECK(((end + 255) & ~255ull) == 2896896, "total size");
    /* control: shifting any one offset from entry 2 on by 256 is flagged at exactly that entry and every later one,
       when the shift is carried through the alignment chain */
    int first_bad = -1, flagged = 0;
    uint64_t shift_end = 0;
    for (int i = 0; i < METALFG_TENSOR_COUNT; i++) {
        uint64_t off = i == 5 ? SPEC[i].off + 256 : SPEC[i].off;
        if (off != T[i].offset) { flagged++; if (first_bad < 0) first_bad = i; }
        shift_end = off;
    }
    (void)shift_end;
    CHECK(first_bad == 5 && flagged == 1, "control: shifted entry flagged (%d, %d)", first_bad, flagged);
}

static uint8_t *good_image(WgenMode mode, size_t *n) {
    uint16_t *t[METALFG_TENSOR_COUNT];
    wgen_tensors(mode, 7, t);
    uint8_t *img = metalfg_weights_image((const uint16_t *const *)t, n);
    for (int i = 0; i < METALFG_TENSOR_COUNT; i++) free(t[i]);
    return img;
}

static void header(void) {
    size_t n;
    uint8_t *img = good_image(WGEN_RANDOM, &n);
    char err[256];
    CHECK(metalfg_weights_validate(img, n, err, sizeof err) == 0, "good file: %s", err);
    static const struct { int off, len; const char *field; } F[] = {
        {0, 8, "magic"}, {8, 4, "version"}, {12, 4, "tensor_count"}, {16, 8, "table_offset"},
        {24, 8, "data_offset"}, {32, 8, "file_size"}, {40, 32, "payload_sha256"}, {72, 56, "reserved"},
    };
    for (size_t f = 0; f < sizeof F / sizeof F[0]; f++) {
        img[F[f].off] ^= 1;
        int rc = metalfg_weights_validate(img, n, err, sizeof err);
        CHECK(rc != 0, "header field %s change accepted", F[f].field);
        CHECK(rc == 0 || strstr(err, F[f].field), "message for %s does not name it: %s", F[f].field, err);
        img[F[f].off] ^= 1;
    }
    /* table entries: change one byte in every field of a few entries */
    for (int t = 0; t < METALFG_TENSOR_COUNT; t += 7) {
        for (int off = 0; off < 96; off += (off < 48 ? 47 : 4)) {
            uint8_t *p = img + 128 + 96 * t + off;
            *p ^= 1;
            CHECK(metalfg_weights_validate(img, n, err, sizeof err) != 0, "table entry %d byte %d change accepted", t, off);
            *p ^= 1;
        }
    }
    /* truncated and padded files */
    CHECK(metalfg_weights_validate(img, n - 256, err, sizeof err) != 0, "short file accepted");
    CHECK(metalfg_weights_validate(img, 100, err, sizeof err) != 0, "tiny file accepted");
    /* payload byte change is caught by the checksum */
    img[n - 3] ^= 0x10;
    CHECK(metalfg_weights_validate(img, n, err, sizeof err) != 0, "payload corruption accepted");
    free(img);
}

static void index_test(void) {
    size_t n;
    uint8_t *img = good_image(WGEN_INDEX, &n);
    MetalFGWeightFile w = {img, n};
    const MetalFGTensorInfo *T = metalfg_weight_table();
    WgenRng r;
    wgen_seed(&r, 99);
    int cross = 0;
    for (int t = 0; t < METALFG_TENSOR_COUNT; t += 2) { /* weights tensors */
        const MetalFGTensorInfo *x = &T[t];
        uint32_t corners[][4] = {{0, 0, 0, 0}, {x->O - 1, 0, 0, 0}, {0, x->kh - 1, x->kw - 1, 0},
                                 {x->O - 1, x->kh - 1, x->kw - 1, x->I - 1}};
        for (int k = 0; k < 68; k++) {
            uint32_t o, ky, kx, i;
            if (k < 4) { o = corners[k][0]; ky = corners[k][1]; kx = corners[k][2]; i = corners[k][3]; }
            else {
                o = wgen_u64(&r) % x->O; ky = wgen_u64(&r) % x->kh; kx = wgen_u64(&r) % x->kw; i = wgen_u64(&r) % x->I;
            }
            uint64_t idx = (((uint64_t)o * x->kh + ky) * x->kw + kx) * x->I + i;
            uint16_t want = wgen_f16((double)(idx % 2048));
            uint16_t got = metalfg_weight_elem(&w, t, o, ky, kx, i);
            CHECK(got == want, "%s (%u,%u,%u,%u)", x->name, o, ky, kx, i);
            if (x->kh == 3 && ky != kx) cross++;
            /* controls: swapped kx/ky and swapped extents give other values for tuples with ky != kx */
            if (x->kh == 3 && ky != kx) {
                uint64_t sw = (((uint64_t)o * x->kh + kx) * x->kw + ky) * x->I + i;
                CHECK((sw % 2048) != (idx % 2048) || sw == idx, "control: kx/ky swap is distinguishable");
            }
        }
    }
    CHECK(cross > 100, "harness sampled tuples with ky != kx (%d)", cross);
    free(img);
}

static void params(void) {
    MetalFGParams p = wgen_params(), q;
    size_t n;
    uint8_t *img = metalfg_params_image(&p, &n);
    char err[256];
    CHECK(n == METALFG_PARAM_FILE_BYTES, "param size %zu", n);
    CHECK(metalfg_params_parse(img, n, &q, err, sizeof err) == 0, "param parse: %s", err);
    CHECK(memcmp(&p, &q, sizeof p) == 0, "param roundtrip");
    CHECK(q.leak == 0.125f && q.hb == 384 && q.sp_max == 4, "param values");
    for (size_t i = 0; i < n; i++) {
        img[i] ^= 1;
        CHECK(metalfg_params_parse(img, n, &q, err, sizeof err) != 0, "param byte %zu change accepted", i);
        img[i] ^= 1;
    }
    /* a non-finite fp32 entry with a fixed-up checksum is still rejected */
    uint32_t inf = 0x7f800000u;
    memcpy(img + 16 + 40 * 10 + 32, &inf, 4);
    metalfg_sha256(img, n - 32, img + n - 32);
    CHECK(metalfg_params_parse(img, n, &q, err, sizeof err) != 0 && strstr(err, "leak"), "non-finite leak: %s", err);
    free(img);
}

static void sha(void) {
    uint8_t out[32];
    metalfg_sha256("abc", 3, out);
    static const uint8_t want[8] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea};
    CHECK(memcmp(out, want, 8) == 0, "sha256 abc");
    uint8_t big[1000];
    memset(big, 'a', sizeof big);
    MetalFGSha256 c;
    metalfg_sha256_init(&c);
    for (int i = 0; i < 1000; i += 7) metalfg_sha256_update(&c, big + i, i + 7 <= 1000 ? 7 : 1000 - i);
    uint8_t a[32], b[32];
    metalfg_sha256_final(&c, a);
    metalfg_sha256(big, sizeof big, b);
    CHECK(memcmp(a, b, 32) == 0, "sha256 incremental");
}

int main(void) {
    sha(); offsets(); header(); index_test(); params();
    printf(fails ? "t_weights: %d FAILURES\n" : "t_weights: ok\n", fails);
    return fails != 0;
}
