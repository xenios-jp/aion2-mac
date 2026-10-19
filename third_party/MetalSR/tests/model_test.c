/* weights-offsets, weights-tail, parameter file checks. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CommonCrypto/CommonDigest.h>
#include <string.h>
#include "../src/model.h"

static int fails;
#define EXPECT(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static void write_file(const char *path, const uint8_t *d, size_t n) { FILE *f = fopen(path, "wb"); fwrite(d, 1, n, f); fclose(f); }

int main(void) {
    size_t off[SR_NUM_TENSORS];
    size_t total = sr_tensor_offsets(off);
    static const size_t table[SR_NUM_TENSORS] = {0, 9216, 9472, 13568, 13824, 161280, 161536, 456448, 456704, 1046528, 1047040,
        1309184, 1310208, 1572352, 1572864, 1703936, 1704448, 1769984, 1770240, 1803008, 1803264, 1836032, 1836288, 1869056,
        1869312, 1885696, 1885952, 1894144, 1894400, 1898496, 1898752, 1900800, 1901056, 1903616};
    for (int i = 0; i < SR_NUM_TENSORS; i++) EXPECT(off[i] == table[i], "offset %d: %zu vs %zu", i, off[i], table[i]);
    EXPECT(total == 1903872 && total == SR_WEIGHT_FILE_BYTES, "total %zu", total);

    char err[300];
    uint8_t *img = calloc(1, total);
    for (size_t i = 0; i < total; i++) img[i] = (uint8_t)(i * 7);
    write_file("build/t_w.bin", img, total);
    uint8_t *w = sr_load_weights("build/t_w.bin", err, sizeof err);
    EXPECT(w && !memcmp(w, img, total), "load");
    free(w);
    write_file("build/t_w_short.bin", img, total - 1);
    EXPECT(!sr_load_weights("build/t_w_short.bin", err, sizeof err), "short file must be rejected");
    printf("  message: %s\n", err);
    EXPECT(strstr(err, "build/t_w_short.bin") && strstr(err, "1903871") && strstr(err, "1903872"), "message names path and sizes");
    write_file("build/t_w_trunc.bin", img, 1903616 + 80);
    EXPECT(!sr_load_weights("build/t_w_trunc.bin", err, sizeof err), "file without end padding must be rejected");
    EXPECT(!sr_load_weights("build/does_not_exist.bin", err, sizeof err), "missing file");

    SrParams p = {2.5, 1.8, 0.0625, 0.5, 8, 0.3, 0.7, 0.9, 1, 1.75, 1.5, 1, 0.5, 0.18, 1e-4, 16}, q;
    uint8_t pb[1024];
    size_t n = sr_write_params(&p, pb);
    write_file("build/t_w.bin.params", pb, n);
    memset(&q, 0, sizeof q);
    EXPECT(sr_load_params("build/t_w.bin", &q, err, sizeof err) == 0, "params load: %s", err);
    EXPECT(q.r_wide == 2.5 && q.n_luma_min == 16 && q.phi_lo == 0.0625 && q.key_ae == (double)(float)0.18, "params values");
    pb[20] ^= 1;
    write_file("build/t_w.bin.params", pb, n);
    EXPECT(sr_load_params("build/t_w.bin", &q, err, sizeof err) != 0, "corrupted params must be rejected");
    printf("  message: %s\n", err);
    pb[20] ^= 1; memcpy(pb, "SRPARAM2", 8);
    write_file("build/t_w.bin.params", pb, n);
    EXPECT(sr_load_params("build/t_w.bin", &q, err, sizeof err) != 0 && strstr(err, "magic"), "bad magic");
    /* params-types (sr.extractor_r2): r_wide, r_fast are fp64, n_luma_min is int32, the rest fp32; wrong codes and non-finite entries are rejected */
    {
        uint8_t good[1024]; size_t gn = sr_write_params(&p, good);
        uint32_t t0; memcpy(&t0, good + 16 + 24, 4);
        uint32_t t2; memcpy(&t2, good + 16 + 40 * 2 + 24, 4);
        uint32_t t15; memcpy(&t15, good + 16 + 40 * 15 + 24, 4);
        EXPECT(t0 == 1 && t2 == 0 && t15 == 2, "entry type codes %u %u %u", t0, t2, t15);
        write_file("build/t_w.bin.params", good, gn);
        EXPECT(sr_load_params("build/t_w.bin", &q, err, sizeof err) == 0, "harness-written file passes: %s", err);
        for (int variant = 0; variant < 2; variant++) {
            uint8_t bad[1024]; memcpy(bad, good, gn);
            if (variant == 0) { uint32_t wrong = 0; memcpy(bad + 16 + 40 * 0 + 24, &wrong, 4); }                       /* r_wide stored as fp32 */
            else { float nanv = NAN; memcpy(bad + 16 + 40 * 5 + 32, &nanv, 4); }                                        /* sigma_min not finite */
            CC_SHA256(bad, 16 + 16 * 40, bad + 16 + 16 * 40);
            write_file("build/t_w.bin.params", bad, gn);
            int rc = sr_load_params("build/t_w.bin", &q, err, sizeof err);
            printf("  %s: %s\n", variant == 0 ? "wrong type code" : "non-finite entry", err);
            EXPECT(rc != 0 && strstr(err, variant == 0 ? "r_wide" : "sigma_min"), "params-types variant %d must be rejected naming the entry", variant);
        }
    }
    printf(fails ? "model_test: %d FAILURES\n" : "model_test: all ok\n", fails);
    return fails != 0;
}
