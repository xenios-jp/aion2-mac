/* Weight file (sr.weights_format) and parameter file (sr.extractor) loading. */
#ifndef SR_MODEL_H
#define SR_MODEL_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    double r_wide, r_fast, phi_lo, phi_hi, kappa_mv, sigma_min, sigma_span, rho_max;
    double q_a, q_b, q_clip, g_beta, alpha_0, key_ae, eps_luma;
    int32_t n_luma_min;
} SrParams;

#define SR_NUM_TENSORS 34
#define SR_WEIGHT_FILE_BYTES 1903872u

typedef struct { const char *layer; int is_bias; uint32_t o, i, k; } SrTensorShape;
extern const SrTensorShape sr_tensor_shapes[SR_NUM_TENSORS];

/* Offsets from the shapes and the 256-byte alignment rule; returns the padded file size. */
size_t sr_tensor_offsets(size_t offsets[SR_NUM_TENSORS]);

/* Reads and checks the weight file (exact size, padding ignored). Returns malloc'd bytes or NULL with a message. */
uint8_t *sr_load_weights(const char *path, char *err, size_t cap);
/* Reads and checks "<weight path>.params". */
int sr_load_params(const char *weight_path, SrParams *out, char *err, size_t cap);
/* Serialises params in file format (for tests and tools); returns bytes written (1 + 16*40 + 16 + 32). */
size_t sr_write_params(const SrParams *p, uint8_t *out);

#endif
