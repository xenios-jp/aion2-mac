/* Weight file and parameter file (specs fg.weights_format, fg.extractor "parameter file", fg.numerics named parameters). */
#ifndef METALFG_WEIGHTS_H
#define METALFG_WEIGHTS_H
#include <stddef.h>
#include <stdint.h>

#define METALFG_TENSOR_COUNT 84
#define METALFG_WEIGHT_HEADER_BYTES 128
#define METALFG_WEIGHT_DATA_OFFSET 8192
#define METALFG_WEIGHT_FILE_BYTES 2896896u

typedef struct {
    char name[48];
    uint32_t O, kh, kw, I; /* weights [O][kh][kw][I]; biases O only (kh = kw = I = 0) */
    uint32_t layout;       /* 0 weights, 1 plain vector */
    uint64_t offset;       /* byte offset from the start of the file */
    uint64_t bytes;        /* size without padding */
} MetalFGTensorInfo;

/* The 84-entry table of fg.weights_format, offsets computed from the alignment rule. */
const MetalFGTensorInfo *metalfg_weight_table(void);
/* Index of a tensor by name, -1 when unknown. */
int metalfg_weight_index(const char *name);
/* Element count of a tensor. */
uint64_t metalfg_tensor_elems(const MetalFGTensorInfo *t);

typedef struct {
    uint8_t *data; /* whole file, malloc'd */
    size_t size;
} MetalFGWeightFile;

/* Validate a file image; 0 on success, else a message naming the field or entry. */
int metalfg_weights_validate(const uint8_t *data, size_t size, char *err, size_t cap);
int metalfg_weights_load(const char *path, MetalFGWeightFile *out, char *err, size_t cap);
void metalfg_weights_free(MetalFGWeightFile *w);
const uint16_t *metalfg_weights_tensor(const MetalFGWeightFile *w, int idx);
/* Element (o, ky, kx, i) of weight tensor idx as the raw binary16 bits. */
uint16_t metalfg_weight_elem(const MetalFGWeightFile *w, int idx, uint32_t o, uint32_t ky, uint32_t kx, uint32_t i);
/* Build a file image from 84 tensors (each metalfg_tensor_elems(t) binary16 values). Caller frees. */
uint8_t *metalfg_weights_image(const uint16_t *const tensors[METALFG_TENSOR_COUNT], size_t *size_out);
int metalfg_write_file_atomic(const char *path, const void *data, size_t size, char *err, size_t cap);

/* ---- named scalars of fg.numerics, from the parameter file ---- */
typedef struct {
    int32_t hm, hf, hb, gx, gy;
    float mt, mfa, mfh, th_ui, th_uia, leak, th_mask, b_steer, sat_n2;
    int32_t d_dyn, d_sim;
    float th_cam;
    int32_t sp_min, sp_max;
    float th_zn;
} MetalFGParams;

#define METALFG_PARAM_COUNT 20
#define METALFG_PARAM_FILE_BYTES (16 + METALFG_PARAM_COUNT * 40 + 32)

int metalfg_params_parse(const uint8_t *data, size_t size, MetalFGParams *out, char *err, size_t cap);
int metalfg_params_load(const char *path, MetalFGParams *out, char *err, size_t cap);
/* Serialise parameters (checksum appended). Caller frees. */
uint8_t *metalfg_params_image(const MetalFGParams *p, size_t *size_out);

#endif
