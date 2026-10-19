/* Setup-time extractor (spec fg.extractor): finds the network blobs and the model parameters in the user's own copy of
   the vendor library, converts the stored tiling to the canonical layout and writes the weight and parameter files.
   Nothing here contains a learned value; the locator file supplies fingerprints, shapes and anchors. */
#ifndef FGX_EXTRACT_H
#define FGX_EXTRACT_H
#include <stddef.h>
#include <stdint.h>

/* exit codes */
#define FGX_OK 0
#define FGX_USAGE 1
#define FGX_NOT_PE 2
#define FGX_NOT_FOUND 3
#define FGX_CHECKSUM 4

/* ---- primitives (self-tested) ---- */
void fgx_blake2b(uint8_t *out, size_t outlen, const void *in, size_t inlen);
/* LZ4 block decode; returns the decoded length or -1 on malformed input or when the output would exceed cap */
long fgx_lz4_decode(const uint8_t *in, size_t inlen, uint8_t *out, size_t cap);

/* ---- tiled forms ---- */
/* Net1 blob (kind 0): weights, zero region, bias tiles. w: O*K*K*I elements, b: O elements. 0 on success. */
int fgx_untile_net1(const uint8_t *blob, uint32_t O, uint32_t K, uint32_t I, uint16_t *w, uint16_t *b, char *err, size_t cap);
/* Net2 weight tiles (kind 1). w: O*K*K*I_stored elements; padded input channels (>= I_real) must be zero. */
int fgx_untile_net2(const uint8_t *blob, uint32_t O, uint32_t K, uint32_t I_stored, uint32_t I_real, uint16_t *w, char *err, size_t cap);

/* ---- locator ---- */
typedef struct {
    uint8_t fp[8];
    uint32_t tensor_index, kind, O, K, I_stored, I_real, variant, blob_bytes;
} FgxBlobRec;

typedef struct {
    char name[25];
    uint8_t fp[8];
    uint16_t wlen;
    int16_t disp;
    uint8_t type, deriv;
    uint16_t second;
    uint8_t area, indirect, tail, nmask;
    uint32_t target;
    uint16_t mask[3][2];
    int32_t addend;
    uint32_t prefilter;      /* FNV-1a hash of the first eight window bytes */
    uint8_t prefix_len;
} FgxParamRec;

typedef struct {
    uint32_t nblobs, nparams;
    uint8_t payload_sha[32], params_sha[32];
    uint32_t container_magic, text_kind;
    uint64_t block_mask, unsupported_mask;
    FgxBlobRec *blobs;
    FgxParamRec *params;
} FgxLocator;

int fgx_locator_parse(const uint8_t *d, size_t n, FgxLocator *loc, char *err, size_t cap);
void fgx_locator_free(FgxLocator *loc);

/* ---- run ---- */
typedef struct {
    int blobs_found, blobs_expected, params_found, params_expected;
    int containers, entries_unpacked, entries_skipped, scan_step;
    uint8_t lib_sha[32], payload_sha[32];
} FgxReport;

/* Extracts from the library image. On success *weights and *params are malloc'd images ready to be written.
   Returns an exit code; err receives the message (names tensors and parameters, never values). */
int fgx_extract(const uint8_t *lib, size_t libsize, const FgxLocator *loc, uint8_t **weights, size_t *wsize, uint8_t **params,
                size_t *psize, FgxReport *rep, char *err, size_t cap);
/* Discovery listing (candidate blobs and containers) as text; malloc'd. */
char *fgx_discover(const uint8_t *lib, size_t libsize, const FgxLocator *loc);

#endif
