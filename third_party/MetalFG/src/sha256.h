/* SHA-256 (FIPS 180-4), used for the payload and parameter-file checksums (spec fg.weights_format, fg.extractor). */
#ifndef METALFG_SHA256_H
#define METALFG_SHA256_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t len;
    uint8_t buf[64];
    size_t fill;
} MetalFGSha256;

void metalfg_sha256_init(MetalFGSha256 *c);
void metalfg_sha256_update(MetalFGSha256 *c, const void *data, size_t n);
void metalfg_sha256_final(MetalFGSha256 *c, uint8_t out[32]);
void metalfg_sha256(const void *data, size_t n, uint8_t out[32]);

#endif
