/* Constants shared by the engine and the setup-time extractor (specs fg.numerics_r3, fg.extractor_r2). */
#ifndef METALFG_CONST_H
#define METALFG_CONST_H
#include <stddef.h>
#include <stdint.h>

/* Menu-detection sample grid: structural constants of the detector, not read from the library (fg.numerics_r3). */
#define METALFG_GX 64
#define METALFG_GY 32

/* 32-bit FNV-1a (fg.extractor_r2 "prefilter hash"). */
static inline uint32_t metalfg_fnv1a32(const uint8_t *p, size_t n) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}

#endif
