/* Texture helpers for pass-level tests: set every texel from a callback, read texels back as floats. */
#pragma once
#include "nnutil.h"

static inline uint32_t tex_bpp(MTLPixelFormat pf) {
    switch (pf) {
    case MTLPixelFormatRGBA32Float: return 16;
    case MTLPixelFormatRGBA16Float: case MTLPixelFormatRG32Float: return 8;
    case MTLPixelFormatRG16Float: case MTLPixelFormatR32Float: case MTLPixelFormatR32Uint: case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatBGRA8Unorm: return 4;
    case MTLPixelFormatR16Float: return 2;
    default: return 1;
    }
}

static inline void tex_set(id<MTLTexture> t, void (^f)(uint32_t x, uint32_t y, float v[4])) {
    uint32_t W = (uint32_t)t.width, H = (uint32_t)t.height, bpp = tex_bpp(t.pixelFormat);
    MTLPixelFormat pf = t.pixelFormat;
    uint8_t *buf = calloc((size_t)W * H, bpp);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            float v[4] = {0, 0, 0, 0};
            f(x, y, v);
            uint8_t *p = buf + ((size_t)y * W + x) * bpp;
            switch (pf) {
            case MTLPixelFormatRGBA32Float: for (int c = 0; c < 4; c++) ((float *)p)[c] = v[c]; break;
            case MTLPixelFormatRGBA16Float: for (int c = 0; c < 4; c++) ((h16 *)p)[c] = (h16)v[c]; break;
            case MTLPixelFormatRG16Float: for (int c = 0; c < 2; c++) ((h16 *)p)[c] = (h16)v[c]; break;
            case MTLPixelFormatR16Float: ((h16 *)p)[0] = (h16)v[0]; break;
            case MTLPixelFormatRG32Float: for (int c = 0; c < 2; c++) ((float *)p)[c] = v[c]; break;
            case MTLPixelFormatR32Float: ((float *)p)[0] = v[0]; break;
            case MTLPixelFormatR32Uint: ((uint32_t *)p)[0] = (uint32_t)v[0]; break;
            case MTLPixelFormatRGBA8Unorm: for (int c = 0; c < 4; c++) p[c] = (uint8_t)lrintf(fminf(fmaxf(v[c], 0), 1) * 255); break;
            default: p[0] = (uint8_t)lrintf(fminf(fmaxf(v[0], 0), 1) * 255); break;
            }
        }
    gpu_upload(t, buf, bpp);
    free(buf);
}

/* returns W * H * comps floats (caller frees); R32Uint is returned as raw bit patterns in the float slots */
static inline float *tex_get(id<MTLTexture> t, int comps) {
    uint32_t W = (uint32_t)t.width, H = (uint32_t)t.height, bpp = tex_bpp(t.pixelFormat);
    MTLPixelFormat pf = t.pixelFormat;
    uint8_t *buf = malloc((size_t)W * H * bpp);
    gpu_download(t, buf, bpp);
    float *o = calloc((size_t)W * H * comps, sizeof(float));
    for (size_t i = 0; i < (size_t)W * H; i++)
        for (int c = 0; c < comps; c++) {
            const uint8_t *p = buf + i * bpp;
            float v = 0;
            switch (pf) {
            case MTLPixelFormatRGBA32Float: v = ((const float *)p)[c]; break;
            case MTLPixelFormatRG32Float: v = c < 2 ? ((const float *)p)[c] : 0; break;
            case MTLPixelFormatR32Float: v = c < 1 ? ((const float *)p)[0] : 0; break;
            case MTLPixelFormatR32Uint: memcpy(&v, p, 4); break;
            case MTLPixelFormatRGBA16Float: v = (float)((const h16 *)p)[c]; break;
            case MTLPixelFormatRG16Float: v = c < 2 ? (float)((const h16 *)p)[c] : 0; break;
            case MTLPixelFormatR16Float: v = c < 1 ? (float)((const h16 *)p)[0] : 0; break;
            case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatBGRA8Unorm: v = p[c] / 255.0f; break;
            default: v = c < 1 ? p[0] / 255.0f : 0; break;
            }
            o[i * comps + c] = v;
        }
    free(buf);
    return o;
}

/* a rig with a library and no useful weights, for passes that do not touch the networks */
static inline NnRig plain_rig(Gpu g) {
    uint16_t *t[METALFG_TENSOR_COUNT];
    wgen_tensors(WGEN_ZERO, 1, t);
    NnRig r;
    if (nn_open(&r, g, t)) exit(2);
    return r;
}

/* Depth record of fg.interm_motion_pin_r4 (rule A), host side: selector in the top three bits, the lower 29 bits of the depth below. */
static inline uint32_t rec_pack_c(float d, uint32_t code) {
    uint32_t b;
    memcpy(&b, &d, 4);
    return (code << 29) | (d >= 0x1p-62f ? (b & 0x1FFFFFFFu) : 0u);
}
static inline uint32_t rec_code_c(uint32_t rec) { return rec >> 29; }
static inline float rec_depth_c(uint32_t rec) {
    uint32_t low = rec & 0x1FFFFFFFu, b = low == 0 ? 0u : (low | 0x20000000u);
    float d;
    memcpy(&d, &b, 4);
    return d;
}
