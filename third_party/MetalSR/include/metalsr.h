/* Public C interface of the SR engine (interface version 2). BSD-3-Clause. */
#ifndef METALSR_H
#define METALSR_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define METALSR_INTERFACE_VERSION 2u

#define METALSR_FLAG_HDR            (1u << 0)
#define METALSR_FLAG_DEPTH_INVERTED (1u << 1)
#define METALSR_FLAG_MV_RENDER_RES  (1u << 2)
#define METALSR_FLAG_AUTO_EXPOSURE  (1u << 3)
#define METALSR_FLAG_REACTIVE       (1u << 4)

typedef struct MetalSREngine MetalSREngine; /* opaque */

typedef struct MetalSRCreateInfo {
    uint32_t    version;      /*  0 */
    uint32_t    max_rw;       /*  4 */
    uint32_t    max_rh;       /*  8 */
    uint32_t    out_w;        /* 12 */
    uint32_t    out_h;        /* 16 */
    uint32_t    flags;        /* 20 */
    const char *weight_path;  /* 24; null = $METALSR_WEIGHTS */
    const char *shader_dir;   /* 32; ignored */
    const char *test_hooks;   /* 40; must be null or empty */
    uint32_t    profile;      /* 48; 1 = per-pass GPU times */
    uint32_t    reserved_;    /* 52 */
} MetalSRCreateInfo;              /* 56 */

typedef struct MetalSRFrameInfo {
    void    *color;           /*   0 */
    void    *depth;           /*   8 */
    void    *motion;          /*  16 */
    void    *output;          /*  24 */
    void    *exposure;        /*  32 */
    void    *reactive;        /*  40; ignored */
    void    *fence;           /*  48 */
    uint32_t rw, rh;          /*  56, 60 */
    float    jitter_x, jitter_y;     /* 64, 68 */
    float    mv_scale_x, mv_scale_y; /* 72, 76 */
    float    pre_exposure;    /*  80 */
    uint32_t reset;           /*  84 */
    uint32_t color_origin[2];    /*  88 */
    uint32_t depth_origin[2];    /*  96 */
    uint32_t motion_origin[2];   /* 104 */
    uint32_t reactive_origin[2]; /* 112; ignored */
    uint32_t output_origin[2];   /* 120 */
} MetalSRFrameInfo;               /* 128 */

typedef struct MetalSRTiming {
    const char *label;        /* 0 */
    double      microseconds; /* 8 */
} MetalSRTiming;                  /* 16 */

/* device: id<MTLDevice>; cmdbuf: id<MTL4CommandBuffer>; both passed as plain pointers. */
MetalSREngine *metalsr_create(void *device, const MetalSRCreateInfo *info, char *err, size_t err_cap);
int        metalsr_encode(MetalSREngine *e, void *cmdbuf, const MetalSRFrameInfo *frame);
uint32_t   metalsr_timings(MetalSREngine *e, MetalSRTiming *out, uint32_t cap);
void       metalsr_destroy(MetalSREngine *e);
/* Size query (sr.test_scenes_r6; CPU only: no device, no file, no allocation; thread-safe). Applies the size and version checks of create and
   encode and derives the sizes they use. Returns 0 accepted (record filled), 1 create would fail, 2 the call size is invalid (message written,
   record unchanged). r_wide and n_luma_min are the parameter-file values. */
typedef struct MetalSRSizes {
    uint32_t wp, hp;          /*  0,  4: padded output size, multiples of 128 */
    uint32_t lw[6], lh[6];    /*  8, 32: level sizes Wp / 2^(k+2), Hp / 2^(k+2) */
    uint32_t path;            /* 56: 0 = N, 1 = W */
    float    sx, sy;          /* 60, 64: render / output scale in double, rounded once to fp32 */
    uint32_t gw, gh;          /* 68, 72: exposure grid of the call, after the n_luma_min floor */
    uint32_t reserved_;       /* 76: 0 */
} MetalSRSizes;                   /* 80 */
int metalsr_sizes(const MetalSRCreateInfo *info, uint32_t rw, uint32_t rh, double r_wide, uint32_t n_luma_min, MetalSRSizes *out, char *err, size_t err_cap);

/* extra (not part of the binary-compatible set): text of the last failed encode */
const char *metalsr_last_error(MetalSREngine *e);
/* extra (tests): exposure state (E_now, E_prev) after the newest completed call */
void metalsr_debug_exposure(MetalSREngine *e, float *now, float *prev);
/* extra (tests): 1 when the engine runs the wide-ratio path (W), else 0 */
int metalsr_debug_path(MetalSREngine *e);
/* extra (tests): compute dispatches and barriers (encoder and queue stage) encoded by the newest call */
void metalsr_debug_counts(MetalSREngine *e, int *dispatches, int *barriers);

#ifdef __cplusplus
}
#endif
#endif
