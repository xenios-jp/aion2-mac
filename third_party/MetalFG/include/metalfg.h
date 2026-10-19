/* metalfg: Metal frame-interpolation engine. Public C interface (spec fg.engine_api). BSD-3-Clause. */
#ifndef METALFG_H
#define METALFG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define METALFG_VISIBLE __attribute__((visibility("default")))

#define METALFG_INTERFACE_VERSION 2u
/* frame record length fixed by the interface version given at create (fg.engine_api_r2) */
#define METALFG_FRAME_RECORD_V1 336u
#define METALFG_FRAME_RECORD_V2 352u

/* create flag bits */
#define METALFG_CREATE_UI_RECOMPOSE   (1u << 0)
#define METALFG_CREATE_MENU_DETECT    (1u << 1)
#define METALFG_CREATE_DISTORTION     (1u << 2)
#define METALFG_CREATE_HDR            (1u << 3)
#define METALFG_CREATE_DIRECT_OUTPUT  (1u << 4)
#define METALFG_CREATE_FORCE_BLEND_1  (1u << 5)
#define METALFG_CREATE_FORMER_MOTION_GRID (1u << 6)   /* fg.motion_grid_r1: path O, the motion grid factor of fg.resources at every height */

/* fg.motion_grid_r1: height threshold of path N (motion grid factor 4 from this display height on) */
#define METALFG_MOTION_GRID_HQ 1440u

/* frame flag bits */
#define METALFG_FRAME_RESET           (1u << 0)
#define METALFG_FRAME_DEPTH_REVERSED  (1u << 1)
#define METALFG_FRAME_UI_IN_COLOUR    (1u << 2)
#define METALFG_FRAME_MOTION_CAMERA   (1u << 3)

#define METALFG_MIN_SIDE 4u
#define METALFG_MAX_WIDTH 8192u
#define METALFG_MAX_HEIGHT 8192u

typedef struct MetalFGCreateInfo {
    uint32_t version;          /*  0 */
    uint32_t display_w;        /*  4 */
    uint32_t display_h;        /*  8 */
    uint32_t flags;            /* 12 */
    uint32_t max_generated;    /* 16 */
    uint32_t profile;          /* 20 */
    const char *weights_file;  /* 24 */
    const char *metallib_dir;  /* 32 */
    float depth_scale;         /* 40 */
    float depth_partition;     /* 44 */
    float depth_min_sep;       /* 48 */
    uint32_t reserved;         /* 52 */
} MetalFGCreateInfo;

typedef struct MetalFGFrameInfo {
    void *colour;              /*   0 */
    void *prev_colour;         /*   8 */
    void *hudless;             /*  16 */
    void *depth;               /*  24 */
    void *motion;              /*  32 */
    void *ui_colour;           /*  40 */
    void *ui_alpha;            /*  48 */
    void *distortion;          /*  56 */
    void *out_interp;          /*  64 */
    void *out_real;            /*  72 */
    void *skip_flag;           /*  80 */
    void *fence;               /*  88 */
    uint32_t content_w;        /*  96 */
    uint32_t content_h;        /* 100 */
    uint32_t colour_off[2];    /* 104 */
    uint32_t depth_off[2];     /* 112 */
    uint32_t motion_off[2];    /* 120 */
    uint32_t distortion_off[2];/* 128 */
    uint32_t distortion_size[2];/* 136 */
    uint32_t out_off[2];       /* 144 */
    float motion_scale_x;      /* 152 */
    float motion_scale_y;      /* 156 */
    float jitter_x;            /* 160 */
    float jitter_y;            /* 164 */
    float delta_time;          /* 168 */
    float near_plane;          /* 172 */
    float far_plane;           /* 176 */
    float fov_deg;             /* 180 */
    float aspect;              /* 184 */
    uint32_t flags;            /* 188 */
    uint32_t gen_index;        /* 192 */
    uint32_t gen_count;        /* 196 */
    uint64_t frame_index;      /* 200 */
    float world_to_view[16];   /* 208 */
    float view_to_clip[16];    /* 272 */
    uint32_t depth_size[2];    /* 336: interface version 2; 0 in a component = content size in that component */
    uint32_t motion_size[2];   /* 344: interface version 2; likewise */
} MetalFGFrameInfo;

typedef struct MetalFGTiming {
    const char *label;         /* 0 */
    double microseconds;       /* 8 */
} MetalFGTiming;

_Static_assert(sizeof(MetalFGCreateInfo) == 56, "create record size");
_Static_assert(offsetof(MetalFGCreateInfo, version) == 0, "version");
_Static_assert(offsetof(MetalFGCreateInfo, display_w) == 4, "display_w");
_Static_assert(offsetof(MetalFGCreateInfo, display_h) == 8, "display_h");
_Static_assert(offsetof(MetalFGCreateInfo, flags) == 12, "flags");
_Static_assert(offsetof(MetalFGCreateInfo, max_generated) == 16, "max_generated");
_Static_assert(offsetof(MetalFGCreateInfo, profile) == 20, "profile");
_Static_assert(offsetof(MetalFGCreateInfo, weights_file) == 24, "weights_file");
_Static_assert(offsetof(MetalFGCreateInfo, metallib_dir) == 32, "metallib_dir");
_Static_assert(offsetof(MetalFGCreateInfo, depth_scale) == 40, "depth_scale");
_Static_assert(offsetof(MetalFGCreateInfo, depth_partition) == 44, "depth_partition");
_Static_assert(offsetof(MetalFGCreateInfo, depth_min_sep) == 48, "depth_min_sep");
_Static_assert(offsetof(MetalFGCreateInfo, reserved) == 52, "reserved");

_Static_assert(sizeof(MetalFGFrameInfo) == METALFG_FRAME_RECORD_V2, "frame record size");
_Static_assert(_Alignof(MetalFGFrameInfo) == 8, "frame record alignment");
_Static_assert(offsetof(MetalFGFrameInfo, colour) == 0, "colour");
_Static_assert(offsetof(MetalFGFrameInfo, prev_colour) == 8, "prev_colour");
_Static_assert(offsetof(MetalFGFrameInfo, hudless) == 16, "hudless");
_Static_assert(offsetof(MetalFGFrameInfo, depth) == 24, "depth");
_Static_assert(offsetof(MetalFGFrameInfo, motion) == 32, "motion");
_Static_assert(offsetof(MetalFGFrameInfo, ui_colour) == 40, "ui_colour");
_Static_assert(offsetof(MetalFGFrameInfo, ui_alpha) == 48, "ui_alpha");
_Static_assert(offsetof(MetalFGFrameInfo, distortion) == 56, "distortion");
_Static_assert(offsetof(MetalFGFrameInfo, out_interp) == 64, "out_interp");
_Static_assert(offsetof(MetalFGFrameInfo, out_real) == 72, "out_real");
_Static_assert(offsetof(MetalFGFrameInfo, skip_flag) == 80, "skip_flag");
_Static_assert(offsetof(MetalFGFrameInfo, fence) == 88, "fence");
_Static_assert(offsetof(MetalFGFrameInfo, content_w) == 96, "content_w");
_Static_assert(offsetof(MetalFGFrameInfo, content_h) == 100, "content_h");
_Static_assert(offsetof(MetalFGFrameInfo, colour_off) == 104, "colour_off");
_Static_assert(offsetof(MetalFGFrameInfo, depth_off) == 112, "depth_off");
_Static_assert(offsetof(MetalFGFrameInfo, motion_off) == 120, "motion_off");
_Static_assert(offsetof(MetalFGFrameInfo, distortion_off) == 128, "distortion_off");
_Static_assert(offsetof(MetalFGFrameInfo, distortion_size) == 136, "distortion_size");
_Static_assert(offsetof(MetalFGFrameInfo, out_off) == 144, "out_off");
_Static_assert(offsetof(MetalFGFrameInfo, motion_scale_x) == 152, "motion_scale_x");
_Static_assert(offsetof(MetalFGFrameInfo, motion_scale_y) == 156, "motion_scale_y");
_Static_assert(offsetof(MetalFGFrameInfo, jitter_x) == 160, "jitter_x");
_Static_assert(offsetof(MetalFGFrameInfo, jitter_y) == 164, "jitter_y");
_Static_assert(offsetof(MetalFGFrameInfo, delta_time) == 168, "delta_time");
_Static_assert(offsetof(MetalFGFrameInfo, near_plane) == 172, "near_plane");
_Static_assert(offsetof(MetalFGFrameInfo, far_plane) == 176, "far_plane");
_Static_assert(offsetof(MetalFGFrameInfo, fov_deg) == 180, "fov_deg");
_Static_assert(offsetof(MetalFGFrameInfo, aspect) == 184, "aspect");
_Static_assert(offsetof(MetalFGFrameInfo, flags) == 188, "flags");
_Static_assert(offsetof(MetalFGFrameInfo, gen_index) == 192, "gen_index");
_Static_assert(offsetof(MetalFGFrameInfo, gen_count) == 196, "gen_count");
_Static_assert(offsetof(MetalFGFrameInfo, frame_index) == 200, "frame_index");
_Static_assert(offsetof(MetalFGFrameInfo, world_to_view) == 208, "world_to_view");
_Static_assert(offsetof(MetalFGFrameInfo, view_to_clip) == 272, "view_to_clip");
_Static_assert(offsetof(MetalFGFrameInfo, depth_size) == METALFG_FRAME_RECORD_V1, "depth_size");
_Static_assert(offsetof(MetalFGFrameInfo, motion_size) == 344, "motion_size");

_Static_assert(sizeof(MetalFGTiming) == 16, "timing record size");
_Static_assert(offsetof(MetalFGTiming, label) == 0, "timing label");
_Static_assert(offsetof(MetalFGTiming, microseconds) == 8, "timing microseconds");

typedef struct MetalFGEngine *MetalFGEngineRef;

/* Build an engine. On failure returns NULL and writes a NUL-terminated message to err (capacity err_cap). */
METALFG_VISIBLE MetalFGEngineRef metalfg_create(void *mtl_device, const MetalFGCreateInfo *info, char *err, size_t err_cap);
/* Encode one generated frame into a Metal 4 command buffer / a classic command buffer. Zero on success. */
METALFG_VISIBLE int metalfg_encode(MetalFGEngineRef engine, void *mtl4_command_buffer, const MetalFGFrameInfo *frame);
METALFG_VISIBLE int metalfg_encode_classic(MetalFGEngineRef engine, void *mtl_command_buffer, const MetalFGFrameInfo *frame);
/* Message of the most recent failure (engine may be NULL: last failed create on this thread). */
METALFG_VISIBLE const char *metalfg_last_error(MetalFGEngineRef engine);
/* Per-pass GPU times of the newest completed call; returns entries written. */
METALFG_VISIBLE uint32_t metalfg_timings(MetalFGEngineRef engine, MetalFGTiming *out, uint32_t capacity);
METALFG_VISIBLE void metalfg_destroy(MetalFGEngineRef engine);
/* Depth-proxy constants from the host's planes: out3[0] scale, out3[1] partition, out3[2] min separation. */
METALFG_VISIBLE int metalfg_depth_config(float near_plane, float far_plane, uint32_t depth_reversed, float *out3);
/* Row-vector clip matrices: out32[0..15] current-to-previous, out32[16..31] previous-to-current (row-major). */
METALFG_VISIBLE int metalfg_camera_matrices(const float *cur_world_to_view, const float *cur_view_to_clip,
                                    const float *prev_world_to_view, const float *prev_view_to_clip,
                                    float fov_deg, float aspect, float near_plane, float far_plane,
                                    uint32_t depth_reversed, uint32_t reset, float *out32);

#ifdef __cplusplus
}
#endif
#endif
