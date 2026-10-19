/* Internal camera helpers (spec fg.engine_api, fg.numerics). */
#ifndef METALFG_CAMERA_H
#define METALFG_CAMERA_H
#include <stdint.h>

/* 4x4 inverse in double precision, row-major; returns 0 when singular. */
int metalfg_invert4_d(const double m[16], double out[16]);
/* Standard perspective matrix, host form (column vectors, column-major). */
void metalfg_fallback_perspective(float fov_deg, float aspect, float nearp, float farp, int reversed, float *m);

#endif
