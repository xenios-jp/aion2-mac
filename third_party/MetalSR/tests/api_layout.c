/* api-layout: asserts every size and offset of sr.engine_api. */
#include <stddef.h>
#include <stdio.h>
#include "metalsr.h"

static int bad;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s\n", #c); bad = 1; } } while (0)

int main(void) {
    CHECK(METALSR_INTERFACE_VERSION == 2);
    CHECK(sizeof(MetalSRCreateInfo) == 56);
    CHECK(offsetof(MetalSRCreateInfo, version) == 0);
    CHECK(offsetof(MetalSRCreateInfo, max_rw) == 4);
    CHECK(offsetof(MetalSRCreateInfo, max_rh) == 8);
    CHECK(offsetof(MetalSRCreateInfo, out_w) == 12);
    CHECK(offsetof(MetalSRCreateInfo, out_h) == 16);
    CHECK(offsetof(MetalSRCreateInfo, flags) == 20);
    CHECK(offsetof(MetalSRCreateInfo, weight_path) == 24);
    CHECK(offsetof(MetalSRCreateInfo, shader_dir) == 32);
    CHECK(offsetof(MetalSRCreateInfo, test_hooks) == 40);
    CHECK(offsetof(MetalSRCreateInfo, profile) == 48);
    CHECK(sizeof(MetalSRFrameInfo) == 128);
    CHECK(offsetof(MetalSRFrameInfo, color) == 0);
    CHECK(offsetof(MetalSRFrameInfo, depth) == 8);
    CHECK(offsetof(MetalSRFrameInfo, motion) == 16);
    CHECK(offsetof(MetalSRFrameInfo, output) == 24);
    CHECK(offsetof(MetalSRFrameInfo, exposure) == 32);
    CHECK(offsetof(MetalSRFrameInfo, reactive) == 40);
    CHECK(offsetof(MetalSRFrameInfo, fence) == 48);
    CHECK(offsetof(MetalSRFrameInfo, rw) == 56);
    CHECK(offsetof(MetalSRFrameInfo, rh) == 60);
    CHECK(offsetof(MetalSRFrameInfo, jitter_x) == 64);
    CHECK(offsetof(MetalSRFrameInfo, jitter_y) == 68);
    CHECK(offsetof(MetalSRFrameInfo, mv_scale_x) == 72);
    CHECK(offsetof(MetalSRFrameInfo, mv_scale_y) == 76);
    CHECK(offsetof(MetalSRFrameInfo, pre_exposure) == 80);
    CHECK(offsetof(MetalSRFrameInfo, reset) == 84);
    CHECK(offsetof(MetalSRFrameInfo, color_origin) == 88);
    CHECK(offsetof(MetalSRFrameInfo, depth_origin) == 96);
    CHECK(offsetof(MetalSRFrameInfo, motion_origin) == 104);
    CHECK(offsetof(MetalSRFrameInfo, reactive_origin) == 112);
    CHECK(offsetof(MetalSRFrameInfo, output_origin) == 120);
    CHECK(sizeof(MetalSRTiming) == 16);
    CHECK(offsetof(MetalSRTiming, microseconds) == 8);
    CHECK(METALSR_FLAG_HDR == 1 && METALSR_FLAG_DEPTH_INVERTED == 2 && METALSR_FLAG_MV_RENDER_RES == 4 &&
          METALSR_FLAG_AUTO_EXPOSURE == 8 && METALSR_FLAG_REACTIVE == 16);
    puts(bad ? "api-layout: FAIL" : "api-layout: ok");
    return bad;
}
