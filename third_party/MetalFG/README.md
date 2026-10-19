# MetalFG

A Metal (Apple GPU) neural frame-interpolation engine, written from the approved specifications of the clean-room
process (`clean/specs/fg.*`). BSD-3-Clause.

The engine takes the current real frame (history is kept internally), depth and motion vectors and produces one
synthetic frame between the previous and the current real frame. Two small convolutional networks sit between
compute passes that prepare motion, warp both frames and synthesise the output.

## Build

    make          # build/libmetalfg.dylib (arm64 + x86_64) and build/metalfg.metallib
    make test     # CPU tests, then GPU tests through ../../bin/gpu-run
    make clean

Shaders are compiled offline: `shaders/precise/*.metal` with `-fno-fast-math` (threshold, key and coordinate passes),
`shaders/fast/*.metal` with `-ffast-math` (none at present: the network kernels moved to `shaders/precise` for the bit-exact arithmetic of fg.arithmetic_pin_r4).

## Interface

`include/metalfg.h` (plain C): `metalfg_create`, `metalfg_encode`, `metalfg_encode_classic`, `metalfg_last_error`, `metalfg_timings`,
`metalfg_destroy`, plus the helpers `metalfg_depth_config` and `metalfg_camera_matrices`.

## Files

| path | content |
|-|-|
| `include/metalfg.h` | public header with the binary records and layout assertions |
| `src/` | library sources (Objective-C / C) |
| `shaders/` | Metal sources |
| `tests/` | unit tests with CPU references (`t_*` CPU only, `g_*` need a GPU) |
| `tools/` | weight/parameter file generator for tests, the extractor |

## Weights

The learned weights are not part of this repository. The extractor (`tools/`) reads them from the user's own copy of
the vendor library at setup time and writes the weight file and its `.params` file. For testing, `tools/genweights`
writes random files in the same format.
