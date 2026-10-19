# Third-party engines

| Directory | What it is | Upstream | License |
| --- | --- | --- | --- |
| `MetalSR/` | Metal 4 neural super-resolution engine (`libmetalsr`) and its setup-time weight extractor (`extractsr`) | xenios-jp clean-room lane `sr`, commit `8373653903199ed919354b53531c94eadbbe8ec5` | BSD-3-Clause: `MetalSR/LICENSE`, `MetalSR/NOTICE` |
| `MetalFG/` | Metal neural frame-interpolation engine (`libmetalfg`) and its setup-time weight extractor (`extractfg`) | xenios-jp clean-room lane `fg`, commit `c73efa1a5ad91aa12b3dc20527a517015c2e053e` | BSD-3-Clause: `MetalFG/LICENSE`, `MetalFG/NOTICE` |

Both engines were written clean-room. The implementers worked only from reviewed behavioural specifications, inside a sandbox with no access to vendor binaries, vendor SDKs, translated vendor code or HelixSR. Every upstream commit names the specifications it implements. Contamination audits found no copied code. They compared the engines with the vendor kernels by text, syntax-tree and compiled-IR similarity, and used independent control reimplementations as a baseline.

No weights, vendor binaries, vendor headers or data derived from vendor code are included. Each extractor reads the learned weights from the user's own copy of the game's DLSS library at setup time. It needs a small locator file for the library version, which holds fingerprints and positions but no weight values. Locator files are not part of this directory.

## Changes from upstream

The trees are `git archive` exports of the commits above, with only these edits:

- The README of each engine was edited: its title, and a reference to upstream working documents removed.
- `MetalFG/docs/STATUS.md`, the upstream development log, is left out.
- `MetalFG/NOTICE` is added, with the same extractor credit as MetalSR's.
- Lab files are left out: benchmark and A/B scripts (MetalSR `tools/*.sh`; MetalFG `tools/perf/`, `tools/tol/`), benchmark programs (MetalSR `tests/kbench.m`, `perf_bench.m`, `dcost.m`, `pipeinfo.m`, `bench_gpu.m`, `mmpeak_gpu.m`) and MetalFG's experiment programs (`tests/x_*.m`). Nothing in the builds or the tests uses them.

The GPU test targets (`./build.sh gputest`, and the GPU part of `make test`) call the upstream workspace's `gpu-run` GPU-lock wrapper, so they don't run from this repository. The library builds and the CPU tests (`./build.sh test`) do.

## Build

    (cd MetalSR && CC="xcrun clang -arch arm64 -arch x86_64" ./build.sh)
    (cd MetalFG && make all build/extractfg)

Both produce universal (arm64 and x86_64) libraries and extractors in each engine's `build/` folder.
