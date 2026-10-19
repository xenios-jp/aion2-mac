# MetalSR

A Metal 4 neural super-resolution engine, written clean-room from approved specifications. BSD-3-Clause.

- `include/metalsr.h`: the public C interface (four functions, interface version 2, plus two test helpers).
- `src/`: engine (`engine.m` orchestration, `passes.m` pass encoders, `gpu.m` Metal 4 plumbing), shaders (`*.metal`),
  weight/parameter file loaders (`model.c`), per-call scalar derivation (`frame.c`).
- `tools/extract.c`: setup-time extractor (`extractsr <library> <locator> --out <dir>`), universal binary.
- `tests/`: CPU references (`ref*.h`, `numerics.h`), synthetic scenes (`scenes.h`) and the tests.
- Build: `./build.sh` (uses `xcrun clang` / `xcrun metal`; produces `build/libmetalsr.dylib`, `.a`, the metallib and
  the extractor). `./build.sh test` also runs the CPU tests; `./build.sh gputest` runs the GPU tests through
  a `gpu-run` GPU-lock wrapper (see ../README.md).
- Weights: the path field of the create record, else `METALSR_WEIGHTS`; the parameter file sits next to it
  (`<weights>.params`). A pipeline cache is kept in the user's caches directory (or next to the weight file).

