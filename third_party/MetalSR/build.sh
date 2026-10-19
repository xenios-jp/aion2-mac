#!/bin/sh
# Build the engine library and the tests into ./build.
#   ./build.sh            library + metallib
#   ./build.sh test       also build and run the CPU tests
#   ./build.sh gputest    also build the GPU tests and run them through clean/bin/gpu-run
set -eu
cd "$(dirname "$0")"
mkdir -p build
rm -f build/*.air
CC="${CC:-xcrun clang}"
CFLAGS="-O2 -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -ffp-contract=off -Iinclude -Isrc -fobjc-arc"
MODCACHE="$PWD/build/mc"

# Shaders: fast math by default; helpers that need IEEE behaviour use the SR_SAFE function pragma (src/sr_math.h).
for f in src/*.metal; do
    n=$(basename "$f" .metal)
    xcrun metal -c -std=metal4.0 -O2 -fmetal-math-mode=fast ${SR_METAL_DEFS:-} -Wno-unused-const-variable \
        -fmodules-cache-path="$MODCACHE" -o "build/$n.air" "$f"
done
xcrun metallib -o build/metalsr.metallib build/*.air
(cd build && xxd -i metalsr.metallib > metallib_data.c)

SRC="src/engine.m src/gpu.m src/passes.m src/model.c src/frame.c build/metallib_data.c"
$CC $CFLAGS -dynamiclib -install_name @rpath/libmetalsr.dylib -o build/libmetalsr.dylib $SRC \
    -Wl,-exported_symbol,'_metalsr_*' -framework Foundation -framework Metal
for s in $SRC; do $CC $CFLAGS -c -o "build/$(basename "$s").o" "$s"; done
rm -f build/libmetalsr.a
ar rcs build/libmetalsr.a build/engine.m.o build/gpu.m.o build/passes.m.o build/model.c.o build/frame.c.o build/metallib_data.c.o

nm -gU build/libmetalsr.dylib | grep -q " T _metalsr_sizes$" || { echo "metalsr_sizes is not exported"; exit 1; }

# setup-time extractor, universal binary (sr.extractor)
$CC -O2 -Wall -Wno-unused-function -arch arm64 -arch x86_64 -o build/extractsr tools/extract.c src/model.c -lm

case "${1:-}" in
test|gputest)
    $CC -O2 -Wall -Iinclude -o build/api_layout tests/api_layout.c
    $CC -O1 -Wall -o build/numerics_test tests/numerics_test.c -lm
    $CC -O1 -Wall -o build/model_test tests/model_test.c src/model.c -lm
    ./build/api_layout
    ./build/numerics_test
    ./build/model_test
    $CC -O2 -Wall -Wno-unused-function -o build/extract_test tests/extract_test.c src/model.c -lm
    ./build/extract_test
    $CC -O2 -Wall -Iinclude -Isrc -o build/sizes_test tests/sizes_test.c src/frame.c
    ./build/sizes_test
    ;;
esac
if [ "${1:-}" = gputest ]; then
    # the filter built with the strict-math helpers in fast math: the head-coverage-ratio3 guard test must fail on it
    mkdir -p build/fast
    xcrun metal -c -std=metal4.0 -O2 -fmetal-math-mode=fast -DSR_NO_SAFE_PRAGMA -Wno-unused-const-variable \
        -fmodules-cache-path="$MODCACHE" -o build/fast/filter.air src/filter.metal
    xcrun metallib -o build/filter_fast.metallib build/fast/filter.air
    # the same with the coordinate product rounded separately (an unfused multiply-add): the sensitivity control of that test
    xcrun metal -c -std=metal4.0 -O2 -fmetal-math-mode=fast -DSR_GUARD_UNFUSED -Wno-unused-const-variable \
        -fmodules-cache-path="$MODCACHE" -o build/fast/filter_unfused.air src/filter.metal
    xcrun metallib -o build/filter_unfused.metallib build/fast/filter_unfused.air
    for t in tests/*_gpu.m; do
        n=$(basename "$t" .m)
        [ "$n" = halfw_gpu ] && continue      # needs a library built with SR_TESTOUT: run below on its own libraries
        $CC $CFLAGS -o "build/$n" "$t" $SRC -framework Foundation -framework Metal
        ../../bin/gpu-run --max-s 300 -- "build/$n"
    done
    # controls of tests/finite_gpu.m: shaders built with injected non-finite stores (filter and conversion pass). With the output guard the
    # test must pass; with the guard removed (SR_NO_OUTPUT_GUARD) it must fail. A control that does not fail stops the build.
    for v in inject inject_noguard; do
        d="build/ctl_$v"; mkdir -p "$d"; rm -f "$d"/*.air
        DEFS="-DSR_INJECT_NONFINITE"
        [ "$v" = inject_noguard ] && DEFS="-DSR_INJECT_NONFINITE -DSR_NO_OUTPUT_GUARD"
        for f in src/*.metal; do
            n=$(basename "$f" .metal)
            xcrun metal -c -std=metal4.0 -O2 -fmetal-math-mode=fast $DEFS -Wno-unused-const-variable \
                -fmodules-cache-path="$MODCACHE" -o "$d/$n.air" "$f"
        done
        xcrun metallib -o "$d/metalsr.metallib" "$d"/*.air
        (cd "$d" && xxd -i metalsr.metallib > metallib_data.c)
        $CC $CFLAGS -o "$d/finite_gpu" tests/finite_gpu.m src/engine.m src/gpu.m src/passes.m src/model.c src/frame.c "$d/metallib_data.c" \
            -framework Foundation -framework Metal
    done
    # tests/halfw_gpu.m (sr.metal_techniques_r3 section 7, items 2 and 3) on libraries built with SR_TESTOUT: SR_HW=1 (V1) must pass; the controls SR_HW=2
    # (no scaling by the largest argument) and SR_HW=3 (weights with 3 mantissa bits) must fail. A control that passes stops the build.
    for hw in 1 2 3; do
        d="build/hw_$hw"; mkdir -p "$d"; rm -f "$d"/*.air
        for f in src/*.metal; do
            n=$(basename "$f" .metal)
            xcrun metal -c -std=metal4.0 -O2 -fmetal-math-mode=fast -DSR_TESTOUT -DSR_HW=$hw -Wno-unused-const-variable \
                -fmodules-cache-path="$MODCACHE" -o "$d/$n.air" "$f"
        done
        xcrun metallib -o "$d/metalsr.metallib" "$d"/*.air
        (cd "$d" && xxd -i metalsr.metallib > metallib_data.c)
        $CC $CFLAGS -o "$d/halfw_gpu" tests/halfw_gpu.m src/engine.m src/gpu.m src/passes.m src/model.c src/frame.c "$d/metallib_data.c" -framework Foundation -framework Metal
    done
    echo "half-precision weights V1 (filter-bound, filter-sharp; must pass)"
    ../../bin/gpu-run --max-s 300 -- build/hw_1/halfw_gpu
    for hw in 2 3; do
        echo "control: SR_HW=$hw (must fail)"
        if ../../bin/gpu-run --max-s 300 -- build/hw_$hw/halfw_gpu; then echo "CONTROL FAILED: halfw_gpu passed with SR_HW=$hw"; exit 1; fi
        echo "control ok: halfw_gpu fails with SR_HW=$hw"
    done
    echo "control: finite_gpu with injected non-finite stores and the output guard (must pass)"
    ../../bin/gpu-run --max-s 300 -- build/ctl_inject/finite_gpu
    echo "control: finite_gpu with injected non-finite stores and no output guard (must fail)"
    if ../../bin/gpu-run --max-s 300 -- build/ctl_inject_noguard/finite_gpu; then
        echo "CONTROL FAILED: finite_gpu passed without the output guard"; exit 1
    fi
    echo "control ok: finite_gpu fails without the output guard"
fi
