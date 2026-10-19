# metalfg build (spec fg.engine_api, "build and shader route").
#   make          library + shader library
#   make test     CPU tests, then GPU tests through the machine-wide GPU serialiser
#   make clean

CC       ?= clang
ARCHS     = -arch arm64 -arch x86_64
BUILD     = build
# Control builds (spec fg.test_scenes_r3, "control build contract"): make CTL=BLEND defines METALFG_CTL=1 and METALFG_CTL_BLEND=1 for the host
# compiler and every shader compile. CTLONLY=1 defines METALFG_CTL alone (changes no behaviour). The release build defines neither.
CTLDEFS  := $(if $(CTL),-DMETALFG_CTL=1 -DMETALFG_CTL_$(CTL)=1,$(if $(CTLONLY),-DMETALFG_CTL=1,))
GPU_RUN  ?= ../../bin/gpu-run --max-s 120 --
HOSTARCH := $(shell uname -m)

SRC_M     = $(wildcard src/*.m)
PRECISE   = $(wildcard shaders/precise/*.metal)
FAST      = $(wildcard shaders/fast/*.metal)
AIR       = $(patsubst shaders/precise/%.metal,$(BUILD)/air/p_%.air,$(PRECISE)) \
            $(patsubst shaders/fast/%.metal,$(BUILD)/air/f_%.air,$(FAST))
SHDEPS    = $(wildcard shaders/*.h shaders/*/*.h include/*.h)

all: $(BUILD)/libmetalfg.dylib $(BUILD)/metalfg.metallib

$(BUILD)/libmetalfg.dylib: $(SRC_M) $(wildcard src/*.h) include/metalfg.h
	@mkdir -p $(BUILD)
	$(CC) -O2 -std=c11 -fobjc-arc -fvisibility=hidden -dynamiclib $(ARCHS) \
	  -install_name @rpath/libmetalfg.dylib $(CTLDEFS) -Iinclude src/*.m \
	  -framework Metal -framework Foundation -o $@

$(BUILD)/air/p_%.air: shaders/precise/%.metal $(SHDEPS)
	@mkdir -p $(BUILD)/air
	xcrun metal -fmodules-cache-path=$(BUILD)/modcache -std=metal4.0 -O3 -fno-fast-math $(CTLDEFS) -Ishaders -Iinclude -c $< -o $@

$(BUILD)/air/f_%.air: shaders/fast/%.metal $(SHDEPS)
	@mkdir -p $(BUILD)/air
	xcrun metal -fmodules-cache-path=$(BUILD)/modcache -std=metal4.0 -O3 -ffast-math $(CTLDEFS) -Ishaders -Iinclude -c $< -o $@

$(BUILD)/metalfg.metallib: $(AIR)
	@mkdir -p $(BUILD)
	xcrun metallib $(AIR) -o $@

# ---- tests --------------------------------------------------------------------------------------
# tests/t_*.c|m : CPU-only, run directly.  tests/g_*.m : need a GPU, run through gpu-run.
# Test programs are built for the host architecture only and compile the engine sources they exercise.
TEST_CPU  = $(patsubst tests/%.c,$(BUILD)/tests/%,$(wildcard tests/t_*.c)) \
            $(patsubst tests/%.m,$(BUILD)/tests/%,$(wildcard tests/t_*.m))
TEST_GPU  = $(patsubst tests/%.m,$(BUILD)/tests/%,$(wildcard tests/g_*.m))
TEST_DYLIB = $(patsubst tests/%.m,$(BUILD)/dtests/%,$(wildcard tests/d_*.m))
TESTFLAGS = $(CTLDEFS) -O1 -g -std=gnu11 -fobjc-arc -Iinclude -Isrc -Itests -Itools -Itools/extract -framework Foundation -framework Metal
ENGINE_SRC = $(SRC_M)
EXTRACT_SRC = tools/extract/extract.c

$(BUILD)/tests/%: tests/%.c $(ENGINE_SRC) $(wildcard src/*.h tests/*.h tests/*.c tools/*.h tools/wgen.c)
	@mkdir -p $(BUILD)/dtests $(BUILD)/tests
	$(CC) $(TESTFLAGS) -DMETALFG_TEST $< $(ENGINE_SRC) tools/wgen.c $(EXTRACT_SRC) -o $@
$(BUILD)/tests/%: tests/%.m $(ENGINE_SRC) $(wildcard src/*.h tests/*.h tests/*.c tools/*.h tools/wgen.c)
	@mkdir -p $(BUILD)/dtests $(BUILD)/tests
	$(CC) $(TESTFLAGS) -DMETALFG_TEST $< $(ENGINE_SRC) tools/wgen.c $(EXTRACT_SRC) -o $@

test: all $(BUILD)/extractfg $(BUILD)/genweights $(TEST_CPU) $(TEST_GPU) $(TEST_DYLIB)
	@set -e; for t in $(TEST_CPU); do echo "== $$t"; $$t; done
	@set -e; for t in $(TEST_GPU); do echo "== $$t (gpu)"; METALFG_METALLIB_DIR=$(BUILD) $(GPU_RUN) $$t; done
	@set -e; for t in $(TEST_DYLIB); do echo "== $$t (gpu, dylib)"; $(GPU_RUN) $$t; done
	@$(MAKE) --no-print-directory ctl-runs
	@$(MAKE) --no-print-directory ctl-check
	@echo "ALL TESTS PASSED"

clean:
	rm -rf $(BUILD)

.PHONY: all test clean ctl-check ctl-runs

# test cases whose control is a build (fg.blend_pack_pin_r3 bp2-inject: the stored candidate images truncate; fg.interm_motion_pin_r4 im2-key-once: the payload pass
# recomputes the key with the correctly rounded division). Each is built in its own directory and its test program runs the case with the control in force.
ctl-runs:
	@echo "== g_arith with each channel-block width forced (METALFG_BLK_TC=4, 8)"; for w in 4 8; do METALFG_NN_PATH=blk METALFG_BLK_TC=$$w METALFG_METALLIB_DIR=$(BUILD) $(GPU_RUN) $(BUILD)/tests/g_arith || exit 1; done
	@echo "== pinned kernels (fg.arithmetic_pin_r4, the anchor N0): network and layer tests on the register-blocked and the generic kernel (METALFG_NN_PATH=blk, scalar)"; for p in blk scalar; do for t in g_arith g_net1 g_net2 g_tol; do METALFG_NN_PATH=$$p METALFG_METALLIB_DIR=$(BUILD) $(GPU_RUN) $(BUILD)/tests/$$t || exit 1; done; done
	@$(MAKE) -s CTL=BP_TRUNC BUILD=build/ctl_bp build/ctl_bp/metalfg.metallib build/ctl_bp/tests/g_blend
	@echo "== build/ctl_bp/tests/g_blend (gpu, METALFG_CTL_BP_TRUNC)"; METALFG_METALLIB_DIR=build/ctl_bp $(GPU_RUN) build/ctl_bp/tests/g_blend
	@$(MAKE) -s CTL=KEY_RECOMPUTE BUILD=build/ctl_kr build/ctl_kr/metalfg.metallib build/ctl_kr/tests/g_interm
	@echo "== build/ctl_kr/tests/g_interm (gpu, METALFG_CTL_KEY_RECOMPUTE)"; METALFG_METALLIB_DIR=build/ctl_kr $(GPU_RUN) build/ctl_kr/tests/g_interm

# control-build check (spec fg.test_scenes_r3): every METALFG_CTL_<NAME> build compiles and differs from the release, the release holds none of them
ctl-check:
	tools/ctl_check.sh

# test-file generator
$(BUILD)/genweights: tools/genweights.c tools/wgen.c tools/wgen.h src/weights.m src/weights.h src/sha256.m
	@mkdir -p $(BUILD)
	$(CC) -O1 -std=gnu11 -fobjc-arc -Iinclude -Isrc -Itools tools/genweights.c tools/wgen.c src/weights.m src/sha256.m -o $@

tools: $(BUILD)/genweights
.PHONY: tools

# setup-time extractor (universal binary, no GPU)
$(BUILD)/extractfg: tools/extract/main.c tools/extract/extract.c tools/extract/extract.h src/weights.m src/weights.h src/sha256.m src/sha256.h
	@mkdir -p $(BUILD)
	$(CC) -O2 -std=gnu11 -fobjc-arc $(ARCHS) -Isrc -Itools/extract tools/extract/main.c tools/extract/extract.c src/weights.m src/sha256.m -o $@

tools: $(BUILD)/extractfg

# tests of the built dylib through its C interface: no engine sources are compiled in
$(BUILD)/dtests/d_%: tests/d_%.m tools/wgen.c tools/wgen.h src/weights.m src/sha256.m
	@mkdir -p $(BUILD)/dtests $(BUILD)/tests
	$(CC) -O1 -g -std=gnu11 -fobjc-arc -Iinclude -Isrc -Itools -framework Foundation -framework Metal $< tools/wgen.c src/weights.m src/sha256.m -o $@
