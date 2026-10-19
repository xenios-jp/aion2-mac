#!/bin/zsh
# Build check of the control builds (spec fg.test_scenes_r3, "Interoperability: control build contract"); run through `make ctl-check`.
#  1. The release build, compiled twice, is bit-identical, and defines neither METALFG_CTL nor any METALFG_CTL_<NAME>.
#  2. A build with METALFG_CTL alone is bit-identical to the release build (it changes no behaviour).
#  3. For every row of the table (tools/controls.txt) a build with METALFG_CTL=1 and METALFG_CTL_<NAME>=1 compiles (host and every shader),
#     its library or shader library differs from the release build, and the symbol occurs in the source tree.
#  4. The release artefacts hold none of the guarded behaviour: no control kernel names, and the planner signature of tests/t_ctl.c
#     differs from the release signature in every control build whose row says "planner" and is equal for METALFG_CTL alone.
set -u
cd "${0:A:h}/.."
ARCH="-arch arm64"
OUT=build/ctl
rm -rf $OUT
mkdir -p $OUT
fail=0
note() { print -r -- "$@"; }
bad() { note "FAIL: $@"; fail=$((fail + 1)); }

build() {   # build <dir> <make variable assignments...>
  local dir=$1; shift
  make -s BUILD=$dir ARCHS="$ARCH" "$@" all -j8 > $OUT/$(basename $dir).log 2>&1
}
sig() {     # sig <dir> <make variable assignments...>: planner signature
  local dir=$1; shift
  make -s BUILD=$dir ARCHS="$ARCH" "$@" $dir/tests/t_ctl > $OUT/sig_$(basename $dir).log 2>&1 || { echo "build-failed"; return; }
  $dir/tests/t_ctl
}
same() { cmp -s "$1/libmetalfg.dylib" "$2/libmetalfg.dylib" && cmp -s "$1/metalfg.metallib" "$2/metalfg.metallib"; }

build $OUT/rel1 || bad "release build"
build $OUT/rel2 || bad "release build (second)"
same $OUT/rel1 $OUT/rel2 && note "ok: the release build, compiled twice, is bit-identical" || bad "release builds differ"
build $OUT/ctlonly CTLONLY=1 || bad "METALFG_CTL alone does not compile"
same $OUT/rel1 $OUT/ctlonly && note "ok: METALFG_CTL alone changes nothing" || bad "METALFG_CTL alone changes the build"

relsig=$(sig $OUT/rel1)
onlysig=$(sig $OUT/ctlonly CTLONLY=1)
[[ "$relsig" == "$onlysig" ]] && note "ok: planner signature with METALFG_CTL alone equals the release ($relsig)" || bad "planner signature changed by METALFG_CTL alone"

# the release artefacts hold no control kernel
for k in ctl_blend up_half_add f19_weights_feat; do
  if strings $OUT/rel1/metalfg.metallib $OUT/rel1/libmetalfg.dylib | grep -q "$k"; then bad "release build contains $k"; fi
done
grep -q "METALFG_CTL" <(strings $OUT/rel1/libmetalfg.dylib) && bad "release dylib mentions METALFG_CTL"
note "ok: no control kernel in the release artefacts"

n=0
while read -r name kind; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  n=$((n + 1))
  d=$OUT/c_$name
  if ! grep -rq "METALFG_CTL_$name" src shaders include; then bad "METALFG_CTL_$name does not appear in the source tree"; fi
  if build $d CTL=$name; then
    if same $OUT/rel1 $d; then bad "METALFG_CTL_$name builds but changes neither library nor shader library"; else note "ok: METALFG_CTL_$name compiles and differs from the release"; fi
  else
    bad "METALFG_CTL_$name does not compile (see $OUT/c_$name.log)"
  fi
  if [[ "$kind" == planner ]]; then
    s=$(sig $d CTL=$name)
    if [[ "$s" == "$relsig" || "$s" == build-failed ]]; then bad "METALFG_CTL_$name does not change the planner signature ($s)"; else note "ok:   planner signature differs ($s)"; fi
  fi
done < tools/controls.txt
note "controls checked: $n"
if (( fail )); then note "ctl-check: $fail FAILURES"; exit 1; fi
note "ctl-check: ok"
