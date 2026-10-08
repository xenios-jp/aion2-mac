#!/bin/bash
# Integration regression: run against a disposable installed bottle.
set -euo pipefail
INSTALL="${1:?Pass an Aion2Mac installation directory}"
GENUINE="${2:?Pass a genuine Steam steamwebhelper.exe}"
source "$INSTALL/scripts/env.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
WRAPPER="$ROOT/fixes/bin/steamwebhelper-wrapper.exe"
cp "$WRAPPER" "$TMP/steamwebhelper.exe"
cp "$WRAPPER" "$TMP/steamwebhelper_real.exe"
# Different wrapper bytes reproduce the upgrade misclassification.
printf 'older-build-fixture' >> "$TMP/steamwebhelper_real.exe"
expect() {
  local wanted=$1; shift
  local actual=0
  WINEDEBUG=-all wine_run "$@" >/dev/null 2>&1 || actual=$?
  [ "$actual" = "$wanted" ] || { echo "Expected $wanted, got $actual" >&2; exit 1; }
}
expect 0 "$WRAPPER" --check-real "$GENUINE"
expect 2 "$WRAPPER" --check-real "$TMP/steamwebhelper_real.exe"
expect 1 "$WRAPPER" --check-real "$TMP/missing.exe"
expect 86 "$TMP/steamwebhelper.exe"
echo 'Steam wrapper regression checks passed.'
