#!/bin/bash
set -euo pipefail
# Exercise the real installer against a disposable old bottle, without Wine.
REPO=$(cd "$(dirname "$0")/.." && pwd)
ARCHIVE=${1:?Pass the release fixes archive}
CHECK_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/aion2-upgrade-check.XXXXXX")
trap 'rm -rf "$CHECK_ROOT"' EXIT
mkdir -p "$CHECK_ROOT/scripts" "$CHECK_ROOT/prefix/drive_c/Program Files (x86)/Steam"
touch "$CHECK_ROOT/.aion2-mac" "$CHECK_ROOT/.ready-v0.1.4"
printf '#!/bin/bash\n# old launcher\n' > "$CHECK_ROOT/scripts/start.command"
chmod +x "$CHECK_ROOT/scripts/start.command"
printf 'game settings sentinel\n' > "$CHECK_ROOT/prefix/settings-sentinel"
touch "$CHECK_ROOT/prefix/drive_c/Program Files (x86)/Steam/steam.exe"
bash "$REPO/install.sh" --root "$CHECK_ROOT" --fixes-archive "$ARCHIVE" --no-app --no-launch
test -f "$CHECK_ROOT/.ready-v0.1.5"
cmp "$REPO/scripts/env.sh" "$CHECK_ROOT/scripts/env.sh"
cmp "$REPO/scripts/start.command" "$CHECK_ROOT/scripts/start.command"
test "$(cat "$CHECK_ROOT/prefix/settings-sentinel")" = 'game settings sentinel'
# An up-to-date bottle must not fetch or re-unpack anything.
bash "$REPO/install.sh" --root "$CHECK_ROOT" --fixes-archive /missing.tar.gz --no-app --no-launch
rm "$CHECK_ROOT/.ready-v0.1.5"
printf 'invalid archive\n' > "$CHECK_ROOT/bad.tar.gz"
if bash "$REPO/install.sh" --root "$CHECK_ROOT" --fixes-archive "$CHECK_ROOT/bad.tar.gz" --no-app --no-launch; then
  echo 'FAIL: accepted an invalid compatibility archive' >&2
  exit 1
fi
test ! -d "$CHECK_ROOT/.install-lock"
test "$(cat "$CHECK_ROOT/prefix/settings-sentinel")" = 'game settings sentinel'
printf 'PASS: existing-bottle upgrade, preserved settings, repeat setup, checksum rejection and lock cleanup\n'
