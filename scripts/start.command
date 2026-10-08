#!/bin/bash
set -euo pipefail
umask 077
source "$(dirname "$0")/env.sh"
mkdir -p "$ROOT/logs"
"$ROOT/scripts/prepare-steam.sh"
SIZE="${AION2_RESOLUTION:-}"
if [ -z "$SIZE" ]; then SIZE=$(WINEDEBUG=-all wine_run "$ROOT/fixes/bin/display-size.exe" 2>/dev/null | tr -d '\r' | tail -1); fi
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || { echo 'Set AION2_RESOLUTION, for example 2560x1080.' >&2; exit 1; }
WIDTH=${SIZE%x*}; HEIGHT=${SIZE#*x}
# Find the Windows user's game configuration without guessing its username.
for USERDIR in "$WINEPREFIX/drive_c/users/"*; do
  [ -d "$USERDIR" ] || continue
  USERNAME=${USERDIR##*/}
  case "$USERNAME" in Public|All\ Users|Default*) continue;; esac
  CONFIG="$USERDIR/AppData/Local/Aion2/Saved_Steam/Config/Windows"
  mkdir -p "$CONFIG"
  if ! grep -q '^Electra.Win.H264UseOldOutputPath=' "$CONFIG/Engine.ini" 2>/dev/null; then
    printf '\n[SystemSettings]\nElectra.Win.H264UseOldOutputPath=1\nElectra.Win.H265UseOldOutputPath=1\n' >> "$CONFIG/Engine.ini"
  fi
done
cd "$WINEPREFIX/drive_c/Program Files (x86)/Steam"
printf 'Launching Aion 2: %s, MSync=%s, Metal HUD=%s, experimental DLSS=%s\n' "$SIZE" "$WINEMSYNC" "$MTL_HUD_ENABLED" "$D3DM_ENABLE_METALFX"
# root is the real Mac desktop, not a constrained emulated display.
DYLD_INSERT_LIBRARIES="$ROOT/fixes/bin/wine-native-errno.dylib" \
  "$WINE" explorer.exe "/desktop=root,$SIZE" 'C:\Program Files (x86)\Steam\steam.exe' \
  -no-cef-sandbox -cef-single-process -noverifyfiles -applaunch 3393110 \
  -fullscreen "-ResX=$WIDTH" "-ResY=$HEIGHT" "$@" >> "$ROOT/logs/aion2.log" 2>&1
