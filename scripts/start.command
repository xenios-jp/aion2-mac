#!/bin/bash
set -euo pipefail
umask 077
source "$(dirname "$0")/env.sh"
mkdir -p "$ROOT/logs"
"$ROOT/scripts/prepare-steam.sh"
if [ ! -f "$WINEPREFIX/drive_c/Program Files (x86)/Steam/steamapps/appmanifest_3393110.acf" ]; then
  echo 'Sign into Steam and install AION 2, then launch AION 2.app again.'
  "$ROOT/scripts/steam.command" steam://install/3393110
  exit 0
fi
SIZE="${AION2_RESOLUTION:-}"
if [ -z "$SIZE" ]; then SIZE=$(WINEDEBUG=-all wine_run "$ROOT/fixes/bin/display-size.exe" 2>/dev/null | tr -d '\r' | tail -1); fi
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || { echo 'Set AION2_RESOLUTION, for example 2560x1080.' >&2; exit 1; }
WIDTH=${SIZE%x*}; HEIGHT=${SIZE#*x}
# Refresh stale routing before the game opens its audio stream.
"$ROOT/scripts/refresh-audio.command" > "$ROOT/logs/audio.log" 2>&1 || true
env -u DYLD_INSERT_LIBRARIES "$ROOT/fixes/bin/audio-follow" \
  "$ROOT/scripts/refresh-audio.command" "$ROOT/logs/audio-follow.lock" \
  >> "$ROOT/logs/audio.log" 2>&1 &
MODE=fullscreen
[ "${AION2_WINDOWED:-0}" = 1 ] && MODE=windowed
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
# A bounded helper presses No only on AION2's exact optional driver dialog.
WINEDEBUG=-all wine_run "$ROOT/fixes/bin/driver-notice.exe" >> "$ROOT/logs/driver-notice.log" 2>&1 &
NOTICE_PID=$!
trap 'kill "$NOTICE_PID" 2>/dev/null || true' EXIT
cd "$WINEPREFIX/drive_c/Program Files (x86)/Steam"
printf 'Launching Aion 2: %s, MSync=%s, Metal HUD=%s, experimental DLSS=%s\n' "$SIZE" "$WINEMSYNC" "$MTL_HUD_ENABLED" "$D3DM_ENABLE_METALFX"
# root is the real Mac desktop, not a constrained emulated display.
DYLD_INSERT_LIBRARIES="$ROOT/fixes/bin/wine-native-errno.dylib" \
  "$WINE" explorer.exe "/desktop=root,$SIZE" 'C:\Program Files (x86)\Steam\steam.exe' \
  -silent -no-cef-sandbox -cef-single-process -noverifyfiles -applaunch 3393110 \
  "-$MODE" "-ResX=$WIDTH" "-ResY=$HEIGHT" "$@" >> "$ROOT/logs/aion2.log" 2>&1
