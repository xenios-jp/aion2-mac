#!/bin/bash
# Steam's initial 32-to-64-bit self-update can replace the CEF wrapper.
set -euo pipefail
source "$(dirname "$0")/env.sh"
URL="${1:?Steam action required}"
case "$URL" in steam://open/library|steam://install/3393110) ;; *) exit 2;; esac
[ -f "$ROOT/.steam-first-launch" ] || exit 0
STEAM="$WINEPREFIX/drive_c/Program Files (x86)/Steam"
HELPER="$STEAM/bin/cef/cef.win64/steamwebhelper.exe"
WRAPPER="$ROOT/fixes/bin/steamwebhelper-wrapper.exe"
# This recovery is only for a fresh bottle without an installed Aion game.
[ ! -f "$STEAM/steamapps/appmanifest_3393110.acf" ] || exit 0
for ((attempt=0; attempt<90; attempt++)); do
  sleep 2
  if [ -f "$HELPER" ] && ! cmp -s "$HELPER" "$WRAPPER"; then
    echo 'Steam updated its UI helper; restarting initial setup once.'
    WINEDEBUG=-all wine_run "$STEAM/steam.exe" -shutdown >/dev/null 2>&1 || true
    "$WINESERVER" -k
    "$WINESERVER" -w
    "$ROOT/scripts/prepare-steam.sh"
    rm -f "$ROOT/.steam-first-launch"
    cd "$STEAM"
    wine_run explorer.exe /desktop=root 'C:\Program Files (x86)\Steam\steam.exe' \
      -no-cef-sandbox -cef-single-process -noverifyfiles "$URL" >> "$ROOT/logs/steam.log" 2>&1
    exit 0
  fi
done
rm -f "$ROOT/.steam-first-launch"
