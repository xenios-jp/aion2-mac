#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEAM="$WINEPREFIX/drive_c/Program Files (x86)/Steam"
HELPER="$STEAM/bin/cef/cef.win64/steamwebhelper.exe"
WRAPPER="$ROOT/fixes/bin/steamwebhelper-wrapper.exe"
[ -f "$STEAM/steam.exe" ] || { echo 'Steam is not installed. Run install.sh without --skip-steam.' >&2; exit 1; }
if [ ! -f "$HELPER" ]; then
  printf 'Downloading Steam client files (first setup can take several minutes)…\n'
  wine_run "$STEAM/steam.exe" -silent -no-cef-sandbox -cef-single-process >> "$ROOT/logs/steam-bootstrap.log" 2>&1 &
  for ((attempt=0; attempt<180; attempt++)); do
    [ ! -f "$HELPER" ] || break
    sleep 2
  done
  if [ ! -f "$HELPER" ]; then echo 'Steam update has not finished. Re-run setup later; see local steam-bootstrap.log.' >&2; exit 1; fi
  wine_run "$STEAM/steam.exe" -shutdown >> "$ROOT/logs/steam-bootstrap.log" 2>&1 || true
  sleep 2
  # This server belongs exclusively to our fresh bottle.
  "$WINESERVER" -k; "$WINESERVER" -w
fi
valid_real() { WINEDEBUG=-all wine_run "$WRAPPER" --check-real "$1" >/dev/null 2>&1; }
REAL="${HELPER%/*}/steamwebhelper_real.exe"
if cmp -s "$HELPER" "$WRAPPER"; then
  valid_real "$REAL" || { echo 'Steam helper backup is missing or is another wrapper. Repair Steam client files before continuing.' >&2; exit 1; }
elif valid_real "$HELPER"; then
  cp "$HELPER" "${HELPER%/*}/steamwebhelper_real.exe"
  cp "$WRAPPER" "$HELPER"
elif valid_real "$REAL"; then
  # Upgrade an older wrapper without overwriting the genuine CEF backup.
  cp "$WRAPPER" "$HELPER"
else
  echo 'Neither Steam helper is a usable original. Repair Steam client files before continuing.' >&2
  exit 1
fi
