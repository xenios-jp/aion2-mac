#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEAM="$WINEPREFIX/drive_c/Program Files (x86)/Steam"
HELPER="$STEAM/bin/cef/cef.win64/steamwebhelper.exe"
WRAPPER="$ROOT/fixes/bin/steamwebhelper-wrapper.exe"
[ -f "$STEAM/steam.exe" ] || { echo 'Steam is not installed. Run install.sh without --skip-steam.' >&2; exit 1; }
client_updated() {
  local last
  /usr/bin/file "$STEAM/steam.exe" | grep -q 'PE32+' || return 1
  [ -f "$HELPER" ] || return 1
  last=$(tail -40 "$STEAM/logs/bootstrap_log.txt" 2>/dev/null |
    awk '/Downloading update|Extracting package|Installing update|Update complete|Verifying installation|Verification complete/ {last=$0} END {print last}')
  [[ "$last" = *"Verification complete"* ]]
}
FIRST_SETUP=0
if [ ! -f "$HELPER" ]; then FIRST_SETUP=1; fi
if [ -f "$ROOT/.steam-first-launch" ] && [ ! -f "$STEAM/steamapps/appmanifest_3393110.acf" ]; then FIRST_SETUP=1; fi
if [ "$FIRST_SETUP" = 1 ]; then
  printf 'AION2_STEAM_BOOTSTRAP:1\nAION2_PROGRESS:Updating Steam to its current 64-bit client. This can take several minutes…\n'
  wine_run "$STEAM/steam.exe" -silent -no-cef-sandbox -cef-single-process >> "$ROOT/logs/steam-bootstrap.log" 2>&1 &
  stable=0
  for ((attempt=0; attempt<300; attempt++)); do
    if client_updated; then stable=$((stable+1)); else stable=0; fi
    # Wait through both the initial client and its 32-to-64-bit self-update.
    [ "$stable" -lt 5 ] || break
    sleep 2
  done
  if [ "$stable" -lt 5 ]; then echo 'Steam update has not finished. Re-run setup later; see local steam-bootstrap.log.' >&2; exit 1; fi
  wine_run "$STEAM/steam.exe" -shutdown >> "$ROOT/logs/steam-bootstrap.log" 2>&1 || true
  sleep 2
  # No game is installed yet; this server belongs exclusively to our fresh bottle.
  "$WINESERVER" -k; "$WINESERVER" -w
  rm -f "$ROOT/.steam-first-launch"
  printf 'AION2_STEAM_BOOTSTRAP:0\nAION2_PROGRESS:Installing Steam’s display compatibility fix…\n'
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
