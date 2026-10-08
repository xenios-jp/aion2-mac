#!/bin/bash
set -euo pipefail
umask 077
source "$(dirname "$0")/env.sh"
mkdir -p "$ROOT/logs"
"$ROOT/scripts/prepare-steam.sh"
URL="${1:-steam://open/library}"
case "$URL" in steam://open/library|steam://install/3393110) ;; *) echo 'Unsupported Steam action' >&2; exit 1;; esac
if [ -f "$ROOT/.steam-first-launch" ]; then
  "$ROOT/scripts/steam-first-run.sh" "$URL" >> "$ROOT/logs/steam-first-run.log" 2>&1 &
fi
"$ROOT/scripts/steam-ready.sh" >/dev/null 2>&1 &
cd "$WINEPREFIX/drive_c/Program Files (x86)/Steam"
wine_run explorer.exe /desktop=root 'C:\Program Files (x86)\Steam\steam.exe' \
  -no-cef-sandbox -cef-single-process -noverifyfiles "$URL" >> "$ROOT/logs/steam.log" 2>&1
