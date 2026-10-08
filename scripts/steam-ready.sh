#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir "$ROOT/.steam-ui-observer-lock" 2>/dev/null || exit 0
trap 'rmdir "$ROOT/.steam-ui-observer-lock" 2>/dev/null || true' EXIT
# A first-update repair can restart wineserver once; keep observing afterward.
for attempt in 1 2; do
  if WINEDEBUG=-all wine_run "$ROOT/fixes/bin/steam-ready.exe" --watch > "$ROOT/logs/steam-ui-state" 2>/dev/null; then exit 0; fi
  sleep 2
done
