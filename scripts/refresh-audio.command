#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export WINEDEBUG=-all MVK_CONFIG_LOG_LEVEL=0
wine_run "$ROOT/fixes/bin/audio-default.exe" --apply
