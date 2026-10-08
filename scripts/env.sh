#!/bin/bash
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export WINEPREFIX="$ROOT/prefix"
ENGINE="$ROOT/Libraries/Wine"
export WINE="$ENGINE/bin/wine" WINESERVER="$ENGINE/bin/wineserver"
export PATH="$ENGINE/bin:$PATH" WINEDLLPATH="$ENGINE/lib/wine"
export DYLD_FALLBACK_LIBRARY_PATH="$ENGINE/lib:/usr/local/lib:/usr/lib"
export WINEDLLOVERRIDES='mscoree,mshtml='
export ROSETTA_ADVERTISE_AVX=1
export WINEMSYNC="${AION2_MSYNC:-1}" WINEESYNC=0
export MTL_HUD_ENABLED="${AION2_HUD:-0}"
export MTL_HUD_LOGGING_ENABLED="${AION2_HUD_LOGGING:-0}"
# Explicit opt-in until Aion's DLSS menu is validated.
export D3DM_ENABLE_METALFX="${AION2_DLSS:-0}"
if [ "$D3DM_ENABLE_METALFX" = 1 ]; then
  export D3DM_VENDOR_ID=4318 D3DM_DEVICE_ID=9988
  export D3DM_DEVICE_DESCRIPTION='NVIDIA GeForce RTX 4080'
fi
export WINEDEBUG="${AION2_WINEDEBUG:--all,err+all}"
# Scope the x86_64 interposer to Wine, not native arm64 shell utilities.
wine_run() { DYLD_INSERT_LIBRARIES="$ROOT/fixes/bin/wine-native-errno.dylib" "$WINE" "$@"; }
