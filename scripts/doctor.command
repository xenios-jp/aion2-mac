#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
printf 'Aion 2 Mac diagnostics\nmacOS: %s\nCPU: %s\n' "$(sw_vers -productVersion)" "$(uname -m)"
"$WINE" --version
for file in "$ENGINE/lib/external/libd3dshared.dylib" "$ROOT/fixes/bin/wine-native-errno.dylib" "$ENGINE/lib/wine/x86_64-windows/winegstreamer.dll" "$WINEPREFIX/drive_c/Program Files (x86)/Steam/bin/cef/cef.win64/steamwebhelper_real.exe"; do
  if [ -f "$file" ]; then printf 'OK: %s\n' "${file#"$ROOT/"}"; else printf 'MISSING: %s\n' "${file#"$ROOT/"}"; fi
done
printf 'MSync=%s HUD=%s experimental-DLSS=%s\n' "$WINEMSYNC" "$MTL_HUD_ENABLED" "$D3DM_ENABLE_METALFX"
printf 'Game logs stay local; do not upload raw logs or your bottle.\n'
