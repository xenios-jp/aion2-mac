#!/bin/bash
set -euo pipefail
umask 077
VERSION=0.1.1
RUNTIME_SHA=a4b5d63493f80698cce5cad8e7212d9a51c8292037b00c478f4652636fcfd331
FIXES_SHA=f3ed553885c687e608af5485f12404b3ebcc43bbe3c19765a01195f7b267c7f1
ROOT="${AION2_MAC_HOME:-$HOME/Library/Application Support/Aion2Mac}"
GPTK="${AION2_GPTK_LIB:-}"
RUNTIME_ARCHIVE= FIXES_ARCHIVE= SKIP_STEAM=0 NO_LAUNCH=0 DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: install.sh --gptk /path/to/apple/payload/lib [options]
  --root PATH             Dedicated installation directory
  --runtime-archive PATH  Use a local, checksum-verified runtime archive
  --fixes-archive PATH    Use a local, checksum-verified fixes archive
  --skip-steam            Prepare the bottle without downloading Steam
  --no-launch             Finish setup without opening Steam
  --dry-run               Validate prerequisites and print the plan
EOF
}
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gptk|--root|--runtime-archive|--fixes-archive)
      [ "$#" -ge 2 ] || die "$1 needs a path"
      case "$1" in
        --gptk) GPTK="$2";; --root) ROOT="$2";;
        --runtime-archive) RUNTIME_ARCHIVE="$2";; --fixes-archive) FIXES_ARCHIVE="$2";;
      esac; shift 2;;
    --skip-steam) SKIP_STEAM=1; shift;;
    --no-launch) NO_LAUNCH=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done
if [ -f "$ROOT/.ready-v$VERSION" ] && [ "$DRY_RUN" = 0 ]; then
  printf 'Already installed: %s/scripts/start.command\n' "$ROOT"
  if [ "$NO_LAUNCH" = 0 ]; then
    if [ -d "$ROOT/AION 2.app" ]; then open "$ROOT/AION 2.app"; else open "$ROOT/scripts/start.command"; fi
  fi
  exit 0
fi
[ "$(uname -s)" = Darwin ] || die 'This installer runs on macOS.'
[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ] || die 'The pinned runtime requires macOS 26 or newer.'
[ "$(uname -m)" = arm64 ] || die 'Use an Apple silicon Mac and a native Terminal.'
if [ -z "$GPTK" ]; then
  GPTK=$(osascript -e 'POSIX path of (choose folder with prompt "Select the Apple toolkit lib folder containing external and wine")') || die 'Setup canceled. Run again when your Apple toolkit is ready.'
fi
for file in external/libd3dshared.dylib external/D3DMetal.framework/Versions/A/D3DMetal wine/x86_64-windows/d3d12.dll wine/x86_64-windows/nvngx-on-metalfx.dll; do
  [ -f "$GPTK/$file" ] || die "Apple payload is incomplete: $GPTK/$file"
done
case "$ROOT" in /*) ;; *) die '--root must be an absolute path';; esac
[ "$ROOT" != / ] && [ "$ROOT" != "$HOME" ] || die 'Choose a dedicated installation directory.'
if [ -d "$ROOT" ] && [ ! -f "$ROOT/.aion2-mac" ] && [ -n "$(ls -A "$ROOT")" ]; then
  die 'The destination already contains unrelated files. Choose an empty --root.'
fi
printf 'Aion 2 Mac %s\nBottle: %s\nApple payload: %s\n' "$VERSION" "$ROOT/prefix" "$GPTK"
if [ "$DRY_RUN" = 1 ]; then
  printf 'Plan: verify pinned downloads, stage D3DMetal, create a fresh bottle, install compatibility fixes, install Steam.\n'
  exit 0
fi
if ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
  osascript -e 'display dialog "AION 2 needs Rosetta 2. Install it now? Apple’s installer will ask you to accept its license in Terminal." buttons {"Cancel", "Install"} default button "Install" cancel button "Cancel"' >/dev/null || die 'Rosetta installation canceled.'
  /usr/sbin/softwareupdate --install-rosetta </dev/tty || die 'Rosetta installation failed. Run: softwareupdate --install-rosetta'
  /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null || die 'Rosetta is still unavailable. Complete its installation and try again.'
fi
mkdir -p "$ROOT"
mkdir "$ROOT/.install-lock" 2>/dev/null || die 'Another installer is running (or remove a stale .install-lock).'
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aion2-mac.XXXXXX")
trap 'rm -rf "$STAGE"; rmdir "$ROOT/.install-lock" 2>/dev/null || true' EXIT
touch "$ROOT/.aion2-mac"
mkdir -p "$ROOT/cache" "$ROOT/logs" "$ROOT/scripts"
download() {
  local url="$1" dest="$2"
  curl --fail --location --retry 2 --proto '=https' --tlsv1.2 "$url" --output "$dest.part"
  mv "$dest.part" "$dest"
}
verify() {
  local actual
  actual=$(shasum -a 256 "$1" | awk '{print $1}')
  [ "$actual" = "$2" ] || die "Checksum mismatch: $1"
}
check_archive() {
  tar -tzf "$1" > "$STAGE/archive-list"
  awk '/^\// || /(^|\/)\.\.(\/|$)/ {bad=1} END {exit bad}' "$STAGE/archive-list" || die 'Unsafe archive member path.'
}
if [ -z "$RUNTIME_ARCHIVE" ]; then
  RUNTIME_ARCHIVE="$ROOT/cache/Libraries.tar.gz"
  [ -f "$RUNTIME_ARCHIVE" ] || download 'https://github.com/dappermint/winecx-gptk/releases/download/runtime-v4.7.3/Libraries.tar.gz' "$RUNTIME_ARCHIVE"
fi
if [ -z "$FIXES_ARCHIVE" ]; then
  FIXES_ARCHIVE="$ROOT/cache/fixes-v$VERSION.tar.gz"
  [ -f "$FIXES_ARCHIVE" ] || download "https://github.com/xenios-jp/aion2-mac/releases/download/v$VERSION/aion2-mac-fixes-v$VERSION.tar.gz" "$FIXES_ARCHIVE"
fi
verify "$RUNTIME_ARCHIVE" "$RUNTIME_SHA"
verify "$FIXES_ARCHIVE" "$FIXES_SHA"
check_archive "$RUNTIME_ARCHIVE"; check_archive "$FIXES_ARCHIVE"
if [ ! -f "$ROOT/.runtime-v4.7.3" ]; then
  printf 'Extracting Wine runtime…\n'
  tar -xzf "$RUNTIME_ARCHIVE" -C "$ROOT"
  touch "$ROOT/.runtime-v4.7.3"
fi
mkdir -p "$STAGE/fixes"
tar -xzf "$FIXES_ARCHIVE" -C "$STAGE/fixes"
ENGINE="$ROOT/Libraries/Wine"
mkdir -p "$ENGINE/lib/external" "$ROOT/fixes"
ditto "$GPTK/external" "$ENGINE/lib/external"
for dll in dxgi d3d10 d3d11 d3d12 nvapi64; do
  [ -f "$GPTK/wine/x86_64-windows/$dll.dll" ] || die "Apple payload missing $dll.dll"
  cp "$GPTK/wine/x86_64-windows/$dll.dll" "$ENGINE/lib/wine/x86_64-windows/$dll.dll"
  ln -sf ../../external/libd3dshared.dylib "$ENGINE/lib/wine/x86_64-unix/$dll.so"
done
cp "$GPTK/wine/x86_64-windows/nvngx-on-metalfx.dll" "$ENGINE/lib/wine/x86_64-windows/nvngx.dll"
ln -sf ../../external/libd3dshared.dylib "$ENGINE/lib/wine/x86_64-unix/nvngx.so"
ditto "$STAGE/fixes" "$ROOT/fixes"
cp "$ROOT/fixes/bin/winegstreamer.dll" "$ENGINE/lib/wine/x86_64-windows/winegstreamer.dll"
ditto "$ROOT/fixes/scripts" "$ROOT/scripts"
chmod +x "$ROOT/scripts/"*.command "$ROOT/scripts/prepare-steam.sh"
source "$ROOT/scripts/env.sh"
printf 'Creating isolated Windows bottle…\n'
WINEDEBUG=-all wine_run wineboot -u > "$ROOT/logs/setup.log" 2>&1
WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine' /v Version /d win10 /f >> "$ROOT/logs/setup.log" 2>&1
for dll in dxgi d3d11 d3d12 nvapi64 nvngx; do
  WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine\AppDefaults\AION2.exe\DllOverrides' /v "$dll" /d builtin /f >> "$ROOT/logs/setup.log" 2>&1
done
for dll in gameoverlayrenderer gameoverlayrenderer64; do
  WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine\AppDefaults\AION2.exe\DllOverrides' /v "$dll" /d '' /f >> "$ROOT/logs/setup.log" 2>&1
done
mkdir -p "$WINEPREFIX/drive_c/windows/system32"
cp "$ROOT/fixes/bin/winegstreamer.dll" "$WINEPREFIX/drive_c/windows/system32/winegstreamer.dll"
for dll in dxgi d3d11 d3d12 nvapi64 nvngx; do cp "$ENGINE/lib/wine/x86_64-windows/$dll.dll" "$WINEPREFIX/drive_c/windows/system32/$dll.dll"; done
if [ "$SKIP_STEAM" = 0 ]; then
  if [ ! -f "$WINEPREFIX/drive_c/Program Files (x86)/Steam/steam.exe" ]; then
    download 'https://cdn.akamai.steamstatic.com/client/installer/SteamSetup.exe' "$ROOT/cache/SteamSetup.exe"
    WINEDEBUG=-all wine_run "$ROOT/cache/SteamSetup.exe" /S >> "$ROOT/logs/setup.log" 2>&1
  fi
  WINEDEBUG=-all wine_run reg delete 'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' /v Steam /f >> "$ROOT/logs/setup.log" 2>&1 || true
  "$ROOT/scripts/prepare-steam.sh"
fi
if [ "$SKIP_STEAM" = 0 ] && [ "$ROOT" = "$HOME/Library/Application Support/Aion2Mac" ]; then
  if [ ! -d "$ROOT/AION 2.app" ]; then "$ROOT/fixes/scripts/build-app.sh" "$ROOT"; fi
  if [ -d "$HOME/Desktop" ] && [ ! -e "$HOME/Desktop/AION 2.app" ] && [ ! -L "$HOME/Desktop/AION 2.app" ]; then
    ln -s "$ROOT/AION 2.app" "$HOME/Desktop/AION 2.app"
  fi
fi
if [ "$SKIP_STEAM" = 0 ]; then touch "$ROOT/.ready-v$VERSION"; fi
printf '\nReady. Open: %s/scripts/start.command\n' "$ROOT"
printf 'Log in to Steam, install AION 2 (app 3393110), and install its prerequisites when prompted.\n'
if [ "$NO_LAUNCH" = 0 ] && [ "$SKIP_STEAM" = 0 ]; then
  if [ -d "$ROOT/AION 2.app" ]; then open "$ROOT/AION 2.app"; else open "$ROOT/scripts/start.command"; fi
fi
