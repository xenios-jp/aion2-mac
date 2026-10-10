#!/bin/bash
set -euo pipefail
umask 077
VERSION=0.1.4
RUNTIME_SHA=a4b5d63493f80698cce5cad8e7212d9a51c8292037b00c478f4652636fcfd331
FIXES_SHA=fdc45a0245242aeaac7f74e312710c96fdd4aabb16b7f5a0259cbf7e6dc414cf
ROOT="${AION2_MAC_HOME:-$HOME/Library/Application Support/Aion2Mac}"
GPTK="${AION2_GPTK_LIB:-}"
RUNTIME_ARCHIVE="${AION2_RUNTIME_ARCHIVE:-}" FIXES_ARCHIVE="${AION2_FIXES_ARCHIVE:-}" SKIP_STEAM=0 NO_LAUNCH=0 NO_APP=0 DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: install.sh [options]
  --gptk PATH             Apple toolkit libraries (mounted volumes are detected)
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
    --no-app) NO_APP=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done
if [ "$DRY_RUN" = 0 ] && [ -f "$ROOT/.aion2-mac" ] &&
   [ -x "$ROOT/scripts/start.command" ] &&
   [ -f "$ROOT/prefix/drive_c/Program Files (x86)/Steam/steam.exe" ] &&
   find -H "$ROOT" -maxdepth 1 -name ".ready-v*" | grep -q .; then
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
  # The app/Terminal installer also works when downloaded as a single script.
  candidates=()
  for volume in /Volumes/*; do
    [ -d "$volume" ] || continue
    for candidate in "$volume/redist/lib" "$volume/lib"; do
      [ -f "$candidate/external/D3DMetal.framework/Versions/A/D3DMetal" ] &&
      [ -f "$candidate/external/libd3dshared.dylib" ] &&
      [ -f "$candidate/wine/x86_64-windows/d3d12.dll" ] &&
      [ -f "$candidate/wine/x86_64-windows/nvngx-on-metalfx.dll" ] && candidates+=("$candidate")
    done
  done
  if [ "${#candidates[@]}" = 1 ]; then
    GPTK="${candidates[0]}"
    printf 'Found Apple toolkit: %s\n' "$GPTK"
  else
    action=$(osascript -e 'button returned of (display dialog "AION 2 needs Apple’s Game Porting Toolkit. Download it from Apple, open the included Evaluation environment disk image, then run AION 2 again. Mounted toolkit libraries are detected automatically." with title "Set up AION 2" buttons {"Cancel", "Choose Toolkit", "Open Apple Download"} default button "Open Apple Download" cancel button "Cancel")') || die 'Setup canceled.'
    if [ "$action" = "Open Apple Download" ]; then
      open 'https://developer.apple.com/games/game-porting-toolkit/'
      die 'Open the evaluation environment disk image, then run AION 2 again.'
    fi
    GPTK=$(osascript -e 'POSIX path of (choose folder with prompt "Choose your extracted Apple evaluation environment folder, or its redist/lib folder")') || die 'Setup canceled.'
    if [ -f "$GPTK/redist/lib/external/libd3dshared.dylib" ]; then GPTK="$GPTK/redist/lib"; fi
    if [ -f "$GPTK/lib/external/libd3dshared.dylib" ]; then GPTK="$GPTK/lib"; fi
  fi
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
progress() { printf "AION2_PROGRESS:%s\n" "$1"; }
step() { printf "AION2_STEP:%s\n" "$1"; progress "$2"; }
step 1 "Checking the files needed for setup…"
download() {
  local url="$1" dest="$2" label="$3" download_pid bytes mb
  progress "Downloading ${label}…"
  curl --silent --show-error --fail --location --retry 2 --connect-timeout 20 --max-time 1800 \
    --proto '=https' --tlsv1.2 "$url" --output "$dest.part" &
  download_pid=$!
  while kill -0 "$download_pid" 2>/dev/null; do
    bytes=$(stat -f '%z' "$dest.part" 2>/dev/null || printf 0)
    mb=$(awk -v bytes="$bytes" 'BEGIN {printf "%.1f", bytes / 1048576}')
    progress "Downloading ${label} — $mb MB received…"
    sleep 1
  done
  wait "$download_pid" || die "Could not download ${label}. Check your connection and try again."
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
  [ -f "$RUNTIME_ARCHIVE" ] || download 'https://github.com/dappermint/winecx-gptk/releases/download/runtime-v4.7.3/Libraries.tar.gz' "$RUNTIME_ARCHIVE" "the Windows compatibility runtime"
fi
if [ -z "$FIXES_ARCHIVE" ]; then
  FIXES_ARCHIVE="$ROOT/cache/fixes-v$VERSION.tar.gz"
  [ -f "$FIXES_ARCHIVE" ] || download "https://github.com/xenios-jp/aion2-mac/releases/download/v$VERSION/aion2-mac-fixes-v$VERSION.tar.gz" "$FIXES_ARCHIVE" "the AION 2 compatibility fixes"
fi
step 2 "Checking downloaded files against their SHA-256 checksums…"
verify "$RUNTIME_ARCHIVE" "$RUNTIME_SHA"
verify "$FIXES_ARCHIVE" "$FIXES_SHA"
check_archive "$RUNTIME_ARCHIVE"; check_archive "$FIXES_ARCHIVE"
step 3 "Unpacking the Windows compatibility runtime…"
if [ ! -f "$ROOT/.runtime-v4.7.3" ]; then
  printf 'Extracting Wine runtime…\n'
  tar -xzf "$RUNTIME_ARCHIVE" -C "$ROOT"
  touch "$ROOT/.runtime-v4.7.3"
fi
step 4 "Installing Apple’s D3DMetal graphics libraries…"
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
# Apple's NVAPI PE advertises the builtin module name nvapi.dll. Wine needs
# that alias even when the game asks for nvapi64.dll.
ln -sf nvapi64.dll "$ENGINE/lib/wine/x86_64-windows/nvapi.dll"
ln -sf ../../external/libd3dshared.dylib "$ENGINE/lib/wine/x86_64-unix/nvapi.so"
cp "$GPTK/wine/x86_64-windows/nvngx-on-metalfx.dll" "$ENGINE/lib/wine/x86_64-windows/nvngx.dll"
ln -sf ../../external/libd3dshared.dylib "$ENGINE/lib/wine/x86_64-unix/nvngx.so"
ditto "$STAGE/fixes" "$ROOT/fixes"
cp "$ROOT/fixes/bin/winegstreamer.dll" "$ENGINE/lib/wine/x86_64-windows/winegstreamer.dll"
ditto "$ROOT/fixes/scripts" "$ROOT/scripts"
chmod +x "$ROOT/scripts/"*.command "$ROOT/scripts/prepare-steam.sh"
source "$ROOT/scripts/env.sh"
step 5 "Creating the separate Windows environment for AION 2…"
WINEDEBUG=-all wine_run wineboot -u > "$ROOT/logs/setup.log" 2>&1
step 6 "Configuring Windows 10 compatibility…"
WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine' /v Version /d win10 /f >> "$ROOT/logs/setup.log" 2>&1
for dll in dxgi d3d11 d3d12 nvapi64 nvngx; do
  case "$dll" in
    dxgi) progress "Configuring the game’s display connection…";;
    d3d11) progress "Enabling D3DMetal for DirectX 11 video playback…";;
    d3d12) progress "Enabling D3DMetal for DirectX 12 graphics…";;
    nvapi64|nvngx) progress "Configuring graphics feature detection…";;
  esac
  WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine\AppDefaults\AION2.exe\DllOverrides' /v "$dll" /d builtin /f >> "$ROOT/logs/setup.log" 2>&1
done
progress "Disabling Steam’s in-game overlay for compatibility…"
for dll in gameoverlayrenderer gameoverlayrenderer64; do
  WINEDEBUG=-all wine_run reg add 'HKCU\Software\Wine\AppDefaults\AION2.exe\DllOverrides' /v "$dll" /d '' /f >> "$ROOT/logs/setup.log" 2>&1
done
progress "Installing the cinematic video and audio decoder…"
mkdir -p "$WINEPREFIX/drive_c/windows/system32"
cp "$ROOT/fixes/bin/winegstreamer.dll" "$WINEPREFIX/drive_c/windows/system32/winegstreamer.dll"
for dll in dxgi d3d11 d3d12 nvapi64 nvngx; do cp "$ENGINE/lib/wine/x86_64-windows/$dll.dll" "$WINEPREFIX/drive_c/windows/system32/$dll.dll"; done
if [ "$SKIP_STEAM" = 0 ]; then
  step 7 "Installing Steam in the game environment…"
  if [ ! -f "$WINEPREFIX/drive_c/Program Files (x86)/Steam/steam.exe" ]; then
    download 'https://cdn.akamai.steamstatic.com/client/installer/SteamSetup.exe' "$ROOT/cache/SteamSetup.exe" "Steam’s official installer"
    progress "Running Steam’s installer…"
    WINEDEBUG=-all wine_run "$ROOT/cache/SteamSetup.exe" /S >> "$ROOT/logs/setup.log" 2>&1
    touch "$ROOT/.steam-first-launch"
  fi
  step 8 "Configuring Steam to render correctly on your Mac…"
  WINEDEBUG=-all wine_run reg delete 'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' /v Steam /f >> "$ROOT/logs/setup.log" 2>&1 || true
  "$ROOT/scripts/prepare-steam.sh"
fi
step 9 "Finishing setup…"
if [ "$NO_APP" = 0 ] && [ "$SKIP_STEAM" = 0 ] && [ "$ROOT" = "$HOME/Library/Application Support/Aion2Mac" ]; then
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
