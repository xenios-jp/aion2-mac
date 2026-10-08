#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="${1:-$REPO/build}"
mkdir -p "$OUT"
APP="$OUT/AION 2.app"
[ ! -e "$APP" ] || { echo "App already exists: $APP" >&2; exit 1; }
osacompile -o "$APP" "$REPO/src/launcher.applescript"
cp "$REPO/assets/Aion.icns" "$APP/Contents/Resources/Aion.icns"
/usr/libexec/PlistBuddy -c 'Set :CFBundleIconFile Aion.icns' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Delete :CFBundleIconName' "$APP/Contents/Info.plist"
cat > "$APP/Contents/Resources/setup.command" <<'SCRIPT'
#!/bin/bash
set -euo pipefail
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aion2-setup.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
curl -fsSL --retry 2 --proto '=https' https://raw.githubusercontent.com/xenios-jp/aion2-mac/v0.1.1/install.sh -o "$STAGE/install.sh"
/bin/bash "$STAGE/install.sh"
SCRIPT
cat > "$APP/Contents/Resources/launch.command" <<'SCRIPT'
#!/bin/bash
set -euo pipefail
ROOT="$HOME/Library/Application Support/Aion2Mac"
"$ROOT/scripts/start.command"
SCRIPT
chmod +x "$APP/Contents/Resources/"*.command
/usr/libexec/PlistBuddy -c 'Set :CFBundleName AION 2' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :LSUIElement bool true' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :CFBundleIdentifier string jp.xenios.aion2mac' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :LSApplicationCategoryType string public.app-category.games' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :CFBundleShortVersionString string 0.1.1' "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
printf 'Built %s\n' "$APP"
