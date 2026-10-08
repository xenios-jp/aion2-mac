#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="${1:-$REPO/build}"
VERSION=0.1.4
mkdir -p "$OUT"
APP="$OUT/AION 2.app"
[ ! -e "$APP" ] || { echo "App already exists: $APP" >&2; exit 1; }
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
if [ -x "$REPO/bin/aion2-launcher" ]; then
  cp "$REPO/bin/aion2-launcher" "$APP/Contents/MacOS/AION2Mac"
else
  xcrun swiftc -O -parse-as-library -module-cache-path "${TMPDIR:-/tmp}/aion2-swift-module-cache" -target arm64-apple-macos26.0 \
    "$REPO/src/Launcher.swift" -o "$APP/Contents/MacOS/AION2Mac"
fi
cp "$REPO/assets/Aion.icns" "$APP/Contents/Resources/Aion.icns"
cp "$REPO/assets/Aion.png" "$APP/Contents/Resources/Aion.png"
if [ -f "$REPO/install.sh" ]; then
  cp "$REPO/install.sh" "$APP/Contents/Resources/install.sh"
else
  curl -fsSL --retry 2 --proto '=https' "https://raw.githubusercontent.com/xenios-jp/aion2-mac/v$VERSION/install.sh" \
    -o "$APP/Contents/Resources/install.sh"
fi
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>AION 2</string>
<key>CFBundleDisplayName</key><string>AION 2</string>
<key>CFBundleIdentifier</key><string>jp.xenios.aion2mac</string>
<key>CFBundleExecutable</key><string>AION2Mac</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleIconFile</key><string>Aion.icns</string>
<key>CFBundleShortVersionString</key><string>$VERSION</string>
<key>CFBundleVersion</key><string>4</string>
<key>LSMinimumSystemVersion</key><string>26.0</string>
<key>LSApplicationCategoryType</key><string>public.app-category.games</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
IDENTITY="${AION2_SIGNING_IDENTITY:--}"
if [ "$IDENTITY" = - ]; then
  codesign --force --sign - "$APP"
  echo 'Preview signature: public downloads require macOS first-open approval.'
else
  case "$IDENTITY" in 'Developer ID Application:'*) ;; *) echo 'Use a Developer ID Application identity for public distribution.' >&2; exit 1;; esac
  codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP"
fi
printf 'Built %s\n' "$APP"
