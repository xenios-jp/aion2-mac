#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$REPO/build"
SOURCE_SHA=32ba159ac9fe352f939eb3709d1784c531153f6dc834db432d6f862f5f85ace4
: "${LLVM_MINGW_ROOT:?Set LLVM_MINGW_ROOT to the LLVM-MinGW 20260922 macOS universal directory}"
export PATH="$LLVM_MINGW_ROOT/bin:$PATH"
command -v bison >/dev/null
mkdir -p "$BUILD/bin"
if [ ! -f "$BUILD/winecx.tar.gz" ]; then
  curl -fL --proto '=https' https://codeload.github.com/dappermint/winecx/tar.gz/e0aa380780b73e20fabcfe78fd42713b94929a53 -o "$BUILD/winecx.tar.gz"
fi
[ "$(shasum -a 256 "$BUILD/winecx.tar.gz" | awk '{print $1}')" = "$SOURCE_SHA" ] || { echo 'Source checksum mismatch' >&2; exit 1; }
if [ ! -d "$BUILD/source" ]; then
  mkdir "$BUILD/source"
  tar -xzf "$BUILD/winecx.tar.gz" --strip-components=1 -C "$BUILD/source"
  for patchfile in "$REPO/patches/"*.patch; do patch -d "$BUILD/source" -p1 < "$patchfile"; done
fi
mkdir -p "$BUILD/wine"
cd "$BUILD/wine"
"$BUILD/source/configure" --enable-archs=x86_64 --with-mingw=x86_64-w64-mingw32-clang \
  --without-x --without-wayland --without-gstreamer --enable-winegstreamer \
  --without-vulkan --without-alsa --without-pulse --without-dbus --without-oss \
  --without-capi --without-cups --without-opencl --without-pcap --without-sane \
  --without-usb --without-v4l2 --disable-tests
make -j8 dlls/winegstreamer/x86_64-windows/winegstreamer.dll
cp dlls/winegstreamer/x86_64-windows/winegstreamer.dll "$BUILD/bin/"
clang -arch x86_64 -O2 -Wall -Wextra -dynamiclib "$REPO/src/wine-native-errno.c" -o "$BUILD/bin/wine-native-errno.dylib"
x86_64-w64-mingw32-clang -municode -O2 -Wall -Wextra -static "$REPO/src/steamwebhelper-wrapper.c" -lshell32 -mwindows -o "$BUILD/bin/steamwebhelper-wrapper.exe"
x86_64-w64-mingw32-clang -O2 "$REPO/src/display-size.c" -o "$BUILD/bin/display-size.exe"
llvm-strip --strip-debug "$BUILD/bin/winegstreamer.dll" "$BUILD/bin/steamwebhelper-wrapper.exe"
printf 'Built compatibility files in %s/bin\n' "$BUILD"
