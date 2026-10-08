#!/bin/bash
# Requires the user's Developer ID identity and locally stored notary credentials.
set -euo pipefail
APP="${1:?Pass the signed AION 2.app path}"
OUT="${2:?Pass the output ZIP path}"
: "${AION2_NOTARY_PROFILE:?Set the name of your notarytool keychain profile}"
[ -d "$APP" ] || { echo 'App not found' >&2; exit 1; }
codesign --verify --deep --strict "$APP"
codesign -dv --verbose=4 "$APP" 2>&1 | grep -q 'Authority=Developer ID Application:' || {
  echo 'Build with AION2_SIGNING_IDENTITY before notarizing.' >&2; exit 1;
}
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aion2-notarize.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
ditto -c -k --norsrc --noextattr --keepParent "$APP" "$STAGE/submit.zip"
xcrun notarytool submit "$STAGE/submit.zip" --keychain-profile "$AION2_NOTARY_PROFILE" --wait
# stapler fails if Apple has not accepted the app; never package a failed submission.
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl --assess --type execute --verbose=2 "$APP"
ditto -c -k --norsrc --noextattr --keepParent "$APP" "$OUT"
printf 'Notarized release: %s\n' "$OUT"
