# Releases

## 0.1.3 — 2026-10-08

Automatically recover once if Steam's first self-update replaces the rendering wrapper. Restricted to fresh setup without an installed Aion game. Verified against the helper-replacement failure in a disposable Steam bottle; sign-in window renders afterward.

## 0.1.2 — 2026-10-08

Prevent Steam helper recursion when upgrading a different wrapper. Preserve the genuine CEF backup, validate it before launch, and fail safely if both copies are wrappers. Includes regression checks for wrapper upgrades and recursive delegation.

## 0.1.1 — 2026-10-08

App wrapper; Rosetta installation prompt; silent Steam game launch and separate library launcher; narrowly matched automatic No for the optional Windows driver download; corrected hex adapter IDs; HUD/capture disabled; live audio output refresh and automatic forwarding; background app launcher; windowed option, screenshots and performance notes. Direct presentation and Aion DLSS menu selection remain experimental.

## 0.1.0 — 2026-10-08

Initial experimental preview: pinned-runtime installer, isolated Steam bottle, software CEF wrapper, errno bridge, Electra video / AAC patches, root-desktop launcher, MSync and optional Metal HUD. Includes troubleshooting, compatibility evidence, source and build instructions. DLSS menu availability and comprehensive gameplay stability remain open.
