# Troubleshooting

Find the symptom below. For instructions referring to a helper file, open Finder, choose **Go → Go to Folder…**, and paste `~/Library/Application Support/Aion2Mac/scripts`. Double-click the named file; it opens in Terminal. Custom installations use their own `scripts` folder.

## macOS blocks the app

The preview app is locally signed but **not Developer ID signed or notarized**. macOS may require you to approve it in System Settings → Privacy & Security → Open Anyway after attempting to open it. Do not disable Gatekeeper. The Terminal command in the README is an alternative.

## Setup asks for Apple’s toolkit

Download Apple’s Game Porting Toolkit and open its included **Evaluation environment** disk image. Setup detects a single mounted complete payload automatically. If it cannot find one, it explains what to download and links to Apple. With multiple mounted toolkits or an extracted folder, use Choose Toolkit. The project cannot include Apple’s libraries in its download.

## Rosetta is missing

Setup checks Rosetta by running an x86_64 system program. The native app offers Apple’s Rosetta installation window and waits for installation to finish. You accept Apple’s license yourself. The optional Terminal installer uses softwareupdate instead.

## Open Steam separately

To manage downloads or your library, double-click **steam.command** in the folder above. This opens the bottle’s Windows Steam; the native Mac Steam app is separate. During setup, use the app’s **Open Steam** or **Show Steam** button instead. Normal game launches keep Steam in the background. Steam remains required for this build's authentication.

## Steam downloads an update during setup

Steam’s official installer is a bootstrapper. Setup waits for the current 64-bit client to finish its initial update and verify its files before applying the display fix and offering Open Steam. Progress shows the actual update stage and download size. Steam can still perform later updates normally.

A bounded repair remains for older, incomplete setups whose first update replaces the UI helper. Only one repair can run at a time, and it does not run once a game manifest exists.

## Steam is black or sluggish

Run `scripts/prepare-steam.sh` from the installation directory after Steam updates finish, then launch normally. It preserves the genuine helper as `steamwebhelper_real.exe` and reinstalls the wrapper. The wrapper uses software CEF rendering, so the client may be slower than native Steam. Disable animated/library-heavy content in Steam settings if needed.

## Steam repeatedly starts helper processes

Use v0.1.2 or newer. Older setup scripts could mistake an existing wrapper for Steam's original helper during an upgrade. The current installer preserves and checks the genuine backup, and the wrapper refuses to launch another wrapper. If setup reports that both copies are wrappers, quit this bottle, repair Steam's client files, then rerun setup.

## Game closes

Report the visible stage and whether Steam reports the game stopped. The errno bridge fixes a reproduced native TLS fault, but other exit-code-1 failures have occurred. Do not infer an anti-cheat kernel requirement or a complete anti-cheat fix from those exits.

Compare with `AION2_MSYNC=0` at the same point. Do not switch Wine versions or mix graphics DLLs without revalidating the patched decoder and errno bridge.

## Cinematics are black or silent

Check `doctor.command` for the patched decoder. The launcher writes the Electra old-output-path settings. Pictures were confirmed in the reference game run; the AAC fallback has passed component tests but still needs in-game confirmation. Report whether only cinematic audio disappears or all audio remains off afterward.

## Audio stays on the previous output

Select the desired output in macOS Sound settings. The launcher forwards output changes into Wine's Windows default-device notifications. If audio remains on the old device, open `scripts/refresh-audio.command`; this does not restart the game. AirPods-to-speakers live switching was confirmed in the reference run, but not every device or game state is certified. The helper does not change microphone selection.

## Multiple Dock icons

The native setup app has a Dock entry during onboarding and exits when it starts the game. The game has its own Dock entry; Steam's UI helper may still create another. Silent Steam suppresses its main window, but does not remove that helper's Dock icon.

## Clicks are displaced

Use the included root-desktop launcher. Avoid `explorer /desktop=Aion2,1280x720`, stretching an outer desktop window, or live display-mode changes. The last two can resize the game without preserving the intended coordinate relationship. Use the desktop’s default resolution first, then an explicit `AION2_RESOLUTION` if needed.

## DLSS is missing

It remains an open compatibility item. Test with `AION2_DLSS=1`, then look in Aion’s graphics/upscaling menu. The flag deploys Apple’s builtin `nvngx` / `nvapi64`, enables MetalFX, and reports a consistent RTX 4080 compatibility identity. The compatibility API currently reports driver 561.09, which can trigger Aion’s driver-update notice. The exact optional driver-download dialog is automatically declined with No; no game binary is patched. Component NGX availability does not establish Aion menu availability. Use FSR if DLSS is absent; do not install a Windows GPU driver or random replacement DLLs into the bottle.

## HUD says Composited

That describes the macOS display presentation path. [Apple explains](https://developer.apple.com/videos/play/tech-talks/110339/) that compositing may add buffering/latency compared with Direct. It does not identify the Direct3D translator or establish whether upscaling is enabled. D3DMetal remains the configured renderer. Fullscreen alone does not guarantee direct presentation.

## Switching fullscreen freezes rendering

Restart in windowed mode: `AION2_WINDOWED=1 AION2_RESOLUTION=1920x1080` before the launcher command. Avoid repeatedly changing modes during a run. This is an unresolved Wine/D3DMetal presentation issue; a Composited label alone does not explain a freeze.

## Performance

Keep the normal quiet launch, compare MSync on/off in the same scene, and collect measurements in a separate instrumented developer setup. Record FPS, frame interval, GPU time, resolution, and graphics preset. First-run shader compilation, Steam software rendering, memory pressure, and Rosetta CPU work can each affect performance. The event benchmark is evidence about synchronization latency, not a promised FPS multiplier. No maximum-performance preset has yet been certified.

## Diagnostics and removal

`scripts/doctor.command` prints versions and file-presence checks without reading login data. Local logs are in the installation’s `logs` directory and can contain authentication tickets; share only reviewed, redacted excerpts. Never publish the prefix, Steam userdata, dumps, or raw launch arguments.

To remove only the Mac launcher, delete **AION 2.app**; the bottle and game files remain. To remove the whole installation, quit the game, its Windows Steam session, and the launcher, then remove `~/Library/Application Support/Aion2Mac`, the app, and any Desktop shortcut. **Removing that data folder deletes this bottle’s installed games and local settings.** Preserve any local saves you need first. Other Wine bottles are independent.

## Steam was closed during onboarding

The app tracks Steam’s live process and window state. Closing it stops the waiting indicator and offers Open Steam Again. A hidden client shows Show Steam. The Play button appears only when Steam marks the game fully installed and its executable exists.
