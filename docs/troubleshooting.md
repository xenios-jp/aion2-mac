# Troubleshooting

## Steam is black or sluggish

Run `scripts/prepare-steam.sh` from the installation directory after Steam updates finish, then launch normally. It preserves the genuine helper as `steamwebhelper_real.exe` and reinstalls the wrapper. The wrapper uses software CEF rendering, so the client may be slower than native Steam. Disable animated/library-heavy content in Steam settings if needed.

## Game closes

Report the visible stage and whether Steam reports the game stopped. The errno bridge fixes a reproduced native TLS fault, but other exit-code-1 failures have occurred. Do not infer an anti-cheat kernel requirement or a complete anti-cheat fix from those exits.

Compare with `AION2_MSYNC=0` at the same point. Do not switch Wine versions or mix graphics DLLs without revalidating the patched decoder and errno bridge.

## Cinematics are black or silent

Check `doctor.command` for the patched decoder. The launcher writes the Electra old-output-path settings. Pictures were confirmed in the reference game run; the AAC fallback has passed component tests but still needs in-game confirmation. Report whether only cinematic audio disappears or all audio remains off afterward.

## Clicks are displaced

Use the included root-desktop launcher. Avoid `explorer /desktop=Aion2,1280x720`, stretching an outer desktop window, or live display-mode changes. The last two can resize the game without preserving the intended coordinate relationship. Use the desktop’s default resolution first, then an explicit `AION2_RESOLUTION` if needed.

## DLSS is missing

It remains an open compatibility item. Test with `AION2_DLSS=1`, then look in Aion’s graphics/upscaling menu. The flag deploys Apple’s builtin `nvngx` / `nvapi64`, enables MetalFX, and reports a consistent RTX 4080 compatibility identity. The compatibility API currently reports driver 561.09, which can trigger Aion’s driver-update notice. Component NGX availability does not establish Aion menu availability. Use FSR if DLSS is absent; do not install a Windows GPU driver or random replacement DLLs into the bottle.

## HUD says Composited

That describes the macOS display presentation path. [Apple explains](https://developer.apple.com/videos/play/tech-talks/110339/) that compositing may add buffering/latency compared with Direct. It does not identify the Direct3D translator or establish whether upscaling is enabled. D3DMetal remains the configured renderer. Fullscreen alone does not guarantee direct presentation.

## Performance

Keep the normal quiet launch, compare MSync on/off in the same scene, and use `AION2_HUD=1`. Record FPS, frame interval, GPU time, resolution, and graphics preset. First-run shader compilation, Steam software rendering, memory pressure, and Rosetta CPU work can each affect performance. The event benchmark is evidence about synchronization latency, not a promised FPS multiplier. No maximum-performance preset has yet been certified.

## Diagnostics and removal

`scripts/doctor.command` prints versions and file-presence checks without reading login data. Local logs are in the installation’s `logs` directory and can contain authentication tickets; share only reviewed, redacted excerpts. Never publish the prefix, Steam userdata, dumps, or raw launch arguments.

To remove the setup, quit its game and Steam session, then remove the dedicated `Aion2Mac` installation directory. Other Wine bottles are independent.
