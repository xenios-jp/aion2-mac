# Performance and presentation

Reference: M5 Pro, 24 GB, macOS 27.0.1, GPTK 4.0 beta 2, WineCX runtime-v4.7.3, MSync enabled. Measurements below are **menu observations, not gameplay averages**.

| Scene | Render size | HUD FPS at capture | GPU time | Presentation |
| --- | --- | --- | --- | --- |
| Global server selection, fullscreen-sized window | 3200×1350 | 53.53 | 18.95 ms | Composited |
| Global server selection, windowed | 1920×1080 | 76.37 | 8.77 ms | Composited |
| Graphics settings, windowed | 1920×1080 | 147.59 | 1.36 ms | Composited |

A settings screen is much cheaper than the rendered world. The two resolutions also have different aspect ratios. These screenshots do not establish a controlled speedup or a gameplay FPS guarantee.

A stationary-camera gameplay sample at 3200×1350 averaged **28.85 FPS** over 60 seconds, with mean GPU time **34.62 ms**. After closing two background utilities, a second sample averaged **30.22 FPS**, with mean GPU time **33.01 ms**. This single pair does not prove causation or a repeatable improvement. The current evidence points toward GPU work as a leading bottleneck; no custom D3DMetal optimization is shipped.

## Reproducible measurements

HUD, capture and profiling are disabled in the distributed launcher. The screenshots below came from earlier diagnostic runs. For development profiling, use Apple's documented HUD configuration in a separate instrumented setup. Never upload a raw game log: it can contain login tickets.

Use the same character, scene, camera, graphics settings and output resolution. Warm shaders first, keep the game focused, then collect at least 60 seconds. Compare average FPS, p95/p99 frame interval, GPU time and visible defects. Change one setting per run. Separate loading, cinematics, menus and gameplay; exclude overlapping helper layers.

Metal 4 is the default on macOS 27. `D3DM_MAX_FPS` is the documented frame cap; it does not increase rendering performance.

## Why the HUD says Composited

[Apple's presentation guide](https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos) requires native macOS fullscreen, an opaque Metal layer, RGB content and Apple silicon for direct-to-display eligibility. Additional hardware/software conditions apply. A borderless Windows fullscreen window is not proof of native fullscreen. Windowed compositing is normal.

The reference setup still reports Composited. Switching back to windowed has also been reported to freeze rendering. Direct presentation and safe fullscreen transitions are unresolved; no renderer patch claiming to fix them is shipped.

## Optimization work

First establish whether a scene is GPU-, CPU- or presentation-limited. The [Metal HUD insights](https://developer.apple.com/documentation/xcode/gaining-performance-insights-with-metal-performance-hud) and a supported GPU trace can identify render-pass splits, copies, barriers and shader costs.

Apple's [TBDR guidance](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering) explains why avoiding unnecessary attachment loads/stores and keeping intermediate results in tile memory can help. These changes require knowledge of resource lifetimes and shader access. Blindly replacing storage modes, removing fences, or merging arbitrary passes can corrupt rendering.

We can modify the open Wine host/window code and test documented D3DMetal controls. A game-specific runtime shim must preserve graphics API semantics and pass correctness tests before shipping. Native tile shaders or a rewritten deferred renderer generally need engine source and a native port. D3DMetal is supplied as Apple's binary; this project does not redistribute a modified binary.

## Translation analysis — 2026-10-10

Local inspection of GPTK 4.0 beta 2 found several costs worth measuring. These are code paths present in the binary, not proof that they dominate AION's frames.

| Path | Observed behavior | Can we fix it easily? |
| --- | --- | --- |
| CPU command translation | The supplied D3DMetal framework is x86_64. Some indirect-command paths batch commands; a conditional fallback emits one command per record. Some state-reset paths allocate temporary vectors. | No measured hot path yet. We cannot turn Apple's binary into an ARM-native renderer through a launcher flag. |
| Resource barriers | The examined resolver tracks subresources and stage dependencies, and can use narrower barriers. Initialization sets its force-all-barriers flag to zero. | No evidence of an accidental force-all-stalls setting. Removing synchronization could corrupt rendering. The legacy resolver does not establish every Metal 4 path's behavior. |
| Texture copies | Compatible copies have a direct path. Certain plane/format cases allocate temporary storage and issue texture-to-buffer then buffer-to-texture transfers with synchronization. | A possible bandwidth cost, but first measure how often the game selects it. Those conversions cannot generally be discarded. |
| Geometry translation | Indirect drawing includes geometry-pipeline preparation and conditional extra dispatches. | Requires per-scene operation counts and timings before selecting a game setting or renderer change. |
| NGX feature detection | Missing NVAPI builtin aliases prevented correct initialization. Repairing the aliases made DLSS visible in the game. | Fixed in working source. This unlocks MetalFX upscaling; a gameplay FPS improvement has not yet been measured. |

MSync is already enabled and GPU capture/debug information remain disabled. Steam's software CEF is a separate CPU/memory cost; its presence does not mean the game's D3D12 rendering is software-rendered. Keep shader caches warm. Start with the game's DLSS Quality/Balanced/Performance options, then compare the same gameplay scene; presets trade internal resolution for image quality.

The latest historical Metal counter collection identifies the game's process but contains no per-layer frame records. It cannot establish current frame time, encoder timing, or a new performance gain. No binary patch is shipped based solely on static findings.

### Frame generation and HUD controls

The Apple bridge passed a standalone frame-generation evaluation and constant-image GPU readback. That test does not validate motion, HUD composition, frame pacing, or AION's swap chain. The shipped Streamline plugin separately checks hardware scheduling and a NVIDIA driver-specific fallback. The fallback entry point is absent from the tested Apple NVAPI implementation. Forcing a console variable is therefore not a verified frame-generation solution.

[NVIDIA's integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md) requires the application to enable interpolation, supply its inputs, and use compatible presentation. Loading the plugin alone is insufficient. AMD frame interpolation remains disabled because of the earlier presentation crash.

Working-source launch scripts enable the MetalFX bridge by default; `AION2_DLSS=0` opts out. `AION2_HUD=1` enables the HUD on the next script launch; capture remains off. These changes are not included in release 0.1.4. On macOS 27, the supported live controls are:

```sh
metalperftrace setup --enable hud --pid GAME_PID
metalperftrace setup --enable per-frame-metrics --pid GAME_PID
metalperftrace setup --enable shader-compiler-metrics --pid GAME_PID
```

Use the actual current Wine game process ID, not a previous run's PID. Disable measurement features after testing with the matching `--disable` command. These controls enable overlays/counters, not GPU frame capture. The HUD does not force DLSS or frame generation. Its MetalFX jitter visualizations are debugging tools, not general quality presets.

## Documentation reviewed

The local GPTK 4.0 beta 2 evaluation README covers setup, supported environments, NGX/MetalFX installation, documented variables, logging, GPU capture and troubleshooting. Its sample covers native window/layer setup and renderer integration. These are distinct from running an unchanged Windows game through translation.

[Current HUD configuration reference](https://developer.apple.com/documentation/xcode/customizing-metal-performance-hud) · [GPTK overview](https://developer.apple.com/games/game-porting-toolkit/)
