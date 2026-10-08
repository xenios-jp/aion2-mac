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

## Documentation reviewed

The local GPTK 4.0 beta 2 evaluation README covers setup, supported environments, NGX/MetalFX installation, documented variables, logging, GPU capture and troubleshooting. Its sample covers native window/layer setup and renderer integration. These are distinct from running an unchanged Windows game through translation.

[Current HUD configuration reference](https://developer.apple.com/documentation/xcode/customizing-metal-performance-hud) · [GPTK overview](https://developer.apple.com/games/game-porting-toolkit/)
