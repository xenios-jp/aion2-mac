# Launch options

Open **AION 2.app** from Applications. The first launch guides setup and Steam installation; subsequent launches start the installed game. Reopening it focuses a game already running in this bottle. To open Steam separately, use `scripts/steam.command` in the installation folder.

Or use Terminal:

```bash
"$HOME/Library/Application Support/Aion2Mac/scripts/start.command"
```

The launcher uses the real Mac desktop and its reported resolution, rather than an emulated desktop with restricted mouse bounds. D3DMetal is the renderer; MSync is enabled. The normal launch keeps diagnostic tracing off.

| Option | Command prefix | Status |
| --- | --- | --- |
| Windowed mode | `AION2_WINDOWED=1 AION2_RESOLUTION=1920x1080` | Useful for screenshots / mode-transition issues |
| Explicit resolution | `AION2_RESOLUTION=2560x1080` | Windows desktop pixels |
| Disable MSync | `AION2_MSYNC=0` | Stability comparison |
| DLSS → MetalFX bridge | `AION2_DLSS=1` | Experimental; Aion menu selection unverified |

For example:

```bash
AION2_WINDOWED=1 AION2_RESOLUTION=1920x1080 "$HOME/Library/Application Support/Aion2Mac/scripts/start.command"
```

GPU frame capture, Metal HUD, HUD logs and encoder instrumentation are disabled in this release.

Audio follows macOS output changes through a small background helper. Manual fallback: open `scripts/refresh-audio.command`. Live switching from AirPods to MacBook speakers was confirmed in the reference run; automatic notification forwarding was checked in both directions.

The DLSS flag enables Apple’s NGX bridge and a consistent NVIDIA compatibility identity. It does not turn the Mac into an NVIDIA GPU, provide NVIDIA frame generation, or guarantee that Aion exposes DLSS. The optional graphics-driver download notice is automatically declined; no Windows NVIDIA driver needs to be installed on macOS. The game’s FSR option is the current fallback.


Run `doctor.command` before opening an issue. Share its concise output and the visible symptom; do not upload your bottle or raw game logs. Logs may contain Steam launch authentication tickets.

## Where the bottle lives

A fresh setup stores its Windows environment at `~/Library/Application Support/Aion2Mac/prefix`, alongside its runtime and fixes. The `.app` is the Mac entry point. The downloaded app detects this project's completed installation even when the app version changes. Running setup again reuses a completed project installation; it does not search other Wine/CrossOver bottles or copy their accounts. Existing installations in other bottles need an explicit migration, which this preview does not automate.

Setup shows nine stages with the current action. Downloads report received bytes. Setup completes Steam’s first client updates before handing over. The Steam handoff shows a waiting indicator and elapsed time until its window opens, then tracks whether it is visible, hidden, or closed. A prefix-scoped observer marks the Steam window open once its visible client window exists. Repeated clicks cannot schedule competing first-update repairs.
