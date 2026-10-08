# Launch options

Double-click:

```text
~/Library/Application Support/Aion2Mac/scripts/start.command
```

Or use Terminal:

```bash
"$HOME/Library/Application Support/Aion2Mac/scripts/start.command"
```

The launcher uses the real Mac desktop and its reported resolution, rather than an emulated desktop with restricted mouse bounds. D3DMetal is the renderer; MSync is enabled. The normal launch keeps diagnostic tracing off.

| Option | Command prefix | Status |
| --- | --- | --- |
| Metal performance HUD | `AION2_HUD=1` | Available |
| HUD timing logs | `AION2_HUD=1 AION2_HUD_LOGGING=1` | For measurements |
| Explicit resolution | `AION2_RESOLUTION=2560x1080` | Windows desktop pixels |
| Disable MSync | `AION2_MSYNC=0` | Stability comparison |
| DLSS → MetalFX bridge | `AION2_DLSS=1` | Experimental; Aion menu selection unverified |

For example:

```bash
AION2_HUD=1 "$HOME/Library/Application Support/Aion2Mac/scripts/start.command"
```

The DLSS flag enables Apple’s NGX bridge and a consistent NVIDIA compatibility identity. It does not turn the Mac into an NVIDIA GPU, provide NVIDIA frame generation, or guarantee that Aion exposes DLSS. A graphics-driver update notice may appear; no Windows NVIDIA driver needs to be installed on macOS. The game’s FSR option is the current fallback.


Run `doctor.command` before opening an issue. Share its concise output and the visible symptom; do not upload your bottle or raw game logs. Logs may contain Steam launch authentication tickets.
