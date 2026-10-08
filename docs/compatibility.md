# Compatibility status

Status recorded on 2026-10-08. The reference machine is an Apple M5 Pro with 24 GB unified memory, macOS 27.0.1, and an external 5120×2160 display configured as a 2560×1080 logical desktop. The installed Steam game build was 25767555.

| Area | Evidence | Remaining work |
| --- | --- | --- |
| Steam interface | Software CEF wrapper rendered the Store/Library | Can remain slow; Steam updates can replace the wrapper |
| Game boot | User reached gameplay; multiple extended runs | Several unexplained exit-code-1 terminations remain; no universal stability claim |
| Native wait crash | Reproduced the actual Darwin wait error with a synthetic Wine TEB; bridge preserves the stack and native errno with empty and active exception lists | Revalidate the TEB offset before changing Wine versions |
| Cinematic pictures | Patched H.264 MFT decoded 18 actual game frames with `IMF2DBuffer`; user confirmed visible cinematics | Other codecs and every cinematic untested |
| Cinematic audio | Missing AAC metadata failed before the patch; raw and regular AAC tests now each decode 35 audible PCM frames, with and without metadata | Aion playback after deploying this patch has not been confirmed |
| Mouse clicks | Emulated-desktop clicks failed when the window moved right/down; root desktop restores physical monitor bounds | User confirmation of all fullscreen menu edges is pending |
| DLSS / MetalFX | D3D12 device, NGX initialization, feature requirements, and SuperSampling availability probes passed; Apple bridge loaded in Aion | Aion originally hid DLSS; corrected identity test has not received menu confirmation |
| MSync | 30,000 event handoffs: 28.89 µs with server sync, 12.32 µs with MSync; both passed | This is not a game-FPS result; sustained gameplay validation pending |
| Presentation | Metal HUD visible; reported Composited | Direct presentation and optimal frame pacing unverified |
| Fresh installer | Fresh bottle, Steam download/wrapper, desktop helper, H.264 2D samples (17 frames), and AAC without metadata (35 PCM frames) passed independently | A fresh-bottle full Aion download/gameplay run is still pending |

## Reproducible runtime

- WineCX runtime-v4.7.3 / Wine 11.17, source commit `e0aa380780b73e20fabcfe78fd42713b94929a53`.
- Upstream `Libraries.tar.gz` SHA-256: `a4b5d63493f80698cce5cad8e7212d9a51c8292037b00c478f4652636fcfd331`.
- Apple D3DMetal 4.0 beta 2, supplied separately by the user.
- Exact-runtime Wine PE decoder rebuilt with the patches in this repository. The original Unix GStreamer component remains unchanged.

The development game runs used an isolated copy of a previously configured prefix. The public installer starts from an empty prefix. The table deliberately distinguishes component tests and observed gameplay from a complete clean-install certification.

## Findings

### Native errno fault

Darwin reads its errno pointer from GS:8. With Wine’s Windows GS base active, that address instead contains `NT_TIB.StackBase`. An error in a native wait can therefore write into the Windows stack boundary. Wine stores the native Darwin TSD in `TEB.Instrumentation[0]` at offset `0x16b8` in the pinned runtime. The bridge recognizes Windows TIB geometry and recovers the native errno pointer. It changes no game or anti-cheat code.

### Electra video

The original macOS decoder excluded NV12 output and did not consistently provide decoder-owned `IMF2DBuffer` samples. The repository carries the relevant common/H.264 changes from winevideo’s Electra patches, rebased onto the exact runtime source. VP9-specific changes were not included because this source does not contain that decoder.

### AAC initialization

The added fallback supplies a minimal AAC-LC AudioSpecificConfig when the application provides none, using its declared sample rate and channel count. It clones the media type, preserves existing metadata, and validates the subtype. It cannot infer an arbitrary missing HE-AAC/SBR configuration; more evidence is needed if a cinematic uses a different audio profile.

### Desktop bounds

The old virtual-desktop launch constrained Wine’s coordinate space. A window moved outside those bounds could hover correctly yet misplace clicks. `/desktop=root` selects the actual Mac desktop and avoids changing the display mode to a smaller emulated size. Related upstream report: [Gcenx/macOS_Wine_builds #99](https://github.com/Gcenx/macOS_Wine_builds/issues/99).
