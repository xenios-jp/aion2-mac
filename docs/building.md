# Rebuilding the compatibility payload

The installer uses prebuilt fixes, so players do not need compilers. Developers can rebuild with Apple Command Line Tools, GNU Bison 3.8.2, and LLVM-MinGW 20260922 (UCRT, macOS universal):

```bash
export PATH="/path/to/bison-3.8.2/bin:$PATH"
export LLVM_MINGW_ROOT="/path/to/llvm-mingw-20260922-ucrt-macos-universal"
bash scripts/build-fixes.sh
```

The script downloads the pinned WineCX source, verifies its archive, applies the two included patches, builds only the PE `winegstreamer.dll`, and builds the errno bridge, Steam wrapper, display-size and driver-notice helpers, and audio-routing helpers. No Apple libraries are needed for the build. Apple libraries are needed for runtime tests.

The release includes `winecx-aion2-corresponding-source-v0.1.3.tar.gz`, containing the exact Wine source with the applied decoder patches. You may modify and replace the LGPL component. The Unix half stays unchanged and must match runtime-v4.7.3.

## Validation performed

- Actual Darwin wait-error reproducer under synthetic Windows GS: native errno and stack sentinel preserved, including an active exception list.
- H.264 MFT: NV12 enumeration, decoder-owned samples, `IMF2DBuffer` and `Lock2D` output. Game cinematic pictures confirmed separately.
- AAC packets: both raw and ordinary AAC, with and without initialization data; 35 nonzero PCM frames for each case.
- D3D12 / NGX: initialization, feature requirements, SuperSampling capability and NVAPI identity probes.
- 30,000 cross-thread auto-reset-event handoffs for MSync comparison.
- Fresh installer including Steam bootstrap, completed-install rerun, signed app build and wrapper-recursion regression checks. Existing-bottle public launcher starts Aion. Full fresh-account gameplay certification remains open.

Test sources for the project-owned errno reproducer are in `tests`. The release source package and repository patches document how the Wine component was produced. Proprietary game media used during development are not distributed.
