# AION 2 on Mac

Run the Windows Steam version of **AION 2** on Apple silicon with D3DMetal.

**Experimental preview.** Gameplay and cinematic pictures work on our test Mac. Fullscreen input, cinematic audio, DLSS, and stability still need in-game verification. [Current status →](docs/compatibility.md)

## Get started

You need **Apple silicon, macOS 26 or newer, Rosetta, and about 120 GB free**. Tested on macOS 27 with D3DMetal 4.0 beta 2.

**First-open limitation:** this preview is not notarized. If macOS blocks it, click **Done**, then use **System Settings → Privacy & Security → Open Anyway** for AION 2. Proper Developer ID signing and notarization are pending.

1. **Open [AION 2.app](https://github.com/xenios-jp/aion2-mac/releases/download/v0.1.4/AION-2-Mac-v0.1.4.zip).** Unzip it and drag the app into Applications. [First-open help](docs/troubleshooting.md#macos-blocks-the-app).
2. **Follow the setup window.** It links to Apple’s toolkit, detects the mounted **Evaluation environment**, offers Rosetta if needed, and prepares the game environment with progress inside the app.
3. **Install and play.** The app opens Steam for sign-in and **AION 2** installation. Return to the setup window and click **Play AION 2**. Future launches go straight to the game with Steam in the background.

Prefer Terminal? Use the same setup:

```bash
curl -fsSL https://raw.githubusercontent.com/xenios-jp/aion2-mac/v0.1.4/install.sh | bash
```

The app uses native macOS controls and requires no Terminal. Setup creates a separate bottle and installs Steam and the compatibility fixes. Apple’s toolkit download and Steam sign-in are handled through their official interfaces; their libraries and account data are not bundled.

## Need help?

[Troubleshooting](docs/troubleshooting.md) · [Screenshots](docs/screenshots.md) · [Performance](docs/performance.md) · [Options](docs/usage.md) · [Compatibility](docs/compatibility.md)

## Credits

Built on [WineCX](https://github.com/dappermint/winecx-gptk), [Wine](https://www.winehq.org/), Apple’s Game Porting Toolkit, [winevideo](https://github.com/Jfishin/winevideo/wiki/UE5-ElectraPlayer), and [notpop’s Steam wrapper](https://github.com/notpop/steam-on-m1-wine).

Project code is MIT licensed; Wine changes are LGPL-2.1-or-later. [Licenses and sources](THIRD_PARTY.md). No game files, Apple binaries, account data, or anti-cheat bypass are included.
