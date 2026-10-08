# AION 2 on Mac

Run the Windows Steam version of **AION 2** on Apple silicon with D3DMetal.

**Experimental preview.** Gameplay and cinematic pictures work on our test Mac. Fullscreen input, cinematic audio, DLSS, and stability still need in-game verification. [Current status →](docs/compatibility.md)

## Get started

You need **Apple silicon, macOS 26 or newer, Rosetta, and about 120 GB free**. Tested on macOS 27 with D3DMetal 4.0 beta 2.

1. **Get Apple’s toolkit.** Download it from [Apple](https://developer.apple.com/games/game-porting-toolkit/) and mount the included **Evaluation environment** disk image.
2. **Open [AION 2.app](https://github.com/xenios-jp/aion2-mac/releases/download/v0.1.3/AION-2-Mac-v0.1.3.zip).** Unzip and open it, then choose the evaluation environment’s `redist/lib` folder. Setup opens Terminal for progress and any license prompts. [First-open help](docs/troubleshooting.md#macos-blocks-the-app).
3. **Install and play.** Sign into Steam and install **AION 2** and its offered prerequisites. Afterwards, **AION 2.app** starts the game with Steam in the background.

Prefer Terminal? Use the same setup:

```bash
curl -fsSL https://raw.githubusercontent.com/xenios-jp/aion2-mac/v0.1.3/install.sh | bash
```

Setup creates a separate bottle, installs Steam and the compatibility fixes, and opens Steam. Apple’s libraries are supplied by you. If Rosetta is missing, setup offers to install it and lets you accept Apple’s license.

## Need help?

[Troubleshooting](docs/troubleshooting.md) · [Screenshots](docs/screenshots.md) · [Performance](docs/performance.md) · [Options](docs/usage.md) · [Compatibility](docs/compatibility.md)

## Credits

Built on [WineCX](https://github.com/dappermint/winecx-gptk), [Wine](https://www.winehq.org/), Apple’s Game Porting Toolkit, [winevideo](https://github.com/Jfishin/winevideo/wiki/UE5-ElectraPlayer), and [notpop’s Steam wrapper](https://github.com/notpop/steam-on-m1-wine).

Project code is MIT licensed; Wine changes are LGPL-2.1-or-later. [Licenses and sources](THIRD_PARTY.md). No game files, Apple binaries, account data, or anti-cheat bypass are included.
