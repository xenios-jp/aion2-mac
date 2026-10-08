# AION 2 on Mac

Run the Windows Steam version of **AION 2** on Apple silicon with D3DMetal.

**Experimental preview.** Gameplay and cinematic pictures work on our test Mac. Fullscreen input, cinematic audio, DLSS, and stability still need in-game verification. [Current status →](docs/compatibility.md)

## Get started

You need **Apple silicon, macOS 26 or newer, Rosetta, and about 120 GB free**. Tested on macOS 27 with D3DMetal 4.0 beta 2.

1. **Get Apple’s toolkit.** Download the evaluation environment from [Apple](https://developer.apple.com/games/game-porting-toolkit/) and mount or extract it.
2. **Run setup.** Paste this into Terminal, then select the toolkit’s `lib` folder containing `external` and `wine`:

```bash
curl -fsSL https://raw.githubusercontent.com/xenios-jp/aion2-mac/v0.1.0/install.sh | bash
```

3. **Install and play.** Sign into Steam, install **AION 2** and its offered prerequisites, then double-click **AION 2.command** on your Desktop.

Setup creates a separate bottle, installs Steam and the compatibility fixes, and opens Steam. Apple’s libraries are supplied by you. If Rosetta is missing, setup prints the command to install it.

## Need help?

[Troubleshooting](docs/troubleshooting.md) · [Launch options / Metal HUD](docs/usage.md) · [Compatibility](docs/compatibility.md) · [Build from source](docs/building.md)

## Credits

Built on [WineCX](https://github.com/dappermint/winecx-gptk), [Wine](https://www.winehq.org/), Apple’s Game Porting Toolkit, [winevideo](https://github.com/Jfishin/winevideo/wiki/UE5-ElectraPlayer), and [notpop’s Steam wrapper](https://github.com/notpop/steam-on-m1-wine).

Project code is MIT licensed; Wine changes are LGPL-2.1-or-later. [Licenses and sources](THIRD_PARTY.md). No game files, Apple binaries, account data, or anti-cheat bypass are included.
