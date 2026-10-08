# Third-party software

| Component | Source | Distribution |
| --- | --- | --- |
| WineCX runtime-v4.7.3 | dappermint/winecx-gptk | Downloaded from its upstream release, with pinned SHA-256 |
| Rebuilt Wine winegstreamer | dappermint/winecx commit e0aa380780b73e20fabcfe78fd42713b94929a53 | LGPL-2.1-or-later; exact patched source included in release |
| Electra video changes | Jfishin/winevideo patches 0005 and common/H.264 parts of 0007 | Wine license; rebased changes in patches/video_decoder.patch |
| Missing AAC initialization fallback | Project change following the public missing-ASC approach documented in winevideo 0006 | LGPL-2.1-or-later as part of Wine; patches/aac_decoder.patch |
| Steam CEF wrapper | notpop/steam-on-m1-wine | MIT; source and original license included |
| Apple D3DMetal / NGX | Apple Game Porting Toolkit | Supplied locally by users; not redistributed |
| AION 2 / Steam | NCSoft / Valve | Downloaded through their own services; not redistributed |

The project’s scripts, display-size helper and original errno bridge use the root MIT license. Wine license text and third-party notices are in `licenses`. The pinned runtime retains its own upstream licenses. This repository is unaffiliated with NCSoft, Valve, Apple, NVIDIA, and CodeWeavers.
