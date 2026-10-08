# Public Mac distribution

The current preview is ad-hoc signed. Browser downloads trigger Gatekeeper; a locally built app opening successfully does not validate the downloaded first-open experience. Users must approve this preview in System Settings → Privacy & Security → Open Anyway. Do not remove quarantine attributes or disable Gatekeeper in setup.

For a standard public Mac download, the publisher needs an Apple Developer Program membership, a Developer ID Application certificate with its private key, and a notarization credential profile stored locally in Keychain. Do not put certificates, private keys, passwords, or API keys in this repository.

Build with the full certificate identity shown by `security find-identity -v -p codesigning`:

```bash
AION2_SIGNING_IDENTITY='Developer ID Application: YOUR NAME (TEAMID)' bash scripts/build-app.sh /path/to/output
```

Configure a Keychain profile using Apple's `notarytool store-credentials`, then notarize and staple the app:

```bash
AION2_NOTARY_PROFILE=your-profile bash scripts/notarize-app.sh '/path/to/output/AION 2.app' /path/to/AION-2-Mac.zip
```

The script verifies the Developer ID signature, waits for Apple's submission result, staples the accepted ticket, and checks Gatekeeper before packaging. It cannot complete without the publisher's signing credentials. Recalculate release checksums after notarizing. Test the final ZIP downloaded through a browser on a Mac that has not already approved the app.

Apple references: [Developer ID](https://developer.apple.com/developer-id/), [notarizing macOS software](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), [opening downloaded apps](https://support.apple.com/102445).
