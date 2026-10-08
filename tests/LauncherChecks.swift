import AppKit
import SwiftUI

@main struct LauncherChecks {
    @MainActor static func main() throws {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["AION2_MAC_HOME"]!)
        precondition(root.path.hasPrefix("/tmp/aion2-launcher-check-"))
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let launcher = Launcher()
        precondition(!launcher.environmentReady && !launcher.gameReady)
        func file(_ relative: String, _ text: String = "") throws {
            let url = root.appendingPathComponent(relative)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        }
        try file(".aion2-mac")
        try file(".ready-v0.1.2") // Upgrading the app must recognize older installations.
        try file("scripts/start.command", "#!/bin/bash\n")
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: root.appendingPathComponent("scripts/start.command").path)
        try file("prefix/drive_c/Program Files (x86)/Steam/steam.exe")
        precondition(launcher.environmentReady)
        let apps = "prefix/drive_c/Program Files (x86)/Steam/steamapps/"
        try file(apps + "appmanifest_3393110.acf", "\"StateFlags\" \"4\"\n\"installdir\" \"AION2\"")
        precondition(!launcher.gameReady) // A manifest alone does not mean the game exists.
        try file(apps + "common/AION2/Aion2/Binaries/Win64/AION2.exe")
        precondition(launcher.gameReady)
        try file(apps + "appmanifest_3393110.acf", "\"StateFlags\" \"1026\"\n\"installdir\" \"AION2\"")
        precondition(!launcher.gameReady) // Download/update still in progress.
        try file(apps + "appmanifest_3393110.acf", "\"StateFlags\" \"4\"\n\"installdir\" \"../elsewhere\"")
        precondition(!launcher.gameReady)
        let boot = "prefix/drive_c/Program Files (x86)/Steam/logs/bootstrap_log.txt"
        try file(boot, "Downloading update (35,027 of 236,054 KB)...")
        precondition(launcher.steamBootstrapStatus() == "Downloading Steam’s update — 34 of 231 MB…")
        try file(boot, "Downloading update (35,027 of 236,054 KB)...\nInstalling update...")
        precondition(launcher.steamBootstrapStatus() == "Installing Steam’s update…")
        try file(boot, "Verification complete")
        precondition(launcher.steamBootstrapStatus() == "Opening Steam’s sign-in window…")
        launcher.applySteamState("ready")
        precondition(!launcher.steamStarting)
        launcher.applySteamState("closed")
        precondition(!launcher.steamStarting && !launcher.steamOpened)
        precondition(launcher.steamStatus.contains("closed"))
        launcher.applySteamState("background")
        precondition(!launcher.steamStarting && launcher.steamStatus.contains("background"))
        launcher.applySteamState("failed")
        precondition(!launcher.steamStarting && !launcher.steamOpened)
        print("PASS: fresh setup, prior-version detection, installed game, partial download, safe manifest paths, real Steam progress parsing, Steam close/background/failure states")
    }
}
