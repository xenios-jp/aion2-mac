import AppKit
import SwiftUI

@MainActor final class Launcher: ObservableObject {
    enum Screen { case welcome, installing, steam, failure }
    @Published var screen: Screen = .welcome
    @Published var toolkit: URL?
    @Published var rosetta = false
    @Published var status = "Checking setup requirements…"
    @Published var step = 1
    @Published var setupElapsed = 0
    private var setupStarted: Date?
    private var steamBootstrap = false
    @Published var error = ""
    @Published var steamOpened = false
    @Published var steamStarting = false
    @Published var steamStatus = ""
    @Published var steamElapsed = 0
    private var steamStarted: Date?
    var steamStartedOnce: Bool { steamStarted != nil }
    let root: URL
    let preview = CommandLine.arguments.contains("--setup-preview")
    private let fm = FileManager.default
    private var timer: Timer?
    private var child: Process?
    private var setupLog: URL?
    init() {
        root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["AION2_MAC_HOME"] ??
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Aion2Mac").path)
    }
    func begin() {
        guard timer == nil else { return }
        if !preview && environmentReady {
            if gameReady { launchGame(); return }
            screen = .steam
        }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }
    var environmentReady: Bool {
        guard fm.fileExists(atPath: root.appendingPathComponent(".aion2-mac").path),
              fm.isExecutableFile(atPath: root.appendingPathComponent("scripts/start.command").path),
              fm.fileExists(atPath: root.appendingPathComponent("prefix/drive_c/Program Files (x86)/Steam/steam.exe").path)
        else { return false }
        return ((try? fm.contentsOfDirectory(atPath: root.path)) ?? []).contains { $0.hasPrefix(".ready-v") }
    }
    var gameReady: Bool {
        let apps = root.appendingPathComponent("prefix/drive_c/Program Files (x86)/Steam/steamapps")
        guard let manifest = try? String(contentsOf: apps.appendingPathComponent("appmanifest_3393110.acf"), encoding: .utf8)
        else { return false }
        func field(_ name: String) -> String? {
            let regex = try! NSRegularExpression(pattern: "\"" + name + "\"\\s*\"([^\"]+)\"")
            guard let match = regex.firstMatch(in: manifest, range: NSRange(manifest.startIndex..., in: manifest)),
                  let range = Range(match.range(at: 1), in: manifest) else { return nil }
            return String(manifest[range])
        }
        guard field("StateFlags") == "4", let directory = field("installdir"),
              !directory.contains("/"), !directory.contains("\\"), directory != ".." else { return false }
        return fm.fileExists(atPath: apps.appendingPathComponent("common/\(directory)/Aion2/Binaries/Win64/AION2.exe").path)
    }
    private func payload(_ folder: URL) -> URL? {
        for candidate in [folder, folder.appendingPathComponent("redist/lib"), folder.appendingPathComponent("lib")] {
            let files = ["external/libd3dshared.dylib", "external/D3DMetal.framework/Versions/A/D3DMetal",
                         "wine/x86_64-windows/d3d12.dll", "wine/x86_64-windows/nvngx-on-metalfx.dll"]
            if files.allSatisfy({ fm.fileExists(atPath: candidate.appendingPathComponent($0).path) }) { return candidate }
        }
        return nil
    }
    func refresh() {
        if screen == .installing {
            if let started = setupStarted { setupElapsed = Int(Date().timeIntervalSince(started)) }
            if steamBootstrap, let update = steamBootstrapStatus() {
                status = update == "Opening Steam’s sign-in window…" ? "Finishing Steam’s initial update…" : update
            }
            return
        }
        guard screen == .welcome || screen == .steam else { return }
        if screen == .welcome {
            if toolkit == nil && !(preview && CommandLine.arguments.contains("--missing-toolkit")) {
                let volumes = (try? fm.contentsOfDirectory(at: URL(fileURLWithPath: "/Volumes"), includingPropertiesForKeys: nil)) ?? []
                let found = volumes.compactMap { payload($0) }
                if found.count == 1 { toolkit = found[0] }
            }
            if !rosetta {
                let check = Process()
                check.executableURL = URL(fileURLWithPath: "/usr/bin/arch")
                check.arguments = ["-x86_64", "/usr/bin/true"]
                check.standardError = FileHandle.nullDevice
                check.standardOutput = FileHandle.nullDevice
                if (try? check.run()) != nil { check.waitUntilExit(); rosetta = check.terminationStatus == 0 }
            }
        }
        if screen == .steam, let started = steamStarted {
            steamElapsed = Int(Date().timeIntervalSince(started))
            if let text = try? String(contentsOf: root.appendingPathComponent("logs/steam-ui-state"), encoding: .utf8),
               let state = text.components(separatedBy: .newlines).last(where: { !$0.isEmpty }) {
                let modified = (try? fm.attributesOfItem(atPath: root.appendingPathComponent("logs/steam-ui-state").path)[.modificationDate]) as? Date
                if ["closed", "failed", "timeout"].contains(state), let modified, modified < started {
                    steamStatus = "Starting Steam…"
                } else { applySteamState(state) }
            } else {
                steamStatus = steamBootstrapStatus() ?? "Starting Steam. Its first update may take several minutes."
            }
        }
        objectWillChange.send()
    }
    func downloadToolkit() { NSWorkspace.shared.open(URL(string: "https://developer.apple.com/games/game-porting-toolkit/")!) }
    func chooseToolkit() {
        let panel = NSOpenPanel()
        panel.title = "Choose Apple’s evaluation environment"
        panel.message = "Select the mounted Evaluation environment or the folder you extracted from it."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: "/Volumes")
        panel.prompt = "Use Toolkit"
        panel.begin { response in
            Task { @MainActor in
                guard response == .OK, let selected = panel.url else { return }
                if let resolved = self.payload(selected) { self.toolkit = resolved; self.error = "" }
                else { self.error = "Open Apple’s included Evaluation environment disk image and choose that volume." }
            }
        }
    }
    func installRosetta() {
        NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Library/CoreServices/Rosetta 2 Updater.app"))
    }
    func setup() {
        guard !preview, let toolkit, rosetta else { return }
        screen = .installing
        setupStarted = Date()
        setupElapsed = 0
        status = "Checking the files needed for setup…"
        do {
            let logURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("aion2-onboarding-\(UUID().uuidString).log")
            fm.createFile(atPath: logURL.path, contents: nil)
            setupLog = logURL
            let log = try FileHandle(forWritingTo: logURL)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = [Bundle.main.resourceURL!.appendingPathComponent("install.sh").path,
                "--root", root.path, "--gptk", toolkit.path, "--no-launch", "--no-app"]
            process.standardInput = FileHandle.nullDevice
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            // Drain on a worker so installer output cannot block its subprocess.
            process.terminationHandler = { process in
                Task { @MainActor in
                    self.child = nil
                    if process.terminationStatus == 0 && self.environmentReady { self.screen = .steam }
                    else { self.fail("Setup couldn’t finish. Your progress is saved. Try again or open the setup log for details.") }
                }
            }
            child = process
            try process.run()
            DispatchQueue.global().async {
                var pending = ""
                while true {
                    let data = pipe.fileHandleForReading.availableData
                    if data.isEmpty { break }
                    try? log.write(contentsOf: data)
                    pending += String(decoding: data, as: UTF8.self)
                    while let newline = pending.firstIndex(of: "\n") {
                        let line = String(pending[..<newline])
                        pending.removeSubrange(...newline)
                        if line.hasPrefix("AION2_STEAM_BOOTSTRAP:") {
                            let active = line.hasSuffix("1")
                            Task { @MainActor in self.steamBootstrap = active }
                        }
                        if line.hasPrefix("AION2_STEP:"), let number = Int(line.dropFirst("AION2_STEP:".count)) {
                            Task { @MainActor in self.step = number }
                        }
                        if line.hasPrefix("AION2_PROGRESS:") {
                            let message = String(line.dropFirst("AION2_PROGRESS:".count))
                            Task { @MainActor in self.status = message }
                        }
                    }
                    if pending.count > 8192 { pending = "" }
                }
                try? log.close()
            }
        } catch { child = nil; fail("Setup couldn’t start. Please try again.") }
    }
    func openSteam() {
        guard !preview else { return }
        if runDetached("steam.command", arguments: ["steam://install/3393110"]) {
            steamOpened = true
            steamStarting = true
            steamStarted = Date()
            steamElapsed = 0
            steamStatus = "Starting Steam. Its first update may take several minutes."
        }
    }
    func applySteamState(_ state: String) {
        switch state {
        case "ready":
            steamOpened = true
            steamStarting = false
            steamStatus = "Steam’s sign-in or library window is open."
        case "background":
            steamOpened = true
            steamStarting = false
            steamStatus = "Steam is running in the background. Click Show Steam to return to it."
        case "closed":
            steamStarting = false
            steamOpened = false
            steamStatus = "Steam was closed. Open it again to continue installing AION 2."
        case "failed", "timeout":
            steamStarting = false
            steamOpened = false
            steamStatus = "Steam couldn’t open. Try again, or check the setup log."
        default:
            steamStarting = true
            steamStatus = steamBootstrapStatus() ?? "Starting Steam. Its first update may take several minutes."
        }
    }
    func steamBootstrapStatus() -> String? {
        let url = root.appendingPathComponent("prefix/drive_c/Program Files (x86)/Steam/logs/bootstrap_log.txt")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        for line in text.components(separatedBy: .newlines).reversed() {
            if let range = line.range(of: "Downloading update (") {
                let counts = String(line[range.upperBound...]).components(separatedBy: " KB")[0].components(separatedBy: " of ")
                if counts.count == 2, let done = Double(counts[0].replacingOccurrences(of: ",", with: "")),
                    let total = Double(counts[1].replacingOccurrences(of: ",", with: "")) {
                    return String(format: "Downloading Steam’s update — %.0f of %.0f MB…", done / 1024, total / 1024)
                }
            }
            if line.contains("Extracting package") { return "Unpacking Steam’s update…" }
            if line.contains("Installing update") { return "Installing Steam’s update…" }
            if line.contains("Verification complete") { return "Opening Steam’s sign-in window…" }
            if line.contains("Verifying installation") { return "Checking Steam’s installation…" }
            if line.contains("Downloading update") { return "Downloading Steam’s first client update…" }
            if line.contains("Update complete") { return "Restarting Steam after its update…" }
        }
        return nil
    }
    func launchGame() {
        guard !preview else { return }
        if runDetached("start.command", arguments: []) { NSApp.terminate(nil) }
    }
    private func prepareLog(_ name: String) throws -> URL {
        try fm.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let url = root.appendingPathComponent("logs/\(name)")
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
        return url
    }
    private func runDetached(_ script: String, arguments: [String]) -> Bool {
        do {
            let output = try FileHandle(forWritingTo: prepareLog("launcher.log"))
            try output.seekToEnd()
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/nohup")
            process.arguments = ["/bin/bash", root.appendingPathComponent("scripts/\(script)").path] + arguments
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = output
            process.standardError = output
            try process.run()
            try output.close()
            return true
        } catch { fail("AION 2 couldn’t open. Try again or open the log for details."); return false }
    }
    func openLog() {
        if let setupLog { NSWorkspace.shared.open(setupLog); return }
        let name = fm.fileExists(atPath: root.appendingPathComponent("logs/onboarding.log").path) ? "onboarding.log" : "launcher.log"
        NSWorkspace.shared.open(root.appendingPathComponent("logs/\(name)"))
    }
    func retry() { error = ""; screen = environmentReady ? .steam : .welcome; refresh() }
    private func fail(_ message: String) { error = message; screen = .failure }
}

struct SetupView: View {
    @ObservedObject var launcher: Launcher
    var body: some View {
        VStack(spacing: 22) {
            if let url = Bundle.main.url(forResource: "Aion", withExtension: "png"), let icon = NSImage(contentsOf: url) {
                Image(nsImage: icon).resizable().frame(width: 88, height: 88)
            }
            VStack(spacing: 8) {
                Text(title).font(.system(size: 26, weight: .semibold))
                Text(subtitle).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            switch launcher.screen {
            case .welcome:
                VStack(alignment: .leading, spacing: 18) {
                    requirement("Apple’s Game Porting Toolkit", ready: launcher.toolkit != nil)
                    if launcher.toolkit == nil {
                        Text("Download the toolkit from Apple, then open the Evaluation environment disk image inside it. We’ll detect it automatically.")
                            .foregroundStyle(.secondary).font(.callout)
                        HStack {
                            Button("Download from Apple…", action: launcher.downloadToolkit)
                            Button("Choose Toolkit…", action: launcher.chooseToolkit)
                        }
                    }
                    if !launcher.rosetta {
                        Divider()
                        requirement("Rosetta 2", ready: false)
                        Button("Install Rosetta…", action: launcher.installRosetta)
                    }
                }.padding(20).background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
                if !launcher.error.isEmpty { Text(launcher.error).foregroundStyle(.red).font(.callout) }
                Button("Set Up AION 2", action: launcher.setup).buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(launcher.toolkit == nil || !launcher.rosetta || launcher.preview)
                Text("Requires Apple silicon, macOS 26 or later, and about 120 GB free.")
                    .font(.caption).foregroundStyle(.secondary)
            case .installing:
                ProgressView().controlSize(.large)
                Text(launcher.status).foregroundStyle(.secondary).multilineTextAlignment(.center)
                Text("Step \(launcher.step) of 9 · \(launcher.setupElapsed / 60)m \(launcher.setupElapsed % 60)s elapsed").font(.caption).foregroundStyle(.secondary)
            case .steam:
                VStack(alignment: .leading, spacing: 12) {
                    Label("Sign in to your Steam account", systemImage: "person.crop.circle")
                    Label("Install AION 2 and its offered prerequisites", systemImage: "arrow.down.circle")
                    Label("Return here to play", systemImage: "play.circle")
                }.padding(20).background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
                if launcher.gameReady {
                    Button("Play AION 2", action: launcher.launchGame).buttonStyle(.borderedProminent).controlSize(.large)
                } else {
                    if !launcher.steamStatus.isEmpty {
                        HStack(spacing: 10) {
                            if launcher.steamStarting { ProgressView().controlSize(.small) }
                            Text(launcher.steamStatus).font(.callout).foregroundStyle(.secondary)
                        }
                        if launcher.steamStarting {
                            Text("Waiting for Steam · \(launcher.steamElapsed / 60)m \(launcher.steamElapsed % 60)s elapsed").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Button(launcher.steamStarting ? (launcher.steamElapsed < 180 ? "Starting Steam…" : "Try Opening Steam Again") : launcher.steamOpened ? "Show Steam" : launcher.steamStartedOnce ? "Open Steam Again" : "Open Steam", action: launcher.openSteam)
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(launcher.steamStarting && launcher.steamElapsed < 180)
                    Text("The Play button appears when installation finishes.").font(.caption).foregroundStyle(.secondary)
                }
            case .failure:
                HStack {
                    Button("Open Log…", action: launcher.openLog)
                    Button("Try Again", action: launcher.retry).buttonStyle(.borderedProminent)
                }
            }
        }.padding(36).frame(width: 500).onAppear { launcher.begin() }
    }
    private var title: String {
        switch launcher.screen {
        case .welcome: return "AION 2 on your Mac"
        case .installing: return "Setting up AION 2"
        case .steam: return launcher.gameReady ? "You’re ready to play" : "One last step in Steam"
        case .failure: return "Let’s try that again"
        }
    }
    private var subtitle: String {
        switch launcher.screen {
        case .welcome: return "We’ll prepare a separate environment for the game."
        case .installing: return "This may take a few minutes. Keep this window open."
        case .steam: return "Your Mac setup is complete."
        case .failure: return launcher.error
        }
    }
    private func requirement(_ text: String, ready: Bool) -> some View {
        HStack {
            Image(systemName: ready ? "checkmark.circle.fill" : "arrow.down.circle")
                .foregroundStyle(ready ? Color.green : Color.secondary)
            Text(text).fontWeight(.medium)
            Spacer()
            if ready { Text("Ready").font(.callout).foregroundStyle(.secondary) }
        }
    }
}

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    weak var model: Launcher?
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if model?.screen == .installing {
            let alert = NSAlert()
            alert.messageText = "Setup is still running"
            alert.informativeText = "Please let setup finish before closing AION 2."
            alert.addButton(withTitle: "Continue Setup")
            alert.runModal()
            return .terminateCancel
        }
        return .terminateNow
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if model?.screen == .installing { return false }
        return true
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        sender.windows.first?.makeKeyAndOrderFront(nil)
        return true
    }
}
#if !LAUNCHER_TESTS
@main struct AionApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var launcher = Launcher()
    var body: some Scene {
        Window("AION 2", id: "setup") {
            SetupView(launcher: launcher).onAppear { delegate.model = launcher; NSApp.windows.first?.delegate = delegate }
        }.windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .help) {
                Button("AION 2 Help") { NSWorkspace.shared.open(URL(string: "https://github.com/xenios-jp/aion2-mac#need-help")!) }
            }
        }
    }
}

#endif
