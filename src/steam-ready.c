#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>
#include <stdio.h>
#include <tlhelp32.h>
#include <string.h>
static BOOL ready;
static BOOL CALLBACK check_window(HWND window, LPARAM unused)
{
    DWORD pid;
    wchar_t path[MAX_PATH], title[256];
    DWORD length = MAX_PATH;
    RECT bounds;
    HANDLE process;
    (void)unused;
    if (!IsWindowVisible(window) || !GetWindowTextW(window, title, 256)) return TRUE;
    GetWindowRect(window, &bounds);
    if (bounds.right - bounds.left < 200 || bounds.bottom - bounds.top < 100) return TRUE;
    GetWindowThreadProcessId(window, &pid);
    process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!process) return TRUE;
    if (QueryFullProcessImageNameW(process, 0, path, &length)) {
        const wchar_t *name = wcsrchr(path, L'\\');
        name = name ? name + 1 : path;
#ifdef STEAM_READY_DEBUG
        wchar_t klass[128];
        GetClassNameW(window, klass, 128);
        wprintf(L"window process=%ls class=%ls size=%ldx%ld signIn=%d\n", name, klass, bounds.right-bounds.left, bounds.bottom-bounds.top, !wcscmp(title,L"Sign in to Steam"));
#endif
        if (!_wcsicmp(name, L"steamwebhelper_real.exe") || !_wcsicmp(name, L"steamwebhelper.exe")) ready = TRUE;
    }
    CloseHandle(process);
    return !ready;
}
static BOOL process_running(const wchar_t *wanted)
{
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W entry = {0};
    BOOL found = FALSE;
    if (snapshot == INVALID_HANDLE_VALUE) return FALSE;
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snapshot, &entry)) do {
        if (!_wcsicmp(entry.szExeFile, wanted)) { found = TRUE; break; }
    } while (Process32NextW(snapshot, &entry));
    CloseHandle(snapshot);
    return found;
}
int main(int argc, char **argv)
{
    BOOL watch = argc > 1 && !strcmp(argv[1], "--watch");
    BOOL seen = FALSE, seen_window = FALSE;
    int missing = 0;
    const char *previous = "";
    HDESK desktop = NULL;
    for (int i = 0; i < (watch ? 1800 : 150); i++) {
        // The observer may start before explorer creates the root desktop.
        if (!desktop) {
            desktop = OpenDesktopW(L"root", 0, FALSE, DESKTOP_READOBJECTS | DESKTOP_ENUMERATE | DESKTOP_WRITEOBJECTS);
            if (desktop) SetThreadDesktop(desktop);
        }
        ready = FALSE;
        EnumWindows(check_window, 0);
        if (!watch && ready) { puts("ready"); return 0; }
        BOOL running = process_running(L"steam.exe");
        if (running) { seen = TRUE; missing = 0; } else missing++;
        if (ready) seen_window = TRUE;
        const char *state = ready ? "ready" : running && seen_window ? "background" :
            !running && seen && missing >= 3 ? "closed" : !seen && i >= 45 ? "failed" : "starting";
        if (watch && strcmp(state, previous)) { puts(state); fflush(stdout); previous = state; }
        if (!strcmp(state, "closed") || !strcmp(state, "failed")) return 0;
        // Once gameplay starts, onboarding no longer needs this observer.
        if (watch && process_running(L"AION2.exe")) return 0;
        Sleep(2000);
    }
    puts("timeout");
    return 1;
}
