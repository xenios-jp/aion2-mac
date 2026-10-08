#define UNICODE
#define _UNICODE
#include <windows.h>
#include <tlhelp32.h>
#include <wchar.h>

static DWORD game_pid;
static BOOL CALLBACK focus_window(HWND window, LPARAM unused)
{
    DWORD pid;
    wchar_t klass[128];
    (void)unused;
    GetWindowThreadProcessId(window, &pid);
    if (pid != game_pid || !IsWindowVisible(window)) return TRUE;
    GetClassNameW(window, klass, 128);
    if (wcscmp(klass, L"UnrealWindow")) return TRUE;
    if (IsIconic(window)) ShowWindow(window, SW_RESTORE);
    SetForegroundWindow(window);
    return FALSE;
}
int main(void)
{
    // Wine's snapshot is scoped to this prefix, unlike a global macOS ps match.
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W entry = {0};
    if (snapshot == INVALID_HANDLE_VALUE) return 1;
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snapshot, &entry)) {
        do {
            if (!_wcsicmp(entry.szExeFile, L"AION2.exe")) {
                game_pid = entry.th32ProcessID;
                break;
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    if (!game_pid) return 1;
    HDESK desktop = OpenDesktopW(L"root", 0, FALSE, DESKTOP_READOBJECTS | DESKTOP_ENUMERATE | DESKTOP_WRITEOBJECTS);
    if (desktop) SetThreadDesktop(desktop);
    EnumWindows(focus_window, 0);
    // A starting game may not have created its window yet. Avoid a second launch.
    return 0;
}
