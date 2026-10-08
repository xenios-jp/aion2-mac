/* Negative test: a matching advisory owned by another executable must stay open. */
#include <windows.h>
#include <stdio.h>
int main(void)
{
    HDESK desktop = OpenDesktopW(L"root", 0, FALSE, DESKTOP_READOBJECTS | DESKTOP_WRITEOBJECTS);
    if (!desktop || !SetThreadDesktop(desktop)) return 2;
    HWND dialog = CreateWindowExW(0, L"#32770", L"Graphics Driver Update Notice",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100, 320, 180, NULL, NULL, NULL, NULL);
    if (!dialog) return 3;
    HWND button = CreateWindowW(L"BUTTON", L"No", WS_CHILD | WS_VISIBLE,
        100, 80, 80, 30, dialog, (HMENU)1004, NULL, NULL);
    if (!button) return 4;
    DWORD until = GetTickCount() + 5000;
    MSG message;
    while ((LONG)(until - GetTickCount()) > 0)
    {
        while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE))
        {
            if (message.hwnd == dialog && message.message == WM_COMMAND && LOWORD(message.wParam) == 1004)
            { puts("FAIL: unrelated dialog was declined"); return 1; }
            TranslateMessage(&message); DispatchMessageW(&message);
        }
        Sleep(20);
    }
    puts("PASS: matching unrelated dialog was left untouched");
    DestroyWindow(dialog);
    CloseDesktop(desktop);
    return 0;
}
