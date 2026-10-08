/* Dismiss only AION2's optional Windows NVIDIA driver-download advisory. */
#include <windows.h>
#include <stdio.h>
#include <wchar.h>
int main(void)
{
    HDESK desktop = NULL;
    for (unsigned int i = 0; i < 600; ++i)
    {
        if (!desktop)
        {
            desktop = OpenDesktopW(L"root", 0, FALSE, DESKTOP_READOBJECTS | DESKTOP_WRITEOBJECTS);
            if (desktop && !SetThreadDesktop(desktop)) { CloseDesktop(desktop); desktop = NULL; }
        }
        if (desktop)
        {
            HWND dialog = FindWindowW(L"#32770", L"Graphics Driver Update Notice");
            DWORD pid = 0, length = 1024;
            wchar_t image[1024], label[32];
            if (dialog)
            {
                GetWindowThreadProcessId(dialog, &pid);
                HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
                BOOL valid = process && QueryFullProcessImageNameW(process, 0, image, &length);
                if (process) CloseHandle(process);
                wchar_t *base = valid ? wcsrchr(image, L'\\') : NULL;
                HWND no = GetDlgItem(dialog, 1004);
                if (base && !_wcsicmp(base + 1, L"AION2.exe") && no &&
                    GetWindowTextW(no, label, 32) && !wcscmp(label, L"No"))
                {
                    PostMessageW(dialog, WM_COMMAND, MAKEWPARAM(1004, BN_CLICKED), (LPARAM)no);
                    puts("Declined AION2's optional Windows graphics-driver download.");
                    fflush(stdout);
                }
            }
        }
        Sleep(1500);
    }
    if (desktop) CloseDesktop(desktop);
    return 0;
}
