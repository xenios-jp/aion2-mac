#include <windows.h>
#include <stdio.h>
int main(void) { printf("%dx%d\n", GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)); return 0; }
