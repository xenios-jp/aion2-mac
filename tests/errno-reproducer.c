#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach.h>
#include <os/os_sync_wait_on_address.h>

extern void _thread_set_tsd_base(uint64_t);

int main(int argc, char **argv)
{
    thread_identifier_info_data_t info;
    mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
    mach_port_t thread = mach_thread_self();
    if (thread_info(thread, THREAD_IDENTIFIER_INFO, (thread_info_t)&info, &count)) return 2;
    mach_port_deallocate(mach_task_self(), thread);
    uintptr_t *teb = aligned_alloc(4096, 0x2000);
    memset(teb, 0, 0x2000);
    void *stack = aligned_alloc(4096, 0x2000);
    int *sentinel = (int *)((char *)stack + 4096);
    *sentinel = 0x12345678;
    int *native_errno = __error();
    *native_errno = 0;
    teb[0] = argc > 1 ? (uintptr_t)teb : UINTPTR_MAX;
    teb[1] = (uintptr_t)sentinel;
    teb[2] = (uintptr_t)stack;
    teb[0x30/8] = (uintptr_t)teb;
    teb[0x16b8/8] = info.thread_handle;
    _thread_set_tsd_base((uintptr_t)teb);
    int *windows_context_errno = __error();
    volatile uintptr_t invalid_address = 0;
    int wait_result = os_sync_wait_on_address((void *)invalid_address, 0, 4,
                                             OS_SYNC_WAIT_ON_ADDRESS_NONE);
    _thread_set_tsd_base(info.thread_handle);
    printf("Native errno preserved: %s\n", windows_context_errno == native_errno ? "yes" : "no");
    printf("Native errno points at Windows StackBase: %s\n", windows_context_errno == sentinel ? "yes" : "no");
    printf("Wait result: %d; native errno: %d; stack sentinel: %x\n",
           wait_result, *native_errno, *sentinel);
    int sentinel_value = *sentinel;
    free(teb);
    free(stack);
    return windows_context_errno == native_errno && sentinel_value == 0x12345678 &&
           wait_result == -1 && *native_errno != 0 ? 0 : 1;
}
