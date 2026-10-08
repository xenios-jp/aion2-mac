// macOS x86_64 errno bridge for native calls made with Wine's Windows GS base.
// Wine stores the Darwin TSD base in TEB.Instrumentation[0] (winternl.h: 0x16b8).
// Darwin's errno slot is GS:8; Windows uses that slot for NT_TIB.StackBase.
#include <stdint.h>
#include <errno.h>

static int *wine_native_errno(void)
{
    uintptr_t exception_list, error_slot;
    __asm__ volatile("movq %%gs:0,%0\n\tmovq %%gs:8,%1"
                     : "=r"(exception_list), "=r"(error_slot));
    uintptr_t stack_limit, self;
    __asm__ volatile("movq %%gs:0x10,%0\n\tmovq %%gs:0x30,%1"
                     : "=r"(stack_limit), "=r"(self));
    // ExceptionList can contain a Wine exception frame during a syscall.
    // A Windows TIB also has page-aligned stack bounds and a page-aligned Self.
    // Darwin's errno pointer / MIG port TSD slots do not have this geometry.
    int windows_tib = self && !(self & 0xfff) &&
        error_slot && !(error_slot & 0xfff) && !(stack_limit & 0xfff) &&
        error_slot > stack_limit && error_slot - stack_limit <= UINT32_MAX;
    if (exception_list == UINTPTR_MAX || windows_tib) {
        uintptr_t teb;
        __asm__ volatile("movq %%gs:0x30,%0" : "=r"(teb));
        uintptr_t darwin_tsd = *(const uintptr_t *)(teb + 0x16b8);
        if (darwin_tsd) error_slot = *(const uintptr_t *)(darwin_tsd + 8);
    }
    return (int *)error_slot;
}

__attribute__((used, section("__DATA,__interpose")))
static const struct { const void *replacement; const void *original; } errno_bridge = {
    (const void *)wine_native_errno, (const void *)__error
};
