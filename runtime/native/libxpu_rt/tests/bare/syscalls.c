/*
 * Freestanding libc support for the Spike bare-metal smoke test.
 *
 * The Zephyr SDK's riscv64-zephyr-elf toolchain ships picolibc, which
 * includes its own ``sbrk`` implementation that reads ``__heap_start``
 * and ``__heap_end`` — both provided by our linker script. So we need
 * only supply the ``_exit`` hook that routes program termination
 * through Spike's HTIF tohost pair.
 */

#include <stdint.h>

extern volatile uint64_t tohost;

__attribute__((noreturn)) void _exit(int code) {
    tohost = ((uint64_t)code << 1) | 1;
    for (;;) {
#if defined(__riscv)
        __asm__ volatile ("wfi");
#endif
    }
}
