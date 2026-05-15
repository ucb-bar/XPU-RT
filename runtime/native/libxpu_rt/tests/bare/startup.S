/*
 * Minimal RISC-V startup for a Spike bare-metal ELF.
 *
 * Spike begins execution at the ELF entry point. We set up sp to the
 * top of the static stack segment and jump into C. No trap vectors,
 * no BSS zeroing beyond what the linker script provides via the
 * ``_fbss`` / ``_ebss`` symbols (the C startup clears it).
 */

.section .text.init, "ax", @progbits
.globl _start
_start:
    /* Set global pointer so small-data accesses resolve. */
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    /* Stack: grows down from the top of the defined stack region. */
    la sp, _sp_top

    /* Zero BSS. */
    la t0, _fbss
    la t1, _ebss
1:
    bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:

    /* Enter C. */
    call main

    /* If main returns, halt via HTIF exit(0). */
    la t0, tohost
    li t1, 1
    sd t1, 0(t0)
3:  wfi
    j 3b
