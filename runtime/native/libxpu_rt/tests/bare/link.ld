/*
 * Spike bare-metal linker script.
 *
 * Places the executable at 0x80000000 (RISC-V reset vector in
 * Spike's default memory map) with text, rodata, data, bss, heap,
 * and stack all in the same RAM region. The heap spans from
 * ``__heap_start`` (after bss) up to the top of the stack.
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64M
}

PROVIDE(_stack_size = 64K);

SECTIONS
{
    . = ORIGIN(RAM);

    .text.init : { *(.text.init) } > RAM
    .text      : { *(.text .text.*) } > RAM
    .rodata    : { *(.rodata .rodata.*) } > RAM

    . = ALIGN(8);
    .data : {
        _fdata = .;
        *(.data .data.*)
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    } > RAM

    .bss (NOLOAD) : {
        . = ALIGN(8);
        _fbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(8);
        _ebss = .;
    } > RAM

    /* HTIF tohost/fromhost — Spike looks up by symbol name, no
     * alignment / addressing constraints beyond being in RAM. */
    .tohost ALIGN(64) : {
        *(.tohost)
    } > RAM

    /* Stack grows down from _sp_top. */
    . = ALIGN(16);
    __heap_start = .;
    . = . + (LENGTH(RAM) - (. - ORIGIN(RAM)) - _stack_size);
    __heap_end = .;
    . = . + _stack_size;
    _sp_top = .;
}
