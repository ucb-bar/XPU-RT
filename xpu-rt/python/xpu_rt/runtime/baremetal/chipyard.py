"""Chipyard-specific baremetal helpers.

Provides constants and utilities for generating code compatible with
chipyard's RISC-V SoC simulation environment (Verilator, VCS, FireSim).

Key chipyard conventions:
- Base address: 0x80000000 (DRAM)
- HTIF (Host-Target Interface) for test pass/fail communication
- Cross-compiler: riscv64-unknown-elf-gcc
- Linker script must include .htif section
"""

from __future__ import annotations

from compgen.runtime.memory_layout import MemoryRegion

# Chipyard standard addresses
CHIPYARD_DRAM_BASE = 0x80000000
CHIPYARD_DRAM_SIZE = 0x10000000  # 256 MiB (default)

# Cross-compilation
CHIPYARD_CROSS_COMPILE = "riscv64-unknown-elf-"
CHIPYARD_MARCH = "rv64gc"
CHIPYARD_MABI = "lp64d"

# Stack/heap defaults
CHIPYARD_STACK_SIZE = 0x6000  # 24 KB
CHIPYARD_HEAP_SIZE = 0x20000  # 128 KB


def chipyard_dram_region(size_bytes: int = CHIPYARD_DRAM_SIZE) -> MemoryRegion:
    """Create a DRAM memory region with chipyard's standard base address."""
    return MemoryRegion(
        name="dram",
        base_addr=CHIPYARD_DRAM_BASE,
        size_bytes=size_bytes,
        device="cpu",
        address_space=0,
    )


def htif_c_section() -> str:
    """Generate the HTIF linker section for chipyard communication.

    The tohost/fromhost registers are used by chipyard's test harness
    to communicate pass/fail status and for serial I/O.
    """
    return """\
    /* HTIF section for chipyard/FireSim communication */
    .htif ALIGN(0x40) : {
        PROVIDE(__tohost = .);
        LONG(0);
        LONG(0);
        PROVIDE(__fromhost = .);
        LONG(0);
        LONG(0);
    }"""


def htif_pass_fail_c() -> str:
    """Generate C code for chipyard test pass/fail reporting.

    Uses the tohost register to signal test completion to the
    simulation harness.
    """
    return """\
/* Chipyard HTIF pass/fail interface */
extern volatile uint64_t __tohost;
extern volatile uint64_t __fromhost;

static inline void htif_exit(int code) {
    __tohost = (code << 1) | 1;
    while (1) { /* wait for host to acknowledge */ }
}

#define TEST_PASS() htif_exit(0)
#define TEST_FAIL() htif_exit(1)
"""


__all__ = [
    "CHIPYARD_CROSS_COMPILE",
    "CHIPYARD_DRAM_BASE",
    "CHIPYARD_DRAM_SIZE",
    "CHIPYARD_HEAP_SIZE",
    "CHIPYARD_MARCH",
    "CHIPYARD_STACK_SIZE",
    "chipyard_dram_region",
    "htif_c_section",
    "htif_pass_fail_c",
]
