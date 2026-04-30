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

import re
from pathlib import Path

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


def htif_data_stream_c() -> str:
    """C helpers for emitting result data over HTIF.

    Pairs with :func:`parse_htif_data_stream` on the host side.
    The guest writes each 32-bit payload via ``htif_emit_u32(word)``
    (LSB=0, payload in the upper 31 bits), then terminates the stream
    with ``htif_exit(code)`` (LSB=1).

    Use this when the host can't directly read the guest's result
    region — the canonical Chipyard limitation: the rv64 host can't
    issue TileLink ``Get`` against the upper-DRAM TSI region the
    fused-SoC convention places kernel data at. HTIF stream-out is
    the simplest workaround; for larger payloads see
    :func:`shared_dram_section`.
    """
    return """\
/* HTIF data-stream emitter — pair with parse_htif_data_stream on host. */
extern volatile uint64_t __tohost;
extern volatile uint64_t __fromhost;

static inline void htif_emit_u32(uint32_t word) {
    /* LSB=0 marks a data word; payload sits in the upper bits.
       The host walks tohost values and concatenates (val>>1) bytes
       little-endian until the first LSB=1 (exit). */
    __tohost = ((uint64_t)word) << 1;
}

static inline void htif_emit_bytes(const void *src, unsigned n_bytes) {
    const unsigned char *p = (const unsigned char *)src;
    /* Pack 4 bytes per HTIF word, little-endian. Tail bytes are
       zero-padded into the last word. */
    unsigned i = 0;
    while (i + 4 <= n_bytes) {
        uint32_t w = (uint32_t)p[i]
                   | ((uint32_t)p[i + 1] << 8)
                   | ((uint32_t)p[i + 2] << 16)
                   | ((uint32_t)p[i + 3] << 24);
        htif_emit_u32(w);
        i += 4;
    }
    if (i < n_bytes) {
        uint32_t w = 0;
        unsigned shift = 0;
        for (; i < n_bytes; ++i) {
            w |= ((uint32_t)p[i]) << shift;
            shift += 8;
        }
        htif_emit_u32(w);
    }
}
"""


def shared_dram_section(symbol: str = "compgen_shared", size_bytes: int = 0x10000) -> str:
    """Generate a linker fragment for a shared host↔guest DRAM region.

    Places ``<symbol>`` at the top of system DRAM (``0x80000000+``,
    NOT the upper-DRAM TSI region used by the fused-SoC convention),
    so the host can do a normal TileLink ``Get`` against it after the
    sim terminates. Use this as an alternative to the HTIF stream-out
    helpers when the result payload is too large to stream word-by-word.

    Args:
        symbol: C symbol the kernel writes into (and the host reads from).
        size_bytes: Region size in bytes — defaults to 64 KiB.

    Returns:
        A linker-script fragment placeable inside a ``SECTIONS { … }``
        block of a Chipyard-style linker file.
    """
    return f"""\
    /* Shared host/guest region in system DRAM (0x80000000+).
       The host can issue TileLink Get against this region; the upper
       DRAM (0x110000000+) is TSI-only and refuses Get. */
    .compgen_shared ALIGN(64) : {{
        PROVIDE({symbol} = .);
        . = . + {hex(size_bytes)};
    }} > REGION_DRAM
"""


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


# ---------------------------------------------------------------------------
# Sim-log parsing — generic across any Chipyard Verilator/VCS/FireSim run.
# ---------------------------------------------------------------------------

# `Time: NNN ps` line VCS emits at $finish (picosecond resolution).
_TIME_LINE_RE = re.compile(r"^\s*Time:\s+(\d+)\s+ps", re.MULTILINE)
# `$finish called from file "<path>", line <N>.` from Chisel
# `printf("%m: $finish");` macros after lowering.
_FINISH_RE = re.compile(
    r'\$finish called from file\s+"(?P<file>[^"]+)",\s*line\s+(?P<line>\d+)',
)
# HTIF `tohost = 0x…` writes the rocket harness echoes.
_TOHOST_RE = re.compile(r"tohost\s*=\s*0x([0-9a-fA-F]+)")
# Common assertion / cache-fault markers across rocket-chip-derived SoCs
# (Rocket cores, accelerators on TileLink, Gemmini-style coprocessors, etc.).
_ASSERT_MARKERS = ("Assertion failed", "Assert", "s2_xcpt", "TLEvent", "TLMonitor")
_ERROR_MARKERS = ("Fatal:",)


def parse_chipyard_finish(log: str) -> dict[str, object]:
    """Extract ``$finish`` reason + cycles + error/assert lines from a sim log.

    Returns a dict with keys ``finish_reason``, ``finish_file``,
    ``finish_line``, ``sim_time_ps``, ``cycles``, ``error_lines``,
    ``asserts``.

    The "happy path" finish reason for any Chipyard SoC that terminates
    via a Chisel ``printf`` + ``$finish`` is the basename (without
    extension) of the file the ``$finish`` was emitted from. Callers
    cross-check that against an expected name (e.g. a top-level
    aggregator module) to declare pass/fail.

    ``cycles`` assumes the standard 1 GHz / 1 ns step every Chipyard sim
    uses today; override by computing ``sim_time_ps // period_ps``.
    """
    finish = _FINISH_RE.search(log)
    time_match = _TIME_LINE_RE.search(log)
    sim_time_ps: int | None = int(time_match.group(1)) if time_match else None
    cycles: int | None = sim_time_ps // 1000 if sim_time_ps is not None else None

    error_lines: list[str] = []
    asserts: list[str] = []
    for line in log.splitlines():
        stripped = line.strip()
        matched_assert = next((m for m in _ASSERT_MARKERS if m in line), None)
        if matched_assert is not None:
            asserts.append(stripped)
            continue
        matched_error = next((m for m in _ERROR_MARKERS if m in line), None)
        if matched_error is not None:
            error_lines.append(stripped)

    finish_reason: str | None = None
    finish_file: str | None = None
    finish_line: int | None = None
    if finish:
        finish_file = finish.group("file")
        finish_line = int(finish.group("line"))
        # Use the basename without extension as the canonical reason —
        # callers grep for the module name, not the full Verilog path.
        finish_reason = Path(finish_file).stem

    return {
        "finish_reason": finish_reason,
        "finish_file": finish_file,
        "finish_line": finish_line,
        "sim_time_ps": sim_time_ps,
        "cycles": cycles,
        "error_lines": error_lines,
        "asserts": asserts,
    }


def parse_htif_exit(log: str) -> tuple[bool, int | None]:
    """Walk HTIF ``tohost = 0x…`` lines and return ``(saw_exit, code)``.

    Per the HTIF contract, the LSB of the tohost word distinguishes data
    (LSB=0) from exit (LSB=1); the exit code is ``(val >> 1)``. Data
    words are not surfaced here — see :func:`parse_htif_data_stream`,
    which collects the payload bytes from LSB=0 writes.
    """
    for m in _TOHOST_RE.finditer(log):
        try:
            val = int(m.group(1), 16)
        except ValueError:
            continue
        if val & 1:
            return True, val >> 1
    return False, None


def parse_htif_data_stream(log: str) -> bytes:
    """Walk HTIF ``tohost = 0x…`` lines and return the concatenated payload.

    Per the HTIF contract, ``tohost`` words with the LSB clear are data
    writes; the data byte/word is ``(val >> 1)``. Concatenates each
    little-endian payload until the first LSB=1 (exit) — at which
    point the stream stops. Exit code is not returned here; use
    :func:`parse_htif_exit` for that.

    Useful for guests that emit result tensors via repeated
    ``__tohost = (data << 1)`` then terminate with ``htif_exit(code)``.
    """
    out = bytearray()
    for m in _TOHOST_RE.finditer(log):
        try:
            val = int(m.group(1), 16)
        except ValueError:
            continue
        if val & 1:
            break
        # Drop the exit-marker LSB and emit the remaining word
        # little-endian. 32-bit payload is the common HTIF convention
        # — wider words can be split into 32-bit chunks guest-side.
        payload = val >> 1
        out.extend(payload.to_bytes(4, "little"))
    return bytes(out)


__all__ = [
    "CHIPYARD_CROSS_COMPILE",
    "CHIPYARD_DRAM_BASE",
    "CHIPYARD_DRAM_SIZE",
    "CHIPYARD_HEAP_SIZE",
    "CHIPYARD_MARCH",
    "CHIPYARD_STACK_SIZE",
    "chipyard_dram_region",
    "htif_c_section",
    "htif_data_stream_c",
    "htif_pass_fail_c",
    "parse_chipyard_finish",
    "parse_htif_data_stream",
    "parse_htif_exit",
    "shared_dram_section",
]
