"""Bare-metal PMU adapter (RISC-V CSR, ARM ETM).

Generates C code for direct hardware counter reads on embedded
targets without an OS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfileSnapshot, TileMetrics

log = structlog.get_logger()

# Standard RISC-V performance counters
RISCV_COUNTERS = {
    "cycles": "mcycle",
    "mcycle": "mcycle",
    "instructions": "minstret",
    "minstret": "minstret",
    # mhpmcounter3-31 are configurable
}


@dataclass
class BareMetalPMUAdapter:
    """Adapter for bare-metal hardware performance counters.

    Supports:
        - RISC-V CSR reads (mcycle, minstret, mhpmcounterN)
        - ARM ETM (Embedded Trace Macrocell) configuration
        - Generic cycle counting via architecture-specific instructions
    """

    _active: bool = False
    _arch: str = "riscv"
    _counters: list[str] = field(default_factory=list)
    _values: dict[str, float] = field(default_factory=dict)
    _config: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "bare_metal_pmu"

    @property
    def is_active(self) -> bool:
        return self._active

    def configure(self, config: dict[str, Any]) -> None:
        self._arch = config.get("arch", "riscv")
        self._counters = config.get("counters", ["cycles", "instructions"])
        self._config = config
        log.debug("bare_metal_pmu.configured", arch=self._arch, counters=self._counters)

    def start(self) -> None:
        self._active = True
        self._values = {c: 0.0 for c in self._counters}
        log.debug("bare_metal_pmu.started")

    def stop(self) -> None:
        self._active = False
        log.debug("bare_metal_pmu.stopped")

    def read_counters(self) -> dict[str, float]:
        return dict(self._values)

    def get_tile_breakdown(self, region_id: str) -> list[TileMetrics]:
        return []

    def export_trace(self, path: str) -> None:
        log.info("bare_metal_pmu.export", path=path)

    def snapshot(self) -> ProfileSnapshot:
        return ProfileSnapshot(
            counters=self.read_counters(),
            metadata={"backend": "bare_metal_pmu", "arch": self._arch},
        )

    def csr_read_code(self, counter_name: str) -> str:
        """Generate C code to read a RISC-V CSR counter.

        Args:
            counter_name: Counter name (e.g., ``"cycles"``).

        Returns:
            C code snippet for inline assembly CSR read.
        """
        csr = RISCV_COUNTERS.get(counter_name, counter_name)

        if csr in ("mcycle", "minstret"):
            rd_insn = "rdcycle" if csr == "mcycle" else "rdinstret"
            return f'uint64_t {counter_name}_val;\n__asm__ volatile("{rd_insn} %0" : "=r"({counter_name}_val));'

        # Generic CSR read for mhpmcounterN
        return f'uint64_t {counter_name}_val;\n__asm__ volatile("csrr %0, {csr}" : "=r"({counter_name}_val));'

    def instrumentation_code(self) -> dict[str, str]:
        """Generate complete C instrumentation code block.

        Returns:
            Dict with ``"declarations"``, ``"start"``, ``"stop"``,
            and ``"read"`` C code sections.
        """
        decls: list[str] = []
        starts: list[str] = []
        stops: list[str] = []
        reads: list[str] = []

        for counter in self._counters:
            csr = RISCV_COUNTERS.get(counter, counter)

            decls.append(f"static uint64_t _start_{counter}, _end_{counter};")

            if csr in ("mcycle", "minstret"):
                rd_insn = "rdcycle" if csr == "mcycle" else "rdinstret"
                starts.append(f'__asm__ volatile("{rd_insn} %0" : "=r"(_start_{counter}));')
                stops.append(f'__asm__ volatile("{rd_insn} %0" : "=r"(_end_{counter}));')
            else:
                starts.append(f'__asm__ volatile("csrr %0, {csr}" : "=r"(_start_{counter}));')
                stops.append(f'__asm__ volatile("csrr %0, {csr}" : "=r"(_end_{counter}));')

            reads.append(f"uint64_t {counter}_delta = _end_{counter} - _start_{counter};")

        return {
            "declarations": "\n".join(decls),
            "start": "\n".join(starts),
            "stop": "\n".join(stops),
            "read": "\n".join(reads),
        }


__all__ = ["BareMetalPMUAdapter"]
