"""CPU target class — abstractions any CPU vendor implements.

Per the unified target hierarchy, this directory holds:

- ``contracts.py`` — Protocols every CPU vendor's package satisfies.
- ``x86/`` — placeholder; future AVX-512 / AVX2 paths.
- ``arm/`` — placeholder; future NEON / SVE paths.

Class-level invariants every CPU shares:

- Single-threaded sequential dispatch is the floor; SIMD intrinsics
  are the per-arch lever.
- No event-tensor coordination over PCIe — intra-CPU sync is
  effectively free; the megakernel scheduler can collapse to a
  serial chain.
- JIT toolchain is per-vendor (clang for x86, gcc for ARM); the
  Protocol abstracts.

Things that are NOT class-level:

- Specific SIMD ISA (AVX-512, NEON, SVE, RVV).
- Specific compiler driver (clang vs gcc vs MSVC).
"""

from __future__ import annotations

from compgen.targets.cpu.contracts import (
    CpuBodyEmitter,
    CpuRuntime,
)

__all__ = ["CpuBodyEmitter", "CpuRuntime"]
