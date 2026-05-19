"""Cross-compile bundle stages.

The host pipeline emits kernel C sources under
``<bundle>/generated_kernels/<provider>/`` plus a description of the
execution order in ``execution_plan.yaml``. For non-x86 targets the
pipeline cannot ``cffi``-compile those sources locally — they need a
cross-toolchain.

This package provides a per-target-triple orchestrator that:

  1. Reads the ``cross_compile`` block from the target profile.
  2. Stages all generated kernel sources + a generated driver into a
     CMake project under ``<bundle>/cross_compile_<triple>/``.
  3. Cross-compiles to a single static ELF + writes it to
     ``<bundle>/program.elf``.

Phase C v1 supports a single target triple: ``riscv64-unknown-elf``
running on Chipyard's spike via HTIF. The orchestrator is in
:mod:`xpu_rt.runtime.cross_compile.riscv64_bare`.
"""

from __future__ import annotations

from .riscv64_bare import (
    CrossCompileError,
    CrossCompileResult,
    cross_compile_riscv64_bundle,
)

__all__ = [
    "CrossCompileError",
    "CrossCompileResult",
    "cross_compile_riscv64_bundle",
]
