"""Ukernel dialect -- stable leaf-call boundary for all kernel backends.

Provides a uniform IR representation for calling into external kernel
implementations, regardless of backend (Triton, CUDA, vendor library,
handwritten assembly).

The ukernel dialect is intentionally small:
    - ``UkernelCallOp``: call a kernel with metadata
    - ``UkernelDeclOp``: declare a kernel interface
    - ``UkernelContract``: kernel interface contract (types, effects, perf bounds)

This is the call boundary that the planner/runtime schedules uniformly.
"""

from __future__ import annotations

__all__: list[str] = []
