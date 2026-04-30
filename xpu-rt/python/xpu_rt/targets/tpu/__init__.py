"""TPU target class — abstractions any TPU implements.

Placeholder for v3/v4/v5 TPU support. The TPU world has different
primitives (XLA HLO, all-reduce primitives baked into hardware,
no per-block parallelism in the GPU sense), so the Protocol shape
differs from GPU. We expose a narrow surface here and let leaves
fill in.

Things class-level for TPUs:

- Tile-based dataflow rather than per-thread-block parallelism.
- HBM/VMEM hierarchy.
- Cross-chip topology (slice + pod-level all-reduce).

Things NOT class-level:

- Specific TPU generation (v3 has 3D mesh; v5 has Trillium SC).
- Specific JAX/XLA bridge.
"""

from __future__ import annotations

from compgen.targets.tpu.contracts import (
    TpuBodyEmitter,
    TpuRuntime,
    TpuTopology,
)

__all__ = ["TpuBodyEmitter", "TpuRuntime", "TpuTopology"]
