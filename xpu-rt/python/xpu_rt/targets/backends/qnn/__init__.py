"""QNN heterogeneous island scheduling — XPU-RT-owned.

Drives the per-island backend assignment for a partitioned MLIR module on
QRB5165 (CPU + Adreno GPU + Hexagon HTA). The decision is made here, in
XPU-RT, on top of the existing scheduler in `xpu-rt/scheduler.py`. Merlin
owns the kernel-emission half: recognizers (`third_party/merlin/tools/kernels/
qnn_emit_recognizers/`), .qnn.cpp emission, on-board ctxbin build.

Pipeline:
    merlin recognizers ─► IslandVariantGroup[]      (XPU-RT owns the type)
    on-board profiler  ─► CostTable                 (data, not constants)
                            │
                            ▼
    XPU-RT scheduler   ─► per-island machine + Gantt
                            │
                            ▼
    merlin codegen     ─► .qnn-ctx + multi-device VMFB

Key design choices:
  * Per-edge transfer cost (volume × dtype × machine-pair × qparam-delta),
    not an N×N machine matrix. Implemented via XPU-RT's per-(predecessor,
    current) cost map (`processing_times_by_pred`).
  * Fusion as a scheduling variable: each Conv→Activation pair emits both
    fused and split alternatives; the scheduler picks per-instance.
  * QDQ-aware transitions: rescale (same dtype, different qp) and dtype
    swap (uint8↔fp16) are separate, separately-measured cost components.
"""

from .cost_table import CostTable, OpKey, BackendKey  # noqa: F401
from .transfer_model import TransferModel, EdgeSpec  # noqa: F401
from .island_dag import (  # noqa: F401
    IslandCandidate,
    IslandVariantGroup,
    QParams,
    BackendName,
    DType,
)
