"""QRB5165 hardware specification for the analytical (roofline) bound.

Numbers below are **published spec sheet values** for the QRB5165
robotics platform (Snapdragon 865 family). They are intentionally
conservative (peak-throughput-from-the-datasheet, not measured
peak), and are used **only** as a roofline lower bound to flag
whether an unbuilt candidate island is worth measuring.

A cost produced by :func:`analytical_bound_us` is **never**
substituted into the production MOSEK schedule unless the caller
explicitly opts in with ``bound_only=True`` — and the rubric in
``.claude/skills/xpu-rt-compile.md`` does not opt in. Roofline cells
exist so the planner can answer the question "could splitting this
op help?" without lying about the cost.

Sources (committed comments):
    CPU  — Kryo 585 (4× Cortex-A77 prime/perf + 4× Cortex-A55).
           Sustained fp32 throughput ≈ 50 GFLOPS (across the 4
           A77 cores at ~2.8 GHz, NEON FMA).
    GPU  — Adreno 650. Theoretical fp16 peak ≈ 1100 GFLOPS.
    DSP  — Hexagon V66 cDSP. int8 vector throughput ≈ 410 GOPS.
           8 MB TCM scratch.
    HTA  — Hexagon Tensor Accelerator (separate coprocessor on the
           V66 cDSP). int8 conv-only; ≈ 3.4 TOPS peak; 8 MB
           scratch.
    HTP  — Hexagon Tensor Processor (newer NPU on QRB5165). int8
           peak ≈ 15 TOPS; not directly callable with the standard
           libQnnHtp.so on our DLCs (segfaults rc=139).
    Memory — LPDDR5 @ ~32 GB/s sustained bandwidth (16-bit interface
             on the QRB5165 module).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

BackendName = Literal["CPU", "GPU", "DSP", "HTA", "HTP"]


@dataclasses.dataclass(frozen=True)
class BackendSpec:
    """Static, published numbers for one backend on QRB5165."""

    name: BackendName
    peak_throughput_ops: float    # ops/sec (fp32 / fp16 / int8 depending on dtype)
    dtype: Literal["fp32", "fp16", "int8"]
    bandwidth_bytes_per_s: float  # sustained main-memory bandwidth
    scratchpad_bytes: int = 0     # local SRAM / TCM (0 if main-memory-backed)
    notes: str = ""


# QRB5165 hardware sheet — frozen constants used by the roofline.
QRB5165_BACKENDS: dict[str, BackendSpec] = {
    "CPU": BackendSpec(
        name="CPU",
        peak_throughput_ops=50e9,        # 50 GFLOPS fp32 across 4× A77
        dtype="fp32",
        bandwidth_bytes_per_s=32e9,      # LPDDR5 sustained
        scratchpad_bytes=0,
        notes="Kryo 585 (4× A77 + 4× A55); we use the A77 cluster.",
    ),
    "GPU": BackendSpec(
        name="GPU",
        peak_throughput_ops=1100e9,      # 1.1 TFLOPS fp16
        dtype="fp16",
        bandwidth_bytes_per_s=32e9,
        scratchpad_bytes=0,
        notes="Adreno 650, fp16 path.",
    ),
    "DSP": BackendSpec(
        name="DSP",
        peak_throughput_ops=410e9,       # 410 GOPS int8
        dtype="int8",
        bandwidth_bytes_per_s=32e9,
        scratchpad_bytes=8 * 1024 * 1024,
        notes="Hexagon V66 cDSP vector unit, int8.",
    ),
    "HTA": BackendSpec(
        name="HTA",
        peak_throughput_ops=3.4e12,      # 3.4 TOPS int8
        dtype="int8",
        bandwidth_bytes_per_s=32e9,
        scratchpad_bytes=8 * 1024 * 1024,
        notes="Hexagon Tensor Accelerator; int8 conv only.",
    ),
    "HTP": BackendSpec(
        name="HTP",
        peak_throughput_ops=15e12,       # 15 TOPS int8
        dtype="int8",
        bandwidth_bytes_per_s=32e9,
        scratchpad_bytes=8 * 1024 * 1024,
        notes="Hexagon Tensor Processor (newer NPU).",
    ),
}


@dataclasses.dataclass(frozen=True)
class OpFootprint:
    """What the roofline needs to know about one op.

    - ``flops``: total compute (multiply-accumulates × 2 if you
      report MACs; report as raw FLOPs to keep consistent with the
      ``peak_throughput_ops`` units).
    - ``bytes_read``: input + weights bytes pulled from main memory
      (counted once per inference, not per-tile).
    - ``bytes_written``: output bytes pushed to main memory.
    """

    flops: float
    bytes_read: float
    bytes_written: float

    @property
    def bytes_total(self) -> float:
        return self.bytes_read + self.bytes_written


def analytical_bound_us(
    op: OpFootprint, backend: BackendName,
) -> tuple[float, str]:
    """Roofline lower bound: max(compute_time, memory_time).

    Returns ``(microseconds, rationale)``. The rationale string
    names which roof was binding ("compute" / "memory") so the
    agent can surface why an unbuilt candidate looks plausible or
    not.

    **This is a LOWER bound**, never a real cost. The caller MUST
    tag the resulting cell ``provenance="analytical_bound"`` and
    refuse to schedule against it unless ``bound_only=True``.
    """
    if backend not in QRB5165_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; "
                         f"known: {sorted(QRB5165_BACKENDS)}")
    spec = QRB5165_BACKENDS[backend]
    compute_s = op.flops / spec.peak_throughput_ops
    memory_s = op.bytes_total / spec.bandwidth_bytes_per_s
    if compute_s >= memory_s:
        return compute_s * 1e6, "compute-bound"
    return memory_s * 1e6, "memory-bound"


def backend_dtype(backend: BackendName) -> str:
    """The dtype the spec assumes for the roofline numbers above."""
    return QRB5165_BACKENDS[backend].dtype


def is_compute_only(op: OpFootprint, backend: BackendName) -> bool:
    """True when the op is squarely compute-bound on this backend."""
    spec = QRB5165_BACKENDS[backend]
    return op.flops / spec.peak_throughput_ops > op.bytes_total / spec.bandwidth_bytes_per_s
