"""Island data model — what the recognizer hands to the scheduler.

An IslandCandidate is one realisation of a logical op chain on one
backend. An IslandVariantGroup is a set of mutually-exclusive
IslandCandidates representing the same logical chain (e.g. fused vs
split, or HTA-uint8 vs GPU-fp16). The scheduler picks one variant per
group such that the global makespan is minimised under the precedence
DAG.

Each candidate carries:
  * the backend it targets (HTA / GPU / CPU)
  * the dtype + qparams of every IO tensor (so transitions between
    candidates can be priced correctly)
  * a key into the CostTable for execute-time lookup
  * the upstream/downstream group ids it depends on (DAG edges)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BackendName = Literal["HTA", "GPU", "CPU"]
DType = Literal["uint8", "int8", "fp16", "fp32", "int32", "sfixed_32"]


@dataclass(frozen=True)
class QParams:
    """Per-tensor quantisation parameters. fp16/fp32 use scale=1, zp=0
    by convention so transition logic can compare without specialising."""
    scale: float = 1.0
    zero_point: int = 0
    per_channel: tuple[float, ...] | None = None  # populated for per-channel weights only

    def is_float(self) -> bool:
        return self.scale == 1.0 and self.zero_point == 0 and self.per_channel is None


@dataclass(frozen=True)
class TensorSpec:
    """Describes one IO tensor of an island. Used by the transfer model
    to compute volume, and by transition logic to detect dtype/qp swaps."""
    name: str
    shape: tuple[int, ...]
    dtype: DType
    qp: QParams = field(default_factory=QParams)

    @property
    def volume_bytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * _bytes_per_elem(self.dtype)


def _bytes_per_elem(d: DType) -> int:
    return {"uint8": 1, "int8": 1, "fp16": 2, "fp32": 4,
            "int32": 4, "sfixed_32": 4}[d]


@dataclass(frozen=True)
class IslandCandidate:
    """One realisation of one island. Picked or rejected as a unit."""
    candidate_id: str            # unique within the workload
    group_id: str                # which IslandVariantGroup this belongs to
    backend: BackendName
    op_key: str                  # CostTable index ("Conv2d_3x3_s2_320x320x3x16" …)
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    fused_with_next: bool = False  # this candidate already absorbed the activation
    static_setup_us: float = 0.0   # one-time init cost (binary load, etc.)
    notes: str = ""


@dataclass
class IslandVariantGroup:
    """A set of mutually-exclusive realisations for one logical chain.

    The scheduler's MILP/greedy picks at most one alternative per group.
    Edges in the DAG are between groups, not between candidates: the
    candidate within a successor group sees the chosen candidate of each
    predecessor group as its actual upstream.
    """
    group_id: str
    alternatives: list[IslandCandidate]
    upstream_group_ids: tuple[str, ...] = ()  # DAG edges (group→group)

    def primary_dtype_in(self) -> DType:
        return self.alternatives[0].inputs[0].dtype if self.alternatives[0].inputs else "fp32"

    def primary_dtype_out(self) -> DType:
        return self.alternatives[0].outputs[0].dtype if self.alternatives[0].outputs else "fp32"
