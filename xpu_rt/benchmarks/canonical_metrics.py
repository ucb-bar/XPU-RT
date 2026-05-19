"""Canonical metric row for cross-backend comparison.

Every backend's native output (KB-vanilla's report.md, KB-v2's
:class:`AgentLoopResult`, autocomp's per-iter eval dicts) maps INTO
this row via a per-backend loader. The report aggregator only ever
reads this canonical shape — that's what makes Q1 / Q2 / Q3 in
``cross_target_fair/report.md`` apples-to-apples.

The schema is deliberately flat (no nested dataclasses) so it can
JSON-round-trip without custom encoders and so the report aggregator
can pivot it via pandas-style operations without conversion.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Enumerations (string-typed for ergonomic JSON round-trip)
# ---------------------------------------------------------------------------


# Recognised backends. The report aggregator gates on these literals;
# unknown backends raise rather than silently miss in Q1 / Q2 / Q3.
BACKENDS = ("kb-vanilla", "kb-v2", "autocomp")

# Recognised targets. ``gemmini`` is the canonical id for stock
# INT8 16×16 Gemmini (default Chipyard checkout). ``gemmini_mx`` is
# the historical alias retained so cached KB-vanilla rows stay
# readable — both resolve to the same SpikeTargetSpec.
TARGETS = ("gemmini", "gemmini_mx", "saturn_opu_v128")

# Recognised workloads.
WORKLOADS = ("smolvla_matmuls", "smolvla_mlp_block")

# Recognised cycle sources — captures whether a backend's cycle count
# came from our Gemmini ``MAIN_LD_ST_EX_CYCLES`` counter, autocomp's
# ``Generated implementation latency`` parse target, the RVV
# ``mcycle`` CSR on Saturn, or a cached prior run.
CYCLE_SOURCES = (
    "MAIN_LD_ST_EX_CYCLES",
    "Generated implementation latency",
    "mcycle",
    "cached",
    "none",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalCellRow:
    """One sample from one backend on one (target, workload, shape).

    ``cycles is None`` means the cell didn't measure a cycle count
    (deferred / compile-error / harness-timeout). The aggregator
    treats those as missing-not-zero so they don't pollute geomeans.

    ``cycle_source`` lets the report's harness-skew column label
    each cell with which physical counter produced the cycle
    number — load-bearing for the fair-comparison framing.
    """

    backend: str
    target: str
    workload: str
    shape_id: str
    repeat: int
    correctness: bool
    cycles: int | None
    rounds_used: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    wall_s: float
    cycle_source: str
    notes: str = ""

    def __post_init__(self) -> None:  # noqa: D401
        # Light validation — string enums must match. We don't validate
        # shape_id because every loader formats it differently
        # (KB-vanilla uses ``"[64, 720]×[720, 320]"``; the block
        # enumerator uses ``"action_expert.layer0.mlp"``).
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {BACKENDS}")
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; expected one of {TARGETS}")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}; expected one of {WORKLOADS}")
        if self.cycle_source not in CYCLE_SOURCES:
            raise ValueError(
                f"unknown cycle_source {self.cycle_source!r}; expected one of {CYCLE_SOURCES}"
            )
        if self.repeat < 0:
            raise ValueError(f"repeat must be >= 0 (got {self.repeat})")


# ---------------------------------------------------------------------------
# JSONL persistence — one row per line for easy streaming.
# ---------------------------------------------------------------------------


def write_jsonl(rows: Iterable[CanonicalCellRow], path: Path) -> int:
    """Write rows as JSON Lines. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), sort_keys=True))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[CanonicalCellRow]:
    """Read JSONL back into typed rows. Missing files yield ``[]``."""
    if not path.is_file():
        return []
    rows: list[CanonicalCellRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        body = json.loads(line)
        rows.append(CanonicalCellRow(**body))
    return rows


def merge_jsonl(*paths: Path) -> list[CanonicalCellRow]:
    """Read multiple JSONL files and concatenate. Order preserved."""
    out: list[CanonicalCellRow] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


# ---------------------------------------------------------------------------
# Convenience constructors used by loaders + tests.
# ---------------------------------------------------------------------------


def shape_id_for_matmul(M: int, K: int, N: int) -> str:
    """Canonical shape id for a single matmul: matches the
    KB-vanilla report.md formatting so cross-backend joins work."""
    return f"[{M}, {K}]×[{K}, {N}]"


def parse_matmul_shape_id(shape_id: str) -> tuple[int, int, int] | None:
    """Inverse of :func:`shape_id_for_matmul`. Returns ``None`` when
    the id isn't a matmul shape (e.g. block ids)."""
    import re

    m = re.match(r"\[(\d+),\s*(\d+)\]\s*[×x*]\s*\[\d+,\s*(\d+)\]", shape_id)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


__all__ = [
    "BACKENDS",
    "CYCLE_SOURCES",
    "CanonicalCellRow",
    "TARGETS",
    "WORKLOADS",
    "merge_jsonl",
    "parse_matmul_shape_id",
    "read_jsonl",
    "shape_id_for_matmul",
    "write_jsonl",
]
