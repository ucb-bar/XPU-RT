"""Empirical cost-calibration data collector.

For a given model, runs the graph-compilation pipeline once per
viable candidate (via ``selection_mode="agent-file"`` with a
synthesized response) and records the resulting min-based CPU
latency. The output is a per-model table

    (candidate_id, candidate_kind, static_relative_cost,
     typed_outcome, latency_min_us, latency_stddev_us, n_iters)

which serves two purposes:

1. **Standalone evidence.** It tells an operator which candidate is
   empirically fastest on a given model — useful even before any
   learned cost model lands. The current `static_relative_cost` is
   uncalibrated (0.78 for fusion, ~1.0 for tile); the empirical
   column lets us see how far off.
2. **Training data for a learned cost model.** The open caveat
   `fusion_cost_model_uncalibrated` calls for a cost model that
   picks the right candidate per workload. This table is the
   per-(model, candidate) ground truth that closure requires.

The data collector is mask-agnostic by design — it relies on
agent-file mode to force each candidate selection. Subsystem masks
(e.g. gating fusion off) would change which candidates surface in
the enumeration step but not the per-candidate measurement once
selected.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.latency_probe import (
    LatencyProbeResult,
    measure_run_dir_latency,
)
from xpu_rt.benchmarks.pass_pool_ablation import run_one_cell


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateMeasurement:
    """One forced-pick measurement."""

    candidate_id: str
    candidate_kind: str
    static_relative_cost: float
    typed_outcome: str  # verified | verification_fail | typed_blocked | error
    error: str
    latency_min_us: float
    latency_median_us: float
    latency_stddev_us: float
    latency_status: str  # ok | skipped | error
    n_iters: int
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "static_relative_cost": self.static_relative_cost,
            "typed_outcome": self.typed_outcome,
            "error": self.error,
            "latency_min_us": self.latency_min_us,
            "latency_median_us": self.latency_median_us,
            "latency_stddev_us": self.latency_stddev_us,
            "latency_status": self.latency_status,
            "n_iters": self.n_iters,
            "run_dir": self.run_dir,
        }


@dataclass
class ModelCalibration:
    """One model's calibration table."""

    model_id: str
    target_id: str
    commit: str = ""
    schema_version: str = "model_calibration_v1"
    generated_at_utc: str = field(default_factory=_utc_now)
    measurements: list[CandidateMeasurement] = field(default_factory=list)

    def best(self) -> CandidateMeasurement | None:
        """Lowest-latency verified candidate, or None when nothing verified."""
        ok = [
            m for m in self.measurements
            if m.typed_outcome == "verified" and m.latency_status == "ok"
        ]
        if not ok:
            return None
        return min(ok, key=lambda m: m.latency_min_us)

    def static_pick(self) -> CandidateMeasurement | None:
        """The candidate greedy's static priority would pick.

        Today greedy sorts by (boundary_required, not_promoted,
        static_relative_cost, candidate_id). We approximate with
        static_relative_cost only — boundary/promoted aren't in
        scope for this calibration since we measure the lowered
        outcome, not the selection process.
        """
        ok = [m for m in self.measurements if m.latency_status == "ok"]
        if not ok:
            return None
        return min(ok, key=lambda m: (m.static_relative_cost, m.candidate_id))

    def regret_pct(self) -> float | None:
        """How much worse is the static pick than the empirical best?

        `(static_min - best_min) / best_min * 100`. Returns None when
        either pick is unavailable.
        """
        b = self.best()
        s = self.static_pick()
        if b is None or s is None or b.latency_min_us == 0.0:
            return None
        return (s.latency_min_us - b.latency_min_us) / b.latency_min_us * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "commit": self.commit,
            "model_id": self.model_id,
            "target_id": self.target_id,
            "measurements": [m.to_dict() for m in self.measurements],
            "best_candidate_id": self.best().candidate_id if self.best() else None,
            "static_pick_candidate_id": (
                self.static_pick().candidate_id if self.static_pick() else None
            ),
            "regret_pct": self.regret_pct(),
        }


# --------------------------------------------------------------------------- #
# Response synthesis
# --------------------------------------------------------------------------- #


def _build_response(
    candidate_id: str,
    candidate_kind: str,
    static_relative_cost: float,
) -> dict[str, Any]:
    """Synthesize an `agent_decision_response_v1` forcing `candidate_id`.

    The validator requires `rationale.evidence` to have >=2 entries
    each referencing a real `candidate.*` or `region.*` field; the
    body of each entry can be a stub since the agent is the
    calibrator forcing a deterministic pick. Two entries that
    reference fields present on every candidate kind
    (`candidate.kind`, `candidate.cost_preview.static_relative_cost`)
    satisfy `rationale_evidence_present` +
    `rationale_references_real_fields`.
    """
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": candidate_id,
        "rationale": {
            "summary": f"Cost-calibration probe forcing {candidate_kind}.",
            "evidence": [
                {"field": "candidate.kind", "value": candidate_kind,
                 "reason": "forced by calibrator"},
                {"field": "candidate.cost_preview.static_relative_cost",
                 "value": static_relative_cost,
                 "reason": (
                     "captured for empirical-vs-static regret analysis"
                 )},
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #


# Kinds the calibrator forces. Each must be a kind that greedy or
# the agent could choose in production. Numerics/placement/dispatch
# candidates are excluded because they layer over a primary pick
# (not standalone decisions).
_FORCEABLE_KINDS: frozenset[str] = frozenset({
    "set_tile_params",
    "fuse_producer_consumer",
    "create_kernel_contract",
    "keep_as_fallback",
})


def _enumerate_candidates(
    model_yaml: Path, target_yaml: Path, enum_dir: Path,
) -> list[dict[str, Any]]:
    """Run greedy to graph-analysis and return viable candidates."""
    if enum_dir.exists():
        shutil.rmtree(enum_dir)
    run_one_cell(
        model_yaml=model_yaml, target_yaml=target_yaml,
        out_dir=enum_dir, mode="greedy",
        stop_after="graph-analysis",
    )
    cas_path = enum_dir / "02_graph_analysis" / "candidate_actions.json"
    if not cas_path.is_file():
        return []
    cas = json.loads(cas_path.read_text())
    viable: list[dict[str, Any]] = []
    for c in cas.get("candidates", []):
        if c.get("kind") not in _FORCEABLE_KINDS:
            continue
        if not c.get("legality", {}).get("ok"):
            continue
        viable.append(c)
    return viable


def calibrate_model(
    *,
    model_yaml: Path,
    target_yaml: Path,
    out_dir: Path,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> ModelCalibration:
    """Run a calibration sweep for one model.

    Steps:
      1. Enumerate viable candidates (greedy → graph-analysis).
      2. For each, run agent-file forcing that pick + measure latency.
      3. Return a `ModelCalibration` with one measurement per viable
         candidate.
    """
    model_yaml = Path(model_yaml).resolve()
    target_yaml = Path(target_yaml).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = ModelCalibration(
        model_id=model_yaml.stem,
        target_id=target_yaml.stem,
    )

    viable = _enumerate_candidates(
        model_yaml, target_yaml, out_dir / "_enum",
    )

    responses_dir = out_dir / "_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    for c in viable:
        cid = c["candidate_id"]
        kind = c["kind"]
        static_cost = float(
            c.get("cost_preview", {}).get("static_relative_cost", 1.0)
        )
        response_path = responses_dir / f"{cid}.json"
        response_path.write_text(
            json.dumps(
                _build_response(cid, kind, static_cost), indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        cell_dir = out_dir / f"cand_{cid}"
        result = run_one_cell(
            model_yaml=model_yaml, target_yaml=target_yaml,
            out_dir=cell_dir, mode="agent-file",
            agent_response_path=response_path,
            stop_after="post-lowering-verification",
        )
        lat: LatencyProbeResult
        if result.typed_outcome in ("verified", "verification_fail"):
            lat = measure_run_dir_latency(
                cell_dir, n_warmup=n_warmup, n_iters=n_iters,
            )
        else:
            from xpu_rt.benchmarks.latency_probe import _skipped
            lat = _skipped(f"typed_outcome={result.typed_outcome}")
        cal.measurements.append(CandidateMeasurement(
            candidate_id=cid,
            candidate_kind=kind,
            static_relative_cost=static_cost,
            typed_outcome=result.typed_outcome,
            error=result.error,
            latency_min_us=lat.latency_min_us,
            latency_median_us=lat.latency_median_us,
            latency_stddev_us=lat.latency_stddev_us,
            latency_status=lat.status,
            n_iters=lat.n_iters,
            run_dir=str(cell_dir),
        ))
    return cal


def emit_calibration(
    cal: ModelCalibration, *, out_path: Path,
) -> Path:
    """Write calibration as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
