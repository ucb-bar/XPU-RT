"""Oracle picker validation harness.

For each calibrated model, runs greedy twice:

- **static**: no calibration env var; uses `static_relative_cost` +
  lex tie-break, the production default.
- **oracle**: ``XPU_RT_CALIBRATION_DIR`` points at the calibration
  table; greedy's cross-site override fires.

Both runs use ``stop_after="post-lowering-verification"`` + the
latency probe. The harness then reports:

- Static delivered min-latency (the production baseline).
- Oracle delivered min-latency.
- Predicted oracle min-latency (from the calibration table).
- **Delivered regret** of static vs oracle: how much did production
  actually leave on the table?
- **Delivered-vs-predicted gap**: did the oracle's delivered latency
  match the calibration's prediction? A large gap means the
  calibration measurement and a fresh-compile execution diverged —
  a new failure mode to investigate.

Closes the loop on the `cost_model_uncalibrated_across_decisions`
caveat by measuring the actual upper bound, not just the calibration
prediction.
"""

from __future__ import annotations

import json
import os
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


_CAL_ENV = "XPU_RT_CALIBRATION_DIR"


@dataclass(frozen=True)
class OracleValidationRow:
    """Static-vs-oracle delivered latency for one model."""

    model_id: str
    target_id: str
    static_candidate_id: str
    static_latency_min_us: float
    static_latency_status: str
    oracle_candidate_id: str
    oracle_latency_min_us: float
    oracle_latency_status: str
    predicted_oracle_min_us: float  # what the calibration said
    # Delivered regret of static vs oracle, in %:
    # (static_min - oracle_min) / oracle_min * 100. Positive = static
    # leaves performance on the table.
    delivered_regret_pct: float | None
    # Delivered-vs-predicted gap, in % of predicted:
    # (delivered - predicted) / predicted * 100. Should be near zero
    # if the calibration measurement generalizes to a fresh compile.
    delivered_vs_predicted_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_id": self.target_id,
            "static_candidate_id": self.static_candidate_id,
            "static_latency_min_us": self.static_latency_min_us,
            "static_latency_status": self.static_latency_status,
            "oracle_candidate_id": self.oracle_candidate_id,
            "oracle_latency_min_us": self.oracle_latency_min_us,
            "oracle_latency_status": self.oracle_latency_status,
            "predicted_oracle_min_us": self.predicted_oracle_min_us,
            "delivered_regret_pct": self.delivered_regret_pct,
            "delivered_vs_predicted_pct": self.delivered_vs_predicted_pct,
        }


@dataclass
class OracleValidationPack:
    schema_version: str = "oracle_validation_pack_v1"
    generated_at_utc: str = field(default_factory=_utc_now)
    commit: str = ""
    calibration_dir: str = ""
    rows: list[OracleValidationRow] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        deliv = [r.delivered_regret_pct for r in self.rows
                 if r.delivered_regret_pct is not None]
        n = len(deliv)
        return {
            "model_count": len(self.rows),
            "measured_count": n,
            "mean_delivered_regret_pct": sum(deliv) / n if n else 0.0,
            "models_with_regret_ge_2pct": sum(1 for r in deliv if r >= 2.0),
            "models_with_regret_ge_5pct": sum(1 for r in deliv if r >= 5.0),
            "max_delivered_regret_pct": max(deliv) if deliv else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "commit": self.commit,
            "calibration_dir": self.calibration_dir,
            "summary": self.summary(),
            "rows": [r.to_dict() for r in self.rows],
        }


def _run_with_env(
    *,
    model_yaml: Path,
    target_yaml: Path,
    out_dir: Path,
    calibration_dir: Path | None,
    n_warmup: int,
    n_iters: int,
) -> tuple[str, LatencyProbeResult, str]:
    """Run greedy + latency probe with / without calibration env var.

    Returns (selected_candidate_id, latency_probe_result, typed_outcome).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    prior = os.environ.get(_CAL_ENV)
    if calibration_dir is not None:
        os.environ[_CAL_ENV] = str(calibration_dir)
    else:
        os.environ.pop(_CAL_ENV, None)
    try:
        result = run_one_cell(
            model_yaml=model_yaml, target_yaml=target_yaml,
            out_dir=out_dir, mode="greedy",
            stop_after="post-lowering-verification",
        )
    finally:
        if prior is None:
            os.environ.pop(_CAL_ENV, None)
        else:
            os.environ[_CAL_ENV] = prior

    if result.typed_outcome in ("verified", "verification_fail"):
        lat = measure_run_dir_latency(
            out_dir, n_warmup=n_warmup, n_iters=n_iters,
        )
    else:
        from xpu_rt.benchmarks.latency_probe import _skipped
        lat = _skipped(f"typed_outcome={result.typed_outcome}")
    return result.selected_candidate_id, lat, result.typed_outcome


def validate_oracle_for_model(
    *,
    model_yaml: Path,
    target_yaml: Path,
    calibration_dir: Path,
    out_dir: Path,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> OracleValidationRow:
    """Run static + oracle greedy on one model and emit a typed row."""
    model_yaml = Path(model_yaml).resolve()
    target_yaml = Path(target_yaml).resolve()
    calibration_dir = Path(calibration_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the calibration's predicted min for the model's "best"
    # candidate. Used to flag delivered-vs-predicted gaps.
    predicted_min_us: float = 0.0
    cal_file = calibration_dir / f"{model_yaml.stem}.json"
    if cal_file.is_file():
        try:
            cd = json.loads(cal_file.read_text(encoding="utf-8"))
            best_id = cd.get("best_candidate_id")
            for m in cd.get("measurements", []):
                if m.get("candidate_id") == best_id and m.get("latency_status") == "ok":
                    predicted_min_us = float(m["latency_min_us"])
                    break
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    static_cid, static_lat, _ = _run_with_env(
        model_yaml=model_yaml, target_yaml=target_yaml,
        out_dir=out_dir / "static",
        calibration_dir=None,
        n_warmup=n_warmup, n_iters=n_iters,
    )
    oracle_cid, oracle_lat, _ = _run_with_env(
        model_yaml=model_yaml, target_yaml=target_yaml,
        out_dir=out_dir / "oracle",
        calibration_dir=calibration_dir,
        n_warmup=n_warmup, n_iters=n_iters,
    )

    delivered_regret: float | None = None
    if (
        static_lat.status == "ok" and oracle_lat.status == "ok"
        and oracle_lat.latency_min_us > 0.0
    ):
        delivered_regret = (
            (static_lat.latency_min_us - oracle_lat.latency_min_us)
            / oracle_lat.latency_min_us * 100.0
        )

    delivered_vs_predicted: float | None = None
    if (
        oracle_lat.status == "ok" and predicted_min_us > 0.0
    ):
        delivered_vs_predicted = (
            (oracle_lat.latency_min_us - predicted_min_us)
            / predicted_min_us * 100.0
        )

    return OracleValidationRow(
        model_id=model_yaml.stem,
        target_id=target_yaml.stem,
        static_candidate_id=static_cid,
        static_latency_min_us=static_lat.latency_min_us,
        static_latency_status=static_lat.status,
        oracle_candidate_id=oracle_cid,
        oracle_latency_min_us=oracle_lat.latency_min_us,
        oracle_latency_status=oracle_lat.status,
        predicted_oracle_min_us=predicted_min_us,
        delivered_regret_pct=delivered_regret,
        delivered_vs_predicted_pct=delivered_vs_predicted,
    )


def emit_pack(pack: OracleValidationPack, *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
