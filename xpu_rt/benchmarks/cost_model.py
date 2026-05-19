"""Learned cost model — train on calibration tables, evaluate via LOMO-CV.

A simple linear-regression cost model that predicts latency_min_us
from candidate features. Training data is the per-model calibration
tables produced by `cost_calibration.calibrate_model`.

Features extracted per candidate:

- One-hot ``candidate_kind`` (4 dims: set_tile_params,
  fuse_producer_consumer, create_kernel_contract, keep_as_fallback).
- ``log(static_relative_cost + 1e-3)`` (numeric).
- For set_tile_params: parsed M, N, K from candidate_id (zeros for
  non-tile candidates).
- ``region_index``: parsed matmul_N → N (zero when no region tag).

Target: ``log(latency_min_us)``.

Evaluation: leave-one-model-out cross-validation. For each held-out
model, predict latencies for all its candidates, pick the lowest-
predicted, measure regret vs the empirical best (the candidate with
the lowest actual latency_min_us in that model's calibration).

The model is small enough that overfitting on the training side is
unlikely; the value question is whether the learned weights generalize
to a held-out model. If LOMO-CV mean regret beats static (which the
calibration audit measured at +27.60% mean), the learned model has
demonstrated real generalization across models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_KIND_ORDER = (
    "set_tile_params",
    "fuse_producer_consumer",
    "create_kernel_contract",
    "keep_as_fallback",
)

_TILE_RE = re.compile(r"tile_M(\d+)_N(\d+)_K(\d+)")
_REGION_IDX_RE = re.compile(r"matmul_(\d+)")


def _features_for_candidate(
    kind: str, candidate_id: str, static_relative_cost: float,
) -> np.ndarray:
    """Return a 9-feature row: 4 kind one-hot + log_cost + M + N + K + region_idx."""
    feats = np.zeros(9, dtype=float)
    if kind in _KIND_ORDER:
        feats[_KIND_ORDER.index(kind)] = 1.0
    feats[4] = float(np.log(max(float(static_relative_cost), 1e-3)))
    m_match = _TILE_RE.search(candidate_id)
    if m_match:
        feats[5] = float(m_match.group(1))  # M
        feats[6] = float(m_match.group(2))  # N
        feats[7] = float(m_match.group(3))  # K
    r_match = _REGION_IDX_RE.search(candidate_id)
    if r_match:
        feats[8] = float(r_match.group(1))
    return feats


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Sample:
    model_id: str
    candidate_id: str
    kind: str
    static_relative_cost: float
    latency_min_us: float


def load_samples(calibration_dir: Path) -> list[_Sample]:
    """Read every *.json under calibration_dir, return verified+ok samples."""
    out: list[_Sample] = []
    for p in sorted(Path(calibration_dir).glob("*.json")):
        if p.stem == "summary":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mid = d.get("model_id", p.stem)
        for m in d.get("measurements", []):
            if m.get("latency_status") != "ok":
                continue
            if m.get("typed_outcome") != "verified":
                continue
            out.append(_Sample(
                model_id=mid,
                candidate_id=m["candidate_id"],
                kind=m["candidate_kind"],
                static_relative_cost=float(m["static_relative_cost"]),
                latency_min_us=float(m["latency_min_us"]),
            ))
    return out


def _features_matrix(samples: list[_Sample]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([
        _features_for_candidate(s.kind, s.candidate_id, s.static_relative_cost)
        for s in samples
    ])
    y = np.log(np.array([s.latency_min_us for s in samples]))
    return X, y


def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS with bias term. Returns beta of shape (n_features+1,)."""
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    # Use lstsq for numerical stability vs (X^T X)^-1.
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return log-latency predictions. Caller exp's to get latency_us."""
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xb @ beta


# --------------------------------------------------------------------------- #
# Leave-one-model-out cross-validation
# --------------------------------------------------------------------------- #


@dataclass
class CVRow:
    model_id: str
    predicted_pick_id: str
    predicted_pick_latency_us: float
    empirical_best_id: str
    empirical_best_latency_us: float
    static_pick_id: str
    static_pick_latency_us: float
    learned_regret_pct: float
    static_regret_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "predicted_pick_id": self.predicted_pick_id,
            "predicted_pick_latency_us": self.predicted_pick_latency_us,
            "empirical_best_id": self.empirical_best_id,
            "empirical_best_latency_us": self.empirical_best_latency_us,
            "static_pick_id": self.static_pick_id,
            "static_pick_latency_us": self.static_pick_latency_us,
            "learned_regret_pct": self.learned_regret_pct,
            "static_regret_pct": self.static_regret_pct,
        }


@dataclass
class CVReport:
    rows: list[CVRow] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.rows)
        if not n:
            return {"model_count": 0}
        learned = np.array([r.learned_regret_pct for r in self.rows])
        static = np.array([r.static_regret_pct for r in self.rows])
        return {
            "model_count": n,
            "learned_mean_regret_pct": float(learned.mean()),
            "static_mean_regret_pct": float(static.mean()),
            "learned_median_regret_pct": float(np.median(learned)),
            "static_median_regret_pct": float(np.median(static)),
            "learned_max_regret_pct": float(learned.max()),
            "static_max_regret_pct": float(static.max()),
            "learned_beats_static_count": int(np.sum(learned < static)),
            "learned_loses_to_static_count": int(np.sum(learned > static)),
            "learned_ties_static_count": int(np.sum(np.isclose(learned, static))),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cost_model_cv_v1",
            "summary": self.summary(),
            "rows": [r.to_dict() for r in self.rows],
        }


def _static_pick(samples: list[_Sample]) -> _Sample:
    """The static priority's pick (lowest static_relative_cost, lex tie)."""
    return min(samples, key=lambda s: (s.static_relative_cost, s.candidate_id))


def _empirical_best(samples: list[_Sample]) -> _Sample:
    return min(samples, key=lambda s: s.latency_min_us)


def leave_one_model_out_cv(samples: list[_Sample]) -> CVReport:
    """LOMO-CV: train on all-but-one model, predict on the held-out."""
    model_ids = sorted({s.model_id for s in samples})
    report = CVReport()
    for held in model_ids:
        train = [s for s in samples if s.model_id != held]
        test = [s for s in samples if s.model_id == held]
        if not train or not test:
            continue
        Xtr, ytr = _features_matrix(train)
        beta = fit_linear(Xtr, ytr)
        Xte, _yte = _features_matrix(test)
        log_pred = predict(beta, Xte)
        # Pick the candidate with the lowest predicted latency.
        learned_idx = int(np.argmin(log_pred))
        learned_pick = test[learned_idx]
        emp_best = _empirical_best(test)
        static = _static_pick(test)
        learned_regret = (
            (learned_pick.latency_min_us - emp_best.latency_min_us)
            / emp_best.latency_min_us * 100.0
        )
        static_regret = (
            (static.latency_min_us - emp_best.latency_min_us)
            / emp_best.latency_min_us * 100.0
        )
        report.rows.append(CVRow(
            model_id=held,
            predicted_pick_id=learned_pick.candidate_id,
            predicted_pick_latency_us=learned_pick.latency_min_us,
            empirical_best_id=emp_best.candidate_id,
            empirical_best_latency_us=emp_best.latency_min_us,
            static_pick_id=static.candidate_id,
            static_pick_latency_us=static.latency_min_us,
            learned_regret_pct=learned_regret,
            static_regret_pct=static_regret,
        ))
    return report
