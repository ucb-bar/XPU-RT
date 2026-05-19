"""Multiplicative per-backend contention model for the closed loop.

After each on-board execution of a MILP-produced schedule we know
the *predicted* time each backend's lane should have taken (sum of
the islands' solo measurements) and the *measured* lane wall time.
Their ratio is the contention factor for that backend in that
loading regime; the next round's MILP uses solo × factor as the
expected cost on the same backend.

The model is intentionally simple: one scalar per backend, EMA'd
across two rounds to dampen oscillation. We cap factors to a
configurable max (default 2.5×) so a transient DDR spike can't
drive the scheduler off a cliff. Per-round factors are persisted to
``contention.jsonl`` for the proof writer.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Mapping

CONTENTION_LOG = "contention.jsonl"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclasses.dataclass
class ContentionState:
    """Per-backend multiplicative contention factor with history."""

    factors: dict[str, float] = dataclasses.field(default_factory=dict)
    history: list[dict[str, float]] = dataclasses.field(default_factory=list)
    max_factor: float = 2.5
    ema_weight: float = 0.5     # 0.5 = average last two; 1.0 = no smoothing
    last_delta: dict[str, float] = dataclasses.field(default_factory=dict)

    def ensure(self, backends: list[str]) -> None:
        for b in backends:
            self.factors.setdefault(b, 1.0)

    def update(
        self,
        *,
        per_backend_predicted_us: Mapping[str, float],
        per_backend_measured_us: Mapping[str, float],
    ) -> dict[str, float]:
        """Compute new per-backend factor = measured / predicted."""
        new: dict[str, float] = dict(self.factors)
        delta: dict[str, float] = {}
        for b, pred in per_backend_predicted_us.items():
            meas = per_backend_measured_us.get(b)
            if meas is None or pred <= 0:
                continue
            raw = float(meas) / float(pred)
            raw = max(min(raw, self.max_factor), 1.0 / self.max_factor)
            # EMA against the previous factor for stability.
            prev = self.factors.get(b, 1.0)
            new[b] = self.ema_weight * raw + (1.0 - self.ema_weight) * prev
            delta[b] = abs(new[b] - prev) / max(prev, 1e-9)
        self.history.append(dict(new))
        self.last_delta = delta
        self.factors = new
        return new

    def apply(
        self,
        solo_latency_matrix: Mapping[str, Mapping[str, float | None]],
    ) -> dict[str, dict[str, float | None]]:
        """Return ``solo_latency_matrix`` scaled by the current factors."""
        out: dict[str, dict[str, float | None]] = {}
        for w, by_b in solo_latency_matrix.items():
            out[w] = {}
            for b, v in by_b.items():
                if v is None:
                    out[w][b] = None
                    continue
                out[w][b] = float(v) * float(self.factors.get(b, 1.0))
        return out

    def is_converged(self, *, tol: float = 0.05, min_rounds: int = 2) -> bool:
        """Δ < ``tol`` for every backend over the last ``min_rounds`` rounds."""
        if len(self.history) < min_rounds:
            return False
        # Check the last two rounds' max delta.
        recent = self.history[-min_rounds:]
        keys = set().union(*(r.keys() for r in recent))
        for k in keys:
            vals = [r.get(k) for r in recent if k in r]
            if len(vals) < min_rounds:
                return False
            base = sum(vals) / len(vals)
            if base <= 0:
                continue
            for v in vals:
                if abs(v - base) / base > tol:
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "factors": dict(self.factors),
            "history": [dict(h) for h in self.history],
            "last_delta": dict(self.last_delta),
            "max_factor": self.max_factor,
            "ema_weight": self.ema_weight,
        }


def per_backend_predicted_from_schedule(
    schedule: Mapping[str, Any],
) -> dict[str, float]:
    """Sum each backend lane's ``predicted_us`` from the MILP schedule."""
    out: dict[str, float] = {}
    for op in schedule.get("ops", []) or []:
        b = op.get("machine")
        if not b:
            continue
        out[b] = out.get(b, 0.0) + float(op.get("predicted_us") or 0.0)
    return out


def per_backend_measured_from_execution(
    execution: Mapping[str, Any],
) -> dict[str, float]:
    """Pull each backend's measured lane finish time from an ExecutionResult.

    ``execution`` is expected to be the dict produced by
    ``execute_schedule.execute_schedule`` — i.e. it has either
    ``lane_finish_ns[backend]`` or
    ``lane_finish_us[backend]``, or per-island timestamps that we
    aggregate.
    """
    if "lane_finish_us" in execution:
        return {str(k): float(v) for k, v in execution["lane_finish_us"].items()}
    if "lane_finish_ns" in execution:
        return {str(k): float(v) / 1_000.0
                for k, v in execution["lane_finish_ns"].items()}
    # Aggregate from per-island times.
    by_b: dict[str, float] = {}
    for r in execution.get("islands", []) or []:
        b = r.get("machine")
        if not b:
            continue
        end_us = r.get("end_us")
        if end_us is None and "end_ns" in r:
            end_us = float(r["end_ns"]) / 1_000.0
        if end_us is None:
            continue
        by_b[b] = max(by_b.get(b, 0.0), float(end_us))
    return by_b


def write_contention_log(
    run_dir: Path | str,
    *,
    round_index: int,
    state: ContentionState,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    p = Path(run_dir) / CONTENTION_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema_version": "qnn_contention_v1",
        "timestamp_utc": _now(),
        "round": round_index,
        "factors": dict(state.factors),
        "last_delta": dict(state.last_delta),
    }
    if extra:
        rec.update({"extra": dict(extra)})
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return p
