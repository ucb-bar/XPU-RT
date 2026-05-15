"""Hint derivation: turn solver state into per-dispatch feedback for Merlin.

Pure functions over the scheduler's outputs (t, alpha, workload, solver_state).
No MOSEK or board dependency — unit-testable from a hand-built Workload.

Hint vocabulary (closed set, target-agnostic):
  prefer_coarser          — slack available, reduce overhead
  prefer_finer            — gaps in the schedule suggest unexploited parallelism
  consider_fuse_with_pred — cross-cluster transfer dominates the duration
  pin_target=<name>       — this op runs much faster on a specific combination
  consider_split_backend  — current target is the slow side; revisit elsewhere

The output JSON schema is documented in docs/merlin_integration.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Thresholds. Tuned to err on the side of fewer hints — Merlin should
# only see a hint when the signal is unambiguous. Adjust per workload via
# kwargs if needed.
_TRANSFER_RATIO_FUSE = 1.5      # cost_by_pred / nominal cost above this → fuse hint
_IDLE_FRACTION_FINER = 0.30     # idle gap before op / op duration above this → finer
_SLACK_RATIO_COARSER = 0.50     # deadline_slack / deadline above this → coarser
_PIN_SPEEDUP_RATIO = 2.0        # min duration / current duration ≤ 1/this → pin


def _assigned_combination(alpha_row: np.ndarray) -> int:
    """argmax over a one-hot row; returns -1 if all zeros / NaN."""
    if alpha_row is None:
        return -1
    finite = np.where(np.isfinite(alpha_row), alpha_row, -np.inf)
    if not np.any(np.isfinite(finite)) or float(np.max(finite)) <= 0.0:
        return -1
    return int(np.argmax(finite))


def _machines_label(workload, k: int) -> str:
    """Human-readable label for combination index k."""
    if k < 0:
        return "unknown"
    combo = workload.get_machine_combinations()[k]
    return "+".join(combo)


def _op_duration(op, k: int) -> float:
    if k < 0 or k >= len(op.processing_times):
        return 0.0
    d = op.processing_times[k]
    if d is None or not math.isfinite(float(d)):
        return 0.0
    return float(d)


def _pred_aware_duration(op, k_pred: int, k_curr: int) -> Optional[float]:
    """Returns the predecessor-aware cost if recorded, else None."""
    if not op.processing_times_by_pred:
        return None
    val = op.processing_times_by_pred.get((k_pred, k_curr))
    if val is None or not math.isfinite(float(val)):
        return None
    return float(val)


def _per_combination_idle_fraction(workload, t: np.ndarray, alpha: np.ndarray
                                   ) -> dict[int, float]:
    """Total idle time on each combination / makespan-on-combination.

    Sweeps ops sorted by start time per combination and sums the gaps.
    """
    n_ops = len(workload.operations)
    n_k = len(workload.get_machine_combinations())
    out: dict[int, float] = {}
    for k in range(n_k):
        windows: list[tuple[float, float]] = []
        for i in range(n_ops):
            ai = _assigned_combination(alpha[i])
            if ai != k:
                continue
            d = _op_duration(workload.operations[i], k)
            windows.append((float(t[i]), float(t[i]) + d))
        if not windows:
            continue
        windows.sort()
        first = windows[0][0]
        last = windows[-1][1]
        busy = sum(b - a for a, b in windows)
        span = last - first
        if span <= 0:
            out[k] = 0.0
        else:
            out[k] = max(0.0, 1.0 - (busy / span))
    return out


def _per_op_idle_gap(workload, t: np.ndarray, alpha: np.ndarray
                     ) -> dict[int, float]:
    """For each op, the idle gap on its combination immediately preceding it.

    Defined as: start_time - end_time_of_prev_op_on_same_combination, or
    start_time itself if the op is first on its combination. Negative
    values clamp to 0.
    """
    n_ops = len(workload.operations)
    n_k = len(workload.get_machine_combinations())
    # Group ops per combination, sorted by start time
    per_combo: dict[int, list[int]] = {}
    for i in range(n_ops):
        k = _assigned_combination(alpha[i])
        if k < 0:
            continue
        per_combo.setdefault(k, []).append(i)
    for k in per_combo:
        per_combo[k].sort(key=lambda i: float(t[i]))

    gaps: dict[int, float] = {}
    for k, ops in per_combo.items():
        prev_end = 0.0
        for i in ops:
            d = _op_duration(workload.operations[i], k)
            gap = max(0.0, float(t[i]) - prev_end)
            gaps[i] = gap
            prev_end = float(t[i]) + d
    return gaps


def derive_dispatch_hints(workload,
                          t: np.ndarray,
                          alpha: np.ndarray,
                          *,
                          run_id: Optional[str] = None,
                          source_schedule: Optional[str] = None,
                          ) -> dict[str, Any]:
    """Build the xpurt_feedback.json payload from scheduler outputs.

    Args:
      workload: the original Workload (post-scheduling — solver_state and
                skipped_op_indices are read off it as side-channels).
      t:        per-op start times (length n_ops).
      alpha:    per-op machine-combination assignments (n_ops × n_combinations).
      run_id:   optional caller-supplied run id; defaults to a UTC timestamp.
      source_schedule: optional path string for cross-referencing.

    Returns:
      A dict matching the schema in docs/merlin_integration.md.
    """
    if t is None or alpha is None:
        return {
            "schema_version": 1,
            "run_id": run_id or _default_run_id(),
            "source_schedule": source_schedule,
            "model_signals": {"solver_failed": True},
            "dispatches": {},
        }

    t = np.asarray(t)
    alpha = np.asarray(alpha)

    n_ops = len(workload.operations)
    n_k = len(workload.get_machine_combinations())

    solver_state = getattr(workload, "solver_state", {}) or {}
    skipped_indices = set(getattr(workload, "skipped_op_indices", []) or [])

    per_combo_idle = _per_combination_idle_fraction(workload, t, alpha)
    per_op_gap = _per_op_idle_gap(workload, t, alpha)

    # Model-level signals.
    makespan = solver_state.get("makespan")
    if makespan is None and n_ops > 0:
        ends = []
        for i in range(n_ops):
            k = _assigned_combination(alpha[i])
            if k >= 0:
                ends.append(float(t[i]) + _op_duration(workload.operations[i], k))
        makespan = max(ends) if ends else 0.0

    total_busy = 0.0
    for i in range(n_ops):
        k = _assigned_combination(alpha[i])
        if k >= 0 and i not in skipped_indices:
            total_busy += _op_duration(workload.operations[i], k)
    makespan_efficiency = (
        (total_busy / (makespan * max(1, n_k))) if makespan and makespan > 0 else None
    )

    skip_triggered = []
    for i in skipped_indices:
        if 0 <= i < n_ops:
            skip_triggered.append(_dispatch_id(workload.operations[i], i))

    deadline_met = True
    for i in range(n_ops):
        op = workload.operations[i]
        if op.deadline_us is None:
            continue
        if i in skipped_indices:
            deadline_met = False
            continue
        k = _assigned_combination(alpha[i])
        end = float(t[i]) + _op_duration(op, k)
        if end > float(op.deadline_us) + 1e-6:
            deadline_met = False

    model_signals = {
        "makespan": float(makespan) if makespan is not None else None,
        "makespan_efficiency": (
            float(makespan_efficiency) if makespan_efficiency is not None else None
        ),
        "deadline_met": bool(deadline_met),
        "skip_triggered": skip_triggered,
        "problem_status": solver_state.get("problem_status"),
        "fusion_applied": bool(solver_state.get("fusion_applied", False)),
    }

    # Per-dispatch hints.
    dispatches: dict[str, dict[str, Any]] = {}
    for i in range(n_ops):
        op = workload.operations[i]
        k = _assigned_combination(alpha[i])
        if k < 0:
            continue
        d = _op_duration(op, k)

        # Predecessor-aware cost ratio: the duration we paid divided by
        # the same combination's nominal duration. Only meaningful if the
        # op has predecessor-aware costs and at least one predecessor.
        transfer_ratio = 1.0
        pred_target = None
        preds = op.get_predecessors() or []
        if preds and op.processing_times_by_pred:
            # Find the predecessor whose combination assignment, paired
            # with this op's k, gives a recorded entry.
            best_ratio = 1.0
            best_pred_target = None
            for p in preds:
                try:
                    pi = workload.operations.index(p)
                except ValueError:
                    continue
                k_pred = _assigned_combination(alpha[pi])
                if k_pred < 0:
                    continue
                eff = _pred_aware_duration(op, k_pred, k)
                if eff is None or d <= 0:
                    continue
                ratio = eff / d
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pred_target = _machines_label(workload, k_pred)
            transfer_ratio = best_ratio
            pred_target = best_pred_target

        idle_gap = float(per_op_gap.get(i, 0.0))
        idle_fraction = (idle_gap / d) if d > 0 else 0.0

        deadline_slack: Optional[float] = None
        if op.deadline_us is not None:
            deadline_slack = float(op.deadline_us) - (float(t[i]) + d)

        # Pin signal: is there a much faster combination available?
        durations_per_k = [_op_duration(op, kk) for kk in range(n_k)]
        valid = [(kk, dd) for kk, dd in enumerate(durations_per_k) if dd > 0]
        pin_hint = None
        if valid and d > 0:
            best_k, best_d = min(valid, key=lambda kv: kv[1])
            if best_k != k and best_d > 0 and (d / best_d) >= _PIN_SPEEDUP_RATIO:
                pin_hint = f"pin_target={_machines_label(workload, best_k)}"

        hints: list[str] = []
        rationale_parts: list[str] = []

        if i in skipped_indices:
            hints.append("prefer_finer")
            rationale_parts.append("op was dropped via skip indicator")

        if transfer_ratio >= _TRANSFER_RATIO_FUSE:
            hints.append("consider_fuse_with_pred")
            rationale_parts.append(
                f"cross-cluster cost ratio {transfer_ratio:.2f} (pred on "
                f"{pred_target})"
            )

        if idle_fraction >= _IDLE_FRACTION_FINER:
            if "prefer_finer" not in hints:
                hints.append("prefer_finer")
            rationale_parts.append(
                f"idle gap before op = {idle_fraction:.2f}× duration"
            )

        if (op.deadline_us is not None and deadline_slack is not None
                and op.deadline_us > 0
                and (deadline_slack / float(op.deadline_us)) >= _SLACK_RATIO_COARSER
                and transfer_ratio < _TRANSFER_RATIO_FUSE
                and "prefer_finer" not in hints):
            # Semantic opposite of prefer_finer; only fire when no idle-gap
            # / skip / miss signal already asked for finer.
            hints.append("prefer_coarser")
            rationale_parts.append(
                f"deadline slack {deadline_slack:.0f}us "
                f"({deadline_slack / float(op.deadline_us):.0%})"
            )

        if pin_hint is not None:
            hints.append(pin_hint)
            rationale_parts.append(
                f"current combo is ≥{_PIN_SPEEDUP_RATIO}× slower than best"
            )

        if not hints:
            continue  # keep payload sparse; absence ≡ no opinion

        dispatch_id = _dispatch_id(op, i)
        dispatches[dispatch_id] = {
            "current_target": _machines_label(workload, k),
            "idle_fraction": round(idle_fraction, 4),
            "transfer_cost_ratio": round(transfer_ratio, 4),
            "deadline_slack_us": (
                round(deadline_slack, 3) if deadline_slack is not None else None
            ),
            "hints": hints,
            "rationale": "; ".join(rationale_parts),
        }

    return {
        "schema_version": 1,
        "run_id": run_id or _default_run_id(),
        "source_schedule": source_schedule,
        "model_signals": model_signals,
        "dispatches": dispatches,
    }


def _dispatch_id(op, fallback_idx: int) -> str:
    """Stable identifier for an operation. Prefers operation_name (which
    merlin_adapter sets to the merlin dispatch name), then operation_id,
    then a synthetic op_<idx> string.
    """
    name = getattr(op, "operation_name", None)
    if name:
        return str(name)
    oid = getattr(op, "operation_id", None)
    if oid is not None:
        return str(oid)
    return f"op_{fallback_idx}"


def _default_run_id() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("xpurt_%Y%m%dT%H%M%SZ")


def write_feedback_json(payload: dict[str, Any], out_path: Path) -> None:
    """Write payload to disk with stable key ordering for diff-friendliness."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
