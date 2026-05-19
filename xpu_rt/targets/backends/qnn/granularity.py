"""Decision logic that turns measurements into granularity proposals.

After each round, the heterogeneous loop has produced three artefacts:

* ``schedule.json`` — what the scheduler thought would run when, with
  ``makespan_us`` and per-dispatch ``start_us`` / ``finish_us`` /
  predicted ``mean_us`` per backend.
* ``profiled_manifest.json`` — what the board actually measured for
  the *same* dispatches (per (canonical, target) ``mean_us``).
* ``graph_dossier_v3.json`` — region-level facts (reuse, working-set
  size, cost share, qparam metadata) that the candidate-generator
  uses to know where it's safe to split.

This module turns those into two lists:

* ``compute_split_candidates`` — dispatches that ran a lot slower
  than predicted and carry a meaningful share of the makespan. These
  are good candidates for being split into finer islands (so the
  scheduler can move pieces of them onto a less-loaded backend).
* ``compute_coarsen_candidates`` — adjacent islands on the same
  backend whose cross-boundary transfer cost dominates their
  combined execute time. These should be fused into a single
  coarser island.

Both functions are pure: they consume dicts, produce dicts. The MCP
tool ``xpu_rt_qnn_decide_granularity`` wraps them and bundles the
output into the existing ``agent_decision_request`` envelope.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

SPLIT_MEASURED_PREDICTED_RATIO = 1.30
SPLIT_REGION_SHARE_OF_MAKESPAN = 0.10
COARSEN_TRANSFER_RATIO = 0.20


@dataclasses.dataclass(frozen=True)
class SplitCandidate:
    """A dispatch the scheduler should consider splitting."""

    dispatch_id: str
    region_id: str | None
    machine: str
    predicted_us: float
    measured_us: float
    ratio: float
    region_share: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CoarsenCandidate:
    """A pair of same-backend islands the scheduler should fuse."""

    first_dispatch_id: str
    second_dispatch_id: str
    machine: str
    combined_compute_us: float
    transfer_us: float
    ratio: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _measured_us_for(profile: Mapping[str, Any], dispatch_id: str, target: str) -> float | None:
    """Look up a ``profiled_manifest`` cell.

    Tolerates several manifest shapes that have appeared in the repo
    so the loop is robust against schema drift:

    1. ``{"dispatches": {<name>: {<target>: {"mean_us": x}}}}``
    2. ``{"results": [{"name": ..., "target": ..., "mean_us": ...}]}``
    """
    if isinstance(profile, Mapping):
        d = profile.get("dispatches")
        if isinstance(d, Mapping):
            cell = d.get(dispatch_id, {})
            if isinstance(cell, Mapping):
                row = cell.get(target) or cell.get(target.lower())
                if isinstance(row, Mapping) and row.get("mean_us") is not None:
                    return float(row["mean_us"])
        for entry in profile.get("results") or []:
            if not isinstance(entry, Mapping):
                continue
            if (
                entry.get("name") == dispatch_id
                and entry.get("target") == target
                and entry.get("mean_us") is not None
            ):
                return float(entry["mean_us"])
    return None


def _machine_to_target(m: str) -> str:
    return {"CPU": "cpu", "GPU": "qnn_gpu", "HTA": "qnn_hta"}.get(m, m.lower())


def _region_for_dispatch(dossier: Mapping[str, Any] | None, dispatch_id: str) -> dict | None:
    if not isinstance(dossier, Mapping):
        return None
    regions = dossier.get("regions") or dossier.get("region_dossiers") or []
    if not isinstance(regions, list):
        return None
    for r in regions:
        if not isinstance(r, Mapping):
            continue
        dispatches = r.get("dispatches") or r.get("dispatch_ids") or []
        if isinstance(dispatches, list) and dispatch_id in dispatches:
            return dict(r)
        if r.get("region_id") == dispatch_id or r.get("id") == dispatch_id:
            return dict(r)
    return None


def _iter_schedule_ops(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    ops = schedule.get("ops")
    if isinstance(ops, list):
        return [dict(o) for o in ops if isinstance(o, Mapping)]
    out: list[dict[str, Any]] = []
    dispatches = schedule.get("dispatches")
    if isinstance(dispatches, Mapping):
        for name, row in dispatches.items():
            d = dict(row) if isinstance(row, Mapping) else {}
            d.setdefault("name", name)
            out.append(d)
    out.sort(key=lambda o: float(o.get("start_us", 0.0)))
    return out


def compute_split_candidates(
    *,
    dossier: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
    schedule: Mapping[str, Any],
    measured_predicted_ratio: float = SPLIT_MEASURED_PREDICTED_RATIO,
    region_share_threshold: float = SPLIT_REGION_SHARE_OF_MAKESPAN,
) -> list[SplitCandidate]:
    """Return dispatches that should be split into finer islands."""
    makespan = float(schedule.get("makespan_us", 0.0)) or 1.0
    out: list[SplitCandidate] = []
    for op in _iter_schedule_ops(schedule):
        name = str(op.get("name") or op.get("id") or "")
        if not name:
            continue
        machine = str(op.get("machine") or op.get("hardware_target") or "CPU")
        predicted = float(op.get("predicted_us") or op.get("processing_us")
                           or (float(op.get("finish_us", 0.0)) - float(op.get("start_us", 0.0))))
        if predicted <= 0.0:
            continue
        measured = _measured_us_for(profile, name, _machine_to_target(machine))
        if measured is None:
            continue
        ratio = measured / predicted
        share = predicted / makespan
        if ratio < measured_predicted_ratio:
            continue
        if share < region_share_threshold:
            continue
        region = _region_for_dispatch(dossier, name)
        rationale = (
            f"measured/predicted = {ratio:.2f} (>{measured_predicted_ratio:.2f}) "
            f"AND predicted share = {share*100:.1f}% of makespan (>{region_share_threshold*100:.0f}%)"
        )
        out.append(SplitCandidate(
            dispatch_id=name,
            region_id=(region or {}).get("region_id"),
            machine=machine,
            predicted_us=predicted,
            measured_us=measured,
            ratio=ratio,
            region_share=share,
            rationale=rationale,
        ))
    out.sort(key=lambda c: c.ratio * c.region_share, reverse=True)
    return out


def compute_coarsen_candidates(
    *,
    schedule: Mapping[str, Any],
    transfer_ratio_threshold: float = COARSEN_TRANSFER_RATIO,
) -> list[CoarsenCandidate]:
    """Return adjacent same-backend pairs that should be fused."""
    ops = _iter_schedule_ops(schedule)
    out: list[CoarsenCandidate] = []
    for prev, nxt in zip(ops, ops[1:]):
        pm = str(prev.get("machine") or prev.get("hardware_target") or "?")
        nm = str(nxt.get("machine") or nxt.get("hardware_target") or "?")
        if pm != nm:
            continue
        if str(nxt.get("dependencies", [])) and prev.get("name") not in (
            nxt.get("dependencies") or []
        ):
            # nxt does not actually consume prev; skip.
            pass
        prev_compute = float(prev.get("finish_us", 0.0)) - float(prev.get("start_us", 0.0))
        nxt_compute = float(nxt.get("finish_us", 0.0)) - float(nxt.get("start_us", 0.0))
        combined = max(0.0, prev_compute + nxt_compute)
        if combined <= 0:
            continue
        transfer = float(prev.get("transfer_to_next_us") or nxt.get("transfer_from_prev_us") or 0.0)
        if transfer <= 0.0:
            # Heuristic when the scheduler did not record an explicit
            # transfer cost: use the gap between finish_us and start_us
            # of the next op on the same backend.
            transfer = max(0.0, float(nxt.get("start_us", 0.0)) - float(prev.get("finish_us", 0.0)))
        ratio = transfer / combined
        if ratio < transfer_ratio_threshold:
            continue
        rationale = (
            f"adjacent {pm} pair "
            f"({prev.get('name')} → {nxt.get('name')}): "
            f"transfer={transfer:.1f}µs / combined-compute={combined:.1f}µs "
            f"= {ratio*100:.0f}%"
        )
        out.append(CoarsenCandidate(
            first_dispatch_id=str(prev.get("name")),
            second_dispatch_id=str(nxt.get("name")),
            machine=pm,
            combined_compute_us=combined,
            transfer_us=transfer,
            ratio=ratio,
            rationale=rationale,
        ))
    out.sort(key=lambda c: c.ratio, reverse=True)
    return out


def predicted_vs_measured_table(
    *,
    profile: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Long-form rows the dashboard's deltas panel renders verbatim."""
    rows: list[dict[str, Any]] = []
    for op in _iter_schedule_ops(schedule):
        name = str(op.get("name") or "")
        if not name:
            continue
        machine = str(op.get("machine") or op.get("hardware_target") or "CPU")
        predicted = float(op.get("predicted_us") or op.get("processing_us")
                           or (float(op.get("finish_us", 0.0)) - float(op.get("start_us", 0.0))))
        measured = _measured_us_for(profile, name, _machine_to_target(machine))
        delta = (measured - predicted) if measured is not None else None
        ratio = (measured / predicted) if (measured is not None and predicted > 0) else None
        rows.append({
            "dispatch": name,
            "machine": machine,
            "predicted_us": predicted,
            "measured_us": measured,
            "delta_us": delta,
            "ratio": ratio,
        })
    rows.sort(key=lambda r: (r.get("ratio") or 0.0), reverse=True)
    return rows
