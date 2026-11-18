"""Scheduling feedback: measured / predicted duration per op.

A deliberately simple return edge from a board run to the scheduler. Every op
gets a multiplicative correction factor, 1.0 until it has been measured. The
scheduler plans with ``profiled_time * factor``; after each run the factors
are updated from what the board actually took.

KEY. ``(network, IR dispatch name, machine kind)``, e.g.
``("dronet", "dispatch_12", "cpu_p")``. Machine kind (the spec's ``cpu_p`` /
``cpu_e``) rather than an individual core: in singletons mode every core of
one kind is priced from the same profile, so one factor per kind gets every
instance's sample. Instances are folded together for the same reason -- the
factor corrects the op, not one execution of it.

ANCHORED TO THE RAW PROFILE, so repeated rounds do not compound. A round's
schedule was solved with ``predicted = raw * factor_used``; the sample that
round contributes is ``measured / raw = (measured / predicted) * factor_used``.
The factor is then the mean of those per-round samples. Example: round 0
measures 1.16x the profile -> factor 1.16. Round 1 is solved with 1.16x and
measures 1.17x the profile (1.0086x its own prediction) -> factor
mean(1.16, 1.17) = 1.165, not 1.16 * 1.0086 compounded on top.

Within one round the per-instance ratios are averaged (``WITHIN_RUN``); across
rounds the per-round samples are averaged (``ACROSS_RUNS``). Every round's
sample is kept in the file so the aggregation can change without re-running.

File format (JSON)::

    {"schema_version": 1,
     "rounds": ["round0", "round1"],
     "ops": {"dronet|dispatch_12|cpu_p": {
                "network": "dronet", "dispatch": "dispatch_12", "kind": "cpu_p",
                "samples": [1.16, 1.17], "n_instances": [30, 30],
                "factor": 1.165}}}
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable, Iterable, Optional

SCHEMA_VERSION = 1

#: Aggregation within one run (over instances) and across runs (over rounds).
WITHIN_RUN: Callable[[list[float]], float] = statistics.fmean
ACROSS_RUNS: Callable[[list[float]], float] = statistics.fmean

#: Profile values at or above this are "unsupported" sentinels / priced-out
#: options, not timings; scaling them would be meaningless.
_NOT_A_TIMING_MS = 1e5


def op_key(network: str, dispatch: str, kind: str) -> str:
    return f"{network}|{dispatch}|{kind.lower()}"


def machine_kind(hardware_target: str) -> str:
    """'CPU_P#3' -> 'cpu_p'; composite 'CPU_P#0+CPU_P#1' -> 'cpu_p'."""
    first = str(hardware_target).split("+")[0]
    return first.split("#")[0].lower()


def empty() -> dict:
    return {"schema_version": SCHEMA_VERSION, "rounds": [], "ops": {}}


def load(path: Optional[str | Path]) -> dict:
    """Factor file, or an empty one (every factor 1.0) when path is None."""
    if path is None:
        return empty()
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {doc.get('schema_version')!r}, "
                         f"expected {SCHEMA_VERSION}")
    return doc


def save(doc: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def factor(doc: dict, network: str, dispatch: str, kind: str) -> float:
    rec = doc.get("ops", {}).get(op_key(network, dispatch, kind))
    return float(rec["factor"]) if rec else 1.0


def update(doc: dict, observations: Iterable[dict], round_name: str) -> dict:
    """Fold one run's observations into a NEW factor document.

    Each observation: ``network, dispatch, kind, measured_ms, predicted_ms``
    where ``predicted_ms`` is what the schedule of THIS run planned (already
    scaled by the factor in ``doc``). Observations with a non-positive
    prediction or measurement are ignored. Ops not observed this round keep
    their previous record unchanged.
    """
    per_op: dict[str, list[float]] = {}
    meta: dict[str, tuple[str, str, str]] = {}
    for ob in observations:
        pred = float(ob["predicted_ms"])
        meas = float(ob["measured_ms"])
        if pred <= 0 or meas <= 0:
            continue
        net, disp, kind = ob["network"], ob["dispatch"], ob["kind"].lower()
        used = factor(doc, net, disp, kind)
        # measured / raw profile, where raw = predicted / factor_used
        r = (meas / pred) * used
        k = op_key(net, disp, kind)
        per_op.setdefault(k, []).append(r)
        meta[k] = (net, disp, kind)

    out = json.loads(json.dumps(doc))      # deep copy; never mutate the input
    out.setdefault("ops", {})
    out["rounds"] = list(out.get("rounds", [])) + [round_name]
    for k, rs in per_op.items():
        net, disp, kind = meta[k]
        rec = out["ops"].get(k) or {"network": net, "dispatch": disp, "kind": kind,
                                    "samples": [], "n_instances": []}
        rec["samples"] = list(rec["samples"]) + [WITHIN_RUN(rs)]
        rec["n_instances"] = list(rec.get("n_instances", [])) + [len(rs)]
        rec["factor"] = ACROSS_RUNS(rec["samples"])
        out["ops"][k] = rec
    return out


def apply(processing_times: dict[str, list[float]],
          networks: Iterable[str],
          combo_kinds: list[str],
          doc: dict) -> dict[str, int]:
    """Scale ``processing_times`` in place.

    ``processing_times`` is ``load_profiled_processing_times``'s first return:
    ``{"<network>_<dispatch>": [ms per machine combination]}``.
    ``combo_kinds[i]`` is the machine kind (``cpu_p``/``cpu_e``) of
    combination i. Returns counts for logging.
    """
    nets = sorted(networks, key=len, reverse=True)    # longest prefix wins
    n_ops = n_vals = 0
    if not doc.get("ops"):
        return {"ops_scaled": 0, "values_scaled": 0}
    for name, times in processing_times.items():
        net = next((n for n in nets if name.startswith(n + "_")), None)
        if net is None:
            continue
        disp = name[len(net) + 1:]
        touched = False
        for i, t in enumerate(times):
            if t is None or i >= len(combo_kinds) or float(t) >= _NOT_A_TIMING_MS:
                continue
            f = factor(doc, net, disp, combo_kinds[i])
            if f != 1.0:
                times[i] = float(t) * f
                n_vals += 1
                touched = True
        n_ops += touched
    return {"ops_scaled": n_ops, "values_scaled": n_vals}


def summarize(doc: dict) -> dict:
    fs = [r["factor"] for r in doc.get("ops", {}).values()]
    if not fs:
        return {"n_ops": 0}
    return {"n_ops": len(fs), "rounds": doc.get("rounds", []),
            "mean": statistics.fmean(fs), "min": min(fs), "max": max(fs)}
