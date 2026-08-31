"""Turn a *predicted* schedule JSON into the same trace rows a measured run emits.

Why this exists
---------------
`trace_metrics.summarise_trace` is the one place in this repo that is allowed to
say what a periodic run achieved -- instance deadline misses, lateness, response
time from the nominal release k*T, achieved frequency, per-core utilization. It
reads a merlin dispatch-scheduler trace CSV.

A host-side policy sweep has no board and therefore no trace. The previous
answer to that was `k1_baselines.predicted()` (retired with the merlin
flow), which re-derived a
*fourth* mini-version of the instance collapse (end > i*T + T, counted per
instance, no rate, no response time, no utilization). So predicted schedules and
measured runs were scored by two different definitions and the numbers were
quietly incomparable.

This module removes the second definition instead of adding a third: it renders
the schedule into trace rows and lets `trace_metrics` do the arithmetic. The
only thing it must get right is the schema and the units.

Units, because this is where it goes wrong
------------------------------------------
`output_scheduled_json` writes ``start_time`` and ``duration`` in **ms**
(``metadata.makespan`` is ms too, despite `metrics.py` printing it under a
``makespan_us`` label). The trace schema is in **us**. Everything here
multiplies by 1000 exactly once.

What is honest and what is not
------------------------------
* ``run_us`` is the solver's own duration for the assigned core combination --
  the profile, not a measurement.
* ``queue_delay_us`` is the predicted gap between when a dispatch became ready
  (its release, or its last dependency finishing) and when the schedule starts
  it. On a measured trace this field is runtime queueing; here it is scheduler
  slack. Both are "time the work was runnable and not running", which is what
  ``queue_share_pct`` reports.
* ``observed_cpu_start``/``observed_cpu_end`` are NOT emitted: there is no
  observation, and emitting them would make `summarise_trace` report a
  migration count for a run that never happened.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

#: Field order of the merlin dispatch-scheduler trace CSV, restricted to the
#: columns a predicted schedule can honestly fill in.
TRACE_FIELDS = (
    "graph_iter", "dispatch_key", "dispatch_id", "ordinal", "total",
    "job_name", "module_name", "target", "cores",
    "planned_start_us", "ready_us", "start_us", "end_us",
    "queue_delay_us", "run_us", "total_latency_us",
)


def split_hardware_target(hardware_target: str) -> Tuple[str, List[str]]:
    """'CPU_P#0+CPU_P#1' -> ('CPU_P', ['0', '1']).

    A dispatch sharded across two *different* clusters has no single ``target``
    label, and `trace_metrics._held_cores` would silently attribute all of its
    cores to whichever cluster came first. Rather than invent an attribution,
    the cluster of the first core is returned together with every core id it
    holds; callers that care can compare against the raw string. Mixed-cluster
    combinations do not occur in any config in this repo -- machine
    combinations are built per machine kind -- so this is a guard, not a path.
    """
    parts = [p for p in str(hardware_target).split("+") if p]
    clusters, cores = [], []
    for p in parts:
        if "#" in p:
            cluster, core = p.split("#", 1)
        else:
            cluster, core = p, "0"
        clusters.append(cluster)
        cores.append(core)
    cluster = clusters[0] if clusters else "CPU"
    return cluster, cores


def trace_rows_from_schedule(schedule: dict) -> List[dict]:
    """Trace rows, in start order, for a schedule JSON from `output_scheduled_json`.

    Rows carry the same ``dispatch_key`` as the schedule's own dispatch names, so
    a renderer can join them back to ``dispatches[key]["hardware_target"]``
    exactly the way it joins a measured trace.
    """
    dispatches: Dict[str, dict] = schedule.get("dispatches") or {}
    end_ms: Dict[str, float] = {}
    for name, d in dispatches.items():
        end_ms[name] = float(d.get("start_time", 0.0)) + float(d.get("duration", 0.0))

    rows: List[dict] = []
    for name, d in dispatches.items():
        start_ms = float(d.get("start_time", 0.0))
        dur_ms = float(d.get("duration", 0.0))
        cluster, cores = split_hardware_target(d.get("hardware_target", "CPU#0"))

        # Ready = the later of the periodic release and the last dependency.
        # `release_us` is already in us (output_scheduled_json multiplies), so
        # it is the one field here that must NOT be scaled again.
        ready_us = float(d.get("release_us", 0.0) or 0.0)
        for dep in d.get("dependencies") or ():
            if dep in end_ms:
                ready_us = max(ready_us, end_ms[dep] * 1000.0)
        start_us = start_ms * 1000.0
        ready_us = min(ready_us, start_us)  # a dispatch cannot start before ready

        rows.append({
            "graph_iter": 0,
            "dispatch_key": name,
            "dispatch_id": d.get("id", ""),
            "ordinal": d.get("ordinal", 1),
            "total": d.get("total", 1),
            "job_name": d.get("job_name", ""),
            "module_name": d.get("module_name", ""),
            "target": cluster,
            "cores": "+".join(cores),
            "planned_start_us": round(start_us, 3),
            "ready_us": round(ready_us, 3),
            "start_us": round(start_us, 3),
            "end_us": round(start_us + dur_ms * 1000.0, 3),
            "queue_delay_us": round(max(0.0, start_us - ready_us), 3),
            "run_us": round(dur_ms * 1000.0, 3),
            "total_latency_us": round(start_us + dur_ms * 1000.0 - ready_us, 3),
        })
    rows.sort(key=lambda r: (r["start_us"], r["dispatch_key"]))
    return rows


def periods_ms(schedule: dict, known=None) -> Dict[str, float]:
    """`metadata.periodic_networks`, the schedule's own record of its periods.

    Taken from the schedule rather than hardcoded so a third model cannot be
    silently dropped from the scoring -- the same reason `trace_metrics` asks
    for the map instead of owning one.

    `known` REPAIRS a schedule written before network names stopped being
    trailing-digit-stripped. Such a schedule records `yolov8_nano_64x` for
    `yolov8_nano_64x96`, and that truncated key then propagates into the
    deadline scorer, where instance 0 is read as instance 960 and the model
    becomes structurally incapable of missing a deadline. Passing the real
    names (they are in the workload spec) maps each stored key back onto the
    network it meant. A key that is already correct is left alone, and a key
    matching no known name is kept as-is rather than guessed at.
    """
    md = schedule.get("metadata") or {}
    raw = {str(k): float(v) for k, v in (md.get("periodic_networks") or {}).items()}
    if not known:
        return raw
    known = set(known)
    out: Dict[str, float] = {}
    for key, period in raw.items():
        if key in known:
            out[key] = period
            continue
        # Longest match first: the stored key is a PREFIX of the real name.
        hits = sorted((n for n in known if n.startswith(key)),
                      key=len, reverse=True)
        out[hits[0] if hits else key] = period
    return out


def machines(schedule: dict) -> List[str]:
    md = schedule.get("metadata") or {}
    return [str(m) for m in (md.get("machines") or [])]


def standalone_service_us(schedule: dict) -> float:
    """Serial sum of every dispatch's duration, in us.

    This is the quantity the *old* accept/reject criterion used as its only
    term, which is why it is worth computing: `candidate_objective` wants it as
    the LAST tie-break, and reproducing it here keeps that term available
    without letting it back into the front of the order.
    """
    return sum(float(d.get("duration", 0.0))
               for d in (schedule.get("dispatches") or {}).values()) * 1000.0


def write_trace_csv(rows: Sequence[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(TRACE_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRACE_FIELDS})
    return path
