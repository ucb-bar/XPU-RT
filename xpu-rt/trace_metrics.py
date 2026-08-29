"""Measured real-time metrics from a merlin dispatch-scheduler trace CSV.

Why this module exists
----------------------
Three copies of "collapse trace rows into periodic instances" had grown up
independently -- in `xpu-rt/metrics.py` (predicted only), in
`scripts/k1_baselines.py` (measured, per instance, no rate), and in
`scripts/plot_k1_evolution.py` (measured, the only one that computed response
time from the nominal release k*T). They disagreed, and the disagreement was
invisible: `metrics.py` reported 160 deadline misses for the same run that
`k1_baselines.py` reported 10, because one counts dispatches and the other
counts job instances. Both numbers are defensible; publishing them under the
same name is not.

So this is the one place that reads a trace. It computes what a periodic
real-time experiment actually has to report, and it names the two miss
definitions differently so they can never be confused again.

Definitions, stated because they are the whole point
----------------------------------------------------
For a periodic job with period T, instance k is *released* at ``k*T`` and its
deadline is ``k*T + D`` (here D == T unless a window is given separately).

* **response time** = completion - k*T, measured from the nominal release, NOT
  from whenever the instance happened to start. A model does not meet its
  frequency by running several invocations back-to-back, and measuring from
  actual start would hide exactly that.
* **lateness** = completion - (k*T + D); positive means a miss.
* **instance miss rate** = missed instances / released instances. This is the
  headline number: "10 misses" reads like a near-miss when it is in fact
  10 out of 10, i.e. total failure on that model.
* **achieved frequency** = instances completed / observed span, in Hz, to be
  compared against 1000/T.

Utilization is per *core*, which requires the trace's ``cores`` column (the set
the runner actually held). Without it the best available attribution is the
cluster label in ``target``, and summing by that over-counts wildly -- a 4-core
cluster shows >100% -- so this module reports per-core utilization only when the
column is present and says so otherwise rather than printing a wrong number.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import job_names
import k1_trace
from typing import Dict, Iterable, List, Optional, Sequence


def model_of(job_name: str, known=None) -> str:
    """'dronet3' -> 'dronet'. The instance index is the numeric suffix.

    `known` is the set of real network names. Pass it whenever it is available:
    without it a network whose own name ends in a digit is split in the wrong
    place, and the consequence here is not cosmetic -- see `job_names`.
    """
    return job_names.model_of(job_name, known)


def instance_index(job_name: str, known=None) -> int:
    return job_names.instance_index(job_name, known)


def pct(xs: Sequence[float], p: float) -> float:
    """Linear-index percentile. Returns 0.0 for an empty sequence."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


#: `rdtime` on the K1. NOT the 1.6 GHz core clock and not 1 MHz -- the device
#: tree's `timebase-frequency` is 24000000, and every cycles->time conversion in
#: this project uses it.
K1_RDTIME_HZ = 24_000_000.0


def normalise_modelblaster(rows: List[dict]) -> List[dict]:
    """Map ModelBlaster's `harness_xpurt` trace onto this module's columns.

    Delegates to `k1_trace.normalise`, which owns the mapping for the whole
    repo -- it had three implementations and they disagreed about whether the
    trace's `dispatch_id` is a record slot or an IR id.

    `fill_queue_delay=False` is the one thing this caller needs differently:
    that producer measures no queueing, and inventing a 0 here would make
    `summarise_trace` report "no queueing" for a run where it was simply never
    measured. Absent stays absent, and `queue_us` comes out None.
    """
    return k1_trace.normalise(rows, fill_queue_delay=False)


def read_trace(path: str) -> List[dict]:
    return k1_trace.read(path, fill_queue_delay=False)


def _held_cores(row: dict) -> List[str]:
    """Cluster-qualified core ids this row occupied, e.g. ['CPU_P#0', ...].

    Returns [] when the trace predates the `cores` column, which is the signal
    to skip per-core utilization rather than approximate it.
    """
    raw = (row.get("cores") or "").strip()
    if not raw:
        return []
    target = (row.get("target") or "").strip() or "CPU"
    return [f"{target}#{c}" for c in raw.split("+") if c != ""]


def idle_intervals(busy: Iterable[tuple], span_end: float,
                   min_gap_us: float = 0.0) -> List[tuple]:
    """Gaps in a set of [start, end) intervals, over [0, span_end).

    Reported as intervals rather than a total because a scalar idle figure
    cannot distinguish one long stall from many short ones, and those call for
    different fixes -- the first is a dependency chain, the second is overhead.
    """
    ivs = sorted(busy)
    out: List[tuple] = []
    cursor = 0.0
    for s, e in ivs:
        if s > cursor + min_gap_us:
            out.append((cursor, s))
        cursor = max(cursor, e)
    if span_end > cursor + min_gap_us:
        out.append((cursor, span_end))
    return out


def summarise_trace(rows: Sequence[dict],
                    periods_ms: Dict[str, float],
                    windows_ms: Optional[Dict[str, float]] = None) -> dict:
    """Every measured metric the periodic experiment has to report.

    ``periods_ms`` maps a model base name to its period in ms; take it from the
    schedule's ``metadata.periodic_networks`` rather than hardcoding it, so a
    third model cannot be silently skipped.
    """
    if not rows:
        return {}
    windows_ms = windows_ms or {}
    # The period map's keys ARE the network names, so the split below needs no
    # extra argument -- provided the schedule that produced them was itself
    # written with un-stripped names (see `postprocessing.output_scheduled_json`).
    # For a schedule written before that fix the keys are already stripped, and
    # `job_names.split_job_name` then falls through to the same stripping, so
    # old artifacts keep scoring exactly as they did.
    known = set(periods_ms) | set(windows_ms)

    service_us = sum(float(r["run_us"]) for r in rows)
    # Not every producer measures queueing separately (ModelBlaster's harness
    # records only start/end cycles). Absent is reported as absent: a 0 here
    # would be indistinguishable from a run that genuinely never queued, and
    # the queue/service split is the field that decides whether a miss calls
    # for a faster kernel or an earlier start.
    have_queue = all((r.get("queue_delay_us") or "") != "" for r in rows)
    queue_us = (sum(float(r["queue_delay_us"]) for r in rows)
                if have_queue else None)
    makespan_us = max(float(r["end_us"]) for r in rows)

    # Per instance: span, and the cores it touched.
    spans: Dict[str, List[float]] = defaultdict(lambda: [1e18, -1e18])
    for r in rows:
        j = r["job_name"]
        spans[j][0] = min(spans[j][0], float(r["start_us"]) / 1000.0)
        spans[j][1] = max(spans[j][1], float(r["end_us"]) / 1000.0)

    per_model: Dict[str, dict] = {}
    for job, (st, en) in spans.items():
        m = model_of(job, known)
        T = periods_ms.get(m)
        if T is None:
            continue
        D = windows_ms.get(m, T)
        k = instance_index(job, known)
        d = per_model.setdefault(m, {
            "period_ms": T, "deadline_ms": D, "instances": 0,
            "misses": 0, "lateness_ms": [], "response_ms": [],
            "first_start_ms": 1e18, "last_end_ms": -1e18,
        })
        d["instances"] += 1
        d["response_ms"].append(en - k * T)
        d["first_start_ms"] = min(d["first_start_ms"], st)
        d["last_end_ms"] = max(d["last_end_ms"], en)
        late = en - (k * T + D)
        if late > 0:
            d["misses"] += 1
            d["lateness_ms"].append(late)

    models: Dict[str, dict] = {}
    for m, d in per_model.items():
        n = d["instances"]
        span_s = max(d["last_end_ms"] - 0.0, 1e-9) / 1000.0
        resp = d["response_ms"]
        models[m] = {
            "period_ms": d["period_ms"],
            "deadline_ms": d["deadline_ms"],
            "instances": n,
            "instance_deadline_misses": d["misses"],
            "instance_deadline_miss_rate_pct": round(100.0 * d["misses"] / n, 1) if n else 0.0,
            "worst_lateness_ms": round(max(d["lateness_ms"]), 2) if d["lateness_ms"] else 0.0,
            "median_lateness_ms": round(pct(d["lateness_ms"], 50), 2) if d["lateness_ms"] else 0.0,
            "response_p50_ms": round(pct(resp, 50), 2),
            "response_p90_ms": round(pct(resp, 90), 2),
            "response_p99_ms": round(pct(resp, 99), 2),
            "achieved_frequency_hz": round(n / span_s, 2) if span_s > 0 else 0.0,
            "required_frequency_hz": round(1000.0 / d["period_ms"], 2),
        }

    out = {
        "n_dispatches": len(rows),
        "service_us": round(service_us, 1),
        "queue_us": round(queue_us, 1) if have_queue else None,
        "queue_share_pct": (round(100 * queue_us / (service_us + queue_us), 1)
                            if have_queue and service_us + queue_us else
                            (0.0 if have_queue else None)),
        "makespan_us": round(makespan_us, 1),
        "periodic_instances": len(spans),
        # Kept for continuity with the earlier per-instance count, but named so
        # it cannot be mistaken for the per-op count metrics.py reports.
        "instance_deadline_misses": sum(m["instance_deadline_misses"]
                                        for m in models.values()),
        "per_model": models,
    }

    # Per-core and per-cluster utilization, only if the trace says which cores
    # were held. A sharded dispatch occupies every core in its set for its whole
    # duration, so its run_us counts once per core -- that is the physical
    # truth, and it is why utilization cannot be derived from run_us alone.
    busy_by_core: Dict[str, List[tuple]] = defaultdict(list)
    have_cores = False
    for r in rows:
        cores = _held_cores(r)
        if not cores:
            continue
        have_cores = True
        s, e = float(r["start_us"]), float(r["end_us"])
        for c in cores:
            busy_by_core[c].append((s, e))

    if have_cores:
        per_core, per_cluster_busy, idle = {}, defaultdict(float), {}
        for c, ivs in sorted(busy_by_core.items()):
            merged, cur_s, cur_e = [], None, None
            for s, e in sorted(ivs):
                if cur_s is None:
                    cur_s, cur_e = s, e
                elif s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            if cur_s is not None:
                merged.append((cur_s, cur_e))
            busy = sum(e - s for s, e in merged)
            per_core[c] = round(100.0 * busy / makespan_us, 1) if makespan_us else 0.0
            per_cluster_busy[c.split("#")[0]] += busy
            gaps = idle_intervals(merged, makespan_us)
            idle[c] = {
                "n_gaps": len(gaps),
                "total_idle_us": round(sum(e - s for s, e in gaps), 1),
                "longest_gap_us": round(max((e - s for s, e in gaps), default=0.0), 1),
            }
        n_by_cluster: Dict[str, int] = defaultdict(int)
        for c in per_core:
            n_by_cluster[c.split("#")[0]] += 1
        out["per_core_utilization_pct"] = per_core
        out["per_cluster_utilization_pct"] = {
            k: round(100.0 * v / (makespan_us * n_by_cluster[k]), 1)
            for k, v in per_cluster_busy.items() if makespan_us and n_by_cluster[k]
        }
        out["idle_intervals"] = idle
    else:
        out["per_core_utilization_pct"] = None
        out["utilization_note"] = (
            "trace has no 'cores' column, so per-core utilization is not "
            "derivable; attributing run_us to the CPU_P/CPU_E label instead "
            "over-counts a multi-core cluster past 100%"
        )

    # Affinity check, free once observed_cpu is in the trace.
    migrated = sum(1 for r in rows
                   if (r.get("observed_cpu_start") or "-1") != "-1"
                   and r.get("observed_cpu_start") != r.get("observed_cpu_end"))
    if any(r.get("observed_cpu_start") for r in rows):
        out["dispatches_that_migrated_mid_run"] = migrated
    return out


def format_summary(tag: str, s: dict) -> str:
    """One compact block per rung, for stdout."""
    if not s:
        return f"{tag}: empty trace"
    lines = [f"{tag}: {s['n_dispatches']} dispatches  "
             f"makespan={s['makespan_us']/1000:.1f} ms  "
             f"queue={s['queue_share_pct']}%"]
    if s.get("dispatches_that_migrated_mid_run") is not None:
        lines[0] += f"  migrated={s['dispatches_that_migrated_mid_run']}"
    for m, d in sorted(s.get("per_model", {}).items()):
        lines.append(
            f"   {m:8s} {d['instances']:3d} inst  "
            f"miss {d['instance_deadline_misses']:3d}"
            f" ({d['instance_deadline_miss_rate_pct']:5.1f}%)  "
            f"worst_late {d['worst_lateness_ms']:8.2f} ms  "
            f"resp p50/p90/p99 {d['response_p50_ms']:7.2f}/"
            f"{d['response_p90_ms']:7.2f}/{d['response_p99_ms']:7.2f} ms  "
            f"freq {d['achieved_frequency_hz']:6.2f}/"
            f"{d['required_frequency_hz']:.0f} Hz")
    util = s.get("per_core_utilization_pct")
    if util:
        lines.append("   cores: " + "  ".join(f"{k}={v}%" for k, v in util.items()))
        lines.append("   clusters: " + "  ".join(
            f"{k}={v}%" for k, v in s.get("per_cluster_utilization_pct", {}).items()))
    elif s.get("utilization_note"):
        lines.append("   utilization: " + s["utilization_note"])
    return "\n".join(lines)
