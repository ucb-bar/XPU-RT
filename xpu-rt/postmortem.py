"""Postmortem: scheduler predictions vs FireSim measured cycles.

Joins ``xpurt_trace.csv`` (per-dispatch actual mtime + predicted start/duration
in scheduler-cycle units) against the scheduler's ``SchedulerReport``.

The two streams live in DIFFERENT clock domains on this bitstream:
  - predicted_duration_ms is whatever unit the workload was built with
    (typically rdcycles at 1 GHz despite the ``_ms`` suffix — name is legacy);
  - actual_end_cycles - actual_start_cycles is mtime ticks (~30 MHz on
    GemminiAndOPUShuttleConfig FireSim, but the exact ratio is bitstream-dependent).

So instead of comparing raw cycles we compute a unit-neutral ratio
(actual_duration / predicted_duration) per dispatch and report deviations
from the median. The median ratio captures the bitstream-wide clock-domain
constant; per-dispatch deviations are real scheduler prediction error.

Usage:
    from postmortem import compare_trace
    out = compare_trace("xpurt_trace.csv", "scheduler_report.json")
    print(out["rms_error_pct"], out["top_outliers"][:3])
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Any, Dict, List, Optional


def _read_trace(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Drop blank rows produced by the harness.
            if not any(v.strip() for v in r.values() if isinstance(v, str)):
                continue
            try:
                actual = int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])
                predicted = float(r["predicted_duration_ms"])
            except (KeyError, ValueError):
                continue
            rows.append({
                "entry_id": int(r["entry_id"]),
                "network": r["network"],
                "instance": int(r["instance"]),
                "dispatch_id": int(r["dispatch_id"]),
                "op": r["op"],
                "name": r["name"],
                "core_kind": r["core_kind"],
                "predicted_duration": predicted,
                "actual_duration": float(actual),
            })
    return rows


def compare_trace(
    trace_csv: str,
    report_json: Optional[str] = None,
    *,
    write_to: Optional[str] = None,
    n_outliers: int = 10,
) -> Dict[str, Any]:
    """Compute predicted-vs-actual statistics from one (trace, report) pair.

    Args:
        trace_csv: Path to xpurt_trace.csv (ModelBlaster output).
        report_json: Path to scheduler_report.json from xpurt's
            ``SchedulerReport.write_json()``. Optional — without it, only
            the trace's own predicted vs actual columns are compared.
        write_to: If given, dump the result dict as JSON to this path.
        n_outliers: How many worst-deviation rows to include in
            ``top_outliers``.

    Returns:
        {
          "n_rows": int,
          "median_ratio": float,             # actual / predicted, median over all dispatches.
                                             # Equals the bitstream-wide clock-domain ratio
                                             # (≈ predicted_rate / actual_rate).
          "rms_error_pct": float,            # RMS of |ratio - median_ratio|/median_ratio * 100
          "p99_error_pct": float,
          "max_error_pct": float,
          "makespan_predicted": float,
          "makespan_actual": float,
          "report_makespan": Optional[float],
          "top_outliers": [
            {entry_id, dispatch_id, op, name, core_kind,
             predicted, actual, ratio, deviation_pct}, ...
          ],
        }
    """
    rows = _read_trace(trace_csv)
    if not rows:
        return {"n_rows": 0, "error": "no usable rows in trace"}

    # Per-row ratio (handles unit-domain mismatch automatically).
    for r in rows:
        if r["predicted_duration"] > 0:
            r["ratio"] = r["actual_duration"] / r["predicted_duration"]
        else:
            r["ratio"] = None

    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    if not ratios:
        return {"n_rows": len(rows), "error": "all predicted durations were 0"}

    median_ratio = statistics.median(ratios)

    # Per-row deviation in percent vs median ratio.
    deviations: List[float] = []
    for r in rows:
        if r["ratio"] is None or median_ratio == 0:
            r["deviation_pct"] = None
            continue
        dev_pct = abs(r["ratio"] - median_ratio) / median_ratio * 100.0
        r["deviation_pct"] = dev_pct
        deviations.append(dev_pct)

    rms = (sum(d * d for d in deviations) / len(deviations)) ** 0.5
    sorted_devs = sorted(deviations)
    p99_idx = max(0, int(len(sorted_devs) * 0.99) - 1)

    # Makespan = max actual_end_cycles, max(predicted_start + predicted_duration).
    makespan_actual = 0.0
    makespan_predicted = 0.0
    with open(trace_csv) as f:
        for r in csv.DictReader(f):
            try:
                makespan_actual = max(makespan_actual, float(r["actual_end_cycles"]))
                end_pred = float(r["predicted_start_ms"]) + float(r["predicted_duration_ms"])
                makespan_predicted = max(makespan_predicted, end_pred)
            except (KeyError, ValueError):
                continue

    report_makespan: Optional[float] = None
    if report_json and os.path.exists(report_json):
        try:
            with open(report_json) as f:
                rep = json.load(f)
            report_makespan = float(rep.get("makespan_cycles", 0.0)) or None
        except (json.JSONDecodeError, OSError):
            report_makespan = None

    top = sorted(
        (r for r in rows if r.get("deviation_pct") is not None),
        key=lambda x: x["deviation_pct"],
        reverse=True,
    )[:n_outliers]
    top_outliers = [
        {
            "entry_id": r["entry_id"],
            "dispatch_id": r["dispatch_id"],
            "op": r["op"],
            "name": r["name"],
            "core_kind": r["core_kind"],
            "predicted": r["predicted_duration"],
            "actual": r["actual_duration"],
            "ratio": r["ratio"],
            "deviation_pct": r["deviation_pct"],
        }
        for r in top
    ]

    result = {
        "n_rows": len(rows),
        "median_ratio": median_ratio,
        "rms_error_pct": rms,
        "p99_error_pct": sorted_devs[p99_idx] if sorted_devs else 0.0,
        "max_error_pct": max(deviations) if deviations else 0.0,
        "makespan_predicted": makespan_predicted,
        "makespan_actual": makespan_actual,
        "report_makespan": report_makespan,
        "top_outliers": top_outliers,
    }

    if write_to:
        os.makedirs(os.path.dirname(os.path.abspath(write_to)), exist_ok=True)
        with open(write_to, "w") as f:
            json.dump(result, f, indent=2)
    return result
