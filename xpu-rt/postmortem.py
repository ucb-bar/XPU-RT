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
    makespan_actual_raw = 0.0
    makespan_predicted = 0.0
    with open(trace_csv) as f:
        for r in csv.DictReader(f):
            try:
                makespan_actual_raw = max(makespan_actual_raw, float(r["actual_end_cycles"]))
                end_pred = float(r["predicted_start_ms"]) + float(r["predicted_duration_ms"])
                makespan_predicted = max(makespan_predicted, end_pred)
            except (KeyError, ValueError):
                continue

    # Normalize actual makespan into predicted's unit (ms). The trace's
    # actual_*_cycles columns store integer µs (legacy column name),
    # so the conversion factor is exactly 1000. Sanity-check against
    # median_ratio (should round to ~1000) so we'd notice if a future
    # bitstream changes the unit.
    actual_to_ms = 1000.0
    if median_ratio and abs(median_ratio - actual_to_ms) / actual_to_ms > 0.10:
        # > 10% off the expected 1000 — fall back to median_ratio.
        actual_to_ms = median_ratio
    makespan_actual = makespan_actual_raw / actual_to_ms
    makespan_delta_pct = (
        100.0 * (makespan_actual - makespan_predicted) / makespan_predicted
        if makespan_predicted else 0.0
    )

    report_makespan: Optional[float] = None
    if report_json and os.path.exists(report_json):
        try:
            with open(report_json) as f:
                rep = json.load(f)
            report_makespan = float(rep.get("makespan_cycles", 0.0)) or None
        except (json.JSONDecodeError, OSError):
            report_makespan = None

    # Filter top_outliers: rows with predicted_duration below the
    # trace's integer-µs floor are dominated by quantization noise
    # (actual gets rounded to 0 or 1 µs, producing a 100% deviation
    # signal that's not a real prediction error). We use 5×
    # (1/median_ratio) as the threshold — i.e. ops whose actual span
    # would land in ≥5 integer ticks. For a ~1000 ratio (µs vs ms)
    # that's 0.005 ms = 5 µs.
    resolution_floor_ms = (5.0 / median_ratio) if median_ratio else 0.0
    candidates = [
        r for r in rows
        if r.get("deviation_pct") is not None
        and r["predicted_duration"] >= resolution_floor_ms
    ]
    top = sorted(candidates, key=lambda x: x["deviation_pct"], reverse=True)[:n_outliers]
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

    # Recompute rms/p99 on the resolution-filtered population so the
    # headline numbers aren't dominated by floor-truncation noise.
    filtered_devs = [
        r["deviation_pct"] for r in rows
        if r.get("deviation_pct") is not None
        and r["predicted_duration"] >= resolution_floor_ms
    ]
    if filtered_devs:
        filtered_sorted = sorted(filtered_devs)
        f_rms = (sum(d * d for d in filtered_devs) / len(filtered_devs)) ** 0.5
        f_p99 = filtered_sorted[max(0, int(len(filtered_sorted) * 0.99) - 1)]
        f_max = max(filtered_devs)
    else:
        f_rms = f_p99 = f_max = 0.0

    result = {
        "n_rows": len(rows),
        "median_ratio": median_ratio,
        "resolution_floor_ms": resolution_floor_ms,
        "n_above_floor": len(filtered_devs),
        # rms/p99/max on rows above the trace's integer-µs floor.
        "rms_error_pct": f_rms,
        "p99_error_pct": f_p99,
        "max_error_pct": f_max,
        # rms/p99/max on ALL rows including sub-resolution noise.
        "raw_rms_error_pct": rms,
        "raw_p99_error_pct": sorted_devs[p99_idx] if sorted_devs else 0.0,
        "raw_max_error_pct": max(deviations) if deviations else 0.0,
        # Both makespans in ms now — directly comparable.
        "makespan_predicted_ms": makespan_predicted,
        "makespan_actual_ms": makespan_actual,
        "makespan_actual_raw": makespan_actual_raw,  # preserve for debug
        "makespan_delta_pct": makespan_delta_pct,
        "report_makespan": report_makespan,
        "top_outliers": top_outliers,
    }

    if write_to:
        os.makedirs(os.path.dirname(os.path.abspath(write_to)), exist_ok=True)
        with open(write_to, "w") as f:
            json.dump(result, f, indent=2)
    return result
