"""Render predicted-vs-actual Gantt charts from an xpurt_trace.csv.

The trace file (written by ModelBlaster's xpurt_main runtime) records per
dispatch:
  - predicted_start_ms, predicted_duration_ms  — scheduler-emitted, in
    workload time units (typically rdcycles at 1 GHz despite the _ms suffix).
  - actual_start_cycles, actual_end_cycles     — measured at runtime, in
    target mtime ticks (typically 1 MHz on this bitstream).

We normalize both streams to milliseconds for a single time axis. The
clock-domain ratio is taken from the median of (actual / predicted) over
all dispatches; that ratio is the bitstream-wide constant, NOT scheduler
error. Per-dispatch deviation from this ratio is the real prediction error.

Output:
  - Two stacked Gantt panels, one per (predicted, actual). Y-axis groups
    by hardware target (CPU_P, CPU_E). Bars colored by core_kind.

Usage:
    python -m xpurt.plot_gantt --trace <path/xpurt_trace.csv> --out gantt.png
    python -m xpurt.plot_gantt --trace ... --out gantt.png --max-rows 30
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from typing import Optional


def _read_trace(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not any(v.strip() for v in r.values() if isinstance(v, str)):
                continue
            try:
                rows.append({
                    "entry_id": int(r["entry_id"]),
                    "network": r["network"],
                    "instance": int(r["instance"]),
                    "dispatch_id": int(r["dispatch_id"]),
                    "op": r["op"],
                    "name": r["name"],
                    "core_kind": r["core_kind"],
                    "hart": int(r["hart"]),
                    "predicted_start": float(r["predicted_start_ms"]),
                    "predicted_duration": float(r["predicted_duration_ms"]),
                    "actual_start": float(r["actual_start_cycles"]),
                    "actual_end": float(r["actual_end_cycles"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def _palette() -> dict:
    """Core-kind palette (legacy; render_fixture_gantt overrides with
    network colors)."""
    return {
        "gemmini":  "#3b82f6",
        "rvv_opu":  "#ef4444",
        "rvv":      "#10b981",
        "scalar":   "#94a3b8",
    }


def _network_palette() -> dict:
    """Color map for the per-network bar colors used on both predicted
    and actual Gantts. The bars on the same time axis can then be read
    by network at a glance — the lane (y-axis position) still encodes
    the (core_kind, hart) target."""
    return {
        "yolov8_nano":     "#3b82f6",  # blue
        "yolov8_nano_64":  "#1e40af",  # darker blue — small yolov8 variant
        "dronet":          "#ef4444",  # red
        "mlp_control":     "#10b981",  # green
    }


def _network_root(name: str) -> str:
    """Strip any trailing instance index from a network name.
    'yolov8_nano_64' stays as-is (longest-prefix wins for known names).
    """
    for prefix in ("yolov8_nano_64", "yolov8_nano", "yolov8", "dronet", "mlp_control"):
        if name.startswith(prefix):
            return prefix
    return name


def render_gantt(trace_csv: str, out_path: str,
                 max_rows: Optional[int] = None,
                 title: Optional[str] = None) -> dict:
    """Build the side-by-side predicted/actual Gantt PNG.

    Returns a small dict of summary stats (for printing): n_rows, scale_ratio,
    predicted_makespan_ms, actual_makespan_ms.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows = _read_trace(trace_csv)
    if not rows:
        raise RuntimeError(f"no usable rows in {trace_csv}")

    # Convert each stream to absolute milliseconds using its native unit:
    #
    #   predicted_*_ms column (fixture-emitter dependent):
    #     - MOSEK bridge writes rdcycles at 1 GHz   → divide by 1e6
    #     - gen_3way_schedule writes ms directly    → divide by 1
    #   actual_*_cycles column: always Zephyr k_cycle_get_64() = mtime, which
    #     is CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=1000000 on FireSim → divide by 1e3
    #
    # We deliberately do NOT force the makespans to coincide — if the actual
    # is 30× the predicted, the chart should show it. That's the whole point.
    raw_max_pred = max(r["predicted_start"] + r["predicted_duration"] for r in rows)
    PRED_PER_MS = 1_000_000.0 if raw_max_pred > 10_000 else 1.0
    ACTUAL_PER_MS = 1_000.0  # mtime @ 1 MHz on this bitstream
    # Keep `scale` as the dispatch-shape ratio for the printed summary.
    scale = (max(r["actual_end"] for r in rows) / raw_max_pred) if raw_max_pred > 0 else 1.0

    if max_rows:
        rows = rows[:max_rows]

    pred_makespan = raw_max_pred / PRED_PER_MS
    actual_makespan = max(r["actual_end"] for r in rows) / ACTUAL_PER_MS

    # Group by (network, instance, core_kind/hart) → vertical lane.
    lane_keys: list[tuple] = []
    lane_for_row: list[int] = []
    seen: dict[tuple, int] = {}
    for r in rows:
        # Lane key: hardware target + hart number. Cleanest layout
        # without exploding lane count when there are many instances.
        k = (r["core_kind"], r["hart"])
        if k not in seen:
            seen[k] = len(lane_keys)
            lane_keys.append(k)
        lane_for_row.append(seen[k])

    # Color bars by network (yolov8 / dronet / mlp_control), not by
    # core_kind — the lane on the y-axis already encodes the target tile.
    palette = _network_palette()

    fig, (ax_p, ax_a) = plt.subplots(2, 1, figsize=(14, max(4, 0.6 * len(lane_keys) + 4)),
                                     sharex=True)
    BAR_H = 0.7

    for r, lane in zip(rows, lane_for_row):
        y = lane
        net = _network_root(r["network"])
        color = palette.get(net, "#94a3b8")
        pstart = r["predicted_start"] / PRED_PER_MS
        pdur = r["predicted_duration"] / PRED_PER_MS
        astart = r["actual_start"] / ACTUAL_PER_MS
        aend = r["actual_end"] / ACTUAL_PER_MS
        adur = aend - astart
        ax_p.broken_barh([(pstart, pdur)], (y - BAR_H/2, BAR_H),
                         facecolors=color, edgecolors="black", linewidth=0.2)
        ax_a.broken_barh([(astart, adur)], (y - BAR_H/2, BAR_H),
                         facecolors=color, edgecolors="black", linewidth=0.2)

    for ax, label, makespan in [(ax_p, "Predicted (scheduler)", pred_makespan),
                                (ax_a, "Actual (FireSim)", actual_makespan)]:
        ax.set_yticks(range(len(lane_keys)))
        ax.set_yticklabels([f"{k[0]}#{k[1]}" for k in lane_keys])
        ax.set_ylabel("Worker")
        ax.set_title(f"{label}  —  makespan {makespan:.3f} ms")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        # Vertical line at makespan
        ax.axvline(makespan, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.invert_yaxis()  # lane 0 at top

    ax_a.set_xlabel("Time (ms)")

    # Legend: one entry per unique network seen.
    seen_nets = sorted({_network_root(r["network"]) for r in rows})
    handles = [mpatches.Patch(color=palette.get(n, "#94a3b8"), label=n) for n in seen_nets]
    ax_p.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_rows": len(rows),
        "scale_ratio": scale,
        "predicted_makespan_ms": pred_makespan,
        "actual_makespan_ms": actual_makespan,
        "n_lanes": len(lane_keys),
    }


def render_fixture_gantt(fixture_json: str, out_path: str,
                         max_dispatches: Optional[int] = None,
                         title: Optional[str] = None) -> dict:
    """Predicted-only Gantt from a ModelBlaster schedule fixture JSON.

    Fixture schema (see schedule_fixtures/*.json):
        dispatches[name] = {
          start_time: float,   # cycles at 1 GHz (== µs)
          duration:   float,
          hardware_target: "CPU_P#0" | "CPU_E#0" | ...,
          job_name:   str,
          ...
        }
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    with open(fixture_json) as f:
        fx = json.load(f)
    dispatches = fx["dispatches"]
    items = list(dispatches.values())
    if max_dispatches:
        items = items[:max_dispatches]
    if not items:
        raise RuntimeError(f"no dispatches in {fixture_json}")

    # Lane = hardware_target string. Sort lanes for stable y-order.
    lanes = sorted({d["hardware_target"] for d in items})
    lane_idx = {lane: i for i, lane in enumerate(lanes)}

    # Color by network root — share the palette with the trace-based
    # render_gantt above so the same color means the same network across
    # both predicted-only and predicted-vs-actual figures.
    def _job_root(j: str) -> str:
        return _network_root(j)
    job_colors = _network_palette()

    # Convert start_time / duration to ms.
    # Fixtures generated by gen_3way_schedule.py store these in ms directly
    # (start_time=200.0 = 200 ms for dronet instance 1). MOSEK fixtures
    # store cycles. Heuristic: if max(start+duration) > 10000, assume cycles.
    raw_max = max(d["start_time"] + d["duration"] for d in items)
    SCALE = 1_000_000.0 if raw_max > 10_000 else 1.0  # cycles→ms or already ms

    makespan = raw_max / SCALE

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * len(lanes) + 2)))
    BAR_H = 0.7

    for d in items:
        lane = lane_idx[d["hardware_target"]]
        s = d["start_time"] / SCALE
        w = d["duration"] / SCALE
        c = job_colors.get(_job_root(d["job_name"]), "#94a3b8")
        ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                       facecolors=c, edgecolors="black", linewidth=0.1)

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.set_xlabel("Time (ms)")
    ax.set_title(
        (title or fixture_json) +
        f"  —  {len(items)} dispatches, makespan {makespan:.1f} ms"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.axvline(makespan, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.invert_yaxis()

    seen = sorted({_job_root(d["job_name"]) for d in items})
    handles = [mpatches.Patch(color=job_colors.get(k, "#94a3b8"), label=k) for k in seen]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_dispatches": len(items), "n_lanes": len(lanes), "makespan_ms": makespan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="path to xpurt_trace.csv (predicted+actual)")
    src.add_argument("--fixture", help="path to schedule fixture JSON (predicted only)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap dispatches for very large schedules (default all)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.trace:
        info = render_gantt(args.trace, args.out, max_rows=args.max_rows, title=args.title)
        print(f"wrote {args.out}")
        print(f"  rows: {info['n_rows']}")
        print(f"  lanes: {info['n_lanes']}")
        print(f"  scale (actual/predicted): {info['scale_ratio']:.6f}")
        print(f"  makespan: predicted={info['predicted_makespan_ms']:.3f} ms  "
              f"actual={info['actual_makespan_ms']:.3f} ms  "
              f"delta={(info['actual_makespan_ms']/info['predicted_makespan_ms']-1)*100:+.2f}%")
    else:
        info = render_fixture_gantt(args.fixture, args.out,
                                    max_dispatches=args.max_rows, title=args.title)
        print(f"wrote {args.out}")
        print(f"  dispatches: {info['n_dispatches']}")
        print(f"  lanes: {info['n_lanes']}")
        print(f"  makespan: {info['makespan_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
