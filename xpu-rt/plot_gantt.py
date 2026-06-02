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


def _instance_shade(base_hex: str, inst: int, n_inst_for_network: int = 4) -> str:
    """Vary brightness of a base color per instance so e.g. dronet0 and
    dronet1 are visually distinct. Returns hex; mixes the base with white
    in proportion ``inst / (n_inst_for_network)`` so inst=0 stays the
    full base color and higher instances get progressively lighter."""
    base = base_hex.lstrip("#")
    r = int(base[0:2], 16)
    g = int(base[2:4], 16)
    b = int(base[4:6], 16)
    # 0.0 (full color) at inst=0, up to 0.55 at the last expected instance.
    frac = min(0.55, 0.0 if n_inst_for_network <= 1 else 0.55 * inst / (n_inst_for_network - 1))
    rr = int(r + (255 - r) * frac)
    gg = int(g + (255 - g) * frac)
    bb = int(b + (255 - b) * frac)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


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
    # Brightness varies per instance so dronet0 and dronet1 are distinct.
    palette = _network_palette()
    n_inst_per_net: dict[str, int] = {}
    for r in rows:
        net = _network_root(r["network"])
        n_inst_per_net[net] = max(n_inst_per_net.get(net, 0), r["instance"] + 1)

    fig, (ax_p, ax_a) = plt.subplots(2, 1, figsize=(14, max(4, 0.6 * len(lane_keys) + 4)),
                                     sharex=True)
    BAR_H = 0.7

    for r, lane in zip(rows, lane_for_row):
        y = lane
        net = _network_root(r["network"])
        base = palette.get(net, "#94a3b8")
        color = _instance_shade(base, r["instance"], n_inst_per_net.get(net, 1))
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

    # Legend: one entry per (network, instance) seen so the shade↔instance
    # mapping is explicit. dronet0 / dronet1 will appear as distinct
    # legend rows with their actual rendered shades.
    seen_keys = sorted({(_network_root(r["network"]), r["instance"]) for r in rows})
    handles = []
    for net, inst in seen_keys:
        base = palette.get(net, "#94a3b8")
        shade = _instance_shade(base, inst, n_inst_per_net.get(net, 1))
        label = f"{net} (inst {inst})" if n_inst_per_net.get(net, 1) > 1 else net
        handles.append(mpatches.Patch(color=shade, label=label))
    ax_p.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

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
    # both predicted-only and predicted-vs-actual figures. Instance index
    # shades the base color so e.g. dronet0 is distinct from dronet1.
    def _job_root(j: str) -> str:
        return _network_root(j)

    def _inst_idx(j: str, root: str) -> int:
        if j == root or not j.startswith(root):
            return 0
        rest = j[len(root):]
        return int(rest) if rest.isdigit() else 0
    job_colors = _network_palette()

    # Compute n_instances per network from the data.
    inst_max: dict[str, int] = {}
    for d in items:
        root = _job_root(d["job_name"])
        inst_max[root] = max(inst_max.get(root, 0), _inst_idx(d["job_name"], root) + 1)

    # Convert start_time / duration to ms.
    # Fixtures generated by gen_3way_schedule.py store these in ms directly
    # (start_time=200.0 = 200 ms for dronet instance 1). MOSEK fixtures
    # store cycles. Heuristic: if max(start+duration) > 10000, assume cycles.
    raw_max = max(d["start_time"] + d["duration"] for d in items)
    SCALE = 1_000_000.0 if raw_max > 10_000 else 1.0  # cycles→ms or already ms

    makespan = raw_max / SCALE

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * len(lanes) + 2)))
    BAR_H = 0.7

    # Derive each instance's periodic slot (when this fixture was
    # generated with enforce_periodic / partition mode) so we can flag
    # dispatches that bleed past their parent network's makespan window.
    # provenance.instances has one row per (network, instance); the slot
    # length is horizon_ms / n_instances_of_that_network.
    inst_slots: dict[tuple[str, int], tuple[float, float]] = {}
    prov = fx.get("_provenance", {})
    horizon = float(prov.get("horizon_target_ms", 0) or 0)
    if horizon > 0:
        n_inst_by_net: dict[str, int] = {}
        for ins in prov.get("instances", []):
            n_inst_by_net[ins["network"]] = n_inst_by_net.get(ins["network"], 0) + 1
        for ins in prov.get("instances", []):
            net = ins["network"]
            inst = ins["instance"]
            n = n_inst_by_net.get(net, 1)
            slot_len = horizon / max(n, 1)
            inst_slots[(net, inst)] = (inst * slot_len, (inst + 1) * slot_len)

    n_out_of_slot = 0
    for d in items:
        lane = lane_idx[d["hardware_target"]]
        s = d["start_time"] / SCALE
        w = d["duration"] / SCALE
        root = _job_root(d["job_name"])
        inst = _inst_idx(d["job_name"], root)
        base = job_colors.get(root, "#94a3b8")
        c = _instance_shade(base, inst, inst_max.get(root, 1))
        # If we know the periodic slot and this dispatch falls outside
        # its parent instance's slot, draw a red border + hatch so the
        # deadline violation is obvious at a glance.
        edge_color = "black"
        edge_width = 0.1
        hatch_pattern: str | None = None
        slot = inst_slots.get((root, inst))
        if slot is not None and (s + 1e-6 < slot[0] or s + w > slot[1] + 1e-6):
            edge_color = "#dc2626"  # red-600
            edge_width = 1.0
            hatch_pattern = "////"
            n_out_of_slot += 1
        if hatch_pattern is not None:
            ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                           facecolors=c, edgecolors=edge_color,
                           linewidth=edge_width, hatch=hatch_pattern)
        else:
            ax.broken_barh([(s, w)], (lane - BAR_H/2, BAR_H),
                           facecolors=c, edgecolors=edge_color, linewidth=edge_width)

    # Draw per-instance slot brackets at the bottom of each lane so the
    # intended periodic windows are visible alongside the actual placement.
    if inst_slots:
        slot_bracket_y = len(lanes) + 0.3
        for (net, inst), (s0, s1) in sorted(inst_slots.items()):
            base = job_colors.get(net, "#94a3b8")
            shade = _instance_shade(base, inst, inst_max.get(net, 1))
            ax.broken_barh([(s0, s1 - s0)], (slot_bracket_y, 0.15),
                           facecolors=shade, edgecolors="black", linewidth=0.3, alpha=0.5)
            ax.text((s0 + s1) / 2, slot_bracket_y + 0.075, f"{net}{inst}",
                    ha="center", va="center", fontsize=5, color="black")

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.set_xlabel("Time (ms)")
    slot_summary = ""
    if inst_slots:
        if n_out_of_slot:
            slot_summary = f", {n_out_of_slot}/{len(items)} OUT-OF-SLOT (red+hatched)"
        else:
            slot_summary = f", all {len(items)} dispatches in their periodic slots"
    ax.set_title(
        (title or fixture_json) +
        f"  —  {len(items)} dispatches, makespan {makespan:.1f} ms{slot_summary}"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.axvline(makespan, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.invert_yaxis()

    seen_keys = sorted({(_job_root(d["job_name"]), _inst_idx(d["job_name"], _job_root(d["job_name"])))
                        for d in items})
    handles = []
    for net, inst in seen_keys:
        base = job_colors.get(net, "#94a3b8")
        shade = _instance_shade(base, inst, inst_max.get(net, 1))
        label = f"{net} (inst {inst})" if inst_max.get(net, 1) > 1 else net
        handles.append(mpatches.Patch(color=shade, label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    # Draw periodic-slot boundary lines if this fixture was generated
    # with enforce_periodic. The provenance carries horizon_ms + the
    # per-network instance counts, from which we derive each network's
    # period (horizon / n_inst) and draw faint vertical bands per
    # network color at every k*period.
    prov = fx.get("_provenance", {})
    horizon = float(prov.get("horizon_target_ms", 0) or 0)
    enforce_periodic = "enforce_periodic" in (prov.get("config", "") or "") or \
                       any("periodic" in (prov.get("config", "") or "").lower() for _ in [0])
    # Heuristic: if the fixture filename includes 'periodic' in its
    # _provenance.config path, treat as periodic.
    if horizon > 0 and ("periodic" in str(prov.get("config", ""))):
        for ins in prov.get("instances", []):
            net = ins["network"]
            net_root = _network_root(net)
            n_inst = sum(1 for j in prov["instances"] if j["network"] == net)
            period = horizon / max(n_inst, 1)
            color = job_colors.get(net_root, "#94a3b8")
            # One line per slot boundary for this network.
            for k in range(1, n_inst):
                ax.axvline(k * period, color=color, linestyle=":",
                           linewidth=0.6, alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_dispatches": len(items), "n_lanes": len(lanes), "makespan_ms": makespan}


def render_composite_gantt(before_json: str, after_json: str, out_path: str,
                           titles: tuple = ("Before", "After")) -> dict:
    """Stacked before/after predicted Gantt for a PR.

    Renders each schedule fixture via render_fixture_gantt (so coloring/instance
    shading/periodic-slot overlays match the single-schedule figures), then
    stacks the two panels into one PNG. Reuses the existing renderer rather than
    duplicating its drawing logic.
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    tmp = tempfile.mkdtemp(prefix="composite_gantt_")
    before_png = os.path.join(tmp, "before.png")
    after_png = os.path.join(tmp, "after.png")
    info_b = render_fixture_gantt(before_json, before_png, title=titles[0])
    info_a = render_fixture_gantt(after_json, after_png, title=titles[1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    for ax, png in zip(axes, (before_png, after_png)):
        ax.imshow(mpimg.imread(png))
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"before": info_b, "after": info_a, "out": out_path}


def render_terminal_gantt(report, *, deadline_us=None, width: int = 80,
                          max_lanes=None) -> str:
    """ASCII Gantt from a SchedulerReport's per-dispatch list (schema >= 2).

    One lane per machine; dispatches that finish after the deadline are shaded;
    a ``|`` column marks the deadline. CLI-native so a schedule can be reasoned
    about in-terminal (over SSH / inside an agent) without opening a PNG.
    Returns the rendered string (also handy for tests). Degrades to a one-line
    notice if the report predates schema v2 (no ``dispatches``).
    """
    import json as _json

    if isinstance(report, str):
        with open(report) as f:
            report = _json.load(f)
    elif hasattr(report, "to_dict"):
        report = report.to_dict()

    dispatches = report.get("dispatches")
    if not dispatches:
        return ("[terminal gantt needs a schema>=2 SchedulerReport with a "
                "'dispatches' list; this report has none]")

    if deadline_us is None:
        deadline_us = report.get("deadline_us")

    # lanes = all known machines (from utilization) plus any seen in dispatch
    # targets. Including utilization keys means idle backends show as empty
    # lanes — which is exactly what makes a rebalance opportunity visible.
    lanes: list[str] = []
    for m in (report.get("utilization") or {}):
        if m and m not in lanes:
            lanes.append(m)
    for d in dispatches:
        for m in str(d.get("target", "")).split("+"):
            if m and m not in lanes:
                lanes.append(m)
    lanes.sort()
    if max_lanes:
        lanes = lanes[:max_lanes]

    makespan = max((float(d.get("finish_us", 0.0)) for d in dispatches), default=0.0) or 1.0
    scale = width / makespan
    dl_col = int(round(deadline_us * scale)) if (deadline_us and deadline_us <= makespan) else None
    n_late = 0

    out = []
    solver = report.get("solver_name", "?")
    out.append(f"Gantt — solver={solver}  makespan={makespan:,.1f} us"
               + (f"  deadline={deadline_us:,.1f} us" if deadline_us else ""))
    out.append("-" * (width + 14))
    lane_w = max((len(m) for m in lanes), default=4)
    for m in lanes:
        row = [" "] * width
        for d in dispatches:
            if m not in str(d.get("target", "")).split("+"):
                continue
            s = int(float(d.get("start_us", 0.0)) * scale)
            e = max(s + 1, int(float(d.get("finish_us", 0.0)) * scale))
            late = deadline_us is not None and float(d.get("finish_us", 0.0)) > deadline_us
            ch = "x" if late else "="
            for i in range(s, min(e, width)):
                if row[i] == " " or (ch == "x" and row[i] == "="):
                    row[i] = ch
        if dl_col is not None and dl_col < width and row[dl_col] == " ":
            row[dl_col] = "|"
        out.append(f"{m.rjust(lane_w)} |{''.join(row)}|")
    if dl_col is not None:
        ruler = [" "] * width
        if dl_col < width:
            ruler[dl_col] = "|"
        out.append(f"{' ' * lane_w} |{''.join(ruler)}|  | = deadline {deadline_us:,.1f} us")

    if deadline_us is not None:
        late_ops = [d for d in dispatches if float(d.get("finish_us", 0.0)) > deadline_us]
        n_late = len(late_ops)
        out.append("")
        out.append(f"  = on-time   x finishes after deadline")
        if n_late:
            over = makespan - deadline_us
            out.append(f"  DEADLINE MISSED by {over:,.1f} us; {n_late} dispatch(es) finish late.")
            worst = sorted(late_ops, key=lambda d: -float(d.get("finish_us", 0.0)))[:5]
            for d in worst:
                out.append(f"    - {d.get('name', d.get('id'))} on {d.get('target')} "
                           f"finishes {float(d.get('finish_us', 0.0)):,.1f} us")
        else:
            out.append(f"  Deadline met with {deadline_us - makespan:,.1f} us of slack.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="path to xpurt_trace.csv (predicted+actual)")
    src.add_argument("--fixture", help="path to schedule fixture JSON (predicted only)")
    src.add_argument("--report", help="path to SchedulerReport JSON (terminal ASCII Gantt)")
    ap.add_argument("--out", default=None, help="output PNG path (required for --trace/--fixture)")
    ap.add_argument("--terminal", action="store_true",
                    help="force terminal ASCII mode (implied by --report)")
    ap.add_argument("--deadline-us", type=float, default=None,
                    help="(terminal) deadline marker in us; overrides report.deadline_us")
    ap.add_argument("--width", type=int, default=80, help="(terminal) chart width in columns")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap dispatches for very large schedules (default all)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.report or args.terminal:
        if not args.report:
            ap.error("--terminal requires --report")
        print(render_terminal_gantt(args.report, deadline_us=args.deadline_us, width=args.width))
        return 0

    if not args.out:
        ap.error("--out is required for --trace/--fixture")

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
