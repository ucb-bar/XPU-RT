#!/usr/bin/env python3
"""Two views of what the QRB5165 timing does to SmolVLA in the Arena env.

  1. Where one inference goes: the 102 profiled dispatches as a Gantt, coloured
     by the hardware each landed on. This is the 3351.7 ms the environment has
     to wait, broken into the stages that make it up.
  2. What that costs in the loop: the sim timeline for each latency mode, one
     row per mode, marking every env step as executing a fresh chunk or stalled
     on a stale one.

Colours are the dataviz reference palette's categorical slots in their fixed
order (blue/orange/aqua) for the hardware identity, and its reserved `critical`
status step for the stalled state -- stalling is a bad state, not a fourth
series. They are used unmodified because that set ships pre-validated; the
skill's own validator is a node script and this host has no node, so inventing
new hexes here would have meant shipping unchecked ones.

    python3 plot_scheduled.py --out ../../plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "scheduled"

# dataviz reference palette, light surface.
SURFACE = "#fcfcfb"
HW_COLOR = {"CPU": "#2a78d6", "DSP": "#eb6834", "HTA": "#1baf7a"}  # slots 1,2,3
STALL = "#d03b3b"   # status: critical
FRESH = "#2a78d6"   # slot 1
INK, MUTED = "#0b0b0b", "#6b6b6b"

# The stage order is the order they actually run in one forward.
STAGE_ORDER = ["vision", "text", "state_proj", "prefill",
               "action_in", "time_in", "decode", "time_out", "action_out"]


def gantt(ax, schedule_path: Path):
    """One inference, as profiled on the board."""
    data = json.loads(schedule_path.read_text())
    hw_map = data.get("metadata", {}).get("profile_hw", {})
    rows = {}
    for key, d in data["dispatches"].items():
        model = str(key).split("_dispatch", 1)[0]
        stage = model.rsplit("_", 1)[0] if model.rsplit("_", 1)[-1].isdigit() else model
        machine = str(d.get("hardware_target", "")).split("#")[0]
        rows.setdefault(stage, []).append(
            (float(d["start_time"]), float(d["duration"]), hw_map.get(machine, machine))
        )
    stages = [s for s in STAGE_ORDER if s in rows] + \
             [s for s in sorted(rows) if s not in STAGE_ORDER]
    for i, stage in enumerate(stages):
        for start, dur, hw in rows[stage]:
            ax.barh(i, dur, left=start, height=.62,
                    color=HW_COLOR.get(hw, MUTED), edgecolor=SURFACE, linewidth=.6)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("ms since the observation was captured", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(s + d for v in rows.values() for s, d, _ in v) * 1.005)
    ax.grid(axis="x", alpha=.25, ls=":", color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    present = [h for h in ("CPU", "DSP", "HTA")
               if any(hw == h for v in rows.values() for *_, hw in v)]
    ax.legend(handles=[Patch(facecolor=HW_COLOR[h], label=h) for h in present],
              fontsize=8.5, frameon=False, ncol=len(present), loc="lower right")


def timeline(ax, runs: list[dict], control_hz: float):
    """Executing vs stalled, per env step, one row per latency mode."""
    labels = []
    for row, r in enumerate(runs):
        ep = r["episodes"][0]
        tl = ep["timeline"]
        dt_ms = 1000.0 / control_hz
        for e in tl:
            ax.barh(row, dt_ms, left=e["t_ms"], height=.55,
                    color=STALL if e["stalled"] else FRESH, linewidth=0)
        lat = r["latency_ms"]
        labels.append(f"board {lat:.0f} ms\n{r.get('replan','?')}" if lat else
                      "free inference\n(no latency)")
        n_stall = ep["stalled_steps"]
        ax.text(tl[-1]["t_ms"] + dt_ms * 3, row,
                f"{ep['steps']} steps, {n_stall} stalled "
                f"({n_stall / max(ep['steps'],1) * 100:.0f}%)"
                f"{'  success' if ep['success'] else '  no success'}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("sim time (ms)", fontsize=9, color=MUTED)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.42)
    ax.grid(axis="x", alpha=.25, ls=":", color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(handles=[Patch(facecolor=FRESH, label="executing a chunk"),
                       Patch(facecolor=STALL, label="stalled, chunk in flight")],
              fontsize=8.5, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=HERE.parent.parent / "plots")
    a = ap.parse_args()

    files = sorted(a.runs.glob("scheduled_rollout_*.json"))
    if not files:
        print(f"  no runs in {a.runs}"); return 1
    runs = [json.loads(f.read_text()) for f in files]
    # baseline first, then increasing latency
    runs.sort(key=lambda r: r["latency_ms"])
    sched_path = Path(runs[-1]["schedule"]["path"])

    a.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    gantt(axes[0], sched_path)
    axes[0].set_title(
        "One SmolVLA inference on the QRB5165 — 102 dispatches, "
        f"{runs[-1]['schedule']['makespan_ms']:.0f} ms end to end",
        fontsize=11, loc="left", color=INK)

    timeline(axes[1], runs, runs[0].get("control_hz", 30.0))
    covers = runs[-1]["chunk_covers_ms"]
    axes[1].set_title(
        f"What that costs in the loop — a chunk covers {covers:.0f} ms of motion "
        f"but takes {runs[-1]['latency_ms']:.0f} ms to produce",
        fontsize=11, loc="left", color=INK)

    fig.tight_layout()
    p = a.out / "smolvla_qrb5165_sim_playback.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
