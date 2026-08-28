#!/usr/bin/env python3
"""Gantt charts from a merlin dispatch-scheduler trace, on physical core lanes.

Why a new renderer rather than reusing one of the two that exist:

* `xpu-rt/plot_gantt.py` reads ModelBlaster's `xpurt_trace.csv` contract
  (`predicted_start_ms`, `actual_start_cycles`, `core_kind`, `hart`, ...). It
  shares zero column names with the merlin trace, and its `ACTUAL_PER_MS` is
  hardcoded to a 1 MHz mtime that is wrong for this board's 24 MHz rdtime.
* merlin's `analysis/plot_dispatch_trace.py` does read the native schema, but
  packs *synthetic* lanes by overlap, because until now the trace did not say
  which core a dispatch ran on.

Now that the trace carries `cores` (the set the runner actually held) this can
draw the truth: one lane per physical core, and a sharded dispatch as a single
bar spanning every core it occupied. Drawing a shard as N independent bars would
depict a machine with more cores than the board has -- the same class of error
that produced one retracted conclusion in this project already.

Falls back to joining against the schedule's `hardware_target` for traces taken
before the `cores` column existed, so old artifacts still render.

Usage:
    plot_k1_trace_gantt.py --trace t.csv --schedule s.json --out p.png \
        [--window-ms 140] [--title "B4 ..."] [--periods mlp=10,dronet=33.3]
    plot_k1_trace_gantt.py --composite B0=t0.csv:s0.json B4=t4.csv:s4.json \
        --out composite.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MM = 1 / 25.4
DOUBLE_COL = 183 * MM

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6,
    "axes.labelsize": 6, "axes.titlesize": 7,
    "xtick.labelsize": 5, "ytick.labelsize": 4.4,
    "legend.fontsize": 5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

# Model colours are assigned by first appearance so any model set works.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
C_DEADLINE = "#D55E00"


def model_of(job_name: str) -> str:
    return job_name.rstrip("0123456789") or job_name


#: rdtime on this board. NOT the 1.6 GHz core clock and not 1 MHz -- the
#: device-tree timebase-frequency is 24000000, and every cycles->time
#: conversion in this project uses it.
K1_RDTIME_HZ = 24_000_000.0


def _normalise_modelblaster(rows: List[dict]) -> List[dict]:
    """Map ModelBlaster's harness_xpurt trace onto the canonical column names.

    Two producers emit measured K1 traces and they disagree on spelling, not on
    meaning: merlin's runner writes `start_us`/`run_us`/`job_name`/`cores`,
    ModelBlaster's writes `actual_start_cycles`/`actual_end_cycles`/
    `network`+`instance`/`core_kind`+`hart`. Normalising once here is what keeps
    this from becoming the fourth renderer that reads exactly one producer.

    Cycles are rdtime ticks at 24 MHz, and the run is stamped from the first
    tick observed rather than from 0, so the axis starts at the run's own t0.
    """
    if not rows or "actual_start_cycles" not in rows[0]:
        return rows
    t0 = min(int(r["actual_start_cycles"]) for r in rows)
    out = []
    for r in rows:
        s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        d = dict(r)
        d["start_us"] = (s - t0) / K1_RDTIME_HZ * 1e6
        d["run_us"] = max(e - s, 0) / K1_RDTIME_HZ * 1e6
        d["job_name"] = f'{r.get("network", "")}{r.get("instance", "")}'
        out.append(d)
    return out


def read_trace(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return _normalise_modelblaster(list(csv.DictReader(f)))


def read_schedule(path: Optional[str]) -> Dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("dispatches", {})


def periods_from_schedule(path: Optional[str]) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        md = json.load(f).get("metadata", {}) or {}
    return {k: float(v) for k, v in (md.get("periodic_networks") or {}).items()}


def lanes_for(row: dict, sched: Dict[str, dict]) -> List[str]:
    """Cluster-qualified core ids this dispatch occupied.

    Prefers the trace's own `cores` column -- the set the runtime enforced --
    and falls back to the schedule's `hardware_target` for older traces.
    """
    raw = (row.get("cores") or "").strip()
    target = (row.get("target") or "").strip()
    if raw and target:
        return [f"{target}#{c}" for c in raw.split("+") if c != ""]

    # ModelBlaster's harness_xpurt emits the same fact under different names:
    # `core_kind` + `hart`, one hart per row. It is a real placement -- the
    # runner pins to that hart -- so refusing to draw it because the column is
    # spelled differently would leave the ONE schema that currently produces
    # measured K1 traces unrenderable, which is how this repo ended up with
    # three Gantt renderers that each read one producer.
    kind = (row.get("core_kind") or "").strip()
    hart = (row.get("hart") or "").strip()
    if kind and hart != "":
        return [f"{kind}#{hart}"]

    ent = sched.get(row.get("dispatch_key", ""))
    if ent:
        return [c for c in str(ent.get("hardware_target", "")).split("+") if c]
    return []


def collect_lane_order(all_lanes: Sequence[str]) -> List[str]:
    """CPU_P#0..n then CPU_E#0..n, so cluster blocks stay contiguous."""
    def key(name: str):
        kind, _, idx = name.partition("#")
        return (0 if kind.endswith("_P") else 1, kind, int(idx or 0))
    return sorted(set(all_lanes), key=key)


def draw(ax, rows: Sequence[dict], sched: Dict[str, dict],
         lane_idx: Dict[str, int], window_ms: float,
         colours: Dict[str, str], periods: Dict[str, float]) -> None:
    for r in rows:
        start = float(r["start_us"]) / 1000.0
        if start > window_ms:
            continue
        dur = float(r["run_us"]) / 1000.0
        lanes = [lane_idx[l] for l in lanes_for(r, sched) if l in lane_idx]
        if not lanes:
            continue
        m = model_of(r.get("job_name", ""))
        lo, hi = min(lanes), max(lanes)
        # One bar spanning the whole held set: that is what the core lock did.
        ax.broken_barh([(start, max(dur, 0.12))], (lo - 0.42, (hi - lo) + 0.84),
                       facecolors=colours.get(m, "#777777"),
                       edgecolors="white", linewidth=0.15)

    # Deadline rules for the LONGEST-period model present. That is the heavy
    # model -- the one whose deadlines are actually in question -- and it also
    # keeps the chart readable: a 10 ms period draws 14 rules across a 140 ms
    # window and turns into hatching, while 33.3 ms draws 5.
    if periods:
        m = max(periods, key=lambda k: periods[k])
        T = periods[m]
        k = 0
        while k * T <= window_ms:
            ax.axvline(k * T, color=C_DEADLINE, lw=0.4, ls=(0, (2, 2)), zorder=0)
            k += 1

    n = len(lane_idx)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(lane_idx.keys()))
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, window_ms)
    ax.spines[["top", "right"]].set_visible(False)
    # Cluster divider.
    kinds = [l.split("#")[0] for l in lane_idx]
    for i in range(1, n):
        if kinds[i] != kinds[i - 1]:
            ax.axhline(i - 0.5, color="0.75", lw=0.4)


def render(panels: List[Tuple[str, str, Optional[str]]], out: str,
           window_ms: float, height_per_panel_mm: float = 26.0) -> None:
    """panels: list of (title, trace_path, schedule_path)."""
    loaded = []
    all_lanes: List[str] = []
    models: List[str] = []
    for title, tp, sp in panels:
        rows = read_trace(tp)
        sched = read_schedule(sp)
        loaded.append((title, rows, sched, periods_from_schedule(sp)))
        for r in rows:
            all_lanes.extend(lanes_for(r, sched))
            m = model_of(r.get("job_name", ""))
            if m not in models:
                models.append(m)
    lane_order = collect_lane_order(all_lanes)
    if not lane_order:
        print(f"{out}: no core placement in any trace or schedule; nothing to "
              f"draw", file=sys.stderr)
        return
    lane_idx = {l: i for i, l in enumerate(lane_order)}
    # Colour by descending period, not by first appearance: otherwise the same
    # model changes colour between figures depending on which dispatch happened
    # to be traced first, which makes two charts impossible to read together.
    all_periods: Dict[str, float] = {}
    for _, _, _, pr in loaded:
        all_periods.update(pr)
    ordered = sorted(models, key=lambda m: (-all_periods.get(m, 0.0), m))
    colours = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(ordered)}
    models = ordered

    h = max(len(panels) * height_per_panel_mm, 30.0) * MM
    fig, axes = plt.subplots(len(panels), 1, figsize=(DOUBLE_COL, h),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for lab, ax, (title, rows, sched, periods) in zip("abcdefgh", axes, loaded):
        draw(ax, rows, sched, lane_idx, window_ms, colours, periods)
        ax.set_title(title, loc="left", pad=2)
        ax.text(-0.062, 1.16, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="right")
    axes[-1].set_xlabel("Time on the K1 (ms)")

    handles = [Patch(facecolor=colours[m], label=m) for m in models]
    tight = max((p for _, _, _, pr in loaded for p in pr.values()), default=None)
    if tight is not None:
        handles.append(plt.Line2D([], [], color=C_DEADLINE, lw=0.6,
                                  ls=(0, (2, 2)),
                                  label=f"deadlines ({tight:g} ms)"))
    axes[0].legend(handles=handles, ncol=len(handles), frameon=False,
                   loc="lower left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        p = out if out.endswith(f".{ext}") else f"{os.path.splitext(out)[0]}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {p}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace")
    ap.add_argument("--schedule")
    ap.add_argument("--title", default="")
    ap.add_argument("--composite", nargs="*", default=[],
                    help="LABEL=trace.csv:schedule.json, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-ms", type=float, default=140.0)
    a = ap.parse_args()

    panels: List[Tuple[str, str, Optional[str]]] = []
    if a.composite:
        for spec in a.composite:
            label, _, rest = spec.partition("=")
            tp, _, sp = rest.partition(":")
            panels.append((label or os.path.basename(tp), tp, sp or None))
    elif a.trace:
        panels.append((a.title or os.path.basename(a.trace), a.trace, a.schedule))
    else:
        ap.error("need --trace or --composite")

    render(panels, a.out, a.window_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
