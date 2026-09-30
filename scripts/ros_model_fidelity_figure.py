#!/usr/bin/env python3
"""The ROS 2 baseline at three tiers of evidence, drawn so the tiers can be read against each other.

Tier A is the ANALYTICAL ROS 2 arm, `scripts/ros_pinning_model.py`:
static per-node partitions, serial node graphs, periodic release, free middleware, per-node costs taken
from XPU-RT's own schedule. The same recurrence costed from the board's own ROS 2 C++ traces is tier B,
`scripts/ros_pinning_profiled.py`; the board runs themselves (`results/codesign_feedback/ros_traced/`)
are tier C. This script draws the three side by side, in three figure options, each in more than one
layout, and never recomputes a number those two modules already compute: every prediction here is a
field of `ros_pinning_profiled.table()` / `submitted_spec_response()` or of `ros_pinning_model`.

  three_tier        per (arm, camera rate) row: camera->goal latency and control cadence, A and B
                    predicted against C measured, saturated rows drawn hollow and kept out of the
                    residual statistics (the recurrence has no queue term; `predict()` names one)
  submitted_metric  the Tier A showdown's own metric -- instance-2 perception response at the 22 ms
                    period against the 23 ms budget -- on the Tier A / recost / profiled inputs, and
                    the control-rate label the figure printed: a 12.40 ms response passed on the
                    composer's command line and the 50 Hz the zero-order hold turns it into. That number
                    is the profile-PREDICTED worst response of `dronet` under a per-network partition
                    model of a different four-network workload (`networks_k1_tri_exact_100ms.json`,
                    recorded in `results/codesign_feedback/microros_baseline_k1/microros_baseline_k1.json`);
                    the flight workload has no dronet and its control node
                    responds in 0.083 ms. It is drawn against the tier-B prediction and the measured pinned arm
                    (per-network pinning, control on its own 100 Hz timer: `p3`)
  story             the story layout recomposed on tier B. No flight replays a tier-B
                    cadence, so the flight panels are the MEASURED arm whose timing tier B predicts
                    (`ros_p345.csv`, the p3 arm at the 45 Hz camera) against the XPU-RT arm it was
                    censused with, every flight of the census drawn rather than one display pair; the
                    schedule panels are the tier-B timeline of that arrangement beside its measured
                    board Gantt (`schedules/measured_gantt_v3_{xpu,p3}.json`)

Every panel carries a tier badge. Every stem gets .png, .pdf and a `_metrics.json` sidecar whose
`inputs` block (figure_constants.sidecar_common) records the sha256 of every file a number came from.
`scripts/verify_ros_model_fidelity.py` re-runs this producer into a tempdir and compares each sidecar
field by field.

    scripts/ros_model_fidelity_figure.py                           # every option, every layout
    scripts/ros_model_fidelity_figure.py --option story --out-dir /tmp/x

No hardware, no GPU. The per-node costs are parsed from ~200 board traces, which is most of the ~1 min
run time. The story's census flight panel reads the per-flight records under
`campaign_percep/records/`, which are not tracked (.gitignore, "Flight-rollout dumps"); every other
input is tracked.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import functools
import glob
import json
import os
import statistics
import sys
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
REFINED = os.path.join(RES, "refined")
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import ros_pinning_profiled as RP   # noqa: E402
import ros_pinning_model as RM   # noqa: E402
import figure_constants as FC   # noqa: E402
from flight_quarantine import flight_rows   # noqa: E402
from hil_envelope_panel import wilson   # noqa: E402
from showdown_atlas import newcombe   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
import showdown_paper_figure as SP   # noqa: E402
import showdown_v3_figure as V   # noqa: E402

SCRIPT = "ros_model_fidelity_figure"
# every stem records the same ~800 trace files; hash each once per run rather than once per stem
FC.sha256_of = functools.lru_cache(maxsize=None)(FC.sha256_of)
C_XPU, C_ROS = "#1f9e5a", "#e2231a"
# One colour per tier for the ROS 2 predictions, so a reader can follow a tier across panels. Tier C
# is the measured ROS 2 arm and keeps the repo's ROS red; the two model tiers are set apart from it.
TIER = {"A": {"badge": "Analytical model", "fc": "#6b6b6b", "data": "#8a8a8a", "marker": "s"},
        "B": {"badge": "Calibrated model · board-profiled costs", "fc": "#0f6f7c", "data": "#138a99", "marker": "D"},
        "C": {"badge": "Measured on K1", "fc": "#1d3557", "data": C_ROS, "marker": "o"}}

# the configuration the paper's text describes -- one core set per network, control on its own 100 Hz
# timer -- at the camera rate the Tier A showdown's schedule was laid out on (22 ms ~ 45 Hz)
PAPER_ARM, PAPER_HZ = "p3", 45
XPU_TRACE, ROS_TRACE = "xpu_a_cpsat_hard.csv", "ros_p345.csv"
CENSUS_CSV = os.path.join(RES, "campaign_percep", "campaign.csv")
RECORDS = os.path.join(RES, "campaign_percep", "records")
GANTT_PREFIX = os.path.join(REPO, "schedules", "measured_gantt_v3")
SUBMITTED_SIDECAR = os.path.join(RES, "warehouse_showdown_board_metrics.json")
# the prediction the Tier A showdown's two control responses (4.89 and 12.40 ms) come from, tracked with
# the README it was produced with. When it is absent the sidecar says so and the drawn literal falls back to
# ros_pinning_model.CONTROL_RATE_PANEL, which holds the same two numbers.
LITERAL_SOURCE = os.environ.get("RMF_LITERAL_SOURCE",
                                os.path.join(RES, "microros_baseline_k1", "microros_baseline_k1.json"))


def literal_source():
    """What the file the 12.40 ms came from says the number is: a predicted dronet response."""
    if not os.path.exists(LITERAL_SOURCE):
        return {"path": LITERAL_SOURCE, "present": False}
    d = json.load(open(LITERAL_SOURCE))
    return {"path": LITERAL_SOURCE, "present": True, "sha256": FC.sha256_of(LITERAL_SOURCE),
            "workload": d.get("workload"), "model": d.get("model"),
            "ros_worst_critical_response_ms": d.get("ros_worst_critical_response_ms"),
            "per_net_worst_response_ms": {k: v.get("worst_response_ms") for k, v in (d.get("per_net") or {}).items()},
            "xpurt_shard_worst_response_ms": d.get("xpurt_shard_worst_response_ms"),
            "xpurt_greedy_worst_response_ms": d.get("xpurt_greedy_worst_response_ms"),
            # the flags the composer was given, against the file: 12.40 is dronet's line, 4.89 the shard line
            "drawn_ros_literal_is_dronet": abs(float((d.get("per_net") or {}).get("dronet", {}).get("worst_response_ms", -1))
                                               - RM.CONTROL_RATE_PANEL["ros_worst_response_ms"]) < 0.005,
            "drawn_xpu_literal_is_shard": abs(float(d.get("xpurt_shard_worst_response_ms", -1))
                                              - RM.CONTROL_RATE_PANEL["xpu_worst_response_ms"]) < 0.005}


# ------------------------------------------------------------------------------------------ data
def gather():
    """Everything every option draws, computed once by the two model modules."""
    costs = RP.pooled_inputs()
    rows = RP.table(costs=costs)
    pm = costs[("perception", 4)]["med"]
    spec = dict(RP.SUBMITTED_SPEC,
                response_submitted_ms=RP.submitted_spec_response(RP.ASSUMED["submitted"]["perception"]),
                response_assumed_ms=RP.submitted_spec_response(RP.ASSUMED["recost"]["perception"]),
                response_profiled_ms=RP.submitted_spec_response(pm), perception_profiled_ms=pm)
    runs = sorted({t for v in costs.values() for t in v["runs"] if not t.startswith("derived")}
                  | {t for r in rows for t in r["runs"]})
    inputs = [os.path.join(RP.TRACED, t, f) for t in runs
              for f in ("trace.csv", "released.csv", "summary.json", "manifest.json")]
    inputs += [os.path.join(REPO, RP.ASSUMED[k]["src"]) for k in RP.ASSUMED]
    return {"costs": costs, "rows": rows, "spec": spec, "inputs": inputs}


def rkey(r):
    return f"{r['arm']}@{r['camera_hz']}"


def row_of(D, arm, hz):
    return next(r for r in D["rows"] if r["arm"] == arm and r["camera_hz"] == hz)


def residual_stats(D):
    """pred - meas on the unsaturated rows only: above saturation the recurrence has no queue term, so a
    residual there measures the missing mechanism rather than the inputs, and is reported apart."""
    un = [r for r in D["rows"] if not r["saturated"]]; sat = [r for r in D["rows"] if r["saturated"]]
    lat = lambda rs, k: {"mean_ms": statistics.mean(r[k] - r["chain_measured_ms"] for r in rs),
                         "median_ms": statistics.median(r[k] - r["chain_measured_ms"] for r in rs),
                         "min_ms": min(r[k] - r["chain_measured_ms"] for r in rs),
                         "max_ms": max(r[k] - r["chain_measured_ms"] for r in rs)}
    hz = lambda rs, k: {"mean_abs_hz": statistics.mean(abs(r[k] - r["ctrl_measured_hz"]) for r in rs),
                        "max_abs_hz": max(abs(r[k] - r["ctrl_measured_hz"]) for r in rs)}
    return {"n_rows": len(D["rows"]), "n_unsaturated": len(un), "n_saturated": len(sat),
            "saturated_rows": [rkey(r) for r in sat],
            "latency_unsaturated": {"A_submitted": lat(un, "chain_assumed_submitted_ms"),
                                    "A_recost": lat(un, "chain_assumed_ms"),
                                    "B_profiled": lat(un, "chain_model_queue_ms")},
            "latency_saturated": {"A_submitted": lat(sat, "chain_assumed_submitted_ms"),
                                  "B_profiled_model_only": lat(sat, "chain_model_ms"),
                                  "B_profiled_with_queue": lat(sat, "chain_model_queue_ms")},
            "ctrl_all_rows": {"A_submitted": hz(D["rows"], "ctrl_assumed_submitted_hz"),
                              "A_recost": hz(D["rows"], "ctrl_assumed_hz"),
                              "B_profiled": hz(D["rows"], "ctrl_pred_hz")}}


def row_record(r):
    keep = ("arm", "camera_hz", "pool", "ctrl_mode", "qos_depth", "saturated", "chain_assumed_submitted_ms",
            "chain_assumed_ms", "chain_model_ms", "chain_model_queue_ms", "chain_measured_ms",
            "ctrl_assumed_submitted_hz", "ctrl_assumed_hz", "ctrl_pred_hz", "ctrl_measured_hz", "ctrl_measured_gap_ms")
    return {k: r[k] for k in keep}


# ------------------------------------------------------------------------------------------ style
def fonts(layout):
    base = {"wide": 13.0, "column": 7.2, "tall": 15.0, "story": 15.0}[layout]
    return dict(base=base, tick=base * 0.9, lab=base, leg=base * 0.82, title=base * 1.08, badge=base * 0.66,
                note=base * 0.78, head=base * 1.3)


def badges(ax, tiers, fz, y=None, x=0.0, wrap=1.0):
    """The tier badges of a panel, left to right in tier order, just above the axes; a row that would run
    past the axes' right edge wraps upward, and the panel's left title is lifted clear of however many
    rows that took. `y` (axes fraction) places them elsewhere, e.g. under a Gantt, without moving the title."""
    fig = ax.figure; r = fig.canvas.get_renderer(); dpi = fig.dpi
    ax.apply_aspect()
    bb = ax.get_window_extent(renderer=r); aw, ah = bb.width, bb.height
    fsz = fz["badge"]; pad_px = 0.28 * fsz * dpi / 72.0
    row_h = (fsz * 1.25 * dpi / 72.0 + 2 * pad_px + 3.0) / ah
    y0 = (1.0 + (4.0 + pad_px) / ah) if y is None else y
    texts, cur, row = [], x, 0
    for t in tiers:                     # first pass: measure, and pack left to right into rows
        tx = ax.text(0, 0, TIER[t]["badge"], transform=ax.transAxes, ha="left", va="bottom", fontsize=fsz, color="white",
                     weight="bold", zorder=50, clip_on=False, bbox=dict(boxstyle="round,pad=0.28", fc=TIER[t]["fc"], ec="white", lw=0.8))
        w = (tx.get_window_extent(renderer=r).width + 2 * pad_px) / aw
        if cur > x and cur + w > wrap:
            cur, row = x, row + 1
        texts.append((tx, cur, row)); cur += w + 6.0 / aw
    for tx, cx, rw in texts:            # second pass: the first row on top, so the badges read in tier order
        tx.set_position((cx + pad_px / aw, y0 + (row - rw) * row_h))
    placed = [TIER[t]["badge"] for t in tiers]
    if y is None:
        t = ax._left_title
        if t.get_text():
            ax.set_title(t.get_text(), loc="left", fontsize=t.get_fontsize(), weight=t.get_weight(),
                         pad=6.0 + (row + 1) * row_h * ah * 72.0 / dpi)
    return placed


def letter(ax, s, fz, dx=-0.06, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=fz["head"], weight="bold", va="bottom", ha="right", clip_on=False)


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="0.92", lw=0.6, zorder=0)


# ------------------------------------------------------------------------------ three_tier panels
def panel_latency(ax, D, st, fz, small=False):
    """camera->goal: A and B predicted (y) against C measured (x), y = x drawn; saturated rows hollow."""
    rows = D["rows"]; ms = 3.2 if small else 7.5
    lo, hi = 20.0, 60.0
    ax.axvspan(45.0, hi, color="#f4f4f4", zorder=0)
    ax.text(hi - 0.6, lo + 1.0, "saturated rows\n(hollow; excluded\nfrom the residuals)", ha="right", va="bottom",
            fontsize=fz["note"], color="0.35", style="italic")
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(lo + 7.0, lo + 5.6, "y = x", rotation=45, ha="center", va="center", fontsize=fz["note"], color="0.25")
    for tier, key in (("A", "chain_assumed_submitted_ms"), ("B", "chain_model_queue_ms")):
        t = TIER[tier]
        for sat in (False, True):
            rs = [r for r in rows if r["saturated"] == sat]
            ax.plot([r["chain_measured_ms"] for r in rs], [r[key] for r in rs], ls="none", marker=t["marker"], ms=ms,
                    mfc=("none" if sat else t["data"]), mec=t["data"], mew=1.3 if small else 1.8, alpha=0.9, zorder=3 if tier == "B" else 2)
    la, lb = st["latency_unsaturated"]["A_submitted"], st["latency_unsaturated"]["B_profiled"]
    lr = st["latency_unsaturated"]["A_recost"]
    txt = (f"unsaturated rows (n = {st['n_unsaturated']}), predicted − measured:\n"
           f"  A · submitted inputs  {la['mean_ms']:+.2f} ms\n"
           f"  A · recost inputs     {lr['mean_ms']:+.2f} ms\n"
           f"  B · profiled inputs   {lb['mean_ms']:+.2f} ms")
    ax.text(0.02, 0.74, txt, transform=ax.transAxes, va="top", ha="left", fontsize=fz["note"] * (0.8 if small else 1.0), family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.8), zorder=6)
    # the 22 unsaturated rows sit within 1.1 ms of each other on the board; the inset spreads them out
    un = [r for r in rows if not r["saturated"]]
    ins = ax.inset_axes([0.30, 0.09, 0.30, 0.26])
    for tier, key in (("A", "chain_assumed_submitted_ms"), ("B", "chain_model_queue_ms")):
        t = TIER[tier]
        ins.plot([r["chain_measured_ms"] for r in un], [r[key] for r in un], ls="none", marker=t["marker"], ms=ms * 0.55,
                 mfc=t["data"], mec=t["data"], alpha=0.85)
    xs = [r["chain_measured_ms"] for r in un]
    ins.set_xlim(min(xs) - 0.2, max(xs) + 0.2); ins.set_ylim(27.5, 29.2)
    ins.tick_params(labelsize=fz["tick"] * 0.7); ins.set_title("unsaturated rows, zoomed", fontsize=fz["note"] * 0.85, pad=2)
    ins.grid(True, color="0.92", lw=0.5)
    ax.indicate_inset_zoom(ins, edgecolor="0.5", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("measured on K1 (ms)", fontsize=fz["lab"]); ax.set_ylabel("predicted (ms)", fontsize=fz["lab"])
    ax.tick_params(labelsize=fz["tick"]); tidy(ax)
    ax.set_title("camera → goal latency", fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 1.9)
    return {"x": "chain_measured_ms", "y": {"A": "chain_assumed_submitted_ms", "B": "chain_model_queue_ms"}}


def panel_ctrl(ax, D, st, fz, small=False):
    rows = D["rows"]; ms = 3.2 if small else 7.5
    ax.plot([0, 105], [0, 105], color="0.25", lw=1.1, ls=(0, (4, 3)), zorder=1)
    for tier, key in (("A", "ctrl_assumed_submitted_hz"), ("B", "ctrl_pred_hz")):
        t = TIER[tier]
        for sat in (False, True):
            rs = [r for r in rows if r["saturated"] == sat]
            ax.plot([r["ctrl_measured_hz"] for r in rs], [r[key] for r in rs], ls="none", marker=t["marker"], ms=ms,
                    mfc=("none" if sat else t["data"]), mec=t["data"], mew=1.3 if small else 1.8, alpha=0.9, zorder=3 if tier == "B" else 2)
    ca, cb = st["ctrl_all_rows"]["A_submitted"], st["ctrl_all_rows"]["B_profiled"]
    cr = st["ctrl_all_rows"]["A_recost"]
    txt = (f"all rows (n = {st['n_rows']}), |predicted − measured|:\n"
           f"  A · submitted  mean {ca['mean_abs_hz']:.2f}  max {ca['max_abs_hz']:.2f} Hz\n"
           f"  A · recost     mean {cr['mean_abs_hz']:.2f}  max {cr['max_abs_hz']:.2f} Hz\n"
           f"  B · profiled   mean {cb['mean_abs_hz']:.2f}  max {cb['max_abs_hz']:.2f} Hz")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=fz["note"] * (0.8 if small else 1.0), family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.8), zorder=6)
    # the one cluster where the input sets separate: control chained to the goal, above saturation
    cp = [r for r in rows if r["ctrl_mode"] == "chained" and r["saturated"]]
    if cp:
        mx = statistics.median(r["ctrl_measured_hz"] for r in cp)
        ax.annotate(f"control chained to the goal,\nsaturated: A {cp[0]['ctrl_assumed_submitted_hz']:.1f} Hz\n"
                    f"B {cp[0]['ctrl_pred_hz']:.1f} Hz · measured\n{min(r['ctrl_measured_hz'] for r in cp):.1f}–"
                    f"{max(r['ctrl_measured_hz'] for r in cp):.1f} Hz",
                    xy=(mx, cp[0]["ctrl_pred_hz"]), xytext=(58, 14), fontsize=fz["note"], ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0))
    ax.set_xlim(0, 105); ax.set_ylim(0, 105); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("measured on K1 (Hz)", fontsize=fz["lab"]); ax.set_ylabel("predicted (Hz)", fontsize=fz["lab"])
    ax.tick_params(labelsize=fz["tick"]); tidy(ax)
    ax.set_title("control command rate", fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 1.9)
    return {"x": "ctrl_measured_hz", "y": {"A": "ctrl_assumed_submitted_hz", "B": "ctrl_pred_hz"}}


def panel_ladder(ax, D, fz, field="ms", compact=False):
    """One line per (arm, rate) row: A and B as their error against C, so every row is visible, not a cluster."""
    rows = D["rows"]; n = len(rows); ys = np.arange(n)[::-1]
    if field == "ms":
        keys = (("A", "chain_assumed_submitted_ms"), ("B", "chain_model_queue_ms")); meas = "chain_measured_ms"; unit = "ms"
    else:
        keys = (("A", "ctrl_assumed_submitted_hz"), ("B", "ctrl_pred_hz")); meas = "ctrl_measured_hz"; unit = "Hz"
    for y, r in zip(ys, rows):
        if r["saturated"]:
            ax.axhspan(y - 0.5, y + 0.5, color="#f1f1f1", zorder=0)
        for tier, k in keys:
            t = TIER[tier]
            ax.plot([r[k] - r[meas]], [y + (0.17 if tier == "A" else -0.17)], ls="none", marker=t["marker"], ms=4.2 if compact else 6.5,
                    mfc=("none" if r["saturated"] else t["data"]), mec=t["data"], mew=1.3, zorder=3)
    ax.axvline(0, color=TIER["C"]["data"], lw=1.6, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels([f"{r['arm']} @ {r['camera_hz']} Hz" + ("  (sat.)" if r["saturated"] else "") for r in rows],
                                          fontsize=fz["tick"] * (0.72 if compact else 0.8))
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(f"predicted − measured ({unit})", fontsize=fz["lab"]); ax.tick_params(axis="x", labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color="0.92", lw=0.6)
    ax.set_title(("camera → goal error per row" if field == "ms" else "control-rate error per row"),
                 fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 1.9)


def tier_legend(fig, fz, loc, ncol=3, anchor=None):
    h = [Line2D([0], [0], ls="none", marker=TIER["A"]["marker"], mfc=TIER["A"]["data"], mec=TIER["A"]["data"], ms=8,
                label="A · analytical model, submitted inputs (XPU-RT's per-dispatch costs)"),
         Line2D([0], [0], ls="none", marker=TIER["B"]["marker"], mfc=TIER["B"]["data"], mec=TIER["B"]["data"], ms=8,
                label="B · same recurrence, per-node costs from the board's ROS 2 traces (+ named queue term)"),
         Line2D([0], [0], color=TIER["C"]["data"], lw=2.2, label="C · measured on K1 (x axis, or the 0 line)"),
         Line2D([0], [0], ls="none", marker="o", mfc="none", mec="0.35", ms=8, label="hollow = saturated row")]
    return fig.legend(handles=h, loc=loc, ncol=ncol, fontsize=fz["leg"], frameon=False, bbox_to_anchor=anchor)


def render_three_tier(D, st, layout, variant=""):
    fz = fonts(layout)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fz["base"], "pdf.fonttype": 42})
    out = {}
    if layout == "wide" and variant == "":
        fig = plt.figure(figsize=(16.0, 9.0))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.34, left=0.05, right=0.985, top=0.80, bottom=0.17)
        axs = [fig.add_subplot(gs[i]) for i in range(3)]
        out["a"] = panel_latency(axs[0], D, st, fz); out["b"] = panel_ctrl(axs[1], D, st, fz); panel_ladder(axs[2], D, fz, "ms")
        for ax, s, tiers in zip(axs, "abc", (["A", "B", "C"],) * 3):
            letter(ax, s, fz); out.setdefault("badges", {})[s] = badges(ax, tiers, fz)
        fig.suptitle("The ROS 2 baseline at three tiers: the submitted model, the same model on board-profiled costs, the board",
                     x=0.05, y=0.965, ha="left", fontsize=fz["head"], weight="bold")
        fig.text(0.05, 0.905, f"deployed camera → perception → nav → control chain · {st['n_rows']} (arm, camera rate) rows over "
                 f"{len({r['arm'] for r in D['rows']})} pinned arms, 5–120 Hz, {sum(len(r['runs']) for r in D['rows'])} board runs",
                 fontsize=fz["note"] * 1.15, color="0.3", ha="left")
        tier_legend(fig, fz, "lower center", ncol=2, anchor=(0.5, 0.0))
    elif layout == "wide" and variant == "ladder":
        fig = plt.figure(figsize=(16.0, 9.0))
        gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.11, right=0.985, top=0.83, bottom=0.17)
        a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        panel_ladder(a0, D, fz, "ms"); panel_ladder(a1, D, fz, "hz")
        la, lb = st["latency_unsaturated"]["A_submitted"], st["latency_unsaturated"]["B_profiled"]
        a0.text(0.62, 0.45, f"unsaturated mean: A {la['mean_ms']:+.2f} · B {lb['mean_ms']:+.2f} ms", transform=a0.transAxes,
                ha="right", va="bottom", fontsize=fz["note"], bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"))
        ca, cb = st["ctrl_all_rows"]["A_submitted"], st["ctrl_all_rows"]["B_profiled"]
        a1.text(0.55, 0.45, f"mean |error|: A {ca['mean_abs_hz']:.2f} · B {cb['mean_abs_hz']:.2f} Hz", transform=a1.transAxes,
                ha="left", va="bottom", fontsize=fz["note"], bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"))
        for ax, s in zip((a0, a1), "ab"):
            letter(ax, s, fz, dx=-0.02); out.setdefault("badges", {})[s] = badges(ax, ["A", "B", "C"], fz)
        fig.suptitle("Every row, both model tiers, against the board", x=0.11, y=0.965, ha="left", fontsize=fz["head"], weight="bold")
        fig.text(0.11, 0.905, "grey band = saturated row: the recurrence has no queue, tier B adds the one named queue term; "
                 "0 line = measured on K1", fontsize=fz["note"] * 1.15, color="0.3", ha="left")
        tier_legend(fig, fz, "lower center", ncol=2, anchor=(0.5, 0.0))
    else:   # column: one paper column, the two scatters stacked
        fig = plt.figure(figsize=(3.5, 8.6))
        gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.17, right=0.97, top=0.86, bottom=0.16)
        a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        out["a"] = panel_latency(a0, D, st, fz, small=True); out["b"] = panel_ctrl(a1, D, st, fz, small=True)
        for ax, s in zip((a0, a1), "ab"):
            letter(ax, s, fz, dx=-0.13); out.setdefault("badges", {})[s] = badges(ax, ["A", "B", "C"], fz)
        fig.suptitle("ROS 2 baseline: model tiers vs the board", x=0.03, y=0.995, ha="left", fontsize=fz["head"] * 0.95, weight="bold")
        h = [Line2D([0], [0], ls="none", marker=TIER["A"]["marker"], mfc=TIER["A"]["data"], mec=TIER["A"]["data"], ms=5, label="A · analytical, submitted inputs"),
             Line2D([0], [0], ls="none", marker=TIER["B"]["marker"], mfc=TIER["B"]["data"], mec=TIER["B"]["data"], ms=5, label="B · same model, board-profiled costs"),
             Line2D([0], [0], ls="none", marker="o", mfc="none", mec="0.35", ms=5, label="hollow = saturated row"),
             Line2D([0], [0], color="0.25", ls=(0, (4, 3)), label="y = x: C, measured on K1")]
        fig.legend(handles=h, loc="lower left", ncol=1, fontsize=fz["leg"], frameon=False, bbox_to_anchor=(0.02, 0.0))
    return fig, out


# ------------------------------------------------------------------------ submitted-metric panels
def submitted_numbers(D):
    sp, rows = D["spec"], D["rows"]; k = sp["reported_instance"]
    r = row_of(D, PAPER_ARM, PAPER_HZ)
    lit = RM.CONTROL_RATE_PANEL
    xgap = FC.ctrl_gap_ms(XPU_TRACE)
    return {
        "instance": k, "period_ms": sp["yolo_period_ms"], "budget_ms": sp["budget_ms"],
        "response_ms": {"A_submitted": sp["response_submitted_ms"][k], "A_recost": sp["response_assumed_ms"][k],
                        "A_recost_as_reported": sp["reported_response_ms"], "B_profiled": sp["response_profiled_ms"][k],
                        "xpu_as_reported": sp["reported_xpu_response_ms"]},
        "per_frame_ms": {"A_submitted": RP.ASSUMED["submitted"]["perception"], "A_recost": RP.ASSUMED["recost"]["perception"],
                         "B_profiled": sp["perception_profiled_ms"]},
        "ratio_to_xpu": {"A_recost_as_reported": sp["reported_response_ms"] / sp["reported_xpu_response_ms"],
                         "A_submitted": sp["response_submitted_ms"][k] / sp["reported_xpu_response_ms"],
                         "B_profiled": sp["response_profiled_ms"][k] / sp["reported_xpu_response_ms"]},
        "series_ms": {"A_submitted": sp["response_submitted_ms"], "A_recost": sp["response_assumed_ms"],
                      "B_profiled": sp["response_profiled_ms"]},
        "control": {"literal_response_ms": lit["ros_worst_response_ms"],
                    "literal_rate_hz": RM.zoh_rate_hz(lit["ros_worst_response_ms"], lit["control_period_ms"]),
                    "literal_source": literal_source(),
                    "A_control_node_ms": RP.ASSUMED["submitted"]["control"], "A_rate_hz": r["ctrl_assumed_submitted_hz"],
                    "B_control_node_ms": D["costs"][("control", 0)]["med"], "B_rate_hz": r["ctrl_pred_hz"],
                    "C_gap_ms": r["ctrl_measured_gap_ms"], "C_rate_hz": r["ctrl_measured_hz"],
                    "C_runs": r["runs"], "arm": PAPER_ARM, "camera_hz": PAPER_HZ, "ctrl_mode": r["ctrl_mode"],
                    "xpu_trace": XPU_TRACE, "xpu_gap_ms": xgap, "xpu_rate_hz": 1000.0 / xgap}}


def panel_response_bars(ax, N, fz, small=False):
    b, rsp = N["budget_ms"], N["response_ms"]
    items = [("ROS 2 · submitted inputs\n%.2f ms per frame" % N["per_frame_ms"]["A_submitted"], rsp["A_submitted"], "A"),
             ("ROS 2 · recost inputs\n%.2f ms per frame" % N["per_frame_ms"]["A_recost"], rsp["A_recost"], "A"),
             ("ROS 2 · board-profiled inputs\n%.2f ms per frame" % N["per_frame_ms"]["B_profiled"], rsp["B_profiled"], "B"),
             ("XPU-RT, as reported\n(board-recost schedule)", rsp["xpu_as_reported"], "X")]
    ys = np.arange(len(items))[::-1]; h = 0.62
    xmax = max(v for _, v, _ in items + [("", rsp["A_recost_as_reported"], "")]) * 1.22
    ax.axvspan(b, xmax, color="#fdeeee", zorder=0)
    for y, (lab, v, t) in zip(ys, items):
        col = C_XPU if t == "X" else TIER[t]["data"]
        ax.barh(y, v, height=h, color=col, edgecolor="white", zorder=3, hatch=("//" if t == "A" else None))
        met = v <= b + 0.01
        ax.text(v + xmax * 0.012, y, f"{v:.2f} ms " + ("✓" if met else "✗"), va="center", ha="left",
                fontsize=fz["lab"] * (0.85 if small else 0.95), weight="bold", color=("#1d6b38" if met else "#8f1d18"), zorder=5)
    ya = ys[1]
    ax.plot([rsp["A_recost_as_reported"]], [ya + h * 0.5 + 0.08], marker="v", ms=8 if not small else 5, color="0.15", zorder=6)
    ax.text(rsp["A_recost_as_reported"], ya + h * 0.5 + 0.22, f"as reported {rsp['A_recost_as_reported']:.2f} ms",
            ha="center", va="bottom", fontsize=fz["note"], color="0.15")
    ax.axvline(b, color="#b3261e", lw=2.0, ls=(0, (5, 3)), zorder=4)
    ax.text(b, ys[0] + 0.62, f"budget {b:.0f} ms", color="#8f1d18", ha="center", va="bottom", fontsize=fz["note"], weight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _, _ in items], fontsize=fz["tick"] * (0.92 if small else 0.95))
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(items) - 0.1)
    ax.set_xlabel(f"perception instance {N['instance']}: release → output (ms)", fontsize=fz["lab"])
    ax.tick_params(axis="x", labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    rt = N["ratio_to_xpu"]
    ax.text(0.99, 0.04, f"ratio to XPU-RT\n{rt['A_recost_as_reported']:.2f}× as reported\n→ {rt['B_profiled']:.2f}× board-profiled",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=fz["note"], color="0.2",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    ax.set_title(f"the submitted metric: one perception frame at a {N['period_ms']:.0f} ms period",
                 fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 1.9)


def panel_response_series(ax, N, fz, small=False):
    ks = np.arange(len(N["series_ms"]["A_submitted"]))
    for key, lab, t, ls in (("A_recost", "recost inputs", "A", (0, (5, 2))), ("A_submitted", "submitted inputs", "A", "-"),
                            ("B_profiled", "board-profiled inputs", "B", "-")):
        ax.plot(ks, N["series_ms"][key], color=TIER[t]["data"], ls=ls, lw=2.2 if not small else 1.4, marker=TIER[t]["marker"],
                ms=7 if not small else 4, label=lab, zorder=3)
    ax.axhline(N["budget_ms"], color="#b3261e", lw=2.0, ls=(0, (5, 3)))
    ax.axhline(N["response_ms"]["xpu_as_reported"], color=C_XPU, lw=2.0)
    ax.text(ks[-1], N["response_ms"]["xpu_as_reported"] - 1.0, "XPU-RT as reported", color=C_XPU, ha="right", va="top", fontsize=fz["note"])
    ax.text(1.5, N["budget_ms"] + 0.8, f"budget {N['budget_ms']:.0f} ms", color="#8f1d18", ha="center", va="bottom", fontsize=fz["note"])
    ax.axvline(N["instance"], color="0.6", lw=1.0, ls=":")
    ax.text(N["instance"] + 0.05, 64, "instance the\nfigure reports", fontsize=fz["note"], color="0.35", va="top")
    ax.set_xticks(ks); ax.set_xlabel("perception instance (5 in the horizon)", fontsize=fz["lab"])
    ax.set_ylabel("release → output (ms)", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"])
    ax.set_ylim(15, 68); ax.legend(fontsize=fz["leg"], loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.88)); tidy(ax)
    ax.set_title("the recurrence falls behind each frame", fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 1.9)


def panel_control(ax, N, fz, small=False, compact_labels=False):
    c = N["control"]
    items = [(f"submitted figure: {c['literal_response_ms']:.2f} ms = predicted dronet response\n"
              "on a different workload (microros_baseline_k1.json),\npassed as the control response → hold at 10 ms",
              c["literal_rate_hz"], "A", "literal"),
             (f"analytical model, control node {c['A_control_node_ms']:.3f} ms", c["A_rate_hz"], "A", ""),
             (f"calibrated model, control node {c['B_control_node_ms']:.3f} ms", c["B_rate_hz"], "B", ""),
             (f"measured, pinned arm `{c['arm']}` @ {c['camera_hz']} Hz camera\n"
              f"control on its own timer, gap {c['C_gap_ms']:.2f} ms", c["C_rate_hz"], "C", ""),
             (f"XPU-RT, measured, gap {c['xpu_gap_ms']:.2f} ms", c["xpu_rate_hz"], "X", "")]
    if compact_labels:
        items = [(f"{c['literal_response_ms']:.2f} ms: predicted dronet\nresponse, other workload,\npassed as control response", c["literal_rate_hz"], "A", "literal"),
                 ("analytical model", c["A_rate_hz"], "A", ""), ("calibrated model", c["B_rate_hz"], "B", ""),
                 (f"measured `{c['arm']}` @ {c['camera_hz']} Hz", c["C_rate_hz"], "C", ""), ("XPU-RT, measured", c["xpu_rate_hz"], "X", "")]
    ys = np.arange(len(items))[::-1]
    for y, (lab, v, t, kind) in zip(ys, items):
        col = C_XPU if t == "X" else TIER[t]["data"]
        if kind == "literal":
            ax.barh(y, v, height=0.62, facecolor="white", edgecolor=TIER["A"]["fc"], hatch="xx", lw=1.6, zorder=3)
        else:
            ax.barh(y, v, height=0.62, color=col, edgecolor="white", zorder=3, hatch=("//" if t == "A" else None))
        ax.text(v + 1.5, y, f"{v:.0f} Hz" if abs(v - round(v)) < 0.05 else f"{v:.1f} Hz", va="center", ha="left",
                fontsize=fz["lab"] * (0.85 if small else 0.95), weight="bold", color="0.15")
    ax.set_yticks(ys); ax.set_yticklabels([l for l, _, _, _ in items], fontsize=fz["tick"] * (0.8 if small else 0.9))
    ax.set_xlim(0, 125); ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_xlabel("control command rate (Hz)", fontsize=fz["lab"])
    ax.tick_params(axis="x", labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color="0.92", lw=0.6)
    ax.set_title("the control-rate label: what was printed vs predicted vs measured", fontsize=fz["title"], weight="bold",
                 loc="left", pad=fz["base"] * 1.9)


def render_submitted(D, layout):
    fz = fonts(layout); N = submitted_numbers(D)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fz["base"], "pdf.fonttype": 42})
    out = {"numbers": N, "badges": {}}
    if layout == "wide":
        fig = plt.figure(figsize=(16.0, 9.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.30, hspace=0.62,
                              left=0.17, right=0.985, top=0.83, bottom=0.08)
        a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, :])
        panel_response_bars(a0, N, fz); panel_response_series(a1, N, fz); panel_control(a2, N, fz)
        for ax, s, tiers, dx in ((a0, "a", ["A", "B"], -0.33), (a1, "b", ["A", "B"], -0.1), (a2, "c", ["A", "B", "C"], -0.24)):
            letter(ax, s, fz, dx=dx); out["badges"][s] = badges(ax, tiers, fz)
        fig.suptitle("The submitted showdown's own numbers, re-costed from the board's ROS 2 traces", x=0.02, y=0.965, ha="left",
                     fontsize=fz["head"], weight="bold")
        fig.text(0.02, 0.905, "same recurrence, same spec (22 ms period, 5 instances, 23 ms budget); only the per-node costs change. "
                 "The pinned baseline misses the budget on every input; the margin halves.", fontsize=fz["note"] * 1.15, color="0.3", ha="left")
    else:
        fig = plt.figure(figsize=(3.5, 6.6))
        gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.44, right=0.95, top=0.88, bottom=0.08)
        a0, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        panel_response_bars(a0, N, fz, small=True); panel_control(a2, N, fz, small=True, compact_labels=True)
        a0.set_title("perception instance 2, 22 ms period", fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 2.2)
        a2.set_title("ROS 2 control command rate", fontsize=fz["title"], weight="bold", loc="left", pad=fz["base"] * 2.2)
        a0.texts[-1].set_visible(False)   # the ratio note does not fit a column; it is in the sidecar and the caption
        a0.set_yticklabels(["submitted\ninputs", "recost\ninputs", "board-profiled\ninputs", "XPU-RT,\nas reported"], fontsize=fz["tick"])
        for ax, s, tiers in ((a0, "a", ["A", "B"]), (a2, "b", ["A", "B", "C"])):
            letter(ax, s, fz, dx=-0.72); out["badges"][s] = badges(ax, tiers, fz)
        fig.suptitle("Submitted metric, re-costed", x=0.03, y=0.985, ha="left", fontsize=fz["head"] * 0.95, weight="bold")
    return fig, out


# ---------------------------------------------------------------------------------- story panels
def census(D=None):
    """The census the flight panels draw: every quarantine-cleared flight of the two arms on course A, goal
    hold 0, each at the latency the registry says it was launched with (figure_constants.lat_ms)."""
    sel = {XPU_TRACE: FC.lat_ms(XPU_TRACE), ROS_TRACE: FC.lat_ms(ROS_TRACE)}
    rows = [r for r in flight_rows(CENSUS_CSV) if r["course"] == "a" and float(r["percep_hold_ms"]) == 0.0
            and r["ctrl_trace"] in sel and abs(float(r["percep_latency_ms"]) - sel[r["ctrl_trace"]]) < 0.05]
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0.0, 0]))
    cells = collections.defaultdict(dict)
    for r in rows:
        c = per[r["ctrl_trace"]][r["cruise_speed"]]
        c[0] += r["outcome"] == "success"; c[1] += 1; c[2] += float(r["gates_passed"]); c[3] += r["outcome"] == "timeout"
        cells[(r["cruise_speed"], r["prop_density"], r["seed"])][r["ctrl_trace"]] = r
    out = {"csv": os.path.relpath(CENSUS_CSV, REPO), "selector_ms": sel, "per_cruise": {}, "pooled": {}}
    for t in sel:
        out["per_cruise"][t] = {cr: {"k": v[0], "n": v[1], "mean_gates": v[2] / v[1], "timeouts": v[3],
                                     "wilson": list(wilson(v[0], v[1]))} for cr, v in sorted(per[t].items())}
        k = sum(v[0] for v in per[t].values()); n = sum(v[1] for v in per[t].values())
        g = sum(v[2] for v in per[t].values())
        out["pooled"][t] = {"k": k, "n": n, "mean_gates": g / n, "timeouts": sum(v[3] for v in per[t].values()), "wilson": list(wilson(k, n))}
    px, pr = out["pooled"][XPU_TRACE], out["pooled"][ROS_TRACE]
    out["difference_xpu_minus_ros"] = list(newcombe(px["k"], px["n"], pr["k"], pr["n"]))
    both = [c for c in cells.values() if len(c) == 2]
    ahead = sum(float(c[XPU_TRACE]["gates_passed"]) > float(c[ROS_TRACE]["gates_passed"]) for c in both)
    behind = sum(float(c[XPU_TRACE]["gates_passed"]) < float(c[ROS_TRACE]["gates_passed"]) for c in both)
    out["paired_cells"] = {"n": len(both), "xpu_ahead": ahead, "level": len(both) - ahead - behind, "xpu_behind": behind}
    return out


def cell_records(trace, lat, cruise, dens):
    """The per-flight records of one census cell, matched on the trace each record says it replayed."""
    pat = os.path.join(RECORDS, f"*_lat{lat:g}_h0_a_d{dens:.2f}_w0.0_g0.0055_c{cruise:.1f}", "ep*.npz")
    out = []
    for p in sorted(glob.glob(pat)):
        d = np.load(p, allow_pickle=True)
        if os.path.basename(str(d["ctrl_trace"])) == trace:
            out.append((p, d))
    return out


def panel_flights(axs, cruise, dens, fz, small=False):
    """Every census flight of one (cruise, density) cell, one lane per arm. The scene is drawn per seed, so
    only the gates -- fixed by the course -- are common; the paths show how far each flight got."""
    rec = {}
    for ax, trace, col, name in ((axs[0], XPU_TRACE, C_XPU, "XPU-RT · CP-SAT"), (axs[1], ROS_TRACE, C_ROS, "ROS 2 · pinned (p3)")):
        rs = cell_records(trace, FC.lat_ms(trace), cruise, dens); g = None; k = 0; res = []
        for p, d in rs:
            xy = d["poses"][:, :2]; g = d["gates_world"]; ok = str(d["outcome"]) == "success"; k += ok
            ax.plot(xy[:, 1], xy[:, 0], color=col, lw=1.0 if small else 1.6, alpha=0.55, zorder=3)
            ax.plot(xy[-1, 1], xy[-1, 0], ls="none", marker=("o" if ok else "X"), ms=(4 if small else 8), mfc=(col if ok else "white"),
                    mec=col, mew=1.2, zorder=4)
            res.append({"record": os.path.relpath(p, REPO), "seed": int(d["seed"]), "outcome": str(d["outcome"]),
                        "gates": int(d["gates_passed"])})
        if g is not None:
            for i, (gx, gy) in enumerate(g):
                ax.plot([gy, gy], [gx - 0.9, gx + 0.9], color="#e0b400", lw=4 if not small else 2.4, solid_capstyle="round", zorder=5)
                ax.text(gy, gx + 1.25, f"G{i + 1}", ha="center", va="bottom", fontsize=fz["note"], color="#8a6d00", weight="bold")
        ax.set_ylim(-10.6, -5.4); ax.set_xlim(5.0, 23.0); ax.invert_yaxis()
        ax.set_yticks([]); ax.tick_params(axis="x", labelsize=fz["tick"])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.text(0.005, 0.03, f"{name}: {k}/{len(rs)} complete", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=fz["lab"] * (0.9 if small else 1.0), weight="bold", color=col)
        rec[trace] = {"k": k, "n": len(rs), "flights": res}
    axs[1].set_xlabel("along the aisle (m)", fontsize=fz["lab"]); axs[0].tick_params(axis="x", labelbottom=False)
    return rec


def panel_census(ax, C, fz, small=False):
    for t, col, lab, dx in ((XPU_TRACE, C_XPU, "XPU-RT · CP-SAT", -0.025), (ROS_TRACE, C_ROS, "ROS 2 · pinned (p3)", 0.025)):
        pc = C["per_cruise"][t]; xs = [float(c) + dx for c in pc]
        p = [v["wilson"][0] for v in pc.values()]; lo = [v["wilson"][0] - v["wilson"][1] for v in pc.values()]
        hi = [v["wilson"][2] - v["wilson"][0] for v in pc.values()]
        ax.errorbar(xs, p, yerr=[lo, hi], color=col, marker="o", ms=5 if small else 8, lw=1.4 if small else 2.2, capsize=3,
                    label=f"{lab}: {C['pooled'][t]['k']}/{C['pooled'][t]['n']} pooled")
    d = C["difference_xpu_minus_ros"]
    ax.text(0.02, 0.98, f"XPU-RT − ROS 2: {100 * d[0]:+.1f} pts [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]\n"
            f"paired cells, XPU-RT: ahead {C['paired_cells']['xpu_ahead']} · level {C['paired_cells']['level']} · "
            f"behind {C['paired_cells']['xpu_behind']}", transform=ax.transAxes, ha="left", va="top", fontsize=fz["note"] * 0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    ax.set_ylim(0, 0.8); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("completion (Wilson 95 %)", fontsize=fz["lab"])
    ax.tick_params(labelsize=fz["tick"]); ax.legend(fontsize=fz["leg"], loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False); tidy(ax)


def tierb_schedule(D, arm=PAPER_ARM, hz=PAPER_HZ, window_ms=100.0):
    """The tier-B timeline of the pinned arrangement, as a schedule document draw_combined_gantt can draw.

    Each node is ros_pinning_model.pin_timeline -- the model's own recurrence, queued executor -- on the
    board-profiled costs, placed on the partitions the measured board Gantt of the same arm records (so
    the only thing different between the two drawn rows is model versus board). The camera callback
    shares the perception partition in this arm, so the perception node's service is camera + perception
    (ros_pinning_profiled.predict's `service_ms`), drawn as its two parts.
    """
    meas = json.load(open(f"{GANTT_PREFIX}_{arm}.json")); md = meas["metadata"]
    harts = collections.defaultdict(set)
    for v in meas["dispatches"].values():
        harts[S.netinfo(v["job_name"])[0]].update(v["hardware_target"].split("+"))
    c = D["costs"]; cp, cc = c[("perception", 4)]["med"], c[("camera", 0)]["med"]
    cn, ck = c[("nav", 0)]["med"], c[("control", 0)]["med"]
    T = 1000.0 / hz; pr = RP.predict({"perception": cp, "nav": cn, "control": ck, "camera": cc}, hz, "timer", 100, 10, True)
    n_p = int(np.ceil(window_ms / T)); n_n = int(np.ceil(window_ms / md["nav_period_ms"])); n_c = int(np.ceil(window_ms / md["ctrl_period_ms"]))
    tl = {"perception": RM.pin_timeline(pr["service_ms"], T, n_p, "queued"),
          "nav": RM.pin_timeline(cn, md["nav_period_ms"], n_n, "queued"),
          "control": RM.pin_timeline(ck, md["ctrl_period_ms"], n_c, "queued")}
    P = "+".join(sorted(harts["yolo"])); disp = {}

    def add(job, hw, s, d):
        disp[str(len(disp))] = {"id": len(disp), "hardware_target": hw, "start_time": s, "duration": d, "job_name": job}
    for x in tl["perception"]:
        add(f"camera{x['instance']}", P, x["start_ms"], cc)
        add(f"yolov8_nano_64x96{x['instance']}", P, x["start_ms"] + cc, cp)
    for x in tl["nav"]:
        add(f"fused_full{x['instance']}", "+".join(sorted(harts["nav"])), x["start_ms"], cn)
    for x in tl["control"]:
        add(f"mlp_control{x['instance']}", "+".join(sorted(harts["ctrl"])), x["start_ms"], ck)
    meta = {"yolo_period_ms": T, "nav_period_ms": md["nav_period_ms"], "ctrl_period_ms": md["ctrl_period_ms"], "busy_pct": {},
            "placement_note": "partitions as measured · costs board-profiled", "ctrl_gap_mean_ms": 1000.0 / pr["ctrl_pred_hz"]}
    facts = {"camera_hz": hz, "service_ms": pr["service_ms"], "costs_ms": {"camera": cc, "perception": cp, "nav": cn, "control": ck},
             "chain_model_ms": pr["chain_model_ms"], "queue_ms": pr["queue_ms"], "chain_model_queue_ms": pr["chain_model_queue_ms"],
             "ctrl_pred_hz": pr["ctrl_pred_hz"], "perception_response_ms": [x["response_ms"] for x in tl["perception"]],
             "partitions": {k: sorted(v) for k, v in harts.items()}, "measured_gantt": os.path.relpath(f"{GANTT_PREFIX}_{arm}.json", REPO)}
    return {"metadata": meta, "dispatches": disp}, facts


def wrap_to(ax, text, fs):
    """Wrap a left title to the width its axes actually occupies, so a long title never runs into the next panel."""
    w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    return "\n".join(textwrap.fill(ln, max(30, int(w_pt / (0.56 * fs)))) for ln in text.split("\n"))


def panel_tierb_gantt(ax, doc, facts, row, fz, k):
    S.draw_combined_gantt(ax, [(doc, "ROS 2 · pinned, calibrated model", TIER["B"]["data"], "ros")])
    V.rescale_fonts(ax, k)
    for t in list(ax.texts):
        s = t.get_text()
        if s.startswith("camera→control") or s.startswith("sensors in"):
            t.set_visible(False)
    ax.set_xlabel("model time (ms) · calibrated model, not a board trace", fontsize=fz["lab"])
    fs = fz["title"] * 0.8
    ax.set_title(wrap_to(ax, f"Calibrated model, pinned arm, {facts['camera_hz']} Hz camera: pin_timeline on board-profiled costs\n"
                         f"camera→goal {facts['chain_model_ms']:.1f} + queue {facts['queue_ms']:.1f} = {facts['chain_model_queue_ms']:.1f} ms "
                         f"vs {row['chain_measured_ms']:.1f} measured · control {facts['ctrl_pred_hz']:.0f} vs "
                         f"{row['ctrl_measured_hz']:.1f} Hz\n"
                         f"without a drop rule the response grows each frame ({facts['perception_response_ms'][0]:.1f} → "
                         f"{facts['perception_response_ms'][-1]:.1f} ms); the board stays flat", fs), fontsize=fs, weight="bold", loc="left")
    leg = ax.get_legend()
    if leg:
        leg.remove()
    ax.legend(handles=[Line2D([0], [0], color=S.C_CTRL, lw=7, label="CTRL (mlp)"), Line2D([0], [0], color=S.C_NAV, lw=7, label="NAV (fused)"),
                       Line2D([0], [0], color=S.C_YOLO, lw=7, label="YOLO"), Line2D([0], [0], color="#8aa", lw=7, label="camera callback")],
                loc="upper right", ncol=4, fontsize=fz["leg"], frameon=True, framealpha=0.95, bbox_to_anchor=(1.0, 1.0))


def render_story(D, layout, cruise=1.0, dens=0.30):
    fz = fonts("story")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fz["base"], "pdf.fonttype": 42})
    N = submitted_numbers(D); C = census(); row = row_of(D, PAPER_ARM, PAPER_HZ)
    doc, facts = tierb_schedule(D)
    out = {"numbers": N, "census": C, "tierb_schedule": facts, "tierb_row": row_record(row), "badges": {}}
    W, H = (20.0, 11.25) if layout == "wide" else (20.0, 25.0)
    fig = plt.figure(figsize=(W, H))
    if layout == "wide":
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.75, left=0.035, right=0.99, top=0.75, bottom=0.06)
        top = outer[0].subgridspec(1, 3, width_ratios=[1.3, 0.8, 1.0], wspace=0.48)
        fl = top[0].subgridspec(2, 1, hspace=0.18); axA = [fig.add_subplot(fl[0]), fig.add_subplot(fl[1])]
        axB = fig.add_subplot(top[1]); axC = fig.add_subplot(top[2])
        bot = outer[1].subgridspec(1, 2, wspace=0.08); axD = fig.add_subplot(bot[0]); axE = fig.add_subplot(bot[1])
        gk = 0.62
    else:
        outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.8, 1.0, 1.25], hspace=0.55, left=0.07, right=0.99, top=0.9, bottom=0.03)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.22)
        fl = top[0].subgridspec(2, 1, hspace=0.18); axA = [fig.add_subplot(fl[0]), fig.add_subplot(fl[1])]
        axB = fig.add_subplot(top[1]); axC = fig.add_subplot(outer[1])
        axD = fig.add_subplot(outer[2]); axE = fig.add_subplot(outer[3])
        gk = 0.9
    tfs = fz["title"] * 0.95
    out["flights"] = panel_flights(axA, cruise, dens, fz)
    axA[0].set_title(wrap_to(axA[0], f"Every census flight of one cell ({cruise:.1f} m/s, density {dens:.2f}): ● complete, ✕ crash", tfs), fontsize=tfs, weight="bold", loc="left")
    letter(axA[0], "A", fz, dx=-0.01, dy=1.02); out["badges"]["A"] = badges(axA[0], ["C"], fz)
    panel_census(axB, C, fz)
    axB.set_title(wrap_to(axB, "Census: all speeds and densities", tfs), fontsize=tfs, weight="bold", loc="left")
    letter(axB, "B", fz, dx=-0.14, dy=1.02); out["badges"]["B"] = badges(axB, ["C"], fz)
    panel_control(axC, N, fz, compact_labels=(layout == "wide"))
    axC.set_title(wrap_to(axC, "Control rate: printed · predicted · measured", tfs), fontsize=tfs, weight="bold", loc="left")
    letter(axC, "C", fz, dx=(-0.34 if layout == "wide" else -0.2), dy=1.02); out["badges"]["C"] = badges(axC, ["A", "B", "C"], fz)
    panel_tierb_gantt(axD, doc, facts, row, fz, gk)
    letter(axD, "D", fz, dx=-0.01, dy=1.02); out["badges"]["D"] = badges(axD, ["B"], fz)
    sides, MI = SP.gantt_two_rows(axE, dict(fz, tick=fz["tick"] * 0.62), fz["base"] * gk * 1.25, names=("xpu", "p3"), prefix=GANTT_PREFIX, camera_hz=PAPER_HZ)
    mx, mr = MI["xpu"], MI["p3"]
    axE.tick_params(axis="x", labelsize=fz["tick"])
    axE.set_title(wrap_to(axE, f"Measured on the K1, {PAPER_HZ} Hz camera, one 100 ms window of each arm's board trace\n"
                          f"XPU-RT · CP-SAT: camera→control {mx['chain_ms_median']:.0f} ms, control every {mx['ctrl_gap_mean_ms']:.1f} ms\n"
                          f"ROS 2 · pinned (p3): camera→control {mr['chain_ms_median']:.0f} ms, control every {mr['ctrl_gap_mean_ms']:.1f} ms, "
                          f"{mr['frames_late']}/{mr['frames_checked']} frames late", fz["title"] * 0.8),
                  fontsize=fz["title"] * 0.8, weight="bold", loc="left", pad=6)
    letter(axE, "E", fz, dx=-0.01, dy=1.02); out["badges"]["E"] = badges(axE, ["C"], fz)
    out["measured_gantt"] = {"sidecars": [os.path.relpath(s, REPO) for s in sides], **MI}
    px, pr = C["pooled"][XPU_TRACE], C["pooled"][ROS_TRACE]
    head = (f"Warehouse showdown, ROS 2 arm re-costed from the board's own traces: the arm the calibrated model describes "
            f"(pinned, control on its own timer) runs {row['ctrl_measured_hz']:.0f} Hz control on the K1 and flies level with XPU-RT")
    fig.suptitle("\n".join(textwrap.wrap(head, 125 if layout == "wide" else 120)), x=0.035, y=0.99, ha="left", va="top",
                 fontsize=fz["head"], weight="bold")
    sub = (f"No flight replays a model cadence: A–B fly the measured cadences of `{ROS_TRACE}` (the arm tier B predicts within "
           f"{abs(row['chain_measured_ms'] - row['chain_model_queue_ms']):.1f} ms camera→goal and "
           f"{abs(row['ctrl_measured_hz'] - row['ctrl_pred_hz']):.1f} Hz control rate) and `{XPU_TRACE}` in the simulator. "
           f"Census: XPU-RT {px['k']}/{px['n']}, ROS 2 {pr['k']}/{pr['n']} complete — a tie, not a win.")
    fig.text(0.035, 0.895 if layout == "wide" else 0.95, "\n".join(textwrap.wrap(sub, 150 if layout == "wide" else 160)), fontsize=fz["note"] * 1.15,
             color="0.25", ha="left", va="top")
    return fig, out, sides


# ---------------------------------------------------------------------------------------- output
STEMS = {  # stem -> (option, layout, variant)
    "ros_model_fidelity_three_tier": ("three_tier", "wide", ""),
    "ros_model_fidelity_three_tier_column": ("three_tier", "column", ""),
    "ros_model_fidelity_three_tier_ladder": ("three_tier", "wide", "ladder"),
    "ros_model_fidelity_submitted_metric": ("submitted_metric", "wide", ""),
    "ros_model_fidelity_submitted_metric_column": ("submitted_metric", "column", ""),
    "ros_model_fidelity_story": ("story", "wide", ""),
    "ros_model_fidelity_story_tall": ("story", "tall", ""),
}


def write(fig, stem, out_dir, side, inputs, dpi):
    base = os.path.join(out_dir, stem)
    for p in (base + ".png", base + ".pdf", base + "_metrics.json"):
        if os.path.abspath(out_dir) == os.path.abspath(REFINED) and os.path.exists(p) and not os.environ.get("RMF_REPLACE_OWN"):
            raise SystemExit(f"{p} exists; this script writes new stems only (set RMF_REPLACE_OWN=1 to re-render its own stems)")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight"); fig.savefig(base + ".pdf", bbox_inches="tight"); plt.close(fig)
    side = {"figure": os.path.relpath(base + ".png", REPO) if base.startswith(REPO) else base + ".png",
            "written": datetime.datetime.now().isoformat(timespec="seconds"), **side,
            **FC.sidecar_common(SCRIPT, [p for p in inputs if os.path.exists(p)])}
    json.dump(side, open(base + "_metrics.json", "w"), indent=1, default=float)
    print("wrote", base + ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--option", choices=["three_tier", "submitted_metric", "story", "all"], default="all")
    ap.add_argument("--out-dir", default=REFINED)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--cruise", type=float, default=1.0, help="the census cell panel A of the story draws")
    ap.add_argument("--density", type=float, default=0.30)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    # RMF_CACHE=<file.pkl> keeps the parsed traces between renders while a layout is being iterated;
    # the verifier never sets it, so every recorded number is re-derived from the traces there
    cache = os.environ.get("RMF_CACHE")
    if cache and os.path.exists(cache):
        import pickle
        D = pickle.load(open(cache, "rb"))
    else:
        D = gather()
        if cache:
            import pickle
            pickle.dump(D, open(cache, "wb"))
    st = residual_stats(D)
    common = {"tiers": {k: v["badge"] for k, v in TIER.items()},
              "tier_sources": {"A": "scripts/ros_pinning_profiled.py ASSUMED['submitted'] (and 'recost'), same recurrence",
                               "B": "scripts/ros_pinning_profiled.py pooled_inputs() + predict()",
                               "C": "results/codesign_feedback/ros_traced/<run>/summary.json (e2e_goal_med_ms, gap_mean_ms)"},
              "profiled_costs_ms": {f"{n}@pool{p}": v["med"] for (n, p), v in D["costs"].items()},
              "profiled_costs_runs": {f"{n}@pool{p}": v["n_runs"] for (n, p), v in D["costs"].items()}}
    for stem, (opt, layout, variant) in STEMS.items():
        if a.option not in ("all", opt):
            continue
        inputs = list(D["inputs"])
        if opt == "three_tier":
            fig, o = render_three_tier(D, st, layout, variant)
            side = {"option": opt, "layout": layout, "variant": variant, **common, "residuals": st,
                    "rows": [row_record(r) for r in D["rows"]], **o}
        elif opt == "submitted_metric":
            fig, o = render_submitted(D, layout)
            side = {"option": opt, "layout": layout, **common, **o}
            inputs += [SUBMITTED_SIDECAR, os.path.join(REPO, RM.CONTROL_RATE_PANEL["sidecar"]),
                       os.path.join(RES, "ctrl_traces", XPU_TRACE)]
        else:
            fig, o, sides = render_story(D, layout, a.cruise, a.density)
            side = {"option": opt, "layout": layout, **common, **o}
            inputs += [SUBMITTED_SIDECAR, os.path.join(REPO, RM.CONTROL_RATE_PANEL["sidecar"]), CENSUS_CSV,
                       os.path.join(RES, "flight_quarantine.csv"), os.path.join(RES, "ctrl_traces", XPU_TRACE),
                       os.path.join(RES, "ctrl_traces", ROS_TRACE), f"{GANTT_PREFIX}_p3.json", f"{GANTT_PREFIX}_xpu.json"]
            inputs += [os.path.join(REPO, f["record"]) for t in o["flights"].values() for f in t["flights"]] + sides
        write(fig, stem, a.out_dir, side, inputs, a.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
