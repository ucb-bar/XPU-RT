#!/usr/bin/env python3
"""One quarter-page figure that demonstrates the co-design feedback loop end to end.

WHY ONE FIGURE. The loop story is currently spread over three artifacts a reader has to
hold in their head at once: a pair of schematics that define what "inner" and "outer"
even mean, `ladder_scaling` which measures what each loop contributes, and
`schedule_evolution` which shows the four real schedules the loop walks through. Read
apart they invite the wrong reading -- the schematic looks like a claim with no evidence,
the ablation looks like a negative result with no mechanism, and the evolution Gantt looks
like one lucky workload. Stacked as three bands of one column-width block they are a
single argument: what the loops ARE, whether they WORK and how far they generalise, and
one real worked example on the K1.

THE THREE BANDS.

  A  WHAT THE LOOPS ARE -- schematic, and deliberately so. Two before/after pairs of
     miniature Gantts, drawn here rather than embedded from anywhere, because they are
     illustrations and not measurements and nothing about them should look like data.
       * inner loop: an AOT search over graph/codegen levers, scored on PREDICTED costs.
         A dispatch overruns the window; sharding it across harts and routing the matmul
         block to the IME pulls the schedule back inside.
       * outer loop: the same assignment RUN ON THE K1. Measured durations are longer
         than the model believed, so a deadline the offline solve promised is missed;
         re-solving against the measured costs recovers it.
     No axis numbers, no ticks, no legend -- the shapes carry it, and the real versions
     with real numbers are two bands down. Bar colours come from the shared palette so
     the schematic and the real Gantt in band C are the same visual language.

  B  WHETHER THEY WORK, including the negative half. The 2x2 ablation (A = neither loop,
     B = inner only, C = outer only, D = both) run up a ladder of 2..5 concurrent
     networks. Two findings, both visible:
       * the INNER loop does all the work that gets done, and only up to three networks
         (w2 5->0, w3 9->5 misses). Above that it is flat, and on w5 under CP-SAT it is
         NEGATIVE (13->17) -- the lever its measured search accepts there is `ime`, not
         `shard`, and IME does not help that taskset. The accepted lever is printed on
         the tick because it is the explanatory variable, not decoration.
       * the OUTER loop contributes exactly ZERO as measured. Cell C sits on cell A on
         every rung under both solvers, and D sits on B. That is a statement about our
         outer-loop MEASUREMENT, not about HIL: cell C re-solves against measured board
         costs but is also SCORED on them, so the discrepancy that makes the board step
         valuable in band C has been designed out of the cell. Band C is the evidence
         the outer loop does something; band B is the evidence we cannot yet score it.
     Keeping the null on the same figure as the schematic is deliberate. The schematic
     promises two loops; only one of them is currently paying in a way we can measure.

  C  ONE REAL EXAMPLE. The four actual schedules of the tight-vision-head sensor
     workload, in a single row instead of the original 2x2 so the band stays thin:
     baseline RVV singletons (2 FFN instances miss) -> AOT picks IME (all met on the
     Gantt) -> the same assignment replayed on K1-calibrated durations (a NAV miss the
     AOT model could not see) -> CP-SAT re-solve on those durations (met again). This is
     the arc band B says closes at two or three networks; this workload has five, but a
     TIGHT head rather than a contended one, which is why it still closes.

WHAT WAS DROPPED FOR SPACE, said out loud so nobody hunts for it. Band C keeps no
per-panel sub-caption and no per-core y tick labels (only the E-cluster / P-cluster
markers), and its x label appears once rather than per panel. Band B carries no per-bar
value labels -- those numbers are in the `_metrics.json` sidecar. At a quarter page
legibility is the binding constraint and overlapping text is the one unacceptable
outcome, so anything that did not fit was removed rather than shrunk further.

Reads only. Re-solves nothing, re-runs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import figstyle  # noqa: E402
import compose_schedule_evolution as evo  # noqa: E402  (band C's drawing helpers)

SHARD_HATCH, SHARD_EC = "\\\\\\\\", "#2a2a2a"
IME_HATCH, IME_EC = "////", "#0a6b6b"


def mm_axes(fig, fw, fh, x, y_top, w, h):
    """`fig.add_axes` in millimetres measured from the TOP-LEFT of the canvas.

    Every band here is height-critical and the bands are described top-down; reasoning
    about three stacked bands in bottom-origin figure fractions is how panels end up
    overlapping each other, which is the one failure mode this figure cannot have.
    """
    return fig.add_axes([x / fw, 1.0 - (y_top + h) / fh, w / fw, h / fh])


def divider(fig, fh, y_top, x0=0.02, x1=0.98):
    """The thin rule that separates two bands."""
    fig.add_artist(Line2D([x0, x1], [1.0 - y_top / fh] * 2, transform=fig.transFigure,
                          color="#cccccc", lw=0.5, zorder=0))


# ---------------------------------------------------------------------------
# Band A -- the two loops, schematically
# ---------------------------------------------------------------------------

# Colours are the shared palette, and the roles line up with band C on purpose: BLUE is
# the detector-sized job, YELLOW the matmul block that the IME can take, GREEN the small
# well-behaved controller, PURPLE the fusion net.
_A_BLUE, _A_YELL, _A_GREEN, _A_PURP = (figstyle.BLUE, figstyle.YELLOW,
                                       figstyle.GREEN, figstyle.PURPLE)

#: (row, start, width, colour, shard?, ime?) on a 0..100 schematic time axis, 3 rows.
#: These are ILLUSTRATIONS. No number here is measured, which is why the band carries no
#: axis at all -- a tick would imply a unit and there is none.
DEADLINE = 74.0
MINI = {
    # inner loop: modelled costs. One long unsharded matmul block overruns.
    "modelled": [(0, 2, 40, _A_BLUE, 0, 0), (0, 44, 48, _A_YELL, 0, 0),
                 (1, 2, 34, _A_GREEN, 0, 0)],
    # inner loop after the AOT levers: the detector is sharded across two harts, the
    # matmul block is on the IME and shorter, and the fusion net fits beside it.
    "optimised": [(0, 2, 30, _A_BLUE, 1, 0), (1, 2, 30, _A_BLUE, 1, 0),
                  (0, 34, 26, _A_YELL, 0, 1), (1, 34, 24, _A_PURP, 0, 0),
                  (2, 2, 22, _A_GREEN, 0, 0)],
    # outer loop before: the SAME assignment, measured on the K1. Everything stretched;
    # the IME block now runs past the window the offline solve promised.
    "measured": [(0, 2, 38, _A_BLUE, 1, 0), (1, 2, 38, _A_BLUE, 1, 0),
                 (0, 42, 44, _A_YELL, 0, 1), (1, 42, 30, _A_PURP, 0, 0),
                 (2, 2, 28, _A_GREEN, 0, 0)],
    # outer loop after: re-solved on those measured costs -- the matmul block is split
    # off the critical hart and the fusion net moves up.
    "re-solved": [(0, 2, 36, _A_BLUE, 1, 0), (1, 2, 36, _A_BLUE, 1, 0),
                  (0, 40, 30, _A_YELL, 0, 1), (2, 2, 30, _A_PURP, 0, 0),
                  (1, 40, 26, _A_GREEN, 0, 0)],
}
#: which mini-Gantts carry a miss, and on which row the offending bar sits
MISSES = {"modelled": 0, "measured": 0}


def mini_gantt(ax, key):
    """One schematic Gantt: a handful of bars, one dashed deadline, nothing else."""
    nrow = 3
    for row, s, w, col, shard, ime in MINI[key]:
        ax.barh(row, w, left=s, height=0.68, color=col, edgecolor="white",
                linewidth=0.3, zorder=3)
        if shard:
            ax.barh(row, w, left=s, height=0.68, facecolor="none", edgecolor=SHARD_EC,
                    linewidth=0.35, hatch=SHARD_HATCH, zorder=3.4)
        if ime:
            ax.barh(row, w, left=s, height=0.68, facecolor="none", edgecolor=IME_EC,
                    linewidth=0.4, hatch=IME_HATCH, zorder=3.4)
    ax.axvline(DEADLINE, color=figstyle.C_DEADLINE, ls=(0, (2.2, 1.6)), lw=0.9,
               zorder=8)
    if key in MISSES:
        # the overrun: the slice of the offending bar that lives past the deadline,
        # outlined in the deadline colour so the eye lands on it first
        row = MISSES[key]
        end = max(s + w for r, s, w, *_ in MINI[key] if r == row)
        ax.barh(row, end - DEADLINE, left=DEADLINE, height=0.68, facecolor="none",
                edgecolor=figstyle.C_DEADLINE, hatch=IME_HATCH, linewidth=0.7, zorder=10)
        ax.scatter([end], [row], marker="X", s=6.0, color=figstyle.C_DEADLINE,
                   edgecolors="white", linewidths=0.35, zorder=11, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(nrow - 0.45, -0.55)                 # row 0 on top
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # a hairline baseline under the cores, so the bars read as a schedule and not as a
    # floating bar chart
    ax.plot([0, 100], [nrow - 0.42] * 2, color="#b0b0b0", lw=0.5, clip_on=False)


def band_a(fig, fw, fh, y_top, gantt_h, centres=None):
    """Two before -> after pairs, side by side, labelled 'inner loop' / 'outer loop'.

    `centres` optionally pins each group's centre in mm. In the two-band layout the four
    stage panels below carry the meaning by ALIGNMENT -- stages 1-2 under `inner loop`,
    3-4 under `outer loop` -- so the schematics have to sit over their own halves rather
    than be spaced for their own convenience.
    """
    pair_w, arrow_w, sub_h = 30.0, 11.0, 3.2
    group_w = pair_w * 2 + arrow_w
    margin = (fw - 2 * group_w) / 3.0               # equal air left, middle, right
    for gi, (title, keys, lever) in enumerate(
            [("inner loop", ("modelled", "optimised"), "shard + IME"),
             ("outer loop", ("measured", "re-solved"), "re-solve")]):
        gx = (centres[gi] - group_w / 2.0 if centres
              else margin + gi * (group_w + margin))
        fig.text((gx + group_w / 2) / fw, 1.0 - (y_top - 0.3) / fh, title,
                 ha="center", va="bottom", fontsize=6.0, fontweight="bold",
                 color=figstyle.BLACK)
        for ki, key in enumerate(keys):
            x = gx + ki * (pair_w + arrow_w)
            ax = mm_axes(fig, fw, fh, x, y_top + 0.7, pair_w, gantt_h)
            mini_gantt(ax, key)
            fig.text((x + pair_w / 2) / fw, 1.0 - (y_top + 0.7 + gantt_h + 0.5) / fh,
                     key, ha="center", va="top", fontsize=4.4, color="#666666")
        # the transition, with the lever that causes it
        ay = 1.0 - (y_top + 0.7 + gantt_h * 0.52) / fh
        ax0 = (gx + pair_w + 1.6) / fw
        ax1 = (gx + pair_w + arrow_w - 1.6) / fw
        fig.add_artist(Line2D([ax0, ax1], [ay, ay], transform=fig.transFigure,
                              color="#8a94a6", lw=0.9))
        fig.add_artist(Line2D([ax1 - 0.006, ax1], [ay + 0.012, ay],
                              transform=fig.transFigure, color="#8a94a6", lw=0.9))
        fig.add_artist(Line2D([ax1 - 0.006, ax1], [ay - 0.012, ay],
                              transform=fig.transFigure, color="#8a94a6", lw=0.9))
        fig.text((ax0 + ax1) / 2, ay + 0.028, lever, transform=fig.transFigure,
                 ha="center", va="bottom", fontsize=4.3, color="#3d6b3d",
                 fontweight="bold")


# ---------------------------------------------------------------------------
# Band B -- the ablation cells
# ---------------------------------------------------------------------------

CELL_ORDER = ("A", "B", "C", "D")
CELL_LABEL = {"A": "neither", "B": "inner (AOT)",
              "C": "outer (HIL)", "D": "both"}
CELL_COLOR = {"A": figstyle.C_MUTED, "B": figstyle.BLUE,
              "C": figstyle.ORANGE, "D": figstyle.GREEN}
SOLVER_TITLE = {"cpsat": "CP-SAT (exact)", "greedy": "greedy (list)"}
#: IME is an accelerator name, not a word. Same spelling as gen_schedule_evolution.py.
_NICE_LEVER = {"ime": "IME"}


def load_ablation(paths):
    """`{rung: {"n_nets": int, solver: {"cells": {...}, "levers": [...]}}}`.

    Lifted from `plot_ladder_scaling.py` so the two figures cannot come to disagree
    about what an ablation summary says. A path that does not exist, does not parse, or
    carries no "runs" list is skipped silently -- the ladder runs in legs, and a leg
    still being written must not take the whole figure down.
    """
    out = {}
    for p in paths:
        try:
            with open(p) as f:
                doc = json.load(f)
            runs = doc["runs"]
            assert isinstance(runs, list)
        except Exception:                            # noqa: BLE001 - see docstring
            continue
        for r in runs:
            rung = os.path.basename(r["workload"]).split("_")[0]
            per = out.setdefault(rung, {"n_nets": len(r.get("feasibility") or {})})
            for sv, s in (r.get("solvers") or {}).items():
                per[sv] = {
                    "cells": {c: (s["cells"][c] or {}).get("instance_misses")
                              for c in CELL_ORDER if c in (s.get("cells") or {})},
                    "levers": s.get("levers") or [],
                }
    return out


def lever_text(levers):
    return "+".join(_NICE_LEVER.get(l, l) for l in levers) if levers else "none"


def cells_panel(ax, data, rungs, solver, ylim):
    """Four bars per rung. Read DOWN a rung for which loop worked, ACROSS for scaling."""
    w = 0.2
    for j, c in enumerate(CELL_ORDER):
        xs = [i + (j - 1.5) * w for i in range(len(rungs))]
        ys = [((data[r].get(solver) or {}).get("cells") or {}).get(c) or 0
              for r in rungs]
        ax.bar(xs, ys, width=w, color=CELL_COLOR[c], edgecolor="white",
               linewidth=0.25, zorder=3)
        # A cleared cell draws nothing, which reads as missing data rather than as the
        # best result on the figure. Mark the zeros.
        for x, y in zip(xs, ys):
            if y == 0:
                ax.plot([x], [0], marker="v", ms=1.5, color=CELL_COLOR[c],
                        clip_on=False, zorder=5)
    ax.set_xticks(range(len(rungs)))
    # The lever the inner search ACCEPTED rides on the tick with the rung name: it is
    # what separates the rungs that move from the rungs that do not.
    ax.set_xticklabels(
        [f"{r} · {data[r]['n_nets']} nets\n{lever_text((data[r].get(solver) or {}).get('levers'))}"
         for r in rungs], fontsize=4.0, linespacing=1.0)
    ax.set_ylim(0, ylim)
    ax.set_yticks([0, round(ylim / 2), int(ylim) - int(ylim) % 2])
    ax.tick_params(axis="y", labelsize=4.2, pad=1)
    ax.tick_params(axis="x", pad=1.0, length=1.5)
    ax.set_title(SOLVER_TITLE.get(solver, solver), fontsize=5.0, pad=1.5)
    ax.grid(axis="y", lw=0.25, color="#dddddd", zorder=0)
    figstyle.despine(ax)


def benefit_panel(ax, data, rungs, solvers):
    """A-B and A-C: misses each loop actually removed. Zero, for the outer, IS the point."""
    style = {"cpsat": dict(marker="o", ls="-"), "greedy": dict(marker="s", ls="--")}
    for sv in solvers:
        for key, cell, col in (("inner", "B", figstyle.BLUE),
                               ("outer", "C", figstyle.ORANGE)):
            ys = []
            for r in rungs:
                cs = (data[r].get(sv) or {}).get("cells") or {}
                ys.append(None if cs.get("A") is None or cs.get(cell) is None
                          else cs["A"] - cs[cell])
            ax.plot(range(len(rungs)), ys, color=col, ms=1.8, lw=0.7,
                    label=f"{key}·{sv}", **style.get(sv, {}))
    ax.axhline(0, color=figstyle.BLACK, lw=0.5, zorder=2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.5, hi + 7.0)                 # headroom for the legend
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(rungs, fontsize=4.2)
    ax.tick_params(axis="y", labelsize=4.2, pad=1)
    ax.tick_params(axis="x", pad=1.0, length=1.5)
    ax.set_title("misses removed by each loop", fontsize=5.0, pad=1.5)
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=3.9,
              handlelength=1.4, columnspacing=0.6, handletextpad=0.28,
              borderpad=0.0, labelspacing=0.12)
    ax.grid(axis="y", lw=0.25, color="#dddddd", zorder=0)
    figstyle.despine(ax)


# ---------------------------------------------------------------------------
# Band C -- the four real schedule stages, in one row
# ---------------------------------------------------------------------------

def _stage2_title(path):
    """Name the AOT stage after the levers the schedule ACTUALLY carries.

    The title was hardcoded "AOT optimise · IME", which was right for the sensor arc and
    wrong the moment the figure was pointed at a sharded one: s5 is cleared by
    shard:yolo + shard:ffn_block and no IME at all. A panel that names the wrong
    transformation is worse than one that names none, so it is read off the dispatches.
    """
    import collections as _c
    try:
        d = json.load(open(path))
    except Exception:
        return "AOT optimise"
    widths, impls = _c.Counter(), set()
    for e in (d.get("dispatches") or {}).values():
        ht = str(e.get("hardware_target") or "")
        widths[len(ht.split("+")) if ht else 0] += 1
        impls.add(str(e.get("implementation") or e.get("impl") or "rvv"))
    marks = []
    if any(k > 1 for k in widths):
        marks.append("shard")
    if "ime" in impls:
        marks.append("IME")
    return "AOT optimise" + (" · " + "+".join(marks) if marks else "")


STAGES = [("1", "Baseline · RVV", "a1_baseline"),
          ("2", "AOT optimise · IME", "a2_aot"),
          ("3", "K1-calibrated replay", "a3_board_recost"),
          ("4", "K1-calibrated re-solve", "a4_board_resolve")]

#: `ffn_block` and `attn_block` are not in figstyle.MODEL_COLOR (they are graph blocks,
#: not shipped networks), so model_color would hand BOTH of them the same muted grey --
#: and they are the two busiest series in this Gantt. Give each an Okabe-Ito slot, and
#: keep the hues the schedule-evolution figure already used so the two agree on sight.
_BLOCK_FALLBACK = {"ffn_block": figstyle.YELLOW, "attn_block": figstyle.SKY}


def stage_axes(ax, sched_json, dl, nets, remap, xmax, breaks, feedback):
    """One stage panel. Delegates every mark to compose_schedule_evolution.draw()."""
    rows = evo.load(sched_json)
    misses, missed = evo.draw(ax, rows, dl, nets, "none", remap, xmax, breaks,
                              feedback, compact=True)
    # draw() sizes its miss "X" for a 2x2 page figure; at a quarter page that marker is
    # wider than the dispatch it flags. The X scatters are the only collections here.
    for coll in ax.collections:
        coll.set_sizes([12.0])
        coll.set_linewidths(0.55)
    # Eight per-core labels cannot fit a 9 mm band without touching. The E/P cluster
    # split is the only part of that axis a reader uses at this size.
    ax.set_yticks([1.5, 5.5])
    ax.set_yticklabels(["E", "P"], fontsize=4.4)
    ax.tick_params(axis="y", pad=1.0, length=0)
    return misses, missed


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="composite co-design loop overview figure")
    ap.add_argument("--summary", action="append", default=None,
                    help="ablation_summary.json (repeatable; the ladder ran in legs). "
                         "Unreadable or unfinished legs are skipped silently.")
    ap.add_argument("--evo-dir",
                    default=os.path.join(_REPO,
                                         "results/codesign_feedback/sensor_evo_ime"),
                    help="dir holding a1_baseline.json .. a4_board_resolve.json")
    ap.add_argument("--spec",
                    default=os.path.join(
                        _REPO,
                        "data/toplevel/_4w_networks_k1_sensor_tight_vision_head.json"),
                    help="the workload spec band C's deadlines come from")
    ap.add_argument("--height-mm", type=float, default=58.5,
                    help="canvas height; the brief caps the figure at a quarter page")
    ap.add_argument("--no-ablation", action="store_true",
                    help="drop band B and align the four stage panels under the two "
                         "schematics: stages 1-2 (predicted costs) under `inner loop`, "
                         "stages 3-4 (measured board costs) under `outer loop`. The "
                         "alignment is the argument -- it says which loop owns which "
                         "beat without a caption having to.")
    ap.add_argument("--stem", default="loop_overview")
    ap.add_argument("--out-dir",
                    default=os.path.join(_REPO, "results/codesign_feedback"))
    a = ap.parse_args()

    summaries = a.summary or [
        os.path.join(_REPO, "results/loop_ablation_v4_small/ablation_summary.json"),
        os.path.join(_REPO, "results/loop_ablation_v4_big/ablation_summary.json"),
        os.path.join(_REPO, "results/loop_ablation_stable/ablation_summary.json"),
    ]

    figstyle.use()
    FW, FH = 183.0, a.height_mm
    fig = plt.figure(figsize=(FW * figstyle.MM, FH * figstyle.MM))
    # figstyle.save() crops with bbox_inches="tight", which would shrink the canvas to
    # its ink and silently break the DOUBLE_COL width the paper's column expects. An
    # invisible full-canvas rectangle pins the crop to the size we asked for.
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor="none", edgecolor="none", zorder=-10))
    notes = {}

    # ---- Band A ---------------------------------------------------------------------
    A_TOP, A_GANTT_H = 3.2, 8.2
    # STAGE-ROW GEOMETRY FIRST, because in the two-band layout band A is centred on the
    # halves the stage panels define. Four panels, a wider gap in the middle to make the
    # inner/outer split visible without a rule through it.
    C_PW, C_GAP, C_MIDGAP, C_X0 = 37.0, 8.0, 13.0, 8.0
    _xs = [C_X0, C_X0 + C_PW + C_GAP,
           C_X0 + 2 * C_PW + C_GAP + C_MIDGAP,
           C_X0 + 3 * C_PW + 2 * C_GAP + C_MIDGAP]
    _centres = [(_xs[0] + _xs[1] + C_PW) / 2.0, (_xs[2] + _xs[3] + C_PW) / 2.0]

    band_a(fig, FW, FH, A_TOP, A_GANTT_H, centres=(_centres if a.no_ablation else None))
    a_bot = A_TOP + 0.7 + A_GANTT_H + 3.0
    divider(fig, FH, a_bot + 0.3)

    # ---- Band B ---------------------------------------------------------------------
    B_TOP = a_bot + 5.9
    data = None if a.no_ablation else load_ablation(summaries)
    notes["summaries_used"] = [p for p in summaries if os.path.exists(p)]
    notes["summaries_missing"] = [p for p in summaries if not os.path.exists(p)]
    B_PLOT_H = 8.6
    if data:
        rungs = sorted(data, key=lambda r: data[r]["n_nets"])
        solvers = [s for s in ("cpsat", "greedy") if any(s in data[r] for r in rungs)]
        ylim = int(max(v for r in rungs for sv in solvers
                       for v in ((data[r].get(sv) or {}).get("cells") or {}).values()
                       if v is not None) * 1.18) + 1
        pw, pgap, px = 45.0, 14.0, 11.0
        b_axes = []
        for sv in solvers[:2]:
            ax = mm_axes(fig, FW, FH, px, B_TOP, pw, B_PLOT_H)
            cells_panel(ax, data, rungs, sv, ylim)
            b_axes.append(ax)
            px += pw + pgap
        ax = mm_axes(fig, FW, FH, px, B_TOP, pw, B_PLOT_H)
        benefit_panel(ax, data, rungs, solvers)
        b_axes.append(ax)
        b_axes[0].set_ylabel("deadline misses", fontsize=4.4, labelpad=1.0)
        b_axes[-1].set_ylabel("misses removed", fontsize=4.4, labelpad=1.0)
        # One legend for the four cells, above the first panel, in the strip the band's
        # divider already reserves. Four entries in one row fit; two rows would not.
        b_axes[0].legend(handles=[Patch(facecolor=CELL_COLOR[c], label=CELL_LABEL[c])
                                  for c in CELL_ORDER],
                         frameon=False, ncol=4, loc="lower left",
                         bbox_to_anchor=(-0.03, 1.34), fontsize=4.1,
                         handlelength=1.0, handleheight=0.8, columnspacing=0.9,
                         handletextpad=0.28, borderpad=0.0)
        notes["ablation"] = {r: {"n_nets": data[r]["n_nets"],
                                 **{sv: data[r][sv] for sv in solvers if sv in data[r]}}
                             for r in rungs}
    elif a.no_ablation:
        notes["ablation"] = "dropped by --no-ablation"
    else:
        print("WARNING: band B dropped - no ablation summary parsed", file=sys.stderr)
        notes["ablation"] = None

    if a.no_ablation:
        b_bot = a_bot                               # no band B, no second divider
    else:
        b_bot = B_TOP + B_PLOT_H + 4.0              # plot + two-line x tick labels
        divider(fig, FH, b_bot + 0.5)

    # ---- Band C ---------------------------------------------------------------------
    C_TOP = b_bot + 5.0
    C_PLOT_H = 10.0
    paths = [os.path.join(a.evo_dir, f"{s}.json") for _, _, s in STAGES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing or not os.path.exists(a.spec):
        print(f"WARNING: band C dropped - missing {missing or a.spec}", file=sys.stderr)
        notes["evolution"] = None
    else:
        # Per-network colour from the shared palette, so a network is the same colour
        # here as in every other figure in the paper (and as in band A above).
        dl = evo.deadlines(a.spec)
        nets = list(dl.keys())
        evo.NETCOLOR = {n: figstyle.model_color(
            n, fallback=_BLOCK_FALLBACK.get(n, figstyle.C_MUTED)) for n in nets}

        loaded = [evo.load(p) for p in paths]
        # The baseline's tail is long; letting it set the scale for all four squashes
        # stages 2-4. Panel 1 keeps its own remap and 2-4 share one, exactly as the 2x2
        # original does -- and panel 1's axis is flagged so the difference is visible.
        first = evo.build_remap([(None, None, loaded[0])])
        rest = evo.build_remap([(None, None, r) for r in loaded[1:]])

        # In the two-band layout the panels sit on the SAME x positions band A was
        # centred on, with a wider gap between stage 2 and 3. That gap is the inner/outer
        # boundary and the alignment is what tells the reader which loop owns which beat.
        cw, cgap, cx = (C_PW, C_GAP, C_X0) if a.no_ablation else (37.0, 8.0, 9.0)
        stage_rows = []
        for i, (num, title, _stem) in enumerate(STAGES):
            if num == "2":
                title = _stage2_title(paths[i])
            remap, xmax, breaks, merged = first if i == 0 else rest
            if a.no_ablation:
                cx = _xs[i]
            ax = mm_axes(fig, FW, FH, cx, C_TOP, cw, C_PLOT_H)
            miss, missed = stage_axes(ax, paths[i], dl, nets, remap, xmax, breaks,
                                      feedback=(i >= 2))
            tks, tlbls = evo.gen_ticks(merged, remap)
            ax.set_xticks(tks)
            ax.set_xticklabels(tlbls, fontsize=4.0)
            ax.tick_params(axis="x", pad=1.0, length=1.5)
            if i == 0:                               # flag the wider time scale
                ax.spines["bottom"].set_color("#d1720b")
                ax.spines["bottom"].set_linewidth(1.2)
                ax.tick_params(axis="x", colors="#d1720b")
            col = "#2f7d4f" if miss == 0 else "#c0392b"
            ax.text(-0.005, 1.30, num, transform=ax.transAxes, fontsize=4.6,
                    weight="bold", color="white", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.24", fc=col, ec="none"))
            # The badge is right-anchored in the same strip, so a long title collides
            # with it -- "AOT optimise · shard + IME" plus "✓ ALL MET" does not fit a
            # 37 mm panel at 4.7 pt. Shrink the title rather than truncate it: the lever
            # names are the informative half and must survive.
            _tfs = 4.7 if len(title) <= 22 else (4.2 if len(title) <= 28 else 3.8)
            ax.text(0.055, 1.30, title, transform=ax.transAxes, fontsize=_tfs,
                    weight="bold", color="#111", ha="left", va="center")
            ax.text(1.0, 1.30, ("✓ ALL MET" if miss == 0 else f"✗ {miss} MISSED"),
                    transform=ax.transAxes, ha="right", va="center", fontsize=4.2,
                    weight="bold", color="white", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))
            _w = set(); _im = set()
            for _e in (json.load(open(paths[i])).get("dispatches") or {}).values():
                _ht = str(_e.get("hardware_target") or "")
                _w.add(len(_ht.split("+")) if _ht else 0)
                _im.add(str(_e.get("implementation") or _e.get("impl") or "rvv"))
            stage_rows.append({"stage": int(num), "title": title,
                               "has_shard": any(x > 1 for x in _w),
                               "has_ime": "ime" in _im,
                               "dispatch_widths": sorted(_w),
                               "implementations": sorted(_im),
                               "instance_misses": miss,
                               "missed_by_network": missed})
            cx += cw + cgap
        fig.text(9.0 / FW, 1.0 - (C_TOP + C_PLOT_H + 3.3) / FH,
                 "onboard time (ms) · K1 cores E0-3 / P0-3 · "
                 "stage 1 is drawn on a wider time scale (orange axis)",
                 fontsize=4.3, color="#555555", ha="left", va="top")
        notes["evolution"] = stage_rows

        # ---- one shared legend, one row, at the very bottom -------------------------
        _sl = {"yolov8_nano_64x96": "yolov8n", "fused_full": "nav",
               "mlp_control": "ctrl", "ffn_block": "ffn", "attn_block": "attn"}
        handles = [Patch(fc=evo.NETCOLOR[n], label=_sl.get(n, n)) for n in nets]
        # ADVERTISE ONLY WHAT IS ACTUALLY DRAWN. A legend listing `shard` and `IME`
        # unconditionally makes a figure promise something its panels may not contain:
        # the sensor arc in band C is IME-ONLY -- every one of its 217 dispatches is
        # width 1 in all four stages, and the whole arc turns on four dispatches moving
        # to the IME engine (six after the board re-solve) -- so "shards the wide
        # dispatches" is not a description of it. The entries are derived from the
        # stage schedules so the claim cannot drift from the data.
        used_shard = any(r.get("has_shard") for r in stage_rows) or True  # band A shards
        used_ime = any(r.get("has_ime") for r in stage_rows) or True
        if used_shard:
            handles.append(Patch(fc="0.85", hatch=SHARD_HATCH, ec=SHARD_EC,
                                 label="shard"))
        if used_ime:
            handles.append(Patch(fc="0.85", hatch=IME_HATCH, ec=IME_EC, label="IME"))
        handles += [
            Line2D([0], [0], color="#e60000", ls=(0, (3, 2)), lw=0.9, label="deadline"),
            Patch(fc="none", ec="#e60000", hatch=IME_HATCH, label="overrun"),
            Patch(fc="#fbf3ec", ec="0.7", label="board-cost stage"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=4.2, frameon=False,
                   bbox_to_anchor=(0.5, 0.3 / FH), columnspacing=0.9,
                   handletextpad=0.28, handlelength=1.1, handleheight=0.8,
                   borderpad=0.0)

    png = figstyle.save(fig, a.stem, a.out_dir)
    print("wrote", png)

    side = os.path.splitext(png)[0] + "_metrics.json"
    notes["canvas_mm"] = [round(v / figstyle.MM, 2) for v in fig.get_size_inches()]
    # What actually landed on disk. figstyle.save() crops with bbox_inches="tight" and
    # a 0.03 in pad, so the file is ~1.5 mm wider and taller than the canvas -- and a
    # figure that reports a size it does not have is the same class of drift as a
    # hand-typed caption.
    try:
        from PIL import Image
        with Image.open(png) as im:
            notes["saved_mm"] = [round(d / 300.0 / figstyle.MM, 2) for d in im.size]
    except Exception:                                # noqa: BLE001 - cosmetic only
        notes["saved_mm"] = None
    notes["cell_meaning"] = CELL_LABEL
    notes["band_a"] = ("schematic only - no measured number appears in band A; the "
                       "mini-Gantts are illustrations of the two loops")
    notes["caveat_cell_C"] = (
        "Cell C re-solves against measured board costs and is also SCORED on them, so "
        "the discrepancy the board step exposes in band C is designed out of it. Its "
        "zero benefit is a statement about the ablation's outer-loop measurement, not "
        "about HIL feedback.")
    with open(side, "w") as f:
        json.dump(notes, f, indent=1)
    print("wrote", side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
