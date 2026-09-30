#!/usr/bin/env python3
"""Where the co-design loop closes, where it stops, and what predicts which.

WHAT THIS SHOWS, and why it is not the figure we wanted. `schedule_evolution_*.png`
tells the loop's story on ONE workload: baseline misses deadlines, the AOT search
clears them, the board reveals misses the AOT costs could not see, the calibrated
re-solve clears them again. That arc is real -- and it closes at TWO networks. This
figure is the same experiment run up a ladder of 2, 3, 4 and 5 concurrent networks,
and the arc stops closing above three.

The panels are ordered as an argument:
  (a),(b) the four ablation cells per rung, one panel per solver. Read DOWN a rung to
          see which loop did the work; read ACROSS rungs to see it stop.
  (c)     the two loop benefits, A-B for the inner loop and A-C for the outer, so the
          collapse and the null result are one glance each instead of subtraction.

THE HONEST READINGS, all four visible in the bars:
  * The INNER loop (AOT + ModelBlaster) does all the work that gets done, and only at
    2-3 networks: it clears w2 outright and halves w3.
  * The OUTER loop contributes NOTHING. Cell C sits on cell A on every rung under both
    solvers, and D sits on B. As measured, its benefit is exactly zero -- see the
    caveat below, because this is a statement about our measurement, not about HIL.
  * CP-SAT never beats greedy. It ties on w2/w3 and LOSES on w4/w5. These counts are
    taken with the deadline budget the scheduler uses (the budget alone accounts for
    17->10 and 30->13 misses), so the loss is not a mis-run.
  * The lever the inner search FINDS predicts everything. `shard` -> the loop works;
    the `ime` fallback -> no misses cleared, and on w5 four MORE. The lever label under
    each rung is therefore the explanatory variable, not decoration.

WHY CELL C IS ZERO, stated on the figure so nobody has to rediscover it. Cell C
re-solves against measured board multipliers but is also SCORED on them, so the
discrepancy that makes the board step valuable in `schedule_evolution` -- costs the
AOT solve could not see -- has been designed out of it. C changes the cost model, not
the code, and where misses come from per-op inflation only different code helps. This
figure is evidence that our outer-loop MEASUREMENT is weak, not that HIL feedback is.

NO INFEASIBILITY ALIBI. Every cell of every rung reports
`total_infeasible_misses: 0` -- the fastest measured implementation of each network
fits its window -- so nothing here is a miss no schedule could have met. The flat
rungs are honest failures.

Input is `ablation_summary.json` (one or more; the ladder was run in two legs at
different budgets). Reads only, never re-solves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

CELL_ORDER = ("A", "B", "C", "D")
CELL_LABEL = {"A": "neither", "B": "inner only (AOT)",
              "C": "outer only (HIL)", "D": "both"}
CELL_COLOR = {"A": figstyle.C_MUTED, "B": figstyle.BLUE,
              "C": figstyle.ORANGE, "D": figstyle.GREEN}
SOLVER_TITLE = {"cpsat": "CP-SAT (exact)", "greedy": "greedy (list)"}


def load(paths):
    """`{rung: {solver: {"cells": {...}, "levers": [...]}}}` keyed w2..w5."""
    out = {}
    for p in paths:
        for r in json.load(open(p))["runs"]:
            rung = os.path.basename(r["workload"]).split("_")[0]
            n_nets = len(r.get("feasibility") or {})
            per = out.setdefault(rung, {"n_nets": n_nets})
            for sv, s in (r.get("solvers") or {}).items():
                cells = {c: (s["cells"][c] or {}).get("instance_misses")
                         for c in CELL_ORDER if c in (s.get("cells") or {})}
                per[sv] = {"cells": cells, "levers": s.get("levers") or []}
    return out


#: IME is an accelerator name, not a word. `gen_schedule_evolution.py` carries the
#: same map; keep the two spellings identical across figures.
_NICE = {"ime": "IME"}


def lever_text(levers):
    """The lever label that goes under a rung -- the explanatory variable."""
    return "+".join(_NICE.get(l, l) for l in levers) if levers else "no lever found"


def cells_panel(ax, data, rungs, solver):
    w = 0.2
    for j, c in enumerate(CELL_ORDER):
        xs, ys = [], []
        for i, rung in enumerate(rungs):
            v = ((data[rung].get(solver) or {}).get("cells") or {}).get(c)
            if v is None:
                continue
            xs.append(i + (j - 1.5) * w)
            ys.append(v)
        ax.bar(xs, ys, width=w, color=CELL_COLOR[c], edgecolor="white",
               linewidth=0.3, zorder=3)
        # A cleared cell has nothing to draw, which reads as missing data rather than
        # as the best result on the figure. Mark the zeros explicitly.
        for x, y in zip(xs, ys):
            if y == 0:
                ax.plot([x], [0], marker="v", ms=2.2, color=CELL_COLOR[c],
                        clip_on=False, zorder=5)
        for x, y in zip(xs, ys):
            # ROTATED, because four bars of equal height print their labels as one
            # illegible run ("10101010"). Vertical labels stay attached to their bar.
            ax.text(x, y + 0.25, str(y), ha="center", va="bottom", fontsize=4.4,
                    color="#333333", rotation=90, zorder=4)
    ax.set_xticks(range(len(rungs)))
    # The lever goes ON the axis with the rung. It is what separates the rungs that
    # work from the rungs that do not, so it belongs where the eye already is.
    ax.set_xticklabels(
        [f"{r}\n{data[r]['n_nets']} nets\n{lever_text((data[r].get(solver) or {}).get('levers'))}"
         for r in rungs], fontsize=4.6)
    ax.set_ylabel("deadline misses (instances)")
    # Headroom for the legend, which otherwise lands on the tallest rung's labels.
    top = max([v for r in rungs
               for v in (((data[r].get(solver) or {}).get("cells") or {}).values())
               if v is not None] or [1])
    ax.set_ylim(0, top * 1.38)
    ax.set_title(SOLVER_TITLE.get(solver, solver))
    ax.grid(axis="y", lw=0.3, color="#dddddd", zorder=0)
    figstyle.despine(ax)


def benefit_panel(ax, data, rungs, solvers):
    """A-B and A-C: how many misses each loop actually removed. Zero is the message."""
    style = {"cpsat": dict(marker="o", ls="-"), "greedy": dict(marker="s", ls="--")}
    for sv in solvers:
        for key, cell, col in (("inner", "B", figstyle.BLUE),
                              ("outer", "C", figstyle.ORANGE)):
            ys = []
            for r in rungs:
                cs = ((data[r].get(sv) or {}).get("cells") or {})
                ys.append(None if cs.get("A") is None or cs.get(cell) is None
                          else cs["A"] - cs[cell])
            ax.plot(range(len(rungs)), ys, color=col, ms=3,
                    label=f"{key} ({sv})", **style.get(sv, {}))
    ax.axhline(0, color=figstyle.BLACK, lw=0.6, zorder=2)
    # The traces sweep from top-left to bottom-right and there is no empty region
    # inside the data's own bounds, so make one above it.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.5, hi + 4.0)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(rungs)
    ax.set_ylabel("misses removed by the loop")
    ax.set_title("loop benefit (positive = fewer misses)")
    # The lines run from top-left to bottom-right, so the only reliably empty region
    # is centre-right. One column, because two spanned the whole panel.
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=4.4)
    ax.grid(axis="y", lw=0.3, color="#dddddd", zorder=0)
    figstyle.despine(ax)
    ax.text(0.02, 0.05, "below zero = the loop made it worse",
            transform=ax.transAxes, fontsize=4.4, color="#777777")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="append", required=True,
                    help="ablation_summary.json (repeatable; the ladder ran in legs)")
    ap.add_argument("--stem", default="ladder_scaling")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    figstyle.use()
    data = load(a.summary)
    rungs = sorted(data, key=lambda r: data[r]["n_nets"])
    solvers = [s for s in ("cpsat", "greedy")
               if any(s in data[r] for r in rungs)]

    fig = plt.figure(figsize=(figstyle.DOUBLE_COL, 62 * figstyle.MM))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for i, sv in enumerate(solvers[:2]):
        cells_panel(axes[i], data, rungs, sv)
        figstyle.panel_label(axes[i], "abc"[i], x=-0.16, y=1.10)
    benefit_panel(axes[2], data, rungs, solvers)
    figstyle.panel_label(axes[2], "c", x=-0.16, y=1.10)

    axes[0].legend(handles=[Patch(facecolor=CELL_COLOR[c], label=CELL_LABEL[c])
                            for c in CELL_ORDER],
                   frameon=False, ncol=2, loc="upper left", fontsize=4.4)

    fig.text(0.5, -0.10,
             "Cell C re-solves on measured board costs and is also SCORED on them, so the "
             "discrepancy the board step exposes in schedule_evolution is designed out of it: "
             "C changes the cost model, not the code.  Every cell reports 0 infeasible "
             "misses, so no bar is a deadline no schedule could meet.",
             ha="center", va="top", fontsize=4.2, color="#666666", wrap=True)

    png = figstyle.save(fig, a.stem, a.out_dir)
    print("wrote", png)

    side = os.path.splitext(png)[0] + "_metrics.json"
    json.dump({"rungs": {r: {"n_nets": data[r]["n_nets"],
                             **{sv: data[r][sv] for sv in solvers if sv in data[r]}}
                         for r in rungs},
               "reads": a.summary}, open(side, "w"), indent=1)
    print("wrote", side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
