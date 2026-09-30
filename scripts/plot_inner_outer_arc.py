#!/usr/bin/env python3
"""The four-beat arc on the top rungs, split by which loop owns which beat.

WHAT IT ANSWERS. The ladder figure shows w4 and w5 flat under every cell without saying
why. Two conditions, neither of them scheduling, hold those rungs flat: a list scheduler
not held to the codegen contract, so the one lever that clears deadlines is discarded
post-hoc; and a global shard switch, so widening `ffn_block` also widens yolo and the
whole lever is rejected. With the contract-bound scheduler and a per-network shard switch
the rungs move, and this figure shows the two loops doing visibly different jobs:

  beat 1 -> 2   INNER (AOT + ModelBlaster): choose graph transformations, on predicted
                costs. w4 10 -> 5 misses, w5 11 -> 7.
  beat 2 -> 3   OUTER, first half (HIL): execute that schedule on the K1 and re-cost it
                on what the hardware actually did. This beat has no analogue in the
                ablation cells and is the one that earns the outer loop: on w5 it exposes
                NINE misses the AOT model did not predict, five of them in `fused_full`,
                a network the AOT solve believed was entirely fine.
  beat 3 -> 4   OUTER, second half: re-solve against the measured costs.

STACKED BY NETWORK ON PURPOSE. A miss total cannot show a reveal that moves misses from
one network to another, and that movement is the finding: w4's residual is entirely
`dronet`, and w5's board re-cost introduces a `fused_full` band that is absent from the
predicted bar beside it. The colours are the shared per-model palette, so a network is
the same colour here as in every other figure in the paper.

HONEST ASYMMETRY, LABELLED ON THE FIGURE. w4 has no reveal to fix -- the board is
slightly KINDER than predicted there (5 -> 4) -- so its beat 3 is not a failure of the
outer loop but a demonstration that the AOT model was already right for that workload.
Only w5 produces a discrepancy, and the greedy re-solve recovers 2 of the 9: a list
scheduler does not optimise deadline misses, so re-solving it against better costs moves
the schedule without aiming at the metric.

Reads the `board_feedback.stages` that `run_codesign_loop.py --board-calibration` writes.
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
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
import figstyle  # noqa: E402
from schedule_eval import summary  # noqa: E402

BEAT = [("baseline", "baseline", "AOT"),
        ("aot-optimized", "AOT\n(inner)", "AOT"),
        ("board-recost", "re-cost\n(outer:\nreveal)", "board"),
        ("board-resolve", "re-solve\n(outer:\nfix)", "board")]


def arc(report_path, spec_path):
    """`[(label, {network: misses}, total, cost_model)]` for one rung."""
    d = json.load(open(report_path))
    bf = d.get("board_feedback") or {}
    base = os.path.dirname(report_path)
    by_stage = {s.get("stage"): s for s in (bf.get("stages") or [])}
    out = []
    for key, label, cost in BEAT:
        st = by_stage.get(key)
        if not st:
            # A REFUSED RE-SOLVE IS NOT ZERO MISSES. When the board re-solve finds
            # nothing better the loop records no `board-resolve` stage at all, and
            # drawing that as an empty bar says the schedule met every deadline -- the
            # opposite of what happened. w4 is exactly this case: greedy's re-solve went
            # 4 -> 8 and was correctly refused, so the deployed schedule is still the
            # re-cost one. Carry the previous beat forward and mark it.
            if key == "board-resolve" and out:
                prev = out[-1]
                out.append((label + "\n[refused: nothing better]",
                            prev[1], prev[2], cost))
            else:
                out.append((label, {}, None, cost))
            continue
        p = st.get("sched") or ""
        p = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
        try:
            s = summary(p, spec_path)
            out.append((label, s.get("misses_by_network") or {},
                        s.get("instance_misses"), cost))
        except Exception:
            out.append((label, {}, st.get("instance_misses"), cost))
    return out, (bf.get("stages") or [{}])[1].get("levers") or []


def attribution(rows, nets, attrib_path=None):
    """`(execution_bound, queueing)` for the last beat, from a MEASURED attribution file.

    Residuals are attributed from the trace's own per-instance cycles
    (`scripts/attribute_board_misses.py`), not from a network's available core widths. On
    this workload `fused_full`'s misses are almost all COLD START and queueing: its warm
    execution is 4.47 ms inside a 5 ms window, and only its first instance, at 2.70x warm,
    is over. A single-width network is not therefore execution-bound.

    What is actually execution-bound is `ffn_block`: warm execution 10.34 ms against a
    10.0 ms window, at the width the loop chose and its fastest measured one. The ladder
    was sized from a profile reading 7.72 ms at 8 cores, so the board is 34% slower than
    the number that declared the rung feasible. Three of five instances are over window
    by execution alone, and no scheduler recovers them.

    The count is read from the attribution JSON rather than inferred from network names,
    so the figure cannot drift from the measurement again.
    """
    last = next((r for r in reversed(rows) if r[2] is not None), None)
    if not last or not attrib_path or not os.path.exists(attrib_path):
        return None
    d = json.load(open(attrib_path))
    ex = int(d.get("execution_bound_instances") or 0)
    return ex, max(0, (last[2] or 0) - ex)


def panel(ax, rows, levers, title, nets, attrib=None):
    xs = range(len(rows))
    bottoms = [0.0] * len(rows)
    for net in nets:
        vals = [float((r[1] or {}).get(net, 0)) for r in rows]
        if not any(vals):
            continue
        ax.bar(xs, vals, bottom=bottoms, width=0.62,
               color=figstyle.model_color(net), edgecolor="white", linewidth=0.4,
               zorder=3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, (label, by, tot, cost) in zip(xs, rows):
        if tot is not None:
            ax.text(x, bottoms[x] + 0.25, str(tot), ha="center", va="bottom",
                    fontsize=5.5, fontweight="bold", zorder=4)
    # Say which cost model each bar is scored on: two of them are predicted and two are
    # measured, and comparing across that line without saying so is how the earlier
    # "outer loop does nothing" reading happened.
    for x, (_l, _b, _t, cost) in zip(xs, rows):
        ax.text(x, max(bottoms) * 0.02, cost, ha="center", va="bottom", fontsize=4.2,
                color=("#ffffff" if _t else "#999999"),
                style=("normal" if cost == "board" else "italic"), zorder=5)
    ax.axvline(0.5, color="#cccccc", lw=0.5, ls=":", zorder=1)
    ax.axvline(1.5, color=figstyle.BLACK, lw=0.7, zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=4.2)
    ax.set_ylabel("deadline misses (instances)")
    ax.set_title(f"{title}\nlevers: {', '.join(levers) or 'none'}", fontsize=6)
    ax.set_ylim(0, max(bottoms) * 1.30 or 1)
    att = attribution(rows, nets, attrib)
    if att and att[0]:
        ax.text(0.5, 0.97,
                f"residual: {att[0]} exec-bound + {att[1]} queueing",
                transform=ax.transAxes, ha="center", va="top", fontsize=4.4,
                color="#444444")
    ax.grid(axis="y", lw=0.3, color="#dddddd", zorder=0)
    figstyle.despine(ax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True,
                    metavar="NAME=REPORT.json=SPEC.json[=ATTRIBUTION.json]",
                    help="repeatable: display name, loop_report.json, workload spec")
    ap.add_argument("--stem", default="inner_outer_arc")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    figstyle.use()
    rungs, attribs = [], {}
    for spec in a.rung:
        parts = spec.split("=")
        name, report, wl = parts[0], parts[1], parts[2]
        if len(parts) > 3:
            attribs[name] = parts[3]
        rows, levers = arc(report, wl)
        rungs.append((name, rows, levers))

    nets = []
    for _n, rows, _l in rungs:
        for _lab, by, _t, _c in rows:
            for k in (by or {}):
                if k not in nets:
                    nets.append(k)

    # Three panels do not fit a double column at the two-panel label sizes; give the
    # row the full width and let the shortened beat labels carry the meaning.
    fig, axes = plt.subplots(1, len(rungs), figsize=(
        figstyle.DOUBLE_COL * min(1.0, 0.40 * len(rungs) + 0.20), 68 * figstyle.MM))
    fig.subplots_adjust(wspace=0.42)
    axes = [axes] if len(rungs) == 1 else list(axes)
    for i, ((name, rows, levers), ax) in enumerate(zip(rungs, axes)):
        panel(ax, rows, levers, name, nets, attribs.get(name))
        figstyle.panel_label(ax, "abcd"[i], x=-0.17, y=1.12)
    axes[0].legend(handles=[Patch(facecolor=figstyle.model_color(n), label=n)
                            for n in nets],
                   frameon=False, ncol=2, loc="upper left", fontsize=4.4)
    fig.text(0.5, -0.08,
             "Bars 1-2 are scored on PREDICTED per-dispatch profiles, bars 3-4 on costs "
             "MEASURED by executing bar 2's schedule on the K1 (w4 394 dispatches, w5 492, "
             "every network bit-exact or within 1.8e-4). Stacked by network because the "
             "reveal moves misses between networks: w5's board re-cost introduces a "
             "fused_full band absent from the predicted bar beside it.",
             ha="center", va="top", fontsize=4.2, color="#666666", wrap=True)
    png = figstyle.save(fig, a.stem, a.out_dir)
    print("wrote", png)
    json.dump({name: [{"beat": r[0], "by_network": r[1], "total": r[2], "cost": r[3]}
                      for r in rows] | {} if False else
               [{"beat": r[0], "by_network": r[1], "total": r[2], "cost": r[3]}
                for r in rows]
               for name, rows, _l in rungs},
              open(os.path.splitext(png)[0] + "_metrics.json", "w"), indent=1)
    print("wrote sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
