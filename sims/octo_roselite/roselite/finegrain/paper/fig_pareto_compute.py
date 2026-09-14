#!/usr/bin/env python3
"""Success vs accelerator work: the ONE axis pair on this plane with a real trade-off.

WHY THIS PAIR AND NOT ANOTHER. PARETO_SEARCH.md sweeps 868 pairs of task-level metrics
over the 44-arm plane. On the tasks where scheduling moves the needle at all (eggplant,
spoon), every pair of *task-quality* axes collapses: success, stable-grasp rate,
grasp->place conversion, time-to-completion, disturbance-on-failure and seed-to-seed
consistency are correlated at |r| = 0.84-0.98 and all improve together, so the fast
schedule dominates outright and the frontier is one point. The only quantity that gets
WORSE as the schedule gets better is the compute bought to make it better.

X IS NOT A MEASUREMENT. Inference rate is invocations per second of mission time, which
the release cadence fixes (1000 / the achieved cadence in arms.tsv, reproduced by the
logged n_inferences / sim_ms to a median 0.30% and a worst 1.39% -- the worst case being
g283_260 on spoon, whose 12 s horizon is short enough for pipeline warm-up to bias the
ratio). It carries no rollout noise: its 95% band is 0.0016 Hz against an axis range of
5.55 Hz. The entire uncertainty of
every point here is vertical, which is why the frontier survives the bootstrap as well as
it does. It also means half of this chart is the schmoo restated -- the honest reading is
"here is the price of the schmoo's x-axis", not "here is a newly discovered trade".

THE TRADE IS PHYSICAL, not an artifact of a shared variable. Success is an outcome of the
rollout; inference rate is the accelerator duty the schedule imposes. Neither appears in
the other's definition -- unlike fig_pareto_per_success.py, whose y-axis divides by the
success count that is also its x-axis.

Three counts per panel, all in the title, because they disagree and the disagreement is
the point: RAW non-dominated arms; RESOLVED, the subset separated from its frontier
neighbour by more than the 95% seed band on both axes; and the median frontier size over
2,000 seed bootstraps. Raw flatters, resolved is conservative, the bootstrap is what
would survive a re-run.

DRAWER IS THE CONTROL. Its success rate spans 34.4-42.7 points against a 3.0-point band,
r(success, rate) = +0.16: the task is insensitive to the schedule, so its "frontier" is
one resolved point and the six raw ones are noise. Nothing on drawer should be read as an
operating-point choice.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pareto_lib as P

OUT = Path(__file__).parent / "fig_pareto_compute.png"

FIG_H, TOP, BOT = 7.0, 0.815, 0.245
AX_H_PT = FIG_H * 72 * (TOP - BOT)          # panel height in points, fixed below
LBL_PT, TIER_PT, BASE_PT = 7.2, 42.0, 22.0  # label size / tier step / clear the ring

fig, axes = plt.subplots(1, 4, figsize=(20.4, FIG_H), dpi=150,
                         gridspec_kw={"wspace": 0.24})
fig.subplots_adjust(top=TOP, bottom=BOT, left=0.045, right=0.988)
summary = []
for ax, (task, name, embod) in zip(axes, P.TASKS):
    r = P.report(task, "inf_rate_hz", -1, "sr", +1)
    x, y, idx, res, hit = r["x"], r["y"], r["idx"], r["res"], r["hit"]
    lo, hi = y.min(), y.max()
    bot_pad = 0.34 * (hi - lo) + r["by"]

    fx = np.array([x[i] for i in idx]); fy = np.array([y[i] for i in idx])
    o = np.argsort(fx)
    ax.step(fx[o], fy[o], where="post", color=P.INK, lw=1.6, alpha=0.55, zorder=2)

    for i, arm in enumerate(P.ARMS):
        on = i in idx
        if hit[i] >= 0.5:                      # survives the seed bootstrap
            ax.scatter(x[i], y[i], s=560, facecolor="none", edgecolor=P.INK,
                       linewidth=0.9, alpha=0.45, zorder=3)
        ax.scatter(x[i], y[i], s=250 if on else 130, c=P.arm_colour(arm), zorder=4,
                   edgecolor=P.INK if on else P.EDGE,
                   linewidth=2.2 if on else 0.9, alpha=1.0 if on else 0.62)
    # Only the frontier is labelled; 44 labels would be unreadable and the interior
    # points are read off the colour ramp instead. Labels run VERTICALLY, because frontier
    # arms sit at distinct inference rates and upright text then cannot collide
    # horizontally. Arms that SHARE a cadence sit at the same x, so their labels are
    # stacked -- and stacked from the TOP MARKER OF THAT COLUMN, not from each arm's own
    # y: anchoring on the arm let the lower arm's label run straight through the upper
    # arm's marker (coke 180/320 through 180/275). A hairline leader keeps each label
    # attached to the point it names.
    col = {}
    for i in idx:
        col.setdefault(P.cadence_ms(P.ARMS[i]), []).append(i)
    tier, ytop = {}, {}
    for cad, members in col.items():
        top_y = max(y[j] for j in members)
        for k, i in enumerate(sorted(members, key=lambda j: -y[j])):
            tier[i], ytop[i] = k, top_y

    yb = lo - 0.16 * (hi - lo)
    ax.errorbar(x.min() + 0.25, yb, yerr=r["by"], color=P.MUTE, capsize=4, lw=1.4,
                zorder=3)
    ax.text(x.min() + 0.42, yb, f"±{r['by']:.1f} pts\n95% seed band", ha="left",
            va="center", fontsize=6.6, color=P.MUTE)
    # Headroom is solved for, not guessed: a rotated label needs BASE + 42 per repeated
    # cadence + its own text length in POINTS, and the panel is a fixed 287 pt tall, so
    # the y-range that just clears the tallest label is a closed form. Guessing a
    # fraction of the data range clipped the eggplant labels at one tier and left a
    # third of the drawer panel blank at the next.
    need = max((ytop[i] - lo + bot_pad) /
               max(1.0 - (BASE_PT + TIER_PT * tier[i]
                          + 0.62 * LBL_PT * len(P.ARMS[i]) + 6.0) / AX_H_PT, 0.15)
               for i in idx)
    ax.set_ylim(lo - bot_pad, lo - bot_pad + need * 1.015)
    ax.set_xlim(x.min() - 0.45, x.max() + 0.45)
    pt = AX_H_PT / (ax.get_ylim()[1] - ax.get_ylim()[0])   # points per success point
    for i in idx:
        dy = (ytop[i] - y[i]) * pt + BASE_PT + TIER_PT * tier[i]
        ax.annotate(P.ARMS[i].replace("_", "/")[1:], (x[i], y[i]),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    va="bottom", rotation=90, fontsize=LBL_PT,
                    fontweight="bold" if i in res else "normal",
                    color=P.INK if i in res else P.MUTE, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=P.EDGE, lw=0.6,
                                    shrinkA=1.0, shrinkB=9.5))
    ax.set_xlabel("inference rate  (invocations per second of mission time)", fontsize=8.4)
    ax.set_title(f"{name}\n{embod}\n"
                 f"{len(idx)} raw · {len(res)} resolved · {r['boot_median']:.0f} bootstrap"
                 f"   (of 44)", fontsize=9.4)
    ax.grid(True, alpha=0.25, zorder=0)

    top = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / np.maximum(v, 1e-6),
                                               lambda v: 1000.0 / np.maximum(v, 1e-6)))
    top.set_xticks([283, 250, 220, 200, 180, 165, 150, 140, 130, 120, 110])
    top.set_xticklabels([283, 250, 220, 200, 180, 165, 150, 140, 130, 120, 110],
                        fontsize=6.2, color=P.MUTE)
    top.set_xlabel("release cadence (ms)", fontsize=7.2, color=P.MUTE, labelpad=2)
    summary.append((task, len(idx), len(res), r["boot_median"], len(r["stable"]),
                    r["corr"], [P.ARMS[i] for i in sorted(idx, key=lambda i: x[i])]))

axes[0].set_ylabel("success rate (%)   —   20 seeds × 24 episodes", fontsize=9)
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="#7f8c8d", mec=P.INK, mew=2.2, ms=11,
           label="non-dominated (20-seed mean)"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=P.INK, mew=0.9, ms=17,
           label="on the frontier in ≥50% of 2,000 seed bootstraps"),
    Line2D([], [], marker="o", ls="", mfc="#45b39d", mec=P.EDGE, ms=8,
           label="dominated   (fill = release cadence, green fast → red slow)")],
    loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=3, frameon=False, fontsize=8.2)

fig.suptitle("Pareto frontier: task success vs accelerator work  —  the only pair of "
             "axes on this plane that genuinely trades.  Preference is UP and LEFT.",
             fontsize=12.6, y=0.972)
fig.text(0.5, 0.108,
         "The x-axis is not a measurement: inference rate is 1000/cadence, fixed by the "
         "schedule (reproduced by the logged n_inferences/sim_ms to a median 0.30%), so "
         "its band is 0.0016 Hz against a 5.55 Hz range and every point's uncertainty is\n"
         "vertical. That makes half this chart the schmoo restated — read it as the PRICE "
         "of the schmoo's x-axis, not as a newly discovered trade. Success and rate "
         "appear in neither's definition, so unlike an energy-per-success chart nothing "
         "here is shared between the axes. RAW counts arms nothing dominates; RESOLVED "
         "keeps only those separated\nfrom their frontier neighbour by more than the "
         "95% seed band on both axes; BOOTSTRAP is the median frontier size over 2,000 "
         "resamples of the 20 seeds. DRAWER IS A CONTROL: r(success, rate) = +0.16, its "
         "whole span is 2.8 bands wide, and its six raw\nfrontier arms collapse to one "
         "under either test — that task does not respond to scheduling and its frontier "
         "is not an operating-point menu. All 868 pairs, and the 230 that survive both a "
         "direction and a shared-variable check, are in PARETO_SEARCH.md.",
         ha="center", va="top", fontsize=7.6, style="italic", color=P.CAPTION)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for t, raw, res, boot, stab, c, arms in summary:
    print(f"  {t:7s} raw={raw:2d} resolved={res:2d} boot={boot:4.1f} stable={stab:2d} "
          f"corr={c:+.2f}\n          {arms}")
