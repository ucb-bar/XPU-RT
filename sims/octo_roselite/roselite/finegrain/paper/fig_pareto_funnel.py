#!/usr/bin/env python3
"""Does a faster schedule buy GRASPS it cannot convert into PLACEMENTS? No. A NEGATIVE.

This is the most physically plausible trade-off the plane could have contained: reaching
the object is a coarse, open-loop-tolerant motion, while closing on it and placing it is
the part that actually needs a fresh observation. A schedule that grabbed more often and
placed no more often would be a genuine trade -- one axis up, the other flat or down.

It does not happen. The precursor stage and the terminal success move together at
r = +0.97 (eggplant), +0.93 (spoon), +0.73 (coke), +0.67 (drawer), and the conversion
rate RISES with the stage rate rather than falling: 37.8% -> 69.7% of eggplant grasps
reach the basket as the grasp rate goes 51 -> 78%. The frontier is one resolved point on
every task, and on spoon a single arm (g130_305) dominates all 43 others on both stages
at once. Freshness helps the whole funnel, not one stage of it.

THE STAGE STRICTLY CONTAINS SUCCESS on all four tasks, which is why every point lies
below the diagonal and why this is a nesting, not a ratio: nothing here divides one axis
by the other, unlike an energy-per-success chart. Stage is consecutive_grasp on the two
widowx tasks (a placement implies a stable grasp); `grasped` on coke, because coke's own
consec_grasp flag fires in only 20-30% of episodes against a 35-46% success rate and so
is the STRICTER event, not the precursor; and "drawer at least half closed" (qpos <=
0.10 m, against the qpos <= 0.05 m that scores) on drawer. Verified nested arm by arm.

Iso-conversion rays are drawn at 40/60/80%: an arm's ray is its grasp-to-place yield. If
the hypothesised trade existed, the fast arms would sit on LOWER rays than the slow ones.
They sit on higher ones.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pareto_lib as P

OUT = Path(__file__).parent / "fig_pareto_funnel.png"
STAGE = {"egg": "stable grasp (consecutive_grasp)",
         "spoon": "stable grasp (consecutive_grasp)",
         "coke": "can grasped (grasped)",
         "drawer": "drawer at least half closed (qpos ≤ 0.10 m)"}
TERM = {"egg": "placed in basket", "spoon": "placed on towel",
        "coke": "lifted (success)", "drawer": "closed (qpos ≤ 0.05 m)"}
LIVE = {"egg", "spoon"}

FIG_W, FIG_H, TOP, BOT, L, R, WS = 20.4, 7.0, 0.815, 0.245, 0.048, 0.988, 0.24
AX_H_PT = FIG_H * 72 * (TOP - BOT)
AX_W_PT = FIG_W * 72 * (R - L) / (4 + 3 * WS)

fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H), dpi=150,
                         gridspec_kw={"wspace": WS})
fig.subplots_adjust(top=TOP, bottom=BOT, left=L, right=R)
out = []
for ax, (task, name, embod) in zip(axes, P.TASKS):
    r = P.report(task, "stage", +1, "sr", +1)
    x, y, idx, res = r["x"], r["y"], r["idx"], r["res"]
    bx, by = r["bx"], r["by"]
    conv = 100.0 * y / x
    best = max(idx, key=lambda i: y[i])
    dom = [i for i in range(len(x))
           if i != best and x[i] <= x[best] and y[i] <= y[best]]

    xl, xh = x.min(), x.max(); lo, hi = y.min(), y.max()
    xlim = (xl - 0.20 * (xh - xl), xh + 0.16 * (xh - xl))
    ylim = (lo - 0.30 * (hi - lo) - by, hi + 0.26 * (hi - lo) + by)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # iso-conversion rays through the origin, labelled ALONG the ray with an opaque
    # backing: laid flat at the right edge they landed on the frontier labels and on the
    # data, and a dashed grey line crossing text reads as a strike-through
    ang = np.degrees(np.arctan2((AX_H_PT / (ylim[1] - ylim[0])),
                                (AX_W_PT / (xlim[1] - xlim[0]))))
    for c in (0.4, 0.6, 0.8):
        ax.plot([0, 300], [0, 300 * c], color=P.MUTE, lw=0.8, ls=(0, (4, 3)),
                alpha=0.55, zorder=1)
        # sit the label a fifth of the way up the panel, on the ray, if the ray passes
        # through the panel there; otherwise ride the right edge
        yy = ylim[0] + 0.14 * (ylim[1] - ylim[0])
        xx = yy / c
        if not (xlim[0] < xx < xlim[1]):
            xx = xlim[1] - 0.02 * (xlim[1] - xlim[0]); yy = c * xx
        if not (ylim[0] < yy < ylim[1] - 0.10 * (ylim[1] - ylim[0])):
            continue
        ax.text(xx, yy, f"{c:.0%} converts", ha="center", va="center", fontsize=6.6,
                color=P.MUTE, rotation=np.degrees(np.arctan(c)) * 0 + ang * c / c * 0
                + np.degrees(np.arctan(c * (AX_H_PT / (ylim[1] - ylim[0]))
                                       / (AX_W_PT / (xlim[1] - xlim[0])))),
                rotation_mode="anchor", zorder=2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8))

    fx = np.array([x[i] for i in idx]); fy = np.array([y[i] for i in idx])
    o = np.argsort(fx)
    ax.step(fx[o], fy[o], where="pre", color=P.INK, lw=1.6, alpha=0.55, zorder=3)
    for i, arm in enumerate(P.ARMS):
        on = i in idx
        ax.scatter(x[i], y[i], s=250 if on else 130, c=P.arm_colour(arm), zorder=4,
                   edgecolor=P.INK if on else P.EDGE,
                   linewidth=2.2 if on else 0.9, alpha=1.0 if on else 0.62)
    ax.errorbar(x[best], y[best], xerr=bx, yerr=by, color=P.INK, lw=1.5, capsize=4,
                zorder=5, alpha=0.85)
    cross = [(x[best] - bx, y[best] - 0.004 * (hi - lo),
              x[best] + bx, y[best] + 0.004 * (hi - lo)),
             (x[best] - 0.004 * (xh - xl), y[best] - by,
              x[best] + 0.004 * (xh - xl), y[best] + by)]
    P.place_labels(ax, [x[i] for i in idx], [y[i] for i in idx],
                   [P.ARMS[i] for i in idx], AX_W_PT, AX_H_PT, obstacles=cross)

    # the cloud runs up and to the right on every task, so the top-left corner is the
    # one place a three-line block cannot land on data or on an iso-conversion ray
    ax.text(0.035, 0.965,
            f"r = {r['corr']:+.2f}\nconversion {conv.min():.0f}→{conv.max():.0f}% "
            f"RISES with the stage\n{len(dom)} of 43 arms dominated by {P.ARMS[best]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
            color=P.INK if task in LIVE else P.MUTE,
            fontweight="bold" if task in LIVE else "normal")
    ax.set_xlabel(f"reached the stage: {STAGE[task]}  (%)", fontsize=8.0)
    ax.set_title(f"{name}   ·   {TERM[task]}\n{embod}\n"
                 f"{len(idx)} raw · {len(res)} resolved · {r['boot_median']:.0f} bootstrap"
                 f"   (of 44)" + ("" if task in LIVE else "   —  span < 3.5 bands"),
                 fontsize=9.4, color=P.INK if task in LIVE else P.MUTE)
    ax.grid(True, alpha=0.25, zorder=0)
    out.append((task, len(idx), len(res), r["boot_median"], r["corr"],
                conv.min(), conv.max(), len(dom), P.ARMS[best]))

axes[0].set_ylabel("completed the task (%)   —   20 seeds × 24 episodes", fontsize=9)
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="#7f8c8d", mec=P.INK, mew=2.2, ms=11,
           label="non-dominated"),
    Line2D([], [], ls=(0, (4, 3)), color=P.MUTE, lw=0.9,
           label="iso-conversion ray (stage → completion yield)"),
    Line2D([], [], marker="o", ls="", mfc="#45b39d", mec=P.EDGE, ms=8,
           label="dominated   (fill = release cadence, green fast → red slow)")],
    loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=3, frameon=False, fontsize=8.2)

fig.suptitle("The funnel does not split: a schedule that buys more grasps converts MORE "
             "of them, not fewer.  Preference is UP and RIGHT.", fontsize=12.6, y=0.972)
fig.text(0.5, 0.108,
         "A NEGATIVE RESULT. The hypothesis was that freshness helps the coarse reach "
         "and not the fine placement, so a fast schedule would grasp more and place no "
         "more. Instead both stages rise together (r = +0.97 / +0.93 / +0.73 / +0.67)\n"
         "and the yield between them rises too, so the fast arms sit on HIGHER "
         "iso-conversion rays, not lower. The stage strictly contains success on every "
         "task — verified arm by arm, which is why nothing lies above the diagonal — but "
         "the two axes are\nnested EVENTS, not a ratio: neither is divided by the other, "
         "so this is not the defect that fig_pareto_per_success.py carries. Coke and "
         "drawer are shown for completeness; their success spans 3.4 and 2.8 noise bands "
         "end to end and\ncarry no evidence either way. Companion: fig_pareto_compute.png "
         "(the one pair that trades) and fig_pareto_missiontime.png. Full sweep of 868 "
         "pairs in PARETO_SEARCH.md.",
         ha="center", va="top", fontsize=7.6, style="italic", color=P.CAPTION)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for t, raw, res, boot, c, cl, ch, dom, best in out:
    print(f"  {t:7s} raw={raw} resolved={res} boot={boot:.1f} r={c:+.2f} "
          f"conv {cl:.1f}-{ch:.1f}%  best={best} dominates {dom}/43")
