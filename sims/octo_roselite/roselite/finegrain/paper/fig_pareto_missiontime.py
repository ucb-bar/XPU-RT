#!/usr/bin/env python3
"""The speed-vs-reliability trade-off, which does not exist. A NEGATIVE result.

The obvious hypothesis for a second Pareto axis is that a schedule can buy success by
taking longer -- a slower, more corrective policy that wins more often but dawdles. The
plane says the opposite, and says it hard: on eggplant one arm, g130_275, has BOTH the
highest success rate and the shortest median time-to-completion of all 44, and dominates
every other arm on both axes simultaneously (43 of 43). On spoon the best arm dominates
41 of 43. r(success, mission time) = -0.84 (egg) and -0.88 (spoon): the schedules that
win more also finish sooner, because both come from the same cause -- an action that
reflects a fresher observation moves the gripper toward where the object actually is.

So the frontier here is ONE POINT, which is the defect this whole search was meant to
fix. It is drawn anyway because a null this clean is the finding: on a task the schedule
can move, a better schedule is simply better, and a paper that shows only the compute
frontier should show this too so the reader knows the alternative was looked for.

MISSION TIME IS CONDITIONED ON SUCCESS and must be: every failure on all four tasks ends
at the horizon (0 of the 52,460 failures, checked over all 84,480 episodes), so
unconditioned mean time is an exact affine function of success rate, 600*(1-p) + t*p on
eggplant (0 of 52,460 failing episodes ended early), and plotting it against success would be plotting success against itself. The
price of conditioning is survivorship: the episodes a weak arm happens to win are its
easy ones, which BIASES ITS TIME DOWNWARD and therefore works AGAINST the conclusion
drawn here. The effect survives the bias.

COKE AND DRAWER ARE NOT EVIDENCE EITHER WAY. Their success spans 3.4 and 2.8 noise bands
end to end and r is +0.09 and +0.00 (no tension either, just no signal); the
four-point frontiers they show are what 44
points scattered inside a noise band always show. Only the two widowx tasks have the
dynamic range to test the hypothesis at all.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import pareto_lib as P

OUT = Path(__file__).parent / "fig_pareto_missiontime.png"
LIVE = {"egg", "spoon"}          # the tasks whose success actually responds to schedule

FIG_W, FIG_H, TOP, BOT, L, R, WS = 20.4, 7.0, 0.815, 0.245, 0.045, 0.988, 0.24
AX_H_PT = FIG_H * 72 * (TOP - BOT)
AX_W_PT = FIG_W * 72 * (R - L) / (4 + 3 * WS)

fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H), dpi=150,
                         gridspec_kw={"wspace": WS})
fig.subplots_adjust(top=TOP, bottom=BOT, left=L, right=R)
out = []
for ax, (task, name, embod) in zip(axes, P.TASKS):
    r = P.report(task, "t_succ_ms", -1, "sr", +1)
    x, y, idx, res = r["x"] / 1000.0, r["y"], r["idx"], r["res"]
    bx, by = r["bx"] / 1000.0, r["by"]
    best = max(idx, key=lambda i: y[i])
    dom = [i for i in range(len(x))
           if i != best and x[i] >= x[best] and y[i] <= y[best]]

    # the quadrant the single best arm dominates: everything slower AND less reliable
    ax.add_patch(Rectangle((x[best], -1e3), 1e4, y[best] + 1e3, facecolor=P.EDGE,
                           alpha=0.16, edgecolor="none", zorder=1))
    fx = np.array([x[i] for i in idx]); fy = np.array([y[i] for i in idx])
    o = np.argsort(fx)
    ax.step(fx[o], fy[o], where="post", color=P.INK, lw=1.6, alpha=0.55, zorder=3)
    for i, arm in enumerate(P.ARMS):
        on = i in idx
        ax.scatter(x[i], y[i], s=250 if on else 130, c=P.arm_colour(arm), zorder=4,
                   edgecolor=P.INK if on else P.EDGE,
                   linewidth=2.2 if on else 0.9, alpha=1.0 if on else 0.62)
    # the 95% seed cross on the one arm the frontier consists of
    ax.errorbar(x[best], y[best], xerr=bx, yerr=by, color=P.INK, lw=1.5, capsize=4,
                zorder=5, alpha=0.85)
    lo, hi = y.min(), y.max()
    ax.set_ylim(lo - 0.28 * (hi - lo) - by, hi + 0.24 * (hi - lo) + by)
    xl, xh = x.min(), x.max()
    ax.set_xlim(xl - 0.22 * (xh - xl), xh + 0.13 * (xh - xl))
    # the 95% cross on the best arm is an obstacle too: text drawn across a whisker
    # reads as a strike-through
    cross = [(x[best] - bx, y[best] - 0.004 * (hi - lo),
              x[best] + bx, y[best] + 0.004 * (hi - lo)),
             (x[best] - 0.004 * (xh - xl), y[best] - by,
              x[best] + 0.004 * (xh - xl), y[best] + by)]
    P.place_labels(ax, [x[i] for i in idx], [y[i] for i in idx],
                   [P.ARMS[i] for i in idx], AX_W_PT, AX_H_PT, obstacles=cross)
    ax.text(0.97, 0.05,
            f"r = {r['corr'] * -1:+.2f}\n{len(dom)} of 43 arms dominated\nby "
            f"{P.ARMS[best]} on BOTH axes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
            color=P.INK if task in LIVE else P.MUTE,
            fontweight="bold" if task in LIVE else "normal")
    ax.set_xlabel("mission time on SUCCESS  (median sim seconds to finish a win)",
                  fontsize=8.4)
    ax.set_title(f"{name}\n{embod}\n"
                 f"{len(idx)} raw · {len(res)} resolved · {r['boot_median']:.0f} bootstrap"
                 f"   (of 44)" + ("" if task in LIVE else "   —  span < 3.5 bands"),
                 fontsize=9.4, color=P.INK if task in LIVE else P.MUTE)
    ax.grid(True, alpha=0.25, zorder=0)
    out.append((task, len(idx), len(res), r["boot_median"], r["corr"], len(dom),
                P.ARMS[best]))

axes[0].set_ylabel("success rate (%)   —   20 seeds × 24 episodes", fontsize=9)
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="#7f8c8d", mec=P.INK, mew=2.2, ms=11,
           label="non-dominated"),
    Line2D([], [], marker="s", ls="", mfc=P.EDGE, mec="none", ms=12, alpha=0.5,
           label="quadrant dominated by the single best arm (slower AND less reliable)"),
    Line2D([], [], marker="o", ls="", mfc="#45b39d", mec=P.EDGE, ms=8,
           label="dominated   (fill = release cadence, green fast → red slow)")],
    loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=3, frameon=False, fontsize=8.2)

fig.suptitle("The trade-off that isn't: success vs mission time.  Preference is UP and "
             "LEFT — and on both widowx tasks one arm owns that corner outright.",
             fontsize=12.6, y=0.972)
fig.text(0.5, 0.108,
         "A NEGATIVE RESULT, plotted so the reader can see the alternative was looked "
         "for. Faster schedules win more AND finish sooner: both follow from acting on a "
         "fresher observation, so the axes are not independent objectives but two\n"
         "readings of the same effect. Time is conditioned on SUCCESS because every "
         "failure on every task runs to the horizon (0 of 52,460 failing episodes ended "
         "early), which makes unconditioned time an exact affine function of success "
         "rate. Conditioning\ncosts survivorship bias — a weak arm is scored only on the "
         "episodes it happened to win, which shortens its time and works AGAINST the "
         "conclusion here. Coke and drawer span 3.4 and 2.8 noise bands end to end with "
         "r = +0.09 and +0.00: their\nfour-point frontiers are scatter inside a band, not "
         "operating points. Companion: fig_pareto_compute.png, the one pair that does "
         "trade.  Full sweep of 868 pairs in PARETO_SEARCH.md.",
         ha="center", va="top", fontsize=7.6, style="italic", color=P.CAPTION)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for t, raw, res, boot, c, dom, best in out:
    print(f"  {t:7s} raw={raw} resolved={res} boot={boot:.1f} r={-c:+.2f} "
          f"best={best} dominates {dom}/43")
