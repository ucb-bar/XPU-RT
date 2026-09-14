#!/usr/bin/env python3
"""C8 -- PERIOD DOMINATES; LATENCY READS BACKWARDS.

Two rows over the same four tasks and the same 44 operating points:
  top     success against RELEASE PERIOD      -- the real relationship
  bottom  success against OBSERVATION AGE     -- the same points, and the trend
                                                 flips sign, because the plane is
                                                 an anti-correlated frontier

Each panel prints its own Spearman rho, computed at draw time. The two google
tasks are actuator-saturated (333 ms actuation grid vs 40 ms on widowx) and are
flat in both rows -- the panel says so with one word, not a paragraph.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V

V.style()
T, PL = V.plane()
arms = sorted(T["egg"])
per = np.array([PL[a][1] for a in arms])
lat = np.array([PL[a][0] for a in arms])

fig = plt.figure(figsize=(7.16, 3.70))
gs = fig.add_gridspec(2, 4, left=0.075, right=0.988, top=0.845, bottom=0.100,
                      wspace=0.14, hspace=0.52)

for r, (x, xlab, tag) in enumerate([(per, "release period (ms)", "PERIOD"),
                                    (lat, "observation age (ms)", "AGE")]):
    for c, (task, name, robot, grid) in enumerate(V.TASKS):
        ax = fig.add_subplot(gs[r, c])
        y = np.array([T[task][a]["success"] for a in arms])
        rho = spearmanr(x, y).statistic
        ax.scatter(x, y, s=17, c=[V.c_period(p) for p in per], edgecolor="white",
                   linewidth=0.45, zorder=4)
        z = np.polyfit(np.log(x), y, 1)
        xs = np.linspace(x.min(), x.max(), 40)
        strong = abs(rho) >= 0.5
        ax.plot(xs, np.polyval(z, np.log(xs)),
                color=V.INK if strong else V.MUTED, lw=1.5 if strong else 1.0,
                ls="-" if strong else (0, (4, 2)), zorder=5)
        ax.text(0.97, 0.94, f"rho {rho:+.2f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.4, color=V.INK if strong else V.MUTED,
                fontweight="bold" if strong else "normal")
        ax.set_ylim(10, 55)
        V.tidy(ax)
        # BOTH rows carry their own x tick labels and axis label: the two rows
        # plot DIFFERENT x variables, so a shared unlabelled axis would be read
        # as one -- which is the exact confusion this figure exists to prevent.
        if r == 0:
            ax.set_title(f"{name}\n{robot} · {grid:.0f} ms actuation", loc="left",
                         pad=3, fontsize=7.6)
        ax.set_xlabel(xlab, fontsize=7.0, labelpad=1.5)
        if c == 0:
            ax.set_ylabel(f"success (%)\nvs {tag}", fontsize=7.2)
        else:
            ax.set_yticklabels([])
        if r == 0 and c == 2:
            ax.text(0.04, 0.08, "actuator-saturated", transform=ax.transAxes,
                    fontsize=6.4, color=V.MUTED)

fig.text(0.075, 0.975, f"MEASURED  ·  {len(arms)} operating points x 4 tasks x "
         f"8-10 seeds x 24 episodes", fontsize=8.2, color=V.INK, fontweight="bold",
         va="top")
fig.text(0.075, 0.938, f"the 44 points are a scheduler frontier, not a factorial grid: "
         f"Spearman(age, period) = {spearmanr(lat, per).statistic:+.2f}",
         fontsize=6.8, color=V.MUTED, va="top")
V.save(fig, "cand_c8_period_vs_latency")
