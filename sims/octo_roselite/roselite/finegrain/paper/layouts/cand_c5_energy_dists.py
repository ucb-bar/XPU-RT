#!/usr/bin/env python3
"""C5 -- ENERGY DISTRIBUTION: four encodings of the same per-episode sample.

The energy result is quoted as a per-arm scalar everywhere else. That scalar is
a MEDIAN of a heavy-tailed per-episode distribution, and which encoding you pick
decides whether a reader can see that.

  1 ridgeline (KDE per arm, stacked)   -- shape, at the cost of a baseline offset
  2 ECDF                               -- exact, no smoothing parameter, crossings
  3 violin + median rule               -- shape + the quoted statistic together
  4 strip of raw episodes              -- 216 dots; nothing is hidden

Same arms, same colour ramp, same x scale in all four.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
TASK = "egg"
T = V.curated()
ARMS = sorted(V.CURATED_ARMS, key=lambda a: T[TASK][a]["period"])
E = {a: np.array([e[V.ECH] for e in T[TASK][a]["episodes"]]) / 1000.0 for a in ARMS}
PER = {a: T[TASK][a]["period"] for a in ARMS}
LO, HI = min(v.min() for v in E.values()), max(v.max() for v in E.values())

fig = plt.figure(figsize=(7.16, 4.05))
gs = fig.add_gridspec(2, 2, left=0.115, right=0.985, top=0.875, bottom=0.085,
                      wspace=0.19, hspace=0.36)
ax = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
xs = np.linspace(np.log10(LO), np.log10(HI), 320)

# 1 ridgeline
from scipy.stats import gaussian_kde
for i, a in enumerate(ARMS):
    d = gaussian_kde(np.log10(E[a]), bw_method=0.32)(xs)
    d = d / d.max() * 0.92
    y = len(ARMS) - 1 - i
    ax[0].fill_between(10 ** xs, y, y + d, color=V.c_period(PER[a]), lw=0, zorder=3 + i)
    ax[0].plot(10 ** xs, y + d, color="white", lw=0.7, zorder=3 + i)
    ax[0].plot([np.median(E[a])], [y], marker="|", ms=7, mew=1.4, color=V.INK,
               zorder=20 + i)
ax[0].set_yticks(np.arange(len(ARMS)) + 0.15)
ax[0].set_yticklabels([f"{V.CURATED_LABEL[a]}  {PER[a]:.0f} ms" for a in ARMS][::-1],
                      fontsize=6.4)
ax[0].set_ylim(-0.25, len(ARMS) + 0.10)
ax[0].set_title("1  ridgeline  ·  tick = median", loc="left", pad=3, fontsize=7.6)

# 2 ECDF
for a in ARMS:
    v = np.sort(E[a])
    ax[1].step(v, np.arange(1, len(v) + 1) / len(v), where="post",
               color=V.c_period(PER[a]), lw=1.3, zorder=3)
ax[1].set_ylabel("fraction of episodes", fontsize=7.0)
ax[1].set_title("2  ECDF  ·  no smoothing parameter", loc="left", pad=3, fontsize=7.6)
ax[1].text(E[V.REFERENCE].max() * 1.1, 0.10, V.CURATED_LABEL[V.REFERENCE],
           fontsize=6.2, color=V.INK2)
ax[1].text(E[V.BASELINE].min() * 0.55, 0.90, V.CURATED_LABEL[V.BASELINE],
           fontsize=6.2, color=V.INK2, ha="right")

# 3 violin
parts = ax[2].violinplot([np.log10(E[a]) for a in ARMS], positions=range(len(ARMS)),
                         showextrema=False, widths=0.86)
for b, a in zip(parts["bodies"], ARMS):
    b.set_facecolor(V.c_period(PER[a])); b.set_alpha(0.95); b.set_edgecolor("white")
    b.set_linewidth(0.6)
ax[2].plot(range(len(ARMS)), [np.log10(np.median(E[a])) for a in ARMS], "_",
           ms=11, mew=1.6, color=V.INK, zorder=6)
ax[2].set_xticks(range(len(ARMS)))
ax[2].set_xticklabels([f"{PER[a]:.0f}" for a in ARMS], fontsize=5.8, rotation=90)
ax[2].set_ylabel("log10 energy", fontsize=7.0)
ax[2].set_xlabel("release period (ms)", fontsize=7.0)
ax[2].set_title("3  violin + median rule", loc="left", pad=3, fontsize=7.6)
V.tidy(ax[2])

# 4 raw strip
rng = np.random.default_rng(11)
for i, a in enumerate(ARMS):
    v = E[a]
    ax[3].plot(v, i + rng.uniform(-0.26, 0.26, len(v)), ".", ms=2.6,
               color=V.c_period(PER[a]), alpha=0.85, zorder=3)
    ax[3].plot([np.median(v)], [i], marker="|", ms=10, mew=1.5, color=V.INK, zorder=6)
ax[3].set_yticks(range(len(ARMS)))
ax[3].set_yticklabels([f"{PER[a]:.0f}" for a in ARMS], fontsize=6.2)
ax[3].set_ylabel("release period (ms)", fontsize=7.0)
ax[3].set_title(f"4  raw strip  ·  {sum(len(v) for v in E.values())} episodes",
                loc="left", pad=3, fontsize=7.6)

for k in (0, 1, 3):
    ax[k].set_xscale("log")
    ax[k].set_xlim(LO * 0.85, HI * 1.18)
    ax[k].set_xlabel("actuator energy per episode  (N^2 m^2 s, x1000)", fontsize=7.0)
    V.tidy(ax[k], grid="x")

fig.text(0.115, 0.955, f"MEASURED  ·  eggplant in basket  ·  one point per episode  ·  "
         f"integrated PD drive torque, NOT calibrated to joules",
         fontsize=8.0, color=V.INK, fontweight="bold")
V.save(fig, "cand_c5_energy_dists")
