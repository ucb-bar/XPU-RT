#!/usr/bin/env python3
"""C3 -- LADDER MATRIX. The curated 9 arms x 4 tasks x 3 metrics, tightened.

The existing fig_metrics3 draws 12 separate axes with 12 legends and a six-line
prose caption. This is the same data with:
  * arms ordered by RELEASE PERIOD, not by name, because period is the driver;
  * one shared x-axis per column, so the ladder is read once;
  * the two reference arms marked by ANNOTATION (ring + word), not by a hue;
  * every panel's prose moved out; only labels and one short annotation remain.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
T = V.curated()
ARMS = sorted(V.CURATED_ARMS, key=lambda a: T["egg"][a]["period"])
PER = [T["egg"][a]["period"] for a in ARMS]
rng = np.random.default_rng(3)


def ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return 0.0, 0.0
    b = rng.choice(v, (3000, len(v))).mean(1)
    return abs(v.mean() - np.percentile(b, 2.5)), abs(np.percentile(b, 97.5) - v.mean())


ROWS = [("success", "success rate (%)", None, 1.0),
        ("mission", "mission time (s)\non successes", None, 1.0),
        ("energy", "actuator energy\n/ ideal arm", "ratio", 1.0)]

fig = plt.figure(figsize=(7.16, 4.30))
gs = fig.add_gridspec(3, 4, left=0.088, right=0.988, top=0.885, bottom=0.135,
                      wspace=0.16, hspace=0.16)

for r, (key, ylab, mode, sc) in enumerate(ROWS):
    for c, (task, name, robot, grid) in enumerate(V.TASKS):
        ax = fig.add_subplot(gs[r, c])
        base = T[task][V.REFERENCE][key] if mode == "ratio" else 1.0
        y = np.array([T[task][a][key] * sc / (base if mode == "ratio" else 1.0)
                      for a in ARMS])
        err = np.zeros((2, len(ARMS)))
        sk = {"success": "success_seeds", "energy": "energy_seeds"}.get(key)
        if sk:
            for i, a in enumerate(ARMS):
                lo, hi = ci(np.array(T[task][a][sk]) / (base if mode == "ratio" else 1.0))
                err[:, i] = lo, hi
        cols = [V.c_period(p) for p in PER]
        ax.bar(range(len(ARMS)), y, color=cols, width=0.74, linewidth=0, zorder=3)
        ax.errorbar(range(len(ARMS)), y, yerr=err, fmt="none", ecolor=V.INK2,
                    elinewidth=0.7, capsize=1.4, zorder=5)
        # reference arms marked by annotation, not by hue
        for a, mark, txt in ((V.REFERENCE, "o", "ideal"), (V.BASELINE, "X", "baseline")):
            i = ARMS.index(a)
            ax.plot([i], [y[i]], marker=mark, ms=6.0, ls="none", mfc="none",
                    mec=V.INK if a == V.REFERENCE else V.STATUS["critical"],
                    mew=1.2, zorder=7)
        if mode == "ratio":
            ax.axhline(1.0, color=V.INK2, lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.set_yscale("log")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xlim(-0.7, len(ARMS) - 0.3)
        V.tidy(ax)
        if r == 0:
            ax.set_title(f"{name}\n{robot}, {grid:.0f} ms actuation", loc="left",
                         pad=3, fontsize=7.6)
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{p:.0f}" for p in PER], fontsize=5.8, rotation=90)
            ax.set_xlabel("release period (ms)", fontsize=6.8)
        if c == 0:
            ax.set_ylabel(ylab, fontsize=7.0)
        else:
            ax.set_yticklabels([])
        if r == 0 and c == 0:
            ax.annotate("ideal", xy=(0, y[0]), xytext=(0.4, y[0] * 1.22), fontsize=6.2,
                        color=V.INK2,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=V.MUTED))
            i = ARMS.index(V.BASELINE)
            ax.annotate("QNN\nbaseline", xy=(i, y[i]), xytext=(i - 2.6, y[0] * 0.85),
                        fontsize=6.2, color=V.STATUS["critical"],
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=V.MUTED))
    if r == 0:
        pass

sm = plt.cm.ScalarMappable(norm=V.norm_period(min(PER), max(PER)), cmap=V.CMAP_PERIOD)
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.006)
cb.set_label("release period (ms)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0); cb.outline.set_visible(False)
fig.text(0.088, 0.960, "MEASURED  ·  9 schedules x 4 tasks x 10 seeds x 24 episodes  "
         "·  arms ordered by RELEASE PERIOD", fontsize=8.2, color=V.INK,
         fontweight="bold")
fig.text(0.088, 0.930, "error bars: bootstrap 95% CI over seeds", fontsize=6.6,
         color=V.MUTED)
V.save(fig, "cand_c3_ladder_matrix", f"order={[V.CURATED_LABEL[a] for a in ARMS]}")
