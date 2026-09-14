#!/usr/bin/env python3
"""B5 -- ENCODING BAKE-OFF. Six ways to draw ONE dataset, side by side.

The dataset is fixed: MEASURED eggplant success at each of the 44 operating
points. Only the encoding changes, so the panels can be compared as encodings
rather than as findings.

  1 grid heatmap on the REQUESTED (period, window) axes -- the familiar view
  2 scatter on the ACHIEVED (period, age) axes -- the honest view
  3 tricontour over the achieved plane -- interpolates where nothing was measured
  4 hexbin -- bins a 44-point cloud that has no density to bin
  5 beeswarm against period alone -- the variable that actually moves the metric
  6 slope of success against each candidate x, as a bar of |Spearman rho|

The point of the panel is that 1, 3 and 4 all LOOK like a response surface and
only 2 and 5 are supported by 44 samples on a one-dimensional frontier.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import vocab as V

V.style()
TASK = "egg"
T, PL = V.plane()
arms = sorted(T[TASK])
lat = np.array([PL[a][0] for a in arms])
per = np.array([PL[a][1] for a in arms])
suc = np.array([T[TASK][a]["success"] for a in arms])
req_p = np.array([float(a[1:].split("_")[0]) for a in arms])
req_w = np.array([float(a.split("_")[1]) for a in arms])
CM = "Blues"
norm = plt.Normalize(suc.min(), suc.max())

fig = plt.figure(figsize=(7.16, 4.15))
gs = fig.add_gridspec(2, 3, left=0.058, right=0.985, top=0.875, bottom=0.085,
                      wspace=0.30, hspace=0.46)
ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# 1 requested grid heatmap
P, W = sorted(set(req_p)), sorted(set(req_w))
M = np.full((len(W), len(P)), np.nan)
for p, w, s in zip(req_p, req_w, suc):
    M[W.index(w), P.index(p)] = s
ax[0].imshow(M, origin="lower", cmap=CM, norm=norm, aspect="auto",
             extent=(0, len(P), 0, len(W)))
for j in range(len(W)):
    for i in range(len(P)):
        if not np.isfinite(M[j, i]):
            ax[0].add_patch(Rectangle((i, j), 1, 1, facecolor=V.ABSENT,
                                      hatch=V.HATCH_INFEASIBLE, edgecolor="white",
                                      linewidth=0))
ax[0].set_xticks(np.arange(len(P)) + .5); ax[0].set_xticklabels([f"{int(p)}" for p in P], fontsize=5.4)
ax[0].set_yticks(np.arange(len(W)) + .5); ax[0].set_yticklabels([f"{int(w)}" for w in W], fontsize=5.4)
ax[0].set_xlabel("period requested"); ax[0].set_ylabel("window requested")
ax[0].set_title("1  requested-grid heatmap", loc="left", pad=3, fontsize=7.6)
ax[0].tick_params(length=0)

# 2 achieved scatter
sc = ax[1].scatter(per, lat, c=suc, cmap=CM, norm=norm, s=44, edgecolor=V.INK2,
                   linewidth=0.5, zorder=4)
ax[1].set_xlabel("period achieved (ms)"); ax[1].set_ylabel("age achieved (ms)")
ax[1].set_title("2  achieved-plane scatter", loc="left", pad=3, fontsize=7.6)
V.tidy(ax[1], grid="both")

# 3 tricontour
tri = mtri.Triangulation(per, lat)
ax[2].tricontourf(tri, suc, levels=9, cmap=CM, norm=norm)
ax[2].plot(per, lat, ".", ms=2.4, color=V.INK2, zorder=5)
ax[2].set_xlabel("period achieved (ms)")
ax[2].set_title("3  tricontour  ·  INTERPOLATES", loc="left", pad=3, fontsize=7.6)
V.tidy(ax[2], grid=None)

# 4 hexbin
ax[3].hexbin(per, lat, C=suc, gridsize=7, cmap=CM, norm=norm, linewidths=0.3,
             edgecolors="white")
ax[3].set_xlabel("period achieved (ms)"); ax[3].set_ylabel("age achieved (ms)")
ax[3].set_title("4  hexbin  ·  n=44, mostly 1/bin", loc="left", pad=3, fontsize=7.6)
V.tidy(ax[3], grid=None)

# 5 beeswarm vs period
rng = np.random.default_rng(0)
jit = rng.uniform(-0.30, 0.30, len(per))
ax[4].scatter(per + jit * 6, suc, s=30, c=[V.c_period(p) for p in per],
              edgecolor="white", linewidth=0.6, zorder=4)
z = np.polyfit(np.log(per), suc, 1)
xs = np.linspace(per.min(), per.max(), 40)
ax[4].plot(xs, np.polyval(z, np.log(xs)), color=V.INK2, lw=1.2, ls=(0, (4, 2)), zorder=5)
ax[4].set_xlabel("period achieved (ms)"); ax[4].set_ylabel("success (%)")
ax[4].set_title(f"5  vs period  ·  rho {spearmanr(per, suc).statistic:+.2f}",
                loc="left", pad=3, fontsize=7.6)
V.tidy(ax[4])

# 6 which x actually explains it
labs = ["period", "age", "makespan", "window req."]
xs_ = [per, lat, np.array([V.arms()[a][2] for a in arms]), req_w]
rr = [spearmanr(x, suc).statistic for x in xs_]
o = np.argsort(-np.abs(rr))
ax[5].barh(range(4), [abs(rr[i]) for i in o], color=[V.CMAP_PERIOD(0.30 + 0.14 * k)
                                                     for k in range(4)][::-1],
           height=0.6, linewidth=0)
for k, i in enumerate(o):
    ax[5].text(abs(rr[i]) + 0.02, k, f"{rr[i]:+.2f}", va="center", fontsize=6.8,
               color=V.INK, fontweight="bold")
ax[5].set_yticks(range(4)); ax[5].set_yticklabels([labs[i] for i in o], fontsize=6.8)
ax[5].set_xlim(0, 1.0); ax[5].invert_yaxis()
ax[5].set_xlabel("|Spearman rho| with success")
ax[5].set_title("6  what explains success", loc="left", pad=3, fontsize=7.6)
V.tidy(ax[5], grid="x")

cb = fig.colorbar(sc, ax=[ax[0], ax[1], ax[2]], fraction=0.016, pad=0.008)
cb.set_label("success (%)", fontsize=6.4); cb.ax.tick_params(labelsize=6.0)
cb.outline.set_visible(False)
fig.text(0.058, 0.955, f"ONE dataset, six encodings  ·  MEASURED eggplant success at "
         f"{len(arms)} operating points, 8-10 seeds x 24 episodes",
         fontsize=8.2, color=V.INK, fontweight="bold")
V.save(fig, "cand_b5_encoding_bakeoff")
