#!/usr/bin/env python3
"""D1 -- FIGURE 1. One page carrying MECHANISM, SEARCH and IMPACT.

  A  the cascade: the baseline's one inference against the pipelined schedule's
     lanes, on one shared 700 ms window, with the release rail.
  B  the search: 108 grid cells, what solved, and the 44 distinct operating
     points it yielded against greedy's 12.
  C  the impact: success against release period on the two widowx tasks, with
     the baseline and the ideal arm marked.

Every panel uses the vocabulary in cand_d2_keysheet: the rail, the lane hues,
the period ramp, the hatched absent cells. Nothing on the canvas is a sentence.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import vocab as V

V.style()
fig = plt.figure(figsize=(7.16, 5.90))
GS = fig.add_gridspec(3, 1, left=0.0, right=1.0, top=1.0, bottom=0.0,
                      height_ratios=[1.00, 0.98, 0.92], hspace=0.0)


def band(i, s):
    fig.text(0.030, s, "", fontsize=1)


# ============================================================ A  MECHANISM
WIN = 700.0
fig.text(0.030, 0.988, "A   MECHANISM   ·   one 700 ms window on the QRB5165, MEASURED",
         fontsize=8.6, color=V.INK, fontweight="bold", va="top")
gsA = fig.add_gridspec(2, 1, left=0.155, right=0.760, top=0.898, bottom=0.742,
                       hspace=0.62)
for r, (title, group, tile) in enumerate([("QNN CPU baseline", "mono", None),
                                          ("XPU-RT pipelined  p150/w300", "p150w300", None)]):
    ax = fig.add_subplot(gsA[r])
    _, rows, _ = V.board_median(group)
    iv = V.inferences(rows)
    st = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    per = float(np.median(np.diff(st))) if len(st) > 1 else span
    for rr in rows:
        if rr["lane"] not in V.LANE or rr["s"] > WIN:
            continue
        ax.barh(V.LANE_ORDER.index(rr["lane"]), min(rr["e"], WIN) - rr["s"],
                left=rr["s"], height=0.60, color=V.LANE[rr["lane"]], linewidth=0,
                zorder=3)
    ax.set_ylim(2.98, -1.05)
    V.release_rail(ax, -0.78, 0, WIN, per, span, ms=5.2)
    ax.set_xlim(-6, WIN + 6)
    ax.set_yticks(range(3))
    ax.set_yticklabels(V.LANE_ORDER, fontsize=6.6)
    for t, l in zip(ax.get_yticklabels(), V.LANE_ORDER):
        t.set_color(V.LANE[l]); t.set_fontweight("bold")
    done = int(np.sum(np.arange(0, WIN, per) + span <= WIN))
    ax.set_title(f"{title}      age {span:.0f} ms  ·  period {per:.0f} ms  ·  "
                 f"{done} action{'s' if done != 1 else ''}", loc="left", pad=2.5,
                 fontsize=7.6)
    V.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    if r == 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("ms since window start", fontsize=7.0, labelpad=1.5)

axS = fig.add_axes([0.790, 0.742, 0.195, 0.156]); axS.axis("off")
axS.set_xlim(0, 1); axS.set_ylim(0, 1)
for i, (val, lab) in enumerate([(f"{sum(V.PARTITION.values())}", "single-graph contexts"),
                                (f"{V.N_DISPATCH}", "dispatches"),
                                (f"{V.N_EVICT}", "context evictions")]):
    axS.text(0.0, 0.86 - 0.33 * i, val, fontsize=15, color=V.INK, fontweight="bold",
             va="center")
    axS.text(0.30, 0.86 - 0.33 * i, lab, fontsize=6.3, color=V.INK2, va="center")
axS.text(0.0, 0.99, "  ".join(f"{l} {V.PARTITION[l]}" for l in V.LANE_ORDER),
         fontsize=6.4, color=V.MUTED, va="top")

# =============================================================== B  SEARCH
fig.text(0.030, 0.668, "B   SEARCH   ·   warm-started CP-SAT over a "
         "(release period x deadline window) grid, PREDICTED",
         fontsize=8.6, color=V.INK, fontweight="bold", va="top")
cells, tally, _ = V.grid_warm()
greedy = V.grid_greedy()
P = sorted({c["p"] for c in cells.values()})
W = sorted({c["w"] for c in cells.values()})
ages = [c["lat_med"] for c in cells.values() if "lat_med" in c]
norm = plt.Normalize(min(ages), max(ages))
axG = fig.add_axes([0.062, 0.408, 0.300, 0.182])
for i, p in enumerate(P):
    for j, w in enumerate(W):
        c = cells.get(f"{p}_{w}")
        if c and "lat_med" in c:
            axG.add_patch(Rectangle((i, j), 1, 1, linewidth=0,
                                    facecolor=V.CMAP_PERIOD(0.12 + 0.80 * norm(c["lat_med"]))))
        else:
            h = V.HATCH_INFEASIBLE if (c or {}).get("status") == "INFEASIBLE" else V.HATCH_UNKNOWN
            axG.add_patch(Rectangle((i, j), 1, 1, facecolor=V.ABSENT, linewidth=0,
                                    hatch=h, edgecolor="white"))
            axG.text(i + .5, j + .5, "x" if h == V.HATCH_INFEASIBLE else "?",
                     ha="center", va="center", fontsize=4.6, color=V.INK2)
axG.set_xlim(0, len(P)); axG.set_ylim(0, len(W))
axG.set_xticks(np.arange(len(P)) + .5); axG.set_xticklabels(P, fontsize=5.0, rotation=90)
axG.set_yticks(np.arange(len(W)) + .5); axG.set_yticklabels(W, fontsize=5.0)
axG.set_xlabel("release period requested (ms)", fontsize=6.8)
axG.set_ylabel("deadline window (ms)", fontsize=6.8)
axG.tick_params(length=0)
for s_ in ("top", "right"):
    axG.spines[s_].set_visible(False)
axG.set_title("108 cells  ·  71 solve", loc="left", pad=2.5, fontsize=7.6)

A = V.arms()
lat = np.array([v[0] for v in A.values()]); per = np.array([v[1] for v in A.values()])
gl = np.array([c["lat_med"] for c in greedy.values()])
gp = np.array([c["cadence"] for c in greedy.values()])
axF = fig.add_axes([0.430, 0.408, 0.245, 0.182])
axF.scatter(gp, gl, s=13, facecolor="none", edgecolor=V.MUTED, linewidth=0.6, zorder=3)
axF.scatter(per, lat, s=22, c=[V.c_period(p) for p in per], edgecolor="white",
            linewidth=0.5, zorder=5)
axF.set_xlabel("achieved release period (ms)", fontsize=6.8)
axF.set_ylabel("achieved observation age (ms)", fontsize=6.8)
axF.set_title(f"44 distinct points  ·  rho {spearmanr(lat, per).statistic:+.2f}",
              loc="left", pad=2.5, fontsize=7.6)
axF.text(0.97, 0.95, "CP-SAT 44", transform=axF.transAxes, ha="right", va="top",
         fontsize=6.4, color=V.CMAP_PERIOD(0.75), fontweight="bold")
axF.text(0.97, 0.85, "greedy 12", transform=axF.transAxes, ha="right", va="top",
         fontsize=6.4, color=V.MUTED)
V.tidy(axF, grid="both")

axFu = fig.add_axes([0.742, 0.408, 0.243, 0.182]); axFu.axis("off")
axFu.set_xlim(-38, 120); axFu.set_ylim(-0.6, 5.6)
for i, (lab, v, sh) in enumerate([("grid cells", 108, 0.30), ("CP-SAT solve", 71, 0.50),
                                  ("CP-SAT distinct", 44, 0.72),
                                  ("greedy solve", 108, 0.18),
                                  ("greedy distinct", 12, 0.18)]):
    y = 4.8 - i * 1.05 - (0.30 if i >= 3 else 0)
    axFu.add_patch(Rectangle((0, y - 0.30), v, 0.58, facecolor=V.CMAP_PERIOD(sh),
                             linewidth=0))
    axFu.text(v + 2.5, y, f"{v}", va="center", fontsize=7.6, color=V.INK,
              fontweight="bold")
    axFu.text(-2.5, y, lab, va="center", ha="right", fontsize=6.4, color=V.INK2)
axFu.text(0, 5.50, "3.7x more operating points", fontsize=6.6,
          color=V.INK, va="top", fontweight="bold")

# =============================================================== C  IMPACT
fig.text(0.030, 0.348, "C   IMPACT   ·   MEASURED in SIMPLER-env, 44 operating points "
         "x 8-10 seeds x 24 episodes", fontsize=8.6, color=V.INK, fontweight="bold",
         va="top")
T, PL = V.plane()
CUR = V.curated()
arms = sorted(T["egg"])
pp = np.array([PL[a][1] for a in arms])
gsC = fig.add_gridspec(1, 3, left=0.062, right=0.985, top=0.286, bottom=0.105,
                       wspace=0.32)
for c, (task, name) in enumerate([("egg", "eggplant in basket"),
                                  ("spoon", "spoon on towel")]):
    ax = fig.add_subplot(gsC[c])
    y = np.array([T[task][a]["success"] for a in arms])
    rho = spearmanr(pp, y).statistic
    ax.scatter(pp, y, s=17, c=[V.c_period(p) for p in pp], edgecolor="white",
               linewidth=0.45, zorder=4)
    z = np.polyfit(np.log(pp), y, 1)
    xs = np.linspace(pp.min(), 700, 60)
    ax.plot(xs, np.polyval(z, np.log(xs)), color=V.INK, lw=1.3, zorder=5)
    for arm, mk, col, lab in ((V.REFERENCE, "o", V.INK, "ideal"),
                              (V.BASELINE, "X", V.STATUS["critical"], "QNN baseline")):
        d = CUR[task][arm]
        ax.plot([d["period"]], [d["success"]], marker=mk, ms=8, ls="none", mfc="none",
                mec=col, mew=1.5, zorder=7)
        ax.annotate(lab, xy=(d["period"], d["success"]),
                    xytext=(-5 if arm == V.BASELINE else 8,
                            11 if arm == V.BASELINE else 9),
                    textcoords="offset points", fontsize=6.2, color=col,
                    ha="right" if arm == V.BASELINE else "left")
    ax.set_xscale("log")
    ax.set_xlim(95, 900)
    ax.set_xticks([110, 200, 400, 685])
    ax.set_xticklabels(["110", "200", "400", "685"])
    ax.set_xticks([], minor=True)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(-3, 60)
    ax.set_xlabel("release period (ms)", fontsize=7.0)
    ax.set_title(f"{name}   ·   rho {rho:+.2f}", loc="left", pad=2.5, fontsize=7.6)
    if c == 0:
        ax.set_ylabel("success rate (%)", fontsize=7.2)
    V.tidy(ax, grid="both")

axD = fig.add_subplot(gsC[2])
BEST = "p105w300"
for i, (task, nm, rb, gr) in enumerate(V.TASKS):
    yy = 3 - i
    a, b = CUR[task][V.BASELINE]["success"], CUR[task][BEST]["success"]
    axD.plot([a, b], [yy, yy], color=V.GRID, lw=3.2, zorder=2, solid_capstyle="round")
    axD.plot([a], [yy], "X", ms=6.5, color=V.STATUS["critical"], mec="white", mew=0.9,
             zorder=6)
    axD.plot([b], [yy], "o", ms=6.0, color=V.CMAP_PERIOD(0.30), mec="white", mew=0.9,
             zorder=5)
    axD.text(max(a, b) + 2.4, yy, f"{b - a:+.0f} pt", va="center", fontsize=6.6,
             color=V.INK, fontweight="bold")
    if abs(b - a) < 2.0:
        axD.annotate("no change", xy=(a, yy), xytext=(9, -9),
                     textcoords="offset points", ha="left", fontsize=5.8,
                     color=V.MUTED)
axD.set_yticks(range(4))
axD.set_yticklabels([f"{n}\n{r}" for _, n, r, g in V.TASKS][::-1],
                    fontsize=6.0)
axD.set_ylim(-0.6, 3.6)
axD.set_xlim(-4, 78)
axD.set_xlabel("success rate (%)", fontsize=7.0)
axD.set_title("baseline -> XPU-RT p105/w300", loc="left", pad=2.5, fontsize=7.6)
V.tidy(axD, grid="x")

fig.legend(handles=V.glyph_handles(["release", "complete"])
           + [plt.Line2D([], [], marker="s", ls="none", ms=5, color=V.LANE[l], label=l)
              for l in V.LANE_ORDER]
           + [Rectangle((0, 0), 1, 1, facecolor=V.ABSENT, hatch=V.HATCH_INFEASIBLE,
                        edgecolor="white", label="x infeasible / ? unknown"),
              plt.Line2D([], [], marker="X", ls="none", ms=6, mfc="none",
                         mec=V.STATUS["critical"], label="QNN baseline"),
              plt.Line2D([], [], marker="o", ls="none", ms=6, mfc="none", mec=V.INK,
                         label="ideal 0 ms")],
           loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=8, handlelength=1.1,
           columnspacing=1.2)
V.save(fig, "cand_d1_splash")
