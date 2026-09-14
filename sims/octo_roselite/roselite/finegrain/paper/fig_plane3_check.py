#!/usr/bin/env python3
"""Sanity checks on the in-flight 44-arm plane sweep.

Three questions, one row each. None of them is a result -- they are the checks
that decide whether the plane's numbers are worth reading at all.

1. DOES THE DOMINANT AXIS BEHAVE?  The 44 operating points are a scheduler
   Pareto front, not a factorial grid: across them Spearman(latency, period) =
   -0.62, so an arm buys a shorter release period by paying observation age.
   Regressing a metric on latency alone therefore reads BACKWARDS. Period is the
   axis plotted; latency is carried as the point colour so the confound stays
   visible instead of being averaged away.

2. DOES ENERGY MOVE WITH SUCCESS, OR AGAINST IT?  If shorter periods bought
   success by spending actuator effort, this would be a genuine trade and the
   Pareto story would be a curve. Plotted so the sign can be read directly.

3. IS THE ENERGY CHANNEL WELL BEHAVED PER EPISODE?  The per-cell number is a
   summary of 24 episodes, and it is only trustworthy if those episodes are one
   population. They are not on close_drawer: a fifth of its episodes integrate to
   ~0, meaning the arm never moved for the full 1017 ticks, and NONE of them ever
   succeed. A bimodal channel makes both the mean and the median unstable, which
   is why the drawer energy panel must not be read as "energy is flat".

4. DOES IT REPRODUCE?  Independent evidence: the earlier 20-seed sweep measured
   success on these same 44 arms with different seeds (100-129 vs 300+) and a
   different harness version. The surfaces should agree where the surface has
   structure -- and should NOT correlate where the earlier work already showed
   the surface is flat to within a few noise bands (coke, drawer). Both outcomes
   are checks; only a strong DISagreement on egg/spoon would be a failure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import plane3_lib as L

HERE = Path(__file__).parent
OUT = HERE / "fig_plane3_check.png"
# 20-seed empirical noise band on a paired mean (g5grid/analyze_merged.py).
FLOOR = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}

T = L.table()
plane = L.arm_plane()
old = json.load(open(HERE / "grid_e2e_success.json"))

LAT = np.array([plane[a][0] for a in plane])
PER = np.array([plane[a][1] for a in plane])
GEOM = spearmanr(LAT, PER).statistic

D_RAW = L.load()
NEAR_ZERO = 100.0     # N2m2s; four orders below every task median

fig, axes = plt.subplots(4, len(L.TASKS), figsize=(20.0, 16.4), dpi=145,
                         gridspec_kw={"hspace": 0.58, "wspace": 0.28})
assert axes.shape == (4, len(L.TASKS)), axes.shape

sc = None
for c, (task, lab, emb) in enumerate(L.TASKS):
    arms = sorted(T[task])
    per = np.array([plane[a][1] for a in arms])
    lat = np.array([plane[a][0] for a in arms])
    suc = np.array([T[task][a]["success"] for a in arms])
    ene = np.array([T[task][a]["energy"] for a in arms])
    ene = ene / ene.min()

    # ---- row 0: success vs the dominant axis -------------------------------
    ax = axes[0, c]
    sc = ax.scatter(per, suc, c=lat, cmap="viridis", s=58, edgecolor="white",
                    linewidth=0.7, zorder=3)
    b, a0 = np.polyfit(per, suc, 1)
    xs = np.linspace(per.min(), per.max(), 2)
    ax.plot(xs, a0 + b * xs, color="#16161C", lw=1.3, ls="--", alpha=0.65, zorder=2)
    r = spearmanr(per, suc)
    ax.set_title(f"{lab}   [{emb}]\nsuccess vs release period:  "
                 f"rho = {r.statistic:+.3f}  (p = {r.pvalue:.2g})", fontsize=9.2)
    ax.set_xlabel("release period requested (ms)", fontsize=8.4)
    if c == 0:
        ax.set_ylabel("success rate (%)\nmean over seeds so far", fontsize=8.6)
    ax.grid(alpha=0.22, lw=0.6, zorder=0)

    # ---- row 1: energy vs the same axis ------------------------------------
    ax = axes[1, c]
    ax.scatter(per, ene, c=lat, cmap="viridis", s=58, edgecolor="white",
               linewidth=0.7, zorder=3)
    b, a0 = np.polyfit(per, ene, 1)
    ax.plot(xs, a0 + b * xs, color="#16161C", lw=1.3, ls="--", alpha=0.65, zorder=2)
    re_ = spearmanr(per, ene)
    rse = spearmanr(suc, ene)
    if abs(rse.statistic) < 0.2:
        verdict = "flat: neither axis moves it"
    elif rse.statistic < 0:
        verdict = "co-optimised, not a trade"
    else:
        verdict = "genuine trade"
    ax.set_title(f"energy vs release period:  rho = {re_.statistic:+.3f} (p = {re_.pvalue:.2g})\n"
                 f"energy vs success:  rho = {rse.statistic:+.3f}   [{verdict}]",
                 fontsize=8.8)
    ax.set_xlabel("release period requested (ms)", fontsize=8.4)
    if c == 0:
        ax.set_ylabel("actuator energy\n/ cheapest arm", fontsize=8.6)
    ax.grid(alpha=0.22, lw=0.6, zorder=0)

    # ---- row 2: per-episode energy distribution ----------------------------
    ax = axes[2, c]
    ep = np.array([x[L.ECH] for (t, a), sd in D_RAW.items() if t == task
                   for d in sd.values() for x in d["episodes"]])
    su = np.array([x["success"] for (t, a), sd in D_RAW.items() if t == task
                   for d in sd.values() for x in d["episodes"]])
    pos = ep > 0
    bins = np.logspace(np.log10(ep[pos].min()), np.log10(ep.max()), 46)
    ax.hist(ep[pos & ~su], bins=bins, color="#c0392b", alpha=0.62, label="failed")
    ax.hist(ep[pos & su], bins=bins, color="#148f77", alpha=0.62, label="succeeded")
    ax.set_xscale("log")
    zf = float((ep < NEAR_ZERO).mean())
    ax.axvspan(bins[0], NEAR_ZERO, color="#16161C", alpha=0.10, zorder=0)
    ax.axvline(NEAR_ZERO, color="#16161C", lw=1.0, ls=":", alpha=0.8)
    zs = int(su[ep < NEAR_ZERO].sum())
    ax.set_title(f"per-episode energy, {len(ep):,} episodes\n"
                 f"arm never moved: {100*zf:.1f}% of episodes\n"
                 f"{zs} of those succeeded", fontsize=8.4)
    ax.set_xlabel("episode energy integral (N²m²s)", fontsize=8.4)
    if c == 0:
        ax.set_ylabel("episodes", fontsize=8.6)
        ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax.grid(alpha=0.22, lw=0.6, zorder=0)

    # ---- row 3: reproduction against the earlier 20-seed sweep -------------
    ax = axes[3, c]
    o = np.array([old[task][a]["rate"] if a in old[task] else np.nan for a in arms])
    src = np.array([old[task].get(a, {}).get("source", "") for a in arms])
    ok = np.isfinite(o) & (src == "measured")
    lim = [min(o[ok].min(), suc[ok].min()) - 3, max(o[ok].max(), suc[ok].max()) + 3]
    ax.plot(lim, lim, color="#16161C", lw=1.0, ls="--", alpha=0.55, zorder=2)
    ax.scatter(o[ok], suc[ok], c=lat[ok], cmap="viridis", s=58, edgecolor="white",
               linewidth=0.7, zorder=3)
    pr = pearsonr(o[ok], suc[ok])
    bias = float(np.mean(suc[ok] - o[ok]))
    ax.set_title(f"replication vs 20-seed sweep, {ok.sum()} arms\n"
                 f"r = {pr.statistic:+.3f}  (p = {pr.pvalue:.2g})\n"
                 f"offset {bias:+.1f} pts = {abs(bias)/FLOOR[task]:.1f} noise bands",
                 fontsize=8.4)
    ax.set_xlabel("earlier sweep, 20 seeds (%)", fontsize=8.4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
    if c == 0:
        ax.set_ylabel("this sweep, seeds so far (%)", fontsize=8.6)
    ax.grid(alpha=0.22, lw=0.6, zorder=0)

cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.013, pad=0.035)
cb.set_label("observation age / latency of the arm (ms)", fontsize=8.2)
cb.ax.tick_params(labelsize=7)

tot = sum(v["n_seeds"] for t in T.values() for v in t.values())
fig.suptitle(f"Plane sweep sanity check — {tot}/1760 cells, all 176 (task, arm) pairs populated",
             fontsize=13.0, x=0.45, y=0.965)
fig.text(0.45, 0.062,
         f"PLANE GEOMETRY: across the 44 operating points Spearman(latency, period) = {GEOM:+.2f}. "
         "They are a scheduler Pareto front, so latency and cadence cannot be varied independently here — "
         "period is plotted, latency is the colour.\n"
         "Rows 1–2 read together: on the widowx tasks the SAME direction (shorter period) raises success and "
         "lowers energy, so these two axes are not trading against each other.\n"
         "Row 3 is the guard on the energy channel itself — close_drawer is bimodal and its energy panel must not be "
         "read as \u2018flat\u2019; row 4 is the independent replication against the earlier 20-seed sweep.",
         ha="center", va="top", fontsize=8.4, style="italic", color="#555")
fig.savefig(OUT, dpi=145, bbox_inches="tight")
print("wrote", OUT)
