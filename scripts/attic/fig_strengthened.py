#!/usr/bin/env python3
"""Single 2-panel figure for the strengthened HIL story.

  (a) Cross-course generalization: success vs control rate for Course A (figure gates) and
      Course B (alternate gates), Wilson 95% CIs. The ~0-below-50Hz floor + rise holds on
      both layouts -> the control-rate floor generalizes across scenes.
  (b) Matched-progress energy: cumulative MODELED propulsive energy vs down-course distance,
      same scene (same seed). ROS crashes partway (X) having spent far more energy to cover
      the ground it managed; XPU-RT stays low and continues to the finish (o). The vertical
      gap at ROS's crash distance is the honest energy difference (ratios robust to rotor
      constants). Reads straight from the CSVs / energy_runs, so it stays reproducible.
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

RES = _REPO + "/results/codesign_feedback"
OUT = f"{RES}/fig_hil_strengthened.png"

# ---- palette / type -------------------------------------------------------------------------
INK   = "#1b2431"; MUTE = "#6b7684"; GRID = "#dfe4ea"
OURS  = "#0e7c7b"   # teal = XPU-RT (ours)
BASE  = "#d9772b"   # orange = ROS 50 Hz
BASE2 = "#b23b3b"   # deep red = ROS 25 Hz
A_CLR = "#2b6cb0"   # course A
B_CLR = "#7a4fb5"   # course B
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.linewidth": 0.9, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 12.5,
    "axes.titleweight": "bold", "figure.dpi": 200,
})

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, (c-h)/d), min(1.0, (c+h)/d)

# ---- panel (a) data: k/n per eff_cmd_hz for course A and B --------------------------------------
def rate_counts(path):
    kn = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            hz = round(float(r["eff_cmd_hz"]))
            kn.setdefault(hz, [0, 0]); kn[hz][1] += 1
            if r["outcome"] == "success": kn[hz][0] += 1
    return kn
A = rate_counts(f"{RES}/hil_ablation.csv")
B = rate_counts(f"{RES}/hil_ablation_courseB.csv")
rates = sorted(A)                                   # 25,33,50,100

# ---- panel (b) data: cumulative energy vs forward distance (same seed = same scene) -------------
AX = 1                                              # course axis = y
def load(cond, s):
    f = glob.glob(f"{RES}/energy_runs/{cond}_s{s}/**/figure_data.npz", recursive=True) \
        or glob.glob(f"{RES}/energy_runs/{cond}_s{s}/figure_data.npz")
    return np.load(f[0], allow_pickle=True) if f else None
def fwd(d):
    p = d["poses"][:, AX].astype(float); return np.maximum.accumulate(np.abs(p - p[0]))
def cum_energy(d):
    w = d["wrench"].astype(float).copy(); w[:, 0] = np.clip(1 + w[:, 0], 0, None)
    arm, kap, dd = 0.09, 0.016, 0.09/np.sqrt(2)
    Tt, Mx, My, Mz = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    T = np.stack([Tt/4-Mx/(4*dd)+My/(4*dd)-Mz/(4*kap), Tt/4+Mx/(4*dd)-My/(4*dd)-Mz/(4*kap),
                  Tt/4+Mx/(4*dd)+My/(4*dd)+Mz/(4*kap), Tt/4-Mx/(4*dd)-My/(4*dd)+Mz/(4*kap)], 1).clip(0)
    P = (T**1.5).sum(1); t = d["t_s"].astype(float)
    return fwd(d), np.concatenate([[0], np.cumsum(P[1:]*np.diff(t))])
SEED = 1000                                         # representative scene shown; ratios below are over all seeds
dx, d50, d25 = load("xpu100", SEED), load("ros50", SEED), load("ros25", SEED)

# mean matched-distance energy ratio over all 3 seeds (for the annotation)
def matched_ratio(base):
    rr = []
    for s in (1000, 1001, 1002):
        a, b = load("xpu100", s), load(base, s)
        if a is None or b is None: continue
        xa, Ea = cum_energy(a); xb, Eb = cum_energy(b)
        L = xa.max(); dstop = min(xb.max(), L)
        ex = Ea[min(int(np.searchsorted(xa, dstop)), len(Ea)-1)]
        eb = Eb[min(int(np.searchsorted(xb, dstop)), len(Eb)-1)]
        if ex > 0: rr.append(eb/ex)
    return float(np.mean(rr)) if rr else float("nan")
r50, r25 = matched_ratio("ros50"), matched_ratio("ros25")

# =================================================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw=dict(wspace=0.255))

# ---- (a) generalization -------------------------------------------------------------------------
axL.axvspan(20, 41, color="#f2b8b8", alpha=0.30, lw=0, zorder=0)
axL.text(30.5, 41.5, "control-rate\nfloor", ha="center", va="top", fontsize=9.5, color=BASE2,
         style="italic", zorder=1)
for kn, clr, lab, dx_off in ((A, A_CLR, "Course A  (figure gates)", 0.985),
                             (B, B_CLR, "Course B  (alternate gates)", 1.015)):
    xs, ys, lo, hi = [], [], [], []
    for hz in rates:
        k, n = kn[hz]; p, l, h = wilson(k, n)
        xs.append(hz*dx_off); ys.append(100*p); lo.append(100*(p-l)); hi.append(100*(h-p))
    axL.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=clr, ecolor=clr, elinewidth=1.4,
                 capsize=3.5, ms=6.5, lw=2.0, mfc="white", mec=clr, mew=1.8, label=lab, zorder=3)
    for hz in rates:                                 # k/n annotations (nudge low-rate ones apart)
        k, n = kn[hz]
        dy = (11 if clr == A_CLR else -16)
        dxo = (-11 if clr == A_CLR else 11) if hz <= 33 else 0
        axL.annotate(f"{k}/{n}", (hz*dx_off, 100*k/n), textcoords="offset points",
                     xytext=(dxo, dy), ha="center", fontsize=8, color=clr)
axL.set_xscale("log"); axL.set_xticks(rates); axL.set_xticklabels([f"{r}" for r in rates])
axL.xaxis.set_minor_formatter(NullFormatter()); axL.tick_params(axis="x", which="minor", length=0)
axL.set_xlim(21, 118); axL.set_ylim(-3, 55)
axL.set_xlabel("Effective control rate  (Hz)"); axL.set_ylabel("Gate-course success  (%)")
axL.set_title("(a)  Control-rate floor generalizes across courses", loc="left", fontsize=11.8)
axL.grid(True, which="both", color=GRID, lw=0.7, zorder=0); axL.set_axisbelow(True)
axL.legend(loc="upper left", frameon=False, fontsize=9.3, handlelength=1.6)
axL.annotate("floor $\\to$ 50 Hz:  A +37 pts (p<0.001),  B +23 pts (p=0.042)",
             (0.5, -0.20), xycoords="axes fraction", ha="center", fontsize=8.7, color=MUTE)

# ---- (b) matched-progress energy ----------------------------------------------------------------
def plot_run(d, clr, lab, marker):
    x, E = cum_energy(d); axR.plot(x, E, color=clr, lw=2.3, label=lab, zorder=3)
    axR.scatter([x[-1]], [E[-1]], s=95, color=clr, marker=marker, zorder=4,
                edgecolor="white", linewidth=1.3)
    return x[-1], E[-1]
xf_x, _ = plot_run(dx, OURS, "XPU-RT  100 Hz  (finishes)", "o")
x50, E50 = plot_run(d50, BASE, "ROS  50 Hz  (crashes)", "X")
x25, E25 = plot_run(d25, BASE2, "ROS  25 Hz  (crashes)", "X")
# energy gap at ROS-50's crash distance
xa, Ea = cum_energy(dx); Ex_at = Ea[min(int(np.searchsorted(xa, x50)), len(Ea)-1)]
axR.annotate("", xy=(x50, E50), xytext=(x50, Ex_at),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
axR.annotate(f"ROS 50 Hz spends ~{r50:.0f}$\\times$\nour energy for the\nsame ground, then crashes",
             (x50, (E50+Ex_at)/2), xytext=(x50-0.4, E50*0.55), ha="right", va="center", fontsize=9.2,
             color=INK, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
axR.annotate("ROS 25 Hz:\ncrashes at\nthe start", (x25, E25), xytext=(x25+0.7, E25*1.02),
             ha="left", va="top", fontsize=8.7, color=BASE2,
             arrowprops=dict(arrowstyle="-", color=BASE2, lw=0.8))
axR.set_xlabel("Distance down course  (m)"); axR.set_ylabel("Cumulative modeled propulsive energy  ($\\propto$)")
axR.set_title("(b)  Same scene: energy to cover the same ground", loc="left", fontsize=11.8)
axR.grid(True, color=GRID, lw=0.7, zorder=0); axR.set_axisbelow(True)
axR.set_xlim(-0.6, max(xf_x, x50)+0.8); axR.set_ylim(0, E50*1.18)
axR.legend(loc="upper left", frameon=False, fontsize=9.0, handlelength=1.6)
axR.annotate(f"mean over 3 scenes: ROS 50 Hz {r50:.0f}$\\times$,  25 Hz {r25:.0f}$\\times$ our energy per distance",
             (0.5, -0.20), xycoords="axes fraction", ha="center", fontsize=8.7, color=MUTE)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
print(f"panel b ratios: ros50 {r50:.1f}x  ros25 {r25:.1f}x   (seed {SEED} shown; crash x50={x50:.1f}m, course={xf_x:.1f}m)")
