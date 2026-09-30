#!/usr/bin/env python3
"""HIL flight-envelope ablation with UNCERTAINTY: success(cruise_speed x command_rate).

The 3-panel envelope (hil_ablation_3panel.py) shows the success fraction per cell as a
colour, which hides that every cell is only n seeds of a Bernoulli trial. This figure
fetches the same measured grid and puts a proper interval on every number:

  * success fraction per cell = k successes / n flights
  * error bars = Wilson score 95% CI (the right interval for a small-n proportion; it
    never leaves [0,1] and is not degenerate at k=0 or k=n, unlike normal-approx / Wald).

Four panels:
  (a) the measured envelope grid (colour = success fraction, text = k/n) -- the raw map.
  (b) HERO: success vs command rate, one line per cruise speed, Wilson 95% CI bars.
  (c) marginal effect of cruise speed  (pool over rate)  -- Wilson 95% CI.
  (d) marginal effect of command rate  (pool over speed) -- Wilson 95% CI.

Data: results/codesign_feedback/hil_ablation.csv  (5 speeds x 4 rates x 6 seeds = 120 flights).
NOTE: this grid uses the fixed controller gain moment_scale=0.0055 (calibrated ~90 Hz); the
low-rate cells are therefore partly under-authority (a gain artifact), so read the rate axis
as "command rate at the deployed gain", not as a pure Nyquist effect. See dima-hil-gain-calibration.
"""
import argparse, csv, math, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

INK = "#22242a"

def wilson(k, n, z=1.96):
    """Wilson score interval for k/n. Returns (phat, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_REPO + "/results/codesign_feedback/hil_ablation.csv")
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/hil_ablation_errorbars")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    cell = defaultdict(lambda: [0, 0])                       # (speed,rate) -> [k, n]
    for r in rows:
        sp = round(float(r["cruise_speed"]), 3); hz = round(float(r["eff_cmd_hz"]))
        cell[(sp, hz)][1] += 1
        cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell})
    rates  = sorted({h for _, h in cell})
    ns, nr = len(speeds), len(rates)

    Z = np.full((nr, ns), np.nan); N = np.zeros((nr, ns), int); Kc = np.zeros((nr, ns), int)
    for (sp, hz), (k, n) in cell.items():
        i = rates.index(hz); j = speeds.index(sp)
        Z[i, j] = k / max(1, n); N[i, j] = n; Kc[i, j] = k

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10})
    # perceptual colour per cruise speed (fast = warm)
    spd_cmap = plt.cm.plasma
    spd_col = {s: spd_cmap(0.12 + 0.72 * k / max(1, ns - 1)) for k, s in enumerate(speeds)}

    fig = plt.figure(figsize=(10.2, 7.2))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.965, top=0.92, bottom=0.085, hspace=0.32, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    # ---- (a) measured envelope grid -------------------------------------------------
    cmap = plt.cm.RdYlGn; norm = Normalize(0, 1)
    for i in range(nr):
        for j in range(ns):
            v = Z[i, j]
            fc = cmap(norm(v)) if not np.isnan(v) else "#ffffff"
            axA.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=fc, ec="white", lw=1.4, zorder=1))
            if not np.isnan(v):
                axA.text(j, i, f"{Kc[i,j]}/{N[i,j]}", ha="center", va="center",
                         fontsize=9, weight="bold", color="white" if v < 0.5 else "#123", zorder=3)
    axA.set_xlim(-.5, ns - .5); axA.set_ylim(-.5, nr - .5)
    axA.set_xticks(range(ns)); axA.set_xticklabels([f"{s:g}" for s in speeds])
    axA.set_yticks(range(nr)); axA.set_yticklabels([f"{h:g}" for h in rates])
    axA.set_xlabel("cruise speed  (m/s)"); axA.set_ylabel("command rate  (Hz)")
    axA.set_title("(a)  measured envelope  ·  success = k / n", loc="left", weight="bold")
    axA.set_aspect("auto")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("success fraction", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ---- (b) HERO: success vs rate, speeds clustered per rate, Wilson CI -------------
    xpos = np.arange(nr)                                          # even categorical slots per rate
    for jj, s in enumerate(speeds):
        ys, los, his = [], [], []
        for i, h in enumerate(rates):
            p, lo, hi = wilson(Kc[i, jj], N[i, jj])
            ys.append(p); los.append(p - lo); his.append(hi - p)
        dx = (jj - (ns - 1) / 2) * 0.15                          # x-jitter so the speed dots separate
        axB.errorbar(xpos + dx, ys, yerr=[los, his], marker="o", ms=6, ls="none", capsize=3,
                     elinewidth=1.6, mew=1.2, color=spd_col[s], label=f"{s:g} m/s", alpha=0.95, zorder=3)
    for x in xpos[:-1]:                                           # faint separators between rate clusters
        axB.axvline(x + 0.5, color="#e6e3dd", lw=1.0, zorder=0)
    axB.set_xlabel("command rate  (Hz)"); axB.set_ylabel("gate-course success fraction")
    axB.set_ylim(-0.04, 1.08); axB.set_xticks(xpos); axB.set_xticklabels([f"{h:g}" for h in rates])
    axB.set_xlim(-0.5, nr - 0.5)
    axB.grid(axis="y", ls=":", lw=0.6, color="#c9c6c0", zorder=0)
    axB.set_title("(b)  success vs command rate  ·  Wilson 95% CI  ·  by speed", loc="left", weight="bold")
    axB.legend(title="cruise speed", fontsize=8, title_fontsize=8, frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(0.005, 0.99))

    # ---- (c) marginal over rate: success vs speed -----------------------------------
    def marg(axis):
        out = []
        keys = speeds if axis == "speed" else rates
        for key in keys:
            k = sum(Kc[i, j] for i in range(nr) for j in range(ns)
                    if (speeds[j] == key if axis == "speed" else rates[i] == key))
            n = sum(N[i, j] for i in range(nr) for j in range(ns)
                    if (speeds[j] == key if axis == "speed" else rates[i] == key))
            out.append((key, *wilson(k, n), k, n))
        return out

    mc = marg("speed")
    xs = [m[0] for m in mc]; ps = [m[1] for m in mc]
    lo = [m[1] - m[2] for m in mc]; hi = [m[3] - m[1] for m in mc]
    axC.errorbar(xs, ps, yerr=[lo, hi], marker="s", ms=7, lw=2.0, capsize=4, color="#1f5fb0", zorder=3)
    for m in mc:
        axC.annotate(f"{m[4]}/{m[5]}", (m[0], m[1]), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5, color="#1f5fb0")
    axC.set_xlabel("cruise speed  (m/s)"); axC.set_ylabel("success fraction  (pooled over rate)")
    axC.set_ylim(-0.04, 1.0); axC.set_xticks(speeds)
    axC.grid(axis="y", ls=":", lw=0.6, color="#c9c6c0", zorder=0)
    axC.set_title("(c)  marginal effect of speed  ·  Wilson 95% CI", loc="left", weight="bold")

    # ---- (d) marginal over speed: success vs rate -----------------------------------
    md = marg("rate")
    xr2 = [m[0] for m in md]; pr = [m[1] for m in md]
    lo2 = [m[1] - m[2] for m in md]; hi2 = [m[3] - m[1] for m in md]
    axD.errorbar(xr2, pr, yerr=[lo2, hi2], marker="D", ms=6.5, lw=2.0, capsize=4, color="#1f9e5a", zorder=3)
    for m in md:
        axD.annotate(f"{m[4]}/{m[5]}", (m[0], m[1]), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=7.5, color="#1f7a46")
    axD.set_xlabel("command rate  (Hz)"); axD.set_ylabel("success fraction  (pooled over speed)")
    axD.set_ylim(-0.04, 1.0); axD.set_xticks(rates)
    axD.grid(axis="y", ls=":", lw=0.6, color="#c9c6c0", zorder=0)
    axD.set_title("(d)  marginal effect of command rate  ·  Wilson 95% CI", loc="left", weight="bold")

    fig.suptitle("HIL gate-course flight envelope — success with 95% confidence intervals "
                 f"(n={sum(N.flat)} flights, 6 seeds/cell)", y=0.975, fontsize=11.5, weight="bold")
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)

    # ---- console summary (the fetched numbers) --------------------------------------
    print("\n== marginal by cruise speed (pooled over rate) ==")
    for m in mc: print(f"  {m[0]:>4g} m/s : {m[4]:>2}/{m[5]:<2}  p={m[1]:.2f}  95%CI[{m[2]:.2f},{m[3]:.2f}]")
    print("== marginal by command rate (pooled over speed) ==")
    for m in md: print(f"  {m[0]:>4g} Hz  : {m[4]:>2}/{m[5]:<2}  p={m[1]:.2f}  95%CI[{m[2]:.2f},{m[3]:.2f}]")

if __name__ == "__main__":
    main()
