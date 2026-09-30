#!/usr/bin/env python3
"""One combined flight-envelope panel: success vs control rate, colour = cruise speed, Wilson 95% CI.

Packs the whole 5-speed x 4-rate x 6-seed ablation into a single axes:
  * each (speed, rate) cell = a scatter point at its measured success fraction, coloured by
    cruise speed, with a Wilson score 95% CI error bar (right interval for a 6-trial proportion);
  * faint per-speed lines trace each speed's rate response;
  * the POOLED rate marginal (all speeds, n=30/rate — the statistically strong trend) is overlaid
    as a bold line with a shaded 95% CI band;
  * x ticks flag the two showdown rates (50 Hz = ROS, 100 Hz = XPU-RT).

Importable: draw_envelope(ax, csv, ...) renders the panel into any axes (reused by the showdown
figure). Run directly to render the standalone figure.

Data: results/codesign_feedback/hil_ablation.csv. NOTE: fixed controller gain (moment_scale=0.0055,
tuned ~90 Hz), so the 25 Hz collapse is partly under-authority (a gain artifact), not pure Nyquist.
"""
import argparse, csv, math, os, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flight_quarantine import flight_rows        # noqa: E402  drops simulator-fault batches

C_XPU = "#1f9e5a"; C_ROS = "#e2231a"; INK = "#22242a"; C_CAL = "#1a5fb4"
DEFAULT_CSV = _REPO + "/results/codesign_feedback/hil_ablation.csv"
CAL_CSV = _REPO + "/results/codesign_feedback/gain_controlled/gain_controlled.csv"   # same grid, gain = 0.5/eff_hz


def fisher_two_sided(k1, n1, k2, n2):
    """Exact two-sided Fisher p for successes k1/n1 vs k2/n2 (sum of tables at most as likely)."""
    from math import comb
    K = k1 + k2; N = n1 + n2
    def pr(a): return comb(n1, a) * comb(n2, K - a) / comb(N, K)
    p_obs = pr(k1)
    return min(1.0, sum(pr(a) for a in range(max(0, K - n2), min(n1, K) + 1) if pr(a) <= p_obs * (1 + 1e-9)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)


def _grid(csv_path):
    # through flight_rows, not a bare DictReader: the sidecar's k/n (showdown_paper_figure.
    # envelope_counts) applies the simulator-fault quarantine, and a panel that draws a different
    # population from the one its sidecar records cannot be checked against it. The two agree today
    # -- neither grid has a quarantined flight -- and this is what keeps that true.
    rows = list(flight_rows(csv_path))
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        sp = round(float(r["cruise_speed"]), 3); hz = round(float(r["eff_cmd_hz"]))
        cell[(sp, hz)][1] += 1
        cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell})
    rates = sorted({h for _, h in cell})
    return cell, speeds, rates


def draw_envelope(ax, csv_path=DEFAULT_CSV, compact=False, colorbar_ax=None, highlight_hz=None, title=True,
                  cal_csv=CAL_CSV):
    """Render the combined envelope panel into `ax`. Returns the ScalarMappable (for a colorbar).

    Two gain policies when both grids exist: the fixed-gain grid (one controller, as deployed, at
    whatever cadence it is given) in black, and the calibrated-gain grid (moment_scale = 0.5/eff_hz,
    each cell with the authority its cadence calls for) in blue. The floor annotation's p-value is
    computed here from the fixed-gain cells, not typed in."""
    cell, speeds, rates = _grid(csv_path)
    ns, nr = len(speeds), len(rates)
    xpos = np.arange(nr)                                          # even categorical slot per rate
    cmap = plt.cm.plasma; norm = Normalize(speeds[0] - 0.05, speeds[-1] + 0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fs = 1.65 if compact else 1.0                                # font scale; compact embed is downscaled ~4x in the composite, so author BIG
    # --- faint per-speed rate-response lines + per-cell scatter with Wilson CI ------------
    for jj, s in enumerate(speeds):
        col = cmap(norm(s)); dx = (jj - (ns - 1) / 2) * 0.15
        ys, los, his = [], [], []
        for i, h in enumerate(rates):
            k, n = cell[(s, h)]
            p, lo, hi = wilson(k, n)
            ys.append(p); los.append(p - lo); his.append(hi - p)
        ax.errorbar(xpos + dx, ys, yerr=[los, his], marker="o", ms=7 * fs, ls="none",
                    capsize=2.5, elinewidth=1.2, mew=1.1 * fs, mec="white",
                    color=col, ecolor=col, alpha=0.72, zorder=4)
    # --- pooled rate marginal (all speeds) with 95% CI band -------------------------------
    pc, plo, phi = [], [], []
    for h in rates:
        k = sum(cell[(s, h)][0] for s in speeds); n = sum(cell[(s, h)][1] for s in speeds)
        p, lo, hi = wilson(k, n); pc.append(p); plo.append(lo); phi.append(hi)
    ax.fill_between(xpos, plo, phi, color=INK, alpha=0.10, zorder=3, lw=0)
    ax.plot(xpos, pc, "-", color=INK, lw=2.6, zorder=5)
    ax.plot(xpos, pc, "D", color=INK, ms=9 * fs, mfc="white", mew=2.2, zorder=6)
    for i, h in enumerate(rates):                                # k/n on the pooled point
        k = sum(cell[(s, h)][0] for s in speeds); n = sum(cell[(s, h)][1] for s in speeds)
        ax.annotate(f"{k}/{n}", (xpos[i], phi[i]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=7)

    # --- the calibrated-gain grid, pooled, when it exists ------------------------------------
    cal_line = None
    if cal_csv and os.path.exists(cal_csv):
        ccell, cspeeds, crates = _grid(cal_csv)
        # drawn only once the calibrated grid covers every rate at three or more speeds; a partial
        # grid is not a curve
        if all(h in crates for h in rates) and len(cspeeds) >= 3:
            cp, clo, chi, cx = [], [], [], []
            for i, h in enumerate(rates):
                if h not in crates:
                    continue
                k = sum(ccell[(s, h)][0] for s in cspeeds if (s, h) in ccell); n = sum(ccell[(s, h)][1] for s in cspeeds if (s, h) in ccell)
                if n:
                    pp, lo, hi = wilson(k, n); cp.append(pp); clo.append(lo); chi.append(hi); cx.append(xpos[i])
                    ax.annotate(f"{k}/{n}", (xpos[i] + 0.12, pp), textcoords="offset points", xytext=(0, -13),
                                ha="center", fontsize=8.5 * fs, weight="bold", color=C_CAL, zorder=7)
            if cp:
                ax.fill_between(cx, clo, chi, color=C_CAL, alpha=0.08, zorder=3, lw=0)
                cal_line, = ax.plot(cx, cp, "-", color=C_CAL, lw=2.2, zorder=5, alpha=0.9)
                ax.plot(cx, cp, "s", color=C_CAL, ms=7 * fs, mfc="white", mew=2.0, zorder=6)
    # --- axes cosmetics + STORY framing (control-rate floor + speed envelope) ------------
    ax.set_ylim(-0.05, 1.17); ax.set_xlim(-0.45, nr - 0.55)
    # floor boundary: between the last rate below 50 Hz and 50 Hz, found in the rates drawn. A fixed index put it
    # between 25 and 33 Hz whenever the grid also sampled 20 Hz, marking 33 Hz feasible against its own data.
    i50_ = next((i for i, h in enumerate(rates) if h >= 45.0), nr - 1)
    xb = i50_ - 0.5
    ax.axvspan(-0.45, xb, color=C_ROS, alpha=0.06, zorder=0)     # under-rate failure zone
    ax.axvspan(xb, nr - 0.55, color=C_XPU, alpha=0.05, zorder=0)  # feasible band
    ax.axvline(xb, color="#b03018", lw=1.5, ls=(0, (5, 3)), alpha=0.6, zorder=1)
    z1 = "under-rate → crash" if not compact else "under-rate"
    z2 = "feasible — speed-limited" if not compact else "feasible"
    ax.text((-0.45 + xb) / 2, 1.11, z1, color="#b81e14", fontsize=11 * fs, weight="bold",
            ha="center", va="center", zorder=8)
    ax.text((xb + nr - 0.55) / 2, 1.11, z2, color="#137a3e", fontsize=11 * fs, weight="bold",
            ha="center", va="center", zorder=8)
    ax.axhline(1.05, color="#d6d3cd", lw=0.8, zorder=0)
    # the cliff across the floor, tested on the fixed-gain grid: the lowest rate sampled against
    # the first rate above the floor, pooled over speed. Both rates are drawn, so the contrast the
    # number refers to is never left implicit.
    k25 = sum(cell[(s, rates[0])][0] for s in speeds); n25 = sum(cell[(s, rates[0])][1] for s in speeds)
    i50 = next((i for i, h in enumerate(rates) if abs(h - 50) < 1.5), min(2, nr - 1))
    k50 = sum(cell[(s, rates[i50])][0] for s in speeds); n50 = sum(cell[(s, rates[i50])][1] for s in speeds)
    pval = fisher_two_sided(k25, n25, k50, n50); dpts = 100.0 * (k50 / max(1, n50) - k25 / max(1, n25))
    ptxt = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.annotate(f"control-rate floor\n{rates[0]:g}→{rates[i50]:g} Hz {dpts:+.0f} pts · {ptxt}", xy=(xb, 0.30), xytext=(0.52, 0.82),
                fontsize=9.5 * fs, weight="bold", color="#111", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.0),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.6, connectionstyle="arc3,rad=-0.2"))
    # XPU-RT operates inside the feasible band (full-size only; omitted in the tiny embed to stay legible)
    if not compact:
        ax.annotate("XPU-RT holds 100 Hz\n(above the floor)", xy=(nr - 1, 0.06), xytext=(nr - 1.5, 0.72),
                    fontsize=9 * fs, weight="bold", color=C_XPU, ha="center", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=C_XPU, lw=1.5))
    for x in xpos[:-1]:
        ax.axvline(x + 0.5, color="#ece9e3", lw=0.8, zorder=0.5)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{h:g}" for h in rates], fontsize=13 * fs)
    for t, h in zip(ax.get_xticklabels(), rates):     # mark XPU-RT's rate; no ROS/50 head-to-head (unsupported)
        if highlight_hz is not None and abs(h - highlight_hz) < 1.5:
            t.set_color(C_XPU); t.set_fontweight("bold")     # the operating point, passed in
    ax.set_xlabel("control rate (Hz)", fontsize=14 * fs, labelpad=6)
    ax.set_ylabel("success fraction" if compact else "gate-course success fraction", fontsize=14 * fs)
    ax.tick_params(labelsize=12 * fs)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if title and not compact:                                    # compact embed drops its own title — the caption covers it
        ntot = sum(n for (_k, n) in (cell[c] for c in cell))     # total flights, counted from the cells
        nper = sum(cell[(s, rates[0])][1] for s in speeds)       # pooled n per rate
        gain_txt = ("black = fixed gain 0.0055 (one controller, as deployed) · blue = gain calibrated per rate (0.5/eff_hz)"
                    if cal_line is not None else "fixed gain (moment_scale 0.0055, calibrated at 50 Hz)")
        ax.set_title("Flight envelope — success rises with control rate, then speed sets the limit\n"
                     f"{ntot} flights · colour = cruise speed · Wilson 95% CI · pooled over speed (n={nper}/rate) · {gain_txt}",
                     fontsize=12.5 * fs, weight="bold", loc="left")
    if cal_line is not None and not compact:                  # the composite's caption names the two policies
        ax.legend(handles=[Line2D([0], [0], color=INK, lw=2.6, marker="D", mfc="white", label="fixed gain (as deployed)"),
                           Line2D([0], [0], color=C_CAL, lw=2.2, marker="s", mfc="white", label="gain calibrated per rate")],
                  loc="lower right", fontsize=9 * fs, frameon=False)
    # label the pooled trend inline near its first point (full-size only)
    if not compact:
        ax.annotate("pooled", (xpos[0], pc[0]), textcoords="offset points", xytext=(9, 11),
                    fontsize=9 * fs, weight="bold", color=INK, va="center", ha="left")

    if colorbar_ax is not None:
        cb = colorbar_ax.figure.colorbar(sm, cax=colorbar_ax)
        cb.set_label("cruise speed (m/s)", fontsize=12 * fs)
        cb.set_ticks(speeds); cb.ax.tick_params(labelsize=10 * fs)
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/hil_envelope_combined")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(8.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[26, 1], left=0.10, right=0.90, top=0.88, bottom=0.135, wspace=0.04)
    ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    draw_envelope(ax, a.csv, colorbar_ax=cax)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)


if __name__ == "__main__":
    main()
