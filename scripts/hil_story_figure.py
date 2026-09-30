#!/usr/bin/env python3
"""One figure that tells the whole HIL story, three panels left-to-right:

  (a) FLIGHT ENVELOPE  — the full speed x rate ablation (course A, 240 flights): a control-rate floor
      gates success, above it cruise speed sets the limit. Reuses hil_envelope_panel.draw_envelope.
  (b) GENERALIZATION   — the same pooled rate-response on a DIFFERENT gate course (B, 120 flights),
      overlaid on course A. The floor + rise reproduces on gates the policy never saw (stack unchanged).
  (c) THE MECHANISM    — WHY below the floor fails: the starved baseline thrashes. Measured mean
      commanded body-moment and modeled propulsive power, as ratios to XPU-RT (log scale). Honestly
      split into MEASURED (moment, straight from the logged wrench) and MODELED (power, mixer +
      momentum theory) so the two claims are never conflated.

Sources: results/codesign_feedback/{hil_ablation.csv, hil_ablation_courseB.csv, flight_energy.csv}.
Output:  results/codesign_feedback/hil_envelope_story.{png,pdf}  (a NEW file — does not overwrite the
         committed hil_envelope_combined used in the paper).
"""
import glob
import re
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hil_envelope_panel import draw_envelope, wilson, fisher_two_sided, C_XPU, C_ROS, INK   # reuse the exact envelope panel
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

RES = _REPO + "/results/codesign_feedback"
C_A, C_B = INK, "#7b3fa0"                                     # course A = black, course B = violet


def pooled_rate_response(csv_path):
    """pooled-over-speed k,n per control rate -> sorted rates, and (p, lo, hi) each."""
    cell = defaultdict(lambda: [0, 0])
    for r in flight_rows(csv_path):
        hz = round(float(r["eff_cmd_hz"])); cell[hz][1] += 1; cell[hz][0] += int(r["outcome"] == "success")
    rates = sorted(cell)
    stats = [(cell[h][0], cell[h][1], *wilson(*cell[h])) for h in rates]   # k,n,p,lo,hi
    return rates, stats


# ------------------------------- panel (b): generalization -------------------------------------
def draw_generalization(ax, fs=1.0, compact=False):
    rA, sA = pooled_rate_response(f"{RES}/hil_ablation.csv")
    rB, sB = pooled_rate_response(f"{RES}/hil_ablation_courseB.csv")
    # the two courses need not sample the same rates (e.g. course A gains a 20 Hz cell); the
    # categorical x-axis is the union of both, and each course is placed at its own rates' indices
    allrates = sorted(set(rA) | set(rB))
    xmap = {h: i for i, h in enumerate(allrates)}
    x = np.arange(len(allrates))
    xb = next((i - 0.5 for i, h in enumerate(allrates) if h >= 45), len(allrates) - 0.55)   # floor boundary (~between 33 and 50 Hz)
    ax.axvspan(-0.45, xb, color=C_ROS, alpha=0.06, zorder=0)
    ax.axvspan(xb, len(allrates) - 0.55, color=C_XPU, alpha=0.05, zorder=0)
    ax.axvline(xb, color="#b03018", lw=1.4, ls=(0, (5, 3)), alpha=0.6, zorder=1)
    labA = "course A" if compact else "course A (figure)"
    labB = "course B" if compact else "course B (unseen gates)"
    for (r, s, col, lab, dx) in [(rA, sA, C_A, labA, -0.04), (rB, sB, C_B, labB, 0.04)]:
        xp = np.array([xmap[h] for h in r], dtype=float)
        p = np.array([t[2] for t in s]); lo = np.array([t[3] for t in s]); hi = np.array([t[4] for t in s])
        ax.fill_between(xp, lo, hi, color=col, alpha=0.12, zorder=2, lw=0)
        ax.plot(xp + dx, p, "-o", color=col, lw=2.4 * fs, ms=6.5 * fs, mfc="white", mew=1.8, zorder=5, label=lab)
        if not compact:                                       # per-point k/n clutters the small embed
            for i, t in enumerate(s):
                ax.annotate(f"{t[0]}/{t[1]}", (xp[i] + dx, hi[i]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8.2 * fs, weight="bold", color=col, zorder=7)
    # the 25 -> 50 Hz contrast, pooled over speed, tested here (two-sided Fisher exact)
    def _contrast(rates, stats):
        i25 = rates.index(min(rates, key=lambda h: abs(h - 25))); i50 = rates.index(min(rates, key=lambda h: abs(h - 50)))
        kA, nA = stats[i25][0], stats[i25][1]; kB, nB = stats[i50][0], stats[i50][1]
        return 100.0 * (kB / nB - kA / nA), fisher_two_sided(kA, nA, kB, nB)
    dA, pA = _contrast(rA, sA); dB, pB = _contrast(rB, sB)
    _p = lambda v: "p<0.001" if v < 0.001 else f"p={v:.3f}"
    box = (f"floor holds on\nunseen gates\n25→50 Hz\nA {dA:+.0f} / B {dB:+.0f} pts" if compact else
           f"floor reproduces on\nunseen gates\nA {dA:+.0f} pts ({_p(pA)})\nB {dB:+.0f} pts ({_p(pB)})")
    ytop = 0.66 if compact else 1.20                          # compact: tighten range so the rise fills the panel
    ax.annotate(box, xy=(xb, 0.34), xytext=(0.30, 0.55 if compact else 0.87), fontsize=9 * fs, weight="bold",
                color="#111", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.0),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.5, connectionstyle="arc3,rad=-0.2"))
    ax.set_ylim(-0.03, ytop); ax.set_xlim(-0.45, len(allrates) - 0.55)
    ax.set_xticks(x); ax.set_xticklabels([f"{h:g}" for h in allrates], fontsize=12 * fs)
    ax.set_xlabel("control rate (Hz)", fontsize=13 * fs, labelpad=5)
    ax.set_ylabel("success fraction" if compact else "success fraction (pooled over speed)", fontsize=12 * fs)
    ax.tick_params(labelsize=11 * fs)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    ax.legend(loc="upper right" if compact else "upper left", fontsize=9.0 * fs, frameon=False,
              bbox_to_anchor=None if compact else (0.0, 1.0), handlelength=1.4, borderaxespad=0.3)
    ax.set_title("Generalization: unseen gates" if compact else "Generalization: same floor, unseen gates",
                 fontsize=11.5 * fs, weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------- panel (c): the mechanism (energy) ------------------------------
ENERGY_CSV = os.environ.get("ENERGY_CSV", f"{RES}/flight_energy.csv")          # flight_energy_v2.csv for the replayed-cadence arms
ARM_NAMES = {"xpu_cpsat": "XPU-RT\nCP-SAT", "xpu_greedy": "XPU-RT\ngreedy", "ros_static6": "ROS 2\nstatic 6 cores",
             "ros_vanilla": "ROS 2\nvanilla", "ros_shipped": "ROS 2\nas shipped"}

# The solver-placed arm is named for the camera rate it was solved at -- xpu_p30free, xpu_p36free,
# xpu_p45free -- so it is one rule rather than one ARM_NAMES entry per rate. Without it a new rate's
# arm falls to the end of the order and a ladder that also holds xpu_greedy would normalise panel D
# to greedy, which is not the arm the figure draws as 1x.
_SOLVER_ARM = re.compile(r"^xpu_p\d+free$")


def canon_arm(cond):
    """the ARM_NAMES key a flight-energy condition is listed under."""
    return "xpu_cpsat" if _SOLVER_ARM.match(cond) else cond


def _cond_rate_hz(cond):
    """The effective command rate a condition flew at, from its first dump (a replayed trace
    carries eff_cmd_hz); None when the dump does not say."""
    for d in sorted(glob.glob(os.path.join(os.path.dirname(ENERGY_CSV), "energy_runs*", f"{cond}_s*", "figure_data.npz"))):
        try:
            z = np.load(d, allow_pickle=True)
            if "eff_cmd_hz" in z:
                return float(z["eff_cmd_hz"])
        except Exception:
            pass
    return None


def draw_mechanism(ax, fs=1.0, compact=False):
    rows = list(csv.DictReader(open(ENERGY_CSV)))
    def agg(pref, key):
        g = [r for r in rows if r["flight"].startswith(pref + "_s")]
        if key == "power":  vals = [float(r["energy"]) / float(r["dur_s"]) for r in g]   # duration-fair
        else:               vals = [float(r["mean_absM"]) for r in g]
        return sum(vals) / len(vals)
    # conditions are whatever the energy runs recorded (scripts/run_energy_experiment.sh names them
    # by runtime and measured control rate: xpu100, ros33, ros17, ...); XPU-RT first, then by rate
    import re as _re
    names = sorted({r["flight"].rsplit("_s", 1)[0] for r in rows})
    def _hz(n):
        r = _cond_rate_hz(n)
        if r is not None:
            return r
        m = _re.search(r"(\d+)$", n); return int(m.group(1)) if m else 0
    order = list(ARM_NAMES)
    names.sort(key=lambda n: (order.index(canon_arm(n)) if canon_arm(n) in order else 99,
                              0 if n.startswith("xpu") else 1, -_hz(n)))
    shades = [C_ROS, "#9a1610", "#5c0b08"]
    def _lab(n):
        base = ARM_NAMES.get(canon_arm(n), "XPU-RT" if n.startswith("xpu") else "ROS 2")
        hz = _hz(n)
        return f"{base}\n{hz:.0f} Hz" if hz else base
    conds = [(n, _lab(n), C_XPU if n.startswith("xpu") else shades[min(i, 2)]) for i, n in enumerate([m for m in names if not m.startswith("xpu")])]
    conds = [(n, _lab(n), C_XPU if i == 0 else "#7fb069") for i, n in enumerate([m for m in names if m.startswith("xpu")])] + conds
    nseed = min(sum(1 for r in rows if r["flight"].startswith(c + "_s")) for c, _, _ in conds)
    momR = {c: agg(c, "mom") for c, _, _ in conds}; pwR = {c: agg(c, "power") for c, _, _ in conds}
    xk = next(c for c, _, _ in conds if c.startswith("xpu")); base_m, base_p = momR[xk], pwR[xk]
    x = np.arange(len(conds)); w = 0.36
    for i, (c, lab, col) in enumerate(conds):
        rm, rp = momR[c] / base_m, pwR[c] / base_p
        ax.bar(x[i] - w / 2, rm, w, color=col, edgecolor=INK, lw=0.8, zorder=3)                       # measured
        ax.bar(x[i] + w / 2, rp, w, color=col, edgecolor=INK, lw=0.8, hatch="////", zorder=3, alpha=0.75)  # modeled
        ax.annotate(f"{rm:.0f}×", (x[i] - w / 2, rm), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=5)
        ax.annotate(f"{rp:.0f}×", (x[i] + w / 2, rp), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5 * fs, weight="bold", color=INK, zorder=5)
    ax.set_yscale("log"); ax.set_ylim(0.6, 900 if compact else 230)   # compact: headroom so the legend clears the bar labels
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    # the reference line needs no label of its own: the y-axis names what the ratio is against and the
    # first pair is drawn at 1x. Labelled at the right edge it landed on top of the tallest bar.
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in conds], fontsize=11 * fs)
    ax.set_ylabel("rel. to XPU-RT (log)" if compact else "relative to XPU-RT (log scale)", fontsize=12 * fs)
    ax.tick_params(labelsize=10 * fs)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb", zorder=0)
    leg = [Patch(fc="#888", ec=INK, label="moment (measured)" if compact else "mean commanded moment (measured)"),
           Patch(fc="#888", ec=INK, hatch="////", alpha=0.75, label="power (modeled)" if compact else "propulsive power (modeled)")]
    ax.legend(handles=leg, loc="upper left", fontsize=8.6 * fs, frameon=False)
    ax.set_title(f"Mechanism: the baseline thrashes (n={nseed} seeds/arm)", fontsize=11.5 * fs, weight="bold", loc="left")
    if not compact:
        ax.text(0.5, -0.185, "duration-fair (per-second); ratios robust to rotor constants",
                transform=ax.transAxes, ha="center", fontsize=8, color="#666", style="italic")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RES}/hil_envelope_story")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(17.6, 6.6))
    # envelope gets the most room + its colorbar; generalization and mechanism share the right third
    gs = fig.add_gridspec(1, 4, width_ratios=[26, 1.1, 20, 15], left=0.045, right=0.985,
                          top=0.85, bottom=0.17, wspace=0.32)
    ax_env = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    ax_gen = fig.add_subplot(gs[2]); ax_mech = fig.add_subplot(gs[3])
    draw_envelope(ax_env, colorbar_ax=cax, title=False)          # concise title instead of the long 2-line default
    ax_env.set_title("Flight envelope: control-rate floor, then speed-limited",
                     fontsize=11.5, weight="bold", loc="left")
    draw_generalization(ax_gen)
    draw_mechanism(ax_mech)
    for ax, lab in [(ax_env, "a"), (ax_gen, "b"), (ax_mech, "c")]:
        ax.text(-0.02, 1.16, f"({lab})", transform=ax.transAxes, fontsize=15, weight="bold",
                va="top", ha="right", color=INK)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)


if __name__ == "__main__":
    main()
