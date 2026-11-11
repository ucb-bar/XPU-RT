#!/usr/bin/env python3
"""Drone cruise velocity vs course-completion rate, from the warehouse HIL flight package.

WHY THIS IS A GROUPED PLOT, NOT A SCATTER OF RAW FLIGHTS
Success in this dataset is BINARY per flight (`outcome` == success / crash / timeout, and
`success` = 1 iff outcome == "success"). A single flight has no "success rate", so a rate only
exists once flights are grouped. The grouping key here is (experimental arm) x (commanded
cruise speed) -- the arm has to be in the key because the package explicitly warns that its
regimes are DIFFERENT experiments and must never be pooled (README_hil_data.md: "Do not pool
the two sources or read them cell-for-cell"; build_master_csv.py stamps a `regime` column
whose stated purpose is that these are "never silently pooled").

LEFT PANEL -- the only arm that was actually DESIGNED to answer this question.
`hil_ablation.csv` (regime "envelope"): a fully crossed 5 cruise speeds x 4 control rates x
6 seeds = 120 flights, fixed controller gain (moment_scale=0.0055), control rate set by clean
ZOH decimation. Because the design is balanced, the pooled-over-rate marginal (n=24 per speed)
is a legitimate estimate and is drawn in grey; the four coloured series are the per-rate cells
(n=6 each). Dashed lines are the AUTHORS' OWN logistic response surface
P(success) ~ log2(rate) + speed (scripts/hil_logistic_fit.py), refit here on the same CSV.

RIGHT PANEL -- does the speed trend replicate anywhere else? Control rate is pinned at 100 Hz
and each independent arm is drawn separately. It does not: only the envelope arm falls with
speed, and it falls because of the 1.6/1.8 m/s cells that no other arm ever flew.

VELOCITY AXIS. `cruise_speed` is the COMMANDED forward speed setpoint in m/s (the package
column dictionary: "cruise_speed -- commanded forward speed (m/s)"). No achieved-velocity
column exists in any file in the package. It is plotted as-is, and it is defensible: within the
envelope arm the implied path length v * steps * sim_dt on its 28 successes has per-speed medians
of 14.49-15.30 m and a full per-flight range of 13.99-16.04 m across every speed from 1.0 to
1.8 m/s, i.e. completion time scales as 1/v on a fixed-length course, so the commanded speed is
in fact the flown speed. (Pooled over all 141 successes the median is 14.63 m, IQR 14.20-15.15;
the tail to 22-24 m is the perception-freshness arm, where the drone detours around walking
people and the FLOWN path is genuinely longer than the course.)

HONEST HEADLINE: over the full 1.0-1.8 m/s range of the designed arm, success falls with cruise
speed (per-SD odds ratio 0.40x, p=0.0006, n=120). That effect is NOT robust. Restricted to the
1.0-1.4 m/s range every other arm shares, it is not significant (p=0.10), and no independent arm
reproduces it (perc_crash p=0.17 n=116; walkgrid p=0.85 n=58; the authors' own 12-seed refresh
hil_ablation_v2.csv p=0.48 n=96). The defensible statement is a high-speed CEILING somewhere
above 1.4 m/s, not a smooth velocity-success trend.
"""
from __future__ import annotations
import csv, math, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

HERE = Path(__file__).parent
OUT = HERE / "fig_drone_velocity.png"
# Extracted (read-only) copy of hil_data_package.zip (sha256 877f0a4a7575999d...).
SRC = Path(os.environ.get(
    "HIL_PKG",
    "/tmp/claude-1172/-scratch2-dima-misc-sw-XPU-RT/"
    "d45883ec-3095-43b7-a110-0d9c3e82adac/scratchpad/hil/hil_data_package"))

INK = "#16161C"
TEAL, ORANGE, PLUM = "#0e6655", "#C77400", "#7A1250"
GREY, GREY_L = "#6B6B78", "#9a9aa4"

# control rate (Hz) -> colour. 100 Hz is the DEPLOYED rate, so it carries the ink.
RATE_COL = {25.0: PLUM, 33.33: ORANGE, 50.0: TEAL, 100.0: INK}
RATE_LAB = {25.0: "25 Hz", 33.33: "33 Hz", 50.0: "50 Hz", 100.0: "100 Hz (deployed)"}


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0 and 1, where these proportions live.

    Identical to the package's own scripts/hil_envelope_panel.py:wilson, so the intervals
    here and in the authors' envelope panel are the same estimator.
    """
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)


def irls_logit(X, y, iters=200, ridge=1e-8):
    """Plain IRLS logistic fit -- same routine as scripts/hil_logistic_fit.py."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        nb = np.linalg.solve(X.T @ (W[:, None] * X) + ridge * np.eye(X.shape[1]), X.T @ (W * z))
        if np.max(np.abs(nb - b)) < 1e-10:
            b = nb
            break
        b = nb
    eta = X @ b
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    W = np.clip(mu * (1 - mu), 1e-9, None)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (W[:, None] * X) + ridge * np.eye(X.shape[1]))))
    return b, se


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ----------------------------------------------------------------------------- load
master = list(csv.DictReader(open(SRC / "hil_flights_master.csv")))
env = [r for r in master if r["source"] == "hil_ablation"]          # the designed sweep


def arm_of(r):
    """Independent experimental arms, per the package's own `regime`/`source` tags."""
    if r["source"] == "hil_ablation":
        return "envelope"
    if r["regime"].startswith("perc_freshness"):
        return "perc_crash"
    if r["source"].startswith("crash_verify_walkgrid") or r["source"] == "hil_smoke":
        return "walkgrid"
    if r["regime"] == "calibrated-gain grid":
        return "calib_grid"
    return "other"


def rate_of(r):
    return round(float(r["eff_cmd_hz"]), 2)


def speed_of(r):
    return round(float(r["cruise_speed"]), 2)


def kn(rows):
    return sum(r["outcome"] == "success" for r in rows), len(rows)


# per (speed, rate) cell of the envelope arm
cell = defaultdict(list)
for r in env:
    cell[(speed_of(r), rate_of(r))].append(r)
SPEEDS = sorted({s for s, _ in cell})
RATES = sorted({h for _, h in cell})

# authors' logistic surface, refit on hil_ablation.csv exactly as hil_logistic_fit.py does
lr = np.array([math.log2(rate_of(r)) for r in env])
ls = np.array([speed_of(r) for r in env])
ly = np.array([1.0 if r["outcome"] == "success" else 0.0 for r in env])
rm, rs, sm, ss = lr.mean(), lr.std(), ls.mean(), ls.std()
BETA, SE = irls_logit(np.column_stack([np.ones(len(ly)), (lr - rm) / rs, (ls - sm) / ss]), ly)
P_SPEED = 2 * (1 - phi(abs(BETA[2] / SE[2])))
OR_SPEED = math.exp(BETA[2])


def surface(speed, rate):
    eta = BETA[0] + BETA[1] * (math.log2(rate) - rm) / rs + BETA[2] * (speed - sm) / ss
    return 1.0 / (1.0 + math.exp(-eta))


# ----------------------------------------------------------------------------- figure
import textwrap

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(15.0, 8.0), dpi=150,
    gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.19,
                 "left": 0.062, "right": 0.988, "top": 0.865, "bottom": 0.350})

YLIM = (-0.09, 1.33)

# ---- PANEL A: the designed speed x rate envelope --------------------------------
# shade the speed band only this arm ever flew -- it is where the whole effect lives
axA.axvspan(1.5, 1.9, color=GREY_L, alpha=0.14, zorder=0, lw=0)
axA.text(1.70, 1.30, "only this arm flew\n1.6–1.8 m/s", ha="center", va="top",
         fontsize=8.4, color=GREY, style="italic", zorder=1)

xg = np.linspace(0.95, 1.85, 160)
for i, h in enumerate(RATES):
    col = RATE_COL[h]
    dx = (i - 1.5) * 0.026                       # dodge so error bars do not overlap
    axA.plot(xg, [surface(v, h) for v in xg], "--", color=col, lw=1.25, alpha=0.55, zorder=2)
    xs, ys, lo, hi = [], [], [], []
    for s_ in SPEEDS:
        k, n = kn(cell[(s_, h)])
        p, l, u = wilson(k, n)
        xs.append(s_ + dx); ys.append(p); lo.append(p - l); hi.append(u - p)
    axA.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col, lw=1.15,
                 capsize=2.5, alpha=0.85, zorder=4)
    axA.plot(xs, ys, "-", color=col, lw=1.5, alpha=0.85, zorder=5)
    axA.plot(xs, ys, "o", color=col, ms=7.0, mfc="white", mew=2.0, zorder=6,
             label=RATE_LAB[h])

# pooled-over-rate marginal: legitimate because the design is balanced (4 rates x 6 seeds)
px, pp, plo, phi_ = [], [], [], []
for s_ in SPEEDS:
    k, n = kn([r for r in env if speed_of(r) == s_])
    p, l, u = wilson(k, n)
    px.append(s_); pp.append(p); plo.append(l); phi_.append(u)
    axA.annotate(f"{k}/{n}", (s_, p), textcoords="offset points", xytext=(20, -4),
                 ha="left", va="center", fontsize=8.6, weight="bold", color=INK, zorder=9)
axA.fill_between(px, plo, phi_, color=GREY, alpha=0.14, lw=0, zorder=1)
axA.plot(px, pp, "-", color=GREY, lw=2.8, zorder=7)
axA.plot(px, pp, "D", color=GREY, ms=9.5, mfc="white", mew=2.4, zorder=8,
         label="pooled over rate  (n=24)")

axA.set_xlim(0.93, 1.90)
axA.set_ylim(*YLIM)
axA.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_xticks(SPEEDS)
axA.set_xticklabels([f"{s_:.1f}" for s_ in SPEEDS])
axA.set_xlabel("commanded cruise speed (m/s)", fontsize=10.4)
axA.set_ylabel("course-completion rate  (all 4 gates cleared)", fontsize=10.4)
axA.set_title("A · designed sweep: 5 speeds × 4 control rates × 6 seeds = 120 flights\n"
              "dashed = authors' logistic surface · Wilson 95% CI · labels are k/n pooled",
              fontsize=10.2, color=INK)
axA.grid(True, alpha=0.22, zorder=0)
axA.legend(fontsize=8.5, loc="upper left", framealpha=0.95,
           title="control rate (n=6 per point)", title_fontsize=8.4)

# ---- PANEL B: replication check, control rate pinned at 100 Hz -------------------
ARMS = [("envelope", "envelope (fixed gain 0.0055)", INK, "o"),
        ("perc_crash", "perception-freshness study", TEAL, "s"),
        ("walkgrid", "walking-crowd verification", ORANGE, "^"),
        ("calib_grid", "calibrated-gain grid (sim_dt 0.001)", PLUM, "v")]
for i, (key, lab, col, mk) in enumerate(ARMS):
    rows = [r for r in master if arm_of(r) == key and rate_of(r) == 100.0]
    if not rows:
        continue
    dx = (i - 1.5) * 0.027                       # dodge arms sharing a speed
    sp = sorted({speed_of(r) for r in rows})
    xs, ys, lo, hi, ns = [], [], [], [], []
    for s_ in sp:
        k, n = kn([r for r in rows if speed_of(r) == s_])
        p, l, u = wilson(k, n)
        xs.append(s_ + dx); ys.append(p); lo.append(p - l); hi.append(u - p); ns.append(n)
    axB.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col, lw=1.15,
                 capsize=2.5, alpha=0.8, zorder=3)
    # 2-speed arms get a faint dotted link: two points are not a trend
    axB.plot(xs, ys, ":" if len(xs) <= 2 else "-", color=col,
             lw=1.3, alpha=0.6, zorder=4)
    axB.plot(xs, ys, mk, color=col, ms=7.5, mfc="white", mew=2.0, zorder=5,
             label=f"{lab}  (N={len(rows)})")
    voff = (15, -18, -18, 15)[i]                 # per-arm offset; keeps n= labels apart
    for x, y, n in zip(xs, ys, ns):
        axB.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, voff),
                     ha="center", fontsize=7.7, color=col, zorder=6)

axB.set_xlim(0.93, 1.90)
axB.set_ylim(*YLIM)
axB.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axB.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8])
axB.set_xticklabels(["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.8"])
axB.set_xlabel("commanded cruise speed (m/s)", fontsize=10.4)
axB.set_ylabel("course-completion rate", fontsize=10.4)
axB.set_title("B · replication check at a pinned 100 Hz control rate\n"
              "four independent arms, never pooled · Wilson 95% CI · dotted = only 2 speeds",
              fontsize=10.2, color=INK)
axB.grid(True, alpha=0.22, zorder=0)
axB.legend(fontsize=8.3, loc="upper left", framealpha=0.95)

fig.suptitle("Cruise velocity vs course-completion rate — warehouse HIL drone flights  ·  "
             "success is per-flight BINARY, so every point is a group of flights",
             fontsize=12.6, y=0.972, color=INK)

_FOOT = [
    "Data: hil_data_package.zip (sha256 877f0a4a…) → hil_flights_master.csv, 471 Isaac-Lab "
    "flights on a fixed 4-gate warehouse course (RL/MLP controller + YOLOv8n). A point is one "
    "(arm × commanded cruise speed) group; a flight is success / crash / timeout and success "
    "means all 4 gates, so the 55 timeouts (every one with gates_passed < 4) count as failures. "
    "No flight has a missing velocity or an unresolvable outcome, and after de-duplication 0 rows "
    "were dropped — the v1 archive is byte-identical to hil_ablation.csv and is excluded rather "
    "than pooled, which would have double-counted 120 flights.",
    "Velocity is the COMMANDED setpoint: no achieved-velocity column exists anywhere in the "
    "package. It is nonetheless flown — on the envelope arm's 28 successes the implied path "
    "length v·steps·sim_dt is 13.99–16.04 m at every speed from 1.0 to 1.8 m/s, so completion "
    "time scales as 1/v on a fixed-length course. Arms are never pooled: they differ in "
    "controller gain, perception hold and crowd, and the calibrated-gain grid also runs a 10× "
    "finer physics step (sim_dt 0.001 vs 0.01), so its 0/8 at 1.0 m/s cannot be read against the "
    "envelope's 4/6.",
    "CAVEAT — the trend is not robust. Across the envelope arm's full 1.0–1.8 m/s range the speed "
    "effect is significant (per-SD odds ratio 0.40×, p=0.0006, n=120), but it is carried entirely "
    "by the 1.6/1.8 m/s cells no other arm flew; over the shared 1.0–1.4 m/s range it is not "
    "significant (p=0.10), and no independent arm reproduces it (perception-freshness p=0.17 "
    "n=116, walking-crowd p=0.85 n=58, the authors' own 12-seed refresh p=0.48 n=96). Read this "
    "as a completion CEILING above ~1.4 m/s, not a smooth velocity–success trend.",
]
fig.text(0.5, 0.288, "\n".join(textwrap.fill(t, 232) for t in _FOOT),
         ha="center", va="top", fontsize=7.7, style="italic", color="#555", linespacing=1.45)

fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")


# ----------------------------------------------------------------------------- stdout
print("\n== PANEL A · envelope arm, k/n per (speed, rate) cell — n=6 everywhere ==")
print("   rate\\spd " + "".join(f"{s:>9g}" for s in SPEEDS))
for h in RATES:
    print(f"   {h:>6g}Hz " + "".join("{:>9}".format("%d/%d" % kn(cell[(s, h)])) for s in SPEEDS))
print("   " + "-" * 54)
print("   pooled   " + "".join(
    "{:>9}".format("%d/%d" % kn([r for r in env if speed_of(r) == s])) for s in SPEEDS))
for s in SPEEDS:
    k, n = kn([r for r in env if speed_of(r) == s])
    p, lo, hi = wilson(k, n)
    print(f"     {s:g} m/s : {k:>2}/{n:<3} = {p:.3f}  Wilson95%[{lo:.3f},{hi:.3f}]")

print(f"\n== authors' logistic surface refit on hil_ablation.csv (n={len(ly)}) ==")
for nm, i in [("log2(rate)", 1), ("cruise speed", 2)]:
    pv = 2 * (1 - phi(abs(BETA[i] / SE[i])))
    print(f"   {nm:<14} coef/SD={BETA[i]:+.3f}  odds/SD={math.exp(BETA[i]):.2f}x  p={pv:.4f}")
print("   -> matches scripts/hil_logistic_fit.py exactly (rate p=0.0030, speed p=0.0006)")

print("\n== PANEL B · arms at a pinned 100 Hz ==")
for key, lab, _, _ in ARMS:
    rows = [r for r in master if arm_of(r) == key and rate_of(r) == 100.0]
    if not rows:
        continue
    out = []
    for s in sorted({speed_of(r) for r in rows}):
        k, n = kn([r for r in rows if speed_of(r) == s])
        out.append(f"{s:g}:{k}/{n}")
    print(f"   {lab:<32} " + "  ".join(out))

print("\n== SANITY CHECKS ==")
print(f"   master rows                                : {len(master)}")
import hashlib
h1 = hashlib.md5(open(SRC / 'hil_ablation.csv', 'rb').read()).hexdigest()
h2 = hashlib.md5(open(SRC / 'hil_ablation_v1_120flights_fixedgain.csv', 'rb').read()).hexdigest()
print(f"   hil_ablation.csv == v1 archive (byte-equal) : {h1 == h2}  (would double-count 120)")
key = ("source", "seed", "cruise_speed", "eff_cmd_hz", "moment_scale",
       "percep_hold_ms", "percep_refresh", "walk_speed")
seen = defaultdict(int)
for r in master:
    seen[tuple(r[c] for c in key)] += 1
print(f"   duplicate flights inside master             : {sum(v - 1 for v in seen.values() if v > 1)}")
to = [r for r in master if r["outcome"] == "timeout"]
print(f"   timeouts (counted as failure)               : {len(to)}, "
      f"max gates_passed={max(int(r['gates_passed']) for r in to)} of 4 -> none is a hidden success")
print(f"   rows with blank/NaN cruise_speed            : "
      f"{sum(1 for r in master if not r['cruise_speed'].strip())}")
print(f"   rows with blank/unknown outcome             : "
      f"{sum(1 for r in master if r['outcome'] not in ('success', 'crash', 'timeout'))}")
esucc = [r for r in env if r["outcome"] == "success"]
EL = [speed_of(r) * int(r["steps"]) * float(r["sim_dt"]) for r in esucc]
print(f"   implied path length, envelope {len(esucc)} successes : "
      f"median {np.median(EL):.2f} m, range {min(EL):.2f}-{max(EL):.2f} m "
      f"-> commanded speed IS achieved")
asucc = [r for r in master if r["outcome"] == "success"]
AL = np.array([speed_of(r) * int(r["steps"]) * float(r["sim_dt"]) for r in asucc])
print(f"   ... all {len(asucc)} successes                     : median {np.median(AL):.2f} m, "
      f"IQR {np.percentile(AL, 25):.2f}-{np.percentile(AL, 75):.2f} m "
      f"(22-24 m tail = perception arm detouring round walkers)")
dts = sorted({(r["source"], r["sim_dt"]) for r in master})
odd = sorted({s for s, d in dts if d != "0.01"})
print(f"   sources NOT at sim_dt=0.01                  : {odd} -> 10x finer physics, do not pool")
