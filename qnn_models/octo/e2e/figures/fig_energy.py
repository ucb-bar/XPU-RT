#!/usr/bin/env python3
"""Actuator cost by schedule -- the metric success rates hide.

Rebuilt over traces_torque2/: ALL FOUR tasks x the NINE corrected arms. Supersedes
the traces_torque/ version, which covered only eggplant+drawer and carried the OLD
mislabelled pipelined latencies (pipe110 was modelled at 117.7 ms when its actual
per-inference span is 385.1 ms -- a 3.3x understatement; see
archive/2026-09-06_pipelined-latency-mislabelled/).

THE MODEL, stated plainly (see also ENERGY_RESULTS.txt):
  tau^2 integral   INT sum_i tau_i(t)^2 dt over the joints, from the sim's own joint
                   forces (get_qf). A copper-loss (I^2 R) proxy: a servo pushing into a
                   rigid constraint has LARGE tau and ~ZERO omega, so this is the only
                   term here that sees a stall or a collision. Ratios between arms are
                   meaningful; absolute joules are not (no motor constant).
  arm effort       INT sum_arm omega_i^2 dt. Gripper fingers EXCLUDED -- damped at 8.0
                   vs 275-1060 for the arm and nearly massless, so unweighted they
                   dominate by ~100x while doing no mechanical work. On google_robot
                   the 2 head joints are excluded for the same reason.

Both are integrated over the WHOLE episode, so an arm that succeeds quickly is charged
less than one that flails to the horizon. That is deliberate: the question asked was
total mission energy, not power. Mission time is plotted separately (fig_metrics.py).

ARMS ARE ORDERED BY CADENCE, not by latency, because cadence is the variable that moves
success. The ordering is therefore NOT monotone in latency -- pipe110fix has the WORST
observation age of any scheduled arm (385.1 ms) and the BEST cadence (111.4 ms). If the
energy curve tracks this ordering, it is tracking cadence.

UNCONDITIONED on success, deliberately. The success-conditioned view is blind: eggplant
cpu685 succeeds 4.2% of the time, so conditioning keeps the 1-in-24 episode that behaved
like the ideal arm and discards all the flailing -- which is the entire effect.

NOT modelled: contact impulses (tau^2 is a proxy, and get_qf() is sampled at tick
boundaries so brief impacts are under-counted), drivetrain and electrical losses.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
# Seeded: the strip-plot jitter is cosmetic, but an unseeded RNG makes the figure
# non-reproducible byte-for-byte, which hides real changes in a diff.
np.random.seed(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent
SRC = HERE.parent.parent.parent.parent.parent / "octo_work/sim_eval/roselite/finegrain/traces_torque2"  # fallback only; the cache is what ships
OUT = Path(__file__).parent / "fig_energy.png"

# name -> (cadence ms, latency ms). MEASURED, ungated (job_torque2.sh).
ARMS = [
    ("lat0",       None,  0.0),
    ("pipe110fix", 111.4, 385.1),
    ("p105w300",   124.8, 258.7),
    ("p130w275",   130.1, 272.3),
    ("p150w300",   150.4, 281.6),
    ("pipe200fix", 219.2, 260.5),
    ("serial283",  283.4, 283.4),
    ("fp32_555",   555.0, 555.0),
    ("cpu685",     684.8, 684.8),
]
SHORT = {"lat0": "ideal", "pipe110fix": "pipe110", "p105w300": "p105/300",
         "p130w275": "p130/275", "p150w300": "p150/300", "pipe200fix": "pipe200",
         "serial283": "serial", "fp32_555": "fp32", "cpu685": "cpu int8"}
# cool -> hot with cadence; the ideal arm is grey because it is not on the cadence axis.
COL = {"lat0": "#7f8c8d", "pipe110fix": "#0e6655", "p105w300": "#148f77",
       "p130w275": "#17a589", "p150w300": "#45b39d", "pipe200fix": "#d68910",
       "serial283": "#e67e22", "fp32_555": "#c0392b", "cpu685": "#7b241c"}

# dof -> (arm joints, everything else). widowx 8 = 6 arm + 2 fingers.
# google_robot 11 = 7 arm + 2 fingers + 2 head.
JG = {8: slice(0, 6), 11: slice(0, 7)}

TASKS = [("egg",    "eggplant in basket",  "GRASP", "widowx"),
         ("spoon",  "spoon on towel",      "GRASP", "widowx"),
         ("coke",   "pick coke can",       "GRASP", "google"),
         ("drawer", "close drawer",        "PUSH",  "google")]


# Prefer the committed scalar cache (39 KB) over the 700 MB trace tree, so the figure
# reproduces from a checkout. Falls back to the traces when the cache is absent.
_CACHE = None
_cf = Path(__file__).parent.parent / "data" / "energy_cache.json"
if _cf.exists():
    _CACHE = json.load(open(_cf))


def load(task, arm):
    if _CACHE is not None:
        c = _CACHE.get(f"{task}_{arm}")
        if c is None:
            return None
        return dict(c, t2=np.asarray(c["t2"]), eff=np.asarray(c["eff"]))
    d = SRC / f"{task}_{arm}"
    if not (d / "summary.json").exists():
        return None
    s = json.load(open(d / "summary.json"))
    dt = s["tick_ms"] / 1000.0          # trace is logged per SIM TICK, not per actuation
    m = np.asarray(json.load(open(d / "ep00_bodies.json"))["link_mass"])
    T2, EF, INF, ACT = [], [], [], []
    for e in s["episodes"]:
        ep = e["episode_id"]
        try:
            qf = np.load(d / f"ep{ep:02d}_qf.npy")
            w = np.load(d / f"ep{ep:02d}_qvel.npy")
        except Exception:
            continue
        a = JG.get(w.shape[1], slice(None))
        T2.append(float((qf ** 2).sum(1).sum() * dt))
        EF.append(float(((w ** 2).sum(0) * dt)[a].sum()))
        INF.append(e["n_inferences"]); ACT.append(e.get("n_actuations", e["ticks"]))
    if not T2:
        return None
    ages = [e.get("act_age_mean_ms") for e in s["episodes"]
            if e.get("act_age_mean_ms") is not None]
    return dict(sr=100.0 * s["n_success"] / s["n_episodes"],
                t2=np.array(T2), eff=np.array(EF), n=len(T2),
                dt=dt, nep=s["n_episodes"],
                grid=s.get("act_ms", s["tick_ms"]),
                age=float(np.mean(ages)) if ages else 0.0,
                # inferences produced per COMMAND the robot can actually issue.
                # >1 means the policy is running faster than the actuator can
                # consume, and the surplus results are discarded unused.
                ipa=float(np.sum(INF)) / float(np.sum(ACT)))


def xlabel(name, cad, lat):
    if cad is None:
        return f"{SHORT[name]}\n-- / {lat:.0f}"
    return f"{SHORT[name]}\n{cad:.0f} / {lat:.0f}"


OFF = 0


def satband(ax, arms, data, label=True):
    """Shade the arms whose policy outruns the robot's own control grid."""
    idx = [i for i, a in enumerate(arms) if data[a]["ipa"] >= 1.0 and a != "lat0"]
    if not idx:
        return
    lo, hi = min(idx), max(idx)
    ax.axvspan(lo - 0.5 + OFF, hi + 0.5 + OFF, color="#5b2c6f", alpha=0.07,
               zorder=0, lw=0)
    ax.axvline(hi + 0.5 + OFF, color="#5b2c6f", alpha=0.45, lw=1.1, ls=":", zorder=1)
    if not label:
        return
    g = data[arms[0]]["grid"]
    ax.text((lo + hi) / 2 + OFF, ax.get_ylim()[1] * 0.90,
            f"policy outruns the {1000/g:.0f} Hz control grid\n"
            f"{data[arms[lo]]['ipa']:.1f}-{data[arms[hi]]['ipa']:.1f} inferences per command "
            "-- the surplus is discarded",
            ha="center", va="top", fontsize=6.4, color="#5b2c6f", style="italic")


fig, axes = plt.subplots(len(TASKS), 2, figsize=(14.6, 17.1), dpi=150,
                         gridspec_kw={"hspace": 0.52, "wspace": 0.22})
missing = []
for r, (task, lab, kind, emb) in enumerate(TASKS):
    data = {}
    for name, cad, lat in ARMS:
        d = load(task, name)
        if d is None:
            missing.append(f"{task}/{name}")
        else:
            data[name] = d
    arms = [a for a, _, _ in ARMS if a in data]
    if not arms:
        continue
    labs = [xlabel(a, c, l) for a, c, l in ARMS if a in data]
    x = np.arange(len(arms))
    base_t2 = float(np.median(data[arms[0]]["t2"]))
    base_ef = float(np.median(data[arms[0]]["eff"]))
    grid = data[arms[0]]["grid"]
    head = f"{lab}  [{kind}, {emb} @ {1000/grid:.0f} Hz = {grid:.0f} ms control grid]"

    # ---- column 1: tau^2, stall / contact ----------------------------------
    ax = axes[r, 0]
    bp = ax.boxplot([data[a]["t2"] for a in arms], patch_artist=True, widths=0.62,
                    showfliers=False, medianprops=dict(color="black", lw=1.6))
    for p, a in zip(bp["boxes"], arms):
        p.set_facecolor(COL[a]); p.set_alpha(0.85); p.set_edgecolor("black")
    # Robust y-limit FIRST: a single outlier two orders up flattens every box to the
    # axis floor. It must be set before the ratio labels are placed, or the labels
    # land outside the final limit and bbox_inches="tight" grows the canvas to
    # tens of thousands of pixels (this happened; see git history).
    hi = max(float(np.percentile(data[a]["t2"], 90)) for a in arms)
    lo = min(float(np.percentile(data[a]["t2"], 5)) for a in arms)
    ax.set_ylim(max(0.0, lo - 0.1 * (hi - lo)), hi * 1.34)
    nclip = sum(int((data[a]["t2"] > hi * 1.34).sum()) for a in arms)
    if nclip:
        ax.text(0.99, 0.03, f"{nclip} point(s) above axis", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.6, style="italic", color="#888")
    top = ax.get_ylim()[1]
    for i, a in enumerate(arms):
        ax.scatter(np.random.normal(i + 1, 0.055, data[a]["n"]), data[a]["t2"],
                   s=6, c="black", alpha=0.28, zorder=4, clip_on=True)
        ax.text(i + 1, top * 0.975, f"{np.median(data[a]['t2']) / base_t2:.2f}x",
                ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.set_xticks(range(1, len(arms) + 1))
    ax.set_xticklabels(labs, fontsize=6.6)
    OFF = 1
    satband(ax, arms, data)
    ax.set_ylabel(r"$\int\sum_i \tau_i^2\,dt$", fontsize=9)
    ax.set_title(f"{head}\nactuator force cost  (stalls & contact)", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)

    # ---- column 2: omega^2, motion ----------------------------------------
    ax = axes[r, 1]
    med = [float(np.median(data[a]["eff"])) for a in arms]
    ax.bar(x, med, 0.64, color=[COL[a] for a in arms], edgecolor="black",
           linewidth=0.5, zorder=3)
    ax.set_ylim(0, max(med) * 1.30)
    for i, a in enumerate(arms):
        ax.text(i, med[i], f"{med[i] / base_ef:.2f}x\n{data[a]['sr']:.0f}%",
                ha="center", va="bottom", fontsize=6.6, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.6)
    OFF = 0
    satband(ax, arms, data, label=False)
    ax.set_ylabel(r"$\int\sum_{arm}\omega_i^2\,dt$", fontsize=9)
    ax.set_title(f"{head}\narm motion effort  (thrashing; % = success rate)",
                 fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("ACTUATOR ENERGY goes flat once the policy outruns the robot's control rate."
             "\nWidowX (25 Hz) is never saturated and cost tracks cadence; google_robot "
             "(3 Hz) is saturated by every schedule here, and cost is flat.",
             fontsize=12.5, y=0.992)
foot = (
    "x labels: arm, then CADENCE / OBSERVATION AGE in ms, both MEASURED on the QRB5165 (ungated). "
    "Arms are ordered by CADENCE, so the axis is NOT monotone in age:\n"
    "pipe110 has the worst age of any scheduled arm (385 ms) and the best cadence (111 ms). "
    "Ratios are vs. the ideal 0 ms arm. 24 episodes per cell, one seed (init_rng=100).\n"
    "\n"
    "THE SHADED BAND is the finding. WidowX actuates every 40 ms, so no schedule here keeps up: "
    "each policy result is held for 3-18 commands (0.36 down to 0.06 inferences per command), and\n"
    "cost rises with cadence across the whole range (spoon: 1.00x -> 2.59x, monotone). "
    "google_robot actuates every 333 ms, so EVERY schedule from 111 to 283 ms cadence outruns it -- "
    "pipe110 computes\n"
    "2.98 inferences per command and throws two thirds away. Across that entire band the observation "
    "age still varies 307-411 ms, yet both ENERGY metrics are flat; only fp32/cpu685, which fall\n"
    "BELOW the grid (0.60 / 0.49 per command, age 780-1010 ms), cost more. On a 3 Hz arm the ideal "
    "0 ms schedule is also the cheapest: it computes exactly 1.00 inferences per command.\n"
    "\n"
    "SCOPE OF THAT CLAIM -- it is about ENERGY, which is what this figure measures. Saturation caps how often the "
    "robot can ACT; it does not close two channels through which a faster cadence still reaches the\n"
    "controller. (i) RESIDUAL STALENESS: the held result is on average ~C/2 old before it is consumed, so at "
    "identical schedule latency the measured age@act still runs 353 ms at cadence 219 vs 411 ms at cadence 283.\n"
    "(ii) ENSEMBLE DEPTH: the harness runs stock ActionEnsembler, so every inference contributes a chunk to the "
    "average -- 2.98 chunks per command at cadence 111 vs 1.17 at cadence 283. Both are invisible to\n"
    "these two metrics, so success is NOT guaranteed to be flat wherever energy is. As of the wide E2E sweep's "
    "coke cells, though, neither channel has produced a demonstrated success effect inside the\n"
    "saturated band: a per-seed trend test across the cadence ladder at fixed age gives rho = -0.11 "
    "[-0.77, +0.54], 5 seeds down / 4 up. The pooled means were monotone (39.6 -> 35.8 -> 32.9%), but that\n"
    "monotonicity came from noise, not from an ordering the seeds agree on. So on google_robot the saturated "
    "band is currently flat in BOTH energy and success. The widowx tasks are where the effect lives\n"
    "and their cells had not landed at the time of writing.\n"
    "\n"
    r"$\tau^2$ is a copper-loss ($I^2R$) proxy: a servo pushing into a rigid constraint has large $\tau$ and "
    r"~zero $\omega$, so it is the only term that sees a stall or collision. The two columns measure "
    "DIFFERENT physics and can\ndiverge -- an arm jammed against the sink has large "
    r"$\tau$ and ~zero $\omega$; an arm swinging freely has the reverse. Both integrate over the whole "
    "episode, so a fast success is charged less than a flail to the horizon. UNCONDITIONED on success."
)
if missing:
    foot += "\nMISSING CELLS (not drawn): " + ", ".join(missing)
fig.text(0.5, 0.150, foot, ha="center", va="top", fontsize=7.4, style="italic",
         color="#555")
fig.subplots_adjust(top=0.947, bottom=0.172)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
if missing:
    print("[warn] missing:", missing)
