#!/usr/bin/env python3
"""Success, mission time and actuator energy — nine schedules x four tasks, n=10 seeds.

Supersedes fig_metrics.py + fig_energy.py for the curated arm set. Three things changed
and all three matter:

1. ENERGY IS NOW A REAL ACTUATOR SIGNAL. The previous figures integrated
   `robot.get_qf()`, which in this SAPIEN build is `compute_passive_force(external=False)`
   -- gravity + Coriolis, blind to contact, stall and drive torque by construction
   (ENERGY_AUDIT.md; tau was identically 0 at tick 0 in 864/864 episodes). This uses
   `t2_drive_arm_sus` from trace_eval2.py: the PD drive torque PhysX actually applies,
   reconstructed per SUBSTEP as clip(K*(q_tgt - q) + D*(v_tgt - v), +-force_limit) and
   validated against the articulation's own equation of motion to 0.024 N.m. The
   `_sus` variant skips the first 4 substeps after each setpoint step, because on widowx
   96.9% of the free-space integral otherwise lands in the single substep after the 5 Hz
   command changes -- which would measure commanded step size, i.e. cadence, not work.

2. n=10 SEEDS, NOT 1. Every point carries a bootstrap 95% CI over seeds (B=4000).
   The old figure's single seed could not distinguish an arm from itself: the audit
   measured the ideal arm's self-ratio at 1.00 [0.45, 2.17].

3. MISSION TIME IS CONDITIONED ON SUCCESS and says so. A failure has no completion time,
   so this biases slow arms DOWNWARD -- they only finish the episodes they can still do.
   n is printed per bar, red where n < 20 episodes contribute.

Energy is UNCONDITIONED on success, deliberately: conditioning keeps only the episodes an
arm happened to win, which on eggplant cpu685 is 1 in 60, and hides the flailing that is
the effect. Ratios between arms are meaningful; absolute values are not calibrated to
joules (ENERGY_AUDIT.md section 4 -- ROBOTIS publishes no winding resistance or Kt).
"""
from __future__ import annotations
import json, glob, re, collections
from pathlib import Path

import numpy as np
np.random.seed(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC = HERE.parent / "traces_torque3"
OUT = HERE / "fig_metrics3.png"

ARMS = [("lat0", "ideal\n0 ms", "#7f8c8d"),
        ("pipe110fix", "pipe110\n111", "#0e6655"), ("p105w300", "p105/300\n125", "#148f77"),
        ("p130w275", "p130/275\n130", "#17a589"), ("p150w300", "p150/300\n150", "#45b39d"),
        ("pipe200fix", "pipe200\n219", "#d68910"), ("serial283", "serial\n283", "#e67e22"),
        ("fp32_555", "fp32\n555", "#c0392b"), ("cpu685", "cpu int8\n685", "#7b241c")]
TASKS = [("egg", "eggplant in basket", "widowx"), ("spoon", "spoon on towel", "widowx"),
         ("coke", "pick coke can", "google"), ("drawer", "close drawer", "google")]

D = collections.defaultdict(dict)
for f in glob.glob(str(SRC / "*" / "energy2.json")):
    k = Path(f).parent.name
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", k)
    D[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))


def ci(v, B=4000, seed=0):
    """Bootstrap 95% CI of the mean over seeds."""
    if len(v) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed); a = np.asarray(v, float)
    m = [r.choice(a, a.size, replace=True).mean() for _ in range(B)]
    return tuple(np.percentile(m, [2.5, 97.5]))


fig, axes = plt.subplots(3, len(TASKS), figsize=(19.5, 12.4), dpi=140,
                         gridspec_kw={"hspace": 0.46, "wspace": 0.20})
for c, (task, tlab, emb) in enumerate(TASKS):
    arms = [a for a, _, _ in ARMS if (task, a) in D]
    x = np.arange(len(arms))
    cols = [col for a, _, col in ARMS if a in arms]
    labs = [lb for a, lb, _ in ARMS if a in arms]

    # ---- row 0: success rate, per-seed mean with bootstrap CI ----------------
    ax = axes[0, c]
    sr = [[100 * d["n_success"] / d["n_episodes"] for d in D[(task, a)].values()] for a in arms]
    mu = [np.mean(v) for v in sr]
    lo = [m - ci(v)[0] for m, v in zip(mu, sr)]; hi = [ci(v)[1] - m for m, v in zip(mu, sr)]
    ax.bar(x, mu, 0.66, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(x, mu, yerr=[lo, hi], fmt="none", ecolor="#16161C", capsize=3, lw=1.2, zorder=4)
    ax.set_title(f"{tlab}  [{emb}]\nsuccess rate", fontsize=9.6)
    if c == 0: ax.set_ylabel("success (%)\nmean over 10 seeds", fontsize=8.6)

    # ---- row 1: mission time on SUCCESSFUL episodes only ---------------------
    ax = axes[1, c]
    mt, ns = [], []
    for a in arms:
        v = [e["duration_s"] for d in D[(task, a)].values() for e in d["episodes"] if e["success"]]
        mt.append(v); ns.append(len(v))
    mu2 = [np.mean(v) if v else np.nan for v in mt]
    e2 = [[m - ci(v)[0] if len(v) > 1 else 0 for m, v in zip(mu2, mt)],
          [ci(v)[1] - m if len(v) > 1 else 0 for m, v in zip(mu2, mt)]]
    ax.bar(x, mu2, 0.66, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(x, mu2, yerr=e2, fmt="none", ecolor="#16161C", capsize=3, lw=1.2, zorder=4)
    for i, (m, n) in enumerate(zip(mu2, ns)):
        if np.isfinite(m):
            ax.text(i, m, f"n={n}", ha="center", va="bottom", fontsize=6.4,
                    color="#c0392b" if n < 20 else "#444")
    ax.set_title("mission time, SUCCESSES only\n(conditioned: biases slow arms down)", fontsize=9.0)
    if c == 0: ax.set_ylabel("completion time (s)", fontsize=8.6)

    # ---- row 2: actuator energy, unconditioned, ratio to the ideal arm -------
    ax = axes[2, c]
    per = {a: [np.median([e["t2_drive_arm_sus"] for e in d["episodes"]])
               for d in D[(task, a)].values()] for a in arms}
    base = np.mean(per["lat0"])
    mu3 = [np.mean(per[a]) / base for a in arms]
    e3 = [[mu3[i] - ci(per[a])[0] / base for i, a in enumerate(arms)],
          [ci(per[a])[1] / base - mu3[i] for i, a in enumerate(arms)]]
    ax.bar(x, mu3, 0.66, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(x, mu3, yerr=e3, fmt="none", ecolor="#16161C", capsize=3, lw=1.2, zorder=4)
    ax.axhline(1.0, color="#16161C", lw=1.0, ls="--", alpha=0.6, zorder=2)
    if max(mu3) / max(min(mu3), 1e-9) > 8:
        ax.set_yscale("log")
    for i, a in enumerate(arms):
        sep = "" if (ci(per[a])[0] / base <= 1 <= ci(per[a])[1] / base) else "*"
        if sep:
            ax.text(i, mu3[i], "*", ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color="#16161C")
    ax.set_title("actuator energy (drive torque)\nUNconditioned  ·  * = CI excludes 1.0", fontsize=9.0)
    if c == 0: ax.set_ylabel(r"$\int\sum\tau_{drive}^2 dt$ / ideal", fontsize=8.6)

    for r in range(3):
        axes[r, c].set_xticks(x)
        axes[r, c].set_xticklabels(labs, fontsize=6.3, rotation=38, ha="right")
        axes[r, c].grid(True, axis="y", alpha=0.3, zorder=0)

fig.suptitle("Success, mission time and ACTUATOR energy — nine schedules × four tasks, "
             "10 seeds × 24 episodes = 8,640 episodes per row", fontsize=13, y=0.975)
fig.subplots_adjust(top=0.918, bottom=0.128)
fig.text(0.5, 0.062,
         "Energy is the PD drive torque PhysX applies, reconstructed per substep and "
         "validated against the articulation's equation of motion to 0.024 N·m. The "
         "previous figures integrated get_qf(), which is gravity + Coriolis only and "
         "blind to contact and stall by construction (ENERGY_AUDIT.md);\n"
         "that proxy compressed spoon's 45× effect into 2.6×. Error bars are bootstrap "
         "95% CIs over 10 seeds (B=4000). Mission time is CONDITIONED on success — a "
         "failure has no completion time — which biases slow arms downward; n is printed, "
         "red where n<20.\n"
         "Energy is UNconditioned: conditioning would keep only the episodes an arm "
         "happened to win (1 in 60 for eggplant cpu685) and hide the flailing. Ratios are "
         "meaningful; absolute values are not calibrated to joules — ROBOTIS publishes no "
         "winding resistance or Kt for these servos.",
         ha="center", va="top", fontsize=7.9, style="italic", color="#555")
fig.savefig(OUT, dpi=140)
print(f"[ok] {OUT}")
for task, _, _ in TASKS:
    arms = [a for a, _, _ in ARMS if (task, a) in D]
    n = sum(len(D[(task, a)]) for a in arms)
    print(f"  {task:7s} {len(arms)} arms, {n} runs")
