#!/usr/bin/env python3
"""Manipulator motion cost by policy schedule, over SUCCESSFUL episodes.

================================ WHAT IS MODELLED ==============================
Two quantities, deliberately kept separate, because the simulator has no physical
friction to integrate and pretending otherwise would invent joules:

(1) JOINT MOTION EFFORT      Eff = INTEGRAL sum_i omega_i(t)^2 dt        [rad^2/s]
    The joint-space analogue of a viscous cost. Reported in its OWN units, not
    watts. It charges steady motion correctly: constant omega gives a constant
    rate, unlike a kinetic-energy-increment model which charges a constant-speed
    reach exactly zero. Multiply by any viscous coefficient b [N.m.s/rad] to get
    joules; RATIOS BETWEEN SCHEDULES ARE INVARIANT TO b, which is what the
    comparison needs.

(2) PHYSICAL WORK            W_lift = sum_t max(0, dU),  U = sum_i m_i g z_i^com
                             W_kin  = sum_t max(0, dT),  T = 0.5 sum_i m_i |v_i|^2
    Real joules from the REAL link masses (robot 2.138 kg, of which ~1.6 kg moves;
    payload 0.0209 kg), summed over every link's centre of mass -- NOT the TCP's
    height times a lumped mass. The arm's COM can descend while the TCP rises, so
    a TCP-based lift term is wrong in magnitude and sometimes in sign.
    Only positive increments are charged (no regeneration).

WHY NOT sum_i d_i omega_i^2 IN WATTS. SAPIEN's joint damping is 152-330 N.m.s/rad
with friction exactly 0. Those are numerical stabilisation values for the PD
controller, not servo data: they yield 39-130 W for a 2 kg arm, which is not a
prediction about a WidowX. Effort is therefore reported unscaled.

WHY NOT JOINT TORQUE WORK. get_qf()/get_qvel() exist, but the joints settle
between control steps, so tau.omega at the harness's tick boundaries reads ~0.
Real actuator work lives in the 500 Hz substeps, behind ManiSkill's internal loop.

STILL NOT MODELLED: static holding torque (a real arm burns power holding a pose;
high-latency arms hold still MORE, so every metric here understates them),
joint-space geometry beyond the link COMs, drivetrain and electrical losses, and
work done on the object during contact.

SELECTION EFFECT: successful episodes only, so each arm is a different
subpopulation -- a high-latency arm only succeeds on the configs it can still do.
n is printed everywhere; eggplant CPU-only has n=0 successes and cannot be scored.
"""
from __future__ import annotations
import json, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "energy_by_schedule.png"
G = 9.81
ARMS = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
ALAB = {"lat0": "ideal\n0 ms", "pipe110": "PIPE\n110", "pipe200": "PIPE\n200",
        "serial283": "SERIAL\n283", "fp32_555": "CPU fp32\n555", "cpu685": "CPU int8\n685"}
COL = {"lat0": "#7f8c8d", "pipe110": "#16a085", "pipe200": "#1a9e8f",
       "serial283": "#e67e22", "fp32_555": "#c0392b", "cpu685": "#96281b"}
TASKS = [("drw", "close drawer", 1/27.0), ("egg", "eggplant in basket", 0.040)]


# ONE source for every arm. Mixing traces_energy2 with traces_energy would compare
# arms across INDEPENDENT runs of a harness that is not run-to-run deterministic
# (NONDETERMINISM.md) -- the arm differences would be partly run-to-run noise.
ROOT = HERE / ("traces_energy2" if (HERE / "traces_energy2" / "_COMPLETE").exists()
               else "traces_energy")


# Joint groups by robot, identified from the qvel width. This split matters: the
# finger joints are damped at 8.0 against 275-1060 for the arm, and carry almost no
# mass, so an unweighted sum over ALL joints lets gripper chatter dominate a
# "manipulator energy" number by ~100x while doing no mechanical work. MEASURED on
# drawer/serial283: finger effort 486 vs arm effort 4.5, with kinetic work flat.
JOINT_GROUPS = {8:  dict(arm=slice(0, 6), finger=slice(6, 8)),    # widowx: 6 + 2 fingers
                11: dict(arm=slice(0, 7), finger=slice(7, 9))}    # google: 7 + 2 fingers + 2 head


def collect(task, arm, dt):
    """Per successful episode: effort, lift work, kinetic work, duration."""
    d = ROOT / f"{task}_{arm}"
    if not (d.exists() and (d / "summary.json").exists()):
        return None
    s = json.load(open(d / "summary.json"))
    ok = {e["episode_id"] for e in s.get("episodes", []) if e.get("success")}
    eff, lift, kin, dur, fing = [], [], [], [], []
    bodies = sorted(d.glob("ep*_bodies.json"))
    m = np.asarray(json.load(open(bodies[0]))["link_mass"]) if bodies else None
    for f in sorted(d.glob("ep*_qvel.npy")):
        ep = int(f.name[2:4])
        if ep not in ok:
            continue
        w = np.load(f)
        g = JOINT_GROUPS.get(w.shape[1])
        e = (w ** 2).sum(0) * dt
        eff.append(e[g["arm"]].sum() if g else e.sum())
        fing.append(e[g["finger"]].sum() if g else 0.0)
        dur.append(len(w) * dt)
        com = d / f"ep{ep:02d}_link_com.npy"
        if com.exists() and m is not None:
            c = np.load(com)                          # (T, n_links, 3)
            U = (m[None, :] * G * c[:, :, 2]).sum(1)  # potential energy, J
            lift.append(np.maximum(0.0, np.diff(U)).sum())
            v = np.diff(c, axis=0) / dt
            T_ = 0.5 * (m[None, :] * (v ** 2).sum(2)).sum(1)
            kin.append(np.maximum(0.0, np.diff(T_)).sum())
    if not eff:
        return None
    return dict(n=len(eff), eff=np.array(eff), dur=np.array(dur),
                fing=np.array(fing),
                lift=np.array(lift) if lift else None,
                kin=np.array(kin) if kin else None)


rows = {t: {a: r for a in ARMS if (r := collect(t, a, dt))} for t, _, dt in TASKS}
present = [(t, l, dt) for t, l, dt in TASKS if rows.get(t)]
if not present:
    raise SystemExit("no traces")

fig, axes = plt.subplots(2, len(present), figsize=(6.4*len(present), 9.2), dpi=140,
                         gridspec_kw={"hspace": 0.33})
axes = np.atleast_2d(axes.reshape(2, -1))
for j, (task, tlab, dt) in enumerate(present):
    arms = [a for a in ARMS if a in rows[task]]
    ax = axes[0, j]
    bp = ax.boxplot([rows[task][a]["eff"] for a in arms], patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color="black", lw=1.7))
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(COL[a]); patch.set_alpha(0.75); patch.set_edgecolor("black")
    for i, a in enumerate(arms):
        v = rows[task][a]["eff"]
        ax.scatter(np.random.normal(i+1, 0.05, len(v)), v, s=9, c="black", alpha=0.35, zorder=4)
        ax.text(i+1, ax.get_ylim()[1], f"n={rows[task][a]['n']}", ha="center", va="top",
                fontsize=7.5, color="#333" if rows[task][a]["n"] >= 8 else "#c0392b",
                fontweight="normal" if rows[task][a]["n"] >= 8 else "bold")
    ax.set_xticks(range(1, len(arms)+1)); ax.set_xticklabels([ALAB[a] for a in arms], fontsize=8)
    ax.set_ylabel(r"ARM joint motion effort  $\int\sum_{arm}\omega_i^2 dt$  [rad$^2$/s]")
    ax.set_title(f"{tlab} — ARM motion effort per SUCCESSFUL episode (fingers excluded)", fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, j]
    x = np.arange(len(arms))
    has_w = all(rows[task][a]["lift"] is not None for a in arms)
    if has_w:
        lift = [float(np.median(rows[task][a]["lift"])) for a in arms]
        kin = [float(np.median(rows[task][a]["kin"])) for a in arms]
        ax.bar(x, lift, 0.6, color=[COL[a] for a in arms], edgecolor="black",
               linewidth=0.5, zorder=3, label=r"lift  $\sum\max(0,\Delta U)$, real link COMs")
        ax.bar(x, kin, 0.6, bottom=lift, color=[COL[a] for a in arms], alpha=0.45,
               hatch="//", edgecolor="black", linewidth=0.5, zorder=3,
               label=r"accelerate  $\sum\max(0,\Delta T)$")
        ax.set_ylabel("physical positive work per episode (J), median")
        ax.set_title("Real mechanical work — actual link masses (~1.6 kg moving, 21 g payload)",
                     fontsize=10.5)
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.bar(x, [rows[task][a]["eff"].mean()/rows[task][a]["dur"].mean() for a in arms], 0.6,
               color=[COL[a] for a in arms], edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_ylabel(r"effort RATE  $\sum_i\omega_i^2$  [rad$^2$/s$^2$]")
        ax.set_title("Effort per unit time (link-COM work pending re-trace)", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([ALAB[a] for a in arms], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Manipulator motion cost by policy schedule — successful episodes only",
             fontsize=13, y=0.985)
fig.text(0.5, 0.015,
         "Effort is reported UNSCALED: the sim's joint damping (152-330 N.m.s/rad, friction 0) is PD "
         "stabilisation, not servo data — scaling it gives 39-130 W for a 2 kg arm, which is not a "
         "prediction about a real WidowX.\nMultiply effort by any viscous coefficient b for joules; ratios "
         "between schedules are invariant to b. Static holding torque is not modelled, which understates "
         "the high-latency arms.", ha="center", fontsize=8.2, style="italic", color="#555")
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}   (source: {ROOT.name})\n")
for task, tlab, dt in present:
    print(f"=== {tlab} (successful episodes only) ===")
    b = rows[task].get("lat0")
    print(f"  {'arm':11s} {'n':>3s} {'effort med':>11s} {'IQR':>17s} {'vs ideal':>9s} "
          f"{'mean':>10s} {'dur(s)':>7s} {'lift(J)':>8s} {'kin(J)':>8s}")
    for a in [x for x in ARMS if x in rows[task]]:
        r = rows[task][a]
        med = float(np.median(r["eff"]))
        q1, q3 = np.percentile(r["eff"], [25, 75])
        rel = med/float(np.median(b["eff"])) if b else float("nan")
        lf = f"{np.median(r['lift']):8.3f}" if r["lift"] is not None else "       -"
        kn = f"{np.median(r['kin']):8.3f}" if r["kin"] is not None else "       -"
        print(f"  {a:11s} {r['n']:3d} {med:11.2f} [{q1:7.1f},{q3:7.1f}] {rel:8.2f}x "
              f"{r['eff'].mean():10.2f} {np.median(r['dur']):7.2f} {lf} {kn} "
              f"| finger {np.median(r['fing']):8.1f}")
    print("  (median-led: effort is heavy-tailed -- one oscillating episode can carry a mean)")
    print()
