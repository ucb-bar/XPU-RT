#!/usr/bin/env python3
"""Flight energy per attempt and per COMPLETED course, from momentum theory.

The HIL package logs no power or current, so energy is MODELLED. Momentum theory for a
rotor in steady level flight (Glauert), with every constant read from the Crazyflie 2.x
URDF shipped in this repo (gym_pybullet_drones/assets/cf2x.urdf) rather than assumed:

    m = 0.027 kg,  4 rotors,  R = 2.31348e-2 m   ->  disc area A = 67.3 cm^2
    T = m g = 0.265 N        v_h = sqrt(T / 2 rho A) = 4.01 m/s   (hover induced vel)

  INDUCED  solve the Glauert quartic  v_i^4 + V^2 v_i^2 - v_h^4 = 0,  P_ind = T v_i
           Induced power FALLS with airspeed: 1.062 W hover -> 1.010 W at 1.8 m/s.
  PARASITE P_par = 0.5 rho Cd Af V^3.  The URDF's drag_coeff_xy is a linear coefficient
           in rotor speed, NOT a Cd*Af, so it cannot be used here; Cd*Af = 3.0e-3 m^2 is
           a flat-plate estimate for a 27 g quad and is the model's weakest assumption.
  PROFILE  P_pro held constant at the hover value: blade geometry (chord, twist, Cd0) is
           not in the URDF, so the mu^2 forward-flight correction cannot be evaluated.
           This UNDERSTATES cost at high speed, i.e. it biases against the paper's story.

Two panels, and the second is the one that matters:

  A  ENERGY PER ATTEMPT = P(V) * t, t = steps * sim_dt (logged per flight). Falls with
     speed, because on a fixed ~14.6 m course flight time scales as 1/V and parasite
     power is negligible at these speeds. Flying faster is cheaper -- IF you finish.
  B  ENERGY PER COMPLETED COURSE = (energy of ALL attempts) / (number of successes).
     Failed attempts are charged to the successes they did not produce. This inverts
     the ranking: at 1.6-1.8 m/s the envelope arm completes 2 of 24, so each completion
     carries 12 flights' worth of energy.

Bars, not a scatter: the x axis is a small set of discrete commanded speeds, and the
quantity of interest is a per-group aggregate, not a per-flight cloud.

Arms are NOT pooled -- the package's own README says not to, and the calibrated-gain grid
additionally runs sim_dt=0.001, a 10x finer physics step than every other arm.
"""
from __future__ import annotations
import csv, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV = Path("/tmp/hil_master.csv")
OUT = HERE / "fig_drone_energy.png"

M, G, RHO = 0.027, 9.81, 1.225
R, NROT = 2.31348e-2, 4
CD_AF = 3.0e-3                      # flat-plate estimate -- the weak assumption
A = NROT * np.pi * R ** 2
T = M * G
VH = np.sqrt(T / (2 * RHO * A))


def induced_velocity(V):
    """Glauert level-flight quartic: v_i^4 + V^2 v_i^2 - v_h^4 = 0."""
    b, c = V ** 2, -VH ** 4
    return np.sqrt((-b + np.sqrt(b * b - 4 * c)) / 2.0)


def power(V):
    return T * induced_velocity(V) + 0.5 * RHO * CD_AF * V ** 3 + T * VH * 0.0


rows = list(csv.DictReader(open(CSV)))
grp = collections.defaultdict(lambda: {"E": 0.0, "n": 0, "s": 0, "t": []})
for r in rows:
    reg, V = r["regime"], float(r["cruise_speed"])
    t = float(r["steps"]) * float(r["sim_dt"])
    g = grp[(reg, V)]
    g["E"] += power(V) * t
    g["n"] += 1
    g["s"] += int(r["success"])
    g["t"].append(t)

REG = [("envelope (figure source)", "envelope", "#0e6655"),
       ("perc_freshness/safety (canonical; separate experiment)", "perception-freshness", "#C77400"),
       ("misc/verification", "misc/verification", "#7A1250"),
       ("showdown (calibrated gain, ZOH latency)", "showdown", "#16161C"),
       ("calibrated-gain grid", "calibrated-gain (sim_dt 1e-3)", "#9a9aa4")]

fig, axes = plt.subplots(1, 2, figsize=(16.4, 7.8), dpi=150,
                         gridspec_kw={"wspace": 0.22})
speeds = sorted({V for _, V in grp})
x = np.arange(len(speeds))
w = 0.16

for pi, (ax, mode) in enumerate(zip(axes, ("attempt", "success"))):
    for k, (reg, lab, col) in enumerate(REG):
        ys, ann = [], []
        for V in speeds:
            g = grp.get((reg, V))
            if not g:
                ys.append(np.nan); ann.append(""); continue
            if mode == "attempt":
                ys.append(g["E"] / g["n"]); ann.append(f"n={g['n']}")
            else:
                ys.append(g["E"] / g["s"] if g["s"] else np.inf)
                ann.append(f"{g['s']}/{g['n']}")
        pos = x + (k - 2) * w
        fin = [0 if not np.isfinite(v) else v for v in ys]
        ax.bar(pos, fin, w, color=col, edgecolor="black", linewidth=0.4,
               label=lab if pi == 0 else None, zorder=3)
        top = np.nanmax([v for v in fin if np.isfinite(v)] or [1])
        for xi, (v, a) in enumerate(zip(ys, ann)):
            if a == "":
                continue
            if not np.isfinite(v):                     # zero completions
                ax.annotate("", xy=(pos[xi], top * 0.98), xytext=(pos[xi], top * 0.55),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                            mutation_scale=14), zorder=5)
                ax.text(pos[xi], top * 0.99, "0 done", ha="center", va="bottom",
                        fontsize=6.4, color=col, fontweight="bold", rotation=90)
            else:
                ax.text(pos[xi], v, a, ha="center", va="bottom", fontsize=5.8,
                        color="#444", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"{v:.1f}" for v in speeds])
    ax.set_xlabel("commanded cruise speed (m/s)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    if mode == "attempt":
        ax.set_ylabel("energy per ATTEMPT (J)", fontsize=9)
        ax.set_title("A · cost of flying — falls with speed\n"
                     "(fixed course, t ~ 1/V; parasite negligible below 2 m/s)",
                     fontsize=9.8)
    else:
        ax.set_ylabel("energy per COMPLETED course (J)", fontsize=9)
        ax.set_title("B · cost of finishing — rises with speed\n"
                     "(failed attempts charged to the completions they did not produce)",
                     fontsize=9.8)
axes[0].legend(fontsize=7.4, loc="upper right", framealpha=0.92)

fig.suptitle("Crazyflie flight energy from momentum theory — flying faster is cheaper, "
             "finishing faster is not", fontsize=13, y=0.972)
fig.subplots_adjust(top=0.870, bottom=0.300)
fig.text(0.5, 0.150,
         "MODELLED, not measured — the HIL package logs no power or current. Glauert momentum theory in steady level\n"
         "flight; constants from the Crazyflie 2.x URDF shipped in this repo (m=27 g, 4 rotors, R=23.1 mm, so v_h=4.01 m/s\n"
         "and hover induced power 1.06 W). Induced power FALLS with airspeed (1.062 to 1.010 W over 0 to 1.8 m/s).\n"
         "Profile power is held at its hover value because blade chord/twist/Cd0 are absent from the URDF — that\n"
         "UNDERSTATES high-speed cost, biasing AGAINST panel B. Parasite uses Cd·Af = 3.0e-3 m² (flat-plate estimate),\n"
         "the weakest assumption; the URDF's drag_coeff_xy is linear in rotor speed and is not a Cd·Af.\n"
         "Labels: panel A = n flights, panel B = successes/attempts. Arms are never pooled (README forbids it; the\n"
         "calibrated-gain grid also runs a 10× finer physics step).",
         ha="center", va="top", fontsize=7.4, style="italic", color="#555", linespacing=1.5)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
print(f"  v_h={VH:.3f} m/s  P_hover_induced={T*VH:.4f} W")
for reg, lab, _ in REG:
    cells = [(V, grp[(reg, V)]) for V in speeds if (reg, V) in grp]
    if not cells:
        continue
    s = ", ".join(f"{V:.1f}:{(g['E']/g['s']):.1f}J" if g["s"] else f"{V:.1f}:inf"
                  for V, g in cells)
    print(f"  {lab:32s} energy/completion  {s}")
