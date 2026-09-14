#!/usr/bin/env python3
"""AUDIT PASS 10 -- the torque the sim's actuators ACTUALLY apply.

ManiSkill drives every joint with a PhysX articulation drive:
    tau_drive = k_i (q_target,i - q_i) + d_i (0 - qd_i),   |tau_drive| <= force_limit_i
(`pd_joint_pos.py:35`; the drive velocity target is left at 0 outside interpolate mode).
None of that is in get_qf(): PhysX applies the drive internally, and set_qf/get_qf carry
only the gravity/Coriolis feed-forward ManiSkill adds on top.

The STIFFNESS half needs q_target and q, neither of which is logged. The DAMPING half
needs only qd, which IS logged. So  d_i * qd_i  is a strictly-recoverable LOWER BOUND on
one component of the real drive torque, and it can be compared like-for-like against the
tau the metric integrates.

Damping/stiffness/force_limit from the agent configs:
  widowx  defaults.py:59-77    google_robot  defaults.py:99-120
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
# in URDF/qf column order
DAMP = {8:  np.array([330., 180., 152.12, 309.62, 201.05, 269.51, 200., 200.]),
        11: np.array([1059.98, 1010.47, 767.28, 680., 674.99, 274.61, 340.53,
                      8., 8., 900., 900.])}
STIF = {8:  np.array([1169.79, 730., 808.46, 1229.13, 1272.28, 1056.33, 1000., 1000.]),
        11: np.array([1700., 1737.05, 979.98, 930., 1212.15, 432.97, 468.,
                      200., 200., 2000., 2000.])}
FLIM = {8:  np.array([200., 200., 100., 100., 100., 100., 60., 60.]),
        11: np.array([300., 300., 100., 100., 100., 100., 100., 60., 60., 100., 100.])}

print("=" * 100)
print("The VISCOUS half of the drive torque, d_i*qd_i, vs the tau the metric integrates")
print("=" * 100)
print(f"{'cell':<20}{'INT sum tau_qf^2':>18}{'INT sum (d qd)^2':>19}{'ratio':>10}"
      f"{'ticks |d qd| >= flim':>22}{'% of ticks':>12}")
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        A, B, nsat, ntot = [], [], 0, 0
        for e in s["episodes"]:
            qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
            qv = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qvel.npy")
            dof = qf.shape[1]
            td = DAMP[dof] * qv
            A.append((qf ** 2).sum() * dt); B.append((td ** 2).sum() * dt)
            nsat += int((np.abs(td) >= FLIM[dof]).any(1).sum()); ntot += len(qf)
        A, B = np.asarray(A), np.asarray(B)
        print(f"{cell:<20}{np.median(A):>18.1f}{np.median(B):>19.1f}"
              f"{np.median(B)/np.median(A):>10.1f}x{nsat:>21}{100*nsat/ntot:>11.1f}%")
    print()
print("""INTERPRETATION. d*qd is only ONE of the two PD terms and it still integrates to
1-3 orders of magnitude more than the quantity plotted as 'actuator force cost'. And on
a large fraction of ticks that term ALONE already exceeds the joint's configured
force_limit, so the sim's own drive is clipping -- which get_qf() cannot show, because
get_qf() never carried the drive torque in the first place.""")
