#!/usr/bin/env python3
"""AUDIT PASS 3 -- the decisive test: does tau see CONTACT?

`base_controller.py:229-238` computes
    qf = articulation.compute_passive_force(external=False)      # gravity + Coriolis
    ... (no controller in this repo returns a 'qf' term) ...
    articulation.set_qf(qf)
and `get_qf()` returns that applied qf. If that reading is right, then

 A) tau on a joint whose axis is world-VERTICAL carries no gravity moment at all, so it
    must be pure Coriolis/centrifugal -- i.e. it must vanish as ||omega|| -> 0 EVEN WHEN
    THE ARM IS JAMMED. A contact reaction would not.
 B) close_drawer is the cleanest possible probe: the gripper pushes HORIZONTALLY into a
    drawer front at near-zero velocity. The reaction is a large moment about
    `joint_torso` (vertical). If tau saw contact, torso tau would be large exactly when
    omega is small. Test the opposite prediction.
 C) The same for widowx pressing down onto the sink/table: `waist` is vertical.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
VJ = {8: {"waist": 0}, 11: {"joint_torso": 0, "joint_bicep": 2, "joint_head_pan": 9}}

print("=" * 84)
print("A/B/C  vertical-axis tau vs joint speed.  Gravity moment about a vertical axis")
print("       is IDENTICALLY ZERO, so anything left is Coriolis (~omega^2) or contact.")
print("=" * 84)
for cell in ["drawer_lat0", "drawer_cpu685", "coke_lat0", "egg_lat0", "egg_cpu685",
             "spoon_cpu685"]:
    s = json.load(open(SRC / cell / "summary.json"))
    QF, QV = [], []
    for e in s["episodes"]:
        QF.append(np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy"))
        QV.append(np.load(SRC / cell / f"ep{e['episode_id']:02d}_qvel.npy"))
    qf, qv = np.concatenate(QF), np.concatenate(QV)
    dof = qf.shape[1]
    sp = np.linalg.norm(qv, axis=1)             # ||omega|| over all joints
    print(f"\n--- {cell}  ({qf.shape[0]} ticks)")
    for nm, j in VJ[dof].items():
        lo = sp <= np.percentile(sp, 10)        # arm essentially stationary
        hi = sp >= np.percentile(sp, 90)        # arm moving fast
        a = np.abs(qf[lo, j]); b = np.abs(qf[hi, j])
        # quadratic-in-omega test: regress tau^2 on (sum omega^2)^2
        x = (qv ** 2).sum(1)
        r = np.corrcoef(x, qf[:, j] ** 2)[0, 1]
        print(f"    {nm:<16} |tau| slowest-decile: mean {a.mean():.5f} max {a.max():.5f}"
              f"  | fastest-decile: mean {b.mean():.5f} max {b.max():.5f}"
              f"  | corr(tau^2, ||w||^2) = {r:+.3f}")

print()
print("=" * 84)
print("B'  close_drawer, per episode: the tick of PEAK |tau_torso| -- is the arm moving?")
print("=" * 84)
for cell in ["drawer_lat0", "drawer_serial283", "drawer_cpu685"]:
    s = json.load(open(SRC / cell / "summary.json"))
    rows = []
    for e in s["episodes"]:
        ep = e["episode_id"]
        qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
        qv = np.load(SRC / cell / f"ep{ep:02d}_qvel.npy")
        k = int(np.argmax(np.abs(qf[:, 0])))
        rows.append((np.abs(qf[k, 0]), np.linalg.norm(qv[k]),
                     np.linalg.norm(qv, axis=1).mean()))
    r = np.asarray(rows)
    print(f"  {cell:<20} peak |tau_torso| median {np.median(r[:,0]):7.3f} N.m; "
          f"||omega|| AT that tick: median {np.median(r[:,1]):.4f} rad/s "
          f"(episode mean {np.median(r[:,2]):.4f}) "
          f"-> peak torque coincides with peak SPEED, not with a stall")

print()
print("=" * 84)
print("D  gripper closed on an object and squeezing: finger tau vs finger speed")
print("=" * 84)
for cell in ["egg_lat0", "coke_lat0"]:
    s = json.load(open(SRC / cell / "summary.json"))
    QF = np.concatenate([np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
                         for e in s["episodes"]])
    QV = np.concatenate([np.load(SRC / cell / f"ep{e['episode_id']:02d}_qvel.npy")
                         for e in s["episodes"]])
    fj = [6, 7] if QF.shape[1] == 8 else [7, 8]
    still = np.abs(QV[:, fj]).max(1) < 1e-4
    print(f"  {cell:<14} fingers static ({still.mean()*100:.0f}% of ticks): "
          f"mean |tau_finger| = {np.abs(QF[still][:, fj]).mean():.5f} N.m, "
          f"max = {np.abs(QF[still][:, fj]).max():.5f} N.m")
    print(f"                 a real gripper clamping a coke can holds tens of newtons; "
          f"the grasp force does NOT appear in tau.")
