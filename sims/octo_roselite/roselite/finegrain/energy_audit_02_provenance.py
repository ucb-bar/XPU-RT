#!/usr/bin/env python3
"""AUDIT PASS 2 -- WHAT IS get_qf() ACTUALLY RETURNING?

The metric is documented as "joint forces ... a copper-loss (I^2R) proxy ... the only
term that sees a stall or a collision". That claim is testable from the traces alone.

Five independent tests, each of which the "actuator torque" reading must pass:

 T1  ZERO AT RESET.        base_agent.reset() calls set_qf(zeros). If get_qf() reported
                           the real joint force, tick 0 would still show gravity load.
 T2  VERTICAL-AXIS JOINTS. A revolute joint whose axis is world-vertical carries ZERO
                           gravity moment but a perfectly ordinary actuator torque
                           (it is the joint that swings the whole arm in azimuth).
                           widowx `waist`, google `joint_torso` and `joint_head_pan`
                           are all axis="0 0 1".
 T3  MAGNITUDE vs LIMIT.   Compare max|tau| to the configured force_limit and to the
                           URDF <limit effort>.
 T4  GRAVITY RECONSTRUCTION. Predict the shoulder-joint gravity hold torque from the
                           logged link frames + masses and regress it on the logged
                           tau. Pure gravity compensation => R^2 ~ 1.
 T5  IMPULSIVENESS.        A contact reaction is impulsive. Gravity compensation is a
                           smooth function of pose. Compare the tick-to-tick increment
                           distribution against the level.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
G = 9.81

def load(cell, ep):
    d = SRC / cell
    return (np.load(d / f"ep{ep:02d}_qf.npy"), np.load(d / f"ep{ep:02d}_qvel.npy"),
            np.load(d / f"ep{ep:02d}_link_com.npy"),
            json.load(open(d / f"ep{ep:02d}_bodies.json")),
            np.load(d / f"ep{ep:02d}_ee_xyz.npy"))

WIDOWX = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
          "wrist_rotate", "left_finger", "right_finger"]
GOOG = ["joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow",
        "joint_forearm", "joint_wrist", "joint_gripper", "joint_finger_right",
        "joint_finger_left", "joint_head_pan", "joint_head_tilt"]
VERT = {8: [0], 11: [0, 2, 4, 6, 9]}      # axis = 0 0 1 in the URDF
FLIM = {8: [200., 200., 100., 100., 100., 100., 60., 60.],
        11: [300., 300., 100., 100., 100., 100., 100., 60., 60., 100., 100.]}

print("=" * 78)
print("T1/T2/T3  per-joint |tau| statistics, pooled over all 24 episodes of a cell")
print("=" * 78)
for cell in ["egg_lat0", "spoon_cpu685", "coke_lat0", "drawer_serial283"]:
    s = json.load(open(SRC / cell / "summary.json"))
    QF = [np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy") for e in s["episodes"]]
    dof = QF[0].shape[1]
    nm = WIDOWX if dof == 8 else GOOG
    allqf = np.concatenate(QF)
    t0 = np.stack([q[0] for q in QF])
    print(f"\n--- {cell}  ({len(QF)} eps, {allqf.shape[0]} ticks, dof={dof})")
    print(f"    T1 tick-0 |tau| max over episodes/joints = {np.abs(t0).max():.3e}   "
          f"(reset() sets qf:=0; a real joint force would show the gravity load)")
    print(f"    {'joint':<20}{'axis':<8}{'max|t|':>10}{'mean|t|':>10}"
          f"{'flim':>8}{'max/flim':>10}")
    for j in range(dof):
        ax = "VERT" if j in VERT[dof] else "horiz"
        mx = np.abs(allqf[:, j]).max()
        print(f"    {nm[j]:<20}{ax:<8}{mx:>10.4f}{np.abs(allqf[:,j]).mean():>10.4f}"
              f"{FLIM[dof][j]:>8.0f}{mx/FLIM[dof][j]:>10.5f}")
    v = [j for j in VERT[dof]]
    print(f"    T2 vertical-axis joints {[nm[j] for j in v]}: "
          f"max|tau| = {np.abs(allqf[:, v]).max():.4e}")
    print(f"    T3 largest |tau| anywhere = {np.abs(allqf).max():.3f} N.m; "
          f"largest force_limit = {max(FLIM[dof]):.0f} N.m  "
          f"=> {np.abs(allqf).max()/max(FLIM[dof])*100:.3f}% of limit")

print()
print("=" * 78)
print("T4  GRAVITY RECONSTRUCTION -- widowx `shoulder`")
print("=" * 78)
print("""The shoulder axis is 0 1 0 in the child frame, i.e. horizontal, perpendicular
to the arm plane. For gravity F_i = (0,0,-m_i g) applied at r_i measured from the axis
point, the moment about that axis is  m_i g (r_x a_y - r_y a_x), and with the axis unit
vector a = (-sin th, cos th, 0) for waist angle th this is  m_i g * rho_i, where rho_i is
the horizontal distance from the shoulder axis along the arm's own azimuth. So

    tau_grav_shoulder = g * SUM_{links distal to the shoulder} m_i * rho_i

The traces log every link's FRAME origin (`link_com.npy`; note that SAPIEN's
Link.get_pose() is the link frame, and in URDF a child link frame sits ON its parent
joint's axis) plus `link_mass`, so rho_i is directly computable. If get_qf() is the
gravity-compensation feed-forward, a one-parameter regression tau ~ a * S must fit
near-perfectly.""")
for cell, ep in [("egg_lat0", 0), ("egg_cpu685", 3), ("spoon_lat0", 1),
                 ("spoon_serial283", 5)]:
    qf, w, lc, b, ee = load(cell, ep)
    m = np.asarray(b["link_mass"]); nm = b["link_names"]
    i_sh = nm.index("upper_arm_link")     # child of the `shoulder` joint -> axis point
    i_base = nm.index("base_link")
    dist = list(range(i_sh, len(nm)))     # upper_arm and everything distal
    p = lc[:, i_sh, :]                                     # axis point, per tick
    az = ee[:, :2] - p[:, :2]                              # arm azimuth, horizontal
    az = az / np.maximum(np.linalg.norm(az, axis=1, keepdims=True), 1e-9)
    rho = ((lc[:, dist, :2] - p[:, None, :2]) * az[:, None, :]).sum(-1)  # (T, ndist)
    S = G * (rho * m[dist]).sum(1)
    tau = qf[:, 1]
    ok = np.isfinite(S) & np.isfinite(tau)
    A = np.vstack([S[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, tau[ok], rcond=None)
    pred = A @ coef
    r2 = 1 - ((tau[ok] - pred) ** 2).sum() / ((tau[ok] - tau[ok].mean()) ** 2).sum()
    print(f"  {cell:<20} ep{ep:02d}  T={len(qf):4d}  "
          f"tau = {coef[0]:+.3f}*S {coef[1]:+.3f}   R^2 = {r2:.4f}   "
          f"resid RMS {np.sqrt(((tau[ok]-pred)**2).mean()):.4f} N.m "
          f"(signal RMS {np.sqrt((tau[ok]**2).mean()):.3f})")

print()
print("=" * 78)
print("T5  IMPULSIVENESS -- is there anything in tau that looks like a contact?")
print("=" * 78)
for cell in ["egg_lat0", "egg_cpu685", "spoon_cpu685", "coke_lat0", "drawer_lat0"]:
    s = json.load(open(SRC / cell / "summary.json"))
    r = []
    for e in s["episodes"]:
        qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
        d = np.abs(np.diff(qf, axis=0))
        d = d[d > 1e-12]
        lev = np.abs(qf).max()
        if d.size:
            r.append((d.max() / max(lev, 1e-9), np.median(d)))
    r = np.asarray(r)
    print(f"  {cell:<20} max per-tick |d tau| as a fraction of the episode's peak "
          f"|tau|: median {np.median(r[:,0]):.3f}  max {r[:,0].max():.3f}")
