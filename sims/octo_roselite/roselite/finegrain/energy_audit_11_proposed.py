#!/usr/bin/env python3
"""AUDIT PASS 11 -- the replacement statistic, computed on the existing traces.

This is what the current data CAN honestly support. It is NOT energy; it is a
dimensionless, duration-normalised, per-joint-rated GRAVITY-HOLD EFFORT:

    GHE(a,e) = (1/T_ae) * INT_0^T_ae  SUM_{i in arm}  ( tau_i(t) / tauhat_i )^2  dt

with
  * tau  = robot.get_qf(), i.e. g(q) + C(q,qd)qd  -- named for what it is
  * tauhat_i = the real servo stall torque for that joint on widowx (sourced, see
    below); 1.0 on google_robot, where no rating is published by anyone
  * arm joints only, consistent with the omega^2 term (fingers/head excluded from BOTH)
  * isolated single-tick solver transients removed
  * divided by episode duration, so mission length is reported separately, not smuggled in
  * PAIRED per episode against the ideal arm -- every cell of a task shares init_rng=100
    and episode ids, so episode e starts from the same scene in every arm

Reported as the median of the per-episode ratios with a paired-bootstrap 95% CI.
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(0)
HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
JG = {8: slice(0, 6), 11: slice(0, 7)}
# tauhat: the best available per-joint torque RATING, in qf column order.
#
# widowx -- the REAL WidowX-250 S servo stall torque at 12.0 V (ROBOTIS e-Manual;
#   Trossen docs/_data/servos_wx250s.csv gives the per-joint assignment). shoulder and
#   elbow are DUAL (a shadow servo mirrors the master via Secondary_ID in
#   interbotix_xsarm_control/config/wx250s.yaml), so their rating is 2x4.1 N.m.
#     waist XM430 4.1 | shoulder 2xXM430 8.2 | elbow 2xXM430 8.2
#     forearm_roll XM430 4.1 | wrist_angle XM430 4.1 | wrist_rotate XL430 1.5
#   NOTE this is NOT the agent-config force_limit ([200,200,100,100,100,100] N.m), which
#   is a system-ID handle 24-49x the physical rating, and NOT the URDF <limit effort>
#   ([10,20,15,2,5,1]), which is 2-2.5x the real servo on the big joints.
#
# google_robot -- NO rating is published for any joint (RT-1 gives no actuator spec; the
#   SimplerEnv URDF sets effort="10.0" uniformly on every joint including the wheels, a
#   placeholder). tauhat is therefore set to 1.0 and the google numbers are an UNWEIGHTED
#   arm-joint sum. They are not comparable to the widowx column -- they never were.
UEFF = {8: np.array([4.1, 8.2, 8.2, 4.1, 4.1, 1.5, 1., 1.]),
        11: np.ones(11)}
B = 20000


def flag_transients(pt, thr=25.0, nb=5.0):
    m = float(np.median(pt))
    if m <= 0:
        return np.zeros(len(pt), bool)
    return (pt > thr * m) & np.r_[True, pt[:-1] < nb * m] & np.r_[pt[1:] < nb * m, True]


D = {}
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        rec = {}
        for e in s["episodes"]:
            ep = e["episode_id"]
            qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
            dof = qf.shape[1]
            n = (qf / UEFF[dof]) ** 2
            n[:, JG[dof].stop:] = 0.0                    # arm joints only
            pt = n.sum(1)
            keep = ~flag_transients((qf ** 2).sum(1))
            rec[ep] = (float(pt[keep].mean()), len(qf) * dt, e["success"])
        D[cell] = dict(rec=rec, sr=100.0 * s["n_success"] / s["n_episodes"])

print("=" * 100)
print("PROPOSED:  gravity-hold effort  GHE = <SUM_arm (tau_i/urdf_effort_i)^2>_t")
print("           dimensionless, per second of mission, paired against the ideal arm")
print("=" * 100)
for t in TASKS:
    base = D[f"{t}_lat0"]["rec"]
    print(f"\n  {t:<8}{'arm':<12}{'GHE (abs)':>12}{'paired x [95% CI]':>26}"
          f"{'median dur s':>14}{'succ%':>8}")
    for a in ARMS:
        c = D[f"{t}_{a}"]["rec"]
        ks = sorted(set(c) & set(base))
        r = np.array([c[k][0] / base[k][0] for k in ks])
        i = rng.integers(0, len(r), (B, len(r)))
        lo, hi = np.percentile(np.median(r[i], 1), [2.5, 97.5])
        sig = "*" if (lo > 1.0 or hi < 1.0) else " "
        print(f"  {'':<8}{a:<12}{np.median([c[k][0] for k in ks]):>12.4f}"
              f"{f'{np.median(r):.3f} [{lo:.3f},{hi:.3f}]{sig}':>26}"
              f"{np.median([c[k][1] for k in ks]):>14.2f}{D[f'{t}_{a}']['sr']:>8.1f}")
print("\n  * = 95% CI excludes 1.00")
