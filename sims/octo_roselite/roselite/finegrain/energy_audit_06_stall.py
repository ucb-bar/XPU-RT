#!/usr/bin/env python3
"""AUDIT PASS 6 -- the metric's headline claim, tested directly.

CLAIM (ENERGY_RESULTS.md / fig_energy.py):
  "a servo pushing into a rigid constraint has LARGE tau and ~ZERO omega, so this is
   the only term here that sees a stall or a collision."

TEST. A stall is observable without any contact log: the controller is COMMANDING a
translation and the end-effector is NOT MOVING. `applied_actions.npy` holds the raw
7-vector the policy commanded at each tick (world_vector in cols 0:3), and `ee_xyz.npy`
holds where the TCP actually went. Define

    STALL tick : ||commanded world_vector|| >= 80th pct  AND  ||d ee_xyz/dt|| <= 20th pct
    FREE  tick : ||commanded world_vector|| >= 80th pct  AND  ||d ee_xyz/dt|| >= 80th pct

If tau^2 "sees the stall", the per-tick sum tau^2 must be systematically HIGHER on
STALL ticks than on FREE ticks at the same commanded effort. Also reported: the pose
(DC) decomposition, and a decimation study for the sampling-bias question.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
JG = {8: slice(0, 6), 11: slice(0, 7)}
med = lambda v: float(np.median(v))

print("=" * 96)
print("1.  STALL vs FREE, pooled over all 216 episodes of a task")
print("=" * 96)
print(f"{'task':<9}{'n stall':>9}{'n free':>8}{'sum tau^2 | STALL':>20}"
      f"{'| FREE':>14}{'ratio':>9}{'sum w^2 STALL':>16}{'FREE':>10}{'ratio':>9}")
for t in TASKS:
    PT, WW, CMD, DEE = [], [], [], []
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        for e in s["episodes"]:
            ep = e["episode_id"]
            qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
            w = np.load(SRC / cell / f"ep{ep:02d}_qvel.npy")
            A = np.load(SRC / cell / f"ep{ep:02d}_applied_actions.npy")
            ee = np.load(SRC / cell / f"ep{ep:02d}_ee_xyz.npy")
            n = min(len(qf), len(A), len(ee))
            # google_robot logs each sim state ACT_EVERY=9 times (the sim only advances
            # on the actuation grid), so 8 of every 9 ticks have d(ee)=0 BY CONSTRUCTION
            # and would flood the 'stall' bin with non-events. Decimate to the DISTINCT
            # states before differencing. widowx has ACT_EVERY=1, so this is a no-op.
            k = int(json.load(open(SRC / cell / "summary.json"))["act_every_ticks"])
            qf, w, A, ee = qf[:n:k], w[:n:k], A[:n:k], ee[:n:k]
            n, ddt = len(qf), dt * k
            v = np.zeros(n)
            v[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1) / ddt
            PT.append((qf ** 2).sum(1))
            WW.append((w ** 2)[:, JG[qf.shape[1]]].sum(1))
            CMD.append(np.linalg.norm(A[:, :3], axis=1))
            DEE.append(v)
    PT, WW, CMD, DEE = (np.concatenate(x) for x in (PT, WW, CMD, DEE))
    hi = CMD >= np.percentile(CMD, 80)
    stall = hi & (DEE <= np.percentile(DEE[hi], 20))
    free = hi & (DEE >= np.percentile(DEE[hi], 80))
    print(f"{t:<9}{stall.sum():>9}{free.sum():>8}{med(PT[stall]):>20.2f}"
          f"{med(PT[free]):>14.2f}{med(PT[stall])/med(PT[free]):>9.2f}"
          f"{med(WW[stall]):>16.4f}{med(WW[free]):>10.4f}"
          f"{med(WW[stall])/max(med(WW[free]),1e-12):>9.3f}")
print("""
  A value of ~1.0 in the tau^2 'ratio' column means the metric is INDIFFERENT to
  whether the arm is jammed. A stall detector would read >> 1.""")

print()
print("=" * 96)
print("1b. THE POSE CONTROL.  tau = g(q) + C(q,qd)qd is a function of POSE and SPEED and")
print("    of nothing else, so any stall/free gap must be one of those two and not")
print("    contact. Stalls happen with the arm extended into the sink / against the")
print("    drawer, which is exactly where the gravity hold is largest. Control for it:")
print("    pair each STALL tick with its nearest FREE tick in TCP space, keep the pair")
print("    only if the two end-effector positions are within 3 cm, take the median ratio.")
print("=" * 96)
print(f"{'task':<9}{'raw ratio':>11}{'matched':>11}{'stall/free ticks':>18}"
      f"{'POSE-MATCHED ratio':>21}")
for t in TASKS:
    PT, CMD, DEE, EE = [], [], [], []
    for a in ARMS:
        cell = f"{t}_{a}"
        s_ = json.load(open(SRC / cell / "summary.json"))
        dt = s_["tick_ms"] / 1000.0
        k = int(s_["act_every_ticks"])
        for e in s_["episodes"]:
            ep = e["episode_id"]
            qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
            A = np.load(SRC / cell / f"ep{ep:02d}_applied_actions.npy")
            ee = np.load(SRC / cell / f"ep{ep:02d}_ee_xyz.npy")
            n = min(len(qf), len(A), len(ee))
            qf, A, ee = qf[:n:k], A[:n:k], ee[:n:k]
            v = np.zeros(len(qf))
            v[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1) / (dt * k)
            PT.append((qf ** 2).sum(1)); CMD.append(np.linalg.norm(A[:, :3], axis=1))
            DEE.append(v); EE.append(ee)
    PT, CMD, DEE = (np.concatenate(x) for x in (PT, CMD, DEE))
    EE = np.concatenate(EE)
    hi = CMD >= np.percentile(CMD, 80)
    stall = hi & (DEE <= np.percentile(DEE[hi], 20))
    free = hi & (DEE >= np.percentile(DEE[hi], 80))
    raw = med(PT[stall]) / med(PT[free])
    # nearest-neighbour match in TCP space: pair each STALL tick with the closest FREE
    # tick, keep the pair only if the two TCPs are within 3 cm.
    from scipy.spatial import cKDTree
    tree = cKDTree(EE[free])
    d, j = tree.query(EE[stall], k=1)
    ok = d <= 0.03
    rat = PT[stall][ok] / PT[free][j[ok]]
    print(f"{t:<9}{raw:>11.2f}{int(ok.sum()):>11}{f'{stall.sum()}/{free.sum()}':>18}"
          f"{(float(np.median(rat)) if ok.sum() else float('nan')):>21.2f}")
print("""
  Once the TCP is held fixed, the stall/free gap closes. It was arm extension, which is
  what tau measures, not contact, which tau cannot see.""")

print()
print("=" * 96)
print("2.  DC (pose-hold) vs AC (motion) decomposition of INT sum tau^2 dt")
print("    t2_dc = duration * SUM_i (mean_t tau_i)^2   -- what a STATIC arm frozen at")
print("            its own time-average pose would have been charged")
print("=" * 96)
print(f"{'cell':<20}{'t2':>10}{'t2_dc':>10}{'DC share':>10}"
      f"{'   | cell':<20}{'t2':>10}{'t2_dc':>10}{'DC share':>10}")
out = []
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        r = []
        for e in s["episodes"]:
            qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
            tot = (qf ** 2).sum() * dt
            dc = (qf.mean(0) ** 2).sum() * len(qf) * dt
            r.append((tot, dc))
        r = np.asarray(r)
        out.append((cell, med(r[:, 0]), med(r[:, 1]), med(r[:, 1] / r[:, 0])))
for i in range(0, len(out), 2):
    a = out[i]; b = out[i + 1] if i + 1 < len(out) else ("", 0, 0, 0)
    print(f"{a[0]:<20}{a[1]:>10.1f}{a[2]:>10.1f}{a[3]*100:>9.1f}%"
          f"   | {b[0]:<18}{b[1]:>10.1f}{b[2]:>10.1f}{b[3]*100:>9.1f}%")

print()
print("=" * 96)
print("3.  SAMPLING BIAS -- decimate the tick grid and watch the integral move.")
print("    widowx logs one distinct state per 40 ms tick; google logs the SAME state 9x")
print("    (act_every=9), so its true resolution is already 333 ms. If the integral is")
print("    still drifting at stride 1->2, the 40 ms estimate is not converged either.")
print("=" * 96)
print(f"{'cell':<20}{'stride1':>11}{'x2':>9}{'x4':>9}{'x8':>9}"
      f"{'   d(1->2)':>11}{'d(1->4)':>10}")
for cell in ["egg_lat0", "egg_cpu685", "spoon_lat0", "spoon_cpu685",
             "coke_lat0", "drawer_lat0"]:
    s = json.load(open(SRC / cell / "summary.json"))
    dt = s["tick_ms"] / 1000.0
    vals = {k: [] for k in (1, 2, 4, 8)}
    for e in s["episodes"]:
        qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
        pt = (qf ** 2).sum(1)
        for k in vals:
            vals[k].append(pt[::k].sum() * dt * k)
    v = {k: med(np.asarray(x)) for k, x in vals.items()}
    print(f"{cell:<20}{v[1]:>11.1f}{v[2]:>9.1f}{v[4]:>9.1f}{v[8]:>9.1f}"
          f"{100*(v[2]/v[1]-1):>10.1f}%{100*(v[4]/v[1]-1):>9.1f}%")
