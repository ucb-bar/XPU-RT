#!/usr/bin/env python3
"""AUDIT PASS 5 -- solver blow-ups, and how much of the metric they carry.

The traces contain single-tick events where |tau| jumps three orders of magnitude and
returns on the NEXT tick, accompanied by joint speeds many times the URDF velocity
limit (3.14 rad/s on every widowx revolute joint). Those are PhysX/inverse-dynamics
transients, not contacts: a contact reaction is sustained for as long as the arm is
pushing, a solver transient lasts exactly one tick.

Definition used here (deliberately conservative):
    a tick is FLAGGED if its sum tau^2 exceeds 25x the episode median AND both
    neighbouring ticks are below 5x it -- an excursion exactly one tick long.
A contact reaction is sustained, so this shape cannot flag one.

Reports, per cell: how many episodes contain one, and what the published median /
box-plot look like once flagged ticks are dropped.
"""
import json
from pathlib import Path
import numpy as np

def flag_transients(pt, thr=25.0, nb=5.0):
    """Isolated single-tick excursions in sum_i tau_i^2.

    A tick is flagged when its sum tau^2 exceeds `thr` x the episode median AND BOTH
    neighbouring ticks are below `nb` x the median -- i.e. the excursion lasts exactly
    one tick and the signal is back to baseline immediately before and after. A contact
    reaction is sustained while the arm pushes, so this cannot flag one; a PhysX
    inverse-dynamics transient is exactly this shape. Verified by eye against
    egg_fp32_555 ep07 t=425 (tau 1.6 -> 608 -> 1.7 N.m) and egg_p130w275 ep16 t=389.
    """
    import numpy as np
    m = float(np.median(pt))
    if m <= 0:
        return np.zeros(len(pt), bool)
    big = pt > thr * m
    prev = np.r_[True, pt[:-1] < nb * m]
    nxt = np.r_[pt[1:] < nb * m, True]
    return big & prev & nxt

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
VLIM = {8:  np.array([np.pi]*6 + [1.0, 1.0]),
        11: np.array([2., 2., 2., 2.5, 3., 3., 3., 1.3, 1.3, 2., 2.])}
JG = {8: slice(0, 6), 11: slice(0, 7)}
med = lambda v: float(np.median(v))

rows = {}
print("=" * 104)
print("Per-cell: flagged (solver-transient) ticks, and the metric with vs without them")
print("=" * 104)
print(f"{'cell':<20}{'eps w/ flag':>12}{'flag ticks':>11}{'med t2':>10}{'med t2*':>10}"
      f"{'d%':>7}{'max t2':>11}{'max t2*':>10}{'med eff':>10}{'med eff*':>10}{'d%':>7}")
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        T2, T2c, EF, EFc, nflag, nep = [], [], [], [], 0, 0
        for e in s["episodes"]:
            ep = e["episode_id"]
            qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
            w = np.load(SRC / cell / f"ep{ep:02d}_qvel.npy")
            dof = qf.shape[1]
            pt = (qf ** 2).sum(1)
            bad = flag_transients(pt)
            nflag += int(bad.sum()); nep += int(bad.any())
            T2.append(pt.sum() * dt); T2c.append(pt[~bad].sum() * dt)
            wa = (w ** 2)[:, JG[dof]].sum(1)
            EF.append(wa.sum() * dt); EFc.append(wa[~bad].sum() * dt)
        T2, T2c, EF, EFc = map(np.asarray, (T2, T2c, EF, EFc))
        rows[cell] = dict(t2=T2, t2c=T2c, eff=EF, effc=EFc, nep=nep, nflag=nflag)
        print(f"{cell:<20}{nep:>7}/{len(T2):<4}{nflag:>11}{med(T2):>10.1f}"
              f"{med(T2c):>10.1f}{100*(med(T2c)/med(T2)-1):>7.1f}{T2.max():>11.1f}"
              f"{T2c.max():>10.1f}{med(EF):>10.2f}{med(EFc):>10.2f}"
              f"{100*(med(EFc)/med(EF)-1):>7.1f}")

print()
print("=" * 104)
print("PUBLISHED RATIOS vs. DE-SPIKED RATIOS  (median episode / ideal-arm median)")
print("=" * 104)
for t in TASKS:
    b, bc = med(rows[f"{t}_lat0"]["t2"]), med(rows[f"{t}_lat0"]["t2c"])
    be, bec = med(rows[f"{t}_lat0"]["eff"]), med(rows[f"{t}_lat0"]["effc"])
    print(f"\n  {t:<8}{'arm':<12}{'t2 x':>9}{'t2* x':>9}{'eff x':>9}{'eff* x':>9}")
    for a in ARMS:
        r = rows[f"{t}_{a}"]
        print(f"  {'':<8}{a:<12}{med(r['t2'])/b:>9.2f}{med(r['t2c'])/bc:>9.2f}"
              f"{med(r['eff'])/be:>9.2f}{med(r['effc'])/bec:>9.2f}")

print()
print("=" * 104)
print("How much of the POOLED (mean) metric is carried by flagged ticks?")
print("=" * 104)
for t in TASKS:
    a1 = sum(rows[f"{t}_{a}"]["t2"].sum() for a in ARMS)
    a2 = sum(rows[f"{t}_{a}"]["t2c"].sum() for a in ARMS)
    e1 = sum(rows[f"{t}_{a}"]["eff"].sum() for a in ARMS)
    e2 = sum(rows[f"{t}_{a}"]["effc"].sum() for a in ARMS)
    nf = sum(rows[f"{t}_{a}"]["nflag"] for a in ARMS)
    ne = sum(rows[f"{t}_{a}"]["nep"] for a in ARMS)
    print(f"  {t:<8} flagged ticks {nf:>5} in {ne:>3}/216 episodes -> they carry "
          f"{100*(1-a2/a1):>5.1f}% of all INT tau^2 and {100*(1-e2/e1):>5.1f}% of all "
          f"INT omega^2")

print()
print("=" * 104)
print("Does the omega^2 term's finger-exclusion claim hold? (fig_energy: 'unweighted")
print("they dominate by ~100x')")
print("=" * 104)
for t in TASKS:
    cell = f"{t}_lat0"
    s = json.load(open(SRC / cell / "summary.json"))
    dt = s["tick_ms"] / 1000.0
    arm, oth = [], []
    for e in s["episodes"]:
        w = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qvel.npy")
        dof = w.shape[1]
        v = (w ** 2).sum(0) * dt
        arm.append(v[JG[dof]].sum()); oth.append(v[JG[dof].stop:].sum())
    print(f"  {t:<8} INT omega^2: arm {np.median(arm):>9.2f}   "
          f"excluded joints {np.median(oth):>9.2f}   ratio "
          f"{np.median(oth)/max(np.median(arm),1e-9):>7.1f}x")
