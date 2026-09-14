#!/usr/bin/env python3
"""AUDIT PASS 4 -- decompose the shipped INT sum tau^2 dt.

Reads energy_audit_cache.json (pass 1). Answers, per task x arm:
  1. how much of the metric is EPISODE LENGTH rather than instantaneous effort
  2. which joints carry it, and what the fingers/head contribute (they are in t2 but
     excluded from eff -- an inconsistency in the shipped code)
  3. what happens to the ARM RANKING under per-joint reweighting
     (a) 1/force_limit_i^2   (b) 1/urdf_effort_i^2  (c) arm joints only
  4. spike dominance: is any episode's t2 carried by one tick?
  5. saturation: does |tau| ever reach the configured force limit?
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).parent
C = json.load(open(HERE / "energy_audit_cache.json"))
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
med = lambda v: float(np.median(v))
get = lambda cell, k: np.array([e[k] for e in C[cell]["eps"]])

print("=" * 100)
print("1.  IS IT A CLOCK?   t2 vs episode DURATION, pooled across the 216 episodes of a task")
print("=" * 100)
print(f"{'task':<8}{'n':>5}{'r(t2,dur)':>12}{'r(log t2,log dur)':>20}"
      f"{'R^2 of t2 ~ k*dur':>20}{'dur range s':>16}")
for t in TASKS:
    d2, du = [], []
    for a in ARMS:
        d2 += list(get(f"{t}_{a}", "t2")); du += list(get(f"{t}_{a}", "dur_s"))
    d2, du = np.asarray(d2), np.asarray(du)
    k = (d2 * du).sum() / (du ** 2).sum()
    r2 = 1 - ((d2 - k * du) ** 2).sum() / ((d2 - d2.mean()) ** 2).sum()
    print(f"{t:<8}{len(d2):>5}{pearsonr(du,d2)[0]:>12.3f}"
          f"{pearsonr(np.log(du),np.log(d2))[0]:>20.3f}{r2:>20.3f}"
          f"{f'{du.min():.1f}-{du.max():.1f}':>16}")

print()
print("=" * 100)
print("2.  INTEGRAL vs RATE.  'x' = median episode integral / ideal-arm median (the")
print("    published number).  'rate x' = median of (t2/duration) / same for the ideal")
print("    arm -- i.e. the same quantity with mission length divided out.")
print("=" * 100)
for t in TASKS:
    b2 = med(get(f"{t}_lat0", "t2"))
    br = med(get(f"{t}_lat0", "t2") / get(f"{t}_lat0", "dur_s"))
    bd = med(get(f"{t}_lat0", "dur_s"))
    print(f"\n  {t}")
    print(f"    {'arm':<12}{'dur s':>8}{'dur x':>8}{'t2':>10}{'t2 x':>8}"
          f"{'rate':>10}{'rate x':>9}{'succ%':>7}")
    for a in ARMS:
        c = f"{t}_{a}"
        du = get(c, "dur_s"); t2 = get(c, "t2")
        print(f"    {a:<12}{med(du):>8.2f}{med(du)/bd:>8.2f}{med(t2):>10.1f}"
              f"{med(t2)/b2:>8.2f}{med(t2/du):>10.2f}{med(t2/du)/br:>9.2f}"
              f"{C[c]['sr']:>7.1f}")

print()
print("=" * 100)
print("3.  WHERE DOES t2 LIVE?  per-joint share of the summed integral (median episode,")
print("    ideal arm).  '*' marks a joint EXCLUDED from the omega^2 term but INCLUDED here.")
print("=" * 100)
for t in TASKS:
    c = f"{t}_lat0"
    nm = C[c]["names"]; dof = C[c]["dof"]
    A = np.array([e["t2_j"] for e in C[c]["eps"]])
    sh = A.sum(0) / A.sum()
    armn = 6 if dof == 8 else 7
    print(f"\n  {t} ({dof} dof)")
    for j, n in enumerate(nm):
        star = " " if j < armn else "*"
        print(f"    {star}{n:<20}{sh[j]*100:>8.3f}%   mean|tau| "
              f"{np.mean([e['tmean'][j] for e in C[c]['eps']]):>8.4f} N.m"
              f"   max|tau| {max(e['tmax'][j] for e in C[c]['eps']):>8.3f}"
              f"   flim {C[c]['flim'][j]:>5.0f}")
    print(f"    -> joints excluded from the omega^2 term contribute "
          f"{sh[armn:].sum()*100:.3f}% of t2")

print()
print("=" * 100)
print("4.  RANKING STABILITY under reweighting (9 arms, median episode).")
print("    t2      = SUM_i tau_i^2                (shipped)")
print("    t2_arm  = arm joints only")
print("    t2_norm = SUM_i (tau_i/force_limit_i)^2   -- dimensionless actuator effort")
print("    t2_urdf = SUM_i (tau_i/urdf_effort_i)^2")
print("=" * 100)
KEYS = ["t2", "t2_arm", "t2_norm", "t2_unorm"]
for t in TASKS:
    v = {k: np.array([med(get(f"{t}_{a}", k)) for a in ARMS]) for k in KEYS}
    print(f"\n  {t}")
    print("    " + f"{'arm':<12}" + "".join(f"{k+' x':>12}" for k in KEYS)
          + f"{'rank shipped':>14}{'rank norm':>11}")
    rk = {k: np.argsort(np.argsort(v[k])) + 1 for k in KEYS}
    for i, a in enumerate(ARMS):
        print(f"    {a:<12}" + "".join(f"{v[k][i]/v[k][0]:>12.2f}" for k in KEYS)
              + f"{rk['t2'][i]:>14d}{rk['t2_norm'][i]:>11d}")
    for k in KEYS[1:]:
        rho = spearmanr(v["t2"], v[k]).statistic
        print(f"      spearman(t2, {k}) = {rho:+.3f}   "
              f"max |rank shift| = {int(np.abs(rk['t2']-rk[k]).max())}")

print()
print("=" * 100)
print("5.  SPIKE DOMINANCE and SATURATION")
print("=" * 100)
print(f"{'cell':<22}{'peak-tick share':>18}{'top-1% share':>15}"
      f"{'max t2/med t2':>16}{'ticks at flim':>15}")
worst = []
for t in TASKS:
    for a in ARMS:
        c = f"{t}_{a}"
        ps = get(c, "peak_tick_share"); tp = get(c, "top1pct_share")
        t2 = get(c, "t2"); sat = get(c, "n_sat_flim").sum()
        worst.append((ps.max(), c, med(ps), med(tp), t2.max() / med(t2), sat))
for ps, c, mps, mtp, rat, sat in sorted(worst, reverse=True)[:10]:
    print(f"{c:<22}{mps*100:>10.2f}% (max {ps*100:.1f}%){mtp*100:>14.1f}%"
          f"{rat:>16.1f}{sat:>15d}")
print(f"\n  total ticks at or above the configured force_limit, ALL 864 episodes: "
      f"{sum(w[5] for w in worst)}")
tot = sum(sum(e['T'] for e in C[f'{t}_{a}']['eps']) for t in TASKS for a in ARMS)
print(f"  total logged ticks: {tot}")

print()
print("=" * 100)
print("6.  ALIASING: distinct sim states behind each logged tick")
print("=" * 100)
for t in TASKS:
    c = f"{t}_lat0"
    T = get(c, "T"); D = get(c, "distinct")
    print(f"  {t:<8} tick_ms={C[c]['tick_ms']:.2f} act_ms={C[c]['act_ms']:.2f} "
          f"act_every={C[c]['act_every']}  ticks/episode {med(T):.0f}, "
          f"DISTINCT states {med(D):.0f}  -> effective sample period "
          f"{C[c]['tick_ms']*med(T)/med(D):.1f} ms")
