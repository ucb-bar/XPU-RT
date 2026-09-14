#!/usr/bin/env python3
"""Fine-grain sweep analysis: spoon 6-arm ladder + the google_robot probe.

Reports, per task:
  * per-arm marginal success with a NAIVE binomial CI and a DESIGN-CORRECTED CI
  * the paired (per-seed) difference vs the zero-latency arm

The design correction matters as much here as it did for the wide sweep: a seed
is a policy-noise replicate over the SAME 24 episode configs, so episodes within
a config are correlated and a binomial on n episodes overstates precision by
DEFF = 1 + (m-1)*ICC. See g5wide/RESULTS.txt section 1.
"""
from __future__ import annotations
import json, glob, math, collections, sys
from pathlib import Path

HERE = Path(__file__).parent
ARM_ORDER = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
LAT = {"lat0": 0.0, "pipe110": 117.7, "pipe200": 231.8,
       "serial283": 283.4, "fp32_555": 555.0, "cpu685": 684.8}

def load():
    rows = []
    for f in glob.glob(str(HERE / "runs" / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name           # <task>_<arm>_rng<seed>
        if "_rng" not in name: continue
        head, seed = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        # Reject pre-fix google_robot runs: they actuated on the 27 Hz fine grid
        # with 1/9-scaled deltas under a planner-interpolated controller and
        # scored 0/24 at ZERO latency (../GOOGLE_ROBOT_PORT.md). Only
        # --actuation native runs are valid for that embodiment.
        if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native":
            continue
        for ep in d.get("episodes", []):
            rows.append(dict(task=task, arm=arm, seed=int(seed),
                             ep=ep["episode_id"], ok=bool(ep["success"])))
    return rows

def icc_deff(rows):
    """ICC with the episode CONFIG as cluster, and the resulting design effect."""
    by = collections.defaultdict(list)
    for r in rows: by[r["ep"]].append(r["ok"])
    K = len(by)
    if K < 2: return 0.0, 1.0, len(rows)
    ns = [len(v) for v in by.values()]
    m = sum(ns) / K
    gm = sum(sum(v) for v in by.values()) / sum(ns)
    msb = sum(len(v) * (sum(v)/len(v) - gm) ** 2 for v in by.values()) / (K - 1)
    msw = sum(sum((x - sum(v)/len(v)) ** 2 for x in v) for v in by.values()) / max(sum(ns) - K, 1)
    icc = max((msb - msw) / (msb + (m - 1) * msw), 0.0) if (msb + (m-1)*msw) > 0 else 0.0
    deff = 1 + (m - 1) * icc
    return icc, deff, sum(ns) / deff

def ci(k, n, deff=1.0):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; ne = n / deff
    se = math.sqrt(max(p * (1 - p), 1e-12) / ne)
    return (100*p, 100*max(p - 1.96*se, 0.0), 100*min(p + 1.96*se, 1.0))

rows = load()
if not rows:
    sys.exit("no runs fetched yet")
for task in sorted({r["task"] for r in rows}):
    tr = [r for r in rows if r["task"] == task]
    print(f"\n{'='*92}\n{task}   ({len(tr)} episodes, "
          f"{len({r['seed'] for r in tr})} seeds, {len({r['ep'] for r in tr})} configs)\n{'='*92}")
    print(f"{'arm':11s} {'lat_ms':>7s} {'n':>5s} {'succ':>6s} "
          f"{'naive 95% CI':>18s} {'ICC':>6s} {'DEFF':>6s} {'n_eff':>6s} {'design 95% CI':>18s}")
    base = None
    for arm in ARM_ORDER:
        ar = [r for r in tr if r["arm"] == arm]
        if not ar: continue
        k, n = sum(r["ok"] for r in ar), len(ar)
        icc, deff, neff = icc_deff(ar)
        p, lo, hi = ci(k, n); _, dlo, dhi = ci(k, n, deff)
        print(f"{arm:11s} {LAT.get(arm,0):7.1f} {n:5d} {p:5.1f}% "
              f"[{lo:6.1f},{hi:6.1f}] {icc:6.3f} {deff:6.2f} {neff:6.0f} [{dlo:6.1f},{dhi:6.1f}]")
        if arm == "lat0": base = ar
    # paired per-seed difference vs the zero-latency arm
    if base:
        bs = collections.defaultdict(list)
        for r in base: bs[r["seed"]].append(r["ok"])
        print(f"\n  paired vs lat0 (per-seed, same configs & policy seed):")
        for arm in ARM_ORDER[1:]:
            ar = [r for r in tr if r["arm"] == arm]
            if not ar: continue
            a_s = collections.defaultdict(list)
            for r in ar: a_s[r["seed"]].append(r["ok"])
            common = sorted(set(a_s) & set(bs))
            if not common: continue
            d = [100*(sum(a_s[s])/len(a_s[s]) - sum(bs[s])/len(bs[s])) for s in common]
            mu = sum(d)/len(d)
            sd = math.sqrt(sum((x-mu)**2 for x in d)/max(len(d)-1,1))
            se = sd/math.sqrt(len(d)) if len(d) > 1 else 0.0
            worse = sum(1 for x in d if x < 0)
            # t, not z: with a handful of seeds the normal quantile understates the
            # interval badly (at 2 seeds t(1)=12.71 vs z=1.96, a 6.5x difference).
            TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                     7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
            df = len(d) - 1
            tc = TCRIT.get(df, 1.96) if df >= 1 else float("nan")
            lo, hi = (mu - tc*se, mu + tc*se) if df >= 1 else (float("nan"), float("nan"))
            flag = "  <-- 95% CI unusable, needs more seeds" if df < 3 else ""
            print(f"    {arm:11s} {mu:+6.1f} pts  95% CI [{lo:+7.1f},{hi:+7.1f}]  "
                  f"({worse}/{len(d)} seeds worse, df={df}){flag}")
