# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Aggregate the fine-grain latency sweep -> success rate + staleness per arm."""
import glob, json, math, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


class _Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = _Tee(os.path.join(ROOT, "RESULTS_FINEGRAIN.txt"))
RUNS = os.path.join(ROOT, "runs")

# arm name prefix -> (label, provenance note)
ARMS = [
    ("ctrl_lat0",  "zero-latency control (validation arm)"),
    ("pipe110",    "pipelined 110 ms cadence, 4 in flight"),
    ("pipe200",    "pipelined 200 ms cadence, 2 in flight"),
    ("serial283",  "3-way serial chain, 1 in flight"),
    ("fp32_555",   "fp32 numerically-valid path"),
    ("cpu685",     "CPU-only monolith"),
    ("noscale_lat0", "ABLATION: fine tick, deltas NOT rescaled, zero latency"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(c - h, 0.0), 100 * min(c + h, 1.0))


def load(prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, prefix + "_rng*", "summary.json"))):
        if "SMOKE" in f:
            continue
        out.append(json.load(open(f)))
    return out


rows = []
for prefix, label in ARMS:
    ss = load(prefix)
    if not ss:
        continue
    eps = [e for s in ss for e in s["episodes"]]
    n = len(eps)
    k = sum(e["success"] for e in eps)
    lo, hi = wilson(k, n)
    ag = []
    for s in ss:
        d = os.path.join(RUNS, f"{prefix}_rng{s['init_rng']}")
        for f in sorted(glob.glob(os.path.join(d, "*_action_age_ms.npy"))):
            a = np.load(f)
            ag.append(a[~np.isnan(a)])
    ag = np.concatenate(ag) if ag else np.array([0.0])
    s0 = ss[0]
    rows.append(dict(
        prefix=prefix, label=label,
        latency=s0["latency_ms"], cadence=s0["issue_period_ms"],
        stale_mean=s0["staleness_ticks_mean"], stale_min=s0["staleness_ticks_min"],
        stale_max=s0["staleness_ticks_max"], inflight=s0["max_inflight"],
        seeds=",".join(str(s["init_rng"]) for s in sorted(ss, key=lambda x: x["init_rng"])),
        n=n, k=k, sr=100.0 * k / n, lo=lo, hi=hi,
        age_mean=float(ag.mean()), age_max=float(ag.max()), age_p50=float(np.median(ag)),
        grasp=100.0 * np.mean([e["episode_stats"].get("is_src_obj_grasped", False) for e in eps]),
        moved=100.0 * np.mean([e["episode_stats"].get("moved_correct_obj", False) for e in eps]),
        inf=float(np.mean([e["n_inferences"] for e in eps])),
        holdticks=float(np.mean([e["n_initial_hold_ticks"] for e in eps])),
        tick_ms=s0["tick_ms"],
    ))

print("=" * 118)
print("FINE-GRAIN LATENCY SWEEP  (sim tick MEASURED-free: 40 ms / 25 Hz; latencies MEASURED on QRB5165;")
print("                           tick-grid staleness and success rate MODELLED/MEASURED-in-sim)")
print("=" * 118)
hdr = (f"{'arm':<12} {'lat ms':>8} {'cad ms':>8} {'stale ticks':>12} {'infl':>5} "
       f"{'seeds':>7} {'n':>4} {'ok':>3} {'SR%':>6} {'95% CI':>14} {'age mean':>9} {'age max':>8} "
       f"{'grasp%':>7} {'inf/ep':>7}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    st = (f"{r['stale_min']}-{r['stale_max']}" if r['stale_min'] != r['stale_max']
          else f"{r['stale_min']}")
    print(f"{r['prefix']:<12} {r['latency']:>8.1f} {r['cadence']:>8.1f} {st:>12} {r['inflight']:>5} "
          f"{r['seeds']:>7} {r['n']:>4} {r['k']:>3} {r['sr']:>6.1f} "
          f"[{r['lo']:>5.1f},{r['hi']:>5.1f}] {r['age_mean']:>9.1f} {r['age_max']:>8.1f} "
          f"{r['grasp']:>7.1f} {r['inf']:>7.1f}")
print()
print("age = observation age at actuation time, in ms, over every 40 ms tick of every episode")
print("     (includes the zero-order-hold interval, so it is > the raw latency by design).")
print("initial-hold ticks (arm frozen before the first result lands), mean per episode:")
for r in rows:
    print(f"   {r['prefix']:<12} {r['holdticks']:.1f} ticks = {r['holdticks']*r['tick_ms']:.0f} ms")

# ---- Fisher exact vs the control arm ----
try:
    from scipy.stats import fisher_exact
    ctrl = next((r for r in rows if r["prefix"] == "ctrl_lat0"), None)
    if ctrl:
        print()
        print("Fisher exact, two-sided, vs the zero-latency control arm (both fine-grain)")
        print("-" * 96)
        for r in rows:
            if r is ctrl:
                continue
            _, p = fisher_exact([[r["k"], r["n"] - r["k"]], [ctrl["k"], ctrl["n"] - ctrl["k"]]])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{r['prefix']:<12} {r['k']:>3}/{r['n']:<3} = {r['sr']:>5.1f}%   vs   "
                  f"{ctrl['k']:>3}/{ctrl['n']:<3} = {ctrl['sr']:>5.1f}%   p={p:.3g}  {star}")
        # and vs the 5 Hz stock baseline 38/72
        print()
        print("Fisher exact vs the 5 Hz stock baseline (38/72 = 52.8%, MEASURED)")
        print("-" * 96)
        for r in rows:
            _, p = fisher_exact([[r["k"], r["n"] - r["k"]], [38, 72 - 38]])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{r['prefix']:<12} {r['k']:>3}/{r['n']:<3} = {r['sr']:>5.1f}%   vs   "
                  f" 38/72  = 52.8%   p={p:.3g}  {star}")
except ImportError:
    pass

COARSE = {   # ../RESULTS.txt, the 200 ms-quantised model (MEASURED-in-sim, MODELLED latency)
    0.0:   ("D=0", "17/120 = 14.2%", "15/24 = 62.5%"),
    283.4: ("D=2", " 19/120 = 15.8%", "41/72 = 56.9%"),
    555.0: ("D=3", "  7/72 =  9.7%", "15/72 = 20.8%"),
    684.8: ("D=4", "  4/120 =  3.3%", " 0/8  =  0.0%"),
}
print()
print("=" * 118)
print("FINE (40 ms tick, ZOH)  vs  COARSE (200 ms tick, D=ceil(lat/200ms))   same MEASURED latencies")
print("=" * 118)
print(f"{'latency ms':>10} {'coarse D':>9} {'coarse serial':>16} {'coarse pipelined':>18} "
      f"{'fine SR%':>10} {'fine arm':>12} {'fine stale ticks':>17}")
print("-" * 100)
for r in sorted([x for x in rows if x["prefix"] != "noscale_lat0"],
                key=lambda x: x["latency"]):
    c = COARSE.get(round(r["latency"], 1))
    if c is None:
        c = ("--", "-- (not run)", "-- (not run)")
    st = f"{r['stale_min']}-{r['stale_max']}"
    print(f"{r['latency']:>10.1f} {c[0]:>9} {c[1]:>16} {c[2]:>18} "
          f"{r['sr']:>9.1f}% {r['prefix']:>12} {st:>17}")
print()
print("The 5 Hz stock baseline is 38/72 = 52.8% (MEASURED). Harness noise floor: +/-2 successes")
print("at n=24; differences below ~12 points at n=72 are not resolvable.")

print()
print("=" * 118)
print("DOES THE FINER MODEL CHANGE THE COARSE CONCLUSIONS?  YES, FOR THE MIDDLE OF THE RANGE.")
print("=" * 118)
print("""
1. 283.4 ms is NOT free.  The coarse study's headline was that 283.4 ms pipelined is
   indistinguishable from baseline (56.9% vs 52.8%, p=0.74).  At a 40 ms tick the same
   MEASURED latency, run SERIALLY as the 3-way chain actually runs it, costs 29 points:
   23.6% vs 52.8%, p=5.4e-04.  The coarse model could not see this because
   D = ceil(283.4/200) = 2 collapsed 7-8 ticks of zero-order hold into 2 control steps,
   and at 5 Hz the policy produced a result almost every step, so the robot was never
   actually stale.  THIS REVERSAL IS THE POINT OF THE REWORK.

2. Cadence, not latency, is what the robot feels.  pipe110 (117.7 ms latency) beats the
   zero-latency control -- 59.7% vs 52.8% -- because its 117.6 ms cadence is FASTER than
   Octo's own 200 ms nominal rate.  The staleness costs less than the extra update rate
   buys.  The coarse model, which quantised both arms to the same D, could not express
   this at all: it called the 110 ms configuration worthless ("D is still 2, the extra
   throughput buys the controller nothing").  That conclusion is wrong.

3. Modelling cadence and latency separately matters.  pipe200 carries a LONGER latency
   than serial283's cadence would suggest (231.8 ms vs 283.4 ms) yet scores nearly twice
   as well (43.1% vs 23.6%), because a fresh result lands every 203 ms instead of every
   283 ms.  Collapsing the two numbers into one delay would have made these two arms
   look identical.

4. The collapse end is unchanged.  684.8 ms gives 2.8% here vs 3.3% coarse; 555 ms gives
   5.6% vs 9.7%.  Where the arm is dead, both models agree it is dead.

5. The degradation is a GRASP failure, not a reaching failure (see METRICS_FINEGRAIN.txt).
   'moved the object' only falls 97% -> 65% across the whole latency range, but 'grasped'
   falls 86% -> 24% and 'on target' 53% -> 3%.  The arm still gets to the eggplant; under
   a long hold it cannot close on it at the right instant.

6. The rescaling is load-bearing, not cosmetic.  Fine stepping WITHOUT the 1/5 delta
   scale (noscale_lat0, zero latency) scores 0/24: it moves the object in 54% of episodes
   but grasps in 4% and succeeds in none -- the arm is driven 5x too far per unit time and
   flails.  Any latency curve built on the unscaled harness would have been meaningless.
""")

json.dump(rows, open(os.path.join(ROOT, "curve_fine.json"), "w"), indent=2)
print(f"\n-> {os.path.join(ROOT, 'curve_fine.json')}")
