#!/usr/bin/env python3
"""Merge structurally-identical arms, and measure the harness noise floor.

Groups come from resolution_audit.py, which derives them from the HARNESS CODE
(snap_tick / apply_tick and the actuation grid), not from measured outcomes.
That is the right definition: two arms with the same (freshest-inference,
obs-snap-tick) sequence are one experiment by construction, so any difference in
their outcomes is harness nondeterminism (NONDETERMINISM.md), not an effect.

Deriving groups from OBSERVED equality -- as an earlier pass here did -- is
circular and also fails: nondeterminism breaks exact equality on ~1 seed in 10,
which silently splits a true group.

Three things follow:
  1. coverage is 39 design arms -> 33 (coke) / 28 (egg) / 28 (spoon) experiments;
  2. a group is ONE operating point whose seeds pool, which raises precision
     where the design was accidentally redundant;
  3. the within-group spread is a large-sample measurement of run-to-run noise,
     and every latency effect must be stated against it.

This variance component is SEPARATE from the ICC/design-effect correction, which
handles clustering by episode config. Both are reported; neither replaces the other.
"""
from __future__ import annotations
import json, glob, math, collections, sys, statistics
from pathlib import Path

HERE = Path(__file__).parent
FINE = HERE.parent / "g5fine"
ARMS = {}
for l in open(HERE / "arms.tsv"):
    n, lat, cad, _ = l.split("\t"); ARMS[n] = (float(lat), float(cad))
AUD = json.load(open(HERE / "resolution_audit.json"))
WIDOWX = ["egg", "spoon"]; GOOGLE = ["coke"]; TASKS = WIDOWX + GOOGLE
TCRIT = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
         9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,
         16:2.120,17:2.110,18:2.101,19:2.093,20:2.086}
def tcrit(df): return TCRIT.get(df, 1.96) if df >= 1 else float("nan")

def load(pattern, keep=None):
    rows = []
    for f in glob.glob(pattern):
        try: d = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name
        if "_rng" not in name: continue
        head, seed = name.rsplit("_rng", 1); task, arm = head.split("_", 1)
        if keep and arm not in keep: continue
        if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native": continue
        eps = d.get("episodes", [])
        rows.append(dict(task=task, arm=arm, seed=int(seed),
                         vec=tuple(int(e["success"]) for e in eps),
                         rate=100*sum(int(e["success"]) for e in eps)/max(len(eps),1),
                         n=len(eps),
                         act_age=sum(e.get("act_age_mean_ms") or 0 for e in eps)/max(len(eps),1),
                         eps=eps))
    return rows

runs = load(str(HERE/"runs"/"*"/"*"/"summary.json"), keep=set(ARMS))
base = load(str(FINE/"runs"/"*"/"*"/"summary.json"), keep={"lat0"})
if not runs: sys.exit("no grid runs fetched yet")

# ---- structural operating points -------------------------------------------
def ops_for(task):
    groups = [list(g) for g in AUD[task]["collapsing_groups"]]
    grouped = {a for g in groups for a in g}
    return groups + [[a] for a in ARMS if a not in grouped]

print("="*118)
print("COVERAGE  --  design arms vs DISTINCT EXPERIMENTS (tick-grid collapse, derived from harness code)")
print("="*118)
for t in TASKS:
    print(f"  {t:6s} ({'widowx, tick 40 ms' if t in WIDOWX else 'google, tick 37.037 ms'}): "
          f"{AUD[t]['n_design_arms']} design arms -> {AUD[t]['n_distinct_experiments']} distinct experiments "
          f"({len(AUD[t]['collapsing_groups'])} collapsing groups)")
print("  The widowx latency axis is BLUNTER than google's (28 vs 33) because its tick is coarser,")
print("  i.e. resolution is worst on exactly the two tasks where cadence matters most.")

# ---- harness noise floor ---------------------------------------------------
by = collections.defaultdict(dict)
for r in runs: by[(r["task"], r["arm"])][r["seed"]] = r

NSEED = max(collections.Counter(len(v) for v in by.values()).items(),
            key=lambda kv: kv[1])[0]          # modal seeds-per-cell, from the data


def noise_floor(task, B=40000, seed=0, nseed=10):
    """Empirical null for an `nseed`-seed PAIRED MEAN DIFFERENCE between two conditions.

    nseed MUST match the number of seeds the contrasts actually average. It was
    hardcoded to 10 while the cells carried 20, which made every band ~sqrt(2) too
    WIDE -- conservative, so it could only have hidden real effects, never manufacture
    one, but wrong either way.

    Why not a sigma multiplier: 89% of same-experiment pairs are bit-identical, so
    the per-seed difference is a spike at zero with rare large jumps, not Gaussian
    jitter. A normal-theory t(9)*sd band understates the tail (it gave +/-1.37
    where the empirical 97.5th percentile is +/-2.08). We therefore bootstrap.

    The pool is SYMMETRISED (+d and -d both included) because which arm of an
    identical pair is called "a" is arbitrary, so the null must be sign-symmetric;
    without this, 4 non-zero draws produce a spuriously one-sided interval.

    Note the pool is already a DIFFERENCE of two runs, so it carries both runs'
    noise -- no further sqrt(2) is applied.
    """
    import random
    diffs = []; ident = 0
    for g in AUD[task]["collapsing_groups"]:
        for i, a in enumerate(g):
            for b in g[i+1:]:
                sa, sb = by.get((task,a), {}), by.get((task,b), {})
                for sd_ in sorted(set(sa) & set(sb)):
                    if sa[sd_]["vec"] == sb[sd_]["vec"]: ident += 1
                    diffs.append(sa[sd_]["rate"] - sb[sd_]["rate"])
    if len(diffs) < 4: return None
    pool = diffs + [-d for d in diffs]
    rng = random.Random(seed)
    means = sorted(abs(sum(rng.choice(pool) for _ in range(nseed))/nseed) for _ in range(B))
    band = means[int(0.975*B)]
    nz = [abs(d) for d in diffs if d != 0]
    # NB: "differs in vector" > "differs in rate" -- nondeterminism can reshuffle WHICH
    # episodes succeed while leaving the count identical (seed 129 of the cadence-200
    # triple: three different vectors, all 11/24). A matching total is NOT evidence
    # that two runs are the same run.
    return dict(n_pairs=len(diffs), ident=ident, band=band, B=B,
                med_nz=(statistics.median(nz) if nz else 0.0),
                max_nz=(max(nz) if nz else 0.0), n_nz=len(nz))

print("\n"+"="*118)
print("HARNESS NOISE FLOOR  --  from arms that are the SAME experiment by construction")
print("="*118)
print(f"Quantity: the {NSEED}-seed PAIRED MEAN DIFFERENCE between two conditions -- exactly what every")
print("contrast in this report is. Band = 97.5th percentile of |that mean| under an EMPIRICAL,")
print("sign-symmetrised bootstrap of the observed same-experiment differences (not a sigma")
print("multiplier: the distribution is ~89% exact zeros with rare jumps, so normal theory is wrong).")
floor = {}
for task in TASKS:
    r = noise_floor(task, nseed=NSEED)
    if not r:
        print(f"  {task:6s}: too few redundant pairs yet -- NO floor; coke's is NOT applied here")
        continue
    floor[task] = r["band"]
    print(f"  {task:6s}: {r['n_pairs']} same-experiment run pairs; "
          f"{r['ident']} bit-identical ({100*r['ident']/r['n_pairs']:.0f}%); "
          f"{r['n_pairs']-r['ident']} differ in VECTOR, {r['n_nz']} differ in RATE")
    print(f"          conditional on differing: median |diff| {r['med_nz']:.2f} pts, max {r['max_nz']:.2f} pts")
    print(f"          => 95% NOISE BAND on a {NSEED}-seed paired mean: +/-{r['band']:.2f} pts "
          f"(empirical 97.5th pct, B={r['B']})")
print("\n  Floors are PER EMBODIMENT and never transferred: widowx replays each policy result over")
print("  several actuations vs google's <1 per result, so nondeterminism has different room to propagate.")
print("\n  MEASURED: the +C/2 hold applies to BOTH embodiments, not just google_robot.")
print("    google_robot -- the result waits for the 333 ms actuation grid;")
print("    widowx       -- the result is replayed as an action chunk spanning C ms, so the MEAN")
print("                    age over the chunk is again ~latency + C/2.")
print("    Across 60 egg runs, mean(hold - C/2) = -2.6 ms (sd 13.2). So a fixed-PREDICTED-age")
print("    cadence ladder is confounded on ALL THREE tasks, and measured act_age is the correct")
print("    x-axis everywhere. inf/cmd = act_ms/cadence on both embodiments (0.222 vs 40/180),")
print("    so ensemble depth is collinear with cadence on widowx too.")
print("  A contrast counts only if it clears its OWN task's band AND its CI excludes zero.")

# ---- merged operating points -----------------------------------------------
def icc_deff(eps_rows):
    bye = collections.defaultdict(list)
    for r in eps_rows:
        for e in r["eps"]: bye[e["episode_id"]].append(int(e["success"]))
    K = len(bye)
    if K < 2: return 0.0, 1.0
    ns=[len(v) for v in bye.values()]; m=sum(ns)/K
    gm=sum(sum(v) for v in bye.values())/sum(ns)
    msb=sum(len(v)*(sum(v)/len(v)-gm)**2 for v in bye.values())/(K-1)
    msw=sum(sum((x-sum(v)/len(v))**2 for x in v) for v in bye.values())/max(sum(ns)-K,1)
    den=msb+(m-1)*msw
    icc=max((msb-msw)/den,0.0) if den>0 else 0.0
    return icc, 1+(m-1)*icc

print("\n"+"="*118)
print("MERGED OPERATING POINTS  --  structural groups pooled; seeds averaged WITHIN a seed")
print("(replicate arms are the same experiment, so they reduce noise per seed; they do not add df)")
print("="*118)
merged = {}
for task in TASKS:
    trs = [r for r in runs if r["task"] == task]
    if not trs: continue
    bl = [r for r in base if r["task"] == task]
    blseed = {r["seed"]: r["rate"] for r in bl}
    print(f"\n{task}  ({'widowx' if task in WIDOWX else 'google_robot'})")
    print(f"{'operating point':34s} {'predAge':>16s} {'cad':>6s} {'measAge':>8s} {'reps':>4s} {'sd':>3s} "
          f"{'nEps':>6s} {'succ':>6s} {'design 95% CI':>15s} {'paired vs lat0':>16s} {'vs noise':>9s}")
    rowsout = []
    for g in sorted(ops_for(task), key=lambda g: (ARMS[g[0]][1], ARMS[g[0]][0])):
        per_seed = collections.defaultdict(list); allrows = []
        for a in g:
            for s, r in by.get((task,a), {}).items():
                per_seed[s].append(r["rate"]); allrows.append(r)
        if not per_seed: continue
        seeds = sorted(per_seed)
        vals = {s: sum(v)/len(v) for s, v in per_seed.items()}
        rate = sum(vals.values())/len(vals)
        nEps = sum(r["n"] for r in allrows)
        icc, deff = icc_deff(allrows)
        p = rate/100; se = math.sqrt(max(p*(1-p),1e-12)/((nEps/deff)))
        lo, hi = 100*max(p-1.96*se,0), 100*min(p+1.96*se,1)
        ma = sum(r["act_age"] for r in allrows)/len(allrows)
        ages = [ARMS[a][0] for a in g]
        agetxt = f"{min(ages):.0f}" if len(g)==1 else f"{min(ages):.0f}-{max(ages):.0f}"
        common = sorted(set(vals) & set(blseed))
        ptxt, vs = "", ""
        if len(common) >= 2:
            d = [vals[s]-blseed[s] for s in common]
            mu = sum(d)/len(d)
            sd = math.sqrt(sum((x-mu)**2 for x in d)/(len(d)-1)); sef = sd/math.sqrt(len(d))
            tc = tcrit(len(d)-1)
            ptxt = f"{mu:+6.1f} [{mu-tc*sef:+6.1f},{mu+tc*sef:+6.1f}]"
            if task in floor: vs = "NOISE" if abs(mu) < floor[task] else "clears"
        label = g[0] if len(g)==1 else f"{g[0]}+{len(g)-1}"
        print(f"{label:34s} {agetxt:>16s} {ARMS[g[0]][1]:6.1f} {ma:8.1f} {len(g):4d} {len(seeds):3d} "
              f"{nEps:6d} {rate:5.1f}% [{lo:5.1f},{hi:5.1f}] {ptxt:>16s} {vs:>9s}")
        rowsout.append((g, vals, rate, ma))
    merged[task] = rowsout
print("\n'vs noise' compares |paired mean vs lat0| to that task's 95% harness-noise band.")

# ============================================================================
# TREND TESTS AND MULTIPLICITY
# ============================================================================
# Rationale: the floor says this is a RARE-JUMP process (89% of same-experiment
# pairs bit-identical; conditional on differing, median jump 4.17 pts, max 12.50
# from only 4 non-zero draws). So a SINGLE pairwise difference is weak evidence --
# one jump manufactures 6-12 points. A MONOTONE ordering across a whole ladder is
# strong evidence, because independent jumps rarely line up in order. Every ladder
# therefore gets a trend statistic over ALL its points, and the endpoint contrast
# is reported beside it, not instead of it.
def spearman(xs, ys):
    n = len(xs)
    if n < 3: return None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else None

def ladder_trend(task, pts):
    """pts: list of (x, [arms]). Per-SEED Spearman rho across the ladder, then a
    paired t interval over seeds. Seed-respecting: each seed sees all k points."""
    seeds = None
    for _, g in pts:
        s = set.intersection(*[set(by.get((task,a), {})) for a in g]) if g else set()
        seeds = s if seeds is None else (seeds & s)
    seeds = sorted(seeds or [])
    if len(seeds) < 3 or len(pts) < 3: return None
    rhos = []
    for s in seeds:
        xs = [x for x, _ in pts]
        ys = [sum(by[(task,a)][s]["rate"] for a in g)/len(g) for _, g in pts]
        r = spearman(xs, ys)
        if r is not None: rhos.append(r)
    if len(rhos) < 3: return None
    m = sum(rhos)/len(rhos)
    sd = math.sqrt(sum((r-m)**2 for r in rhos)/(len(rhos)-1))
    se = sd/math.sqrt(len(rhos)); tc = tcrit(len(rhos)-1)
    tstat = m/se if se > 0 else float('inf')
    neg = sum(1 for r in rhos if r < 0); pos = sum(1 for r in rhos if r > 0)
    return dict(rho=m, lo=m-tc*se, hi=m+tc*se, n=len(rhos), neg=neg, pos=pos, t=tstat)

def tp_two_sided(t, df):
    """Two-sided p from Student t, via a continued-fraction incomplete beta."""
    if df <= 0: return float('nan')
    if not math.isfinite(t): return 0.0
    if t == 0: return 1.0
    x = df/(df+t*t)
    x = min(max(x, 1e-15), 1-1e-15)
    def betacf(a, b, x, itmax=200, eps=3e-12):
        qab, qap, qam = a+b, a+1.0, a-1.0
        c, d = 1.0, 1.0-qab*x/qap
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0/d; h = d
        for m in range(1, itmax+1):
            m2 = 2*m
            aa = m*(b-m)*x/((qam+m2)*(a+m2))
            d = 1.0+aa*d;  c = 1.0+aa/c
            if abs(d) < 1e-30: d = 1e-30
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0/d; h *= d*c
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d = 1.0+aa*d;  c = 1.0+aa/c
            if abs(d) < 1e-30: d = 1e-30
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0/d; de = d*c; h *= de
            if abs(de-1.0) < eps: break
        return h
    a, b = df/2.0, 0.5
    lbeta = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    if x < (a+1)/(a+b+2): ib = math.exp(a*math.log(x)+b*math.log(1-x)-lbeta)*betacf(a,b,x)/a
    else: ib = 1-math.exp(b*math.log(1-x)+a*math.log(x)-lbeta)*betacf(b,a,1-x)/b
    return max(min(ib, 1.0), 0.0)

def bh(pvals, q=0.05):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i]); m = len(pvals)
    thr = [0.0]*m; kmax = -1
    for r, i in enumerate(idx, start=1):
        if pvals[i] <= q*r/m: kmax = r
    keep = set(idx[:kmax]) if kmax > 0 else set()
    return keep, (q*kmax/m if kmax > 0 else 0.0)

def op_rate(task, g, s): return sum(by[(task,a)][s]["rate"] for a in g)/len(g)
def op_meas(task, g):
    v=[by[(task,a)][s]["act_age"] for a in g for s in by.get((task,a),{})]
    return sum(v)/len(v) if v else None

def endpoint(task, glo, ghi):
    seeds = sorted(set.intersection(*[set(by.get((task,a),{})) for a in glo+ghi]))
    if len(seeds) < 3: return None
    d = [op_rate(task,ghi,s)-op_rate(task,glo,s) for s in seeds]
    m = sum(d)/len(d); sd = math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1))
    se = sd/math.sqrt(len(d)); tc = tcrit(len(d)-1)
    t = m/se if se>0 else float('inf')
    return dict(mu=m, lo=m-tc*se, hi=m+tc*se, n=len(d), p=tp_two_sided(t, len(d)-1))

MAXJUMP = {}
for task in TASKS:
    r = noise_floor(task, nseed=NSEED)
    if r: MAXJUMP[task] = (r["max_nz"], r["n_nz"])

print("\n"+"="*118)
print("LADDERS  --  TREND first, endpoint contrast second")
print("="*118)
print("A single pairwise difference is weak evidence under a rare-jump noise process; a monotone")
print("ordering across a whole ladder is strong, because independent jumps rarely line up in order.")
print("rho = mean per-seed Spearman across the ladder (negative = success falls as x rises).")

ladders = []   # (task, kind, label, pts)
for task in TASKS:
    ops = ops_for(task)
    ops = [g for g in ops if all(by.get((task,a)) for a in g)]
    if not ops: continue
    bycad = collections.defaultdict(list)
    for g in ops: bycad[round(ARMS[g[0]][1])].append(g)
    for cad, gs in bycad.items():
        if len(gs) < 3: continue
        pts = sorted(((op_meas(task,g), g) for g in gs), key=lambda p: p[0])
        ladders.append((task, "latency@fixed-cadence", f"cadence {cad} ms", pts))
    bylat = collections.defaultdict(list)
    for g in ops: bylat[round(ARMS[g[0]][0]/8)*8].append(g)
    for lat, gs in bylat.items():
        if len(gs) < 3 or len({round(ARMS[g[0]][1]) for g in gs}) < 3: continue
        pts = sorted(((ARMS[g[0]][1], g) for g in gs), key=lambda p: p[0])
        ladders.append((task, "cadence@fixed-age", f"pred age ~{lat} ms", pts))

# primary = per task, the largest ladder of each kind
primary = set()
for task in TASKS:
    for kind in ("cadence@fixed-age", "latency@fixed-cadence"):
        cands = [l for l in ladders if l[0]==task and l[1]==kind]
        if cands: primary.add(id(max(cands, key=lambda l: len(l[3]))))

results = []
for task, kind, label, pts in ladders:
    tr = ladder_trend(task, pts); ep = endpoint(task, pts[0][1], pts[-1][1])
    if not tr or not ep: continue
    results.append(dict(task=task, kind=kind, label=label, npts=len(pts),
                        tr=tr, ep=ep, primary=id((task,kind,label,pts)) in primary))
# re-mark primary (ids change); recompute by matching largest per task/kind
for task in TASKS:
    for kind in ("cadence@fixed-age","latency@fixed-cadence"):
        c=[r for r in results if r["task"]==task and r["kind"]==kind]
        if c:
            best=max(c,key=lambda r:r["npts"])
            for r in c: r["primary"]=(r is best)

prim=[r for r in results if r["primary"]]
expl=[r for r in results if not r["primary"]]
if prim:
    prim_p = [tp_two_sided(r["tr"]["t"], r["tr"]["n"]-1) for r in prim]
    keep, thr = bh(prim_p)
for grp, name in ((prim,"PRIMARY (pre-specified: largest ladder of each kind per task)"),
                  (expl,"EXPLORATORY (all other ladders -- not confirmatory)")):
    if not grp: continue
    print(f"\n--- {name} ---")
    print(f"{'task':6s} {'kind':22s} {'ladder':16s} {'k':>2s} {'rho':>7s} {'rho 95% CI':>16s} "
          f"{'seeds -/+':>10s} | {'endpoint':>7s} {'95% CI':>16s} {'p':>7s} {'verdict':>28s}")
    for i, r in enumerate(grp):
        tr, ep = r["tr"], r["ep"]
        band = floor.get(r["task"])
        mx, nnz = MAXJUMP.get(r["task"], (None, 0))
        if band is None: verdict = "no floor for this task yet"
        elif abs(ep["mu"]) < band: verdict = "within noise band"
        elif mx and abs(ep["mu"]) <= mx:
            verdict = "clears band; < 1 jump" if not (tr["hi"]<0 or tr["lo"]>0) else "clears band; TREND carries it"
        else: verdict = "exceeds band and max jump"
        print(f"{r['task']:6s} {r['kind']:22s} {r['label']:16s} {r['npts']:2d} {tr['rho']:+7.3f} "
              f"[{tr['lo']:+6.2f},{tr['hi']:+6.2f}] {tr['neg']:4d}/{tr['pos']:<5d} | "
              f"{ep['mu']:+7.1f} [{ep['lo']:+6.1f},{ep['hi']:+6.1f}] {ep['p']:7.4f} {verdict:>28s}")

print(f"\nMULTIPLICITY: {len(results)} ladder tests in this family "
      f"({len(prim)} primary, {len(expl)} exploratory).")
if prim:
    print(f"  Benjamini-Hochberg at q=0.05 across the {len(prim)} PRIMARY trend tests: "
          f"{len(keep)} survive"
          + (f" (adjusted threshold p<={thr:.4f})." if keep else f"; smallest primary p={min(prim_p):.4f}."))
print("  Exploratory ladders are NOT multiplicity-controlled and are hypothesis-generating only.")
for task in TASKS:
    if task in MAXJUMP:
        mx, nnz = MAXJUMP[task]
        print(f"  {task}: noise floor PROVISIONAL -- tail estimated from only {nnz} non-zero "
              f"difference(s); observed max single jump {mx:.2f} pts, so the true 99th pct may be higher.")
