#!/usr/bin/env python3
"""Closed-loop analysis of the 39-point CP-SAT operating-point plane.

TWO X-AXES, AND THEY ARE NOT THE SAME AXIS
------------------------------------------
The schedule's `latency_ms` is the age we DESIGNED. It is not the age the policy
experienced. google_robot actuates on a 3 Hz native grid (act_ms = 333.3), so a
result produced at cadence C waits in the hold buffer ~C/2 before the robot can
consume it: effective age ~ schedule_latency + C/2. Every summary records the
age actually measured at actuation (`act_age_mean_ms`), so we regress on that
and report the predicted age only as the design coordinate.

Consequence for the ladders: a "fixed predicted age" ladder is NOT a fixed
effective-age ladder (the hold term grows with cadence), and a "fixed cadence"
ladder does vary effective age one-for-one. Both spans are printed so the
confound stays visible instead of being asserted away.

SECOND CHANNEL: stock action ensembling (finegrain_eval.py:566, ActionEnsembler
(CHUNK, 0.0), applied at :606) averages EVERY inference chunk, whether or not the
robot could act on it, so ensemble depth scales with inferences-per-command.
inf/cmd is reported per cell and entered as a regressor for the google tasks.

EMBODIMENTS ARE KEPT SEPARATE, but note the +C/2 hold applies to BOTH: widowx replays each
inf/cmd < 1, so neither the hold-buffer nor the ensemble-depth confound applies
there and the cadence axis means what it was designed to mean.

All (age, cadence) DESIGN values are PREDICTED from the schedule JSON, never
board-measured.
"""
from __future__ import annotations
import json, glob, math, collections, sys
from pathlib import Path

HERE = Path(__file__).parent
FINE = HERE.parent / "g5fine"
ARMS = {}
for l in open(HERE / "arms.tsv"):
    n, lat, cad, _ = l.split("\t"); ARMS[n] = (float(lat), float(cad))

WIDOWX = ["egg", "spoon"]
GOOGLE = ["coke"]
TASKS  = WIDOWX + GOOGLE

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
        head, seed = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        if keep and arm not in keep: continue
        if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native":
            continue
        for ep in d.get("episodes", []):
            ni, na = ep.get("n_inferences"), ep.get("n_actuations")
            rows.append(dict(task=task, arm=arm, seed=int(seed), ep=ep["episode_id"],
                             ok=bool(ep["success"]),
                             act_age=ep.get("act_age_mean_ms"),
                             infcmd=(ni/na if (ni and na) else None),
                             setup=d.get("policy_setup"), act_ms=d.get("act_ms")))
    return rows

def icc_deff(rows):
    by = collections.defaultdict(list)
    for r in rows: by[r["ep"]].append(r["ok"])
    K = len(by)
    if K < 2: return 0.0, 1.0, len(rows)
    ns = [len(v) for v in by.values()]; m = sum(ns)/K
    gm = sum(sum(v) for v in by.values())/sum(ns)
    msb = sum(len(v)*(sum(v)/len(v)-gm)**2 for v in by.values())/(K-1)
    msw = sum(sum((x-sum(v)/len(v))**2 for x in v) for v in by.values())/max(sum(ns)-K,1)
    den = msb+(m-1)*msw
    icc = max((msb-msw)/den, 0.0) if den > 0 else 0.0
    return icc, 1+(m-1)*icc, sum(ns)/(1+(m-1)*icc)

def ci(k, n, deff=1.0):
    if n == 0: return (0.0,0.0,0.0)
    p = k/n; se = math.sqrt(max(p*(1-p),1e-12)/(n/deff))
    return (100*p, 100*max(p-1.96*se,0.0), 100*min(p+1.96*se,1.0))

def paired(a_rows, b_rows):
    A = collections.defaultdict(list); B = collections.defaultdict(list)
    for r in a_rows: A[r["seed"]].append(r["ok"])
    for r in b_rows: B[r["seed"]].append(r["ok"])
    common = sorted(set(A) & set(B))
    if len(common) < 2: return None
    d = [100*(sum(A[s])/len(A[s]) - sum(B[s])/len(B[s])) for s in common]
    mu = sum(d)/len(d)
    sd = math.sqrt(sum((x-mu)**2 for x in d)/(len(d)-1))
    se = sd/math.sqrt(len(d)); tc = tcrit(len(d)-1)
    return (mu, mu-tc*se, mu+tc*se, len(d), sum(1 for x in d if x < 0))

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def ols_cluster(X, y, cl):
    k = len(X[0]); n = len(y)
    XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a]*y[i] for i in range(n)) for a in range(k)]
    M = [row[:]+[1.0 if i==j else 0.0 for j in range(k)] for i,row in enumerate(XtX)]
    for c in range(k):
        p = max(range(c,k), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12: return None
        M[c], M[p] = M[p], M[c]
        d = M[c][c]; M[c] = [v/d for v in M[c]]
        for r in range(k):
            if r != c and M[r][c]:
                f = M[r][c]; M[r] = [v-f*w for v,w in zip(M[r], M[c])]
    inv = [row[k:] for row in M]
    beta = [sum(inv[a][b]*Xty[b] for b in range(k)) for a in range(k)]
    res = [y[i]-sum(X[i][a]*beta[a] for a in range(k)) for i in range(n)]
    groups = collections.defaultdict(list)
    for i,g in enumerate(cl): groups[g].append(i)
    meat = [[0.0]*k for _ in range(k)]
    for idx in groups.values():
        s = [sum(X[i][a]*res[i] for i in idx) for a in range(k)]
        for a in range(k):
            for b in range(k): meat[a][b] += s[a]*s[b]
    G = len(groups); scale = G/max(G-1,1) * (n-1)/max(n-k,1)
    cov = [[scale*sum(inv[a][u]*meat[u][v]*inv[v][b] for u in range(k) for v in range(k))
            for b in range(k)] for a in range(k)]
    return beta, [math.sqrt(max(cov[a][a],0.0)) for a in range(k)], G

def pearson(x, y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    if sx==0 or sy==0: return float('nan')
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)

grid = load(str(HERE/"runs"/"*"/"*"/"summary.json"), keep=set(ARMS))
base = load(str(FINE/"runs"/"*"/"*"/"summary.json"), keep={"lat0"})
rows = grid + base
if not grid: sys.exit("no grid runs fetched yet")
order = sorted({r["arm"] for r in grid}, key=lambda a: (ARMS[a][1], ARMS[a][0]))

print(f"grid episodes: {len(grid)}   arms: {len({r['arm'] for r in grid})}/39")
print("DESIGN (age, cadence) values are PREDICTED from the schedule, NOT board-measured.")
print("MEASURED age is act_age_mean_ms, the age at actuation recorded by the sim.")

percell={}; nseeds={}; neps={}; measage={}; infcmd={}
for task in TASKS:
    tr = [r for r in rows if r["task"] == task]
    if not tr: continue
    b = [r for r in tr if r["arm"] == "lat0"]
    emb = "widowx (act 40 ms)" if task in WIDOWX else "google_robot (act 333 ms)"
    print(f"\n{'='*118}\n{task}   {emb}   ({len([r for r in tr if r['arm']!='lat0'])} grid episodes)\n{'='*118}")
    print(f"{'arm':10s} {'predAge':>8s} {'cad':>6s} {'measAge':>8s} {'hold':>6s} {'inf/cmd':>7s} "
          f"{'sd':>3s} {'n':>5s} {'succ':>6s} {'design 95% CI':>15s} | {'paired vs lat0':>15s} {'95% CI':>17s}")
    for arm in ["lat0"]+order:
        ar = [r for r in tr if r["arm"] == arm]
        if not ar: continue
        k, n = sum(r["ok"] for r in ar), len(ar)
        icc, deff, _ = icc_deff(ar); p,_,_ = ci(k,n); _,dlo,dhi = ci(k,n,deff)
        lat, cad = (0.0, 0.0) if arm == "lat0" else ARMS[arm]
        ma = mean([r["act_age"] for r in ar]); ic = mean([r["infcmd"] for r in ar])
        ns = len({r["seed"] for r in ar})
        percell[(task,arm)]=p; nseeds[(task,arm)]=ns; neps[(task,arm)]=n
        measage[(task,arm)]=ma; infcmd[(task,arm)]=ic
        pr = paired(ar, b) if arm != "lat0" and b else None
        ptxt = f"{pr[0]:+6.1f} pts  [{pr[1]:+6.1f},{pr[2]:+6.1f}]" if pr else ""
        hold = (ma-lat) if (ma is not None and arm!="lat0") else None
        flag = "" if (ns==10 and n==240) or arm=="lat0" else f"  <-- INCOMPLETE {ns}/10 seeds"
        f8 = lambda v: f"{v:8.1f}" if v is not None else f"{'-':>8}"
        f6 = lambda v: f"{v:6.1f}" if v is not None else f"{'-':>6}"
        f7 = lambda v: f"{v:7.2f}" if v is not None else f"{'-':>7}"
        print(f"{arm:10s} {lat:8.1f} {cad:6.1f} {f8(ma)} {f6(hold)} {f7(ic)} "
              f"{ns:3d} {n:5d} {p:5.1f}% [{dlo:5.1f},{dhi:5.1f}] | {ptxt}{flag}")

def ladder(title, groups, xlab, xget):
    print(f"\n\n{'#'*118}\n{title}\n{'#'*118}")
    for key in sorted(groups):
        arms = groups[key]
        if len(arms) < 2: continue
        print(f"\n{xlab} {key}   ({len(arms)} points)")
        for task in TASKS:
            cells = [(xget(task,a), percell.get((task,a)), measage.get((task,a)),
                      infcmd.get((task,a)), a) for a in arms]
            cells = [c for c in cells if c[1] is not None]
            if len(cells) < 2: continue
            cells.sort(key=lambda c: c[0])
            lo_a, hi_a = cells[0][4], cells[-1][4]
            A=[r for r in rows if r["task"]==task and r["arm"]==hi_a]
            B=[r for r in rows if r["task"]==task and r["arm"]==lo_a]
            pr = paired(A,B)
            ptxt = f"  paired {pr[0]:+6.1f} [{pr[1]:+6.1f},{pr[2]:+6.1f}]" if pr else "  (paired n/a)"
            ma_lo, ma_hi = cells[0][2], cells[-1][2]
            span_meas = f"measAge {ma_lo:.0f}->{ma_hi:.0f}" if (ma_lo and ma_hi) else "measAge n/a"
            ic_lo, ic_hi = cells[0][3], cells[-1][3]
            span_ic = f"inf/cmd {ic_lo:.2f}->{ic_hi:.2f}" if (ic_lo and ic_hi) else ""
            print(f"  {task:6s} " + " ".join(f"{c[0]:.0f}:{c[1]:.1f}%" for c in cells))
            print(f"         {span_meas}  {span_ic}{ptxt}")

# CONTRAST 1 -- predicted age varied at fixed cadence
bycad = collections.defaultdict(list)
for a in order: bycad[round(ARMS[a][1])].append(a)
ladder("CONTRAST 1  --  PREDICTED age varied at FIXED cadence.\n"
       "The hold term is ~constant down each ladder, so measured age moves with predicted age here.",
       bycad, "cadence ~", lambda t,a: ARMS[a][0])

# CONTRAST 2 -- cadence varied at fixed PREDICTED age
bylat = collections.defaultdict(list)
for a in order: bylat[round(ARMS[a][0]/8)*8].append(a)
bylat = {k:v for k,v in bylat.items() if len({round(ARMS[a][1]) for a in v})>1}
ladder("CONTRAST 2  --  cadence varied at FIXED PREDICTED age.\n"
       "WARNING: on google_robot this is NOT a fixed EFFECTIVE-age ladder -- the hold term\n"
       "grows ~C/2 with cadence, so measured age still rises down the ladder. The printed\n"
       "measAge span is the amount of that leakage; read it before attributing to cadence.",
       bylat, "predicted age ~", lambda t,a: ARMS[a][1])

# ---- regressions, split by embodiment --------------------------------------
def cells_for(tasks):
    out=[]
    for t in tasks:
        cell=collections.defaultdict(list)
        for r in rows:
            if r["task"]==t and r["arm"] in ARMS: cell[(r["arm"],r["seed"])].append(r)
        if not cell: continue
        gm = sum(sum(x["ok"] for x in v)/len(v) for v in cell.values())/len(cell)
        for (arm,seed),v in cell.items():
            ma = mean([x["act_age"] for x in v]); ic = mean([x["infcmd"] for x in v])
            out.append(dict(task=t, arm=arm, seed=seed,
                            y=100*(sum(x["ok"] for x in v)/len(v) - gm),
                            pred=ARMS[arm][0], cad=ARMS[arm][1], meas=ma, ic=ic))
    return out

def report(name, C, specs):
    print(f"\n{'-'*118}\n{name}   ({len(C)} arm x seed cells)\n{'-'*118}")
    for label, keys in specs:
        d = [c for c in C if all(c[k] is not None for k in keys)]
        if len(d) < 12: print(f"  {label}: too few cells"); continue
        X=[[1.0]+[c[k]/100.0 for k in keys] for c in d]
        y=[c["y"] for c in d]; cl=[(c["task"],c["seed"]) for c in d]
        r = ols_cluster(X,y,cl)
        if not r: print(f"  {label}: singular"); continue
        beta,se,G = r
        print(f"  {label}  (n={len(d)}, {G} seed clusters)")
        for i,k in enumerate(keys, start=1):
            lo,hi = beta[i]-1.96*se[i], beta[i]+1.96*se[i]
            unit = "per 100 ms" if k in ("pred","cad","meas") else "per 1.0 inf/cmd (x100)"
            print(f"     {k:5s} {beta[i]:+8.2f} pts {unit:22s} 95% CI [{lo:+8.2f},{hi:+8.2f}]"
                  + ("  *" if lo*hi>0 else ""))
        if len(keys)==2:
            a=[c[keys[0]] for c in d]; b=[c[keys[1]] for c in d]
            rr=pearson(a,b); vif=1/(1-rr*rr) if abs(rr)<0.999 else float('inf')
            warn = "  <-- COLLINEAR: cannot separate these two" if abs(rr)>0.9 else ""
            print(f"     collinearity r({keys[0]},{keys[1]}) = {rr:+.3f}   VIF = {vif:.1f}{warn}")

print(f"\n\n{'#'*118}\nJOINT FITS  --  OLS on per-(arm,seed) cell means, SEs clustered by seed.\n"
      f"Slopes are percentage points. '*' = 95% CI excludes zero.\n{'#'*118}")
CW = cells_for(WIDOWX); CG = cells_for(GOOGLE)
report("widowx (egg+spoon) -- act 40 ms, inf/cmd<1: no hold-buffer or ensemble confound", CW,
       [("success ~ PREDICTED age + cadence", ["pred","cad"]),
        ("success ~ MEASURED age + cadence",  ["meas","cad"])])
report("google_robot (coke) -- act 333 ms: hold buffer and ensemble depth both active", CG,
       [("success ~ PREDICTED age + cadence",   ["pred","cad"]),
        ("success ~ MEASURED age + cadence",    ["meas","cad"]),
        ("success ~ MEASURED age + inf/cmd",    ["meas","ic"]),
        ("success ~ cadence + inf/cmd",         ["cad","ic"])])

# ---- the plane -------------------------------------------------------------
print(f"\n\n{'#'*118}\nTHE PLANE  --  success by cadence (rows) x PREDICTED age (cols), per embodiment.\n{'#'*118}")
for grp, name in ((WIDOWX,"widowx (egg+spoon)"), (GOOGLE,"google_robot (coke)")):
    print(f"\n{name}")
    agebins = sorted({int(round(ARMS[a][0]/20)*20) for a in order})
    print("cad\\predAge " + "".join(f"{b:>7d}" for b in agebins))
    for c in sorted({round(ARMS[a][1]) for a in order}):
        line = f"{c:10d} "
        for bnd in agebins:
            vs=[percell[(t,a)] for a in order for t in grp
                if round(ARMS[a][1])==c and int(round(ARMS[a][0]/20)*20)==bnd and (t,a) in percell]
            line += f"{sum(vs)/len(vs):6.1f}%" if vs else "      ."
        print(line)

# ---- completeness ----------------------------------------------------------
print(f"\n\n{'#'*118}\nCOMPLETENESS  --  every cell is reported on the n ACTUALLY COMPLETED.\n{'#'*118}")
short=[(t,a,nseeds.get((t,a),0),neps.get((t,a),0)) for t in TASKS for a in order
       if nseeds.get((t,a),0)!=10 or neps.get((t,a),0)!=240]
tot=sum(neps.get((t,a),0) for t in TASKS for a in order)
print(f"dispatched {len(TASKS)*len(order)} cells x 240 eps = {len(TASKS)*len(order)*240}; "
      f"completed {tot} over {len(TASKS)*len(order)-len(short)}/{len(TASKS)*len(order)} full cells")
if short:
    print(f"{len(short)} INCOMPLETE cell(s) -- flagged above, NOT to be read as n=10:")
    for t,a,ns,ne in short: print(f"    {t:6s} {a:10s} {ns}/10 seeds  {ne}/240 eps")
else: print("every cell has the full 10 seeds x 24 episodes.")

# ---- anchor ----------------------------------------------------------------
print(f"\n\n{'#'*118}\nANCHOR  --  PREDICTED design coordinate vs BOARD measurement.\n{'#'*118}")
plat,pcad = ARMS["g150_290"]
print(f"  g150_290 predicted: age {plat:.1f} ms, cadence {pcad:.1f} ms")
print(f"  p150w300 on board : age {281.6:.1f} ms, cadence {146.5:.1f} ms (cadence median n=3)")
print(f"     age gap     {plat-281.6:+.1f} ms ({100*(plat-281.6)/281.6:+.1f}%)")
print(f"     cadence gap {pcad-146.5:+.1f} ms ({100*(pcad-146.5)/146.5:+.1f}%, prediction pessimistic)")
print("  Board error is NOT one-directional (-4.3% here, 1.31x SLOWER on a 4-deep pipeline),")
print("  so this anchor bounds the extrapolation only near this cell. Every design coordinate")
print("  in this report is PREDICTED; the measAge column is the sim's own measurement.")

# ---- resolution / degeneracy audit -----------------------------------------
print(f"\n\n{'#'*118}\nRESOLUTION AUDIT  --  how many of the 39 design points are DISTINCT EXPERIMENTS?\n"
      f"The sim resolves latency onto the tick grid (latency_resolution_ms ~ 37.04 ms), so design\n"
      f"points closer together than one tick produce the SAME condition. Arms below share an\n"
      f"identical per-episode outcome vector on every common seed: they are one experiment, not\n"
      f"several, and a ladder containing them overstates the latency span it actually tested.\n{'#'*118}")
sig = collections.defaultdict(dict)
for f in glob.glob(str(HERE/"runs"/"*"/"*"/"summary.json")):
    name = Path(f).parent.name
    if "_rng" not in name: continue
    head, seed = name.rsplit("_rng", 1); task, arm = head.split("_", 1)
    if arm not in ARMS: continue
    try: d = json.load(open(f))
    except Exception: continue
    sig[(task, arm)][int(seed)] = tuple(int(e["success"]) for e in d.get("episodes", []))
for task in TASKS:
    arms_t = [a for a in order if (task,a) in sig]
    if not arms_t: continue
    # Compare on the INTERSECTION of completed seeds, so partial coverage does not
    # hide a duplicate; union-find the identical arms into components.
    parent = {a: a for a in arms_t}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(arms_t):
        for bb in arms_t[i+1:]:
            common = set(sig[(task,a)]) & set(sig[(task,bb)])
            if len(common) < 2: continue
            if all(sig[(task,a)][s] == sig[(task,bb)][s] for s in common):
                parent[find(a)] = find(bb)
    comp = collections.defaultdict(list)
    for a in arms_t: comp[find(a)].append(a)
    dup = [v for v in comp.values() if len(v) > 1]
    print(f"\n{task}: {len(arms_t)} arms run -> {len(comp)} distinct experiments "
          f"({len(arms_t)-len(comp)} arm(s) duplicated by tick quantisation)")
    if dup:
        for v in sorted(dup, key=lambda v: -len(v)):
            ages = ", ".join(f"{ARMS[a][0]:.1f}" for a in sorted(v, key=lambda a: ARMS[a][0]))
            cads = {round(ARMS[a][1]) for a in v}
            print(f"   IDENTICAL: {sorted(v)}")
            print(f"      predicted ages [{ages}] at cadence {sorted(cads)} -> same tick count, same run")
    else:
        print("   no two arms are simulation-identical on the seeds completed so far")
print("\nNOTE: on google_robot inf/cmd = act_ms / cadence almost exactly (333.3/C), so ensemble")
print("depth is NOT an independent regressor there -- it is cadence reparameterised. Any fit")
print("that enters both is collinear by construction; the two channels cannot be separated by")
print("this design, only by an experiment that changes ensembling at fixed cadence.")
