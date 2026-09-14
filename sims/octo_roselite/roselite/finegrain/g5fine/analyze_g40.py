#!/usr/bin/env python3
"""Does a cadence effect APPEAR when google_robot is unsaturated?

drawer at the native 3 Hz is saturated by every schedule here (each policy result drives
<1 command), and the plane sweep found no cadence effect. At 27 Hz the same task is
UNSATURATED. Same controller, same arms, same seeds -- control rate is the only variable.

Prediction: a cadence effect at 27 Hz that is absent at 3 Hz. If it does not appear,
the widowx/google asymmetry is NOT explained by actuation rate and fig_energy's framing
needs revisiting.

Trend test, not endpoint contrast: the harness noise is a rare-jump process, so a single
pairwise difference is weak evidence while a monotone ordering across the ladder is
strong. Per-seed Spearman across arms ordered by cadence, averaged over seeds with a
paired-t interval, plus the -/+ seed sign counts.
"""
import json, collections, re, math
from pathlib import Path

RUNS = Path(__file__).parent.parent / "runs_g40"
# arm -> (measured cadence ms, measured age ms)
ARMS = [("pipe110fix",111.4,385.1),("p105w300",124.8,258.7),("p130w275",130.1,272.3),
        ("p150w300",150.4,281.6),("pipe200fix",219.2,260.5),("serial283",283.4,283.4),
        ("fp32_555",555.0,555.0),("cpu685",684.8,684.8)]
TC={9:2.262,8:2.306,7:2.365,6:2.447,5:2.571,4:2.776,3:3.182,2:4.303}

def spearman(xs, ys):
    n=len(xs)
    if n<3: return None
    def rank(v):
        s=sorted(range(n), key=lambda i:v[i]); r=[0.0]*n
        i=0
        while i<n:
            j=i
            while j+1<n and v[s[j+1]]==v[s[i]]: j+=1
            for k in range(i,j+1): r[s[k]]=(i+j)/2.0+1
            i=j+1
        return r
    rx,ry=rank(xs),rank(ys)
    mx,my=sum(rx)/n,sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else None

data=collections.defaultdict(dict)   # (rate, arm) -> {seed: rate}
for f in RUNS.rglob("*/summary.json"):
    m=re.match(r"drawer_(.+?)_(native|fine27)_rng(\d+)$", f.parent.name)
    if not m: continue
    arm,rate,seed=m.group(1),m.group(2),int(m.group(3))
    d=json.load(open(f))
    data[(rate,arm)][seed]=100.0*d["n_success"]/d["n_episodes"]

print(f"{'':22s}{'NATIVE 3 Hz (saturated)':>26s}   {'FINE 27 Hz (unsaturated)':>26s}")
print(f"  {'arm':12s} {'cad':>6s} {'age':>6s} {'mean':>8s} {'n':>3s}   {'mean':>8s} {'n':>3s}   delta")
for a,cad,age in ARMS:
    n_=data.get(("native",a),{}); f_=data.get(("fine27",a),{})
    if not n_ and not f_: continue
    mn=sum(n_.values())/len(n_) if n_ else float("nan")
    mf=sum(f_.values())/len(f_) if f_ else float("nan")
    print(f"  {a:12s} {cad:6.1f} {age:6.1f} {mn:7.1f}% {len(n_):3d}   {mf:7.1f}% {len(f_):3d}   {mf-mn:+6.1f}")

print("\nTREND across the cadence ladder (per-seed Spearman; negative = success falls as cadence rises)")
for rate,lab in (("native","NATIVE 3 Hz  (saturated)"),("fine27","FINE 27 Hz (unsaturated)")):
    seeds=set.intersection(*[set(data[(rate,a)]) for a,_,_ in ARMS if data.get((rate,a))]) \
          if any(data.get((rate,a)) for a,_,_ in ARMS) else set()
    arms=[(a,c) for a,c,_ in ARMS if data.get((rate,a))]
    if len(arms)<3 or not seeds:
        print(f"  {lab}: insufficient data"); continue
    rhos=[]
    for s in sorted(seeds):
        xs=[c for _,c in arms]; ys=[data[(rate,a)][s] for a,_ in arms]
        r=spearman(xs,ys)
        if r is not None: rhos.append(r)
    n=len(rhos); m=sum(rhos)/n
    sd=math.sqrt(sum((r-m)**2 for r in rhos)/(n-1)) if n>1 else 0.0
    h=TC.get(n-1,2.262)*sd/math.sqrt(n) if n>1 else float("nan")
    neg=sum(1 for r in rhos if r<0); pos=sum(1 for r in rhos if r>0)
    print(f"  {lab}: rho={m:+.3f} [{m-h:+.3f}, {m+h:+.3f}]  seeds -/+ = {neg}/{pos}  (n={n} seeds, k={len(arms)} arms)")
print("\nA cadence effect present at 27 Hz and absent at 3 Hz confirms SATURATION as the")
print("mechanism. Same controller and seeds throughout; control rate is the only variable.")
