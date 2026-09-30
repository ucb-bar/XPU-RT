"""Gates reached, not just pass/fail: the graded outcome, paired seed for seed."""
import csv, glob, os, sys, statistics as st, collections, random
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)
R=os.path.join(_REPO, 'results', 'codesign_feedback') + os.sep
CELL=dict(person_h='2.4',walk_speed='0.0',percep_hold_ms='0.0',moment_scale='0.0055')
def load(trace, lat):
    out={}
    for f in glob.glob(R+'campaign*/campaign*.csv'):
        for r in flight_rows(f):
            if not all(r.get(k)==v for k,v in CELL.items()): continue
            if (r.get('ctrl_trace') or '').split('/')[-1]!=trace or r.get('percep_latency_ms')!=lat: continue
            try: g=float(r.get('gates_passed'))
            except (TypeError, ValueError): continue
            if r['outcome']=='success': g=4.0
            out[(r['seed'],r['cruise_speed'],r['course'],r['prop_density'])]=g
    return out
def boot(pairs, n=4000, seed=7):
    rng=random.Random(seed); d=[a-b for a,b in pairs]; m=len(d)
    s=sorted(st.fmean(rng.choices(d,k=m)) for _ in range(n))
    return st.fmean(d), s[int(.025*n)], s[int(.975*n)]
X=load('xpu_a_cpsat_hard.csv','56.8')
ARMS=[('ROS out of the box','ros_vanilla445.csv','242.0'),
      ('ROS one process','ros_vanilla4t45.csv','121.0'),
      ('ROS timer-driven','ros_vanilla4tm45.csv','242.0'),
      ('ROS hand-pinned','ros_p345.csv','55.8'),
      ('ROS hand-pinned + QoS1','ros_p3_q145.csv','31.1'),
      ('ROS 8 cores, 2 instances','ros_vanilla4x245.csv','37.0')]
print(f"{'ROS arm':<26} {'pairs':>6} {'ROS gates':>10} {'XPU-RT gates':>13}   mean difference [95% bootstrap]")
for lab,tr,lat in ARMS:
    d=load(tr,lat); common=sorted(set(d)&set(X))
    if len(common)<30: print(f'{lab:<26} {len(common):>6}  (waiting)'); continue
    pairs=[(X[k], d[k]) for k in common]
    m,lo,hi=boot(pairs)
    star=' ***' if lo>0 else ''
    print(f'{lab:<26} {len(common):>6} {st.fmean(d[k] for k in common):>10.2f} {st.fmean(X[k] for k in common):>13.2f}   {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}')
