"""How much, and how, each ROS arm fails against the best XPU-RT arm on matched cells."""
import csv, glob, os, sys, collections
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'scripts'))
from showdown_atlas import newcombe
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)
R=os.path.join(_REPO, 'results', 'codesign_feedback') + os.sep
CELL=dict(person_h='2.4',walk_speed='0.0',percep_hold_ms='0.0',moment_scale='0.0055')
def load(trace, lat):
    out={}
    for f in glob.glob(R+'campaign*/campaign*.csv'):
        for r in flight_rows(f):
            if not all(r.get(k)==v for k,v in CELL.items()): continue
            if (r.get('ctrl_trace') or '').split('/')[-1]!=trace or r.get('percep_latency_ms')!=lat: continue
            out[(r['seed'],r['cruise_speed'],r['course'],r['prop_density'])]=r
    return out
BEST=('xpu_shard45.csv','36.0'); FALLBACK=('xpu_a_cpsat_hard.csv','56.8')
best=load(*BEST)
if len(best)<40:
    print(f'best XPU-RT arm has only {len(best)} flights so far; using the 56.8 ms arm as the reference\n')
    best=load(*FALLBACK); ref='XPU-RT 56.8 ms'
else: ref='XPU-RT 36 ms (sharded)'
ARMS=[('ROS out of the box','ros_vanilla445.csv','242.0'),
      ('ROS timer-driven','ros_vanilla4tm45.csv','242.0'),
      ('ROS timer, all 8 harts','ros_vanilla8tm45.csv','241.9'),
      ('ROS one process','ros_vanilla4t45.csv','121.0'),
      ('ROS hand-pinned','ros_p345.csv','55.8'),
      ('ROS hand-pinned + QoS1','ros_p3_q145.csv','31.1'),
      ('ROS 8 cores, 2 instances','ros_vanilla4x245.csv','37.0'),
      ('ROS 8 cores + timer','ros_vanilla4x2tm45.csv','37.8')]
def gates(rows, keys):
    g=collections.Counter()
    for k in keys:
        r=rows[k]
        if r['outcome']!='crash': continue
        try: g[int(float(r.get('gates_passed') or -1))]+=1
        except ValueError: pass
    return g
print(f'reference: {ref}   ({len(best)} flights)\n')
print(f"{'ROS arm':<26} {'paired':>6} {'ROS ok':>8} {ref[:12]:>12}  {'sep':>20}   ROS crashes by gate reached")
for lab,tr,lat in ARMS:
    d=load(tr,lat); common=sorted(set(d)&set(best))
    if len(common)<12: print(f'{lab:<26} {len(common):>6}   (waiting on flights)'); continue
    nr=sum(1 for k in common if d[k]['outcome']!='timeout'); kr=sum(1 for k in common if d[k]['outcome']=='success')
    nx=sum(1 for k in common if best[k]['outcome']!='timeout'); kx=sum(1 for k in common if best[k]['outcome']=='success')
    dd,lo,hi=newcombe(kx,nx,kr,nr)
    g=gates(d,common); tot=sum(g.values())
    gs=" ".join(f"G{i}:{g.get(i,0)*100//max(tot,1)}%" for i in range(4) if g.get(i))
    print(f'{lab:<26} {len(common):>6} {kr:>4}/{nr:<3} {kx:>7}/{nx:<3}  {dd*100:+6.1f} [{lo*100:+5.1f},{hi*100:+5.1f}]   {gs}')
