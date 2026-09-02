"""Fill in the points the time-budget ablation didn't cover, on one instance:
the remaining constructive pickers, the metaheuristics at several budgets, and
warm-started CP-SAT. Merged with the ablation to give a full time/quality set.
"""
import json, os, sys, time
REPO = os.environ.get("XPURT_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"xpu-rt"))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from solver_bench import build
from schedule_decoder import DecoderContext, evaluate
import greedy_scheduler as gs, metaheuristics as mh
import cpsat_scheduler
from cpsat_scheduler import cpsat_schedule

SPEC="networks_periodic_dronet50ms_yolov8_firesim_q31"
w=build(os.path.join(REPO,"data/toplevel",SPEC+".json"),1,{"dronet":1})
ctx=DecoderContext(w); ws=mh.heft_schedule(w)
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","pareto_fill.json")
rows=[]
def add(name, family, budget, fn):
    t0=time.perf_counter()
    tt,aa=fn(); wall=time.perf_counter()-t0
    o,m,_=evaluate(ctx,tt,aa,True)
    rows.append(dict(solver=name, family=family, budget=budget,
                     objective=round(o,3), misses=m, wall_s=round(wall,2)))
    print(f"{name:<22}{o:>10.2f}{m:>6}{wall:>9.2f}s", flush=True)
    json.dump({"spec":SPEC,"ops":len(w.operations),"rows":rows},open(OUT,"w"),indent=1)

for n,f in (("greedy",gs.greedy_schedule),("greedy_periodic",gs.greedy_periodic_schedule),
            ("decomposed",gs.decomposed_schedule)):
    add(n,"constructive",0,lambda f=f: f(w))
for b in (15,60,120):
    add(f"pso@{b}s","metaheuristic",b,lambda b=b: mh.pso_schedule(w,time_budget=b,seed=0))
    add(f"sa@{b}s","metaheuristic",b,lambda b=b: mh.sa_schedule(w,time_budget=b,seed=0))
for b in (15,60,120,300):
    add(f"cpsat-warm@{b}s","exact",b,lambda b=b: cpsat_schedule(w,time_limit=b,warm_start=ws))
