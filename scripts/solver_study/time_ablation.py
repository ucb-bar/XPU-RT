"""Solve-time vs solution-quality ablation for the slow (exact) solvers.

Every earlier exact-solver number in this study came from a different budget —
20 s through 180 s depending on which harness produced it — which makes the
cross-solver comparisons hard to read. This sweeps the budget explicitly and
records, per point, what the solver actually proved: CP-SAT's status and bound,
cvxpy's status. A FEASIBLE answer at a 4x gap and a proven OPTIMAL one must not
look alike in the output.

Writes incrementally: a long run that gets interrupted keeps what it had.
"""
import argparse, json, os, sys, time
import numpy as np
REPO = os.environ.get("XPURT_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver_bench import build
from schedule_decoder import DecoderContext, evaluate
import cpsat_scheduler
from cpsat_scheduler import cpsat_schedule
from scheduler import schedule as milp
import metaheuristics as mh
import greedy_scheduler as gs

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--instance-map", default="")
ap.add_argument("--budgets", default="15,60,120,300,600")
ap.add_argument("--solvers", default="cpsat,milp:MOSEK,milp:HIGHS,milp:SCIPY")
ap.add_argument("--out", required=True)
a = ap.parse_args()

imap = {}
for part in (x for x in a.instance_map.split(",") if x):
    k, v = part.split("="); imap[k.strip()] = int(v)
w = build(os.path.join(REPO, "data/toplevel", a.spec + ".json"), 1, imap)
ctx = DecoderContext(w)
budgets = [float(x) for x in a.budgets.split(",")]
rows = []

# Reference points: the cheap methods, so the exact solvers' numbers can be
# read against what zero seconds already buys.
for name, fn in (("greedy_reserved", gs.greedy_reserved_schedule),
                 ("heft", mh.heft_schedule), ("heft_edf", mh.heft_edf_schedule)):
    t0 = time.perf_counter(); tt, aa = fn(w); wall = time.perf_counter() - t0
    o, m, _ = evaluate(ctx, tt, aa, True)
    rows.append(dict(solver=name, budget=0, objective=round(o, 3), misses=m,
                     wall_s=round(wall, 2), status="heuristic", bound=None))

def flush():
    json.dump({"spec": a.spec, "ops": len(w.operations), "rows": rows},
              open(a.out, "w"), indent=1)

print(f"{a.spec}: {len(w.operations)} ops")
print(f"{'solver':<14}{'budget':>8}{'objective':>12}{'status':>20}{'bound':>10}{'wall':>9}")
for r in rows:
    print(f"{r['solver']:<14}{r['budget']:>8.0f}{r['objective']:>12.2f}"
          f"{r['status']:>20}{'-':>10}{r['wall_s']:>8.1f}s")
flush()

for solver in a.solvers.split(","):
    for b in budgets:
        t0 = time.perf_counter()
        status, bound, obj, miss = "?", None, None, None
        try:
            if solver == "cpsat":
                tt, aa = cpsat_schedule(w, time_limit=b)
                info = dict(cpsat_scheduler.LAST_SOLVE)
                status = info.get("status", "?")
                bound = round(info.get("best_bound") or 0.0, 2)
            else:
                backend = solver.split(":", 1)[1]
                res = milp(w, time_limit=b, cvxpy_solver=backend)
                tt, aa = res[0], res[1]
                status = "milp"
            wall = time.perf_counter() - t0
            if tt is None:
                status = "no solution"
            else:
                obj, miss, _ = evaluate(ctx, tt, aa, True)
        except Exception as e:
            wall = time.perf_counter() - t0
            status = f"{type(e).__name__}"
        rows.append(dict(solver=solver, budget=b,
                         objective=None if obj is None else round(obj, 3),
                         misses=miss, wall_s=round(wall, 1),
                         status=status, bound=bound))
        print(f"{solver:<14}{b:>8.0f}"
              f"{'FAIL' if obj is None else format(obj, '.2f'):>12}"
              f"{status:>20}{'-' if bound is None else format(bound, '.2f'):>10}"
              f"{wall:>8.1f}s", flush=True)
        flush()
