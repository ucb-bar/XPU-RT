"""Head-to-head on ONE fixed workload instance.

`run_xpurt_schedule.py` wraps every solver in the periodic-instance
refinement loop, which rebuilds the workload between passes and diverges on
some workloads. That conflates "is this solver good" with "does the loop
converge for this solver". Here the workload is built once, at fixed instance
counts, and every method schedules that same instance.
"""
import argparse, json, os, sys, time
import numpy as np

REPO = os.environ.get("XPURT_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

from workload_factory import create_workload_from_network_hierarchy, build_machine_combinations
from profile_loader import load_profiled_processing_times
from schedule_decoder import DecoderContext, evaluate
import greedy_scheduler as gs
import metaheuristics as mh


def build(spec_path, instances, instance_map=None):
    nd = json.load(open(spec_path))
    hw = nd["hardware"]
    machines, combos = build_machine_combinations(
        {k.upper(): v for k, v in hw["machines"].items()})
    phw = {k.lower(): v for k, v in hw["profile_hw"].items()}
    combo_hw = [phw[c[0].split("#")[0].lower()] for c in combos]
    tt = np.zeros((len(machines), len(machines)))
    prof = hw.get("profile", {})
    tt_override = (prof.get("topo_tag_per_hw")
                   or (prof.get("topo_tag") if prof.get("topo_tag_override") else None))
    pt, *_ = load_profiled_processing_times(
        networks=nd["networks"], repo_base_path=REPO, machine_combinations=combos,
        combo_hw=combo_hw, profile_target=prof.get("target", "spacemit_x60"),
        cpu_p_profile_hw=phw.get("cpu_p", "RVV"), cpu_e_profile_hw=phw.get("cpu_e", "scalar"),
        rng=np.random.default_rng(42), p_core_speedup=float(hw.get("p_core_speedup", 1.0)),
        topo_tag_override=tt_override)
    # Per-network counts matter: giving a 1000 ms-period network the same
    # instance count as a 10 ms-period one invents 60x the work it needs and
    # makes every deadline metric meaningless.
    for nid, info in nd["networks"].items():
        if info.get("period") is None:
            continue
        info["num_instances"] = int((instance_map or {}).get(nid, instances))
    return create_workload_from_network_hierarchy(
        networks_data=nd, repo_base_path=REPO, machines=machines, transfer_times=tt,
        p_core_speedup=float(hw.get("p_core_speedup", 1.0)), random_seed=42,
        processing_times=pt, machine_combinations=combos)


def methods(budget, cvxpy_time, cpsat_time):
    from cpsat_scheduler import cpsat_schedule
    from scheduler import schedule as milp
    return {
        "greedy":          lambda w: gs.greedy_schedule(w),
        "greedy_periodic": lambda w: gs.greedy_periodic_schedule(w),
        "greedy_reserved": lambda w: gs.greedy_reserved_schedule(w),
        "decomposed":      lambda w: gs.decomposed_schedule(w),
        "heft":            lambda w: mh.heft_schedule(w),
        "heft_edf":        lambda w: mh.heft_edf_schedule(w),
        "pso":             lambda w: mh.pso_schedule(w, time_budget=budget, seed=0),
        "sa":              lambda w: mh.sa_schedule(w, time_budget=budget, seed=0),
        "cpsat":           lambda w: cpsat_schedule(w, time_limit=cpsat_time),
        "milp:MOSEK":      lambda w: milp(w, time_limit=cvxpy_time, cvxpy_solver="MOSEK")[:2],
        "milp:HIGHS":      lambda w: milp(w, time_limit=cvxpy_time, cvxpy_solver="HIGHS")[:2],
        "milp:SCIPY":      lambda w: milp(w, time_limit=cvxpy_time, cvxpy_solver="SCIPY")[:2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--instance-map", default="",
                    help="per-network overrides, e.g. mlp_control=63,dronet=1")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--cvxpy-time", type=float, default=60.0)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    imap = {}
    for part in (x for x in a.instance_map.split(",") if x):
        k, v = part.split("=")
        imap[k.strip()] = int(v)
    w = build(os.path.join(REPO, "data/toplevel", a.spec + ".json"), a.instances, imap)
    ctx = DecoderContext(w)
    print(f"{a.spec}: {len(w.operations)} ops, {len(w.get_machine_combinations())} lanes, "
          f"instances={imap or a.instances}")
    print(f"{'method':<16}{'objective':>12}{'all-ops':>11}{'misses':>8}{'wall':>9}")
    rows = []
    sel = set(x for x in a.only.split(",") if x)
    for name, fn in methods(a.budget, a.cvxpy_time, a.cpsat_time).items():
        if sel and name not in sel:
            continue
        t0 = time.perf_counter()
        try:
            t, alpha = fn(w)
            wall = time.perf_counter() - t0
            obj, misses, all_end = evaluate(ctx, t, alpha, True)
            print(f"{name:<16}{obj:>12.2f}{all_end:>11.2f}{misses:>8}{wall:>8.1f}s")
            rows.append(dict(method=name, objective=round(obj, 3),
                             all_ops=round(all_end, 3), misses=misses,
                             wall_s=round(wall, 2)))
            _flush(a, w, rows)
        except Exception as e:
            wall = time.perf_counter() - t0
            print(f"{name:<16}{'FAILED':>12}  {type(e).__name__}: {str(e)[:60]}")
            rows.append(dict(method=name, objective=None, all_ops=None,
                             misses=None, wall_s=round(wall, 2),
                             error=f"{type(e).__name__}: {e}"))
            _flush(a, w, rows)
    _flush(a, w, rows)


def _flush(a, w, rows):
    """Persist after every method, not just at the end: a long run that gets
    interrupted otherwise loses every result it already had."""
    if a.out:
        json.dump({"spec": a.spec, "ops": len(w.operations),
                   "instances": a.instances, "rows": rows},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
