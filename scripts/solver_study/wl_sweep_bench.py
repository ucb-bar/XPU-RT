"""Solver comparison on the real wl_sweep problem set.

Split roots: the solvers live in the flow-c checkout, the workloads and their
profiled dispatch graphs live in the rose-infra checkout. solver_bench.py
assumes one root for both, so the build is reproduced here with them split.

Instance counts come from each spec's own `num_instances` (gen_random_workload
emits one per periodic network, sized from the horizon it laid the sporadic
tasks into), so every solver schedules the workload as authored.
"""
import json, os, sys, time, glob
import numpy as np

CODE = os.environ.get("XPURT_CODE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.environ.get("XPURT_DATA_ROOT",
                       "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt")
OUT  = os.environ.get("XPURT_BENCH_OUT", ".")
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "xpu-rt"))

from workload_factory import create_workload_from_network_hierarchy, build_machine_combinations
from profile_loader import load_profiled_processing_times
from schedule_decoder import DecoderContext, evaluate
import greedy_scheduler as gs
import metaheuristics as mh


def build(spec_path):
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
        networks=nd["networks"], repo_base_path=DATA, machine_combinations=combos,
        combo_hw=combo_hw, profile_target=prof.get("target", "firesim_f2_rocket_saturn"),
        cpu_p_profile_hw=phw.get("cpu_p", "gemmini_q31"),
        cpu_e_profile_hw=phw.get("cpu_e", "V256D128_rvv"),
        rng=np.random.default_rng(42), p_core_speedup=float(hw.get("p_core_speedup", 1.0)),
        topo_tag_override=tt_override)
    w = create_workload_from_network_hierarchy(
        networks_data=nd, repo_base_path=DATA, machines=machines, transfer_times=tt,
        p_core_speedup=float(hw.get("p_core_speedup", 1.0)), random_seed=42,
        processing_times=pt, machine_combinations=combos)
    return w, nd


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
        "cpsat:warm":      lambda w: cpsat_schedule(w, time_limit=cpsat_time,
                                                    warm_start=mh.heft_edf_schedule(w)),
    }


def main():
    os.environ.setdefault(
        "XPURT_CPSAT_PYTHON",
        "/tmp/claude-1172/-scratch2-dima-misc-sw-XPU-RT/cb67e7aa-73a1-4f96-95bd-e31812fb2543/scratchpad/cpsat-venv/bin/python")
    specs = sorted(glob.glob(f"{DATA}/data/toplevel/wl_sweep/*.json"))
    print(f"{len(specs)} wl_sweep specs; solvers from {CODE}, data from {DATA}\n")
    allrows = {}
    for sp in specs:
        name = os.path.basename(sp)[:-5]
        try:
            w, nd = build(sp)
        except Exception as e:
            print(f"===== {name} ===== BUILD FAILED: {type(e).__name__}: {str(e)[:120]}")
            allrows[name] = {"error": f"build: {type(e).__name__}: {e}"}
            continue
        ctx = DecoderContext(w)
        lanes = f"{nd['hardware']['profile_hw'].get('cpu_p')}+{nd['hardware']['profile_hw'].get('cpu_e')}"
        print(f"===== {name} =====")
        print(f"  {len(w.operations)} ops, {len(w.get_machine_combinations())} combos, lanes {lanes}")
        print(f"  {'method':<16}{'objective':>12}{'all-ops':>11}{'misses':>8}{'wall':>9}")
        rows = []
        for mname, fn in methods(20.0, 120.0, 60.0).items():
            t0 = time.perf_counter()
            try:
                t, alpha = fn(w)
                wall = time.perf_counter() - t0
                obj, misses, all_end = evaluate(ctx, t, alpha, True)
                print(f"  {mname:<16}{obj:>12.2f}{all_end:>11.2f}{misses:>8}{wall:>8.1f}s", flush=True)
                rows.append(dict(method=mname, objective=round(obj,3), all_ops=round(all_end,3),
                                 misses=int(misses), wall_s=round(wall,2)))
            except Exception as e:
                wall = time.perf_counter() - t0
                print(f"  {mname:<16}{'FAILED':>12}  {type(e).__name__}: {str(e)[:70]}", flush=True)
                rows.append(dict(method=mname, error=f"{type(e).__name__}: {e}", wall_s=round(wall,2)))
        allrows[name] = {"lanes": lanes, "ops": len(w.operations), "rows": rows}
        json.dump(allrows, open(f"{OUT}/wl_results.json","w"), indent=1)
    json.dump(allrows, open(f"{OUT}/wl_results.json","w"), indent=1)
    print("\nWL SWEEP DONE")

if __name__ == "__main__":
    main()
