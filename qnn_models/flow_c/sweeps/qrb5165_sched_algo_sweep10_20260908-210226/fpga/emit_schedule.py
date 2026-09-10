#!/usr/bin/env python3
"""Emit ONE sweep10 (arm, workload, solver) schedule as a `scheduled_*.json`
the modelblaster xpurt codegen can ingest.

Why this exists rather than `scripts/run_xpurt_schedule.py`:

  * `run_xpurt_schedule.py` only exposes four solvers (milp, greedy,
    greedy_periodic, decomposed). Ten of the twelve sweep10 entries -- and
    every one of the winners -- come from the standalone XPU-RT solver tree
    (`metaheuristics.py`, `cpsat_scheduler.py`), which that script cannot
    reach.
  * `run_xpurt_schedule.py` also wraps greedy in a periodic-instance
    refinement loop and post-trims the schedule. sweep10 does neither: it
    builds the workload ONCE at the spec's own `num_instances` and hands that
    same instance to every solver.

So the workload build here is byte-for-byte `sweep10_runner.build` (which is
itself verbatim `wl_sweep_bench.build`), and only the emit step is new: the
solver's `(t, alpha)` goes straight into `postprocessing.output_scheduled_json`
-- the same function `run_xpurt_schedule.py` uses, unchanged and shared by both
trees -- so the JSON the codegen sees is the format it already accepts.

Net effect: every ELF built from these schedules corresponds to exactly one row
of `results/all_results.json`, with no harness difference in between.

Usage:
  XPURT_CODE_ROOT=/scratch2/dima/misc_sw/XPU-RT \
  XPURT_DATA_ROOT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt \
  emit_schedule.py --arm wl_sweep --name networks_bimodal_gempair \
                   --solver pso --out /path/scheduled_x.json
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np

CODE = os.environ["XPURT_CODE_ROOT"]
DATA = os.environ["XPURT_DATA_ROOT"]
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "xpu-rt"))

from workload_factory import create_workload_from_network_hierarchy, build_machine_combinations
from profile_loader import load_profiled_processing_times
from postprocessing import (output_scheduled_json,
                            trim_periodic_after_nonperiodic_makespan)
from schedule_decoder import DecoderContext, evaluate

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "scripts", "sweep10_runner.py")


def _load_runner():
    """Reuse sweep10_runner's make_solver/validate verbatim -- the winners have
    to come out of the same code that produced the predictions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep10_runner", os.path.abspath(_RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(spec_path):
    """Verbatim sweep10_runner.build, but keeps the profile side-channels that
    `output_scheduled_json` needs for per-dispatch `module_name`."""
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
    pt, prof_p, prof_e, prof_by_net = load_profiled_processing_times(
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
    return w, nd, prof_p, prof_e, prof_by_net


def _best_of_fast(w):
    """`best-of-fast` is not a solver in `make_solver` -- sweep10 SYNTHESISED it
    in the analyzer from the six sub-second heuristics' own rows
    (`sweep10_analyze.py:86`, feasible-then-fastest). Reproduce that selection
    here so the row gets a real schedule, and so its dedupe identity with the
    member it picks is measured rather than assumed."""
    import greedy_scheduler as gs
    import metaheuristics as mh
    ctx = DecoderContext(w)
    cands = []
    for name, f in (("greedy", gs.greedy_schedule),
                    ("greedy_periodic", gs.greedy_periodic_schedule),
                    ("greedy_reserved", gs.greedy_reserved_schedule),
                    ("decomposed", gs.decomposed_schedule),
                    ("heft", mh.heft_schedule),
                    ("heft_edf", mh.heft_edf_schedule)):
        try:
            t, al = f(w)
            o, ms, _ = evaluate(ctx, t, al, True)
            cands.append(((1 if ms > 0 else 0, ms, o), name, (t, al)))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])
    if not cands:
        raise RuntimeError("best-of-fast: no member returned a schedule")
    return cands[0][2], cands[0][1]


def _mask_undeclared(w, nd, cost_model_path):
    """Forbid the cells the binding manifest cannot execute.

    `build_cost_model.py` already drops these from the cost model, and says why:
    "A cell the binding manifest cannot execute must not be in the cost model.
    The scheduler has no `forbidden` flag -- it will happily place a tile on the
    cheapest lane it is offered." But that decision never reaches the SOLVER,
    because the solver's workload is built straight out of `gen/profile/` by
    `load_profiled_processing_times`, which knows nothing about bindings. The
    two paths only meet at `flowc/schedule.py::ingest`, which refuses to emit a
    runtime -- i.e. after the solve, and only if the solver happened to take the
    bait.

    It did. `vint/vint_encoders@gpu` has a measured 55.854 ms in
    `gen/profile/GPU/qrb5165_flowc/vint/` and no GPU context in
    `bindings/vint.json`, so on `vint_{intro,multi}_cg` CP-SAT proves an OPTIMAL
    72.279 ms that cannot be built. Phase 3 missed it only by accident of
    timing: its solves ran 2026-09-08 16:31 and that profile row was not written
    until 18:21, so the cell was invisible to the solver then and is visible now.

    Masking to +inf here makes the constraint the cost model asserts an actual
    constraint on the search, rather than a post-hoc rejection.
    """
    cm = json.load(open(cost_model_path))
    dropped = cm.get("dropped_undeclared_cells") or []
    if not dropped:
        return []
    phw = {k.lower(): v for k, v in nd["hardware"]["profile_hw"].items()}
    combos = w.get_machine_combinations()
    combo_be = [phw[c[0].split("#")[0].lower()].lower() for c in combos]
    here = os.path.dirname(os.path.abspath(__file__))
    masked = []
    for entry in dropped:
        cell, _, be = entry.partition("@")
        net, _, tile = cell.partition("/")
        man = os.path.join(here, "..", "bindings", f"{net}.json")
        if not os.path.exists(man):
            man = os.path.join(here, "..", "..", "..", "bindings", f"{net}.json")
        if not os.path.exists(man):
            continue
        names = [b["name"] for b in json.load(open(man))["bindings"]]
        if tile not in names:
            continue
        k = names.index(tile)                       # k-th binding == dispatch_k
        pat = re.compile(rf"^{re.escape(net)}\d*_dispatch_{k}$")
        cs = [i for i, b in enumerate(combo_be) if b == be]
        if not cs:
            continue
        for op in w.operations:
            if not pat.match(str(getattr(op, "operation_name", "") or "")):
                continue
            for c in cs:
                if np.isfinite(op.processing_times[c]):
                    masked.append(f"{op.operation_name}@{be}")
                    op.processing_times[c] = float("inf")
    return sorted(set(masked))


def _sched_hash(doc):
    """Dedupe key = the op -> (combination, start, duration) assignment, i.e.
    everything the codegen reads out of the schedule. NOT the objective: two
    different assignments can share a makespan, and collapsing those would
    silently drop coverage."""
    disp = doc["dispatches"]
    items = sorted(disp.items()) if isinstance(disp, dict) else \
        sorted(((str(i), d) for i, d in enumerate(disp)))
    h = hashlib.sha256()
    for k, d in items:
        h.update(k.encode())
        h.update(b"\0")
        h.update(json.dumps(d, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)          # wl_sweep | wl_sweep_shard
    ap.add_argument("--name", required=True)         # networks_<family>_<cfg>
    ap.add_argument("--solver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-out", default=None)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune-periodic", action="store_true",
                    help="apply postprocessing.trim_periodic_after_nonperiodic_makespan "
                         "to the solved schedule before emitting it, as the specs\' "
                         "scheduler.prune_periodic asks for")
    ap.add_argument("--mask-undeclared", default=None,
                    help="cost_model.json whose dropped_undeclared_cells "
                         "are masked out of the solver's workload")
    a = ap.parse_args()

    runner = _load_runner()
    spec = os.path.join(DATA, "data", "toplevel", a.arm, a.name + ".json")
    w, nd, prof_p, prof_e, prof_by_net = build(spec)
    masked = _mask_undeclared(w, nd, a.mask_undeclared) if a.mask_undeclared else []
    ctx = DecoderContext(w)

    picked = None
    t0 = time.perf_counter()
    if a.solver == "best-of-fast":
        (t, alpha), picked = _best_of_fast(w)
    else:
        fn = runner.make_solver(a.solver, a.budget, a.cpsat_time, a.cpsat_workers, a.seed)
        t, alpha = fn(w)
    wall = round(time.perf_counter() - t0, 3)
    obj, misses, all_end = evaluate(ctx, t, alpha, True)
    val = runner.validate(ctx, t, alpha)

    # `prune_periodic` is written into every ported spec by
    # `mk_workloads_qrb5165.py:282` and nothing in this tree ever read it: the
    # trim lives in `scripts/run_xpurt_schedule.py`, which sweep10 deliberately
    # bypasses. It is purely post-hoc -- it takes an already-computed (t, alpha)
    # and drops periodic operations whose window does not overlap
    # [0, non-periodic makespan). It cannot move placement, and because every
    # dropped op starts at or after the cut it cannot move the non-periodic
    # makespan either. That is asserted here rather than assumed: the objective
    # is re-evaluated on the trimmed workload and has to come back bit-identical.
    pruned = None
    if a.prune_periodic:
        pre = dict(objective=float(obj), all_ops=float(all_end),
                   misses=int(misses), ops=int(ctx.n))
        w, t, alpha = trim_periodic_after_nonperiodic_makespan(
            w, t, alpha, horizon_ms=nd.get("horizon_ms"))
        ctx = DecoderContext(w)
        obj, misses, all_end = evaluate(ctx, t, alpha, True)
        val = runner.validate(ctx, t, alpha)
        pruned = dict(pre_objective=round(pre["objective"], 6),
                      pre_all_ops=round(pre["all_ops"], 6),
                      pre_misses=pre["misses"], pre_ops=pre["ops"],
                      post_ops=int(ctx.n),
                      dropped_ops=pre["ops"] - int(ctx.n),
                      objective_unchanged=abs(float(obj) - pre["objective"]) <= 1e-9)
        if not pruned["objective_unchanged"]:
            raise SystemExit(
                f"prune_periodic moved the non-periodic objective: "
                f"{pre['objective']} -> {float(obj)}. The trim is post-hoc and "
                f"cannot do that; this is a bug, not a result.")

    phw = {k.upper(): v for k, v in nd["hardware"]["profile_hw"].items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    output_scheduled_json(
        combined_workload=w, t=t, alpha=alpha, output_path=a.out,
        profiled_times_p=prof_p, profiled_times_e=prof_e,
        profile_hw=phw, profiled_times_by_network=prof_by_net)

    doc = json.load(open(a.out))
    disp = doc["dispatches"]
    vals = list(disp.values()) if isinstance(disp, dict) else disp
    rec = dict(arm=a.arm, workload=a.name, solver=a.solver, seed=a.seed,
               ops=int(ctx.n), periodic_ops=int(ctx.periodic.sum()),
               combos=int(ctx.n_combos), wall_s=wall,
               objective=round(float(obj), 6), all_ops=round(float(all_end), 6),
               misses=int(misses), validation=val,
               dispatches=len(vals),
               json_makespan=round(max(x["start_time"] + x["duration"] for x in vals), 6),
               sched_hash=_sched_hash(doc), picked=picked,
               masked_cells=masked, pruned=pruned,
               schedule=os.path.abspath(a.out))
    print(json.dumps(rec))
    if a.meta_out:
        json.dump(rec, open(a.meta_out, "w"), indent=1)


if __name__ == "__main__":
    main()
