"""
Scheduling script for IREE dispatch graphs from hierarchical network dependencies.
Parses top-level network dependency JSON files and schedules them onto
CPU_P (performant) and CPU_E (efficient) cores.
"""

from __future__ import annotations

import sys
import os
import pathlib
import json
import argparse
import numpy as np

# Add the in-tree xpu-rt directory to sys.path BEFORE site-packages so
# our local edits take priority over any shadowed install in the
# active conda env (xpurt has greedy_scheduler.py installed as a flat
# site-packages module — without insert(0, ...), the install shadows
# our updates).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "xpu-rt"))

import time

from workload import Workload, Operation
from workload_factory import (
    create_workload_from_network_hierarchy,
    build_machine_combinations,
    machine_type_prefix,
)
from schedulers import get_scheduler, available_schedulers
from metrics import compute_metrics
# The greedy family lives in greedy_scheduler (no cvxpy dependency); the MILP
# path is reached lazily through the registry via get_scheduler("mosek").
from greedy_scheduler import (
    greedy_schedule,
    greedy_periodic_schedule,
    decomposed_schedule,
)
from profile_loader import (
    load_profiled_processing_times,
    compute_pdb_hash,
    _LAST_LOAD_CSV_PATHS,
)
from postprocessing import trim_periodic_after_nonperiodic_makespan, output_scheduled_json
from granularity_advisor import analyze_granularity, from_workload
import plot

# Hardware constants — SpacemiT x60
CPU_P = "CPU_P"
CPU_E = "CPU_E"


def load_networks_config(json_path: str) -> tuple[dict, dict]:
    """
    Load a network dependencies JSON file and extract hardware/scheduler config.

    Reads from:
      - hardware.machines, hardware.profile_hw, hardware.profile, hardware.p_core_speedup
      - scheduler.*

    Returns:
      (networks_data, config) where networks_data is the full parsed JSON
      and config is a flat dict of resolved settings with defaults.
    """
    with open(json_path, 'r') as f:
        networks_data = json.load(f)

    hw = networks_data.get("hardware", {})
    sched = networks_data.get("scheduler", {})
    profile_cfg = hw.get("profile", {})
    profile_hw = hw.get("profile_hw", {})

    raw_seed = sched.get("random_seed", 0)
    seed = None if (isinstance(raw_seed, int) and raw_seed < 0) else int(raw_seed)

    # Parse hardware.machines into {machine_type: core_count}
    machines_cfg = hw.get("machines", {})
    if isinstance(machines_cfg, dict) and machines_cfg and all(isinstance(v, int) for v in machines_cfg.values()):
        machine_core_counts = {k.strip().upper(): v for k, v in machines_cfg.items() if v > 0}
        if not machine_core_counts:
            machine_core_counts = {CPU_P: 1, CPU_E: 1}
    else:
        machine_core_counts = {CPU_P: 1, CPU_E: 1}

    # Generalised profile_hw map: every key in `hardware.machines` may have
    # a corresponding entry in `hardware.profile_hw` mapping it to a
    # bitstream/backend tag (e.g. "HTP", "GPU_fp16", "RVV"). The legacy
    # cpu_p/cpu_e fields below are kept for back-compat with two-machine
    # configs and for profile_loader's "p"/"e" output bucketing — when more
    # than two kinds are present, the first key is the "p" bucket and the
    # rest fall into "e".
    profile_hw_map = {k.strip().lower(): v
                      for k, v in profile_hw.items()
                      if isinstance(v, str)}
    cfg = {
        "profile_hw_map":   profile_hw_map,
        "cpu_p_profile_hw": profile_hw.get("cpu_p", "RVV"),
        "cpu_e_profile_hw": profile_hw.get("cpu_e", "scalar"),
        "profile_target": profile_cfg.get("target", "spacemit_x60"),
        "profile_topo_tag": profile_cfg.get("topo_tag", "topo_0_1_2_3"),
        # When True, the profile_topo_tag above forces every machine
        # combination to look up profile data under that single topo,
        # regardless of combination size. Use this when the registry
        # models a multi-hart cluster as a single machine: the scheduler
        # sees 1 unit per kind, but the dispatched ops use the multi-core
        # measurements that match the cluster's actual width.
        "profile_topo_tag_override": bool(profile_cfg.get("topo_tag_override", False)),
        # Optional per-hw topo override, e.g.
        #   "topo_tag_per_hw": { "RVV": "topo_0", "scalar": "topo_0_1_2_3" }
        # Use this when one cluster maps to a single-hart machine
        # (singleton) and another cluster maps to a multi-hart unit —
        # each kind's machine model is one "slot" but profile lookups
        # need to read different topos to reflect the actual cluster
        # widths. Takes precedence over the scalar topo_tag_override.
        "profile_topo_tag_per_hw": profile_cfg.get("topo_tag_per_hw") or None,
        "gen_root": profile_cfg.get("gen_root", None),
        "p_core_speedup": float(hw.get("p_core_speedup", 1.5)),
        "random_seed": seed,
        "solver_verbosity": int(sched.get("solver_verbosity", 0)),
        "time_limit": sched.get("time_limit", None),
        "use_profiled": bool(sched.get("use_profiled", False)),
        "prune_periodic": bool(sched.get("prune_periodic", True)),
        "restrict_makespan_to_nonperiodic": bool(sched.get("restrict_makespan_to_nonperiodic", True)),
        "machine_combination_mode": str(sched.get("machine_combination_mode", "singletons")),
        "enforce_same_processor_combinations": bool(sched.get("enforce_same_processor_combinations", True)),
        "machine_core_counts": machine_core_counts,
    }

    return networks_data, cfg

def schedule_iree_networks(
    networks_json_path: str = None,
    solver: str = "milp",
    solver_verbosity: int | None = None,
    time_limit: float | None = None,
    random_seed: int | None = None,
    p_core_speedup: float | None = None,
    use_profiled: bool | None = None,
    prune_periodic: bool | None = None,
    restrict_makespan_to_nonperiodic: bool | None = None,
    scheduler: str = "mosek",
    max_periodic_iters: int = 4,
    freshness_weight: float = 0.0,
    contention_model=None,
) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Main function to schedule networks from a hierarchical network dependencies JSON file.
    CLI arguments override JSON config values when not None.

    `solver` selects the scheduling algorithm:
      - "milp"            (default) — global optimization via cvxpy/mosek
                              (`scheduler.schedule`).  One pass; no
                              periodic-instance refinement.
      - "greedy"            — list-scheduling heuristic via
                              `greedy_scheduler.greedy_schedule`. Runs
                              an iterative periodic-instance refinement
                              loop (start with num_instances=1 per
                              periodic network if
                              restrict_makespan_to_nonperiodic is set,
                              then grow only as actual contention pushes
                              the non-periodic makespan out — caps at
                              `max_periodic_iters`).
      - "greedy_periodic"   — same loop as `greedy`, but the per-iter
                              picker prioritizes non-periodic ops over
                              periodic ones. Periodic ops only get
                              scheduled when no non-periodic is ready,
                              with an emergency-promote when delaying
                              would miss the periodic max_end_t
                              window. Use when the heterogeneous
                              workload has a non-periodic critical
                              path (e.g. yolov8) that vanilla greedy
                              fragments by interleaving periodic
                              instances (e.g. dronet 50ms).
    """
    if solver not in ("milp", "greedy", "greedy_periodic", "decomposed"):
        raise ValueError(
            f"solver must be 'milp' | 'greedy' | 'greedy_periodic' | 'decomposed', "
            f"got {solver!r}"
        )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_base_path = os.path.abspath(os.path.join(script_dir, '..'))

    if networks_json_path is None:
        networks_json_path = os.path.join(repo_base_path, 'data', 'toplevel', 'networks_periodic_profile.json')
    if not os.path.isabs(networks_json_path):
        networks_json_path = os.path.join(repo_base_path, networks_json_path)

    print("=" * 60)
    print("Loading network hierarchy from JSON...")
    print("=" * 60)
    print(f"\nLoading networks from: {networks_json_path}")

    networks_data, cfg = load_networks_config(networks_json_path)

    # CLI overrides (None = use JSON config value)
    cfg["p_core_speedup"] = p_core_speedup if p_core_speedup is not None else cfg["p_core_speedup"]
    cfg["random_seed"] = random_seed if random_seed is not None else cfg["random_seed"]
    cfg["solver_verbosity"] = solver_verbosity if solver_verbosity is not None else cfg["solver_verbosity"]
    cfg["time_limit"] = time_limit if time_limit is not None else cfg["time_limit"]
    cfg["use_profiled"] = use_profiled if use_profiled is not None else cfg["use_profiled"]
    cfg["prune_periodic"] = prune_periodic if prune_periodic is not None else cfg["prune_periodic"]
    cfg["restrict_makespan_to_nonperiodic"] = restrict_makespan_to_nonperiodic if restrict_makespan_to_nonperiodic is not None else cfg["restrict_makespan_to_nonperiodic"]

    machine_core_counts = cfg["machine_core_counts"]
    machine_combination_mode = cfg["machine_combination_mode"]
    enforce_same_processor_combinations = cfg["enforce_same_processor_combinations"]
    cpu_p_profile_hw = cfg["cpu_p_profile_hw"]
    cpu_e_profile_hw = cfg["cpu_e_profile_hw"]
    profile_target = cfg["profile_target"]
    profile_topo_tag = cfg["profile_topo_tag"]
    profile_topo_tag_override = cfg["profile_topo_tag_override"]
    profile_topo_tag_per_hw = cfg["profile_topo_tag_per_hw"]
    # `gen_root` selects which profile tree the timings come from. It was parsed
    # into cfg and then never read by anything, so a config naming an alternate
    # tree silently got the default one -- i.e. a run could be labelled with one
    # timing basis while actually using another. Defaulting to "gen" preserves
    # every existing config, whose value is either absent or literally "gen".
    gen_root = cfg.get("gen_root") or "gen"
    effective_p_core_speedup = cfg["p_core_speedup"]
    effective_random_seed = cfg["random_seed"]
    effective_solver_verbosity = cfg["solver_verbosity"]
    effective_time_limit = cfg["time_limit"]
    effective_use_profiled = cfg["use_profiled"]
    effective_prune_periodic = cfg["prune_periodic"]
    effective_restrict_makespan_to_nonperiodic = cfg["restrict_makespan_to_nonperiodic"]

    rng = np.random.default_rng(effective_random_seed)

    # Print network information
    networks = networks_data.get('networks', {})
    edges = networks_data.get('edges', [])
    print(f"\nFound {len(networks)} networks:")
    for network_id, network_info in networks.items():
        print(f"  - {network_id} (id: {network_info.get('id')}, identifier: {network_info.get('identifier')})")

    print(f"\nFound {len(edges)} network-level dependencies:")
    for edge in edges:
        print(f"  - {edge.get('from')} → {edge.get('to')}")

    print("\nResolved runtime configuration:")
    print(f"  Machine core counts: {machine_core_counts}")
    print(f"  Profile HW mapping: {CPU_P}->{cpu_p_profile_hw}, {CPU_E}->{cpu_e_profile_hw}")
    print(f"  Profile target/topology: target={profile_target}, topo_tag={profile_topo_tag}")
    print(f"  p_core_speedup: {effective_p_core_speedup}")
    print(f"  random_seed: {'nondeterministic' if effective_random_seed is None else effective_random_seed}")
    print(f"  solver_verbosity: {effective_solver_verbosity}")
    print(f"  time_limit: {effective_time_limit}")
    print(f"  use_profiled: {effective_use_profiled}")
    print(f"  prune_periodic: {effective_prune_periodic}")
    print(f"  restrict_makespan_to_nonperiodic: {effective_restrict_makespan_to_nonperiodic}")
    print(f"  machine_combination_mode: {machine_combination_mode}")
    print(f"  enforce_same_processor_combinations: {enforce_same_processor_combinations}")

    # Build machines list and machine combinations (cumulative core groups per type)
    # machine_combination_mode is now honoured (it used to be parsed and only
    # printed). "singletons" -- its long-standing default -- gives every core its
    # own combination, which is what a multi-core target needs to express real
    # concurrency; "prefix" keeps the cumulative-group reading. The two are
    # identical when every kind has one core, so no pre-K1 config changes.
    # Capability legality, before anything is scheduled.
    #
    # IME is an instruction available on cluster-0 cores only -- measured with a
    # per-core SIGILL probe, artifacts/k1_bringup/*/ime_capability_probe.txt.
    # An IME dispatch on cluster 1 does not run slowly, it traps. That has to be
    # rejected here rather than discovered on the board, and until now it was
    # not: capabilities.py was unit-tested and called by nothing.
    try:
        from capabilities import check_profile_hw_map
        check_profile_hw_map(cfg["profile_hw_map"])
    except ImportError:
        pass  # capabilities.py is K1-specific; other targets need not have it

    # Optional per-dispatch implementation axis (rvv vs ime/NPU). Off by default
    # (spec `scheduler.enable_impls: true` turns it on). When on, each core-group
    # combination is emitted once per legal implementation, so a dispatch that is
    # cheaper on the IME (measured: transformer MLP M>=16 wins 1.3-2.4x over RVV,
    # crossover M~10) can be placed there while attention/GEMV (M<=8) stays on RVV.
    # IME is CLUSTER-0-ONLY and that is enforced structurally: K1_CAPABILITIES
    # gives CPU_E no `ime`, so build_machine_combinations_with_impls emits zero
    # ime combinations on cluster 1 (harts 4-7 SIGILL on smt.vmadot). A combo's
    # ime cost comes from its ime_x60 profile; absent that CSV the cell is excluded
    # (INFEASIBLE_COST) and the solver simply never places there — no free NPU.
    combo_impls = None
    enable_impls = bool(networks_data.get("scheduler", {}).get("enable_impls", False))
    if enable_impls:
        from capabilities import build_machine_combinations_with_impls, K1_CAPABILITIES
        # Expose only rvv+ime as placement choices (scalar is a correctness
        # fallback, never a preferred placement); intersect with each kind's caps.
        machine_impls = {
            k: [i for i in ("rvv", "ime") if i in K1_CAPABILITIES.get(k, frozenset())]
            for k in machine_core_counts
        }
        _gran = "per_core" if machine_combination_mode == "singletons" else machine_combination_mode
        machines, machine_combinations, combo_impls = build_machine_combinations_with_impls(
            machine_core_counts, machine_impls, K1_CAPABILITIES, granularity=_gran)
        n_ime = sum(1 for x in combo_impls if x == "ime")
        print(f"  Impl-aware combinations: {len(machine_combinations)} total, "
              f"{n_ime} ime (cluster-0 only), rest rvv")
    else:
        machines, machine_combinations = build_machine_combinations(
            machine_core_counts, mode=machine_combination_mode)
    n_cores = len(machines)
    transfer_times = np.zeros((n_cores, n_cores))

    # Map each combination to its profile hw and topo tag.
    # The general path uses the full profile_hw_map (machine-kind → hw
    # backend), so configs with three or more machine kinds (e.g.
    # CPU/GPU/HTP) work the same way as the legacy CPU_P/CPU_E split.
    profile_hw_map = cfg["profile_hw_map"]
    combo_hw = []
    for ci, combo in enumerate(machine_combinations):
        core_type = machine_type_prefix(combo[0])
        hw = profile_hw_map.get(core_type.lower())
        if hw is None:
            # Fall back to the legacy two-machine convention.
            hw = cpu_p_profile_hw if core_type == CPU_P else cpu_e_profile_hw
        # For an ime-impl combination, read the IME profile instead of the RVV
        # one: swap the leading rvv->ime in the hw label (rvv_x60 -> ime_x60),
        # which is where the curated IME profiles land. combo_hw drives the CSV
        # lookup in load_profiled_processing_times, so this is the whole plumbing.
        if combo_impls is not None and combo_impls[ci] == "ime" and hw \
                and hw.lower().startswith("rvv"):
            hw = "ime" + hw[3:]  # rvv_x60 -> ime_x60, RVV -> ime
        combo_hw.append(hw)

    # Optional: build profiled processing times if requested
    processing_times: dict[str, list[float]] | None = None
    combined_profiled_p: dict[int, dict] | None = None
    combined_profiled_e: dict[int, dict] | None = None
    profiled_by_network: dict[str, dict[str, dict[int, dict]]] | None = None
    if effective_use_profiled:
        print("\nUsing profiled runtimes where available...")
        # Resolve which override flavor to pass: per-hw dict wins over
        # single-string override.
        if profile_topo_tag_per_hw:
            tt_override = dict(profile_topo_tag_per_hw)
        elif profile_topo_tag_override:
            tt_override = profile_topo_tag
        else:
            tt_override = None
        processing_times, combined_profiled_p, combined_profiled_e, profiled_by_network = load_profiled_processing_times(
            networks=networks,
            repo_base_path=repo_base_path,
            machine_combinations=machine_combinations,
            combo_hw=combo_hw,
            profile_target=profile_target,
            cpu_p_profile_hw=cpu_p_profile_hw,
            cpu_e_profile_hw=cpu_e_profile_hw,
            rng=rng,
            p_core_speedup=effective_p_core_speedup,
            topo_tag_override=tt_override,
            gen_root=gen_root,
        )

    # "Periodic" means the op belongs to a network the workload declared
    # with a `period`.  Not "the op has a time window": a sporadic network
    # carries min_start_t/max_end_t too, so the window test folds yolov8
    # into the periodic set and reports a workload built around it as
    # having no non-periodic work at all.
    periodic_net_ids = {nid for nid, info in networks.items()
                        if info.get("period") is not None}

    def _is_periodic_op(workload, op) -> bool:
        # job_names is indexed by JOB id, not by operation index.
        job_id = getattr(op, "job_id", None)
        if job_id is None or job_id >= len(workload.job_names):
            return False
        jn = workload.job_names[job_id]
        if not isinstance(jn, str):
            return False
        return any(jn.startswith(nid) and jn[len(nid):].isdigit() and jn != nid
                   for nid in periodic_net_ids)

    def _build_workload():
        return create_workload_from_network_hierarchy(
            networks_data=networks_data,
            repo_base_path=repo_base_path,
            machines=machines,
            transfer_times=transfer_times,
            p_core_speedup=effective_p_core_speedup,
            random_seed=effective_random_seed,
            processing_times=processing_times,
            machine_combinations=machine_combinations,
        )

    if solver == "milp":
        # Single global solve. Build workload once, run the selected registry
        # scheduler (default "mosek" == the CVXPY/MOSEK MILP), post-process trim.
        print(f"\nCreating workload from network hierarchy...")
        combined_workload = _build_workload()

        # MILP contention term: the MOSEK/MILP path (unlike greedy) has no notion
        # of co-runner slowdown. Mirror greedy's `base * contention_factor(op,
        # placement)` by folding the measured per-placement multiplier into each
        # (op, combination) processing time BEFORE the solve, so MOSEK optimizes
        # against contention-scaled costs (e.g. cross-cluster placements, measured
        # 1.185x, become genuinely more expensive). Non-circular: the factor keys
        # off the COMBINATION's placement (same/other cluster), not on which ops
        # actually co-run — identical modeling choice to greedy_scheduler._duration.
        # Off unless a contention model is passed (--contention). Never raises.
        if contention_model is not None:
            combos = combined_workload.get_machine_combinations()
            n_scaled = 0
            for op in combined_workload.operations:
                pt = op.processing_times
                for k, combo in enumerate(combos):
                    if k >= len(pt):
                        continue
                    try:
                        placement = contention_model.placement_for_combination(combo)
                        factor = float(contention_model.contention_factor(op, placement))
                    except Exception:
                        factor = 1.0
                    if factor > 0 and factor != 1.0:
                        pt[k] = pt[k] * factor
                        n_scaled += 1
            print(f"  MILP contention: folded measured co-runner factors into "
                  f"{n_scaled} (op,combination) costs")

        print(f"\nWorkload created successfully!")
        print(f"  Total operations: {len(combined_workload.operations)}")
        print(f"  Scheduler machines ({len(combined_workload.machines)} cores): {combined_workload.machines}")
        print(
            f"  Machine combination options: {len(combined_workload.get_machine_combinations())} "
            f"(mode={machine_combination_mode})"
        )
        print(f"  Job names: {combined_workload.job_names}")

        operations_with_multiple_predecessors = [
            op for op in combined_workload.operations if len(op.predecessors) > 1
        ]
        print(f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")
        independent_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
        print(f"  Independent jobs (can run in parallel): {independent_jobs}")

        print("\n" + "=" * 60)
        print(f"Scheduling combined workload (scheduler={scheduler})...")
        print("=" * 60)
        # Route the single-pass solve through the schedulers registry so
        # `--scheduler` can profile any registered algorithm
        # (heft/peft/edf/cpsat/milp_*/...). get_scheduler("mosek") is the
        # CVXPY/MOSEK MILP, so the default behaviour is unchanged.
        scheduler_fn = get_scheduler(scheduler)

        # Freshness-aware objective (opt-in): identify the operations that belong to
        # a producer network named in a freshness_edge, so the MILP can pull their
        # start times early (fresh output for consumers) instead of only minimizing
        # makespan (which delays producers and makes consumers read stale inputs).
        fresh_kwargs = {}
        if freshness_weight and freshness_weight > 0.0:
            producer_tasks = {
                str(e.get("producer_task", "")).lower()
                for e in networks_data.get("freshness_edges", [])
                if e.get("producer_task")
            }
            producer_idx = []
            jn = combined_workload.job_names
            for i, op in enumerate(combined_workload.operations):
                name = ""
                if op.job_id is not None and 0 <= op.job_id < len(jn):
                    name = str(jn[op.job_id]).lower()
                if any(name.startswith(pt) for pt in producer_tasks):
                    producer_idx.append(i)
            print(f"  Freshness-aware: weight={freshness_weight}, producers={sorted(producer_tasks)}, "
                  f"{len(producer_idx)} producer ops pulled early")
            fresh_kwargs = {"freshness_weight": freshness_weight,
                            "freshness_producer_op_indices": producer_idx}

        solver_t0 = time.perf_counter()
        result = scheduler_fn(
            combined_workload,
            solver_verbosity=effective_solver_verbosity,
            time_limit=effective_time_limit,
            restrict_makespan_to_nonperiodic=effective_restrict_makespan_to_nonperiodic,
            prune_cross_period_constraints=effective_prune_periodic,
            **fresh_kwargs,
        )
        solver_wall_time_s = time.perf_counter() - solver_t0
        t, alpha, _, _ = result

        if effective_prune_periodic:
            combined_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(
                combined_workload, t, alpha,
                horizon_ms=networks_data.get("horizon_ms"),
            )
    else:
        # solver == "greedy" or "greedy_periodic": iterative periodic-
        # instance refinement. See greedy_scheduler for the per-pass
        # algorithm; only the picker discipline differs between the two.
        # Loop strategy:
        #   - low-seed: force num_instances=1 per periodic network that
        #     does not ask for a count itself (one that does is pinned to
        #     what it asked for and skips the loop), when
        #     restrict_makespan_to_nonperiodic is set (otherwise the
        #     workload_factory horizon S_np/(1-F_p) inflates the seed
        #     and the joint schedule converges to a bad equilibrium —
        #     non-periodic finishes late *because of* over-allocated
        #     periodic, justifying the over-allocation).
        #   - per pass: build workload, run greedy, measure makespan
        #     over non-periodic ops only (when the flag is set), grow
        #     periodic counts to ceil(max(makespan, horizon)/period) for
        #     any short network; iterate until counts are stable or
        #     `max_periodic_iters` is hit.  "Non-periodic" here means a
        #     network the workload did not declare with a `period` -- a
        #     sporadic one with a window still counts -- see
        #     `_is_periodic_op`.
        #
        # A workload that declares `num_instances` (gen_random_workload
        # emits one per periodic network, sized from the horizon it laid
        # the sporadic tasks into) gets exactly that count: the refinement
        # loop below sizes counts for workloads that DON'T say, and a
        # document that does say has already decided.  Two things went
        # wrong when it did not:
        #   - overwriting the count with 1 and then growing from the
        #     *non-periodic* makespan meant a workload of nothing but
        #     periodic tasks measured a makespan of 0 and converged at one
        #     instance of each network — mlp_control ran once, at t=0, and
        #     never again;
        #   - growing past a declared count undoes the generator's
        #     --cap-instances and --max-ops budgets at schedule time, which
        #     is where the operation count actually costs something.
        solver_t0 = time.perf_counter()
        declared_instances: dict[str, int] = {}
        for net_id, net_info in networks.items():
            if net_info.get("period") is None:
                continue
            declared = net_info.get("num_instances")
            if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
                declared_instances[net_id] = declared
            elif effective_restrict_makespan_to_nonperiodic:
                networks_data["networks"][net_id]["num_instances"] = 1

        # The workload's own span, when it states one: periodic instances
        # cover at least this much even if the non-periodic work finishes
        # earlier (or there is none at all).
        try:
            declared_horizon = max(0.0, float(networks_data.get("horizon_ms") or 0.0))
        except (TypeError, ValueError):
            declared_horizon = 0.0
        if declared_instances or declared_horizon:
            print(f"  Workload declares horizon {declared_horizon:.0f} ms and "
                  f"{len(declared_instances)} explicit periodic instance "
                  f"counts; those counts are used as given. Networks without "
                  f"one are still sized by the refinement loop below.")

        # Pick the per-pass picker function.
        if solver == "greedy_periodic":
            _greedy_fn = greedy_periodic_schedule
        elif solver == "decomposed":
            _greedy_fn = decomposed_schedule
        else:
            _greedy_fn = greedy_schedule

        combined_workload = None
        t = None
        alpha = None
        prev_counts: dict[str, int] = {}
        for it in range(max_periodic_iters):
            print(f"\n--- {solver} iteration {it + 1} ---")
            combined_workload = _build_workload()
            print(f"  Total operations: {len(combined_workload.operations)}")
            print(f"  Job names: {combined_workload.job_names}")

            t, alpha = _greedy_fn(combined_workload)

            # Measure makespan, optionally restricted to non-periodic ops
            # (matches the MILP solver's objective when
            # restrict_makespan_to_nonperiodic is on).
            machine_combinations_iter = combined_workload.get_machine_combinations()
            iter_makespan = 0.0
            iter_makespan_all = 0.0
            for i, op in enumerate(combined_workload.operations):
                combo_idx = int(np.argmax(alpha[i]))
                dur = op.get_duration_for_combination(
                    combo_idx, machine_combinations_iter, combined_workload.machines
                )
                finish = float(t[i]) + float(dur)
                if finish > iter_makespan_all:
                    iter_makespan_all = finish
                if effective_restrict_makespan_to_nonperiodic and \
                        _is_periodic_op(combined_workload, op):
                    continue
                if finish > iter_makespan:
                    iter_makespan = finish
            if effective_restrict_makespan_to_nonperiodic:
                print(f"  Makespan: {iter_makespan:.2f} ms (non-periodic only; "
                      f"all-ops max-end = {iter_makespan_all:.2f} ms)")
            else:
                print(f"  Makespan: {iter_makespan:.2f} ms")

            # Refine periodic counts: each periodic net needs
            # ceil(makespan/period) instances. Don't shrink — periodic
            # workloads only need to grow to cover larger makespans.  A net
            # that declared its own count is pinned to it instead.
            needed_counts: dict[str, int] = {}
            for net_id, net_info in networks.items():
                period = net_info.get("period")
                window_duration = net_info.get("window_duration")
                if period is None or window_duration is None:
                    continue
                try:
                    T = float(period)
                except (TypeError, ValueError):
                    continue
                if T <= 0:
                    continue
                needed = max(1, int(np.ceil(max(iter_makespan, declared_horizon) / T)))
                if net_id in declared_instances:
                    asked = declared_instances[net_id]
                    if needed > asked:
                        print(f"  Periodic '{net_id}': the schedule runs to "
                              f"{max(iter_makespan, declared_horizon):.0f} ms, which would "
                              f"take {needed} instances, but the workload asks "
                              f"for {asked} — keeping {asked}")
                    needed = asked
                current = int(net_info.get("num_instances") or prev_counts.get(net_id, 0))
                if current == 0:
                    current = sum(
                        1 for n in combined_workload.job_names
                        if isinstance(n, str) and n.startswith(net_id) and n[len(net_id):].isdigit()
                    )
                print(f"  Periodic '{net_id}': period={T:.0f}ms current={current} needed={needed}")
                needed_counts[net_id] = needed
                prev_counts[net_id] = current

            if all(prev_counts.get(k, 0) >= v for k, v in needed_counts.items()):
                print("  All periodic counts cover makespan — converged.")
                break
            bumped = []
            for net_id, needed in needed_counts.items():
                cur = prev_counts.get(net_id, 0)
                if cur < needed:
                    networks_data["networks"][net_id]["num_instances"] = needed
                    bumped.append((net_id, cur, needed))
            if not bumped:
                break
            print("  Bumping num_instances:", ", ".join(f"{n}: {a}->{b}" for n, a, b in bumped))
        # A workload of nothing but periodic tasks has a non-periodic
        # makespan of 0 by construction; report the span that means
        # something for it instead of a bare zero.
        # Keep the "<N> ms (after <k> iteration" shape: the sweep parses it.
        if effective_restrict_makespan_to_nonperiodic and iter_makespan <= 0.0:
            print(f"\nFinal greedy makespan: {iter_makespan_all:.2f} ms "
                  f"(after {it + 1} iteration{'s' if it else ''}; over all "
                  f"operations, this workload has no non-periodic work)")
        else:
            print(f"\nFinal greedy makespan: {iter_makespan:.2f} ms (after {it + 1} iteration{'s' if it else ''})")
        solver_wall_time_s = time.perf_counter() - solver_t0

        if effective_prune_periodic:
            combined_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(
                combined_workload, t, alpha,
                horizon_ms=networks_data.get("horizon_ms"),
            )

    # Name of the algorithm actually run (for metrics / report labeling): the
    # registry scheduler on the MILP path, else the greedy-family solver.
    algo_name = scheduler if solver == "milp" else solver
    # Calculate makespan (non-periodic operations only, matching the solver objective)
    machine_combinations = combined_workload.get_machine_combinations()
    completion_times = []
    for i in range(len(combined_workload.operations)):
        op = combined_workload.operations[i]
        combo_idx = int(np.argmax(alpha[i]))
        dur = op.get_duration_for_combination(combo_idx, machine_combinations, combined_workload.machines)
        if not _is_periodic_op(combined_workload, op):
            completion_times.append(float(t[i]) + float(dur))
    makespan = max(completion_times) if completion_times else 0.0

    all_ops_makespan = 0.0
    for i in range(len(combined_workload.operations)):
        op = combined_workload.operations[i]
        combo_idx = int(np.argmax(alpha[i]))
        dur = op.get_duration_for_combination(combo_idx, machine_combinations, combined_workload.machines)
        all_ops_makespan = max(all_ops_makespan, float(t[i]) + float(dur))

    print(f"\nScheduling completed!")
    if completion_times:
        print(f"Makespan (non-periodic): {makespan:.2f} ms "
              f"(all operations: {all_ops_makespan:.2f} ms)")
    else:
        print(f"Makespan (all operations): {all_ops_makespan:.2f} ms "
              f"(no non-periodic work in this workload)")

    # Build combination labels for display
    def _combo_label(combo: list[str]) -> str:
        core_type = machine_type_prefix(combo[0])
        n = len(combo)
        return f"{n}x{core_type}"

    combo_labels = [_combo_label(c) for c in machine_combinations]

    # Count operations assigned to each combination
    combo_counts = {lbl: 0 for lbl in combo_labels}
    for i in range(len(alpha)):
        combo_idx = int(np.argmax(alpha[i]))
        combo_counts[combo_labels[combo_idx]] += 1

    print("\nCombination assignments:")
    for lbl in combo_labels:
        print(f"  {lbl}: {combo_counts[lbl]} operations")

    # Count operations per network (group by job_id)
    network_stats = {}
    for op_idx, op in enumerate(combined_workload.operations):
        job_id = op.job_id
        if job_id not in network_stats:
            network_stats[job_id] = {
                "name": combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}",
                "combo_counts": {lbl: 0 for lbl in combo_labels},
            }

        combo_idx = int(np.argmax(alpha[op_idx]))
        network_stats[job_id]["combo_counts"][combo_labels[combo_idx]] += 1

    print(f"\nPer-network combination assignments:")
    for job_id in sorted(network_stats.keys()):
        stats = network_stats[job_id]
        counts_text = ", ".join(
            f"{lbl}={stats['combo_counts'][lbl]}"
            for lbl in combo_labels if stats['combo_counts'][lbl] > 0
        )
        print(f"  {stats['name'].capitalize()}: {counts_text}")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create title showing the networks. Dedup by base model kind (e.g.
    # "mlp_control0".."mlp_control57" -> one "Mlp_control" entry) -- without
    # this, a schedule with many periodic instances produces a title
    # hundreds of characters long, which forces bbox_inches='tight' to
    # blow up the whole figure's canvas to fit it (observed: a 642-dispatch
    # schedule with 58 mlp_control instances rendered a 26113x1768px image
    # that was almost entirely title whitespace).
    network_names = [combined_workload.job_names[i] if i < len(combined_workload.job_names) else f"Job {i}"
                     for i in sorted(set(op.job_id for op in combined_workload.operations))]
    seen_kinds = []
    for name in network_names:
        kind = plot._kind_from_job_name(name)
        if kind not in seen_kinds:
            seen_kinds.append(kind)
    title_networks = " + ".join([kind.capitalize() for kind in seen_kinds])
    
    # Output naming: solver tag + optional `_profiled` suffix.  MILP
    # historically wrote to `plots/iree_combined_schedule_period.png`
    # (a single hardcoded path that got overwritten across runs); the
    # merged path standardizes to `plots/<base>{_greedy}{_profiled}.png`
    # and `schedules/scheduled_<base>{_greedy}{_profiled}.json`.  MILP
    # outputs intentionally have no `_milp` infix so existing consumers
    # (e.g. xpurt_demo's default SCHEDULE_JSON path) keep working.
    input_json_basename = os.path.basename(networks_json_path)
    input_json_name = os.path.splitext(input_json_basename)[0]
    if solver == "greedy":
        solver_tag = "_greedy"
        title_solver = "Greedy "
    elif solver == "greedy_periodic":
        solver_tag = "_greedy_periodic"
        title_solver = "Greedy-periodic "
    elif solver == "decomposed":
        solver_tag = "_decomposed"
        title_solver = "Decomposed-EDF "
    else:
        # MILP/registry path: default "mosek" keeps no infix for back-compat;
        # other registry schedulers are tagged so a sweep (heft/peft/edf/...)
        # doesn't clobber the canonical mosek outputs.
        solver_tag = "" if scheduler == "mosek" else f"_{scheduler}"
        title_solver = "" if scheduler == "mosek" else f"{scheduler} "
    profiled_tag = "_profiled" if effective_use_profiled else ""
    plot_path = f"plots/{input_json_name}{solver_tag}{profiled_tag}.png"
    json_output_path = f"schedules/scheduled_{input_json_name}{solver_tag}{profiled_tag}.json"
    # Hardware-name labels on the y-axis: instead of "CPU_P#0" / "CPU_E#0",
    # show "gemmini_q31 (CPU_P#0)" / "RVV (CPU_E#0)". Resolves to the
    # bitstream-level identity from the schedule input's profile_hw map.
    # Builds the per-machine-kind label from the generalised profile_hw_map
    # so 3-way (and higher) configs render correctly.
    plot_profile_hw = {k.upper(): v for k, v in profile_hw_map.items()}
    plot_profile_hw.setdefault(CPU_P, cpu_p_profile_hw)
    plot_profile_hw.setdefault(CPU_E, cpu_e_profile_hw)
    # The Gantt render must never be able to destroy the run's data. It used to:
    # this call precedes output_scheduled_json, and on large schedules
    # matplotlib/FreeType raised "raster overflow" while rasterising a glyph,
    # aborting the process after the solve had already succeeded. A sweep lost
    # 14 of 45 cells that way -- every cell at contention B>=3, i.e. exactly the
    # oversubscribed points the experiment exists to measure.
    #
    # plot.py now scales dpi down and retries, so this should be rare; the guard
    # stays because a cosmetic artifact is never worth a solved schedule. The
    # failure is printed loudly rather than swallowed, since a per-cell Gantt is
    # a required deliverable and a silently missing one would be worse than a
    # noisy one.
    plot_ok, plot_error = True, None
    try:
        plot.plot_optimization_schedule(
            combined_workload.get_durations(),
            t,
            alpha,
            num_jobs,
            len(combined_workload.machines),
            combined_workload.machines,
            combined_workload.get_transfer_times(),
            save_path=plot_path,
            plot_title=f"{title_networks} {title_solver}Schedule ({n_cores} cores) (Periodic)",
            workload=combined_workload,
            profile_hw=plot_profile_hw,
        )
        print(f"\nPlot saved to {plot_path}")
    except Exception as exc:  # noqa: BLE001 - a plot must not abort the solve
        plot_ok, plot_error = False, f"{type(exc).__name__}: {exc}"
        print(f"\nWARN: Gantt plot FAILED and was skipped: {plot_error}")
        print(f"WARN: the schedule itself is unaffected and will still be "
              f"written to {json_output_path}")

    os.makedirs("schedules", exist_ok=True)
    print(f"\nOutputting scheduled JSON...")
    # Snapshot the CSVs the loader read this run and hash them.
    # Embed both in the fixture metadata so the runtime loader can
    # detect when the PDB-on-disk has drifted from the PDB the solve
    # was performed against — the trap that produced v8's 9x
    # predicted/measured gap.
    _pdb_hash, _pdb_files = compute_pdb_hash(list(_LAST_LOAD_CSV_PATHS))
    print(f"  pdb_hash = sha256:{_pdb_hash[:16]}... over "
          f"{len(_pdb_files)} CSV(s)")
    output_scheduled_json(
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=combined_profiled_p,
        profiled_times_e=combined_profiled_e,
        profile_hw=plot_profile_hw,
        profiled_times_by_network=profiled_by_network,
        pdb_hash=_pdb_hash,
        pdb_files=_pdb_files,
        combo_impls=combo_impls,
    )

    # PER-DISPATCH RUNTIME FEEDBACK. `derive_dispatch_hints` wants the
    # solver's (t, alpha) directly. They are in hand here, which is why this
    # lives in the solver rather than in a script that reads the written
    # schedule back: reconstructing alpha from a serialized schedule means
    # inferring a one-hot assignment from a machine label, and any dispatch
    # whose label is ambiguous silently becomes a hint about the wrong
    # combination.
    #
    # This is the channel merlin consumed through merlin_adapter, which is
    # retired. Nothing else called `feedback.py`.
    if getattr(args, "emit_feedback", False):
        import feedback as _feedback
        _payload = _feedback.derive_dispatch_hints(
            combined_workload, t, alpha,
            run_id=args.feedback_run_id,
            source_schedule=os.path.basename(json_output_path),
        )
        _fb_path = os.path.join(os.path.dirname(json_output_path) or ".",
                                "xpurt_feedback.json")
        _feedback.write_feedback_json(_payload, pathlib.Path(_fb_path))
        print(f"feedback -> {_fb_path}  "
              f"({len(_payload.get('dispatches', {}))} dispatches with hints)")

    # Feedback-driven compilation: surface any dispatch-granularity mismatch
    # between periodic and non-periodic jobs in this schedule. Advisory
    # only -- xpu-rt can't split a coarse dispatch itself; this just flags
    # it so a human (or an upstream partitioner) can act on it. Same signal
    # is already embedded in the JSON's metadata["granularity_advice"]; this
    # just makes it visible without opening the file.
    try:
        advice = analyze_granularity(from_workload(combined_workload, t, alpha))
        for a in advice:
            if a.recommended != "unchanged":
                print(f"WARN: granularity advisor -- {a.reason}")
    except Exception as e:
        print(f"warning: granularity advisor failed ({e})")
    # Emit per-run metrics next to the schedule JSON (additive — does not affect
    # the regression diff on the schedule JSON itself).
    try:
        import json as _json
        metrics_dict = compute_metrics(
            combined_workload,
            t,
            alpha,
            scheduler_name=algo_name,
            solver_wall_time_s=solver_wall_time_s,
        )
        metrics_path = json_output_path.replace(".json", "_metrics.json")
        with open(metrics_path, "w") as f:
            _json.dump(metrics_dict, f, indent=2)
        print(f"Metrics written to: {metrics_path}")
        # `op_deadline_miss`, spelled out. `metrics.py` counts misses PER
        # DISPATCH and says so at length, but it also keeps a
        # `deadline_miss_count` alias for compatibility -- and this line used
        # to print that alias as a bare "deadline_miss". A reader takes that
        # for the number of late INSTANCES, which is the quantity a frequency
        # claim rests on, and the two differ by more than an order of
        # magnitude: one B4 cell reports 229 late dispatches and 13 late
        # instances out of 47. The alias is fine in the JSON, where the
        # op_-prefixed name sits beside it; on a summary line with no context
        # it is a trap, and it caught me.
        print(f"  makespan_us={metrics_dict['makespan_us']:.2f}  "
              f"op_deadline_miss={metrics_dict['op_deadline_miss_count']}"
              f" (dispatches, NOT instances)  "
              f"cross_dev={metrics_dict['cross_device_transitions']}  "
              f"solver_s={solver_wall_time_s:.3f}")
    except Exception as exc:
        print(f"[warn] metrics emission failed: {exc}")

    # Emit a structured SchedulerReport (schema v2, with the per-dispatch list)
    # next to the schedule JSON, so the scheduler advisor and terminal Gantt can
    # consume real runs. Additive and best-effort.
    try:
        from profiling import SchedulerReport
        report = SchedulerReport.from_solver_state(
            combined_workload,
            t,
            alpha,
            solver_name=algo_name,
            solve_wall_s=solver_wall_time_s,
            solver_status="feasible",
        )
        report_path = json_output_path.replace(".json", "_report.json")
        report.write_json(report_path)
        print(f"Scheduler report written to: {report_path}")
    except Exception as exc:
        print(f"[warn] scheduler report emission failed: {exc}")

    return combined_workload, t, alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE networks from hierarchical network dependencies JSON file"
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default="data/toplevel/networks_periodic_profile.json",
        help="Path to the top-level networks dependencies JSON file (default: data/toplevel/networks_periodic_profile.json)"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="milp",
        choices=["milp", "greedy", "greedy_periodic", "decomposed"],
        help="Scheduling algorithm. 'milp' (default) is the global cvxpy/mosek "
             "solver. 'greedy' is a list-scheduling heuristic with iterative "
             "periodic-instance refinement — fast, no external solver needed, "
             "and suitable for large workloads where the MILP times out. "
             "'greedy_periodic' is the same loop but the per-iter picker "
             "prioritizes non-periodic ops over periodic ones (use for "
             "heterogeneous workloads where the non-periodic critical path "
             "shouldn't be fragmented by periodic instances).",
    )
    parser.add_argument(
        "--max-periodic-iters",
        type=int,
        default=4,
        help="(greedy only) cap on iterations of the periodic-instance "
             "refinement loop (default: 4).",
    )
    parser.add_argument(
        "--solver-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="(milp only) MOSEK solver verbosity level: 0=silent, 1=errors, "
             "2=warnings, 3=info, 4=detailed progress.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20,
        help="(milp only) Maximum optimization time in seconds (override).",
    )
    parser.add_argument(
        "--profiled",
        dest="profiled",
        action="store_true",
        default=True,
        help="Enable profiled runtimes (default: enabled).",
    )
    # Compatibility alias for the previous run_greedy_schedule.py CLI.
    parser.add_argument(
        "--use-profiled",
        dest="profiled",
        action="store_true",
        help="Compat alias for --profiled.",
    )
    parser.add_argument(
        "--no-profiled",
        dest="profiled",
        action="store_false",
        help="Disable profiled runtimes.",
    )
    parser.add_argument(
        "--prune-periods",
        dest="prune_periodic",
        action="store_true",
        default=None,
        help="Enable pruning of periodic operations (override).",
    )
    parser.add_argument(
        "--no-prune-periods",
        dest="prune_periodic",
        action="store_false",
        help="Disable pruning of periodic operations (override).",
    )
    parser.add_argument(
        "--restrict-makespan-to-nonperiodic",
        dest="restrict_makespan_to_nonperiodic",
        action="store_true",
        default=None,
        help="Restrict makespan objective to non-periodic operations (override).",
    )
    parser.add_argument(
        "--include-periodic-in-makespan",
        dest="restrict_makespan_to_nonperiodic",
        action="store_false",
        help="Include periodic operations in makespan objective (override).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for synthetic runtime generation. If omitted, uses JSON config (scheduler.random_seed/seed) or 0. Use -1 for nondeterministic.",
    )
    parser.add_argument(
        "--p-core-speedup",
        type=float,
        default=None,
        help="Override P-core speedup factor. If omitted, uses JSON config (hardware/scheduler p_core_speedup) or 1.5.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="mosek",
        choices=available_schedulers(),
        help="Scheduler to use. Default 'mosek' preserves the existing CVXPY/MOSEK pipeline; additional baselines are added in later milestones.",
    )
    parser.add_argument(
        "--contention",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Apply measured co-runner contention multipliers to the greedy "
            "duration lookup. OFF by default. Bare flag uses "
            "artifacts/k1_run/contention.json; pass a path to override. A "
            "missing artifact is a no-op."
        ),
    )
    parser.add_argument(
        "--freshness-weight",
        type=float,
        default=0.0,
        help=(
            "Freshness-aware objective weight (MILP only). 0.0 (default) = pure "
            "makespan. A positive value adds w * sum(producer start times) to the "
            "objective, pulling operations of any network named as a producer_task "
            "in the spec's freshness_edges as early as possible, so consumers read "
            "fresh inputs. Minimizing makespan alone delays producers and yields "
            "stale outputs; this term counteracts that."
        ),
    )
    parser.add_argument(
        "--emit-feedback",
        action="store_true",
        help=(
            "Also write xpurt_feedback.json beside the schedule: per-dispatch "
            "RUNTIME hints (prefer_coarser / prefer_finer / "
            "consider_fuse_with_pred / pin_target / consider_split_backend) "
            "derived from the solved schedule. This is the other feedback "
            "channel -- compile_advice.json says how to REWRITE the graph, "
            "this says how to place and size what is already there. Off by "
            "default: without it the run is byte-identical to before."
        ),
    )
    parser.add_argument(
        "--feedback-run-id",
        default=None,
        help="run_id recorded in xpurt_feedback.json (default: UTC timestamp). "
             "The ingest merges hints by set-union on the same run_id, so "
             "repeated emissions during one campaign accumulate.",
    )
    args = parser.parse_args()

    # Contention is additive and off unless asked for: installing None here
    # leaves the schedulers on the plain solo profile.
    _model = None
    if args.contention is not None:
        import contention_model
        import greedy_scheduler

        _path = None if args.contention is True else args.contention
        _model = contention_model.load(_path)
        if _model is None:
            print(
                f"--contention: no artifact at "
                f"{_path or contention_model.DEFAULT_PATH}; running without it"
            )
        else:
            print(
                f"--contention: loaded {_model.path} "
                f"(placements: {', '.join(_model.placements())})"
            )
        greedy_scheduler.configure_contention(_model)

    seed = None if args.random_seed is not None and args.random_seed < 0 else args.random_seed

    schedule_iree_networks(
        networks_json_path=args.networks_json,
        solver=args.solver,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
        p_core_speedup=args.p_core_speedup,
        use_profiled=args.profiled,
        prune_periodic=args.prune_periodic,
        restrict_makespan_to_nonperiodic=args.restrict_makespan_to_nonperiodic,
        scheduler=args.scheduler,
        max_periodic_iters=args.max_periodic_iters,
        freshness_weight=args.freshness_weight,
        contention_model=_model,
    )
