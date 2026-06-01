"""
Scheduling script for IREE dispatch graphs from hierarchical network dependencies.
Parses top-level network dependency JSON files and schedules them onto
CPU_P (performant) and CPU_E (efficient) cores.
"""

from __future__ import annotations

import sys
import os
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
from profile_loader import load_profiled_processing_times
from postprocessing import trim_periodic_after_nonperiodic_makespan, output_scheduled_json
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
    machines, machine_combinations = build_machine_combinations(machine_core_counts)
    n_cores = len(machines)
    transfer_times = np.zeros((n_cores, n_cores))

    # Map each combination to its profile hw and topo tag.
    # The general path uses the full profile_hw_map (machine-kind → hw
    # backend), so configs with three or more machine kinds (e.g.
    # CPU/GPU/HTP) work the same way as the legacy CPU_P/CPU_E split.
    profile_hw_map = cfg["profile_hw_map"]
    combo_hw = []
    for combo in machine_combinations:
        core_type = machine_type_prefix(combo[0])
        hw = profile_hw_map.get(core_type.lower())
        if hw is None:
            # Fall back to the legacy two-machine convention.
            hw = cpu_p_profile_hw if core_type == CPU_P else cpu_e_profile_hw
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
        )

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
        solver_t0 = time.perf_counter()
        result = scheduler_fn(
            combined_workload,
            solver_verbosity=effective_solver_verbosity,
            time_limit=effective_time_limit,
            restrict_makespan_to_nonperiodic=effective_restrict_makespan_to_nonperiodic,
            prune_cross_period_constraints=effective_prune_periodic,
        )
        solver_wall_time_s = time.perf_counter() - solver_t0
        t, alpha, _, _ = result

        if effective_prune_periodic:
            combined_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(
                combined_workload, t, alpha
            )
    else:
        # solver == "greedy" or "greedy_periodic": iterative periodic-
        # instance refinement. See greedy_scheduler for the per-pass
        # algorithm; only the picker discipline differs between the two.
        # Loop strategy:
        #   - low-seed: force num_instances=1 per periodic network when
        #     restrict_makespan_to_nonperiodic is set (otherwise the
        #     workload_factory horizon S_np/(1-F_p) inflates the seed
        #     and the joint schedule converges to a bad equilibrium —
        #     non-periodic finishes late *because of* over-allocated
        #     periodic, justifying the over-allocation).
        #   - per pass: build workload, run greedy, measure makespan
        #     over non-periodic ops only (when the flag is set), grow
        #     periodic counts to ceil(makespan/period) for any short
        #     network; iterate until counts are stable or
        #     `max_periodic_iters` is hit.
        solver_t0 = time.perf_counter()
        if effective_restrict_makespan_to_nonperiodic:
            for net_id, net_info in networks.items():
                if net_info.get("period") is not None:
                    networks_data["networks"][net_id]["num_instances"] = 1

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
            periodic_net_ids = {
                nid for nid, info in networks.items()
                if info.get("period") is not None
            }
            def _is_periodic_op(op_idx: int) -> bool:
                jn = (combined_workload.job_names[op_idx]
                      if op_idx < len(combined_workload.job_names) else "")
                if not isinstance(jn, str):
                    return False
                for nid in periodic_net_ids:
                    if jn.startswith(nid) and jn[len(nid):].isdigit() and jn != nid:
                        return True
                return False
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
                if effective_restrict_makespan_to_nonperiodic and _is_periodic_op(i):
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
            # workloads only need to grow to cover larger makespans.
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
                needed = max(1, int(np.ceil(iter_makespan / T)))
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
        print(f"\nFinal greedy makespan: {iter_makespan:.2f} ms (after {it + 1} iteration{'s' if it else ''})")
        solver_wall_time_s = time.perf_counter() - solver_t0

        if effective_prune_periodic:
            combined_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(
                combined_workload, t, alpha
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
        is_periodic = (op.min_start_t is not None) or (op.max_end_t is not None)
        if not is_periodic:
            completion_times.append(float(t[i]) + float(dur))
    makespan = max(completion_times) if completion_times else 0.0

    print(f"\nScheduling completed!")
    print(f"Makespan (non-periodic): {makespan:.2f} ms")

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
    
    # Create title showing the networks
    network_names = [combined_workload.job_names[i] if i < len(combined_workload.job_names) else f"Job {i}" 
                     for i in sorted(set(op.job_id for op in combined_workload.operations))]
    title_networks = " + ".join([name.capitalize() for name in network_names])
    
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

    os.makedirs("schedules", exist_ok=True)
    print(f"\nOutputting scheduled JSON...")
    output_scheduled_json(
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=combined_profiled_p,
        profiled_times_e=combined_profiled_e,
        profile_hw=plot_profile_hw,
        profiled_times_by_network=profiled_by_network,
    )

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
        print(f"  makespan_us={metrics_dict['makespan_us']:.2f}  "
              f"deadline_miss={metrics_dict['deadline_miss_count']}  "
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
    args = parser.parse_args()

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
    )
