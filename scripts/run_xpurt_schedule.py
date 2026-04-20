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

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import (
    create_workload_from_network_hierarchy,
    build_machine_combinations,
    machine_type_prefix,
)
from scheduler import schedule
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

    cfg = {
        "cpu_p_profile_hw": profile_hw.get("cpu_p", "RVV"),
        "cpu_e_profile_hw": profile_hw.get("cpu_e", "scalar"),
        "profile_target": profile_cfg.get("target", "spacemit_x60"),
        "profile_topo_tag": profile_cfg.get("topo_tag", "topo_0_1_2_3"),
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
    solver_verbosity: int | None = None,
    time_limit: float | None = None,
    random_seed: int | None = None,
    p_core_speedup: float | None = None,
    use_profiled: bool | None = None,
    prune_periodic: bool | None = None,
    restrict_makespan_to_nonperiodic: bool | None = None,
) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Main function to schedule networks from a hierarchical network dependencies JSON file.
    CLI arguments override JSON config values when not None.
    """
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

    # Map each combination to its profile hw and topo tag
    # e.g. [CPU_P#0, CPU_P#1] → hw="RVV", topo="topo_0_1"
    combo_hw = []
    for combo in machine_combinations:
        core_type = machine_type_prefix(combo[0])
        if core_type == CPU_P:
            combo_hw.append(cpu_p_profile_hw)
        else:
            combo_hw.append(cpu_e_profile_hw)

    # Optional: build profiled processing times if requested
    processing_times: dict[str, list[float]] | None = None
    combined_profiled_p: dict[int, dict] | None = None
    combined_profiled_e: dict[int, dict] | None = None
    if effective_use_profiled:
        print("\nUsing profiled runtimes where available...")
        processing_times, combined_profiled_p, combined_profiled_e = load_profiled_processing_times(
            networks=networks,
            repo_base_path=repo_base_path,
            machine_combinations=machine_combinations,
            combo_hw=combo_hw,
            profile_target=profile_target,
            cpu_p_profile_hw=cpu_p_profile_hw,
            cpu_e_profile_hw=cpu_e_profile_hw,
            rng=rng,
            p_core_speedup=effective_p_core_speedup,
        )

    # Create workload from network hierarchy
    print(f"\nCreating workload from network hierarchy...")
    combined_workload = create_workload_from_network_hierarchy(
        networks_data=networks_data,
        repo_base_path=repo_base_path,
        machines=machines,
        transfer_times=transfer_times,
        p_core_speedup=effective_p_core_speedup,
        random_seed=effective_random_seed,
        processing_times=processing_times,
        machine_combinations=machine_combinations,
    )
    
    print(f"\nWorkload created successfully!")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Scheduler machines ({len(combined_workload.machines)} cores): {combined_workload.machines}")
    print(
        f"  Machine combination options: {len(combined_workload.get_machine_combinations())} "
        f"(mode={machine_combination_mode})"
    )
    print(f"  Job names: {combined_workload.job_names}")
    
    # Print some statistics
    operations_with_multiple_predecessors = [
        op for op in combined_workload.operations if len(op.predecessors) > 1
    ]
    print(f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")
    
    # Count independent jobs (operations with no predecessors)
    independent_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    print(f"  Independent jobs (can run in parallel): {independent_jobs}")
    
    # Schedule the combined workload
    print("\n" + "=" * 60)
    print("Scheduling combined workload...")
    print("=" * 60)
    result = schedule(
        combined_workload,
        solver_verbosity=effective_solver_verbosity,
        time_limit=effective_time_limit,
        restrict_makespan_to_nonperiodic=effective_restrict_makespan_to_nonperiodic,
        prune_cross_period_constraints=effective_prune_periodic,
    )
    t, alpha, _, _ = result  # Always returns 4 values now
    
    # Post-process schedule: optionally trim periodic/background operations that
    # occur entirely after the non-periodic makespan.
    if effective_prune_periodic:
        combined_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(combined_workload, t, alpha)
    
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
    
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path="plots/iree_combined_schedule_period.png",
        plot_title=f"{title_networks} Schedule ({n_cores} cores) (Periodic)",
        workload=combined_workload
    )
    
    print(f"\nPlot saved to plots/iree_combined_schedule_period.png")
    
    # Output combined JSON file with scheduling information
    os.makedirs("schedules", exist_ok=True)
    # Extract base filename from input JSON path
    input_json_basename = os.path.basename(networks_json_path)
    input_json_name = os.path.splitext(input_json_basename)[0]  # Remove .json extension
    json_output_path = f"schedules/scheduled_{input_json_name}.json"
    if effective_use_profiled:
        json_output_path = json_output_path.replace(".json", "_profiled.json")
    
    print(f"\nOutputting scheduled JSON...")
    output_scheduled_json(
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=combined_profiled_p,
        profiled_times_e=combined_profiled_e
    )
    
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
        "--solver-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="MOSEK solver verbosity level override: 0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress. If omitted, use JSON/default.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20,
        help="Maximum optimization time in seconds (override). If omitted, use JSON/default.",
    )
    parser.add_argument(
        "--profiled",
        dest="profiled",
        action="store_true",
        default=True,
        help="Enable profiled runtimes (default: enabled).",
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
    args = parser.parse_args()
    
    seed = None if args.random_seed is not None and args.random_seed < 0 else args.random_seed
    
    schedule_iree_networks(
        networks_json_path=args.networks_json,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
        p_core_speedup=args.p_core_speedup,
        use_profiled=args.profiled,
        prune_periodic=args.prune_periodic,
        restrict_makespan_to_nonperiodic=args.restrict_makespan_to_nonperiodic,
    )
