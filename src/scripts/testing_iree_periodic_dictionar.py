"""
Test script for scheduling IREE dispatch graphs from hierarchical network dependencies.
Parses top-level network dependency JSON files and schedules them on CPU_P (performant) 
and CPU_E (efficient) cores. CPU_P is 1.5x faster than CPU_E.
"""

import sys
import os
import json
import argparse
import csv
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_network_hierarchy
from scheduler import schedule
import plot

def load_networks_graph(json_path: str) -> dict:
    """Load a network dependencies JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_hardware_specification(json_path: str) -> dict:
    """Load a hardware specification JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_profiled_times(csv_path: str) -> dict[int, dict]:
    """
    Load profiled runtimes from a CSV file.

    Expected columns:
      - dispatch_id
      - module_name (optional)
      - mean_time
      - mean_unit (assumed 'ms' if missing)
    
    Returns:
      dict mapping dispatch_id (int) -> {"time_ms": float, "module_name": str}
    """
    profiled: dict[int, dict] = {}
    if not os.path.exists(csv_path):
        return profiled
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dispatch_id_str = row.get("dispatch_id")
            if not dispatch_id_str:
                continue
            try:
                dispatch_id = int(dispatch_id_str)
            except ValueError:
                continue
            
            module_name = row.get("module_name", "")
            try:
                mean_time = float(row.get("mean_time", 0.0))
            except ValueError:
                continue
            unit = row.get("mean_unit", "ms")
            if unit == "us":
                mean_time_ms = mean_time / 1000.0
            elif unit == "s":
                mean_time_ms = mean_time * 1000.0
            else:
                mean_time_ms = mean_time
            profiled[dispatch_id] = {
                "time_ms": mean_time_ms,
                "module_name": module_name,
            }
    return profiled

def output_scheduled_json(
    combined_workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    output_path: str,
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None
):
    """
    Output a combined JSON file with all dispatches, their hardware targets, and start times.
    
    Args:
        combined_workload: Combined workload after scheduling
        t: Start times array from scheduling
        alpha: Assignment matrix from scheduling
        output_path: Path to save the output JSON file
        profiled_times_p: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for P-core
        profiled_times_e: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for E-core
    """
    machine_combinations = combined_workload.get_machine_combinations()
    
    # First pass: collect all dispatch info with completion times
    dispatch_info_list = []
    
    for op_idx in range(len(combined_workload.operations)):
        op = combined_workload.operations[op_idx]
        
        # Get dispatch name from operation
        dispatch_name = op.operation_name if hasattr(op, 'operation_name') and op.operation_name else f"op_{op_idx}"
        
        # Get hardware target (which combination was assigned)
        combo_idx = np.argmax(alpha[op_idx])
        hardware_target = "+".join(machine_combinations[combo_idx]) if len(machine_combinations[combo_idx]) > 1 else machine_combinations[combo_idx][0]
        
        # Get start time
        start_time = float(t[op_idx])
        
        # Get duration for the assigned combination
        duration = op.get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )
        
        # Get dispatch ID
        dispatch_id = op.operation_id if hasattr(op, 'operation_id') and op.operation_id is not None else op_idx
        
        # Get job name
        job_id = op.job_id if hasattr(op, 'job_id') and op.job_id is not None else 0
        job_name = combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}"
        
        # Get module name from profiled data if available
        module_name = None
        if profiled_times_p and isinstance(dispatch_id, int) and dispatch_id in profiled_times_p:
            module_name = profiled_times_p[dispatch_id].get("module_name")
        elif profiled_times_e and isinstance(dispatch_id, int) and dispatch_id in profiled_times_e:
            module_name = profiled_times_e[dispatch_id].get("module_name")
        
        completion_time = start_time + float(duration)
        
        dispatch_info_list.append({
            'op_idx': op_idx,
            'dispatch_name': dispatch_name,
            'dispatch_id': dispatch_id,
            'hardware_target': hardware_target,
            'start_time': start_time,
            'duration': float(duration),
            'completion_time': completion_time,
            'job_name': job_name,
            'module_name': module_name,
            'op': op,
        })
    
    # Build time dependency mapping: for each hardware target, track dispatches sorted by completion time
    hardware_dispatch_map = {}  # hardware_target -> list of (completion_time, dispatch_name, start_time)
    
    for info in dispatch_info_list:
        hw_target = info['hardware_target']
        if hw_target not in hardware_dispatch_map:
            hardware_dispatch_map[hw_target] = []
        hardware_dispatch_map[hw_target].append((
            info['completion_time'],
            info['dispatch_name'],
            info['start_time']
        ))
    
    # Sort each hardware target's dispatches by completion time
    for hw_target in hardware_dispatch_map:
        hardware_dispatch_map[hw_target].sort(key=lambda x: x[0])  # Sort by completion_time
    
    # Build combined dispatches dictionary
    combined_dispatches = {}
    
    for info in dispatch_info_list:
        dispatch_name = info['dispatch_name']
        hardware_target = info['hardware_target']
        start_time = info['start_time']
        op = info['op']
        
        # Get dependencies (from operation predecessors)
        dependencies = []
        for pred_op in op.predecessors:
            # Find the index of this predecessor in the combined workload
            pred_idx = None
            for idx, combined_operation in enumerate(combined_workload.operations):
                if combined_operation == pred_op:
                    pred_idx = idx
                    break
            if pred_idx is not None:
                pred_dispatch_name = combined_workload.operations[pred_idx].operation_name if hasattr(combined_workload.operations[pred_idx], 'operation_name') and combined_workload.operations[pred_idx].operation_name else f"op_{pred_idx}"
                dependencies.append(pred_dispatch_name)
        
        # Find time dependency: previous dispatch on same hardware target
        time_dependency = None
        if hardware_target in hardware_dispatch_map:
            hw_dispatches = hardware_dispatch_map[hardware_target]
            # Find the dispatch that finished most recently before this one starts
            for completion_time, prev_dispatch_name, prev_start_time in hw_dispatches:
                if completion_time <= start_time and prev_dispatch_name != dispatch_name:
                    time_dependency = prev_dispatch_name
                elif completion_time > start_time:
                    break  # No need to check further (sorted by completion time)
        
        # Create dispatch entry
        dispatch_entry = {
            "id": info['dispatch_id'],
            "ordinal": 1,  # Keep original structure
            "total": 1,
            "dependencies": dependencies,
            "hardware_target": hardware_target,
            "start_time": start_time,
            "duration": info['duration'],
            "job_name": info['job_name']
        }
        
        # Add module_name if available
        if info['module_name']:
            dispatch_entry["module_name"] = info['module_name']
        
        # Add time_dependency if found
        if time_dependency:
            dispatch_entry["time_dependency"] = time_dependency
        
        combined_dispatches[dispatch_name] = dispatch_entry
    
    # Create output JSON structure
    output_data = {
        "dot_file": "combined_schedule_periodic.json",
        "dispatches": combined_dispatches,
        "metadata": {
            "makespan": float(max(
                t[i] + combined_workload.operations[i].get_duration_for_combination(
                    np.argmax(alpha[i]), machine_combinations, combined_workload.machines
                )
                for i in range(len(combined_workload.operations))
            )),
            "num_operations": len(combined_workload.operations),
            "machines": combined_workload.machines,
            "machine_combinations": [combo if isinstance(combo, list) else [combo] for combo in machine_combinations]
        }
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nScheduled JSON saved to: {output_path}")


def _trim_periodic_after_nonperiodic_makespan(workload: Workload, t: np.ndarray, alpha: np.ndarray) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Post-process the schedule to discard periodic/background operations that occur
    entirely after the last non-periodic operation completes.
    
    An operation is considered periodic/background if it has a time-window bound
    (min_start_t or max_end_t set). Non-periodic operations have both as None.
    
    We:
      1) Compute the makespan over non-periodic operations only.
      2) Drop any periodic operation whose window starts at or after this makespan.
         (i.e., its period does not overlap the non-periodic makespan interval).
    """
    if t is None or alpha is None or len(workload.operations) == 0:
        return workload, t, alpha

    # 1) Compute makespan over non-periodic operations
    nonperiodic_completion_times: list[float] = []
    for i, op in enumerate(workload.operations):
        is_periodic = (getattr(op, "min_start_t", None) is not None) or (getattr(op, "max_end_t", None) is not None)
        if is_periodic:
            continue
        # Completion time based on chosen machine
        combo_idx = int(np.argmax(alpha[i]))
        dur = op.get_duration_for_combination(combo_idx, workload.get_machine_combinations(), workload.machines)
        nonperiodic_completion_times.append(float(t[i] + dur))

    if not nonperiodic_completion_times:
        # No non-periodic ops: nothing to trim
        return workload, t, alpha

    nonperiodic_makespan = max(nonperiodic_completion_times)

    # 2) Build keep mask: always keep non-periodic ops; for periodic, keep only
    #    those whose window overlaps [0, nonperiodic_makespan).
    keep_indices: list[int] = []
    for i, op in enumerate(workload.operations):
        min_start_t = getattr(op, "min_start_t", None)
        max_end_t = getattr(op, "max_end_t", None)
        is_periodic = (min_start_t is not None) or (max_end_t is not None)

        if not is_periodic:
            keep_indices.append(i)
            continue

        # If no explicit window, treat as non-periodic (already handled above).
        if min_start_t is None or max_end_t is None:
            keep_indices.append(i)
            continue

        # Period window [min_start_t, max_end_t) overlaps [0, nonperiodic_makespan) iff:
        #   min_start_t < nonperiodic_makespan and max_end_t > 0
        if (min_start_t < nonperiodic_makespan) and (max_end_t > 0):
            keep_indices.append(i)
        # else: drop this periodic op (it is entirely after the relevant horizon)

    if len(keep_indices) == len(workload.operations):
        # Nothing trimmed
        return workload, t, alpha

    # Build trimmed workload and schedule arrays
    trimmed_ops = [workload.operations[i] for i in keep_indices]
    trimmed_t = np.array([t[i] for i in keep_indices])
    trimmed_alpha = np.array([alpha[i] for i in keep_indices])

    trimmed_workload = Workload(
        trimmed_ops,
        workload.machines,
        workload.transfer_times,
        job_names=workload.job_names,
        machine_combinations=workload.machine_combinations,
    )

    return trimmed_workload, trimmed_t, trimmed_alpha

def schedule_iree_networks(
    networks_json_path: str = None,
    hardware_json_path: str = None,
    solver_verbosity: int = 0,
    time_limit: float = None,
    random_seed: int | None = 0,
    use_profiled: bool = False,
    prune_periodic: bool = True,
    restrict_makespan_to_nonperiodic: bool = True,
) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Main function to schedule networks from a hierarchical network dependencies JSON file.
    
    Parameters:
    - hardware_json_path: Path to a hardware specification JSON file (not currently used, but can be extended to define machines)s
    - networks_json_path: Path to the top-level networks dependencies JSON file.
                          If None, uses the default networks_deps.json file.
    - solver_verbosity: MOSEK solver verbosity level (0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress).
    - time_limit: Maximum optimization time in seconds. MOSEK will return the best solution found within this time limit.
    """
    # Get script directory and repo base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_base_path = os.path.join(script_dir, '..', '..')
    
    # Use default networks JSON if not provided
    if networks_json_path is None:
        networks_json_path = os.path.join(
        script_dir, 
        '..', 
            'data',
            'toplevel',
            'networks_periodic.json'
        )
    
    # Resolve to absolute path
    if not os.path.isabs(networks_json_path):
        networks_json_path = os.path.join(repo_base_path, networks_json_path)

    if hardware_json_path is None:
        hardware_json_path = os.path.join(
            script_dir,
            '..',
            'data',
            'hardware',
            'hardware_spec.json'
        )
    if not os.path.isabs(hardware_json_path):
        hardware_json_path = os.path.join(repo_base_path, hardware_json_path)
    
    print("=" * 60)
    print("Loading network hierarchy from JSON...")
    print("=" * 60)
    print(f"\nLoading networks from: {networks_json_path}")
    
    # Load network dependencies JSON
    networks_data = load_networks_graph(networks_json_path)
    hardware_data = load_hardware_specification(hardware_json_path) 

    # Print network information
    networks = networks_data.get('networks', {})
    edges = networks_data.get('edges', [])
    print(f"\nFound {len(networks)} networks:")
    for network_id, network_info in networks.items():
        print(f"  - {network_id} (id: {network_info.get('id')}, identifier: {network_info.get('identifier')})")
    
    print(f"\nFound {len(edges)} network-level dependencies:")
    for edge in edges:
        print(f"  - {edge.get('from')} → {edge.get('to')}")
    
    #Print hardware information
    print(f"\nHardware specification:")
    machines ={}
    hardware_types = hardware_data.get('hardware', 0)
    for machineid, machine_info in hardware_types.items():
        machines[machineid] = machine_info.get('cores', 0)
    
    printf(f"  Machines: {machines}")
    
    # Define machines (dual-core device)
    # Example with 2 CPU cores, 3 GPU cores, and 1 NPU core. Adjust as needed.
    # machines = {"CPU": 2, "GPU":3, "NPU": 1 }
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))

    # Optional: build profiled processing times if requested
    processing_times: dict[str, list[float]] | None = None
    combined_profiled_p: dict[int, dict] | None = None
    combined_profiled_e: dict[int, dict] | None = None
    if use_profiled:
        print("\nUsing profiled runtimes where available...")
        processing_times = {}
        combined_profiled_p = {}
        combined_profiled_e = {}

        # Paths to profiled runtimes (P-core and E-core) for known networks
        dronet_profile_csv_p = os.path.join(
            script_dir,
            "..",
            "data",
            "dronet_rvv",
            "topo_0_1_2_3",
            "results.csv",
        )
        dronet_profile_csv_e = os.path.join(
            script_dir,
            "..",
            "data",
            "dronet_scalar",
            "topo_0_1_2_3",
            "results.csv",
        )
        mlp_profile_csv_p = os.path.join(
            script_dir,
            "..",
            "data",
            "mlp_rvv",
            "topo_0_1_2_3",
            "results.csv",
        )
        mlp_profile_csv_e = os.path.join(
            script_dir,
            "..",
            "data",
            "mlp_scalar",
            "topo_0_1_2_3",
            "results.csv",
        )

        dronet_profiled_p = load_profiled_times(dronet_profile_csv_p)
        dronet_profiled_e = load_profiled_times(dronet_profile_csv_e)
        mlp_profiled_p = load_profiled_times(mlp_profile_csv_p)
        mlp_profiled_e = load_profiled_times(mlp_profile_csv_e)
        
        # Combine profiled data for JSON output
        if dronet_profiled_p:
            combined_profiled_p.update(dronet_profiled_p)
        if dronet_profiled_e:
            combined_profiled_e.update(dronet_profiled_e)
        if mlp_profiled_p:
            combined_profiled_p.update(mlp_profiled_p)
        if mlp_profiled_e:
            combined_profiled_e.update(mlp_profiled_e)

        p_core_speedup = 1.5

        def get_profiles_for_dispatch_path(path: str) -> tuple[dict[int, dict] | None, dict[int, dict] | None]:
            fname = os.path.basename(path)
            if "dronet_dispatch_deps" in fname:
                return dronet_profiled_p or None, dronet_profiled_e or None
            if "mlp_dispatch_deps" in fname:
                return mlp_profiled_p or None, mlp_profiled_e or None
            return None, None

        for net_id, net_info in networks.items():
            dispatch_deps_path = net_info.get("dispatch_deps_path", "")
            full_dispatch_path = os.path.join(repo_base_path, dispatch_deps_path)
            if not os.path.exists(full_dispatch_path):
                continue

            prof_p, prof_e = get_profiles_for_dispatch_path(dispatch_deps_path)
            if prof_p is None and prof_e is None:
                continue

            with open(full_dispatch_path, "r") as f:
                dispatch_data = json.load(f)
            dispatches = dispatch_data.get("dispatches", {})

            net_prefix = f"{net_id}_"
            for dispatch_name, dispatch_info in dispatches.items():
                dispatch_id = dispatch_info.get("id", None)
                cpu_p_time: float
                cpu_e_time: float

                p_ms = None
                e_ms = None
                if isinstance(dispatch_id, int) and prof_p and dispatch_id in prof_p:
                    p_ms = prof_p[dispatch_id]["time_ms"]
                if isinstance(dispatch_id, int) and prof_e and dispatch_id in prof_e:
                    e_ms = prof_e[dispatch_id]["time_ms"]

                if p_ms is not None:
                    cpu_p_time = float(p_ms)
                    if e_ms is not None:
                        cpu_e_time = float(e_ms)
                    else:
                        cpu_e_time = float(p_ms * p_core_speedup)
                elif e_ms is not None:
                    cpu_e_time = float(e_ms)
                    cpu_p_time = float(e_ms / p_core_speedup)
                else:
                    p_ms_synth = float(np.random.uniform(2.0, 10.0))
                    cpu_p_time = p_ms_synth
                    cpu_e_time = p_ms_synth * p_core_speedup

                prefixed_name = f"{net_prefix}{dispatch_name}"
                processing_times[prefixed_name] = [cpu_p_time, cpu_e_time]

    # Create workload from network hierarchy
    print(f"\nCreating workload from network hierarchy...")
    combined_workload = create_workload_from_network_hierarchy(
        networks_data=networks_data,
        repo_base_path=repo_base_path,
        machines=machines,
        transfer_times=transfer_times,
        p_core_speedup=1.5,
        random_seed=random_seed,
        processing_times=processing_times,
    )
    
    print(f"\nWorkload created successfully!")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")
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
        solver_verbosity=solver_verbosity,
        time_limit=time_limit,
        restrict_makespan_to_nonperiodic=restrict_makespan_to_nonperiodic,
        prune_cross_period_constraints=prune_periodic,
    )
    t, alpha, _, _ = result  # Always returns 4 values now
    
    # Post-process schedule: optionally trim periodic/background operations that
    # occur entirely after the non-periodic makespan.
    if prune_periodic:
        combined_workload, t, alpha = _trim_periodic_after_nonperiodic_makespan(combined_workload, t, alpha)
    
    # Calculate makespan
    makespan = max(t[i] + combined_workload.operations[i].get_durations()[np.argmax(alpha[i])] 
                   for i in range(len(combined_workload.operations)))
    
    print(f"\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each core
    cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
    cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)
    
    print(f"\nCore assignments:")
    print(f"  CPU_P (performant): {cpu_p_count} operations")
    print(f"  CPU_E (efficient): {cpu_e_count} operations")
    
    # Count operations per network (group by job_id)
    network_stats = {}
    for op_idx, op in enumerate(combined_workload.operations):
        job_id = op.job_id
        if job_id not in network_stats:
            network_stats[job_id] = {'cpu_p': 0, 'cpu_e': 0, 'name': combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}"}
        
        # Find which machine this operation is assigned to
        assigned_machine = np.argmax(alpha[op_idx])
        if assigned_machine == 0:
            network_stats[job_id]['cpu_p'] += 1
        else:
            network_stats[job_id]['cpu_e'] += 1
    
    print(f"\nPer-network core assignments:")
    for job_id in sorted(network_stats.keys()):
        stats = network_stats[job_id]
        print(f"  {stats['name'].capitalize()}: CPU_P={stats['cpu_p']}, CPU_E={stats['cpu_e']}")
    
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
        plot_title=f"{title_networks} Schedule on Dual-Core Device (Periodic)",
        workload=combined_workload
    )
    
    print(f"\nPlot saved to plots/iree_combined_schedule_period.png")
    
    # Output combined JSON file with scheduling information
    os.makedirs("schedules", exist_ok=True)
    # Extract base filename from input JSON path
    input_json_basename = os.path.basename(networks_json_path)
    input_json_name = os.path.splitext(input_json_basename)[0]  # Remove .json extension
    json_output_path = f"schedules/scheduled_{input_json_name}.json"
    if use_profiled:
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
        default=None,
        help="Path to the top-level networks dependencies JSON file (default: src/data/toplevel/networks_periodic.json)"
    )
    parser.add_argument(
        "--solver-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="MOSEK solver verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Maximum optimization time in seconds. MOSEK will return the best solution found within this time limit.",
    )
    parser.add_argument(
        "--profiled",
        action="store_true",
        help="Use profiled runtimes where available (currently supports Dronet and MLP).",
    )
    parser.add_argument(
        "--no-prune-periods",
        action="store_true",
        help="Disable pruning of periodic operations scheduled entirely after the non-periodic makespan.",
    )
    parser.add_argument(
        "--include-periodic-in-makespan",
        action="store_true",
        help="Include periodic operations in the makespan objective instead of only non-periodic tasks.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for synthetic runtime generation (default: 0 for reproducible results). Use -1 for nondeterministic.",
    )
    args = parser.parse_args()
    
    seed = None if args.random_seed is not None and args.random_seed < 0 else args.random_seed
    
    schedule_iree_networks(
        networks_json_path=args.networks_json,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
        use_profiled=args.profiled,
        prune_periodic=not args.no_prune_periods,
        restrict_makespan_to_nonperiodic=not args.include_periodic_in_makespan,
    )

