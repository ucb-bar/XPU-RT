"""
Test script for scheduling IREE dispatch graphs (Fast, Dronet, and MLP) on a dual-core device
using profiled runtimes where available.

Currently profiled data is available for Fast, Dronet, and MLP:
- Fast:
  * `src/data/fastdepth_rvv/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P, vector/RVV).
  * `src/data/fastdepth_scalar/topo_0_1_2_3/results.csv` contains measurements for the efficient core (CPU_E, scalar).
- Dronet:
  * `src/data/dronet_rvv/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P, vector/RVV).
  * `src/data/dronet_scalar/topo_0_1_2_3/results.csv` contains measurements for the efficient core (CPU_E, scalar).
- MLP:
  * `src/data/mlp_rvv/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P, vector/RVV).
  * `src/data/mlp_scalar/topo_0_1_2_3/results.csv` contains measurements for the efficient core (CPU_E, scalar).
- If a dispatch has only P-core or only E-core data, the missing value is derived using the
  scaling factor (CPU_P is 1.5x faster than CPU_E).
"""

import sys
import os
import json
import csv
import argparse
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_dependencies
from scheduler import schedule
import plot
from validate_schedule import validate_schedule


def load_dispatch_graph(json_path: str) -> dict:
    """Load a dispatch dependencies JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_profiled_times(csv_path: str) -> dict:
    """
    Load profiled runtimes from a CSV file.

    The CSV is expected to have at least:
      - dispatch_id
      - module_name
      - mean_time
      - mean_unit (assumed 'ms')

    Returns:
      dict mapping dispatch_id (int) -> {"time_ms": float, "module_name": str}
    """
    profiled: dict[int, dict] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse dispatch_id
            dispatch_id_str = row.get("dispatch_id")
            if dispatch_id_str is None or dispatch_id_str == "":
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
            # Convert to milliseconds if needed (currently ms already)
            if unit == "us":
                mean_time_ms = mean_time / 1000.0
            elif unit == "s":
                mean_time_ms = mean_time * 1000.0
            else:
                # Assume ms by default
                mean_time_ms = mean_time

            profiled[dispatch_id] = {
                "time_ms": mean_time_ms,
                "module_name": module_name,
            }
    return profiled


def create_workload_from_json_with_profile(
    json_path: str,
    name_prefix: str = "",
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None,
    p_core_speedup: float = 1.5,
) -> tuple[Workload, str]:
    """
    Create a workload from a dispatch dependencies JSON file, optionally using
    profiled runtimes for CPU_P and CPU_E from separate CSV files.

    Parameters:
      - json_path: Path to the dispatch_deps.json file.
      - name_prefix: Optional prefix to add to dispatch names (to avoid conflicts when combining).
      - profiled_times_p: Optional dict mapping dispatch_id -> profiled P-core runtime (ms).
      - profiled_times_e: Optional dict mapping dispatch_id -> profiled E-core runtime (ms).
      - p_core_speedup: CPU_P is `p_core_speedup` times faster than CPU_E (used as fallback if E-core data missing).

    Returns:
      - Tuple (Workload object, job_name) where job_name is derived from filename / prefix.
    """
    # Load the JSON file
    dispatch_data = load_dispatch_graph(json_path)

    # Get dispatches
    original_dispatches = dispatch_data.get("dispatches", {})

    # Update dispatch names in the data structure if prefix is provided
    if name_prefix:
        prefixed_dispatches = {}
        for dispatch_name, dispatch_info in original_dispatches.items():
            prefixed_name = f"{name_prefix}{dispatch_name}"
            prefixed_info = dispatch_info.copy()
            # Update dependencies to use prefixed names
            if "dependencies" in prefixed_info:
                prefixed_info["dependencies"] = [
                    f"{name_prefix}{dep}" if dep in original_dispatches else dep
                    for dep in prefixed_info["dependencies"]
                ]
            prefixed_dispatches[prefixed_name] = prefixed_info
        dispatch_data = {"dispatches": prefixed_dispatches}
        dispatches = prefixed_dispatches
    else:
        dispatches = original_dispatches

    # Generate processing times for dual-core device
    # Map dispatch names to processing times (the function expects names)
    processing_times_by_name: dict[str, list[float]] = {}

    for dispatch_name, dispatch_info in dispatches.items():
        cpu_p_time: float
        cpu_e_time: float

        # Get dispatch ID from JSON
        json_dispatch_id = dispatch_info.get("id", None)

        # Try to get profiled P-core time
        p_ms = None
        if (
            profiled_times_p
            and isinstance(json_dispatch_id, int)
            and json_dispatch_id in profiled_times_p
        ):
            entry_p = profiled_times_p[json_dispatch_id]
            p_ms = entry_p["time_ms"]
            module_name = entry_p.get("module_name", "")
            # Debug: exact ID match between JSON dispatch and CSV entry
            print(
                f"[PROFILE MATCH-ID] json_id={json_dispatch_id}, "
                f"dispatch_name='{dispatch_name}', module_name='{module_name}', "
                f"P-core runtime={p_ms} ms"
            )

        # Try to get profiled E-core time
        e_ms = None
        if (
            profiled_times_e
            and isinstance(json_dispatch_id, int)
            and json_dispatch_id in profiled_times_e
        ):
            entry_e = profiled_times_e[json_dispatch_id]
            e_ms = entry_e["time_ms"]
            if p_ms is not None:
                # Both P and E times found
                print(
                    f"[PROFILE MATCH-ID] json_id={json_dispatch_id}, "
                    f"E-core runtime={e_ms} ms"
                )

        if p_ms is not None:
            # Use profiled P-core time
            cpu_p_time = float(p_ms)
            if e_ms is not None:
                # Use profiled E-core time
                cpu_e_time = float(e_ms)
            else:
                # Derive E-core time from P-core time using speedup factor
                cpu_e_time = float(p_ms * p_core_speedup)
        elif e_ms is not None:
            # Only E-core time available, derive P-core from it
            cpu_e_time = float(e_ms)
            cpu_p_time = float(e_ms / p_core_speedup)
        else:
            # No profile for this dispatch; fall back to synthetic numbers
            # Choose synthetic P-core times in a similar ballpark to profiled data (~5 ms)
            p_ms_synth = float(np.random.uniform(2.0, 10.0))  # 2–10 ms
            cpu_p_time = p_ms_synth
            cpu_e_time = p_ms_synth * p_core_speedup

        # Order: [CPU_P, CPU_E]
        processing_times_by_name[dispatch_name] = [cpu_p_time, cpu_e_time]

    # Define machines (dual-core device)
    machines = ["CPU_P", "CPU_E"]

    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))

    # Create workload from dependencies
    workload = create_workload_from_dependencies(
        dispatch_data=dispatch_data,
        processing_times=processing_times_by_name,
        machines=machines,
        transfer_times=transfer_times,
    )

    # Extract job name from filename (e.g., "dronet_dispatch_deps.json" -> "dronet")
    filename = os.path.basename(json_path)
    job_name = filename.replace("_dispatch_deps.json", "").replace(".json", "")
    if name_prefix:
        # If prefix was added, use it as job name
        job_name = name_prefix.rstrip("_")

    return workload, job_name


def combine_workloads(
    workloads: list[Workload],
    job_names: list[str] | None = None,
    job_id_mapping: list[int] | None = None,
) -> Workload:
    """
    Combine multiple workloads into a single workload.
    Each workload becomes a separate job, preserving job_id assignments.

    Parameters:
      - workloads: List of Workload objects to combine.
      - job_names: Optional list of job names corresponding to workloads.
      - job_id_mapping: Optional list mapping workload index to job_id.
                        If None, each workload gets a unique job_id based on its index.
                        If provided, workloads with the same job_id will share the same color.

    Returns:
      - Combined Workload object with job names.
    """
    if not workloads:
        raise ValueError("At least one workload must be provided")

    # All workloads should have the same machines and transfer times
    machines = workloads[0].machines
    transfer_times = workloads[0].get_transfer_times()

    # Combine all operations
    all_operations: list[Operation] = []
    combined_job_names: list[str | None] = []

    for i, workload in enumerate(workloads):
        # Get job name for this workload
        workload_job_name = None
        if job_names and i < len(job_names):
            workload_job_name = job_names[i]
        elif hasattr(workload, "job_names") and workload.job_names:
            # Use first job name from workload if available
            workload_job_name = workload.job_names[0] if workload.job_names else None

        # Assign job_id to all operations in this workload
        # Use job_id_mapping if provided, otherwise use index as job_id
        if job_id_mapping and i < len(job_id_mapping):
            job_id = job_id_mapping[i]
        else:
            job_id = i  # Use index as job_id

        for op in workload.operations:
            # Set explicit job_id for this operation
            op.job_id = job_id
            all_operations.append(op)

            # Track job names: add name when we encounter first operation of this job
            # If multiple workloads share the same job_id, use the first name encountered
            if op.job_id == job_id:
                # Extend list if needed
                while len(combined_job_names) <= job_id:
                    combined_job_names.append(None)
                # Only set name if not already set (first workload with this job_id wins)
                if combined_job_names[job_id] is None:
                    if workload_job_name:
                        combined_job_names[job_id] = workload_job_name
                    else:
                        combined_job_names[job_id] = f"Job {job_id}"

    # Remove None entries and ensure we have names for all jobs
    final_job_names: list[str] = []
    for j in range(len(combined_job_names)):
        if j < len(combined_job_names) and combined_job_names[j]:
            final_job_names.append(combined_job_names[j])
        else:
            final_job_names.append(f"Job {j}")

    # Create combined workload
    combined_workload = Workload(
        all_operations, machines, transfer_times, job_names=final_job_names
    )

    return combined_workload


def output_scheduled_json(
    all_workloads: list,
    all_job_names: list,
    combined_workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    output_path: str,
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None,
):
    """
    Output a combined JSON file with all dispatches, their hardware targets, and start times.

    Args:
        all_workloads: List of original Workload objects
        all_job_names: List of job names corresponding to workloads
        combined_workload: Combined workload after scheduling
        t: Start times array from scheduling
        alpha: Assignment matrix from scheduling
        output_path: Path to save the output JSON file
        profiled_times_p: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for P-core
        profiled_times_e: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for E-core
    """
    import json

    machine_combinations = combined_workload.get_machine_combinations()

    # Build mapping from operation index in combined workload to dispatch info
    operation_to_dispatch = {}
    current_idx = 0

    for workload_idx, workload in enumerate(all_workloads):
        job_name = (
            all_job_names[workload_idx]
            if workload_idx < len(all_job_names)
            else f"job_{workload_idx}"
        )

        for op in workload.operations:
            operation_to_dispatch[current_idx] = {
                "operation": op,
                "job_name": job_name,
                "workload_idx": workload_idx,
            }
            current_idx += 1

    # First pass: collect all dispatch info with completion times
    dispatch_info_list = []

    for op_idx in range(len(combined_workload.operations)):
        op_info = operation_to_dispatch.get(op_idx)
        if not op_info:
            continue

        op = op_info["operation"]
        job_name = op_info["job_name"]

        # Get dispatch name from operation
        dispatch_name = (
            op.operation_name
            if hasattr(op, "operation_name") and op.operation_name
            else f"op_{op_idx}"
        )

        # Get hardware target (which combination was assigned)
        combo_idx = np.argmax(alpha[op_idx])
        hardware_target = (
            "+".join(machine_combinations[combo_idx])
            if len(machine_combinations[combo_idx]) > 1
            else machine_combinations[combo_idx][0]
        )

        # Get start time
        start_time = float(t[op_idx])

        # Get duration for the assigned combination
        duration = combined_workload.operations[op_idx].get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )

        # Get dispatch ID
        dispatch_id = (
            op.operation_id
            if hasattr(op, "operation_id") and op.operation_id is not None
            else op_idx
        )

        # Get module name from profiled data if available
        module_name = None
        if (
            profiled_times_p
            and isinstance(dispatch_id, int)
            and dispatch_id in profiled_times_p
        ):
            module_name = profiled_times_p[dispatch_id].get("module_name")
        elif (
            profiled_times_e
            and isinstance(dispatch_id, int)
            and dispatch_id in profiled_times_e
        ):
            module_name = profiled_times_e[dispatch_id].get("module_name")

        completion_time = start_time + float(duration)

        dispatch_info_list.append(
            {
                "op_idx": op_idx,
                "dispatch_name": dispatch_name,
                "dispatch_id": dispatch_id,
                "hardware_target": hardware_target,
                "start_time": start_time,
                "duration": float(duration),
                "completion_time": completion_time,
                "job_name": job_name,
                "module_name": module_name,
                "op": op,
                "combined_op": combined_workload.operations[op_idx],
            }
        )

    # Build time dependency mapping: for each hardware target, track dispatches sorted by completion time
    hardware_dispatch_map = (
        {}
    )  # hardware_target -> list of (completion_time, dispatch_name, start_time)

    for info in dispatch_info_list:
        hw_target = info["hardware_target"]
        if hw_target not in hardware_dispatch_map:
            hardware_dispatch_map[hw_target] = []
        hardware_dispatch_map[hw_target].append(
            (info["completion_time"], info["dispatch_name"], info["start_time"])
        )

    # Sort each hardware target's dispatches by completion time
    for hw_target in hardware_dispatch_map:
        hardware_dispatch_map[hw_target].sort(
            key=lambda x: x[0]
        )  # Sort by completion_time

    # Build combined dispatches dictionary
    combined_dispatches = {}

    for info in dispatch_info_list:
        dispatch_name = info["dispatch_name"]
        hardware_target = info["hardware_target"]
        start_time = info["start_time"]

        # Get dependencies (from combined workload operation predecessors)
        dependencies = []
        combined_op = info["combined_op"]
        for pred_op in combined_op.predecessors:
            # Find the index of this predecessor in the combined workload
            pred_idx = None
            for idx, combined_operation in enumerate(combined_workload.operations):
                if combined_operation == pred_op:
                    pred_idx = idx
                    break
            if pred_idx is not None and pred_idx in operation_to_dispatch:
                pred_info = operation_to_dispatch[pred_idx]
                pred_dispatch_name = (
                    pred_info["operation"].operation_name
                    if hasattr(pred_info["operation"], "operation_name")
                    and pred_info["operation"].operation_name
                    else f"op_{pred_idx}"
                )
                dependencies.append(pred_dispatch_name)

        # Find time dependency: previous dispatch on same hardware target
        time_dependency = None
        if hardware_target in hardware_dispatch_map:
            hw_dispatches = hardware_dispatch_map[hardware_target]
            # Find the dispatch that finished most recently before this one starts
            for completion_time, prev_dispatch_name, prev_start_time in hw_dispatches:
                if (
                    completion_time <= start_time
                    and prev_dispatch_name != dispatch_name
                ):
                    time_dependency = prev_dispatch_name
                elif completion_time > start_time:
                    break  # No need to check further (sorted by completion time)

        # Create dispatch entry
        dispatch_entry = {
            "id": info["dispatch_id"],
            "ordinal": 1,  # Keep original structure
            "total": 1,
            "dependencies": dependencies,
            "hardware_target": hardware_target,
            "start_time": start_time,
            "duration": info["duration"],
            "job_name": info["job_name"],
        }

        # Add module_name if available
        if info["module_name"]:
            dispatch_entry["module_name"] = info["module_name"]

        # Add time_dependency if found
        if time_dependency:
            dispatch_entry["time_dependency"] = time_dependency

        combined_dispatches[dispatch_name] = dispatch_entry

    # Create output JSON structure
    output_data = {
        "dot_file": "combined_schedule.json",
        "dispatches": combined_dispatches,
        "metadata": {
            "makespan": float(
                max(
                    t[i]
                    + combined_workload.operations[i].get_duration_for_combination(
                        np.argmax(alpha[i]),
                        machine_combinations,
                        combined_workload.machines,
                    )
                    for i in range(len(combined_workload.operations))
                )
            ),
            "num_operations": len(combined_workload.operations),
            "machines": combined_workload.machines,
            "machine_combinations": [
                combo if isinstance(combo, list) else [combo]
                for combo in machine_combinations
            ],
        },
    }

    # Save to file
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nScheduled JSON saved to: {output_path}")


def add_dependency(source_workload: Workload, target_workload: Workload) -> None:
    """
    Make target workload's first operations depend on source workload's last operations.
    Source workload's last operations are those that are not predecessors of any other operation.
    """
    # Find last operations in source (operations that are not predecessors of any other operation in source)
    source_last_ops: list[Operation] = []

    for op in source_workload.operations:
        # Check if this operation is a predecessor of any other operation in source
        is_predecessor = False
        for other_op in source_workload.operations:
            if op in other_op.predecessors:
                is_predecessor = True
                break
        if not is_predecessor:
            source_last_ops.append(op)

    # If no explicit last operations found, use all operations as potential last ops (fallback)
    if not source_last_ops:
        source_last_ops = source_workload.operations

    # Find first operations in target (operations with no predecessors)
    target_first_ops = [op for op in target_workload.operations if not op.predecessors]

    # If no first operations found, use the first operation
    if not target_first_ops and target_workload.operations:
        target_first_ops = [target_workload.operations[0]]

    # Add dependencies: each target first operation depends on all source last operations
    for target_op in target_first_ops:
        for source_op in source_last_ops:
            target_op.add_predecessor(source_op)


def schedule_iree_networks_profiled(
    no_depth_network: bool = False,
    use_mobilenet: bool = False,
    use_diffusion: bool = False,
    fusion_threshold: float = None,
    verbose: bool = False,
    solver_verbosity: int = 0,
    time_limit: float = None,
):
    """
    Schedule Fast, Dronet, and either 5 MLP instances, 1 MobilenetV2, or 1 Diffusion model on a dual-core device using
    profiled runtimes where available.

    - Fast uses profiled runtimes:
      * CPU_P from `src/data/fastdepth_rvv/topo_0_1_2_3/results.csv` (vector/RVV)
      * CPU_E from `src/data/fastdepth_scalar/topo_0_1_2_3/results.csv` (scalar)
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - Dronet uses profiled runtimes:
      * CPU_P from `src/data/dronet_rvv/topo_0_1_2_3/results.csv` (vector/RVV)
      * CPU_E from `src/data/dronet_scalar/topo_0_1_2_3/results.csv` (scalar)
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - MLP instances use profiled runtimes:
      * CPU_P from `src/data/mlp_rvv/topo_0_1_2_3/results.csv` (vector/RVV)
      * CPU_E from `src/data/mlp_scalar/topo_0_1_2_3/results.csv` (scalar)
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - MobilenetV2 uses profiled runtimes:
      * CPU_P from `src/data/mobilenet_v2_rvv/topo_0_1_2_3/results.csv` (vector/RVV)
      * CPU_E from `src/data/mobilenet_v2_scalar/topo_0_1_2_3/results.csv` (scalar)
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - Diffusion uses profiled runtimes:
      * CPU_P from `src/data/diffusion_rvv/topo_0_1_2_3/results.csv` (vector/RVV)
      * CPU_E from `src/data/diffusion_scalar/topo_0_1_2_3/results.csv` (scalar)
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - Dependency chains:
        Chain 1: Fast → Dronet (if Fast is included)
                 OR MobilenetV2 → Dronet (if use_mobilenet is True)
                 OR Diffusion → Dronet (if use_diffusion is True)
                 OR MobilenetV2 + Diffusion → Dronet (if both are enabled)
        Chain 2: MLP0 → MLP1 → MLP2 → MLP3 → MLP4 (if use_mobilenet and use_diffusion are False)
      The two chains are independent and can run in parallel.
      If no_depth_network is True, Fast is skipped.
      Special case: If both use_mobilenet and use_diffusion are True, both feed into Dronet.

    Args:
        no_depth_network: If True, skip loading and scheduling Fast (depth network).
        use_mobilenet: If True, use a single MobilenetV2 instead of 5 MLP instances.
        use_diffusion: If True, use a single Diffusion model instead of 5 MLP instances.
    """
    # Paths to JSON files (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir,
        "..",
        "pytorch_workload",
        "samples",
    )

    fast_path = os.path.join(base_path, "fast_dispatch_deps.json")
    dronet_path = os.path.join(base_path, "dronet_dispatch_deps.json")
    mlp_path = os.path.join(base_path, "mlp_dispatch_deps.json")
    mobilenet_path = os.path.join(base_path, "mobilenet_v2_dispatch_deps.json")
    diffusion_path = os.path.join(base_path, "diffusion_dispatch_deps.json")

    # Paths to profiled Dronet runtimes
    # P-core: vector/RVV data
    dronet_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "dronet_rvv",
        "topo_0_1_2_3",
        "results.csv",
    )
    # E-core: scalar data
    dronet_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "dronet_scalar",
        "topo_0_1_2_3",
        "results.csv",
    )

    # Paths to profiled MLP runtimes
    # P-core: vector/RVV data
    mlp_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "mlp_rvv",
        "topo_0_1_2_3",
        "results.csv",
    )
    # E-core: scalar data
    mlp_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "mlp_scalar",
        "topo_0_1_2_3",
        "results.csv",
    )

    # Paths to profiled Fast (fastdepth) runtimes
    # P-core: vector/RVV data
    fast_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "fastdepth_rvv",
        "topo_0_1_2_3",
        "results.csv",
    )
    # E-core: scalar data
    fast_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "fastdepth_scalar",
        "topo_0_1_2_3",
        "results.csv",
    )

    # Paths to profiled MobilenetV2 runtimes
    # P-core: vector/RVV data
    mobilenet_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "mobilenet_v2_rvv",
        "topo_0_1_2_3",
        "results.csv",
    )
    # E-core: use same vector data (scalar data may not be available)
    mobilenet_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "mobilenet_v2_scalar",
        "topo_0_1_2_3",
        "results.csv",
    )

    print("=" * 60)
    print("Loading dispatch graphs (with profiled runtimes)...")
    print("=" * 60)

    # Load profiled runtimes for Dronet (CPU_P and CPU_E)
    print(f"\n0. Loading profiled Dronet P-core runtimes from: {dronet_profile_csv_p}")
    dronet_profiled_times_p = load_profiled_times(dronet_profile_csv_p)
    print(f"   Loaded {len(dronet_profiled_times_p)} profiled P-core entries")

    print(f"\n1. Loading profiled Dronet E-core runtimes from: {dronet_profile_csv_e}")
    dronet_profiled_times_e = load_profiled_times(dronet_profile_csv_e)
    print(f"   Loaded {len(dronet_profiled_times_e)} profiled E-core entries")

    # Initialize MLP, MobilenetV2, and Diffusion profiled times variables
    mlp_profiled_times_p = None
    mlp_profiled_times_e = None
    mobilenet_profiled_times_p = None
    mobilenet_profiled_times_e = None
    diffusion_profiled_times_p = None
    diffusion_profiled_times_e = None

    # Load profiled runtimes for MLP, MobilenetV2, and/or Diffusion (CPU_P and CPU_E)
    if not use_mobilenet and not use_diffusion:
        print(f"\n2. Loading profiled MLP P-core runtimes from: {mlp_profile_csv_p}")
        mlp_profiled_times_p = load_profiled_times(mlp_profile_csv_p)
        print(f"   Loaded {len(mlp_profiled_times_p)} profiled P-core entries")

        print(f"\n3. Loading profiled MLP E-core runtimes from: {mlp_profile_csv_e}")
        mlp_profiled_times_e = load_profiled_times(mlp_profile_csv_e)
        print(f"   Loaded {len(mlp_profiled_times_e)} profiled E-core entries")
    else:
        print(
            f"\n2. Loading profiled MobilenetV2 P-core runtimes from: {mobilenet_profile_csv_p}"
        )
        mobilenet_profiled_times_p = load_profiled_times(mobilenet_profile_csv_p)
        print(f"   Loaded {len(mobilenet_profiled_times_p)} profiled P-core entries")

        print(
            f"\n3. Loading profiled MobilenetV2 E-core runtimes from: {mobilenet_profile_csv_e}"
        )
        mobilenet_profiled_times_e = load_profiled_times(mobilenet_profile_csv_e)
        print(f"   Loaded {len(mobilenet_profiled_times_e)} profiled E-core entries")

    # Load profiled runtimes for Diffusion (CPU_P and CPU_E) if enabled
    if use_diffusion:
        # Paths to profiled Diffusion runtimes
        # P-core: vector/RVV data
        diffusion_profile_csv_p = os.path.join(
            script_dir,
            "..",
            "data",
            "diffusion_rvv",
            "topo_0_1_2_3",
            "results.csv",
        )
        # E-core: scalar data
        diffusion_profile_csv_e = os.path.join(
            script_dir,
            "..",
            "data",
            "diffusion_scalar",
            "topo_0_1_2_3",
            "results.csv",
        )

        step_num = 2 if use_mobilenet else 2
        print(
            f"\n{step_num + 1}. Loading profiled Diffusion P-core runtimes from: {diffusion_profile_csv_p}"
        )
        diffusion_profiled_times_p = load_profiled_times(diffusion_profile_csv_p)
        print(f"   Loaded {len(diffusion_profiled_times_p)} profiled P-core entries")

        print(
            f"\n{step_num + 2}. Loading profiled Diffusion E-core runtimes from: {diffusion_profile_csv_e}"
        )
        diffusion_profiled_times_e = load_profiled_times(diffusion_profile_csv_e)
        print(f"   Loaded {len(diffusion_profiled_times_e)} profiled E-core entries")

    # Load profiled runtimes for Fast (CPU_P and CPU_E) - only if not skipping depth network
    fast_profiled_times_p = None
    fast_profiled_times_e = None
    fast_workload = None
    fast_job_name = None

    if not no_depth_network:
        print(f"\n4. Loading profiled Fast P-core runtimes from: {fast_profile_csv_p}")
        fast_profiled_times_p = load_profiled_times(fast_profile_csv_p)
        print(f"   Loaded {len(fast_profiled_times_p)} profiled P-core entries")

        print(f"\n5. Loading profiled Fast E-core runtimes from: {fast_profile_csv_e}")
        fast_profiled_times_e = load_profiled_times(fast_profile_csv_e)
        print(f"   Loaded {len(fast_profiled_times_e)} profiled E-core entries")

        # Create workloads from JSON files
        print(f"\n6. Loading fast dispatch graph from: {fast_path}")
        fast_workload, fast_job_name = create_workload_from_json_with_profile(
            fast_path,
            name_prefix="fast_",
            profiled_times_p=fast_profiled_times_p,
            profiled_times_e=fast_profiled_times_e,
        )
        print(
            f"   Created {fast_job_name} workload with {len(fast_workload.operations)} operations"
        )
    else:
        print("\n4-6. Skipping Fast (depth network) as requested")

    if not no_depth_network:
        print(f"\n7. Loading dronet dispatch graph from: {dronet_path}")
    else:
        print(f"\n4. Loading dronet dispatch graph from: {dronet_path}")
    dronet_workload, dronet_job_name = create_workload_from_json_with_profile(
        dronet_path,
        name_prefix="dronet_",
        profiled_times_p=dronet_profiled_times_p,
        profiled_times_e=dronet_profiled_times_e,
    )
    print(
        f"   Created {dronet_job_name} workload with {len(dronet_workload.operations)} operations"
    )

    # Create MLP workloads, MobilenetV2 workload, and/or Diffusion workload
    mlp_workloads: list[Workload] = []
    mlp_job_names: list[str] = []
    mobilenet_workload = None
    mobilenet_job_name = None
    diffusion_workload = None
    diffusion_job_name = None

    if not use_mobilenet and not use_diffusion:
        # Create 5 MLP workloads, each with a unique prefix (using profiled runtimes)
        if not no_depth_network:
            print(
                f"\n8. Loading MLP dispatch graph (5 instances, profiled runtimes)..."
            )
        else:
            print(
                f"\n5. Loading MLP dispatch graph (5 instances, profiled runtimes)..."
            )
        for i in range(5):
            mlp_prefix = f"mlp{i}_"
            mlp_workload, mlp_job_name = create_workload_from_json_with_profile(
                mlp_path,
                name_prefix=mlp_prefix,
                profiled_times_p=mlp_profiled_times_p,
                profiled_times_e=mlp_profiled_times_e,
            )
            mlp_job_name = f"mlp{i}"  # Use numbered name
            mlp_workloads.append(mlp_workload)
            mlp_job_names.append(mlp_job_name)
            print(
                f"   Created {mlp_job_name} workload with {len(mlp_workload.operations)} operations"
            )
    else:
        # Create single MobilenetV2 workload
        if not no_depth_network:
            print(f"\n8. Loading MobilenetV2 dispatch graph (profiled runtimes)...")
        else:
            print(f"\n5. Loading MobilenetV2 dispatch graph (profiled runtimes)...")
        mobilenet_workload, mobilenet_job_name = create_workload_from_json_with_profile(
            mobilenet_path,
            name_prefix="mobilenet_",
            profiled_times_p=mobilenet_profiled_times_p,
            profiled_times_e=mobilenet_profiled_times_e,
        )
        mobilenet_job_name = "mobilenet_v2"
        print(
            f"   Created {mobilenet_job_name} workload with {len(mobilenet_workload.operations)} operations"
        )

    # Create Diffusion workload if enabled
    if use_diffusion:
        if not no_depth_network:
            step_num = 8 if not use_mobilenet else 9
        else:
            step_num = 5 if not use_mobilenet else 6
        print(f"\n{step_num}. Loading Diffusion dispatch graph (profiled runtimes)...")
        diffusion_workload, diffusion_job_name = create_workload_from_json_with_profile(
            diffusion_path,
            name_prefix="diffusion_",
            profiled_times_p=diffusion_profiled_times_p,
            profiled_times_e=diffusion_profiled_times_e,
        )
        diffusion_job_name = "diffusion"
        print(
            f"   Created {diffusion_job_name} workload with {len(diffusion_workload.operations)} operations"
        )

    # Make Dronet depend on Fast, MobilenetV2, and/or Diffusion
    # Special case: if both mobilenet and diffusion are enabled, dronet depends on both
    step_num = 9 if not no_depth_network else 6
    dependencies_added = []

    # Add dependency on Fast if included
    if not no_depth_network:
        print(
            f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {fast_job_name}..."
        )
        add_dependency(fast_workload, dronet_workload)
        dependencies_added.append(fast_job_name)
        step_num += 1

    # Add dependency on MobilenetV2 if enabled
    if use_mobilenet:
        print(
            f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {mobilenet_job_name}..."
        )
        add_dependency(mobilenet_workload, dronet_workload)
        dependencies_added.append(mobilenet_job_name)
        step_num += 1

    # Add dependency on Diffusion if enabled
    if use_diffusion:
        print(
            f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {diffusion_job_name}..."
        )
        add_dependency(diffusion_workload, dronet_workload)
        dependencies_added.append(diffusion_job_name)
        step_num += 1

    if not dependencies_added:
        print("\n6. Dronet is independent (no dependencies)")

    # Make each MLP instance depend on the previous one (MLP instances are independent of Fast/Dronet)
    # Note: MobilenetV2 and Diffusion dependencies are handled above (they feed into dronet)
    if not use_mobilenet and not use_diffusion:
        step_num = 10 if not no_depth_network else 7
        print(
            f"\n{step_num}. Adding dependencies between MLP instances (MLPs are independent of Fast/Dronet)..."
        )
        for i in range(1, len(mlp_workloads)):
            print(f"   {mlp_job_names[i]} depends on {mlp_job_names[i-1]}...")
            add_dependency(mlp_workloads[i - 1], mlp_workloads[i])
    else:
        # MobilenetV2 and Diffusion are already set up as dependencies for dronet above
        # They don't have dependencies between themselves
        pass

    # Combine workloads
    # Dependency structure:
    # - Fast (if included) → Dronet
    # - MobilenetV2 (if enabled) → Dronet
    # - Diffusion (if enabled) → Dronet
    # - MLP instances form their own chain: MLP0 → MLP1 → MLP2 → MLP3 → MLP4 (if not using mobilenet/diffusion)
    if not no_depth_network:
        print(f"\n11. Combining workloads...")
    else:
        print(f"\n8. Combining workloads...")
    print("   Dependency chains:")
    # Build dependency chain for dronet
    dronet_deps = []
    if not no_depth_network:
        dronet_deps.append(fast_job_name)
    if use_mobilenet:
        dronet_deps.append(mobilenet_job_name)
    if use_diffusion:
        dronet_deps.append(diffusion_job_name)

    if dronet_deps:
        print(f"     Chain 1: {' + '.join(dronet_deps)} → {dronet_job_name}")
    else:
        print(f"     Chain 1: {dronet_job_name} (independent)")

    if not use_mobilenet and not use_diffusion:
        print(f"     Chain 2: {' → '.join(mlp_job_names)}")
    print("   (Chains are independent and can run in parallel)")

    if not use_mobilenet and not use_diffusion:
        if not no_depth_network:
            all_workloads = [fast_workload, dronet_workload] + mlp_workloads
            # Use a single name "MLP" for all MLP instances in the legend
            all_job_names_for_legend = [fast_job_name, dronet_job_name] + ["MLP"] * len(
                mlp_workloads
            )
            # Create job_id mapping: Fast=0, Dronet=1, all MLPs=2 (same color for all MLPs)
            job_id_mapping = [0, 1] + [2] * len(mlp_workloads)
        else:
            all_workloads = [dronet_workload] + mlp_workloads
            # Use a single name "MLP" for all MLP instances in the legend
            all_job_names_for_legend = [dronet_job_name] + ["MLP"] * len(mlp_workloads)
            # Create job_id mapping: Dronet=0, all MLPs=1 (same color for all MLPs)
            job_id_mapping = [0] + [1] * len(mlp_workloads)
    else:
        # Build workload list: first network (if included), then mobilenet/diffusion (if enabled), then dronet
        all_workloads = []
        all_job_names_for_legend = []
        job_id_mapping = []
        job_id = 0

        if not no_depth_network:
            all_workloads.append(fast_workload)
            all_job_names_for_legend.append(fast_job_name)
            job_id_mapping.append(job_id)
            job_id += 1

        if use_mobilenet:
            all_workloads.append(mobilenet_workload)
            all_job_names_for_legend.append(mobilenet_job_name)
            job_id_mapping.append(job_id)
            job_id += 1

        if use_diffusion:
            all_workloads.append(diffusion_workload)
            all_job_names_for_legend.append(diffusion_job_name)
            job_id_mapping.append(job_id)
            job_id += 1

        # Dronet comes last (depends on all above)
        all_workloads.append(dronet_workload)
        all_job_names_for_legend.append(dronet_job_name)
        job_id_mapping.append(job_id)

    combined_workload = combine_workloads(
        all_workloads,
        job_names=all_job_names_for_legend,
        job_id_mapping=job_id_mapping,
    )

    print("\nCombined workload statistics:")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")

    # Print some statistics
    operations_with_multiple_predecessors = [
        op for op in combined_workload.operations if len(op.predecessors) > 1
    ]
    print(
        f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}"
    )

    # Count independent jobs (operations with no predecessors)
    independent_jobs = sum(
        1 for op in combined_workload.operations if not op.predecessors
    )
    print(f"  Independent jobs (can run in parallel): {independent_jobs}")

    # Schedule the combined workload
    print("\n" + "=" * 60)
    if not no_depth_network:
        if not use_mobilenet and not use_diffusion:
            print(
                "Scheduling combined workload (with profiled Fast, Dronet, and MLP runtimes)..."
            )
        elif use_mobilenet and use_diffusion:
            print(
                "Scheduling combined workload (with profiled Fast, Dronet, MobilenetV2, and Diffusion runtimes)..."
            )
        elif use_mobilenet:
            print(
                "Scheduling combined workload (with profiled Fast, Dronet, and MobilenetV2 runtimes)..."
            )
        else:  # use_diffusion
            print(
                "Scheduling combined workload (with profiled Fast, Dronet, and Diffusion runtimes)..."
            )
    else:
        if not use_mobilenet and not use_diffusion:
            print(
                "Scheduling combined workload (with profiled Dronet and MLP runtimes, no depth network)..."
            )
        elif use_mobilenet and use_diffusion:
            print(
                "Scheduling combined workload (with profiled Dronet, MobilenetV2, and Diffusion runtimes, no depth network)..."
            )
        elif use_mobilenet:
            print(
                "Scheduling combined workload (with profiled Dronet and MobilenetV2 runtimes, no depth network)..."
            )
        else:  # use_diffusion
            print(
                "Scheduling combined workload (with profiled Dronet and Diffusion runtimes, no depth network)..."
            )
    print("=" * 60)
    result = schedule(
        combined_workload,
        fusion_threshold=fusion_threshold,
        verbose=verbose,
        solver_verbosity=solver_verbosity,
        time_limit=time_limit,
    )
    t, alpha, _, _ = result  # Always returns 4 values now

    # Check if scheduling was successful
    if t is None or alpha is None:
        print(
            "\nScheduling failed (infeasible or error). Cannot proceed with analysis."
        )
        return combined_workload, None, None

    # Calculate makespan
    makespan = max(
        t[i] + combined_workload.operations[i].get_durations()[np.argmax(alpha[i])]
        for i in range(len(combined_workload.operations))
    )

    print("\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")

    # Count operations assigned to each core
    cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
    cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)

    print("\nCore assignments:")
    print(f"  CPU_P (performant): {cpu_p_count} operations")
    print(f"  CPU_E (efficient): {cpu_e_count} operations")

    # Create plot
    os.makedirs("plots", exist_ok=True)

    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)

    # Create title showing the dependency chain and that networks are profiled
    if not use_mobilenet and not use_diffusion:
        mlp_chain = " → ".join([name.capitalize() for name in mlp_job_names])
        if not no_depth_network:
            plot_title = (
                f"{fast_job_name.capitalize()} (profiled) → {dronet_job_name.capitalize()} (profiled) + "
                f"{mlp_chain} (profiled) Schedule on Dual-Core Device"
            )
        else:
            plot_title = (
                f"{dronet_job_name.capitalize()} (profiled) + "
                f"{mlp_chain} (profiled) Schedule on Dual-Core Device"
            )
    else:
        # Build chain showing mobilenet and/or diffusion feeding into dronet
        chains = []
        if use_mobilenet:
            chains.append(mobilenet_job_name.capitalize())
        if use_diffusion:
            chains.append(diffusion_job_name.capitalize())
        if chains:
            chain_str = " + ".join(chains) + " → " + dronet_job_name.capitalize()
        else:
            chain_str = dronet_job_name.capitalize()

        if not no_depth_network:
            plot_title = f"{fast_job_name.capitalize()} (profiled) → {chain_str} (profiled) Schedule on Dual-Core Device"
        else:
            plot_title = f"{chain_str} (profiled) Schedule on Dual-Core Device"

    # Add makespan to plot title
    plot_title = f"{plot_title} (Makespan: {makespan:.2f} ms)"

    # Determine plot filename
    plot_filename = "iree_combined_schedule_profiled.png"
    if fusion_threshold is not None and fusion_threshold > 0:
        plot_filename = "iree_combined_schedule_profiled_fusion.png"

    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path=f"plots/{plot_filename}",
        plot_title=plot_title,
        workload=combined_workload,
    )

    print(f"\nPlot saved to plots/{plot_filename}")

    # Output combined JSON file with scheduling information
    os.makedirs("schedules", exist_ok=True)
    json_output_path = "schedules/combined_schedule_profiled.json"
    if fusion_threshold is not None and fusion_threshold > 0:
        json_output_path = "schedules/combined_schedule_profiled_fusion.json"
    if no_depth_network:
        json_output_path = json_output_path.replace(".json", "_no_depth.json")
    if use_mobilenet and use_diffusion:
        json_output_path = json_output_path.replace(
            ".json", "_mobilenet_diffusion.json"
        )
    elif use_mobilenet:
        json_output_path = json_output_path.replace(".json", "_mobilenet.json")
    elif use_diffusion:
        json_output_path = json_output_path.replace(".json", "_diffusion.json")

    # Combine profiled times for JSON output
    combined_profiled_p = {}
    combined_profiled_e = {}
    if not no_depth_network and fast_profiled_times_p:
        combined_profiled_p.update(fast_profiled_times_p)
        combined_profiled_e.update(fast_profiled_times_e)
    combined_profiled_p.update(dronet_profiled_times_p)
    combined_profiled_e.update(dronet_profiled_times_e)
    if not use_mobilenet and not use_diffusion:
        if mlp_profiled_times_p:
            combined_profiled_p.update(mlp_profiled_times_p)
        if mlp_profiled_times_e:
            combined_profiled_e.update(mlp_profiled_times_e)
    else:
        if use_mobilenet and mobilenet_profiled_times_p:
            combined_profiled_p.update(mobilenet_profiled_times_p)
        if use_mobilenet and mobilenet_profiled_times_e:
            combined_profiled_e.update(mobilenet_profiled_times_e)
        if use_diffusion and diffusion_profiled_times_p:
            combined_profiled_p.update(diffusion_profiled_times_p)
        if use_diffusion and diffusion_profiled_times_e:
            combined_profiled_e.update(diffusion_profiled_times_e)

    print(f"\nOutputting scheduled JSON...")
    output_scheduled_json(
        all_workloads=all_workloads,
        all_job_names=all_job_names_for_legend,
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=combined_profiled_p if combined_profiled_p else None,
        profiled_times_e=combined_profiled_e if combined_profiled_e else None,
    )

    # Validate schedule against original JSON and profiled data
    print("\n" + "=" * 60)
    print("Validating schedule against original JSON and profiled data...")
    print("=" * 60)

    # Combine all JSON dispatch data
    combined_json_data = {"dispatches": {}}

    # Load and combine JSON files
    json_files = []
    if not no_depth_network:
        json_files.append(("fast_", fast_path))
    json_files.append(("dronet_", dronet_path))
    if not use_mobilenet and not use_diffusion:
        for i in range(5):
            json_files.append((f"mlp{i}_", mlp_path))
    else:
        if use_mobilenet:
            json_files.append(("mobilenet_", mobilenet_path))
        if use_diffusion:
            json_files.append(("diffusion_", diffusion_path))

    for prefix, json_path in json_files:
        dispatch_data = load_dispatch_graph(json_path)
        original_dispatches = dispatch_data.get("dispatches", {})
        for dispatch_name, dispatch_info in original_dispatches.items():
            prefixed_name = f"{prefix}{dispatch_name}"
            prefixed_info = dispatch_info.copy()
            # Update dependencies to use prefixed names
            if "dependencies" in prefixed_info:
                prefixed_info["dependencies"] = [
                    f"{prefix}{dep}" if dep in original_dispatches else dep
                    for dep in prefixed_info["dependencies"]
                ]
            combined_json_data["dispatches"][prefixed_name] = prefixed_info

    # Combine all profiled times
    combined_profiled_p = {}
    combined_profiled_e = {}

    if not no_depth_network:
        combined_profiled_p.update(fast_profiled_times_p)
        combined_profiled_e.update(fast_profiled_times_e)

    combined_profiled_p.update(dronet_profiled_times_p)
    combined_profiled_e.update(dronet_profiled_times_e)

    if not use_mobilenet and not use_diffusion:
        combined_profiled_p.update(mlp_profiled_times_p)
        combined_profiled_e.update(mlp_profiled_times_e)
    else:
        if use_mobilenet:
            combined_profiled_p.update(mobilenet_profiled_times_p)
            combined_profiled_e.update(mobilenet_profiled_times_e)
        if use_diffusion:
            combined_profiled_p.update(diffusion_profiled_times_p)
            combined_profiled_e.update(diffusion_profiled_times_e)

    # Run validation
    os.makedirs("validation_reports", exist_ok=True)
    validation_file = "validation_reports/validation_report.txt"

    # Build network-specific profiled times to avoid dispatch_id collisions
    profiled_times_by_network = {}
    if not no_depth_network:
        profiled_times_by_network["fast"] = {
            "p": fast_profiled_times_p,
            "e": fast_profiled_times_e,
        }
    profiled_times_by_network["dronet"] = {
        "p": dronet_profiled_times_p,
        "e": dronet_profiled_times_e,
    }
    if not use_mobilenet and not use_diffusion:
        profiled_times_by_network["mlp"] = {
            "p": mlp_profiled_times_p,
            "e": mlp_profiled_times_e,
        }
    else:
        if use_mobilenet:
            profiled_times_by_network["mobilenet"] = {
                "p": mobilenet_profiled_times_p,
                "e": mobilenet_profiled_times_e,
            }
        if use_diffusion:
            profiled_times_by_network["diffusion"] = {
                "p": diffusion_profiled_times_p,
                "e": diffusion_profiled_times_e,
            }

    is_valid, validation_results = validate_schedule(
        combined_workload,
        t,
        alpha,
        combined_json_data,
        profiled_times_p=combined_profiled_p if combined_profiled_p else None,
        profiled_times_e=combined_profiled_e if combined_profiled_e else None,
        profiled_times_by_network=profiled_times_by_network,
        output_file=validation_file,
    )

    if is_valid:
        print("✓ Validation PASSED: All checks passed")
    else:
        print("✗ Validation FAILED: See validation report for details")
        print(f"  Errors: {len(validation_results['errors'])}")
        print(f"  Warnings: {len(validation_results['warnings'])}")

    print(f"\nValidation report saved to: {validation_file}")

    return combined_workload, t, alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE dispatch graphs (Fast, Dronet, and MLP/MobilenetV2/Diffusion) using profiled runtimes"
    )
    parser.add_argument(
        "--no-depth-network",
        action="store_true",
        help="Skip loading and scheduling the depth network (Fast/glpdepth). Only schedule Dronet and MLP/MobilenetV2/Diffusion.",
    )
    parser.add_argument(
        "--use-mobilenet",
        action="store_true",
        help="Use a single MobilenetV2 instead of 5 MLP instances.",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="Use a single Diffusion model instead of 5 MLP instances.",
    )
    parser.add_argument(
        "--fusion-threshold",
        type=float,
        default=None,
        help="Fusion threshold in time units. Operations with duration <= threshold will be fused with neighbors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing problem statistics and timing.",
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
    args = parser.parse_args()
    schedule_iree_networks_profiled(
        no_depth_network=args.no_depth_network,
        use_mobilenet=args.use_mobilenet,
        use_diffusion=args.diffusion,
        fusion_threshold=args.fusion_threshold,
        verbose=args.verbose,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
    )
