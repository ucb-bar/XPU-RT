"""
Test script for scheduling IREE dispatch graphs (Fast, Dronet, and MLP) on a dual-core device
using profiled runtimes where available.

Currently profiled data is available for Fast, Dronet, and MLP:
- Fast:
  * `src/data/fastdepth/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P).
  * `src/data/fastdepth/topo_0_1/results.csv` contains measurements for the efficient core (CPU_E).
- Dronet:
  * `src/data/dronet/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P).
  * `src/data/dronet/topo_0_1/results.csv` contains measurements for the efficient core (CPU_E).
- MLP:
  * `src/data/mlp/topo_0_1_2_3/results.csv` contains measurements for the performant core (CPU_P).
  * `src/data/mlp/topo_0_1/results.csv` contains measurements for the efficient core (CPU_E).
- If a dispatch has only P-core or only E-core data, the missing value is derived using the
  scaling factor (CPU_P is 1.5x faster than CPU_E).
"""

import sys
import os
import json
import csv
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_dependencies
from scheduler import schedule
import plot


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
        if profiled_times_p and isinstance(json_dispatch_id, int) and json_dispatch_id in profiled_times_p:
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
        if profiled_times_e and isinstance(json_dispatch_id, int) and json_dispatch_id in profiled_times_e:
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
    combined_workload = Workload(all_operations, machines, transfer_times, job_names=final_job_names)

    return combined_workload


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


def schedule_iree_networks_profiled():
    """
    Schedule Fast, Dronet, and 5 MLP instances on a dual-core device using
    profiled runtimes where available.

    - Fast uses profiled runtimes:
      * CPU_P from `src/data/fastdepth/topo_0_1_2_3/results.csv`
      * CPU_E from `src/data/fastdepth/topo_0_1/results.csv`
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - Dronet uses profiled runtimes:
      * CPU_P from `src/data/dronet/topo_0_1_2_3/results.csv`
      * CPU_E from `src/data/dronet/topo_0_1/results.csv`
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - MLP instances use profiled runtimes:
      * CPU_P from `src/data/mlp/topo_0_1_2_3/results.csv`
      * CPU_E from `src/data/mlp/topo_0_1/results.csv`
      * If only one core's data is available, the other is derived via scaling (t_E = 1.5 * t_P).
    - Dependency chains:
        Chain 1: Fast → Dronet
        Chain 2: MLP0 → MLP1 → MLP2 → MLP3 → MLP4
      The two chains are independent and can run in parallel.
    """
    # Paths to JSON files (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir,
        "..",
        "..",
        "merlin",
        "samples",
        "robotic-NN",
        "pytorch_workload",
        "computation_graph",
    )

    fast_path = os.path.join(base_path, "fast_dispatch_deps.json")
    dronet_path = os.path.join(base_path, "dronet_dispatch_deps.json")
    mlp_path = os.path.join(base_path, "mlp_dispatch_deps.json")

    # Paths to profiled Dronet runtimes
    dronet_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "dronet",
        "topo_0_1_2_3",
        "results.csv",
    )
    dronet_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "dronet",
        "topo_0_1",
        "results.csv",
    )
    
    # Paths to profiled MLP runtimes
    mlp_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "mlp",
        "topo_0_1_2_3",
        "results.csv",
    )
    mlp_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "mlp",
        "topo_0_1",
        "results.csv",
    )
    
    # Paths to profiled Fast (fastdepth) runtimes
    fast_profile_csv_p = os.path.join(
        script_dir,
        "..",
        "data",
        "fastdepth",
        "topo_0_1_2_3",
        "results.csv",
    )
    fast_profile_csv_e = os.path.join(
        script_dir,
        "..",
        "data",
        "fastdepth",
        "topo_0_1",
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
    
    # Load profiled runtimes for MLP (CPU_P and CPU_E)
    print(f"\n2. Loading profiled MLP P-core runtimes from: {mlp_profile_csv_p}")
    mlp_profiled_times_p = load_profiled_times(mlp_profile_csv_p)
    print(f"   Loaded {len(mlp_profiled_times_p)} profiled P-core entries")
    
    print(f"\n3. Loading profiled MLP E-core runtimes from: {mlp_profile_csv_e}")
    mlp_profiled_times_e = load_profiled_times(mlp_profile_csv_e)
    print(f"   Loaded {len(mlp_profiled_times_e)} profiled E-core entries")
    
    # Load profiled runtimes for Fast (CPU_P and CPU_E)
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
    print(f"   Created {fast_job_name} workload with {len(fast_workload.operations)} operations")

    print(f"\n7. Loading dronet dispatch graph from: {dronet_path}")
    dronet_workload, dronet_job_name = create_workload_from_json_with_profile(
        dronet_path,
        name_prefix="dronet_",
        profiled_times_p=dronet_profiled_times_p,
        profiled_times_e=dronet_profiled_times_e,
    )
    print(f"   Created {dronet_job_name} workload with {len(dronet_workload.operations)} operations")

    # Create 5 MLP workloads, each with a unique prefix (using profiled runtimes)
    print(f"\n8. Loading MLP dispatch graph (5 instances, profiled runtimes)...")
    mlp_workloads: list[Workload] = []
    mlp_job_names: list[str] = []
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
        print(f"   Created {mlp_job_name} workload with {len(mlp_workload.operations)} operations")

    # Make Dronet depend on Fast
    print(f"\n9. Adding dependency: {dronet_job_name} depends on {fast_job_name}...")
    add_dependency(fast_workload, dronet_workload)

    # Make each MLP instance depend on the previous one (MLP instances are independent of Fast/Dronet)
    print("\n10. Adding dependencies between MLP instances (MLPs are independent of Fast/Dronet)...")
    for i in range(1, len(mlp_workloads)):
        print(f"   {mlp_job_names[i]} depends on {mlp_job_names[i-1]}...")
        add_dependency(mlp_workloads[i - 1], mlp_workloads[i])

    # Combine workloads
    # Fast and Dronet form one dependency chain: Fast → Dronet
    # MLP instances form their own independent chain: MLP0 → MLP1 → MLP2 → MLP3 → MLP4
    # These two chains can run in parallel (after Fast completes, Dronet and MLP0 can both start)
    print("\n11. Combining workloads...")
    print("   Dependency chains:")
    print(f"     Chain 1: {fast_job_name} → {dronet_job_name}")
    print(f"     Chain 2: {' → '.join(mlp_job_names)}")
    print("   (Chains are independent and can run in parallel)")

    all_workloads = [fast_workload, dronet_workload] + mlp_workloads
    # Use a single name \"MLP\" for all MLP instances in the legend
    all_job_names_for_legend = [fast_job_name, dronet_job_name] + ["MLP"] * len(mlp_workloads)

    # Create job_id mapping: Fast=0, Dronet=1, all MLPs=2 (same color for all MLPs)
    job_id_mapping = [0, 1] + [2] * len(mlp_workloads)

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
    print(f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")

    # Count independent jobs (operations with no predecessors)
    independent_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    print(f"  Independent jobs (can run in parallel): {independent_jobs}")

    # Schedule the combined workload
    print("\n" + "=" * 60)
    print("Scheduling combined workload (with profiled Fast, Dronet, and MLP runtimes)...")
    print("=" * 60)
    t, alpha = schedule(combined_workload)

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

    # Create title showing the dependency chain and that Dronet is profiled
    mlp_chain = " → ".join([name.capitalize() for name in mlp_job_names])
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path="plots/iree_combined_schedule_profiled.png",
        plot_title=(
            f"{fast_job_name.capitalize()} (profiled) → {dronet_job_name.capitalize()} (profiled) + "
            f"{mlp_chain} (profiled) Schedule on Dual-Core Device"
        ),
        workload=combined_workload,
    )

    print("\nPlot saved to plots/iree_combined_schedule_profiled.png")

    return combined_workload, t, alpha


if __name__ == "__main__":
    schedule_iree_networks_profiled()


