"""
Test script for scheduling IREE dispatch graphs with machine combinations.
Supports scheduling to CPU_P, CPU_E, or both concurrently (CPU_P + CPU_E).
Uses synthetic data and assumes ideal parallelism for the combined option.
"""

import sys
import os
import json
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


class OperationWithCombinations(Operation):
    """
    Extended Operation class that supports explicit durations for machine combinations.
    For the combined CPU_P+CPU_E option, we store a separate duration that assumes ideal parallelism.
    """
    
    def __init__(self, processing_times: list[float], predecessors=None, 
                 operation_id=None, operation_name=None, job_id=None,
                 combined_duration: float = None):
        """
        @param processing_times: list of durations for each machine [cpu_p_time, cpu_e_time]
        @param combined_duration: duration when running on both CPU_P and CPU_E concurrently (ideal parallelism)
        """
        super().__init__(processing_times, predecessors, operation_id, operation_name, job_id)
        self.combined_duration = combined_duration
    
    def get_duration_for_combination(self, combination_idx: int, machine_combinations: list[list[str]], machines: list[str]) -> float:
        """
        Get the duration for a specific machine combination.
        Overrides the base class to handle the combined CPU_P+CPU_E option specially.
        """
        if combination_idx < 0 or combination_idx >= len(machine_combinations):
            raise ValueError(f"Invalid combination index: {combination_idx}")
        
        combo = machine_combinations[combination_idx]
        
        # If singleton combination, return duration for that machine
        if len(combo) == 1:
            machine_idx = machines.index(combo[0])
            if machine_idx < len(self.processing_times):
                return self.processing_times[machine_idx]
            else:
                raise ValueError(f"Machine {combo[0]} not found in processing_times")
        
        # For multi-machine combinations, check if it's the combined CPU_P+CPU_E option
        if len(combo) == 2 and set(combo) == {'CPU_P', 'CPU_E'}:
            # Use the explicitly stored combined duration if available
            if self.combined_duration is not None:
                return self.combined_duration
            else:
                # Fallback: use ideal parallelism assumption (min of the two, scaled down)
                cpu_p_idx = machines.index('CPU_P')
                cpu_e_idx = machines.index('CPU_E')
                cpu_p_time = self.processing_times[cpu_p_idx]
                cpu_e_time = self.processing_times[cpu_e_idx]
                # Ideal parallelism: assume we can achieve ~60% of the faster time
                # (perfect load balancing would give ~50%, but accounting for overhead)
                return min(cpu_p_time, cpu_e_time) * 0.7
        
        # For other multi-machine combinations, use max (default behavior)
        durations = []
        for machine_name in combo:
            machine_idx = machines.index(machine_name)
            if machine_idx < len(self.processing_times):
                durations.append(self.processing_times[machine_idx])
        if durations:
            return max(durations)
        else:
            raise ValueError(f"Could not find durations for combination {combo}")


def create_workload_from_json_with_combinations(json_path: str, name_prefix: str = "") -> tuple:
    """
    Create a workload from a dispatch dependencies JSON file with support for machine combinations.
    
    Parameters:
    - json_path: Path to the dispatch_deps.json file
    - name_prefix: Optional prefix to add to dispatch names (to avoid conflicts when combining)
    
    Returns:
    - Tuple of (Workload object, job_name) where job_name is derived from filename
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
    
    # Define machines and machine combinations
    machines = ["CPU_P", "CPU_E"]
    machine_combinations = [
        ["CPU_P"],           # Option 1: CPU_P alone
        ["CPU_E"],           # Option 2: CPU_E alone
        ["CPU_P", "CPU_E"],  # Option 3: Both concurrently
    ]
    
    # Generate processing times for each dispatch
    # Map dispatch names to Operation objects with combination support
    operations_map = {}
    job_id = 0
    
    for dispatch_name, dispatch_info in dispatches.items():
        # Generate random P-core time in milliseconds (2–10 ms)
        p_ms_synth = float(np.random.uniform(2.0, 10.0))
        cpu_p_time = p_ms_synth
        cpu_e_time = p_ms_synth * 1.5  # CPU_P is 1.5x faster than CPU_E
        
        # For ideal parallelism on both cores, assume we can achieve ~60% of the faster time
        # (perfect load balancing would give ~50%, but accounting for overhead)
        # NOTE: To test with slower combined time, change 0.6 to something > 1.0 (e.g., 1.5)
        combined_time = min(cpu_p_time, cpu_e_time) * 0.6  # Faster (ideal parallelism)
        # combined_time = max(cpu_p_time, cpu_e_time) * 1.2  # Slower than both (to test constraints)
        
        # Extract ID and name from dispatch info
        operation_id = dispatch_info.get("id", None)
        operation_name = dispatch_name
        
        # Create operation with combination support
        operation = OperationWithCombinations(
            processing_times=[cpu_p_time, cpu_e_time],
            operation_id=operation_id,
            operation_name=operation_name,
            job_id=job_id,
            combined_duration=combined_time,
        )
        
        operations_map[dispatch_name] = operation
    
    # Set up predecessor relationships
    for dispatch_name, dispatch_info in dispatches.items():
        dependencies = dispatch_info.get("dependencies", [])
        operation = operations_map[dispatch_name]
        
        for dep_name in dependencies:
            if dep_name in operations_map:
                operation.add_predecessor(operations_map[dep_name])
    
    # Create list of operations
    operations = list(operations_map.values())
    
    # Determine job names
    job_names = []
    seen_prefixes = set()
    
    for operation in operations:
        if not operation.predecessors:
            op_name = operation.operation_name or f"dispatch_{len(job_names)}"
            if "_" in op_name:
                parts = op_name.split("_")
                if len(parts) >= 2:
                    prefix = parts[0]
                    job_name = prefix.capitalize()
                    if job_name not in seen_prefixes:
                        job_names.append(job_name)
                        seen_prefixes.add(job_name)
                else:
                    if op_name not in seen_prefixes:
                        job_names.append(op_name.capitalize())
                        seen_prefixes.add(op_name)
            else:
                if op_name not in seen_prefixes:
                    job_names.append(op_name.capitalize())
                    seen_prefixes.add(op_name)
    
    if not job_names:
        num_jobs = sum(1 for op in operations if not op.predecessors)
        job_names = [f"Job {i}" for i in range(num_jobs)]
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))
    
    # Create workload with machine combinations
    workload = Workload(
        operations=operations,
        machines=machines,
        transfer_times=transfer_times,
        job_names=job_names,
        machine_combinations=machine_combinations,
    )
    
    # Extract job name from filename
    filename = os.path.basename(json_path)
    job_name = filename.replace("_dispatch_deps.json", "").replace(".json", "")
    if name_prefix:
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
    """
    if not workloads:
        raise ValueError("At least one workload must be provided")
    
    # All workloads should have the same machines, transfer times, and machine combinations
    machines = workloads[0].machines
    transfer_times = workloads[0].get_transfer_times()
    machine_combinations = workloads[0].get_machine_combinations()
    
    # Verify all workloads have the same machine combinations
    for workload in workloads[1:]:
        if workload.get_machine_combinations() != machine_combinations:
            raise ValueError("All workloads must have the same machine combinations")
    
    # Combine all operations
    all_operations: list[Operation] = []
    combined_job_names: list[str | None] = []
    
    for i, workload in enumerate(workloads):
        workload_job_name = None
        if job_names and i < len(job_names):
            workload_job_name = job_names[i]
        elif hasattr(workload, "job_names") and workload.job_names:
            workload_job_name = workload.job_names[0] if workload.job_names else None
        
        if job_id_mapping and i < len(job_id_mapping):
            job_id = job_id_mapping[i]
        else:
            job_id = i
        
        for op in workload.operations:
            op.job_id = job_id
            all_operations.append(op)
            
            if op.job_id == job_id:
                while len(combined_job_names) <= job_id:
                    combined_job_names.append(None)
                if combined_job_names[job_id] is None:
                    if workload_job_name:
                        combined_job_names[job_id] = workload_job_name
                    else:
                        combined_job_names[job_id] = f"Job {job_id}"
    
    final_job_names: list[str] = []
    for j in range(len(combined_job_names)):
        if j < len(combined_job_names) and combined_job_names[j]:
            final_job_names.append(combined_job_names[j])
        else:
            final_job_names.append(f"Job {j}")
    
    combined_workload = Workload(
        all_operations, machines, transfer_times, job_names=final_job_names, machine_combinations=machine_combinations
    )
    
    return combined_workload


def add_dependency(source_workload: Workload, target_workload: Workload) -> None:
    """Make target workload's first operations depend on source workload's last operations."""
    source_last_ops: list[Operation] = []
    
    for op in source_workload.operations:
        is_predecessor = False
        for other_op in source_workload.operations:
            if op in other_op.predecessors:
                is_predecessor = True
                break
        if not is_predecessor:
            source_last_ops.append(op)
    
    if not source_last_ops:
        source_last_ops = source_workload.operations
    
    target_first_ops = [op for op in target_workload.operations if not op.predecessors]
    
    if not target_first_ops and target_workload.operations:
        target_first_ops = [target_workload.operations[0]]
    
    for target_op in target_first_ops:
        for source_op in source_last_ops:
            target_op.add_predecessor(source_op)


def schedule_iree_networks_grouped():
    """
    Schedule Dronet on a dual-core device with machine combinations.
    Operations can be scheduled to CPU_P, CPU_E, or both concurrently (with ideal parallelism).
    """
    # Paths to JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir,
        "..",
        "pytorch_workload",
        "samples",
    )
    
    dronet_path = os.path.join(base_path, "dronet_dispatch_deps.json")
    
    print("=" * 60)
    print("Loading Dronet dispatch graph (with machine combinations support)...")
    print("=" * 60)
    
    # Create workload from JSON file
    print(f"\n1. Loading dronet dispatch graph from: {dronet_path}")
    dronet_workload, dronet_job_name = create_workload_from_json_with_combinations(
        dronet_path, name_prefix="dronet_"
    )
    print(f"   Created {dronet_job_name} workload with {len(dronet_workload.operations)} operations")
    
    # Use the workload directly (no need to combine)
    combined_workload = dronet_workload
    
    print("\nWorkload statistics:")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")
    print(f"  Machine combinations: {combined_workload.get_machine_combinations()}")
    
    # Schedule the workload
    print("\n" + "=" * 60)
    print("Scheduling workload (with machine combinations)...")
    print("=" * 60)
    result = schedule(combined_workload)
    t, alpha, _, _ = result  # Always returns 4 values now
    
    # Calculate makespan
    machine_combinations = combined_workload.get_machine_combinations()
    makespan = 0
    for i in range(len(combined_workload.operations)):
        combo_idx = np.argmax(alpha[i])
        duration = combined_workload.operations[i].get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )
        makespan = max(makespan, t[i] + duration)
    
    print("\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each combination
    combo_counts = [0] * len(machine_combinations)
    for i in range(len(alpha)):
        combo_idx = np.argmax(alpha[i])
        combo_counts[combo_idx] += 1
    
    print("\nMachine combination assignments:")
    for idx, combo in enumerate(machine_combinations):
        combo_str = "+".join(combo) if len(combo) > 1 else combo[0]
        print(f"  {combo_str}: {combo_counts[idx]} operations")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create labels for machine combinations for the plot
    combination_labels = []
    for combo in machine_combinations:
        if len(combo) == 1:
            combination_labels.append(combo[0])
        else:
            combination_labels.append("+".join(combo))
    
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(machine_combinations),  # Use num_combinations instead of num_machines
        combination_labels,  # Pass combination labels instead of machine names
        combined_workload.get_transfer_times(),
        save_path="plots/iree_dronet_schedule_grouped.png",
        plot_title=(
            f"{dronet_job_name.capitalize()} Schedule with Machine Combinations"
        ),
        workload=combined_workload,
    )
    
    print("\nPlot saved to plots/iree_dronet_schedule_grouped.png")
    
    return combined_workload, t, alpha


if __name__ == "__main__":
    schedule_iree_networks_grouped()

