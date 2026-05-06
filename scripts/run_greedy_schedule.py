"""
Test script for scheduling IREE dispatch graphs (dronet and MLP) on a dual-core device using a greedy scheduler.
Parses dispatch dependency JSON files and schedules them in parallel on CPU_P (performant) 
and CPU_E (efficient) cores. CPU_P is 1.5x faster than CPU_E.
Uses a greedy scheduling algorithm instead of MILP optimization.
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
from workload_factory import (
    create_workload_from_dependencies,
    create_workload_from_network_hierarchy,
    build_machine_combinations,
    machine_type_prefix,
)
from profile_loader import load_profiled_processing_times
import plot
from schedule_validation import overlap_fixer, count_overlaps, validate_schedule

# Reuse the toplevel-JSON loader and CPU constants from the MILP runner so
# `--networks-json` accepts exactly the same spec.  The greedy path mirrors
# the workload setup (machine combinations, profile loading, periodic
# expansion) end-to-end and only swaps `schedule()` for `greedy_schedule()`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_xpurt_schedule import load_networks_config, CPU_P, CPU_E  # noqa: E402

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
                return min(cpu_p_time, cpu_e_time) * 0.6
        
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

def greedy_schedule(workload: Workload) -> tuple:
    """
    Greedy scheduling algorithm that assigns each operation to the machine combination that gives the earliest completion time.
    Supports both traditional machine assignments and machine combinations (grouped backends).
    
    Parameters:
    - workload: Workload object containing operations, machines, and transfer times
    
    Returns:
    - t: numpy array of start times for each operation
    - alpha: numpy array of shape (num_operations, num_combinations) where alpha[i, j] = 1 if operation i is assigned to combination j
    """
    num_operations = len(workload.operations)
    machines = workload.machines
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()
    
    # Initialize arrays
    t = np.zeros(num_operations)  # Start times
    alpha = np.zeros((num_operations, num_combinations))  # Combination assignments (one-hot)
    
    # Track when each combination becomes available
    # For combinations that overlap, we need to track when the shared machines are available
    # We'll track the latest time any machine in the combination becomes available
    combination_available_time = np.zeros(num_combinations)
    
    # Track which operations have been scheduled
    scheduled = [False] * num_operations
    
    # Schedule operations iteratively
    while not all(scheduled):
        # Find an operation that can be scheduled (all predecessors are scheduled)
        best_op_idx = None
        best_completion_time = float('inf')
        best_combination_idx = None
        best_start_time = 0
        
        for i in range(num_operations):
            if scheduled[i]:
                continue
            
            # Check if all predecessors are scheduled
            op = workload.operations[i]
            can_schedule = True
            
            for pred in op.predecessors:
                pred_idx = workload.operations.index(pred)
                if not scheduled[pred_idx]:
                    can_schedule = False
                    break
            
            if not can_schedule:
                continue
            
            # For this operation, find the combination that gives the earliest completion time
            for combo_idx in range(num_combinations):
                # Check if this combination conflicts with any already scheduled operation
                # (i.e., if any scheduled operation uses an overlapping combination)
                can_use_combo = True
                earliest_start = combination_available_time[combo_idx]
                
                # Check for conflicts with already scheduled operations
                for j in range(num_operations):
                    if not scheduled[j]:
                        continue
                    
                    # Find which combination the scheduled operation uses
                    scheduled_combo_idx = np.argmax(alpha[j, :])
                    
                    # If combinations overlap, we need to ensure they don't run at the same time
                    if workload.combinations_overlap(combo_idx, scheduled_combo_idx):
                        # The scheduled operation's end time
                        scheduled_op_duration = workload.operations[j].get_duration_for_combination(
                            scheduled_combo_idx, machine_combinations, machines
                        )
                        scheduled_op_end_time = t[j] + scheduled_op_duration
                        # This combination can't start until the conflicting operation finishes
                        earliest_start = max(earliest_start, scheduled_op_end_time)
                
                # Consider all predecessors and their transfer times
                for pred in op.predecessors:
                    pred_idx = workload.operations.index(pred)
                    pred_combo_idx = np.argmax(alpha[pred_idx, :])
                    
                    # Get predecessor's duration for its combination
                    pred_duration = workload.operations[pred_idx].get_duration_for_combination(
                        pred_combo_idx, machine_combinations, machines
                    )
                    pred_end_time = t[pred_idx] + pred_duration
                    
                    # Calculate transfer time from predecessor's combination to this candidate combination
                    # For transfer time, we use the first machine from each combination
                    # (This is a simplification; in practice, transfer time might depend on the specific machines)
                    pred_combo = machine_combinations[pred_combo_idx]
                    candidate_combo = machine_combinations[combo_idx]
                    
                    # Use the first machine from each combination for transfer time calculation
                    pred_machine_idx = machines.index(pred_combo[0])
                    candidate_machine_idx = machines.index(candidate_combo[0])
                    transfer_time = transfer_times[pred_machine_idx, candidate_machine_idx]
                    
                    pred_ready_time = pred_end_time + transfer_time
                    earliest_start = max(earliest_start, pred_ready_time)

                # Honor periodic / windowed time-bounds carried by the
                # Operation (set by create_workload_from_network_hierarchy
                # when expanding periodic networks: instance i gets
                # min_start_t = start + i*period, max_end_t = min_start +
                # window_duration).  Without this, all periodic instances
                # collapse to t=0 since they have no predecessors and the
                # only nonzero floor was the prior op on the machine.
                if op.min_start_t is not None:
                    earliest_start = max(earliest_start, float(op.min_start_t))

                # Get duration for this combination
                duration = workload.operations[i].get_duration_for_combination(
                    combo_idx, machine_combinations, machines
                )
                completion_time = earliest_start + duration

                # If this combo can't finish within the window
                # (max_end_t), skip it — the scheduler must pick a faster
                # combo or another op.  When ALL combos miss the window,
                # we fall back to the best (latest-ending) one and let
                # validation flag it; that's the best a non-backtracking
                # greedy can do.
                if op.max_end_t is not None and completion_time > float(op.max_end_t):
                    # Track as a fallback only if nothing else fits.
                    if completion_time < best_completion_time:
                        best_completion_time = completion_time
                        best_op_idx = i
                        best_combination_idx = combo_idx
                        best_start_time = earliest_start
                    continue

                if completion_time < best_completion_time:
                    best_completion_time = completion_time
                    best_op_idx = i
                    best_combination_idx = combo_idx
                    best_start_time = earliest_start
        
        if best_op_idx is None:
            # This shouldn't happen if the dependency graph is acyclic
            # But if it does, schedule the first unscheduled operation on the first combination
            for i in range(num_operations):
                if not scheduled[i]:
                    best_op_idx = i
                    best_combination_idx = 0
                    best_start_time = combination_available_time[0]
                    break
        
        # Schedule the operation
        t[best_op_idx] = best_start_time
        alpha[best_op_idx, best_combination_idx] = 1.0
        scheduled[best_op_idx] = True
        
        # Update combination availability
        # For combinations that overlap, we need to update the availability of all overlapping combinations
        duration = workload.operations[best_op_idx].get_duration_for_combination(
            best_combination_idx, machine_combinations, machines
        )
        operation_end_time = best_start_time + duration
        
        # Update the availability of this combination
        combination_available_time[best_combination_idx] = operation_end_time
        
        # Also update availability of all overlapping combinations
        # (They can't start until this operation finishes)
        for combo_idx in range(num_combinations):
            if combo_idx != best_combination_idx and workload.combinations_overlap(best_combination_idx, combo_idx):
                combination_available_time[combo_idx] = max(
                    combination_available_time[combo_idx],
                    operation_end_time
                )
    
    return t, alpha

def output_scheduled_json(
    all_workloads: list,
    all_job_names: list,
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
        job_name = all_job_names[workload_idx] if workload_idx < len(all_job_names) else f"job_{workload_idx}"
        
        for op in workload.operations:
            operation_to_dispatch[current_idx] = {
                'operation': op,
                'job_name': job_name,
                'workload_idx': workload_idx
            }
            current_idx += 1
    
    # First pass: collect all dispatch info with completion times
    dispatch_info_list = []
    
    for op_idx in range(len(combined_workload.operations)):
        op_info = operation_to_dispatch.get(op_idx)
        if not op_info:
            continue
        
        op = op_info['operation']
        job_name = op_info['job_name']
        
        # Get dispatch name from operation
        dispatch_name = op.operation_name if hasattr(op, 'operation_name') and op.operation_name else f"op_{op_idx}"
        
        # Get hardware target (which combination was assigned)
        combo_idx = np.argmax(alpha[op_idx])
        hardware_target = "+".join(machine_combinations[combo_idx]) if len(machine_combinations[combo_idx]) > 1 else machine_combinations[combo_idx][0]
        
        # Get start time
        start_time = float(t[op_idx])
        
        # Get duration for the assigned combination
        duration = combined_workload.operations[op_idx].get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )
        
        # Get dispatch ID
        dispatch_id = op.operation_id if hasattr(op, 'operation_id') and op.operation_id is not None else op_idx
        
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
            'combined_op': combined_workload.operations[op_idx]
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
        
        # Get dependencies (from combined workload operation predecessors)
        dependencies = []
        combined_op = info['combined_op']
        for pred_op in combined_op.predecessors:
            # Find the index of this predecessor in the combined workload
            pred_idx = None
            for idx, combined_operation in enumerate(combined_workload.operations):
                if combined_operation == pred_op:
                    pred_idx = idx
                    break
            if pred_idx is not None and pred_idx in operation_to_dispatch:
                pred_info = operation_to_dispatch[pred_idx]
                pred_dispatch_name = pred_info['operation'].operation_name if hasattr(pred_info['operation'], 'operation_name') and pred_info['operation'].operation_name else f"op_{pred_idx}"
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
        "dot_file": "combined_schedule.json",
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

def load_dispatch_graph(json_path: str) -> dict:
    """Load a dispatch dependencies JSON file."""
    with open(json_path, 'r') as f:
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

def create_workload_from_json(
    json_path: str,
    name_prefix: str = "",
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None,
    p_core_speedup: float = 1.5,
) -> tuple:
    """
    Create a workload from a dispatch dependencies JSON file.
    
    Parameters:
    - json_path: Path to the dispatch_deps.json file
    - name_prefix: Optional prefix to add to dispatch names (to avoid conflicts when combining)
    
    Returns:
    - Tuple of (Workload object, job_name) where job_name is derived from filename
    """
    # Load the JSON file
    dispatch_data = load_dispatch_graph(json_path)
    
    # Get dispatches
    original_dispatches = dispatch_data.get('dispatches', {})
    
    # Update dispatch names in the data structure if prefix is provided
    if name_prefix:
        prefixed_dispatches = {}
        for dispatch_name, dispatch_info in original_dispatches.items():
            prefixed_name = f"{name_prefix}{dispatch_name}"
            prefixed_info = dispatch_info.copy()
            # Update dependencies to use prefixed names
            if 'dependencies' in prefixed_info:
                prefixed_info['dependencies'] = [
                    f"{name_prefix}{dep}" if dep in original_dispatches else dep
                    for dep in prefixed_info['dependencies']
                ]
            prefixed_dispatches[prefixed_name] = prefixed_info
        dispatch_data = {'dispatches': prefixed_dispatches}
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
        
        processing_times_by_name[dispatch_name] = [cpu_p_time, cpu_e_time]
    
    # Define machines (dual-core device)
    machines = ['CPU_P', 'CPU_E']
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))
    
    # Create workload from dependencies
    workload = create_workload_from_dependencies(
        dispatch_data=dispatch_data,
        processing_times=processing_times_by_name,
        machines=machines,
        transfer_times=transfer_times
    )
    
    # Extract job name from filename (e.g., "dronet_dispatch_deps.json" -> "dronet")
    filename = os.path.basename(json_path)
    job_name = filename.replace('_dispatch_deps.json', '').replace('.json', '')
    if name_prefix:
        # If prefix was added, use it as job name
        job_name = name_prefix.rstrip('_')
    
    return workload, job_name

def create_workload_from_json_with_combinations(
    json_path: str,
    name_prefix: str = "",
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None,
    p_core_speedup: float = 1.5,
) -> tuple:
    """
    Create a workload from a dispatch dependencies JSON file with support for machine combinations.
    
    Parameters:
    - json_path: Path to the dispatch_deps.json file
    - name_prefix: Optional prefix to add to dispatch names (to avoid conflicts when combining)
    - profiled_times_p: Optional dict mapping dispatch_id -> profiled P-core runtime (ms)
    - profiled_times_e: Optional dict mapping dispatch_id -> profiled E-core runtime (ms)
    - p_core_speedup: CPU_P is `p_core_speedup` times faster than CPU_E (used as fallback if E-core data missing)
    
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
        cpu_p_time: float
        cpu_e_time: float
        combined_time: float
        
        # Get dispatch ID from JSON
        json_dispatch_id = dispatch_info.get("id", None)
        
        # Try to get profiled P-core time
        p_ms = None
        if profiled_times_p and isinstance(json_dispatch_id, int) and json_dispatch_id in profiled_times_p:
            entry_p = profiled_times_p[json_dispatch_id]
            p_ms = entry_p["time_ms"]
            module_name = entry_p.get("module_name", "")
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
            p_ms_synth = float(np.random.uniform(2.0, 10.0))  # 2–10 ms
            cpu_p_time = p_ms_synth
            cpu_e_time = p_ms_synth * p_core_speedup
        
        # For ideal parallelism on both cores, assume we can achieve ~60% of the faster time
        combined_time = min(cpu_p_time, cpu_e_time) * 0.6
        
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
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))
    
    # Create workload with machine combinations
    workload = Workload(operations, machines, transfer_times, machine_combinations=machine_combinations)
    
    # Extract job name from filename
    filename = os.path.basename(json_path)
    job_name = filename.replace('_dispatch_deps.json', '').replace('.json', '')
    if name_prefix:
        job_name = name_prefix.rstrip('_')
    
    return workload, job_name

def combine_workloads(workloads: list, job_names: list = None, job_id_mapping: list = None) -> Workload:
    """
    Combine multiple workloads into a single workload.
    Each workload becomes a separate job, preserving job_id assignments.
    
    Parameters:
    - workloads: List of Workload objects to combine
    - job_names: Optional list of job names corresponding to workloads
    - job_id_mapping: Optional list mapping workload index to job_id. 
                      If None, each workload gets a unique job_id based on its index.
                      If provided, workloads with the same job_id will share the same color.
    
    Returns:
    - Combined Workload object with job names
    """
    if not workloads:
        raise ValueError("At least one workload must be provided")
    
    # All workloads should have the same machines and transfer times
    machines = workloads[0].machines
    transfer_times = workloads[0].get_transfer_times()
    
    # Get machine combinations from first workload (all should have the same)
    machine_combinations = workloads[0].get_machine_combinations()
    
    # Combine all operations
    all_operations = []
    combined_job_names = []
    
    for i, workload in enumerate(workloads):
        # Get job name for this workload
        workload_job_name = None
        if job_names and i < len(job_names):
            workload_job_name = job_names[i]
        elif hasattr(workload, 'job_names') and workload.job_names:
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
    final_job_names = []
    for j in range(len(combined_job_names)):
        if j < len(combined_job_names) and combined_job_names[j]:
            final_job_names.append(combined_job_names[j])
        else:
            final_job_names.append(f"Job {j}")
    
    # Create combined workload with machine combinations if available
    if machine_combinations is not None:
        combined_workload = Workload(all_operations, machines, transfer_times, job_names=final_job_names, machine_combinations=machine_combinations)
    else:
        combined_workload = Workload(all_operations, machines, transfer_times, job_names=final_job_names)
    
    return combined_workload

def add_dependency(source_workload: Workload, target_workload: Workload):
    """
    Make target workload's first operations depend on source workload's last operations.
    Source workload's last operations are those that are not predecessors of any other operation.
    
    Parameters:
    - source_workload: Workload that should complete first
    - target_workload: Workload that should start after source completes
    """
    # Find last operations in source (operations that are not predecessors of any other operation in source)
    source_last_ops = []
    source_ops_set = set(source_workload.operations)
    
    for op in source_workload.operations:
        # Check if this operation is a predecessor of any other operation in source
        is_predecessor = False
        for other_op in source_workload.operations:
            if op in other_op.predecessors:
                is_predecessor = True
                break
        if not is_predecessor:
            source_last_ops.append(op)
    
    # If no explicit last operations found, use operations with no successors
    # (operations that aren't predecessors of anything)
    if not source_last_ops:
        # Use all operations as potential last ops (fallback)
        source_last_ops = source_workload.operations
    
    # Find first operations in target (operations with no predecessors)
    target_first_ops = [op for op in target_workload.operations if not op.predecessors]
    
    # If no first operations found, use the first operation
    if not target_first_ops:
        target_first_ops = [target_workload.operations[0]] if target_workload.operations else []
    
    # Add dependencies: each target first operation depends on all source last operations
    for target_op in target_first_ops:
        for source_op in source_last_ops:
            target_op.add_predecessor(source_op)

def schedule_iree_networks(use_glpdepth=False, use_profiled=False, use_grouped=False, no_depth_network=False, use_mobilenet=False, use_diffusion=False, mlp_debug=False):
    """
    Main function to schedule fast/glpdepth, dronet (depends on fast/glpdepth), and 5 MLP instances, MobilenetV2, or Diffusion on dual-core device.
    Each MLP instance depends on the previous one (MLP0 → MLP1 → MLP2 → MLP3 → MLP4).
    MobilenetV2 and Diffusion are independent (no dependencies).
    Uses a greedy scheduling algorithm instead of MILP optimization.
    
    Parameters:
    - use_glpdepth: If True, use glpdepth instead of fast as the first network
    - use_profiled: If True, use profiled runtimes from CSV files (currently only supported for glpdepth when use_glpdepth is True)
    - use_grouped: If True, use machine combinations (CPU_P, CPU_E, CPU_P+CPU_E) as scheduling options
    - no_depth_network: If True, skip loading and scheduling the depth network (fast/glpdepth). Only schedule Dronet and MLP/MobilenetV2/Diffusion.
    - use_mobilenet: If True, use a single MobilenetV2 instead of 5 MLP instances.
    - use_diffusion: If True, use a single Diffusion model instead of 5 MLP instances.
    - mlp_debug: If True, only schedule a single MLP instance instead of 5 (for debugging).
    """
    # Path to JSON files (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir, 
        '..', 
        'pytorch_workload', 
        'samples'
    )
    
    # Select first network based on argument
    if use_glpdepth:
        first_network_name = "glpdepth"
        first_network_file = "glpdepth_dispatch_deps.json"
        first_network_prefix = "glpdepth_"
    else:
        first_network_name = "fast"
        first_network_file = "fast_dispatch_deps.json"
        first_network_prefix = "fast_"
    
    first_network_path = os.path.join(base_path, first_network_file)
    dronet_path = os.path.join(base_path, 'dronet_dispatch_deps.json')
    mlp_path = os.path.join(base_path, 'mlp_dispatch_deps.json')
    mobilenet_path = os.path.join(base_path, 'mobilenet_v2_dispatch_deps.json')
    diffusion_path = os.path.join(base_path, 'diffusion_dispatch_deps.json')
    
    # Initialize profiled data variables
    first_network_profiled_times_p = None
    first_network_profiled_times_e = None
    mobilenet_profiled_times_p = None
    mobilenet_profiled_times_e = None
    diffusion_profiled_times_p = None
    diffusion_profiled_times_e = None
    
    # Load profiled data if requested
    if use_profiled:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load first network profiled data (only if not skipping depth network)
        if not no_depth_network:
            if use_glpdepth:
                # Load glpdepth profiled data
                glpdepth_profile_csv_p = os.path.join(
                    script_dir,
                    "..",
                    "data",
                    "glpdepth",
                    "topo_0_1_2_3",
                    "results.csv",
                )
                glpdepth_profile_csv_e = os.path.join(
                    script_dir,
                    "..",
                    "data",
                    "glpdepth",
                    "topo_0_1",
                    "results.csv",
                )
                print("=" * 60)
                print("Loading profiled runtimes...")
                print("=" * 60)
                print(f"\nLoading profiled glpdepth P-core runtimes from: {glpdepth_profile_csv_p}")
                first_network_profiled_times_p = load_profiled_times(glpdepth_profile_csv_p)
                print(f"   Loaded {len(first_network_profiled_times_p)} profiled P-core entries")
                
                print(f"\nLoading profiled glpdepth E-core runtimes from: {glpdepth_profile_csv_e}")
                first_network_profiled_times_e = load_profiled_times(glpdepth_profile_csv_e)
                print(f"   Loaded {len(first_network_profiled_times_e)} profiled E-core entries")
        
        # Load MobilenetV2 profiled data if using mobilenet
        if use_mobilenet:
            mobilenet_profile_csv_p = os.path.join(
                script_dir,
                "..",
                "data",
                "mobilenet_v2_rvv",
                "topo_0_1_2_3",
                "results.csv",
            )
            mobilenet_profile_csv_e = os.path.join(
                script_dir,
                "..",
                "data",
                "mobilenet_v2_scalar",
                "topo_0_1_2_3",
                "results.csv",
            )
            if no_depth_network:
                # Only print header if we skipped depth network (otherwise it was already printed)
                print("=" * 60)
                print("Loading profiled runtimes...")
                print("=" * 60)
            print(f"\nLoading profiled MobilenetV2 P-core runtimes from: {mobilenet_profile_csv_p}")
            mobilenet_profiled_times_p = load_profiled_times(mobilenet_profile_csv_p)
            print(f"   Loaded {len(mobilenet_profiled_times_p)} profiled P-core entries")
            
            print(f"\nLoading profiled MobilenetV2 E-core runtimes from: {mobilenet_profile_csv_e}")
            mobilenet_profiled_times_e = load_profiled_times(mobilenet_profile_csv_e)
            print(f"   Loaded {len(mobilenet_profiled_times_e)} profiled E-core entries")
        
        # Load Diffusion profiled data if using diffusion
        if use_diffusion:
            diffusion_profile_csv_p = os.path.join(
                script_dir,
                "..",
                "data",
                "diffusion_rvv",
                "topo_0_1_2_3",
                "results.csv",
            )
            diffusion_profile_csv_e = os.path.join(
                script_dir,
                "..",
                "data",
                "diffusion_scalar",
                "topo_0_1_2_3",
                "results.csv",
            )
            if no_depth_network:
                # Only print header if we skipped depth network (otherwise it was already printed)
                print("=" * 60)
                print("Loading profiled runtimes...")
                print("=" * 60)
            print(f"\nLoading profiled Diffusion P-core runtimes from: {diffusion_profile_csv_p}")
            diffusion_profiled_times_p = load_profiled_times(diffusion_profile_csv_p)
            print(f"   Loaded {len(diffusion_profiled_times_p)} profiled P-core entries")
            
            print(f"\nLoading profiled Diffusion E-core runtimes from: {diffusion_profile_csv_e}")
            diffusion_profiled_times_e = load_profiled_times(diffusion_profile_csv_e)
            print(f"   Loaded {len(diffusion_profiled_times_e)} profiled E-core entries")
        
        # Load Dronet profiled data (always load if use_profiled is True)
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
        if no_depth_network and not use_mobilenet and not use_diffusion:
            # Only print header if we skipped depth network and didn't print it for mobilenet/diffusion
            print("=" * 60)
            print("Loading profiled runtimes...")
            print("=" * 60)
        print(f"\nLoading profiled Dronet P-core runtimes from: {dronet_profile_csv_p}")
        dronet_profiled_times_p = load_profiled_times(dronet_profile_csv_p)
        print(f"   Loaded {len(dronet_profiled_times_p)} profiled P-core entries")
        
        print(f"\nLoading profiled Dronet E-core runtimes from: {dronet_profile_csv_e}")
        dronet_profiled_times_e = load_profiled_times(dronet_profile_csv_e)
        print(f"   Loaded {len(dronet_profiled_times_e)} profiled E-core entries")
    else:
        # Initialize dronet profiled times as None if not using profiled
        dronet_profiled_times_p = None
        dronet_profiled_times_e = None
    
    print("=" * 60)
    print("Loading dispatch graphs...")
    print("=" * 60)
    
    # Choose workload creation function based on whether we're using combinations
    if use_grouped:
        create_workload_func = create_workload_from_json_with_combinations
        print("Using machine combinations: CPU_P, CPU_E, CPU_P+CPU_E")
    else:
        create_workload_func = create_workload_from_json
    
    # Create workloads from JSON files
    first_network_workload = None
    first_network_job_name = None
    
    # If mlp_debug is True, skip depth network and dronet
    if mlp_debug:
        no_depth_network = True
        print("\n1. MLP debug mode: Skipping depth network (fast/glpdepth) and dronet")
        dronet_workload = None
        dronet_job_name = None
    else:
        if not no_depth_network:
            print(f"\n1. Loading {first_network_name} dispatch graph from: {first_network_path}")
            first_network_workload, first_network_job_name = create_workload_func(
                first_network_path,
                name_prefix=first_network_prefix,
                profiled_times_p=first_network_profiled_times_p,
                profiled_times_e=first_network_profiled_times_e,
            )
            print(f"   Created {first_network_job_name} workload with {len(first_network_workload.operations)} operations")
        else:
            print("\n1. Skipping depth network (fast/glpdepth) as requested")
        
        step_num = 2 if not no_depth_network else 1
        print(f"\n{step_num}. Loading dronet dispatch graph from: {dronet_path}")
        dronet_workload, dronet_job_name = create_workload_func(
            dronet_path,
            name_prefix="dronet_",
            profiled_times_p=dronet_profiled_times_p if use_profiled else None,
            profiled_times_e=dronet_profiled_times_e if use_profiled else None,
        )
        print(f"   Created {dronet_job_name} workload with {len(dronet_workload.operations)} operations")
    
    # Create MLP workloads, MobilenetV2 workload, and/or Diffusion workload
    mlp_workloads = []
    mlp_job_names = []
    mobilenet_workload = None
    mobilenet_job_name = None
    diffusion_workload = None
    diffusion_job_name = None
    
    step_num = 3 if not no_depth_network else 2
    
    if not use_mobilenet and not use_diffusion:
        # Create MLP workloads (5 instances by default, 1 if mlp_debug is True)
        num_mlps = 1 if mlp_debug else 5
        print(f"\n{step_num}. Loading MLP dispatch graph ({num_mlps} instance{'s' if num_mlps > 1 else ''})...")
        for i in range(num_mlps):
            mlp_prefix = f"mlp{i}_"
            mlp_workload, mlp_job_name = create_workload_func(mlp_path, name_prefix=mlp_prefix)
            mlp_job_name = f"mlp{i}"  # Use numbered name
            mlp_workloads.append(mlp_workload)
            mlp_job_names.append(mlp_job_name)
            print(f"   Created {mlp_job_name} workload with {len(mlp_workload.operations)} operations")
    else:
        # Create MobilenetV2 workload if requested
        if use_mobilenet:
            print(f"\n{step_num}. Loading MobilenetV2 dispatch graph...")
            mobilenet_workload, mobilenet_job_name = create_workload_func(
                mobilenet_path,
                name_prefix="mobilenet_",
                profiled_times_p=mobilenet_profiled_times_p,
                profiled_times_e=mobilenet_profiled_times_e,
            )
            mobilenet_job_name = "mobilenet_v2"
            print(f"   Created {mobilenet_job_name} workload with {len(mobilenet_workload.operations)} operations")
            step_num += 1
        
        # Create Diffusion workload if requested
        if use_diffusion:
            print(f"\n{step_num}. Loading Diffusion dispatch graph...")
            diffusion_workload, diffusion_job_name = create_workload_func(
                diffusion_path,
                name_prefix="diffusion_",
                profiled_times_p=diffusion_profiled_times_p,
                profiled_times_e=diffusion_profiled_times_e,
            )
            diffusion_job_name = "diffusion"
            print(f"   Created {diffusion_job_name} workload with {len(diffusion_workload.operations)} operations")
    
    # Make dronet depend on first network, mobilenet, and/or diffusion
    # Special case: if both mobilenet and diffusion are enabled, dronet depends on both
    if mlp_debug:
        # Skip dependency setup for mlp_debug mode
        pass
    else:
        step_num = 4 if not no_depth_network else 3
        dependencies_added = []
        
        # Add dependency on first network if included
        if not no_depth_network:
            print(f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {first_network_job_name}...")
            add_dependency(first_network_workload, dronet_workload)
            dependencies_added.append(first_network_job_name)
            step_num += 1
        
        # Add dependency on mobilenet if enabled
        if use_mobilenet:
            if dependencies_added:
                print(f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {mobilenet_job_name}...")
            else:
                print(f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {mobilenet_job_name}...")
            add_dependency(mobilenet_workload, dronet_workload)
            dependencies_added.append(mobilenet_job_name)
            step_num += 1
        
        # Add dependency on diffusion if enabled
        if use_diffusion:
            if dependencies_added:
                print(f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {diffusion_job_name}...")
            else:
                print(f"\n{step_num}. Adding dependency: {dronet_job_name} depends on {diffusion_job_name}...")
            add_dependency(diffusion_workload, dronet_workload)
            dependencies_added.append(diffusion_job_name)
            step_num += 1
        
        if not dependencies_added:
            print("\n3. Dronet is independent (no dependencies)")
    
    # Make each MLP instance depend on the previous one (MLP instances are independent of first network/Dronet)
    # Note: MobilenetV2 and Diffusion dependencies are handled above (they feed into dronet)
    if not use_mobilenet and not use_diffusion:
        step_num = 5 if not no_depth_network else 4
        if len(mlp_workloads) > 1:
            first_network_label = first_network_name.capitalize() if not no_depth_network else "Dronet"
            print(f"\n{step_num}. Adding dependencies between MLP instances (MLPs are independent of {first_network_label}/Dronet)...")
            for i in range(1, len(mlp_workloads)):
                print(f"   {mlp_job_names[i]} depends on {mlp_job_names[i-1]}...")
                add_dependency(mlp_workloads[i-1], mlp_workloads[i])
        else:
            print(f"\n{step_num}. Single MLP instance (no dependencies between MLPs)")
    else:
        # MobilenetV2 and Diffusion are already set up as dependencies for dronet above
        # They don't have dependencies between themselves
        pass
    
    # Combine workloads
    # Dependency structure:
    # - First network (if included) → Dronet
    # - MobilenetV2 (if enabled) → Dronet
    # - Diffusion (if enabled) → Dronet
    # - MLP instances form their own chain: MLP0 → MLP1 → MLP2 → MLP3 → MLP4 (if not using mobilenet/diffusion)
    # OR in mlp_debug mode, only MLP is scheduled
    if mlp_debug:
        step_num = 2
        print(f"\n{step_num}. Combining workloads...")
        print(f"   Dependency chains:")
        print(f"     Chain 1: {' → '.join(mlp_job_names) if len(mlp_job_names) > 1 else mlp_job_names[0] + ' (independent)'}")
        all_workloads = mlp_workloads
        all_job_names = mlp_job_names
        # Create job_id mapping: MLP=0
        job_id_mapping = [0] * len(mlp_workloads)
        all_job_names_for_legend = ["MLP"] * len(mlp_workloads)
    else:
        step_num = 6 if not no_depth_network else 5
        print(f"\n{step_num}. Combining workloads...")
        print(f"   Dependency chains:")
        # Build dependency chain for dronet
        dronet_deps = []
        if not no_depth_network:
            dronet_deps.append(first_network_job_name)
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
        print(f"   (Chains are independent and can run in parallel)")
        
        if not use_mobilenet and not use_diffusion:
            if not no_depth_network:
                all_workloads = [first_network_workload, dronet_workload] + mlp_workloads
                all_job_names = [first_network_job_name, dronet_job_name] + mlp_job_names
                # Create job_id mapping: First network=0, Dronet=1, all MLPs=2 (same color for all MLPs)
                job_id_mapping = [0, 1] + [2] * len(mlp_workloads)  # All MLPs get job_id=2
                # Use a single name "MLP" for all MLP instances in the legend
                all_job_names_for_legend = [first_network_job_name, dronet_job_name] + ["MLP"] * len(mlp_workloads)
            else:
                all_workloads = [dronet_workload] + mlp_workloads
                all_job_names = [dronet_job_name] + mlp_job_names
                # Create job_id mapping: Dronet=0, all MLPs=1 (same color for all MLPs)
                job_id_mapping = [0] + [1] * len(mlp_workloads)
                # Use a single name "MLP" for all MLP instances in the legend
                all_job_names_for_legend = [dronet_job_name] + ["MLP"] * len(mlp_workloads)
        else:
            # Build workload list: first network (if included), then mobilenet/diffusion (if enabled), then dronet
            all_workloads = []
            all_job_names = []
            job_id_mapping = []
            all_job_names_for_legend = []
            job_id = 0
            
            if not no_depth_network:
                all_workloads.append(first_network_workload)
                all_job_names.append(first_network_job_name)
                job_id_mapping.append(job_id)
                all_job_names_for_legend.append(first_network_job_name)
                job_id += 1
            
            if use_mobilenet:
                all_workloads.append(mobilenet_workload)
                all_job_names.append(mobilenet_job_name)
                job_id_mapping.append(job_id)
                all_job_names_for_legend.append(mobilenet_job_name)
                job_id += 1
            
            if use_diffusion:
                all_workloads.append(diffusion_workload)
                all_job_names.append(diffusion_job_name)
                job_id_mapping.append(job_id)
                all_job_names_for_legend.append(diffusion_job_name)
                job_id += 1
            
            # Dronet comes last (depends on all above)
            all_workloads.append(dronet_workload)
            all_job_names.append(dronet_job_name)
            job_id_mapping.append(job_id)
            all_job_names_for_legend.append(dronet_job_name)
    
    combined_workload = combine_workloads(all_workloads, job_names=all_job_names_for_legend, job_id_mapping=job_id_mapping)
    
    print(f"\nCombined workload statistics:")
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
    
    # Schedule the combined workload using greedy algorithm
    print("\n" + "=" * 60)
    print("Scheduling combined workload using greedy algorithm...")
    print("=" * 60)
    t, alpha = greedy_schedule(combined_workload)
    
    # Calculate makespan
    machine_combinations = combined_workload.get_machine_combinations()
    makespan = max(
        t[i] + combined_workload.operations[i].get_duration_for_combination(
            np.argmax(alpha[i]), machine_combinations, combined_workload.machines
        )
        for i in range(len(combined_workload.operations))
    )
    
    print(f"\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each combination
    num_combinations = len(machine_combinations)
    combination_counts = [0] * num_combinations
    for i in range(len(alpha)):
        combo_idx = np.argmax(alpha[i])
        combination_counts[combo_idx] += 1
    
    print(f"\nCombination assignments:")
    for combo_idx, count in enumerate(combination_counts):
        combo_str = "+".join(machine_combinations[combo_idx]) if len(machine_combinations[combo_idx]) > 1 else machine_combinations[combo_idx][0]
        print(f"  {combo_str}: {count} operations")
    
    # For backward compatibility, also show per-machine counts if using singleton combinations
    if all(len(combo) == 1 for combo in machine_combinations):
        # Traditional machine-based assignment
        cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
        cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)
        print(f"\nCore assignments (backward compatibility):")
        print(f"  CPU_P (performant): {cpu_p_count} operations")
        print(f"  CPU_E (efficient): {cpu_e_count} operations")
    
    # Count operations per network (based on order in all_workloads: first network (if included), then mobilenet/diffusion (if enabled), then dronet, then MLP instances)
    current_idx = 0
    first_network_indices = []
    mobilenet_indices = []
    diffusion_indices = []
    dronet_indices = []
    mlp_indices_list = []
    
    if not mlp_debug:
        if not no_depth_network:
            first_network_indices = list(range(current_idx, current_idx + len(first_network_workload.operations)))
            current_idx += len(first_network_workload.operations)
        
        # Mobilenet and diffusion come before dronet when enabled
        if use_mobilenet:
            mobilenet_indices = list(range(current_idx, current_idx + len(mobilenet_workload.operations)))
            current_idx += len(mobilenet_workload.operations)
        if use_diffusion:
            diffusion_indices = list(range(current_idx, current_idx + len(diffusion_workload.operations)))
            current_idx += len(diffusion_workload.operations)
        
        if dronet_workload:
            dronet_indices = list(range(current_idx, current_idx + len(dronet_workload.operations)))
            current_idx += len(dronet_workload.operations)
    
    # Calculate indices for MLP instances (only if not using mobilenet/diffusion)
    if not use_mobilenet and not use_diffusion:
        for mlp_workload in mlp_workloads:
            mlp_indices = list(range(current_idx, current_idx + len(mlp_workload.operations)))
            mlp_indices_list.append(mlp_indices)
            current_idx += len(mlp_workload.operations)
    
    # Count per-network combination assignments
    print(f"\nPer-network combination assignments:")
    for combo_idx in range(num_combinations):
        combo_str = "+".join(machine_combinations[combo_idx]) if len(machine_combinations[combo_idx]) > 1 else machine_combinations[combo_idx][0]
        print(f"  {combo_str}:")
        if mlp_debug:
            # Only MLP in debug mode
            for i, mlp_indices in enumerate(mlp_indices_list):
                mlp_count = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == combo_idx)
                print(f"    {mlp_job_names[i].capitalize()}: {mlp_count} operations")
        else:
            if not no_depth_network and first_network_indices:
                first_network_count = sum(1 for i in first_network_indices if np.argmax(alpha[i]) == combo_idx)
                print(f"    {first_network_job_name.capitalize()}: {first_network_count} operations")
            if dronet_indices:
                dronet_count = sum(1 for i in dronet_indices if np.argmax(alpha[i]) == combo_idx)
                print(f"    {dronet_job_name.capitalize()}: {dronet_count} operations")
            if not use_mobilenet and not use_diffusion:
                for i, mlp_indices in enumerate(mlp_indices_list):
                    mlp_count = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == combo_idx)
                    print(f"    {mlp_job_names[i].capitalize()}: {mlp_count} operations")
            else:
                if use_mobilenet:
                    mobilenet_count = sum(1 for idx in mobilenet_indices if np.argmax(alpha[idx]) == combo_idx)
                    print(f"    {mobilenet_job_name.capitalize()}: {mobilenet_count} operations")
                if use_diffusion:
                    diffusion_count = sum(1 for idx in diffusion_indices if np.argmax(alpha[idx]) == combo_idx)
                    print(f"    {diffusion_job_name.capitalize()}: {diffusion_count} operations")
    
    # For backward compatibility, also show per-machine counts if using singleton combinations
    if all(len(combo) == 1 for combo in machine_combinations):
        print(f"\nPer-network core assignments (backward compatibility):")
        if mlp_debug:
            # Only MLP in debug mode
            for i, mlp_indices in enumerate(mlp_indices_list):
                mlp_cpu_p = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 0)
                mlp_cpu_e = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 1)
                print(f"  {mlp_job_names[i].capitalize()}: CPU_P={mlp_cpu_p}, CPU_E={mlp_cpu_e}")
        else:
            if not no_depth_network and first_network_indices:
                first_network_cpu_p = sum(1 for i in first_network_indices if np.argmax(alpha[i]) == 0)
                first_network_cpu_e = sum(1 for i in first_network_indices if np.argmax(alpha[i]) == 1)
                print(f"  {first_network_job_name.capitalize()}: CPU_P={first_network_cpu_p}, CPU_E={first_network_cpu_e}")
            if dronet_indices:
                dronet_cpu_p = sum(1 for i in dronet_indices if np.argmax(alpha[i]) == 0)
                dronet_cpu_e = sum(1 for i in dronet_indices if np.argmax(alpha[i]) == 1)
                print(f"  {dronet_job_name.capitalize()}: CPU_P={dronet_cpu_p}, CPU_E={dronet_cpu_e}")
            if not use_mobilenet and not use_diffusion:
                for i, mlp_indices in enumerate(mlp_indices_list):
                    mlp_cpu_p = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 0)
                    mlp_cpu_e = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 1)
                    print(f"  {mlp_job_names[i].capitalize()}: CPU_P={mlp_cpu_p}, CPU_E={mlp_cpu_e}")
            else:
                if use_mobilenet:
                    mobilenet_cpu_p = sum(1 for idx in mobilenet_indices if np.argmax(alpha[idx]) == 0)
                    mobilenet_cpu_e = sum(1 for idx in mobilenet_indices if np.argmax(alpha[idx]) == 1)
                    print(f"  {mobilenet_job_name.capitalize()}: CPU_P={mobilenet_cpu_p}, CPU_E={mobilenet_cpu_e}")
                if use_diffusion:
                    diffusion_cpu_p = sum(1 for idx in diffusion_indices if np.argmax(alpha[idx]) == 0)
                    diffusion_cpu_e = sum(1 for idx in diffusion_indices if np.argmax(alpha[idx]) == 1)
                    print(f"  {diffusion_job_name.capitalize()}: CPU_P={diffusion_cpu_p}, CPU_E={diffusion_cpu_e}")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create title showing the dependency chain
    if not use_mobilenet and not use_diffusion:
        mlp_chain = " → ".join([name.capitalize() for name in mlp_job_names])
    else:
        # Build chain showing mobilenet and/or diffusion feeding into dronet
        chains = []
        if use_mobilenet:
            chains.append(mobilenet_job_name.capitalize())
        if use_diffusion:
            chains.append(diffusion_job_name.capitalize())
        if chains:
            mlp_chain = " + ".join(chains) + " → " + dronet_job_name.capitalize()
        else:
            mlp_chain = dronet_job_name.capitalize()
    
    plot_suffix = "Greedy"
    if use_profiled:
        plot_suffix += " (Profiled)"
    if use_grouped:
        plot_suffix += " (Grouped)"
    if mlp_debug:
        plot_suffix += " (MLP Debug)"
    plot_filename = "iree_combined_schedule_greedy.png"
    if use_profiled and use_grouped:
        plot_filename = "iree_combined_schedule_greedy_profiled_grouped.png"
    elif use_profiled:
        plot_filename = "iree_combined_schedule_greedy_profiled.png"
    elif use_grouped:
        plot_filename = "iree_combined_schedule_greedy_grouped.png"
    if mlp_debug:
        plot_filename = plot_filename.replace(".png", "_mlp_debug.png")
    elif use_diffusion:
        plot_filename = plot_filename.replace(".png", "_diffusion.png")
    
    # Determine if using combinations and prepare labels
    machine_combinations = combined_workload.get_machine_combinations()
    using_combinations = len(machine_combinations) > len(combined_workload.machines)
    
    # Create plot title
    if mlp_debug:
        plot_title = f"{mlp_chain} Schedule on Dual-Core Device ({plot_suffix})"
    elif not no_depth_network:
        plot_title = f"{first_network_job_name.capitalize()} → {dronet_job_name.capitalize()} + {mlp_chain} Schedule on Dual-Core Device ({plot_suffix})"
    else:
        plot_title = f"{dronet_job_name.capitalize()} + {mlp_chain} Schedule on Dual-Core Device ({plot_suffix})"
    
    # Add makespan to plot title
    plot_title = f"{plot_title} (Makespan: {makespan:.2f} ms)"
    
    if using_combinations:
        # Create labels for machine combinations
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
            save_path=f"plots/{plot_filename}",
            plot_title=plot_title,
            workload=combined_workload
        )
    else:
        # Traditional machine-based plotting
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
            workload=combined_workload
        )
    
    print(f"\nPlot saved to plots/{plot_filename}")
    
    # Output combined JSON file with scheduling information
    json_output_path = "schedules/combined_schedule_greedy.json"
    if use_profiled:
        json_output_path = "schedules/combined_schedule_greedy_profiled.json"
    if use_grouped:
        json_output_path = json_output_path.replace(".json", "_grouped.json")
    if mlp_debug:
        json_output_path = "schedules/combined_schedule_greedy_mlp_debug.json"
    elif no_depth_network:
        json_output_path = json_output_path.replace(".json", "_no_depth.json")
    if use_mobilenet and use_diffusion:
        json_output_path = json_output_path.replace(".json", "_mobilenet_diffusion.json")
    elif use_mobilenet:
        json_output_path = json_output_path.replace(".json", "_mobilenet.json")
    elif use_diffusion:
        json_output_path = json_output_path.replace(".json", "_diffusion.json")
    
    # Collect all profiled data for JSON output
    # Merge profiled data from all workloads (first_network, dronet, mlp/mobilenet)
    all_profiled_times_p = {}
    all_profiled_times_e = {}
    
    if use_profiled:
        # Add first network profiled data
        if not no_depth_network and first_network_profiled_times_p:
            all_profiled_times_p.update(first_network_profiled_times_p)
        if not no_depth_network and first_network_profiled_times_e:
            all_profiled_times_e.update(first_network_profiled_times_e)
        
        # Add mobilenet profiled data if using mobilenet
        if use_mobilenet and mobilenet_profiled_times_p:
            all_profiled_times_p.update(mobilenet_profiled_times_p)
        if use_mobilenet and mobilenet_profiled_times_e:
            all_profiled_times_e.update(mobilenet_profiled_times_e)
        
        # Add diffusion profiled data if using diffusion
        if use_diffusion and diffusion_profiled_times_p:
            all_profiled_times_p.update(diffusion_profiled_times_p)
        if use_diffusion and diffusion_profiled_times_e:
            all_profiled_times_e.update(diffusion_profiled_times_e)
        
        # Add dronet profiled data
        if dronet_profiled_times_p:
            all_profiled_times_p.update(dronet_profiled_times_p)
        if dronet_profiled_times_e:
            all_profiled_times_e.update(dronet_profiled_times_e)
    
    print(f"\nOutputting scheduled JSON...")
    output_scheduled_json(
        all_workloads=all_workloads,
        all_job_names=all_job_names,
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=all_profiled_times_p if all_profiled_times_p else None,
        profiled_times_e=all_profiled_times_e if all_profiled_times_e else None
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
        json_files.append((first_network_prefix, first_network_path))
    json_files.append(("dronet_", dronet_path))
    if not use_mobilenet and not use_diffusion:
        for i in range(5 if not mlp_debug else 1):
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
    
    # Build network-specific profiled times to avoid dispatch_id collisions
    profiled_times_by_network = {}
    if use_profiled:
        if not no_depth_network and first_network_profiled_times_p:
            network_key = "glpdepth" if use_glpdepth else "fast"
            profiled_times_by_network[network_key] = {
                "p": first_network_profiled_times_p,
                "e": first_network_profiled_times_e
            }
        # Add dronet profiled times if available
        if dronet_profiled_times_p:
            profiled_times_by_network["dronet"] = {
                "p": dronet_profiled_times_p,
                "e": dronet_profiled_times_e
            }
        if not use_mobilenet and not use_diffusion:
            # MLP profiled times are not loaded in greedy scheduler currently
            pass
        else:
            if use_mobilenet and mobilenet_profiled_times_p:
                profiled_times_by_network["mobilenet"] = {
                    "p": mobilenet_profiled_times_p,
                    "e": mobilenet_profiled_times_e
                }
            if use_diffusion and diffusion_profiled_times_p:
                profiled_times_by_network["diffusion"] = {
                    "p": diffusion_profiled_times_p,
                    "e": diffusion_profiled_times_e
                }
    
    # Run validation
    os.makedirs("validation_reports", exist_ok=True)
    validation_file = "validation_reports/greedy_validation_report.txt"
    
    is_valid, validation_results = validate_schedule(
        combined_workload,
        t,
        alpha,
        combined_json_data,
        profiled_times_p=all_profiled_times_p if all_profiled_times_p else None,
        profiled_times_e=all_profiled_times_e if all_profiled_times_e else None,
        profiled_times_by_network=profiled_times_by_network if profiled_times_by_network else None,
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

def greedy_schedule_iree_networks(
    networks_json_path: str,
    *,
    use_profiled: bool = True,
    p_core_speedup: float | None = None,
    random_seed: int | None = None,
    max_periodic_iters: int = 4,
    save_outputs: bool = True,
) -> tuple[Workload, np.ndarray, np.ndarray, float]:
    """Greedy-schedule a toplevel networks JSON (same spec as run_xpurt_schedule).

    Mirrors `schedule_iree_networks` in run_xpurt_schedule.py up to the
    workload-creation step, then swaps the MILP `schedule()` call for the
    greedy `greedy_schedule()` defined in this module.  The result is a
    fast pessimistic makespan estimate that bounds what the MILP can
    achieve under the same dependency / hardware constraints — useful as
    a horizon for periodic-instance count, and as a sanity check for the
    MILP's solution.

    Periodic networks are expanded by `create_workload_from_network_hierarchy`,
    so this entry point handles spec files with `period` / `window_duration`
    fields directly — no manual periodic unrolling needed.

    Periodic-count refinement: after the first greedy pass we compare the
    achieved makespan against ceil(makespan/period) for every periodic
    network.  If any network is short of that count, inject a
    `num_instances` override into networks_data and re-run greedy.  We
    iterate up to `max_periodic_iters` times or until counts converge —
    in practice 2-3 passes are enough.  This is the "greedy-bootstrapped
    horizon" suggestion: instead of the analytical
    `s_np / (1 - F_p)` heuristic in workload_factory, we use the actual
    greedy makespan, which already accounts for DAG serialisation and
    HW contention.

    Returns (workload, t, alpha, makespan) for the final (converged) pass.
    """
    repo_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Greedy scheduler: loading network hierarchy from JSON...")
    print("=" * 60)
    print(f"\nLoading networks from: {networks_json_path}")

    networks_data, cfg = load_networks_config(networks_json_path)
    if p_core_speedup is not None:
        cfg["p_core_speedup"] = p_core_speedup
    if random_seed is not None:
        cfg["random_seed"] = random_seed
    if use_profiled is not None:
        cfg["use_profiled"] = use_profiled

    machine_core_counts = cfg["machine_core_counts"]
    cpu_p_profile_hw = cfg["cpu_p_profile_hw"]
    cpu_e_profile_hw = cfg["cpu_e_profile_hw"]
    profile_target = cfg["profile_target"]
    profile_topo_tag = cfg["profile_topo_tag"]
    profile_topo_tag_override = cfg["profile_topo_tag_override"]
    profile_topo_tag_per_hw = cfg["profile_topo_tag_per_hw"]
    effective_p_core_speedup = cfg["p_core_speedup"]
    effective_random_seed = cfg["random_seed"]
    effective_use_profiled = cfg["use_profiled"]
    # Honor the schedule's restrict_makespan_to_nonperiodic flag in the
    # iterative-greedy refinement loop. Without it, each pass measures
    # makespan over ALL ops (including periodic), which feeds back into
    # ceil(makespan/period) and makes the scheduler add MORE periodic
    # instances each iteration, runaway. The right semantic is
    # makespan = max-end-time of non-periodic ops (the actual workload
    # boundary); periodic instances are then sized to cover that.
    effective_restrict_makespan_to_nonperiodic = bool(
        cfg.get("restrict_makespan_to_nonperiodic", True)
    )

    rng = np.random.default_rng(effective_random_seed)

    networks = networks_data.get("networks", {})
    edges = networks_data.get("edges", [])
    print(f"\nFound {len(networks)} networks:")
    for network_id, network_info in networks.items():
        period = network_info.get("period")
        ann = f" (periodic: period={period}ms)" if period is not None else ""
        print(f"  - {network_id} (id: {network_info.get('id')}){ann}")
    if edges:
        print(f"\nFound {len(edges)} network-level dependencies")

    print("\nResolved runtime configuration:")
    print(f"  Machine core counts: {machine_core_counts}")
    print(f"  Profile HW mapping: {CPU_P}->{cpu_p_profile_hw}, {CPU_E}->{cpu_e_profile_hw}")
    print(f"  Profile target/topology: target={profile_target}, topo_tag={profile_topo_tag}")

    machines, machine_combinations = build_machine_combinations(machine_core_counts)
    n_cores = len(machines)
    transfer_times = np.zeros((n_cores, n_cores))

    combo_hw = []
    for combo in machine_combinations:
        core_type = machine_type_prefix(combo[0])
        combo_hw.append(cpu_p_profile_hw if core_type == CPU_P else cpu_e_profile_hw)

    processing_times = None
    if effective_use_profiled:
        print("\nUsing profiled runtimes where available...")
        if profile_topo_tag_per_hw:
            tt_override = dict(profile_topo_tag_per_hw)
        elif profile_topo_tag_override:
            tt_override = profile_topo_tag
        else:
            tt_override = None
        processing_times, _, _ = load_profiled_processing_times(
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

    # Iterative greedy: build workload, run greedy, measure makespan.  If
    # any periodic network's instance count is below ceil(makespan/period),
    # inject a num_instances override and re-run.  Converges when no
    # network needs more instances.
    #
    # When restrict_makespan_to_nonperiodic is set, do an initial
    # zero-periodic pass first: build a workload with num_instances=1
    # for every periodic network (the smallest model the workload
    # factory will accept) and use the resulting non-periodic makespan
    # as the seed for periodic-count sizing. The default
    # profile_horizon already inflates by 1/(1-F_p) to account for
    # periodic CPU steal, which produces a high initial periodic
    # count; under that count, the joint greedy schedule finds
    # equilibria where yolov8 *also* runs to that horizon (because the
    # 24 dronet instances contend with yolov8 on hart 1), so the loop
    # converges immediately at the inflated count without ever testing
    # whether fewer periodic instances would let non-periodic finish
    # sooner.  Seeding from no-contention finish gives a much tighter
    # starting point; the iter loop then grows from there if real
    # contention pushes non-periodic finish out.
    workload = None
    t = None
    alpha = None
    makespan = 0.0
    prev_counts: dict[str, int] = {}

    if effective_restrict_makespan_to_nonperiodic:
        for net_id, net_info in networks.items():
            if net_info.get("period") is not None:
                networks_data["networks"][net_id]["num_instances"] = 1

    for it in range(max_periodic_iters):
        print(f"\n--- Greedy iteration {it + 1} ---")
        print("Creating workload from network hierarchy...")
        workload = create_workload_from_network_hierarchy(
            networks_data=networks_data,
            repo_base_path=repo_base_path,
            machines=machines,
            transfer_times=transfer_times,
            p_core_speedup=effective_p_core_speedup,
            random_seed=effective_random_seed,
            processing_times=processing_times,
            machine_combinations=machine_combinations,
        )
        print(f"  Total operations: {len(workload.operations)}")
        print(f"  Job names: {workload.job_names}")

        print("Greedy-scheduling combined workload...")
        t, alpha = greedy_schedule(workload)

        machine_combinations_list = workload.get_machine_combinations()
        # Identify periodic networks so we can optionally exclude their
        # ops from the makespan reference. job_names look like
        # "<id>0", "<id>1", ... for periodic instances; non-periodic
        # job_names equal the network id.
        periodic_net_ids = {
            net_id for net_id, info in networks.items()
            if info.get("period") is not None
        }
        def _is_periodic_op(op_idx: int) -> bool:
            jn = workload.job_names[op_idx] if op_idx < len(workload.job_names) else ""
            if not isinstance(jn, str):
                return False
            for nid in periodic_net_ids:
                # periodic instance: "<id><digits>" with non-empty digits.
                if jn.startswith(nid) and jn[len(nid):].isdigit() and jn != nid:
                    return True
            return False
        makespan = 0.0
        makespan_all = 0.0  # for diagnostic
        for i, op in enumerate(workload.operations):
            combo_idx = int(np.argmax(alpha[i]))
            dur = op.get_duration_for_combination(
                combo_idx, machine_combinations_list, workload.machines
            )
            finish = float(t[i]) + float(dur)
            if finish > makespan_all:
                makespan_all = finish
            if effective_restrict_makespan_to_nonperiodic and _is_periodic_op(i):
                continue
            if finish > makespan:
                makespan = finish
        if effective_restrict_makespan_to_nonperiodic:
            print(f"  Makespan: {makespan:.2f} ms (non-periodic only; "
                  f"all-ops max-end = {makespan_all:.2f} ms)")
        else:
            print(f"  Makespan: {makespan:.2f} ms")

        # For each periodic network, count current instances (#networks
        # whose identifier starts with `<base>` and has a numeric suffix)
        # vs the count needed to fully cover the makespan.  Bump via
        # num_instances override if short.  Don't shrink — periodic
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
            needed = max(1, int(np.ceil(makespan / T)))
            current = int(net_info.get("num_instances") or prev_counts.get(net_id, 0))
            if current == 0:
                # No prior override — count from job_names ('<id>0', '<id>1', ...).
                current = sum(
                    1 for n in workload.job_names
                    if isinstance(n, str) and n.startswith(net_id) and n[len(net_id):].isdigit()
                )
            print(f"  Periodic '{net_id}': period={T:.0f}ms current={current} needed={needed}")
            needed_counts[net_id] = needed
            prev_counts[net_id] = current

        # If every periodic network already has enough instances, we're done.
        if all(prev_counts.get(k, 0) >= v for k, v in needed_counts.items()):
            print("  All periodic counts cover makespan — converged.")
            break

        # Otherwise, inject overrides and iterate.  Bump only the ones that
        # are short; leave the rest alone so we don't over-grow networks
        # whose makespan share is small.
        bumped = []
        for net_id, needed in needed_counts.items():
            cur = prev_counts.get(net_id, 0)
            if cur < needed:
                networks_data["networks"][net_id]["num_instances"] = needed
                bumped.append((net_id, cur, needed))
        if not bumped:
            break
        print("  Bumping num_instances:", ", ".join(f"{n}: {a}->{b}" for n, a, b in bumped))

    print(f"\nFinal greedy makespan: {makespan:.2f} ms (after {it + 1} iteration{'s' if it else ''})")

    if save_outputs:
        # Mirror run_xpurt_schedule.py's outputs but tagged "greedy" so
        # they sit alongside MILP runs without overwriting.
        from postprocessing import output_scheduled_json

        os.makedirs("plots", exist_ok=True)
        os.makedirs("schedules", exist_ok=True)

        json_basename = os.path.splitext(os.path.basename(networks_json_path))[0]
        suffix = "_profiled" if effective_use_profiled else ""
        plot_path = f"plots/{json_basename}_greedy{suffix}.png"
        sched_path = f"schedules/scheduled_{json_basename}_greedy{suffix}.json"

        # Title + counts
        num_jobs = sum(1 for op in workload.operations if not op.predecessors)
        network_names_in_plot = [
            workload.job_names[i] if i < len(workload.job_names) else f"Job {i}"
            for i in sorted(set(op.job_id for op in workload.operations))
        ]
        title = " + ".join(name.capitalize() for name in network_names_in_plot)

        plot.plot_optimization_schedule(
            workload.get_durations(),
            t,
            alpha,
            num_jobs,
            len(workload.machines),
            workload.machines,
            workload.get_transfer_times(),
            save_path=plot_path,
            plot_title=f"{title} Greedy Schedule ({n_cores} cores) — makespan {makespan:.0f} ms",
            workload=workload,
        )
        print(f"Plot saved to {plot_path}")

        output_scheduled_json(
            combined_workload=workload,
            t=t,
            alpha=alpha,
            output_path=sched_path,
        )
        print(f"Scheduled JSON saved to {sched_path}")

    return workload, t, alpha, makespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE dispatch graphs (fast/glpdepth, dronet, and MLP/MobilenetV2/Diffusion) on a dual-core device using greedy scheduling"
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default=None,
        help="Path to a toplevel networks_*.json (same spec as run_xpurt_schedule.py). "
             "When provided, runs the network-hierarchy path with periodic expansion "
             "and prints the greedy makespan; bypasses the legacy hardcoded-network mode."
    )
    parser.add_argument(
        "--use-glpdepth",
        action="store_true",
        help="Use glpdepth instead of fast as the first network (default: use fast)"
    )
    parser.add_argument(
        "--use-profiled",
        action="store_true",
        help="Use profiled runtimes from CSV files (currently only supported for glpdepth when --use-glpdepth is set, MobilenetV2 when --use-mobilenet is set, or Diffusion when --diffusion is set)"
    )
    parser.add_argument(
        "--use-grouped",
        action="store_true",
        help="Use machine combinations (CPU_P, CPU_E, CPU_P+CPU_E) as additional scheduling options"
    )
    parser.add_argument(
        "--no-depth-network",
        action="store_true",
        help="Skip loading and scheduling the depth network (fast/glpdepth). Only schedule Dronet and MLP/MobilenetV2."
    )
    parser.add_argument(
        "--use-mobilenet",
        action="store_true",
        help="Use a single MobilenetV2 instead of 5 MLP instances."
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="Use a single Diffusion model instead of 5 MLP instances."
    )
    parser.add_argument(
        "--mlp-debug",
        action="store_true",
        help="Only schedule a single MLP instance instead of 5 (for debugging)."
    )
    args = parser.parse_args()

    # Network-hierarchy path (preferred): same JSON format as run_xpurt_schedule.
    if args.networks_json:
        greedy_schedule_iree_networks(
            networks_json_path=args.networks_json,
            use_profiled=args.use_profiled if args.use_profiled else True,
        )
        sys.exit(0)

    # Check mutual exclusivity: mlp_debug is exclusive with mobilenet/diffusion
    if args.mlp_debug and (args.use_mobilenet or args.diffusion):
        parser.error("--mlp-debug is mutually exclusive with --use-mobilenet and --diffusion")

    schedule_iree_networks(
        use_glpdepth=args.use_glpdepth,
        use_profiled=args.use_profiled,
        use_grouped=args.use_grouped,
        no_depth_network=args.no_depth_network,
        use_mobilenet=args.use_mobilenet,
        use_diffusion=args.diffusion,
        mlp_debug=args.mlp_debug
    )

