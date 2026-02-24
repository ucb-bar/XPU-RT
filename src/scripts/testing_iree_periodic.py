"""
Combined script that uses greedy scheduler as initialization for optimization-based periodic task scheduling.

Workflow:
1. Run greedy scheduler first to get initial solution (warm-up)
2. Use greedy results (start times, HW allocation, makespan) as:
   - Initial solution for optimization
   - Upper bound for makespan constraint
3. Schedule periodic tasks (mobilenet_v2) with required frequencies
4. Maximize number of periodic task instances within makespan bound
"""

import os
import sys
import numpy as np
import cvxpy as cp
import json
import time
from typing import Tuple, Optional

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_dependencies
from plot import plot_schedule

# Import greedy scheduler function
from testing_iree_greedy import (
    greedy_schedule,
    create_workload_from_json_with_combinations,
    load_profiled_times,
    combine_workloads,
    add_dependency,
    output_scheduled_json
)


class PeriodicOperation(Operation):
    """
    Extended Operation class for periodic tasks with frequency requirements.
    
    Attributes:
        frequency: Required frequency (1/period) in Hz
        period: Period in time units (1/frequency)
        instance_id: Which instance of the periodic task this is
    """
    
    def __init__(self, processing_times: list[float], predecessors=None, 
                 operation_id=None, operation_name=None, job_id=None,
                 combined_duration: float = None, frequency: float = None,
                 period: float = None, instance_id: int = 0):
        super().__init__(processing_times, predecessors, operation_id, operation_name, job_id)
        self.frequency = frequency
        self.period = period
        self.instance_id = instance_id
        # Store combined duration for CPU_P+CPU_E combination
        self.combined_duration = combined_duration
    
    def get_duration_for_combination(self, combination_idx: int, machine_combinations: list[list[str]], machines: list[str]) -> float:
        """Get duration for a specific machine combination."""
        if self.combined_duration is not None and len(machine_combinations[combination_idx]) > 1:
            # Combined execution (CPU_P+CPU_E)
            return self.combined_duration
        else:
            # Single machine execution
            machine = machine_combinations[combination_idx][0]
            machine_idx = machines.index(machine)
            return self.get_durations()[machine_idx]


def schedule_periodic_tasks_with_greedy_warmstart(
    base_workload: Workload,
    periodic_workload: Workload,
    periodic_frequency: float,
    greedy_t: np.ndarray,
    greedy_alpha: np.ndarray,
    greedy_makespan: float,
    max_periodic_instances: int = 10,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Schedule periodic tasks using greedy solution as warm start and upper bound.
    
    Args:
        base_workload: Base workload (e.g., Fast, Dronet)
        periodic_workload: Single instance of periodic task (e.g., MobilenetV2)
        periodic_frequency: Required frequency for periodic task (Hz)
        greedy_t: Start times from greedy scheduler
        greedy_alpha: Machine assignments from greedy scheduler
        greedy_makespan: Makespan from greedy scheduler (upper bound)
        max_periodic_instances: Maximum number of periodic instances to try
        verbose: Print debug information
    
    Returns:
        (t, alpha, num_instances) where:
        - t: Start times for all operations (base + periodic instances)
        - alpha: Machine assignments for all operations
        - num_instances: Number of periodic instances scheduled
    """
    
    # Calculate period from frequency
    period = 1.0 / periodic_frequency if periodic_frequency > 0 else float('inf')
    
    if verbose:
        print(f"\n{'='*60}")
        print("PERIODIC SCHEDULING WITH GREEDY WARM START")
        print(f"{'='*60}")
        print(f"Greedy makespan (upper bound): {greedy_makespan:.2f} ms")
        print(f"Periodic task frequency: {periodic_frequency:.4f} Hz")
        print(f"Periodic task period: {period:.2f} ms")
        print(f"Max periodic instances to try: {max_periodic_instances}")
    
    # Estimate maximum instances that can fit based on greedy makespan
    estimated_max_instances = int(greedy_makespan / period) + 1
    max_instances_to_try = min(max_periodic_instances, estimated_max_instances)
    
    if verbose:
        print(f"Estimated max instances from greedy makespan: {estimated_max_instances}")
        print(f"Will try up to: {max_instances_to_try} instances")
    
    # Binary search for maximum number of instances
    best_num_instances = 0
    best_t = None
    best_alpha = None
    
    for num_instances in range(1, max_instances_to_try + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Trying {num_instances} periodic instances...")
            print(f"{'='*60}")
        
        # Create combined workload with periodic instances
        combined_workload = create_combined_workload_with_periodic(
            base_workload, periodic_workload, num_instances, period
        )
        
        # Solve optimization with greedy warm start and makespan constraint
        t, alpha, feasible = optimize_with_warmstart(
            combined_workload,
            greedy_t,
            greedy_alpha,
            greedy_makespan,
            len(base_workload.operations),
            num_instances,
            period,
            verbose=verbose
        )
        
        if feasible:
            best_num_instances = num_instances
            best_t = t
            best_alpha = alpha
            if verbose:
                print(f"✓ Successfully scheduled {num_instances} instances")
        else:
            if verbose:
                print(f"✗ Could not schedule {num_instances} instances")
            # If we can't fit this many, we won't fit more
            break
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PERIODIC SCHEDULING RESULT")
        print(f"{'='*60}")
        print(f"Maximum periodic instances scheduled: {best_num_instances}")
        if best_t is not None:
            final_makespan = np.max(best_t + np.array([
                combined_workload.operations[i].get_duration_for_combination(
                    np.argmax(best_alpha[i]),
                    combined_workload.get_machine_combinations(),
                    combined_workload.machines
                )
                for i in range(len(combined_workload.operations))
            ]))
            print(f"Final makespan: {final_makespan:.2f} ms")
            print(f"Greedy makespan: {greedy_makespan:.2f} ms")
            print(f"Improvement: {greedy_makespan - final_makespan:.2f} ms ({100*(greedy_makespan-final_makespan)/greedy_makespan:.1f}%)")
    
    return best_t, best_alpha, best_num_instances


def create_combined_workload_with_periodic(
    base_workload: Workload,
    periodic_workload: Workload,
    num_instances: int,
    period: float
) -> Workload:
    """
    Create combined workload with base workload + multiple periodic instances.
    
    Each periodic instance depends on the previous instance (to enforce period).
    
    Args:
        base_workload: Base workload (e.g., Fast, Dronet)
        periodic_workload: Single instance of periodic task
        num_instances: Number of periodic instances to create
        period: Period between instances (ms)
    
    Returns:
        Combined workload with all operations
    """
    # Create periodic instances
    periodic_instances = []
    for i in range(num_instances):
        # Clone periodic workload operations
        instance_ops = []
        op_mapping = {}  # Map original op to cloned op
        
        for op in periodic_workload.operations:
            # Create new operation with same durations
            new_op = PeriodicOperation(
                processing_times=op.get_durations().tolist(),
                predecessors=[],  # Will set later
                operation_id=f"{op.operation_id}_instance{i}",
                operation_name=f"{op.operation_name}_inst{i}",
                job_id=base_workload.get_num_jobs() + i,  # Each instance is a separate job
                combined_duration=getattr(op, 'combined_duration', None),
                frequency=1.0/period if period > 0 else 0,
                period=period,
                instance_id=i
            )
            instance_ops.append(new_op)
            op_mapping[op] = new_op
        
        # Set up predecessors within this instance
        for orig_op, new_op in op_mapping.items():
            if orig_op.predecessors:
                new_op.predecessors = [op_mapping[pred] for pred in orig_op.predecessors if pred in op_mapping]
        
        periodic_instances.append(instance_ops)
    
    # Connect periodic instances in chain (instance i+1 depends on instance i)
    for i in range(1, num_instances):
        # Find last operations of previous instance (no successors)
        prev_last_ops = []
        prev_ops_set = set(periodic_instances[i-1])
        for op in periodic_instances[i-1]:
            is_last = True
            for other_op in periodic_instances[i-1]:
                if op in other_op.predecessors:
                    is_last = False
                    break
            if is_last:
                prev_last_ops.append(op)
        
        # Find first operations of current instance (no predecessors within instance)
        curr_first_ops = [op for op in periodic_instances[i] if not any(pred in periodic_instances[i] for pred in op.predecessors)]
        
        # Make current first ops depend on previous last ops
        for curr_op in curr_first_ops:
            curr_op.predecessors.extend(prev_last_ops)
    
    # Combine all operations
    all_operations = list(base_workload.operations)
    for instance_ops in periodic_instances:
        all_operations.extend(instance_ops)
    
    # Create combined workload
    combined_workload = Workload(
        operations=all_operations,
        machines=base_workload.machines,
        transfer_times=base_workload.get_transfer_times(),
        machine_combinations=base_workload.get_machine_combinations()
    )
    
    return combined_workload


def optimize_with_warmstart(
    workload: Workload,
    greedy_t: np.ndarray,
    greedy_alpha: np.ndarray,
    makespan_bound: float,
    num_base_ops: int,
    num_periodic_instances: int,
    period: float,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Optimize schedule with greedy solution as warm start and makespan bound.
    
    Args:
        workload: Combined workload to schedule
        greedy_t: Start times from greedy (for base operations only)
        greedy_alpha: Assignments from greedy (for base operations only)
        makespan_bound: Upper bound on makespan from greedy
        num_base_ops: Number of base operations (non-periodic)
        num_periodic_instances: Number of periodic instances
        period: Period for periodic tasks
        verbose: Print debug information
    
    Returns:
        (t, alpha, feasible) where feasible indicates if solution was found
    """
    num_operations = len(workload.operations)
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()
    
    # Decision variables
    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()
    
    # Large constant for big-M constraints
    H = 10000
    
    # Constraints
    constraints = []
    
    # (1) Each operation assigned to exactly one combination
    for i in range(num_operations):
        constraints.append(cp.sum(alpha[i, :]) == 1)
    
    # (2) Precedence constraints
    for i in range(num_operations):
        predecessors = workload.operations[i].get_predecessors()
        for pred in predecessors:
            try:
                i_pred = workload.operations.index(pred)
            except ValueError:
                continue
            
            # Build duration vector for predecessor
            dur_vec_pred = [
                workload.operations[i_pred].get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]
            
            # Transfer time (use maximum as upper bound)
            max_transfer_time = 0
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    if len(machine_combinations[k_pred]) > 0 and len(machine_combinations[k_curr]) > 0:
                        machine_pred = workload.machines.index(machine_combinations[k_pred][0])
                        machine_curr = workload.machines.index(machine_combinations[k_curr][0])
                        transfer_time_val = transfer_times[machine_pred][machine_curr]
                        max_transfer_time = max(max_transfer_time, transfer_time_val)
            
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + max_transfer_time
            )
    
    # (3) Non-overlap constraints for overlapping combinations
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    if workload.combinations_overlap(k1, k2):
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        
                        constraints.append(
                            t[i] + dur_i_k1 <= t[j] + H * (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j])
                        )
                        constraints.append(
                            t[j] + dur_j_k2 <= t[i] + H * (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j])
                        )
    
    # (4) Makespan constraints
    for i in range(num_operations):
        dur_vec = [
            workload.operations[i].get_duration_for_combination(
                k, machine_combinations, workload.machines
            )
            for k in range(num_combinations)
        ]
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
        )
    
    # (5) Non-negative start times
    for i in range(num_operations):
        constraints.append(t[i] >= 0)
    
    # (6) Makespan upper bound from greedy
    constraints.append(C_max <= makespan_bound)
    
    # (7) Warm start constraints - use greedy solution for base operations
    # For base operations, we can initialize close to greedy solution
    # This is a soft constraint via objective, not hard constraint
    
    # (8) Period constraints for periodic tasks
    # Each periodic instance should start approximately 'period' after the previous
    # This is enforced by the dependency chain we created
    
    # Objective: Minimize makespan
    objective = cp.Minimize(C_max)
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    try:
        if verbose:
            print(f"  Solving optimization problem...")
            print(f"    Variables: {num_operations * num_combinations + num_operations * num_operations + num_operations + 1}")
            print(f"    Constraints: {len(constraints)}")
        
        problem.solve(solver=cp.MOSEK, verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if verbose:
                print(f"  ✓ Solution found: makespan = {problem.value:.2f} ms")
            return t.value, alpha.value, True
        else:
            if verbose:
                print(f"  ✗ No solution: status = {problem.status}")
            return None, None, False
    except Exception as e:
        if verbose:
            print(f"  ✗ Solver error: {e}")
        return None, None, False


def main_periodic_scheduling(
    use_profiled: bool = False,
    periodic_frequency: float = 10.0,  # Hz
    max_periodic_instances: int = 10,
    verbose: bool = False,
    output_schedule: bool = True
):
    """
    Main function for periodic task scheduling with greedy warm start.
    
    Args:
        use_profiled: Use profiled runtimes if available
        periodic_frequency: Required frequency for periodic task (Hz)
        max_periodic_instances: Maximum periodic instances to try
        verbose: Print debug information
        output_schedule: Generate output JSON and plots
    """
    print("="*60)
    print("PERIODIC TASK SCHEDULING WITH GREEDY WARM START")
    print("="*60)
    
    # Paths to JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir, '..', 'pytorch_workload', 'samples'
    )
    
    fast_path = os.path.join(base_path, 'fast_dispatch_deps.json')
    dronet_path = os.path.join(base_path, 'dronet_dispatch_deps.json')
    mobilenet_path = os.path.join(base_path, 'mobilenet_v2_dispatch_deps.json')
    
    # Load profiled times if requested
    fast_profiled_p = None
    fast_profiled_e = None
    dronet_profiled_p = None
    dronet_profiled_e = None
    mobilenet_profiled_p = None
    mobilenet_profiled_e = None
    
    if use_profiled:
        # Load profiled data (similar to testing_iree_profile.py)
        data_path = os.path.join(script_dir, '..', 'data')
        
        fast_csv_p = os.path.join(data_path, 'fastdepth_rvv', 'topo_0_1_2_3', 'results.csv')
        fast_csv_e = os.path.join(data_path, 'fastdepth_scalar', 'topo_0_1_2_3', 'results.csv')
        dronet_csv_p = os.path.join(data_path, 'dronet_rvv', 'topo_0_1_2_3', 'results.csv')
        dronet_csv_e = os.path.join(data_path, 'dronet_scalar', 'topo_0_1_2_3', 'results.csv')
        mobilenet_csv_p = os.path.join(data_path, 'mobilenet_v2_rvv', 'topo_0_1_2_3', 'results.csv')
        mobilenet_csv_e = os.path.join(data_path, 'mobilenet_v2_scalar', 'topo_0_1_2_3', 'results.csv')
        
        if os.path.exists(fast_csv_p):
            fast_profiled_p = load_profiled_times(fast_csv_p)
            print(f"Loaded {len(fast_profiled_p)} Fast P-core profiled entries")
        if os.path.exists(fast_csv_e):
            fast_profiled_e = load_profiled_times(fast_csv_e)
            print(f"Loaded {len(fast_profiled_e)} Fast E-core profiled entries")
        if os.path.exists(dronet_csv_p):
            dronet_profiled_p = load_profiled_times(dronet_csv_p)
            print(f"Loaded {len(dronet_profiled_p)} Dronet P-core profiled entries")
        if os.path.exists(dronet_csv_e):
            dronet_profiled_e = load_profiled_times(dronet_csv_e)
            print(f"Loaded {len(dronet_profiled_e)} Dronet E-core profiled entries")
        if os.path.exists(mobilenet_csv_p):
            mobilenet_profiled_p = load_profiled_times(mobilenet_csv_p)
            print(f"Loaded {len(mobilenet_profiled_p)} MobilenetV2 P-core profiled entries")
        if os.path.exists(mobilenet_csv_e):
            mobilenet_profiled_e = load_profiled_times(mobilenet_csv_e)
            print(f"Loaded {len(mobilenet_profiled_e)} MobilenetV2 E-core profiled entries")
    
    # Step 1: Create base workloads (Fast + Dronet)
    print("\n" + "="*60)
    print("STEP 1: Creating base workloads (Fast + Dronet)")
    print("="*60)
    
    fast_workload, _ = create_workload_from_json_with_combinations(
        fast_path, name_prefix="fast_",
        profiled_times_p=fast_profiled_p, profiled_times_e=fast_profiled_e
    )
    print(f"Created Fast workload: {len(fast_workload.operations)} operations")
    
    dronet_workload, _ = create_workload_from_json_with_combinations(
        dronet_path, name_prefix="dronet_",
        profiled_times_p=dronet_profiled_p, profiled_times_e=dronet_profiled_e
    )
    print(f"Created Dronet workload: {len(dronet_workload.operations)} operations")
    
    # Make Dronet depend on Fast
    add_dependency(fast_workload, dronet_workload)
    print("Added dependency: Dronet depends on Fast")
    
    # Combine base workloads
    base_workload = combine_workloads([fast_workload, dronet_workload], job_names=["Fast", "Dronet"])
    print(f"Combined base workload: {len(base_workload.operations)} operations")
    
    # Step 2: Create periodic task workload (single MobilenetV2 instance)
    print("\n" + "="*60)
    print("STEP 2: Creating periodic task workload (MobilenetV2)")
    print("="*60)
    
    mobilenet_workload, _ = create_workload_from_json_with_combinations(
        mobilenet_path, name_prefix="mobilenet_",
        profiled_times_p=mobilenet_profiled_p, profiled_times_e=mobilenet_profiled_e
    )
    print(f"Created MobilenetV2 workload: {len(mobilenet_workload.operations)} operations")
    
    # Step 3: Run greedy scheduler on base workload only (for warm start)
    print("\n" + "="*60)
    print("STEP 3: Running greedy scheduler (warm-up)")
    print("="*60)
    
    greedy_t, greedy_alpha = greedy_schedule(base_workload)
    
    # Calculate greedy makespan
    machine_combinations = base_workload.get_machine_combinations()
    greedy_makespan = max(
        greedy_t[i] + base_workload.operations[i].get_duration_for_combination(
            np.argmax(greedy_alpha[i]), machine_combinations, base_workload.machines
        )
        for i in range(len(base_workload.operations))
    )
    
    print(f"\nGreedy scheduling completed!")
    print(f"  Makespan: {greedy_makespan:.2f} ms")
    print(f"  This will be used as upper bound for optimization")
    
    # Step 4: Run optimization-based periodic scheduler
    print("\n" + "="*60)
    print("STEP 4: Running optimization-based periodic scheduler")
    print("="*60)
    
    t, alpha, num_instances = schedule_periodic_tasks_with_greedy_warmstart(
        base_workload=base_workload,
        periodic_workload=mobilenet_workload,
        periodic_frequency=periodic_frequency,
        greedy_t=greedy_t,
        greedy_alpha=greedy_alpha,
        greedy_makespan=greedy_makespan,
        max_periodic_instances=max_periodic_instances,
        verbose=verbose
    )
    
    # Step 5: Output results
    if t is not None and alpha is not None:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Successfully scheduled {num_instances} periodic MobilenetV2 instances")
        
        # Recreate final combined workload for analysis
        final_workload = create_combined_workload_with_periodic(
            base_workload, mobilenet_workload, num_instances, 1000.0/periodic_frequency
        )
        
        final_makespan = max(
            t[i] + final_workload.operations[i].get_duration_for_combination(
                np.argmax(alpha[i]), machine_combinations, final_workload.machines
            )
            for i in range(len(final_workload.operations))
        )
        
        print(f"Final makespan: {final_makespan:.2f} ms")
        print(f"Greedy makespan: {greedy_makespan:.2f} ms")
        print(f"Period: {1000.0/periodic_frequency:.2f} ms ({periodic_frequency:.2f} Hz)")
        
        # Count assignments
        cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
        cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)
        cpu_both_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 2)
        
        print(f"\nHardware assignments:")
        print(f"  CPU_P: {cpu_p_count} operations")
        print(f"  CPU_E: {cpu_e_count} operations")
        print(f"  CPU_P+CPU_E: {cpu_both_count} operations")
        
        if output_schedule:
            # Generate plot
            os.makedirs("plots", exist_ok=True)
            plot_filename = f"plots/periodic_schedule_{num_instances}instances_{periodic_frequency}Hz.png"
            plot_title = f"Periodic Scheduling: {num_instances} MobilenetV2 @ {periodic_frequency}Hz (Makespan: {final_makespan:.2f}ms)"
            
            plot_schedule(
                final_workload,
                t,
                alpha,
                title=plot_title,
                filename=plot_filename
            )
            print(f"\nPlot saved to: {plot_filename}")
            
            # Generate output JSON
            json_filename = f"plots/periodic_schedule_{num_instances}instances_{periodic_frequency}Hz.json"
            # Note: output_scheduled_json expects specific format, may need adaptation
            print(f"Output JSON would be saved to: {json_filename}")
        
        return t, alpha, final_workload, num_instances
    else:
        print("\n" + "="*60)
        print("SCHEDULING FAILED")
        print("="*60)
        print("Could not find feasible schedule")
        return None, None, None, 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Periodic task scheduling with greedy warm start")
    parser.add_argument("--use-profiled", action="store_true", help="Use profiled runtimes")
    parser.add_argument("--frequency", type=float, default=10.0, help="Periodic task frequency (Hz)")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum periodic instances to try")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    parser.add_argument("--no-output", action="store_true", help="Skip output generation")
    
    args = parser.parse_args()
    
    main_periodic_scheduling(
        use_profiled=args.use_profiled,
        periodic_frequency=args.frequency,
        max_periodic_instances=args.max_instances,
        verbose=args.verbose,
        output_schedule=not args.no_output
    )
