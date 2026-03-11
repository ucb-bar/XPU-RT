#
# Problem formulation from https://www.sciencedirect.com/science/article/pii/S037722172300382X#sec0014 Section 2.1.
#

import cvxpy as cp
import numpy as np
import time
import os
import sys

# Ensure local modules are imported correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload, Window
from packing import greedy_packing, convex_packing, combine_solved_windows
from typing import Tuple, Optional

# Import from local fusion module (not the system package)
try:
    from .fusion import fuse_operations, expand_schedule, print_fusion_report
except ImportError:
    from fusion import fuse_operations, expand_schedule, print_fusion_report


def schedule_window(window: Window) -> Tuple[np.ndarray, np.ndarray]:
    num_operations = len(window.operations)
    machine_combinations = window.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = window.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters
    H = 5000

    # Constraints
    constraints = []
    # (2) Each operation must be assigned to exactly one machine combination
    for i in range(num_operations):
        constraints.append(cp.sum(alpha[i, :]) == 1)
    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    for i in range(num_operations):
        predecessors = window.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = None
            try:
                i_pred = window.operations.index(pred)
            except ValueError:
                # happens when the predecessor is not in the window
                i_pred = None

            # check if there is a required predecessor in this window
            if i_pred is not None:
                # Build duration vector for predecessor
                dur_vec_pred = [
                    window.operations[i_pred].get_duration_for_combination(
                        k, machine_combinations, window.machines
                    )
                    for k in range(num_combinations)
                ]

                # For transfer time, use maximum as upper bound (DCP-compliant)
                max_transfer_time = 0
                for k_pred in range(num_combinations):
                    for k_curr in range(num_combinations):
                        machine_pred = window.machines.index(
                            machine_combinations[k_pred][0]
                        )
                        machine_curr = window.machines.index(
                            machine_combinations[k_curr][0]
                        )
                        transfer_time_val = transfer_times[machine_pred][machine_curr]
                        max_transfer_time = max(max_transfer_time, transfer_time_val)

                transfer_time_weighted = max_transfer_time

                constraints.append(
                    t[i]
                    >= t[i_pred]
                    + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :]))
                    + transfer_time_weighted
                )
    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    for i in range(num_operations):
        for j in range(i + 1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if window.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        # Get duration for combination k2
                        dur_j_k2 = window.operations[j].get_duration_for_combination(
                            k2, machine_combinations, window.machines
                        )
                        constraints.append(
                            t[i]
                            >= t[j]
                            + dur_j_k2
                            - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        # Get duration for combination k1
                        dur_i_k1 = window.operations[i].get_duration_for_combination(
                            k1, machine_combinations, window.machines
                        )
                        constraints.append(
                            t[j]
                            >= t[i]
                            + dur_i_k1
                            - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    # (6)
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [
            window.operations[i].get_duration_for_combination(
                k, machine_combinations, window.machines
            )
            for k in range(num_combinations)
        ]
        constraints.append(C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])))
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(t[i] >= 0)

    # term to maximize consecutive empty space on each machine combination
    empty_space = cp.Variable(num_combinations)
    for k in range(num_combinations):
        for i in range(num_operations):
            for j in range(i + 1, num_operations):
                # Only consider if both operations could be on this combination (though they can't overlap)
                dur_j_k = window.operations[j].get_duration_for_combination(
                    k, machine_combinations, window.machines
                )
                constraints.append(
                    empty_space[k]
                    >= t[i]
                    - (
                        t[j]
                        + dur_j_k
                        - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H
                    )
                )

    # objective_func = 150*C_max + cp.sum(empty_space)
    objective_func = C_max

    # Optimization problem
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value


def schedule(
    workload: Workload,
    fusion_threshold: Optional[float] = None,
    verbose: bool = False,
    solver_verbosity: int = 0,
    time_limit: Optional[float] = None,
    restrict_makespan_to_nonperiodic: bool = True,
    prune_cross_period_constraints: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Workload], Optional[dict]]:
    """
    Schedule a workload, optionally with operation fusion.

    Args:
        workload: The workload to schedule
        fusion_threshold: If provided, fuse operations with duration <= threshold (in time units).
                        If None, no fusion is performed.
        verbose: If True, print problem statistics and timing information.
        solver_verbosity: MOSEK solver verbosity level (0=silent, >0=enables verbose output).
        time_limit: Maximum optimization time in seconds. If None, no time limit is set.
                   MOSEK will return the best solution found within the time limit.
        restrict_makespan_to_nonperiodic: If True, C_max only tracks non-periodic operations
                   (those without min_start_t / max_end_t). Periodic/background operations
                   still obey all constraints but do not affect the makespan objective.
        prune_cross_period_constraints: If True, skip precedence and non-overlap constraints
                   between operations whose time windows provably do not overlap in time.
                   This reduces redundant constraints for periodic tasks in disjoint periods.

    Returns:
        (t, alpha, fused_workload, fusion_map) where:
        - t: Start times for operations (original operations if fusion was used)
        - alpha: Machine assignments for operations (original operations if fusion was used)
        - fused_workload: The fused workload (None if no fusion)
        - fusion_map: Mapping from fused op index to original op indices (None if no fusion)
    """
    original_workload = workload
    fusion_map = None

    # Apply fusion only if threshold is explicitly provided and positive
    # If fusion_threshold is None (flag not passed) or <= 0, skip fusion
    if fusion_threshold is not None and fusion_threshold > 0:
        workload, fusion_map = fuse_operations(workload, fusion_threshold)
        print(
            f"Fusion applied: {len(original_workload.operations)} operations -> {len(workload.operations)} fused operations"
        )
        # Print detailed fusion report for debugging (to file)
        import os

        os.makedirs("fusion_reports", exist_ok=True)
        report_file = f"fusion_reports/fusion_report_{len(original_workload.operations)}to{len(workload.operations)}.txt"
        print_fusion_report(
            original_workload, workload, fusion_map, output_file=report_file
        )
    else:
        # No fusion requested - use original workload as-is
        if verbose:
            print("Fusion skipped: fusion_threshold not provided or <= 0")

    num_operations = len(workload.get_operations())
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters
    H = 5000

    # Constraints
    constraints = []
    # (2) Each operation must be assigned to exactly one machine combination
    for i in range(num_operations):
        constraints.append(cp.sum(alpha[i, :]) == 1)

    def _periods_overlap(op_a, op_b) -> bool:
        """Return True if the time windows of two operations can overlap."""
        a_start = getattr(op_a, "min_start_t", None)
        a_end = getattr(op_a, "max_end_t", None)
        b_start = getattr(op_b, "min_start_t", None)
        b_end = getattr(op_b, "max_end_t", None)
        # If any bound is missing, conservatively assume they may overlap
        if a_start is None or a_end is None or b_start is None or b_end is None:
            return True
        # Intervals [a_start, a_end) and [b_start, b_end) overlap iff both:
        # a_start < b_end and b_start < a_end
        return (a_start < b_end) and (b_start < a_end)

    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    for i in range(num_operations):
        op_i = workload.operations[i]
        predecessors = op_i.get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Optionally prune precedence constraints between non-overlapping periods
            if prune_cross_period_constraints:
                pred_start = getattr(pred, "min_start_t", None)
                pred_end = getattr(pred, "max_end_t", None)
                succ_start = getattr(op_i, "min_start_t", None)
                succ_end = getattr(op_i, "max_end_t", None)
                if (
                    pred_start is not None
                    and pred_end is not None
                    and succ_start is not None
                    and succ_end is not None
                ):
                    # If predecessor's window ends before successor's window starts,
                    # precedence is automatically satisfied by time-window constraints.
                    if pred_end <= succ_start:
                        continue
                    # If successor's window ends before predecessor's window starts,
                    # the precedence is impossible under the windows; raise early.
                    if succ_end <= pred_start:
                        raise ValueError(
                            f"Infeasible precedence: successor window [{succ_start}, {succ_end}) "
                            f"before predecessor window [{pred_start}, {pred_end})."
                        )

            # Build duration vector for predecessor
            dur_vec_pred = [
                workload.operations[i_pred].get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]

            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            max_transfer_time = 0

            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    machine_1 = workload.combination_to_machine(
                        machine_combinations[k_pred]
                    )
                    machine_2 = workload.combination_to_machine(
                        machine_combinations[k_curr]
                    )
                    machine_pred = list(workload.machines.keys()).index(machine_1)
                    machine_curr = list(workload.machines.keys()).index(machine_2)
                    transfer_time_val = transfer_times[machine_pred][machine_curr]
                    max_transfer_time = max(max_transfer_time, transfer_time_val)

            transfer_time_weighted = max_transfer_time

            constraints.append(
                t[i]
                >= t[i_pred]
                + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :]))
                + transfer_time_weighted
            )
    # Time window constraints: operations must respect min_start_t and max_end_t if specified
    for i in range(num_operations):
        op = workload.operations[i]
        # Constraint: operation must start after min_start_t (if specified)
        if op.min_start_t is not None:
            constraints.append(t[i] >= op.min_start_t)
        # Constraint: operation must end before max_end_t (if specified)
        if op.max_end_t is not None:
            # Build duration vector for all combinations
            dur_vec = [
                op.get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]
            # Operation completion time = start_time + duration_for_chosen_combination
            # Must be <= max_end_t
            constraints.append(
                t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])) <= op.max_end_t
            )
            # constraints.append(
            #     t[i] <= op.max_end_t
            # )

    first_assignment = {}
    for o1 in range(num_operations):
        for o2 in range(o1 + 1, num_operations):
            first_assignment[(o1, o2)] = {}
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    if workload.combinations_overlap(k1, k2):
                        first_assignment[(o1, o2)][(k1, k2)] = cp.Variable(boolean=True)

                        dur_o1_k1 = workload.operations[
                            o1
                        ].get_duration_for_combination(
                            k1, machine_combinations, workload.machines
                        )
                        dur_o2_k2 = workload.operations[
                            o2
                        ].get_duration_for_combination(
                            k2, machine_combinations, workload.machines
                        )

                        constraints.append(
                            t[o1]
                            >= t[o2]
                            + dur_o2_k2
                            - (1 - first_assignment[(o1, o2)][(k1, k2)]) * H
                            - (2 - alpha[o1, k1] - alpha[o2, k2]) * H
                        )
                        constraints.append(
                            t[o2]
                            >= t[o1]
                            + dur_o1_k1
                            - first_assignment[(o1, o2)][(k1, k2)] * H
                            - (2 - alpha[o1, k1] - alpha[o2, k2]) * H
                        )

    # (6) Makespan constraints:
    if restrict_makespan_to_nonperiodic:
        # C_max tracks only NON-periodic operations (operations without explicit time-window bounds).
        # Periodic/background operations (with min_start_t or max_end_t set) do NOT constrain C_max.
        non_periodic_ops_exist = False
        for i in range(num_operations):
            op = workload.operations[i]
            # Treat an operation as periodic/background if it has any time-window bound
            is_periodic = (
                getattr(op, "min_start_t", None) is not None
                or getattr(op, "max_end_t", None) is not None
            )
            if is_periodic:
                continue
            non_periodic_ops_exist = True
            # Build duration vector for all combinations
            dur_vec = [
                workload.operations[i].get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]
            constraints.append(
                C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
            )
        # If there are no non-periodic operations, C_max is unconstrained from below
        # (objective will be trivial), which is acceptable: only periodic tasks exist.
    else:
        # Original behavior: C_max covers all operations (including periodic ones)
        for i in range(num_operations):
            dur_vec = [
                workload.operations[i].get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]
            constraints.append(
                C_max
                >= t[i]
                + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
                + cp.sum(list_overflow) * H
            )

    # Debug: Print durations for first operation to verify they're correct
    if num_operations > 0:
        print(f"\nDEBUG DURATIONS for first operation:")
        for k in range(num_combinations):
            combo = machine_combinations[k]
            combo_str = "+".join(combo) if len(combo) > 1 else combo[0]
            dur = workload.operations[0].get_duration_for_combination(
                k, machine_combinations, workload.machines
            )
            print(f"  Combination {k} ({combo_str}): {dur:.3f} ms")

    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(t[i] >= 0)

    # Optimization problem
    objective = cp.Minimize(C_max)
    problem = cp.Problem(objective, constraints)

    # Print problem statistics
    if verbose:
        print(f"\n{'='*60}")
        print("OPTIMIZATION PROBLEM STATISTICS")
        print(f"{'='*60}")
        print(f"Number of operations: {num_operations}")
        print(f"Number of machine combinations: {num_combinations}")
        num_vars = (
            num_operations * num_combinations
            + num_operations * num_operations
            + num_operations
            + 1
        )
        print(f"Number of variables: {num_vars}")
        print(
            f"  - alpha (operation->combination): {num_operations * num_combinations}"
        )
        print(f"  - beta (operation ordering): {num_operations * num_operations}")
        print(f"  - t (start times): {num_operations}")
        print(f"  - C_max (makespan): 1")
        print(f"Number of constraints: {len(constraints)}")
        print(f"{'='*60}\n")
        print("Starting optimization...")
        start_time = time.time()

    # Configure MOSEK solver parameters
    # CVXPY's verbose parameter controls MOSEK output
    # For additional MOSEK-specific parameters, we can pass them through mosek_params
    # Note: CVXPY's verbose=True already enables MOSEK logging
    mosek_verbose = verbose or (solver_verbosity > 0)
    mosek_params = {}

    # Set time limit if specified (in seconds)
    if time_limit is not None and time_limit > 0:
        mosek_params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = time_limit
        if verbose:
            print(
                f"Time limit set to {time_limit:.1f} seconds ({time_limit/60:.1f} minutes)"
            )

    if mosek_params:
        problem.solve(solver=cp.MOSEK, verbose=mosek_verbose, mosek_params=mosek_params)
    else:
        problem.solve(solver=cp.MOSEK, verbose=mosek_verbose)

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"{'='*60}")

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)

    t_result = t.value
    alpha_result = alpha.value

    # Check if optimization was successful
    if t_result is None or alpha_result is None:
        print(
            "Warning: Optimization failed (infeasible or error). Cannot expand schedule."
        )
        return None, None, workload if fusion_threshold else None, fusion_map

    # Check problem status - warn if not optimal
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: Problem status is '{problem.status}' (not optimal).")
        if problem.status == cp.SOLVER_ERROR:
            print("  Solver encountered an error. Solution may be invalid.")
        elif problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            print("  Problem is infeasible. Solution is invalid.")
            return None, None, workload if fusion_threshold else None, fusion_map
        elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
            print("  Problem is unbounded. Solution may be invalid.")
        else:
            print("  Solution may not be optimal but should be feasible.")

    # Validate solution dimensions
    if len(t_result) != num_operations or alpha_result.shape[0] != num_operations:
        print(
            f"Error: Solution dimensions don't match. Expected {num_operations} operations, got {len(t_result)} start times and {alpha_result.shape[0]} assignments."
        )
        return None, None, workload if fusion_threshold else None, fusion_map

    # Expand schedule back to original operations if fusion was used
    if fusion_threshold is not None and fusion_threshold > 0 and fusion_map is not None:
        t_result, alpha_result = expand_schedule(
            workload, fusion_map, original_workload, t_result, alpha_result
        )
        print(
            f"Schedule expanded: {len(workload.operations)} fused operations -> {len(original_workload.operations)} original operations"
        )

        # Validate expansion
        if len(t_result) != len(original_workload.operations):
            print(
                f"ERROR: Expanded schedule has {len(t_result)} operations but original workload has {len(original_workload.operations)} operations!"
            )
        else:
            print(
                f"Validation: All {len(original_workload.operations)} original operations have been scheduled."
            )

        return t_result, alpha_result, workload, fusion_map

    # Validate non-fused schedule
    if len(t_result) != len(workload.operations):
        print(
            f"ERROR: Schedule has {len(t_result)} operations but workload has {len(workload.operations)} operations!"
        )
    else:
        print(
            f"Validation: All {len(workload.operations)} operations have been scheduled."
        )

    return t_result, alpha_result, None, None


def schedule_additional_objectives(
    workload: Workload, nominal_start_times: list[float], gap_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    @param nominal_start_times: list of nominal start times for each operation. If there is no desired start time, set index to -1
    @param gap_bound: the maximum maximum allowable gap between operations to bound the optimization problem
    """
    num_operations = len(workload.get_operations())
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # desired frequency
    z = cp.Variable(num_operations)

    # interrupt tolerance
    G_max = cp.Variable()  # TODO have a G_max for each machine
    g = cp.Variable(num_operations)

    # Hyperparameters
    H = 5000

    # Constraints
    constraints = []
    # (2) Each operation must be assigned to exactly one machine combination
    for i in range(num_operations):
        constraints.append(cp.sum(alpha[i, :]) == 1)
    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    for i in range(num_operations):
        predecessors = workload.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Build duration vector for predecessor
            dur_vec_pred = [
                workload.operations[i_pred].get_duration_for_combination(
                    k, machine_combinations, workload.machines
                )
                for k in range(num_combinations)
            ]

            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            max_transfer_time = 0
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    machine_pred = workload.machines.index(
                        machine_combinations[k_pred][0]
                    )
                    machine_curr = workload.machines.index(
                        machine_combinations[k_curr][0]
                    )
                    transfer_time_val = transfer_times[machine_pred][machine_curr]
                    max_transfer_time = max(max_transfer_time, transfer_time_val)

            # For transfer time, we use the maximum transfer time as an upper bound
            # This is conservative but ensures correctness and is DCP-compliant
            # The actual transfer time will be <= max_transfer_time, which is safe for scheduling
            # For backward compatibility (singleton combinations), this still works correctly
            # since the maximum is just the max over all machine pairs
            transfer_time_weighted = max_transfer_time

            constraints.append(
                t[i]
                >= t[i_pred]
                + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :]))
                + transfer_time_weighted
            )
    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    for i in range(num_operations):
        for j in range(i + 1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(
                            k2, machine_combinations, workload.machines
                        )
                        constraints.append(
                            t[i]
                            >= t[j]
                            + dur_j_k2
                            - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(
                            k1, machine_combinations, workload.machines
                        )
                        constraints.append(
                            t[j]
                            >= t[i]
                            + dur_i_k1
                            - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    # (6)
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [
            workload.operations[i].get_duration_for_combination(
                k, machine_combinations, workload.machines
            )
            for k in range(num_combinations)
        ]
        constraints.append(C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])))
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(t[i] >= 0)

    # desired frequency
    for i in range(num_operations):
        if nominal_start_times[i] >= 0:
            constraints.append(z[i] >= t[i] - nominal_start_times[i])
            constraints.append(z[i] >= -(t[i] - nominal_start_times[i]))
        constraints.append(z[i] >= 0)

    # interrupt tolerance
    for i in range(num_operations):
        for j in range(i + 1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(
                            k2, machine_combinations, workload.machines
                        )
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(
                            k1, machine_combinations, workload.machines
                        )
                        constraints.append(
                            g[i]
                            >= (t[i] - t[j] - dur_j_k2)
                            - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        constraints.append(
                            g[j]
                            >= (t[j] - t[i] - dur_i_k1)
                            - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )

    for i in range(num_operations):
        constraints.append(G_max <= g[i])
        constraints.append(g[i] >= 0)
        constraints.append(g[i] <= gap_bound)

    # Optimization problem
    objective_func = C_max + cp.sum(z) - 0.1 * G_max
    # objective_func = C_max + cp.sum(z)
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value


def schedule_with_greedy_packing(
    workload: Workload, n_splits: int
) -> Tuple[np.ndarray, np.ndarray]:
    windows = greedy_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha


def schedule_with_convex_packing(workload: Workload, n_splits: int) -> Tuple[int, int]:
    windows = convex_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha
