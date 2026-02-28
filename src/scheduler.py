#
# Problem formulation from https://www.sciencedirect.com/science/article/pii/S037722172300382X#sec0014 Section 2.1.
#

import cvxpy as cp
import numpy as np
from workload import Workload, Window
from packing import greedy_packing, convex_packing, combine_solved_windows
from typing import Tuple

def schedule_window(window: Window) -> Tuple[np.ndarray, np.ndarray]:
    num_operations = len(window.operations)
    machine_combinations = window.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = window.get_transfer_times()
    machines = window.machines  # Get machine concurrency limits

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
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
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
                dur_vec_pred = [window.operations[i_pred].get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
                
                # For transfer time, use maximum as upper bound (DCP-compliant)
                max_transfer_time = 0
                for k_pred in range(num_combinations):
                    for k_curr in range(num_combinations):
                        # Get machine names and find their indices
                        machine_pred_name = machine_combinations[k_pred][0]
                        machine_curr_name = machine_combinations[k_curr][0]
                        machine_names = list(window.machines.keys())
                        machine_pred_idx = machine_names.index(machine_pred_name) if machine_pred_name in machine_names else 0
                        machine_curr_idx = machine_names.index(machine_curr_name) if machine_curr_name in machine_names else 0
                        transfer_time_val = transfer_times[machine_pred_idx][machine_curr_idx]
                        max_transfer_time = max(max_transfer_time, transfer_time_val)
                
                transfer_time_weighted = max_transfer_time
                
                constraints.append(
                    t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
                )
    
    # (4) and (5) Overlap constraints with machine concurrency limits
    # Only enforce strict ordering when concurrency limit is 1
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if window.combinations_overlap(k1, k2):
                        # Get the machines in both combinations
                        shared_machines = set(machine_combinations[k1]) & set(machine_combinations[k2])
                        
                        # Check if ANY shared machine has concurrency limit of 1
                        must_not_overlap = any(machines.get(m, 1) == 1 for m in shared_machines)
                        
                        if must_not_overlap:
                            # Machine can only run 1 operation: enforce strict ordering
                            # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                            dur_j_k2 = window.operations[j].get_duration_for_combination(k2, machine_combinations, window.machines)
                            constraints.append(
                                t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                            )
                            # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                            dur_i_k1 = window.operations[i].get_duration_for_combination(k1, machine_combinations, window.machines)
                            constraints.append(
                                t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                            )
                        # If concurrency > 1, allow overlaps by not adding strict ordering constraints   
    # (6)
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [window.operations[i].get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
        )
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )

    # term to maximize consecutive empty space on each machine combination
    empty_space = cp.Variable(num_combinations)
    for k in range(num_combinations):
        for i in range(num_operations):
            for j in range(i+1, num_operations):
                # Only consider if both operations could be on this combination (though they can't overlap)
                dur_j_k = window.operations[j].get_duration_for_combination(k, machine_combinations, window.machines)
                constraints.append(
                    empty_space[k] >= t[i] - (t[j] + dur_j_k - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H)
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

def schedule(workload: Workload) -> Tuple[np.ndarray, np.ndarray]:
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
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    for i in range(num_operations):
        predecessors = workload.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Build duration vector for predecessor
            dur_vec_pred = [workload.operations[i_pred].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
            
            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            max_transfer_time = 0
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    machine_name_to_idx = {name: idx for idx, name in enumerate(workload.machines)}
                    machine_pred = machine_name_to_idx[machine_combinations[k_pred][0]]
                    machine_curr = machine_name_to_idx[machine_combinations[k_curr  ][0]]
                    transfer_time_val = transfer_times[machine_pred][machine_curr]
                    max_transfer_time = max(max_transfer_time, transfer_time_val)
            
            # For transfer time, we use the maximum transfer time as an upper bound
            # This is conservative but ensures correctness and is DCP-compliant
            # The actual transfer time will be <= max_transfer_time, which is safe for scheduling
            # For backward compatibility (singleton combinations), this still works correctly
            # since the maximum is just the max over all machine pairs
            transfer_time_weighted = max_transfer_time
            
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
            )
    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        constraints.append(
                            t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        constraints.append(
                            t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    # (6)
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
        )
    
    # Debug: Print durations for first operation to verify they're correct
    if num_operations > 0:
        print(f"\nDEBUG DURATIONS for first operation:")
        for k in range(num_combinations):
            combo = machine_combinations[k]
            combo_str = "+".join(combo) if len(combo) > 1 else combo[0]
            dur = workload.operations[0].get_duration_for_combination(k, machine_combinations, workload.machines)
            print(f"  Combination {k} ({combo_str}): {dur:.3f} ms")
    
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )

    # Optimization problem
    objective = cp.Minimize(C_max)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value

def schedule_additional_objectives(workload: Workload, nominal_start_times: list[float], gap_bound: float) -> Tuple[np.ndarray, np.ndarray]:
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
    G_max = cp.Variable() # TODO have a G_max for each machine
    g = cp.Variable(num_operations)

    # Hyperparameters
    H = 5000

    # Constraints
    constraints = []
    # (2) Each operation must be assigned to exactly one machine combination
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )

    max_transfer_time = 0
    for k_pred in range(num_combinations):
        for k_curr in range(num_combinations):
            machine_pred = workload.machines.index(machine_combinations[k_pred][0])
            machine_curr = workload.machines.index(machine_combinations[k_curr][0])
            transfer_time_val = transfer_times[machine_pred][machine_curr]
            max_transfer_time = max(max_transfer_time, transfer_time_val)
    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    for i in range(num_operations):
        predecessors = workload.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Build duration vector for predecessor
            dur_vec_pred = [workload.operations[i_pred].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
            
            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            #TODO: Ailsa: Why is this inside the loop lol? Shouldn't this be moved out. 
            
            # For transfer time, we use the maximum transfer time as an upper bound
            # This is conservative but ensures correctness and is DCP-compliant
            # The actual transfer time will be <= max_transfer_time, which is safe for scheduling
            # For backward compatibility (singleton combinations), this still works correctly
            # since the maximum is just the max over all machine pairs
            transfer_time_weighted = max_transfer_time
            
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
            )
    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        constraints.append(
                            t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        constraints.append(
                            t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    # (6)
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
        )
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )

    # desired frequency
    for i in range(num_operations):
        if nominal_start_times[i] >= 0:
            constraints.append(
                z[i] >= t[i] - nominal_start_times[i]
            )
            constraints.append(
                z[i] >= -(t[i] - nominal_start_times[i])
            )
        constraints.append(
            z[i] >= 0
        )

    # interrupt tolerance
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        constraints.append(
                            g[i] >= (t[i] - t[j] - dur_j_k2) - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        constraints.append(
                            g[j] >= (t[j] - t[i] - dur_i_k1) - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )

    for i in range(num_operations):
        constraints.append(
            G_max <= g[i]
        )
        constraints.append(
            g[i] >= 0
        )
        constraints.append(
            g[i] <= gap_bound
        )

    # Optimization problem
    objective_func = C_max + cp.sum(z) - 0.1*G_max
    # objective_func = C_max + cp.sum(z)
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value

def schedule_with_greedy_packing(workload: Workload, n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    windows = greedy_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha

def schedule_with_convex_packing(workload: Workload, n_splits: int) -> Tuple[int, int]:
    windows = convex_packing(workload,  n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha