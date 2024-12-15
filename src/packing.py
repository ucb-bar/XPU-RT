import cvxpy as cp
import numpy as np

from workload import Workload, Window
from schedule_validation import overlap_fixer, count_overlaps
from typing import List, Tuple

def lp_schedule(workload: Workload) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the MILP scheduling problem with relaxed integer constraints. The resultant program is an LP.
    """
    num_operations = len(workload.get_operations())
    num_machines = len(workload.machines)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_machines))
    beta = cp.Variable((num_operations, num_operations))
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters
    H = 5000

    # Constraints
    constraints = []
    # (2)
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    # (3)
    for i in range(num_operations):
        pred = workload.operations[i].get_predecessor()
        i_pred = None if pred is None else workload.operations.index(pred)

        # check if there is a required predecessor
        if i_pred is not None:
            machine_pred = np.argmax(alpha[i_pred, :])
            machine_curr = np.argmax(alpha[i, :])

            transfer_time = transfer_times[machine_pred][machine_curr]

            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(workload.operations[i_pred].get_durations()[:], alpha[i_pred, :])) + transfer_time
            )
    # (4)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k in range(num_machines):
                constraints.append(
                    t[i] >= t[j] + workload.operations[j].get_durations()[k] - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H
                )
    # (5)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k in range(num_machines):
                constraints.append(
                    t[j] >= t[i] + workload.operations[i].get_durations()[k] - (3 - alpha[i, k] - alpha[j, k] - beta[i, j]) * H
                )
    # (6)
    for i in range(num_operations):
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(workload.operations[i].get_durations()[:], alpha[i, :]))
        )
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )

    for i in range(num_operations):
        for j in range(num_operations):
            constraints.extend(
                [beta[i, j] >= 0, beta[i, j] <= 1]
            )
    for i in range(num_operations):
        for j in range(num_machines):
            constraints.extend(
                [alpha[i, j] >= 0, alpha[i, j] <= 1]
            )

    # Optimization problem
    objective = cp.Minimize(C_max)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    # Create a boolean mask where the maximum value in each row is True to get the machine assignment
    mask = alpha.value == alpha.value.max(axis=1, keepdims=True)
    alpha = mask.astype(int)
    t = t.value

    overlaps = count_overlaps(workload, t, alpha)

    for _ in range(len(workload.operations)):
        t = overlap_fixer(workload, t, alpha)

    new_overlaps = count_overlaps(workload, t, alpha)

    print(f"Overlaps before: {overlaps}, Overlaps after: {new_overlaps}")
    return t, alpha

def greedy_packing(workload: Workload, n_splits: int) -> list[Window]:
    """
    Greedy packing algorithm that packs operations into n_splits+1 windows.
    """
    estimated_time = sum([np.mean(operation.get_durations()) for operation in workload.get_operations()])
    window_time = estimated_time / (n_splits + 1)

    operations = workload.get_operations().copy()
    windows = []
    for i in range(n_splits+1):
        window_operations = []
        window_duration = 0
        if i == n_splits:
            # append all remaining operations to the last window
            windows.append(Window(window_time, operations, workload.machines, workload.get_transfer_times()))
        else:
            while True:
                operation = operations[0]
                if window_duration + np.mean(operation.get_durations()) <= window_time:
                    operations.pop(0)
                    window_operations.append(operation)
                    window_duration += np.mean(operation.get_durations())
                else:
                    break
            windows.append(Window(window_time, window_operations, workload.machines, workload.get_transfer_times()))

    for i in range(n_splits+1):
        print(f"Window {i}: {len(windows[i].operations)} operations")

    return windows

def convex_packing(workload: Workload, n_splits: int) -> List[Window]:
    """
    approximate the optimal packing of jobs into windows using convex optimization.
    
    1) solves optimization problem without integer constraints
    2) splits the operations into n_splits windows based on the start times

    Returns a list of windows each containing a subset of the operations
    """
    operations = workload.get_operations().copy()
    t, alpha = lp_schedule(workload)

    # sort operations by start time
    times_and_ops = []
    for i in range(len(t)):
        times_and_ops.append([t[i], operations[i]])
    
    times_and_ops.sort(key=lambda time_and_op: time_and_op[0])

    # split operations into n_splits windows
    max_idx = np.argmax(t)
    max_start_time = np.max(t)
    window_time = max_start_time / (n_splits+1)
    window_operations = [[] for _ in range(n_splits+1)]
    for time, operation in times_and_ops:
        window_idx = min(int(time // window_time), n_splits)
        window_operations[window_idx].append(operation)
    
    windows = []
    for operations in window_operations:
        windows.append(Window(window_time, operations, workload.machines, workload.get_transfer_times()))
    
    return windows

def combine_solved_windows(original_workload, windows, solutions):
    """
    takes a list of (t, alpha) and combines them into a single (t, alpha) for the entire workload
    """
    original_operations = original_workload.get_operations()
    t = np.zeros(len(original_operations))
    alpha = np.zeros((len(original_operations), len(original_workload.machines)))
    transfer_times = original_workload.get_transfer_times()

    start_time = 0

    for window, (t_window, alpha_window) in zip(windows, solutions):
        for i, operation in enumerate(window.operations):
            idx = original_operations.index(operation)
            t[idx] = t_window[i] + start_time
            alpha[idx] = alpha_window[i]

        # find the best start time for the next window
        latest_operation_idx = np.argmax(t)
        latest_operation = original_workload.operations[latest_operation_idx]
        duration = latest_operation.get_durations()[np.argmax(alpha[latest_operation_idx])]
        start_time = t[latest_operation_idx] + duration + np.mean(transfer_times)

    for i in range(len(original_operations)):
        pred = original_operations[i].get_predecessor()
        i_pred = None if pred is None else original_operations.index(pred)

        if i_pred is not None:
            machine_pred = np.argmax(alpha[i_pred, :])
            machine_curr = np.argmax(alpha[i, :])
            transfer_time = transfer_times[machine_pred][machine_curr]
            t[i] = max(t[i], t[i_pred] + original_operations[i_pred].get_durations()[np.argmax(alpha[i_pred])] + transfer_time)
    
    # greedily pushes back operations that overlap in time
    for _ in range(len(original_operations)):
        t = overlap_fixer(original_workload, t, alpha)
    
    return t, alpha