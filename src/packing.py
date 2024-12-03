from workload import Workload, Job, Operation, Window
import cvxpy as cp
import numpy as np
from typing import Tuple

def convex_schedule(workload: Workload, transfer_times):
    num_operations = len(workload.get_operations())
    num_machines = len(workload.machines)

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

    # Create a boolean mask where the maximum value in each row is True
    mask = alpha.value == alpha.value.max(axis=1, keepdims=True)

    # Convert the boolean mask to integers (True becomes 1, False becomes 0)
    alpha = mask.astype(int)

    t = t.value

    # group start times by machine and operation
    times_and_ops = []
    for i in range(len(t)):
        times_and_ops.append([t[i], workload.operations[i]])
    original_order = {op: idx for idx, op in enumerate(workload.operations)}

    start_times = []
    for i in range(num_machines):
        start_times.append([times_and_ops[j] for j in range(num_operations) if alpha[j, i] == 1])
    
    for i in range(num_machines):
        start_times[i].sort(key=lambda time_and_op: time_and_op[0])

    # make sure start times are after the previous operation's end time
    for i in range(num_machines):
        for j in range(1, len(start_times[i])):
            start_times[i][j][0] = max(start_times[i][j][0], start_times[i][j-1][0] + start_times[i][j-1][1].get_durations()[i])
    
    # flatten back to times_and_ops and recover original ordering
    times_and_ops = [time_and_op for machine in start_times for time_and_op in machine]
    times_and_ops.sort(key=lambda time_and_op: original_order[time_and_op[1]])
    t = [time_and_op[0] for time_and_op in times_and_ops]
    t = np.array(t)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t, alpha

def greedy_packing(workload: Workload, n_splits: int) -> list[Window]:
    """
    Greedy packing algorithm that packs jobs into windows.
    """
    total_duration = sum([np.mean(operation.get_durations()) for operation in workload.get_operations()])
    estimated_time = total_duration
    window_time = estimated_time / (n_splits - 1)

    operations = workload.get_operations().copy()
    windows = []
    for i in range(n_splits - 1):
        window_operations = []
        window_duration = 0
        if i == n_splits - 1:
            windows.append(Window(window_time, operations, workload.machines))
        else:
            while True:
                operation = operations[0]
                if window_duration + np.mean(operation.get_durations()) <= window_time:
                    operations.pop(0)
                    window_operations.append(operation)
                    window_duration += np.mean(operation.get_durations())
                else:
                    break

        windows.append(Window(window_time, window_operations, workload.machines))

    return windows

def convex_packing(workload: Workload, n_splits: int) -> list[Window]:
    """
    approximate the optimal packing of jobs into windows using convex optimization.
    
    1) solves optimization problem without integer constraints
    2) splits the operations into n_splits windows based on the start times

    Returns a list of windows each containing a subset of the operations
    Sanitizes the operations so the ones at start of windows don't have predecessors
    """
    operations = workload.get_operations()
    t, alpha = convex_schedule(workload, workload.get_transfer_times())

    # sort operations by start time
    times_and_ops = []
    for i in range(len(t)):
        times_and_ops.append([t[i], operations[i]])
    
    times_and_ops.sort(key=lambda time_and_op: time_and_op[0])

    # split operations into n_splits windows
    max_idx = np.argmax(t)
    max_start_time = np.max(t) + np.mean(operations[max_idx].get_durations())
    window_time = max_start_time / (n_splits - 1)
    # TODO this should just be n_splits and pass in n_splits - 1
    window_operations = [[] for _ in range(n_splits - 1)]
    for time, operation in times_and_ops:
        window_idx = int(time // window_time)
        window_operations[window_idx].append(operation)
    
    # sanitize operations
    for i in range(len(window_operations)):
        for j in range(len(window_operations[i])):
            operation = window_operations[i][j]
            if operation.predecessor is not None:
                # look if the predecessor is in a previous window
                for k in range(i):
                    if operation.predecessor in window_operations[k]:
                        # remove the predecessor from the current operation
                        operation.predecessor = None
                        break
    
    windows = []
    for operations in window_operations:
        windows.append(Window(window_time, operations, workload.machines))
    
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
    
    for _ in range(len(original_operations)):
        t = overlap_solver(original_workload, t, alpha)
    
    return t, alpha

def overlap_solver(workload, t, alpha):
    """
    Resolves overlaps by pushing them forward in time
    """
    transfer_times = workload.get_transfer_times()
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[i].predecessor == workload.operations[j]:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[j].predecessor == workload.operations[i]:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    t[j] = t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    t[i] = t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time

    return t

def check_for_overlaps(workload, t, alpha):
    """
    checks if the schedule has any overlaps
    """
    transfer_times = workload.get_transfer_times()
    count = 0

    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[i].predecessor == workload.operations[j]:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[j].predecessor == workload.operations[i]:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    count += 1
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    count += 1
    return count