from workload import Workload, Job, Operation, Window
import cvxpy as cp
import numpy as np
from typing import Tuple

def convex_schedule(workload: Workload):
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
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(workload.operations[i_pred].get_durations()[:], alpha[i_pred, :]))
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
            constraints.append(
                beta[i, j] >= 0
            )
    for i in range(num_operations):
        for j in range(num_machines):
            constraints.append(
                alpha[i, j] >= 0
            )

    # Optimization problem
    objective = cp.Minimize(C_max)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    # Create a boolean mask where the maximum value in each row is True
    mask = alpha.value == alpha.value.max(axis=1, keepdims=True)

    # Convert the boolean mask to integers (True becomes 1, False becomes 0)
    alpha = mask.astype(int)
    print(alpha)

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
    return None

def convex_packing(workload: Workload) -> list[Window]:
    """
    approximate the optimal packing of jobs into windows using convex optimization.
    
    solves optimization problem without integer constraints, then rounds results to nearest integer to obtain feasible solution, splitting that into windows
    """

    t, alpha = convex_schedule(workload)
    
    