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
    num_machines = len(window.machines)
    transfer_times = window.get_transfer_times()

    alpha = cp.Variable((num_operations, num_machines), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
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
        pred = window.operations[i].get_predecessor()
        i_pred = None
        if pred is not None:
            try:
                i_pred = window.operations.index(pred)
            except ValueError:
                # happens when the predecessor is not in the window
                i_pred = None

        # check if there is a required predecessor
        if i_pred is not None:
            machine_pred = np.argmax(alpha[i_pred, :])
            machine_curr = np.argmax(alpha[i, :])

            transfer_time = transfer_times[machine_pred][machine_curr]

            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(window.operations[i_pred].get_durations()[:], alpha[i_pred, :])) + transfer_time
            )
    # (4)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k in range(num_machines):
                constraints.append(
                    t[i] >= t[j] + window.operations[j].get_durations()[k] - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H
                )
    # (5)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k in range(num_machines):
                constraints.append(
                    t[j] >= t[i] + window.operations[i].get_durations()[k] - (3 - alpha[i, k] - alpha[j, k] - beta[i, j]) * H
                )
    # (6)
    for i in range(num_operations):
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(window.operations[i].get_durations()[:], alpha[i, :]))
        )
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )

    # term to maximize consecutive empty space on each machine
    empty_space = cp.Variable(num_machines)
    for k in range(num_machines):
        for i in range(num_operations):
            for j in range(i+1, num_operations):
                constraints.append(
                    empty_space[k] >= t[i] - (t[j] + window.operations[j].get_durations()[k] - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H)
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
    num_machines = len(workload.machines)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_machines), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
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
    num_machines = len(workload.machines)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_machines), boolean=True)
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
            for k in range(num_machines):
                constraints.append(
                    g[i] >= (t[i] - t[j] - workload.operations[j].get_durations()[k]) - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H
                )

    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k in range(num_machines):
                constraints.append(
                    g[j] >= (t[j] - t[i] - workload.operations[i].get_durations()[k]) - (3 - alpha[i, k] - alpha[j, k] - beta[i, j]) * H
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
    windows = convex_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha