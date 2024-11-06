#
# Problem formulation from https://www.sciencedirect.com/science/article/pii/S037722172300382X#sec0014 Section 2.1.
#

import cvxpy as cp
import numpy as np
from workload import Workload, Operation
import plot

# Create the workload
mpc = Operation([200, 100])
dnn = Operation([800, 200])
ekf = Operation([10, 1000]) # TODO handle incompatible machines
workload = Workload([mpc, dnn, ekf])
num_operations = workload.num_operations
num_machines = workload.num_machines

alpha = cp.Variable((num_operations, num_machines), boolean=True)
beta = cp.Variable((num_operations, num_operations), boolean=True)
t = cp.Variable(num_operations)
C_max = cp.Variable()

# Hyperparameters
H = 1000

# Constraints
constraints = []
# (2)
for i in range(num_operations):
    constraints.append(
        cp.sum(alpha[i, :]) == 1
    )
# (3)
# for i in range(num_operations):
#     i_pred = workload.operations[i].get_predecessor()

#     # check if there is no required predecessor
#     if i_pred is None:
#         continue

#     constraints.append(
#         t[i] >= t[i_pred] + cp.sum(cp.multiply(workload.operations[i_pred].get_durations()[:], alpha[i_pred, :]))
#     )
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
problem.solve(solver=cp.MOSEK, verbose=True)

print("Status: ", problem.status)
print("Optimal value: ", problem.value)

# Plot the optimized schedule
durations = [op.get_durations() for op in workload.operations]
plot.plot_optimization_schedule(durations, t, alpha, num_machines)