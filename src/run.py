from workload import Workload, Operation
import plot
from scheduler import schedule

# Create the workload
mpc = Operation([200, 100])
dnn = Operation([800, 200])
ekf = Operation([10, 1000]) # TODO handle incompatible machines
workload = Workload([mpc, dnn, ekf])

# Schedule the workload
t, alpha = schedule(workload)

# Plot the optimized schedule
durations = [op.get_durations() for op in workload.operations]
plot.plot_optimization_schedule(durations, t, alpha, 2)