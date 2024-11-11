from workload import Workload, Operation
import plot
from scheduler import schedule
from generate_syn_workload import generate_syn_workload

# Create the workload
print("Creating synthetic workload...")
workload = generate_syn_workload(1, 15, 5)

# Schedule the workload
print("Scheduling the workload...")
t, alpha = schedule(workload)

# Plot the optimized schedule
print("Plotting the optimized schedule...")
durations = [op.get_durations() for op in workload.operations]
plot.plot_optimization_schedule(durations, t, alpha, workload.num_machines)