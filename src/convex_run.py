from workload import Workload, Job, Operation, Window
import plot
from scheduler import schedule_window
from workload_factory import generate_syn_workload
from packing import convex_schedule


window = generate_syn_workload(50, 3)

t, alpha = convex_schedule(window)

# Plot the optimized schedule
print("Plotting the optimized window schedule...")
durations = []
for i in range(len(window.operations)):
    # group together operations that belong to the same job
    operation = window.operations[i]
    if operation.predecessor is None:
        durations.append([operation.get_durations()])
    else:
        durations[-1].append(operation.get_durations())
plot.plot_optimization_schedule(durations, t, alpha, len(window.machines), "plots/convex_schedule.png")