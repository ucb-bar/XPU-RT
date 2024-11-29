from workload import Workload, Job, Operation, Window
import plot
from scheduler import schedule_window
from workload_factory import generate_test_window

# Create the window
print("Creating synthetic window...")
window = generate_test_window()

# Schedule the workload
print("Scheduling the workload...")
t, alpha = schedule_window(window)

# Plot the optimized schedule
print("Plotting the optimized schedule...")
durations = [op.get_durations() for op in window.operations]
plot.plot_optimization_schedule(durations, t, alpha, len(window.machines))