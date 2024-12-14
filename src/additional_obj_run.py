from workload import Workload, Operation
import plot
from scheduler import schedule_additional_objectives
from generate_syn_workload import generate_syn_workload
import numpy as np

def run_additional_objectives():
    # Generate synthetic workload
    workload = generate_syn_workload(10, 3)

    nominal_start_times = np.ones(len(workload.operations)) * -1
    period = 500
    for i in range(3):
        nominal_start_times[i] = i * period

    transfer_times = np.array([
        [0, 10, 50],
        [10, 0, 200],
        [50, 200, 0]
    ])

    # Schedule the workload
    t, alpha = schedule_additional_objectives(workload, transfer_times, nominal_start_times)

    plot.plot_optimization_schedule(workload.get_durations(), t, alpha, len(workload.machines), transfer_times, save_path="plots/additional_objectives_schedule.png")

run_additional_objectives()