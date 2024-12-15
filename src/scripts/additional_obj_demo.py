"""
This script demonstrates the use of additional objectives in the optimization problem.
"""

# add parent path to sys path to enable imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from workload_factory import generate_syn_workload, generate_syn_transfer_times
from scheduler import schedule_additional_objectives
from plot import plot_optimization_schedule

def run_additional_objectives():
    # Generate synthetic workload
    transfer_times = generate_syn_transfer_times(3)
    workload = generate_syn_workload(10, 3, transfer_times)

    # Set nominal start times based on the first job having a period of 500
    # Set all operations without a desired start time to -1
    nominal_start_times = np.ones(len(workload.operations)) * -1
    period = 500
    for i in range(3):
        nominal_start_times[i] = i * period

    # Schedule the workload
    t, alpha = schedule_additional_objectives(workload, nominal_start_times)

    plot_optimization_schedule(workload.get_durations(), t, alpha, len(workload.machines), transfer_times, save_path="plots/additional_objectives_schedule.png")

if __name__ == "__main__":
    run_additional_objectives()