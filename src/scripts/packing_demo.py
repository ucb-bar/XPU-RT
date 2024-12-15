"""
Shows examples of greedy and convex packing algorithms
"""

# add parent path to sys path to enable imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot
from scheduler import schedule_with_greedy_packing, schedule_with_convex_packing
from workload_factory import create_syn_sequential_workload
import numpy as np

def greedy_pack_test() -> float:
    """
    Example of the greedy packing algorithm, returns the duration of the schedule
    """
    transfer_times = np.array([
            [0, 10, 50],
            [10, 0, 200],
            [50, 200, 0]
    ])
    workload = create_syn_sequential_workload(5, 3, 3, transfer_times)
        
    t, alpha = schedule_with_greedy_packing(workload, 3)

    plot.plot_optimization_schedule(workload.get_durations(), t, alpha, len(workload.machines), transfer_times, save_path="plots/greedy_schedule.pdf", plot_title="Greedy Packing Schedule")

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]
    return max_time

def convex_pack_test() -> float:
    """
    Example of the convex packing algorithm, returns the duration of the schedule
    """

    transfer_times = np.array([
        [0, 10, 50],
        [10, 0, 200],
        [50, 200, 0]
    ])
    workload = create_syn_sequential_workload(5, 3, 3, transfer_times)
    
    t, alpha = schedule_with_convex_packing(workload, 3)

    plot.plot_optimization_schedule(workload.get_durations(), t, alpha, len(workload.machines), transfer_times, save_path="plots/convex_schedule.pdf", plot_title="Convex Packing Schedule")

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]
    return max_time

if __name__ == "__main__":
    # convex_pack_test()
    greedy_pack_test()
