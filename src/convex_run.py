from workload import Workload, Job, Operation, Window
import plot
from scheduler import schedule_window
from workload_factory import generate_syn_workload, create_sequential_workload
from packing import convex_packing, combine_solved_windows, greedy_packing, check_for_overlaps
import numpy as np
import csv

def greedy_pack_test() -> float:
    workload = create_sequential_workload(10, 3)
    workload_durations = workload.get_durations()
    transfer_times = np.array([
            [0, 10, 50],
            [10, 0, 200],
            [50, 200, 0]
        ])
    workload.set_transfer_times(transfer_times)
    windows = greedy_packing(workload, 5)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]
    return max_time

def convex_pack_test() -> float:
    workload = create_sequential_workload(10, 3)
    workload_durations = workload.get_durations()
    transfer_times = np.array([
            [0, 10, 50],
            [10, 0, 200],
            [50, 200, 0]
        ])
    workload.set_transfer_times(transfer_times)
    windows = convex_packing(workload, 6)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)
    plot.plot_optimization_schedule(workload_durations, t, alpha, len(workload.machines), transfer_times, save_path="plots/combined_convex_schedule.png")

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]
    return max_time

def test_packing():
    for i in range(10):
        greedy_time = greedy_pack_test()
        convex_time = convex_pack_test()
        # log to csv
        with open('packing_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([greedy_time, convex_time])

convex_pack_test()