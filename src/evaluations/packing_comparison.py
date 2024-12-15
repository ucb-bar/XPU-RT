import time
from workload import Workload, Operation
import plot
from scheduler import schedule_window
from generate_syn_workload import generate_syn_workload
import numpy as np
import csv
from workload_factory import create_sequential_job, create_sequential_workload_with_n_machines
from packing import greedy_packing, convex_packing, combine_solved_windows

def schedule_greedy(workload: Workload, n_splits):
    start_time = time.time()
    windows = greedy_packing(workload, n_splits)
    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)
    runtime = time.time() - start_time

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]

    return runtime, max_time

def schedule_convex(workload: Workload, n_splits):
    start_time = time.time()
    windows = convex_packing(workload, n_splits+1) #TODO this is because of jank math in convex_packing
    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)
    runtime = time.time() - start_time

    # find max time
    max_idx = np.argmax(t)
    max_operation = workload.operations[max_idx]
    max_machine = np.argmax(alpha[max_idx])
    max_time = t[max_idx] + max_operation.get_durations()[max_machine]

    return runtime, max_time

def param_sweep():
    # Define parameter ranges for sweeping
    num_jobs_list = np.arange(10, 40)  # Number of jobs
    num_machines_list = np.arange(1, 10)  # Number of machines

    # Prepare to record results
    results = []

    # Results file
    results_file = "greedy_runtime_results.csv"
    fieldnames = ["num_jobs", "num_machines", "convex_runtime", "convex_max_time", "greedy_runtime", "greedy_max_time"]

    # Ensure the CSV file is initialized with a header
    with open(results_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    # Perform the parameter sweep
    for num_jobs in num_jobs_list:
        for num_machines in num_machines_list:
            print(f"Generating workload with {num_jobs} jobs and {num_machines} machines...")
            
            # Generate synthetic workload

            # create symmetric transfer times with diagonals being 0
            transfer_times = np.random.randint(50, 1000, (num_machines, num_machines))
            for i in range(num_machines):
                transfer_times[i, i] = 0
            transfer_times = (transfer_times + transfer_times.T) // 2

            workload = create_sequential_workload_with_n_machines(num_jobs, 2, num_machines)
            workload.set_transfer_times(transfer_times)
            
            convex_runtime, convex_max_time = schedule_convex(workload, 5)
            greedy_runtime, greedy_max_time = schedule_greedy(workload, 5)

            # Append the current result to the file
            with open(results_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow({
                    "num_jobs": num_jobs,
                    "num_machines": num_machines,
                    "convex_runtime": convex_runtime,
                    "convex_max_time": convex_max_time,
                    "greedy_runtime": greedy_runtime,
                    "greedy_max_time": greedy_max_time
                })

    print("Parameter sweep completed.")

if __name__ == "__main__":
    param_sweep()
    