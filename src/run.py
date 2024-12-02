import time
from workload import Workload, Operation
import plot
from scheduler import schedule
from generate_syn_workload import generate_syn_workload
import numpy as np
import csv

# Define parameter ranges for sweeping
num_jobs_list = np.arange(1, 13)  # Number of jobs
num_machines_list = np.arange(1, 5)  # Number of machines

# Prepare to record results
results = []

# Results file
results_file = "runtime_results.csv"

# Ensure the CSV file is initialized with a header
with open(results_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["num_jobs", "num_machines", "runtime", "plot"])
    writer.writeheader()

# Perform the parameter sweep
for num_jobs in num_jobs_list:
    for num_machines in num_machines_list:
        print(f"Generating workload with {num_jobs} jobs and {num_machines} machines...")
        
        # Generate synthetic workload
        workload = generate_syn_workload(num_jobs, num_machines)
        
        # Schedule the workload and record runtime
        print(f"Scheduling workload for {num_jobs} jobs and {num_machines} machines...")
        start_time = time.time()
        t, alpha = schedule(workload)
        runtime = time.time() - start_time
        print(f"Scheduling completed in {runtime:.2f} seconds.")

        # Organize durations for plotting
        durations = []
        for i in range(len(workload.operations)):
            operation = workload.operations[i]
            if operation.predecessor is None:
                durations.append([operation.get_durations()])
            else:
                durations[-1].append(operation.get_durations())
        
        # Save the plot
        plot_filename = f"runtime/schedule_jobs{num_jobs}_machines{num_machines}.png"
        print(f"Saving plot to {plot_filename}...")
        try:
            plot.plot_optimization_schedule(durations, t, alpha, len(workload.machines), plot_filename)
        
            # Append the current result to the file
            with open(results_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["num_jobs", "num_machines", "runtime", "plot"])
                writer.writerow({
                    "num_jobs": num_jobs,
                    "num_machines": num_machines,
                    "runtime": runtime,
                    "plot": plot_filename
                })
        except Exception as e:
            print(f"Error saving plot: {e}")

print("Parameter sweep completed.")