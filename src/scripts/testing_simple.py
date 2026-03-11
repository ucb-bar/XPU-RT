"""
Simple testing script with zero transfer times between machines.
This simplifies the scheduling problem by removing transfer time constraints.
"""

import time
import os
from workload import Workload, Operation
import plot
from scheduler import schedule
from workload_factory import create_syn_sequential_workload
import numpy as np
import csv
from workload_factory import create_sequential_job


def create_zero_transfer_times(n_machines: int) -> np.ndarray:
    """
    Creates a zero matrix for transfer times (no transfer time between machines).
    """
    return np.zeros((n_machines, n_machines))


def simple_test():
    """
    Simple test with zero transfer times between machines.
    """
    machines = ["cpu", "gpu", "fpga"]
    n_machines = len(machines)

    # Create zero transfer times (no transfer time between machines)
    transfer_times = create_zero_transfer_times(n_machines)

    # Create a workload with 4 jobs
    operations1 = []
    operations2 = []
    operations3 = []
    operations4 = []

    for _ in range(5):
        processing_times = [np.random.randint(50, 1000) for _ in range(n_machines)]
        operations1.append(Operation(processing_times))

    for _ in range(3):
        processing_times = [np.random.randint(50, 1000) for _ in range(n_machines)]
        operations2.append(Operation(processing_times))

    for _ in range(6):
        processing_times = [np.random.randint(50, 1000) for _ in range(n_machines)]
        operations3.append(Operation(processing_times))

    for _ in range(4):
        processing_times = [np.random.randint(50, 1000) for _ in range(n_machines)]
        operations4.append(Operation(processing_times))

    job1 = create_sequential_job(operations1)
    job2 = create_sequential_job(operations2)
    job3 = create_sequential_job(operations3)
    job4 = create_sequential_job(operations4)

    operations = (
        job1.get_operations()
        + job2.get_operations()
        + job3.get_operations()
        + job4.get_operations()
    )

    workload = Workload(operations, machines, transfer_times)

    print("Scheduling workload...")
    t, alpha = schedule(workload)
    print("Scheduling completed.")

    # Count number of jobs (operations with no predecessor start a new job)
    num_jobs = sum(1 for op in operations if not op.predecessors)

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plot.plot_optimization_schedule(
        workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(workload.machines),
        machines,
        transfer_times,
        save_path="plots/simple_test.png",
    )
    print(f"Plot saved to plots/simple_test.png")


def param_sweep_simple():
    """
    Parameter sweep with zero transfer times.
    """
    # Define parameter ranges for sweeping
    num_jobs_list = np.arange(1, 13)  # Number of jobs
    num_machines_list = np.arange(1, 5)  # Number of machines

    # Results file
    results_file = "runtime_results_simple.csv"

    # Ensure the CSV file is initialized with a header
    with open(results_file, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["num_jobs", "num_machines", "runtime", "plot"]
        )
        writer.writeheader()

    # Perform the parameter sweep
    for num_jobs in num_jobs_list:
        for num_machines in num_machines_list:
            print(
                f"Generating workload with {num_jobs} jobs and {num_machines} machines..."
            )

            # Create zero transfer times
            transfer_times = create_zero_transfer_times(num_machines)
            n_operations_per_job = 3  # Default number of operations per job
            workload = create_syn_sequential_workload(
                num_jobs, n_operations_per_job, num_machines, transfer_times
            )

            # Schedule the workload and record runtime
            print(
                f"Scheduling workload for {num_jobs} jobs and {num_machines} machines..."
            )
            start_time = time.time()
            t, alpha = schedule(workload)
            runtime = time.time() - start_time
            print(f"Scheduling completed in {runtime:.2f} seconds.")

            # Organize durations for plotting
            durations = []
            for i in range(len(workload.operations)):
                operation = workload.operations[i]
                if not operation.predecessors:
                    durations.append([operation.get_durations()])
                else:
                    durations[-1].append(operation.get_durations())

            # Save the plot
            os.makedirs("runtime", exist_ok=True)
            plot_filename = (
                f"runtime/schedule_simple_jobs{num_jobs}_machines{num_machines}.png"
            )
            print(f"Saving plot to {plot_filename}...")
            try:
                plot.plot_optimization_schedule(
                    durations,
                    t,
                    alpha,
                    num_jobs,
                    num_machines,
                    workload.machines,
                    transfer_times,
                    save_path=plot_filename,
                )

                # Append the current result to the file
                with open(results_file, mode="a", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=["num_jobs", "num_machines", "runtime", "plot"]
                    )
                    writer.writerow(
                        {
                            "num_jobs": num_jobs,
                            "num_machines": num_machines,
                            "runtime": runtime,
                            "plot": plot_filename,
                        }
                    )
            except Exception as e:
                print(f"Error saving plot: {e}")

    print("Parameter sweep completed.")


if __name__ == "__main__":
    simple_test()
