import time
import os
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
import csv
from workload import Workload, Operation
import plot
from scheduler import schedule
from workload_factory import create_sequential_job
from greedy import greedy
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def plot_greedy_schedule(
    start_times: Dict[Tuple[int, int], float],
    machine_allocations: Dict[Tuple[int, int], int],
    durations: Dict[Tuple[int, int], float],
    transfer_times: Dict[int, Dict[int, float]],
    save_path: str = None,
):
    """
    Plots the schedule produced by the greedy scheduling algorithm, incorporating
    operation colors, transfer times, and optional saving functionality.
    """
    # Define colors
    base_colors = plt.cm.tab20.colors
    color_gradients = np.linspace(0.6, 1.0, 5)
    transfer_color = "black"

    # Identify unique jobs and machines
    unique_jobs = sorted(set(job[0] for job in start_times.keys()))
    unique_machines = sorted(set(machine_allocations.values()))
    machine_mapping = {machine: i + 1 for i, machine in enumerate(unique_machines)}

    fig, ax = plt.subplots(figsize=(12, 6))
    current_operation_index = 0

    # Iterate over jobs and their operations
    for job_index, job in enumerate(unique_jobs):
        base_color = np.array(base_colors[job_index % len(base_colors)])

        for op in sorted(op for j, op in start_times.keys() if j == job):
            start_time = start_times[(job, op)]
            machine = machine_allocations[(job, op)]
            machine_idx = machine_mapping[machine]
            operation_duration = durations[(job, op)]
            gradient_factor = color_gradients[min(op, len(color_gradients) - 1)]
            operation_color = tuple(base_color * gradient_factor)

            ax.broken_barh(
                [(start_time, operation_duration)],
                (machine_idx - 0.4, 0.8),
                facecolors=operation_color,
                edgecolor="black",
            )

            # Draw transfer time if applicable
            if op > 0:
                prev_machine = machine_allocations[(job, op - 1)]
                if (
                    prev_machine in transfer_times
                    and machine in transfer_times[prev_machine]
                ):
                    transfer_time = transfer_times[prev_machine][machine]
                    ax.broken_barh(
                        [(start_time - transfer_time, transfer_time)],
                        (machine_mapping[prev_machine] - 0.4, 0.8),
                        facecolors=transfer_color,
                        edgecolor="black",
                    )

            current_operation_index += 1

    ax.set_yticks(list(machine_mapping.values()))
    ax.set_yticklabels(list(machine_mapping.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Greedy Schedule Execution Timeline")

    plt.savefig(save_path)
    plt.close()
    print(f"Greedy schedule plot saved to {save_path}")

    # def plot_greedy_schedule(
    #     start_times: Dict[Tuple[int, int], float],
    #     machine_allocations: Dict[Tuple[int, int], int],
    #     durations: Dict[Tuple[int, int], float],
    #     transfer_times: Dict[int, Dict[int, float]],
    #     save_path: str = None
    # ):
    #     """
    #     Plots the schedule produced by the greedy scheduling algorithm, incorporating
    #     operation colors, transfer times, and optional saving functionality.
    #     """
    #     # Define colors
    #     base_colors = plt.cm.tab20.colors
    #     color_gradients = np.linspace(0.6, 1.0, 5)
    #     transfer_color = 'black'

    #     # Identify unique jobs and machines
    #     unique_jobs = sorted(set(job[0] for job in start_times.keys()))
    #     unique_machines = sorted(set(machine_allocations.values()))
    #     machine_mapping = {machine: i + 1 for i, machine in enumerate(unique_machines)}

    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     current_operation_index = 0

    #     # Iterate over jobs and their operations
    #     for job_index, job in enumerate(unique_jobs):
    #         base_color = np.array(base_colors[job_index % len(base_colors)])

    #         for op in sorted(op for j, op in start_times.keys() if j == job):
    #             start_time = start_times[(job, op)]
    #             machine = machine_allocations[(job, op)]
    #             machine_idx = machine_mapping[machine]
    #             operation_duration = durations[(job, op)]
    #             gradient_factor = color_gradients[min(op, len(color_gradients) - 1)]
    #             operation_color = tuple(base_color * gradient_factor)

    #             ax.broken_barh([(start_time, operation_duration)],
    #                            (machine_idx - 0.4, 0.8),
    #                            facecolors=operation_color,
    #                            edgecolor='black')

    #             # Draw transfer time if applicable
    #             if op > 0:
    #                 prev_machine = machine_allocations[(job, op - 1)]
    #                 if prev_machine in transfer_times and machine in transfer_times[prev_machine]:
    #                     transfer_time = transfer_times[prev_machine][machine]
    #                     ax.broken_barh([(start_time - transfer_time, transfer_time)],
    #                                    (machine_mapping[prev_machine] - 0.4, 0.8),
    #                                    facecolors=transfer_color,
    #                                    edgecolor='black')

    #             current_operation_index += 1

    ax.set_yticks(list(machine_mapping.values()))
    ax.set_yticklabels(list(machine_mapping.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Greedy Schedule Execution Timeline")

    plt.savefig(save_path)
    plt.close()
    print(f"Greedy schedule plot saved to {save_path}")


def transfer_time_test():
    machines = ["cpu", "gpu", "fpga", "tpu", "asic"]
    transfer_times = np.array(
        [
            [5, 25, 10, 5, 10],
            [5, 30, 10, 5, 10],
            [10, 10, 20, 15, 10],
            [5, 5, 5, 20, 10],
            [10, 10, 10, 20, 10],
        ]
    )

    # Create workload once and reuse for both schedulers
    jobs = []
    for _ in range(5):
        operations = [
            Operation([np.random.randint(50, 1000) for _ in range(len(machines))])
            for _ in range(np.random.randint(3, 7))
        ]
        jobs.append(create_sequential_job(operations))

    # Flatten operations from jobs
    operations = [op for job in jobs for op in job.get_operations()]
    workload = Workload(operations, machines, transfer_times)

    # Create job duration mapping for greedy scheduler
    job_durations = {
        (idx, machine): op.get_durations()[machine_idx]
        for idx, job in enumerate(jobs)
        for op in job.get_operations()
        for machine_idx, machine in enumerate(machines)
    }

    # === Schedule using Optimization-Based Approach ===
    print("Scheduling workload using optimization approach...")
    start_time = time.time()
    try:
        t, alpha = schedule(workload)
        schedule_runtime = time.time() - start_time
        print(f"Optimization scheduling completed in {schedule_runtime:.2f} seconds.")
    except Exception as e:
        print(f"Error scheduling workload: {e}")
        return

    # === Schedule using Greedy Approach ===
    print("Scheduling workload using greedy approach...")
    start_time = time.time()
    try:
        t_greedy, alpha_greedy = greedy(jobs, machines, job_durations, transfer_times)
        print("Greedy Schedule Start Times:", t_greedy)  # Debug output
        print("Greedy Schedule Alpha:", alpha_greedy)  # Debug output
        greedy_runtime = time.time() - start_time
        print(f"Greedy scheduling completed in {greedy_runtime:.2f} seconds.")
    except Exception as e:
        print(f"Error scheduling workload with greedy algorithm: {e}")
        return

    # Convert dictionary results to numpy arrays for greedy scheduling
    t_greedy_array = np.array(
        [t_greedy[job_idx] for job_idx in sorted(t_greedy.keys())]
    )
    alpha_greedy_array = np.zeros((len(t_greedy), len(machines)))

    for job_idx, machine in alpha_greedy.items():
        machine_idx = machines.index(machine)  # Convert machine name to index
        alpha_greedy_array[job_idx, machine_idx] = 1  # Assign job to machine

    # Organize durations for plotting
    durations = []
    for job in jobs:
        job_durations_list = [op.get_durations() for op in job.get_operations()]
        durations.append(job_durations_list)

    # Define plot directory
    plot_dir = "/scratch/kris/scheduler/src/scripts/plots"
    os.makedirs(plot_dir, exist_ok=True)
    machines = ["asic", "cpu", "gpu", "fpga", "tpu"]

    # Save optimization-based scheduling plot
    schedule_plot_filename = os.path.join(plot_dir, "schedule_transfer_time_test1.png")
    print(f"Saving schedule plot to {schedule_plot_filename}...")
    try:
        plot.plot_optimization_schedule(
            durations,
            t,
            alpha,
            num_machines=len(machines),
            machines=machines,
            num_jobs=len(jobs),
            transfer_times=transfer_times,
            save_path=schedule_plot_filename,
        )

    except Exception as e:
        print(f"Error saving schedule plot: {e}")

    print("Scheduling workload using greedy approach...")
    start_time = time.time()
    try:
        t_greedy, alpha_greedy = greedy(jobs, machines, job_durations, transfer_times)
        greedy_runtime = time.time() - start_time
        print(f"Greedy scheduling completed in {greedy_runtime:.2f} seconds.")
    except Exception as e:
        print(f"Error scheduling workload with greedy algorithm: {e}")
        return
    plot_dir = "/scratch/kris/scheduler/src/scripts/plots"
    os.makedirs(plot_dir, exist_ok=True)
    # schedule_plot_filename = os.path.join(plot_dir, "schedule_transfer_time_test1.png")
    # print(f"Saving schedule plot to {schedule_plot_filename}...")
    # plot_greedy_schedule(t, alpha, save_path=schedule_plot_filename)
    greedy_plot_filename = os.path.join(
        plot_dir, "greedy_schedule_transfer_time_test1.png"
    )
    print(f"Saving greedy schedule plot to {greedy_plot_filename}...")
    plot_greedy_schedule(
        t_greedy,
        alpha_greedy,
        durations,
        transfer_times,
        save_path=greedy_plot_filename,
    )
    results_file = "runtime_results.csv"
    fieldnames = ["schedule_runtime", "greedy_runtime", "schedule_plot", "greedy_plot"]
    with open(results_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(
            {
                "schedule_runtime": schedule_runtime,
                "greedy_runtime": greedy_runtime,
                "schedule_plot": schedule_plot_filename,
                "greedy_plot": greedy_plot_filename,
            }
        )
    print("Transfer time test completed.")
    # greedy_plot_filename = os.path.join(plot_dir, "greedy_schedule_transfer_time_test1.png")
    # print(f"Saving greedy schedule plot to {greedy_plot_filename}...")
    # try:
    #    plot_greedy_schedule(t_greedy_array, alpha_greedy_array, save_path=greedy_plot_filename)
    # except Exception as e:
    #     print(f"Error saving greedy schedule plot: {e}")

    # greedy_plot_filename = os.path.join(plot_dir, "greedy_schedule_transfer_time_test1.png")
    # print(f"Saving greedy schedule plot to {greedy_plot_filename}...")
    # try:
    #     plot_greedy_schedule(
    #         durations, t_greedy_array, alpha_greedy_array,
    #         num_machines=len(machines), machines=machines,  num_jobs=len(jobs),
    #         transfer_times=transfer_times, save_path=greedy_plot_filename
    #     )
    # except Exception as e:
    #     print(f"Error saving greedy schedule plot: {e}")

    # Save results to CSV
    results_file = "runtime_results.csv"
    fieldnames = ["schedule_runtime", "greedy_runtime", "schedule_plot", "greedy_plot"]

    with open(results_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(
            {
                "schedule_runtime": schedule_runtime,
                "greedy_runtime": greedy_runtime,
                "schedule_plot": schedule_plot_filename,
                "greedy_plot": greedy_plot_filename,
            }
        )

    print("Transfer time test completed.")


if __name__ == "__main__":
    transfer_time_test()
