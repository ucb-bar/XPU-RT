import os
import matplotlib.pyplot as plt
import numpy as np


def plot_combined_schedule(
    durations: list[list[list[float]]],
    t,
    alpha,
    num_machines,
    transfer_times,
    num_jobs,
    save_path=None,
    plot_title="Schedule",
    start_times=None,
    machine_allocations=None,
):
    """
    Parses CVXPY optimization outputs to plot a schedule of jobs on machines over time.

    Parameters:
    - durations: List of jobs, where each job is a list of operations,
                 and each operation is a list of runtimes for different machines.
    - t: CVXPY variable (vector) containing start times for each operation in each job.
    - alpha: CVXPY variable (matrix) containing machine assignments for each operation.
    - num_machines: Total number of machines.
    - transfer_times: Transfer time matrix between machines.
    - save_path: Optional file path to save the plot.
    - plot_title: Title for the plot.
    - num_jobs: Number of jobs (used for naming the file).
    """

    # If using greedy scheduling, use start_times and machine_allocations
    if start_times is not None and machine_allocations is not None:
        job_colors = {}
        unique_jobs = sorted(set(job[0] for job in start_times.keys()))
        color_map = plt.cm.get_cmap("tab10", len(unique_jobs))
        for idx, job in enumerate(unique_jobs):
            job_colors[job] = color_map(idx)

        machine_mapping = {
            machine: i + 1
            for i, machine in enumerate(sorted(set(machine_allocations.values())))
        }

        for (job, op), start_time in start_times.items():
            machine = machine_allocations[(job, op)]
            machine_idx = machine_mapping[machine]
            operation_duration = durations[(job, op)]

            ax.barh(
                machine_idx,
                width=operation_duration,
                left=start_time,
                color=job_colors[job],
                edgecolor="black",
                label=f"Job {job}, Op {op}",
            )

        ax.set_yticks(list(machine_mapping.values()))
        ax.set_yticklabels(list(machine_mapping.keys()))

    # Ensure job and machine numbers are provided
    if num_jobs is None or num_machines is None:
        raise ValueError(
            "num_jobs and num_machines must be provided to generate dynamic filenames."
        )

    # Define save directory and create it if it doesn't exist
    plot_dir = "/scratch/kris/scheduler/src/scripts/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Define the dynamic filename based on job and machine count
    if save_path is None:
        save_path = os.path.join(
            plot_dir, f"schedule_jobs{num_jobs}_machines{num_machines}.png"
        )

    # Convert optimization variables to NumPy arrays if needed
    start_times = np.array(t.value).flatten() if not isinstance(t, np.ndarray) else t
    alpha_values = np.array(alpha.value) if not isinstance(alpha, np.ndarray) else alpha

    # Validate number of operations
    num_operations = sum(len(job) for job in durations)
    if num_operations != len(start_times):
        raise ValueError(
            f"Mismatch: durations specify {num_operations} operations, "
            f"but start_times has {len(start_times)} entries."
        )

    # Determine machine assignments
    machine_assignments = np.argmax(alpha_values, axis=1)

    # Plot settings
    fig, ax = plt.subplots(figsize=(10, 6))
    base_colors = plt.cm.tab20.colors
    color_gradients = np.linspace(0.6, 1.0, 5)
    transfer_color = "black"

    # Plot each operation
    current_operation_index = 0
    for job_index, job_durations in enumerate(durations):
        base_color = np.array(base_colors[job_index % len(base_colors)])
        for operation_index, operation_runtimes in enumerate(job_durations):
            start_time = start_times[current_operation_index]
            machine = machine_assignments[current_operation_index]
            operation_duration = operation_runtimes[machine]
            gradient_factor = color_gradients[
                min(operation_index, len(color_gradients) - 1)
            ]
            operation_color = tuple(base_color * gradient_factor)

            ax.broken_barh(
                [(start_time, operation_duration)],
                (machine - 0.4, 0.8),
                facecolors=operation_color,
                edgecolor="black",
            )

            if operation_index > 0:
                prev_machine = machine_assignments[current_operation_index - 1]
                transfer_time = transfer_times[prev_machine][machine]
                ax.broken_barh(
                    [(start_time - transfer_time, transfer_time)],
                    (prev_machine - 0.4, 0.8),
                    facecolors=transfer_color,
                    edgecolor="black",
                )

            current_operation_index += 1

    # Labels and title
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(plot_title)

    plt.tight_layout()

    # Save the plot with dynamic filename
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")

    # Optional: show a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for label, handle in zip(labels, handles)}

    # Ensure 'Transfer Time' appears last in the legend
    if "Transfer\nTime" in unique_labels:
        transfer_handle = unique_labels.pop("Transfer\nTime")
        unique_labels["Transfer\nTime"] = transfer_handle

    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper right",
        title="Jobs",
        bbox_to_anchor=(1.18, 1),
    )

    plt.tight_layout()

    # Save the plot
    # check if the save path has a folder
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
