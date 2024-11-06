import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_schedule(durations, t, alpha, num_machines):
    """
    Parses CVXPY optimization outputs to plot a schedule of jobs on machines over time.

    Parameters:
    - durations: list of durations for each job (list of lists or array)
    - t: CVXPY variable (vector) containing start times for each job
    - alpha: CVXPY variable (matrix) containing machine assignments for each job
    - num_machines: total number of machines
    """

    # Get the optimized start times and machine assignments from CVXPY variables
    start_times = np.array(t.value).flatten()  # Convert CVXPY start times to a 1D numpy array
    alpha_values = np.array(alpha.value)  # Convert CVXPY alpha matrix to a numpy array
    
    # Determine machine assignments based on alpha values
    machine_assignments = np.argmax(alpha_values, axis=1)  # Finds the machine (index) assigned for each job

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20.colors  # Color map for distinguishing jobs

    # Plot each job as a bar on its assigned machine row
    for i, (duration, start_time, machine) in enumerate(zip(durations, start_times, machine_assignments)):
        job_duration = duration[machine] if isinstance(duration, (list, np.ndarray)) else duration
        ax.broken_barh([(start_time, job_duration)], (machine - 0.4, 0.8),
                       facecolors=colors[i % len(colors)], edgecolor='black', label=f'Job {i+1}' if i < len(colors) else "")

    # Customize the y-axis to show each machine as a separate row
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(num_machines)])
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Optimized Job Schedule Over Time")

    # Optional: show a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for label, handle in zip(labels, handles)}
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', title="Jobs", bbox_to_anchor=(1.15, 1))

    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # save the plot
    plt.savefig("plots/schedule.png", dpi=300, bbox_inches='tight')