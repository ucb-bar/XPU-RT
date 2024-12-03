import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_schedule(durations: list[list[list[float]]], t, alpha, num_machines, transfer_times, save_path="plots/schedule.png"):
    """
    Parses CVXPY optimization outputs to plot a schedule of jobs on machines over time,
    ensuring operations within the same job are plotted with gradients of the same color.

    Parameters:
    - durations: list of jobs, where each job is a list of operations, and each operation
                 is a list of runtimes for different machines.
    - t: CVXPY variable (vector) containing start times for each operation in each job.
    - alpha: CVXPY variable (matrix) containing machine assignments for each operation.
    - num_machines: total number of machines.
    """
    # Get the optimized start times and machine assignments from CVXPY variables
    if not isinstance(t, np.ndarray):
        start_times = np.array(t.value).flatten()  # Convert CVXPY start times to a 1D numpy array
    else:
        start_times = t
    
    # check if alpha is a numpy array
    if not isinstance(alpha, np.ndarray):
        alpha_values = np.array(alpha.value)  # Convert CVXPY alpha matrix to a numpy array
    else:
        alpha_values = alpha

    # Ensure the total number of operations matches the start times
    num_operations = sum(len(job) for job in durations)
    if num_operations != len(start_times):
        raise ValueError(f"Mismatch: durations specify {num_operations} operations, "
                         f"but start_times has {len(start_times)} entries.")

    # Determine machine assignments based on alpha values
    machine_assignments = np.argmax(alpha_values, axis=1)  # Finds the machine (index) assigned for each operation

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    base_colors = plt.cm.tab20.colors  # Use 'tab20' colormap for better color distinction
    color_gradients = np.linspace(0.6, 1.0, 5)  # Gradients to adjust color intensity
    transfer_color = 'black'  # Color for transfer times

    # Plot each operation as a bar on its assigned machine row
    current_operation_index = 0
    for job_index, job_durations in enumerate(durations):
        base_color = np.array(base_colors[job_index % len(base_colors)])  # Base color for the job
        for operation_index, operation_runtimes in enumerate(job_durations):
            # Fetch relevant data
            start_time = start_times[current_operation_index]
            machine = machine_assignments[current_operation_index]
            operation_duration = operation_runtimes[machine]  # Get runtime for the assigned machine
            gradient_factor = color_gradients[min(operation_index, len(color_gradients) - 1)]  # Gradation for this operation
            operation_color = tuple(base_color * gradient_factor)  # Adjust base color intensity

            # Plot operation as a horizontal bar
            ax.broken_barh([(start_time, operation_duration)], 
                           (machine - 0.4, 0.8),
                           facecolors=operation_color,
                           edgecolor='black',
                           label=f'Job {job_index + 1}' if operation_index == 0 else None)
            
            # Plot transfer time if this is not the first operation
            if operation_index > 0:
                prev_machine = machine_assignments[current_operation_index - 1]
                transfer_time = transfer_times[prev_machine][machine]
                ax.broken_barh([(start_time - transfer_time, transfer_time)], 
                            (prev_machine - 0.4, 0.8),
                            facecolors=transfer_color,
                            edgecolor='black',
                            label='Transfer\nTime' if current_operation_index == 1 else None)
            
            current_operation_index += 1

    # Customize the y-axis to show each machine as a separate row
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(num_machines)])
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Optimized Job Schedule With Convex Packing")

    # Optional: show a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for label, handle in zip(labels, handles)}

    # Ensure 'Transfer Time' appears last in the legend
    if 'Transfer\nTime' in unique_labels:
        transfer_handle = unique_labels.pop('Transfer\nTime')
        unique_labels['Transfer\nTime'] = transfer_handle
        
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', title="Jobs", bbox_to_anchor=(1.18, 1))

    plt.tight_layout()

    # Save the plot
    # check if the save path has a folder
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
