import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_optimization_schedule(durations, t, alpha, num_jobs, num_machines, machines, transfer_times, save_path="plots/schedule.png", plot_title="Schedule", workload=None):
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
    - workload: Optional Workload object to extract job names and operation IDs.
    """

    # Ensure job and machine numbers are provided
    if num_jobs is None or num_machines is None:
        raise ValueError("num_jobs and num_machines must be provided to generate dynamic filenames.")

    # Define save directory and create it if it doesn't exist
    plot_dir = "/scratch/kris/scheduler/src/scripts/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Define the dynamic filename based on job and machine count
    if save_path is None:
        save_path = os.path.join(plot_dir, f"schedule_jobs{num_jobs}_machines{num_machines}.png")
    else:
        # Create directory for custom save path if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if there's a directory component
            os.makedirs(save_dir, exist_ok=True)

    # Convert optimization variables to NumPy arrays if needed
    start_times = np.array(t.value).flatten() if not isinstance(t, np.ndarray) else t
    alpha_values = np.array(alpha.value) if not isinstance(alpha, np.ndarray) else alpha

    # Validate number of operations
    num_operations = sum(len(job) for job in durations)
    if num_operations != len(start_times):
        raise ValueError(f"Mismatch: durations specify {num_operations} operations, "
                         f"but start_times has {len(start_times)} entries.")

    # Determine machine assignments
    machine_assignments = np.argmax(alpha_values, axis=1)

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use highly distinct colors - manually specify first few for maximum distinction
    # These colors are chosen to be maximally perceptually distinct (RGB values)
    # First 3 colors are specifically chosen to be very different from each other
    highly_distinct_colors = [
        (0.2, 0.6, 1.0),      # Bright Sky Blue - very distinct, not dark
        (1.0, 0.5, 0.0),      # Bright Orange - very distinct
        (0.0, 0.8, 0.4),      # Bright Green - very distinct
        (0.9, 0.1, 0.1),      # Bright Red
        (0.7, 0.0, 0.9),      # Bright Purple
        (1.0, 0.8, 0.0),      # Bright Yellow
        (0.0, 0.7, 0.9),      # Cyan
        (0.9, 0.5, 0.7),      # Pink
        (0.5, 0.3, 0.1),      # Brown
        (0.3, 0.3, 0.3),      # Gray
    ]
    
    # For small number of jobs, use the most distinct colors
    if num_jobs <= len(highly_distinct_colors):
        base_colors = highly_distinct_colors[:num_jobs]
    elif num_jobs <= 20:
        # Extend with tab20 colors
        colors1 = highly_distinct_colors
        colors2 = plt.cm.tab20(np.linspace(0, 1, 20))
        base_colors = colors1 + [tuple(c[:3]) for c in colors2[:num_jobs - len(colors1)]]
    else:
        # For many jobs, use all available colors
        colors1 = highly_distinct_colors
        colors2 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors3 = plt.cm.Set3(np.linspace(0, 1, 12))
        base_colors = colors1 + [tuple(c[:3]) for c in colors2] + [tuple(c[:3]) for c in colors3]
        base_colors = base_colors[:num_jobs]
    
    # Ensure base_colors are tuples
    base_colors = [tuple(c) if isinstance(c, (list, tuple, np.ndarray)) else c for c in base_colors]
    
    color_gradients = np.linspace(0.7, 1.0, 5)  # Slightly brighter gradients
    transfer_color = 'black'
    
    # Extract job names and operation info from workload if provided
    job_names_list = []
    operation_ids_list = []
    
    if workload is not None:
        # Get job names from workload
        job_names_list = workload.job_names if hasattr(workload, 'job_names') and workload.job_names else []
        # Get operation IDs
        operation_ids_list = [op.operation_id for op in workload.operations] if hasattr(workload, 'operations') else []
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    # Map job_id to color index
    # Build a mapping from job_id to color index to ensure consistent colors
    # CRITICAL: This ensures that if job_ids are unique, color_indices are unique
    job_id_to_color_index = {}
    num_unique_job_ids = num_jobs  # Default to num_jobs
    if workload is not None:
        # Collect all unique job_ids and map them to color indices
        unique_job_ids = []
        for op in workload.operations:
            if op.job_id is not None and op.job_id not in unique_job_ids:
                unique_job_ids.append(op.job_id)
        unique_job_ids.sort()  # Sort to ensure consistent ordering
        num_unique_job_ids = len(unique_job_ids)  # Use actual number of unique job_ids
        for idx, job_id in enumerate(unique_job_ids):
            job_id_to_color_index[job_id] = idx
        # Verify uniqueness: each unique job_id maps to a unique color_index
        color_indices = list(job_id_to_color_index.values())
        assert len(color_indices) == len(set(color_indices)), f"Color indices must be unique! Got: {color_indices}"
        # Debug: print the mapping
        if len(unique_job_ids) <= 5:  # Only print for small number of jobs
            print(f"Job ID to color index mapping: {job_id_to_color_index}")
            # Also print job_id distribution
            job_id_counts = {}
            for op in workload.operations:
                if op.job_id is not None:
                    job_id_counts[op.job_id] = job_id_counts.get(op.job_id, 0) + 1
            print(f"Job ID distribution: {job_id_counts}")
    
    # CRITICAL FIX: Ensure base_colors has enough colors for all unique job_ids
    # If we have more unique job_ids than num_jobs, we need to extend base_colors
    if num_unique_job_ids > len(base_colors):
        # Extend base_colors to have at least num_unique_job_ids colors
        colors_to_add = num_unique_job_ids - len(base_colors)
        if colors_to_add > 0:
            # Use additional distinct colors from our curated list first
            # This ensures we use the most perceptually distinct colors available
            remaining_distinct = [c for c in highly_distinct_colors if c not in base_colors]
            if len(remaining_distinct) >= colors_to_add:
                # Use distinct colors from our curated list
                colors_being_added = remaining_distinct[:colors_to_add]
                base_colors.extend(colors_being_added)
                if len(job_id_to_color_index) <= 5:
                    print(f"DEBUG: Adding distinct colors: {colors_being_added}")
            else:
                # Use all remaining distinct colors, then fill with tab20
                base_colors.extend(remaining_distinct)
                still_needed = colors_to_add - len(remaining_distinct)
                if still_needed > 0:
                    # Use tab20 colors, but skip ones that are too similar to existing colors
                    additional_colors = plt.cm.tab20(np.linspace(0, 1, 20))
                    # Filter out colors that are too similar to existing ones
                    for color in additional_colors:
                        if still_needed <= 0:
                            break
                        color_tuple = tuple(color[:3])
                        # Check if this color is sufficiently different from existing colors
                        is_distinct = True
                        for existing_color in base_colors:
                            # Calculate color distance (simple Euclidean distance in RGB space)
                            color_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(color_tuple, existing_color)))
                            if color_dist < 0.3:  # Threshold for "too similar"
                                is_distinct = False
                                break
                        if is_distinct:
                            base_colors.append(color_tuple)
                            still_needed -= 1
        if len(job_id_to_color_index) <= 5:
            print(f"DEBUG: Extended base_colors from {len(base_colors) - colors_to_add} to {len(base_colors)} to support {num_unique_job_ids} unique job_ids")
    
    # Plot each operation
    current_operation_index = 0
    jobs_in_legend = set()  # Track which jobs we've added to legend
    
    for job_index, job_durations in enumerate(durations):
        # Process each operation individually to get its correct job_id and color
        for operation_index, operation_runtimes in enumerate(job_durations):
            # Get the actual job_id for THIS specific operation
            op_job_id = None
            if workload is not None and current_operation_index < len(workload.operations):
                current_op = workload.operations[current_operation_index]
                op_job_id = current_op.job_id if current_op.job_id is not None else None
            
            # Determine color index for THIS operation based on its job_id
            # CRITICAL: Always use job_id_to_color_index lookup when job_id is available
            # This ensures unique job_ids get unique color_indices
            # NEVER use job_index when job_id is available - job_index is just the position in durations list
            if op_job_id is not None:
                # MUST use the mapping - this guarantees unique color_indices for unique job_ids
                if op_job_id in job_id_to_color_index:
                    op_color_index = job_id_to_color_index[op_job_id]
                    # Debug: verify lookup immediately after assignment
                    if len(jobs_in_legend) < 5 and current_operation_index >= 62 and current_operation_index <= 64:
                        print(f"DEBUG IMMEDIATE: op={current_operation_index}, job_id={op_job_id}, lookup={job_id_to_color_index[op_job_id]}, op_color_index={op_color_index}, base_colors[{op_color_index}]={base_colors[op_color_index] if op_color_index < len(base_colors) else 'OUT_OF_BOUNDS'}")
                else:
                    # This should never happen if mapping was built correctly
                    raise ValueError(f"job_id {op_job_id} not found in mapping {job_id_to_color_index}. "
                                   f"This indicates a bug in the mapping construction.")
            else:
                # No job_id available, use job_index as fallback (backward compatibility)
                op_color_index = job_index
            
            # Ensure color_index is valid (modulo to handle cases where we have more jobs than colors)
            # Store the original color_index before modulo (for debugging)
            original_color_index = op_color_index
            op_color_index = op_color_index % len(base_colors)
            op_base_color = np.array(base_colors[op_color_index])
            
            # Debug: check if modulo changed the value
            if len(jobs_in_legend) < 5 and current_operation_index >= 62 and current_operation_index <= 64:
                print(f"DEBUG AFTER MODULO: op={current_operation_index}, original={original_color_index}, after_modulo={op_color_index}, len(base_colors)={len(base_colors)}")
            
            # Get job name for legend (only for first operation of each unique job_id)
            if op_job_id is not None:
                name_index = op_job_id
            else:
                name_index = job_index
            
            job_label = None
            if name_index < len(job_names_list) and job_names_list[name_index]:
                job_label = job_names_list[name_index]
            else:
                job_label = f"Job {name_index}"
            
            # Add to legend only once per unique job_id
            # Use job_id as the key to ensure we don't add duplicates
            if op_job_id is not None:
                legend_key = f"job_id_{op_job_id}"
            else:
                legend_key = f"job_label_{job_label}"
            
            # Debug: check op_color_index right before legend check
            if len(jobs_in_legend) < 5 and current_operation_index >= 62 and current_operation_index <= 64:
                print(f"DEBUG BEFORE LEGEND: op={current_operation_index}, op_color_index={op_color_index}, original={original_color_index}, job_index={job_index}, legend_key={legend_key}, in_legend={legend_key in jobs_in_legend}")
            
            if legend_key not in jobs_in_legend:
                # Debug output for first few jobs
                if len(jobs_in_legend) < 5:
                    # Verify op_color_index is correct - it should match the lookup result
                    if op_job_id is not None and op_job_id in job_id_to_color_index:
                        expected_color_index = job_id_to_color_index[op_job_id]
                        if op_color_index != expected_color_index:
                            print(f"ERROR: Job '{job_label}' (job_id={op_job_id}) has color_index={op_color_index} but expected {expected_color_index}, original={original_color_index}, job_index={job_index}")
                        else:
                            print(f"Job '{job_label}' (job_id={op_job_id}, color_index={op_color_index}) -> color RGB={op_base_color}")
                    else:
                        print(f"Job '{job_label}' (job_id={op_job_id}, color_index={op_color_index}, job_index={job_index}) -> color RGB={op_base_color}")
                legend_handles.append(mpatches.Patch(facecolor=op_base_color, edgecolor='black', label=job_label))
                legend_labels.append(job_label)
                jobs_in_legend.add(legend_key)
            
            start_time = start_times[current_operation_index]
            machine = machine_assignments[current_operation_index]
            operation_duration = operation_runtimes[machine]
            # Use a subtle gradient for operations within the same job
            gradient_factor = color_gradients[min(operation_index, len(color_gradients) - 1)]
            operation_color = tuple(op_base_color * gradient_factor)

            # Draw the operation box
            rect = ax.broken_barh([(start_time, operation_duration)], 
                           (machine - 0.4, 0.8),
                           facecolors=operation_color,
                           edgecolor='black',
                           linewidth=0.5)
            
            # Add operation ID text on the box
            if current_operation_index < len(operation_ids_list) and operation_ids_list[current_operation_index] is not None:
                operation_id = operation_ids_list[current_operation_index]
                # Center the text in the box
                text_x = start_time + operation_duration / 2
                text_y = machine
                # Use white text if box is dark, black if light
                color_intensity = np.mean(operation_color)
                text_color = 'white' if color_intensity < 0.5 else 'black'
                ax.text(text_x, text_y, str(operation_id), 
                       ha='center', va='center', 
                       fontsize=7, fontweight='bold',
                       color=text_color)

            if operation_index > 0:
                prev_machine = machine_assignments[current_operation_index - 1]
                transfer_time = transfer_times[prev_machine][machine]
                if transfer_time > 0:  # Only plot if there's actual transfer time
                    ax.broken_barh([(start_time - transfer_time, transfer_time)], 
                                (prev_machine - 0.4, 0.8),
                                facecolors=transfer_color,
                                edgecolor='black',
                                linewidth=0.5)

            current_operation_index += 1
    
    # Add transfer time to legend if there are any transfers
    if np.any(transfer_times > 0):
        legend_handles.append(mpatches.Patch(facecolor=transfer_color, edgecolor='black', label='Transfer Time'))
        legend_labels.append('Transfer Time')

    # Labels and title
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(plot_title)

    # Add legend with job labels
    ax.legend(legend_handles, legend_labels, loc='upper right', title="Jobs", 
              bbox_to_anchor=(1.15, 1), framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Save the plot with dynamic filename
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
