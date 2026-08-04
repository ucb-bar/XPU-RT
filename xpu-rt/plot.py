import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# XPU-RT repo root /plots (this file lives in xpu-rt/)
REPO_PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


# Per-model-kind colormap assignments. Periodic instances of the same
# model (e.g. dronet0, dronet1, ..., dronet26) get distinct shades from
# the same matplotlib colormap so the schedule plot reads as "all blues
# are dronet, all oranges are yolov8" rather than 30 unrelated colors.
# Add new entries here when introducing a new model family.
KIND_TO_CMAP = {
    "dronet": "Blues",
    "yolov8_nano": "Oranges",
    "yolov8": "Oranges",
    "mlp": "Greens",
    "mobilenet": "Purples",
    "resnet50": "Reds",
    "tinyyolo": "YlOrBr",
}
# Fallback cmaps for unknown model kinds, in priority order. We avoid
# Greys here because the transfer-time bars are already gray/black.
_FALLBACK_CMAPS = ["Greens", "Purples", "Reds", "YlOrBr", "PuRd",
                   "BuGn", "OrRd", "GnBu"]


def _kind_from_job_name(name):
    """Return the model 'kind' for a job name. e.g. 'dronet0' -> 'dronet',
    'dronet10' -> 'dronet', 'yolov8_nano' -> 'yolov8_nano'. Strips the
    trailing periodic-instance suffix (digits at end of name)."""
    if not name:
        return "unknown"
    # Strip trailing digits (periodic instance index)
    return re.sub(r"\d+$", "", name)


def _build_family_colors(job_id_to_color_index, job_names_list, num_colors):
    """Build a base_colors list where color_index -> RGB triple, with
    instances of the same model kind drawn from the same colormap.

    Returns a list of length `num_colors`. Slots not covered by any
    job_id default to a neutral gray (so any caller-side fallback path
    that hits an unmapped index still gets a sensible color).
    """
    # Group color_index slots by model kind.
    kind_to_color_indices = defaultdict(list)
    for job_id, color_idx in job_id_to_color_index.items():
        name = job_names_list[job_id] if job_id < len(job_names_list) else None
        kind = _kind_from_job_name(name)
        kind_to_color_indices[kind].append(color_idx)

    # Assign a colormap to each kind. Known kinds use their declared
    # cmap; unknown kinds draw from _FALLBACK_CMAPS in deterministic
    # order (sorted by kind name) so reruns produce stable colors.
    used_cmaps = set(KIND_TO_CMAP.get(k) for k in kind_to_color_indices
                     if k in KIND_TO_CMAP)
    fallback_iter = iter(c for c in _FALLBACK_CMAPS if c not in used_cmaps)
    kind_to_cmap = {}
    for kind in sorted(kind_to_color_indices.keys()):
        if kind in KIND_TO_CMAP:
            kind_to_cmap[kind] = KIND_TO_CMAP[kind]
        else:
            kind_to_cmap[kind] = next(fallback_iter, "Greys")

    # Default unused slots to gray; loop fills in known slots.
    base_colors = [(0.5, 0.5, 0.5)] * num_colors

    # For each kind, pick distinct shades from its colormap. We avoid
    # the very-light end (<0.35) where the bar would blend into the
    # white grid background, and the very-dark end (>0.95) where the
    # text overlay would be unreadable. Single-instance kinds get a
    # mid-tone (0.7) so they don't look washed out.
    for kind, indices in kind_to_color_indices.items():
        cmap = plt.get_cmap(kind_to_cmap[kind])
        indices_sorted = sorted(indices)
        n = len(indices_sorted)
        if n == 1:
            shades = [0.7]
        else:
            shades = np.linspace(0.4, 0.95, n)
        for shade, color_idx in zip(shades, indices_sorted):
            if 0 <= color_idx < num_colors:
                base_colors[color_idx] = tuple(cmap(shade)[:3])

    return base_colors


def plot_optimization_schedule(durations, t, alpha, num_jobs, num_machines, machines, transfer_times, save_path="plots/schedule.png", plot_title="Schedule", workload=None, profile_hw=None):
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
    plot_dir = str(REPO_PLOTS_DIR)
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

    # Determine machine/combination assignments
    machine_assignments = np.argmax(alpha_values, axis=1)
    
    # Check if we're using machine combinations
    # IMPORTANT: Use transfer_times shape to detect, not len(machines), because
    # 'machines' parameter might be combination labels (3 elements) while transfer_times
    # is always indexed by actual machines (2 elements)
    num_actual_machines = transfer_times.shape[0]
    using_combinations = False
    machine_combinations = None
    if workload is not None and hasattr(workload, 'get_machine_combinations'):
        machine_combinations = workload.get_machine_combinations()
        # If we have more combinations than actual machines, we're using combinations
        if len(machine_combinations) > num_actual_machines:
            using_combinations = True
        # Also check if alpha has more columns than actual machines (most reliable indicator)
        elif alpha_values.shape[1] > num_actual_machines:
            using_combinations = True
            # If we detected via alpha shape but don't have combinations, try to get them
            if machine_combinations is None:
                machine_combinations = workload.get_machine_combinations()
    
    # Additional check: if alpha has more columns than actual machines, we must be using combinations
    if not using_combinations and alpha_values.shape[1] > num_actual_machines:
        using_combinations = True
        if workload is not None and hasattr(workload, 'get_machine_combinations'):
            machine_combinations = workload.get_machine_combinations()
    
    # Final safety check: always use combinations if alpha columns > actual machines
    # This is the most reliable indicator
    if alpha_values.shape[1] > num_actual_machines:
        using_combinations = True
        if machine_combinations is None and workload is not None and hasattr(workload, 'get_machine_combinations'):
            machine_combinations = workload.get_machine_combinations()
    
    # Get actual machine list from workload if available (for transfer time lookups)
    actual_machines = machines  # Default to what was passed
    if workload is not None and hasattr(workload, 'machines'):
        actual_machines = workload.machines

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
    
    # Family-aware color override: if we have job names, group instances of
    # the same model kind under the same colormap (Blues for dronet*,
    # Oranges for yolov8*, etc.). This makes schedules with many periodic
    # instances readable — without it, dronet0..dronet26 each get a
    # different unrelated tab20 color.
    if workload is not None and job_names_list and job_id_to_color_index:
        family_n = max(num_unique_job_ids, len(base_colors))
        base_colors = _build_family_colors(
            job_id_to_color_index, job_names_list, family_n)

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
            
            # Get operation duration - handle both machine and combination indices
            # Check if we're using combinations: use transfer_times shape (actual machines) not len(machines)
            # This is the most reliable check - if alpha has more columns than actual machines, we MUST be using combinations
            is_using_combos = (alpha_values.shape[1] > num_actual_machines) or using_combinations
            
            # Try to get duration using combinations if applicable
            if is_using_combos and workload is not None and current_operation_index < len(workload.operations):
                # Get machine combinations if not already set
                if machine_combinations is None and hasattr(workload, 'get_machine_combinations'):
                    machine_combinations = workload.get_machine_combinations()
                if machine_combinations is not None and hasattr(workload.operations[current_operation_index], 'get_duration_for_combination'):
                    try:
                        # machine is actually a combination index
                        operation_duration = workload.operations[current_operation_index].get_duration_for_combination(
                            machine, machine_combinations, actual_machines
                        )
                    except (ValueError, IndexError, AttributeError):
                        # Fallback to traditional indexing if combination lookup fails
                        if machine < len(operation_runtimes):
                            operation_duration = operation_runtimes[machine]
                        else:
                            operation_duration = operation_runtimes[0] if len(operation_runtimes) > 0 else 1.0
                else:
                    # Fallback: use machine index (shouldn't happen, but safe)
                    if machine < len(operation_runtimes):
                        operation_duration = operation_runtimes[machine]
                    else:
                        operation_duration = operation_runtimes[0] if len(operation_runtimes) > 0 else 1.0
            else:
                # Traditional machine-based indexing
                if machine < len(operation_runtimes):
                    operation_duration = operation_runtimes[machine]
                else:
                    # Safety fallback
                    operation_duration = operation_runtimes[0] if len(operation_runtimes) > 0 else 1.0
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
                # For transfer times, we need to map combination indices to machine indices
                # Use the same robust check as for duration lookup - use num_actual_machines (from transfer_times)
                is_using_combos_for_transfer = alpha_values.shape[1] > num_actual_machines
                
                # Debug: print once to see what's happening
                if current_operation_index == 1:
                    print(f"DEBUG TRANSFER: alpha_shape={alpha_values.shape}, num_actual_machines={num_actual_machines}, "
                          f"len(machines)={len(machines)}, alpha_cols={alpha_values.shape[1]}, "
                          f"is_using_combos={is_using_combos_for_transfer}, prev_machine={prev_machine}, machine={machine}, "
                          f"transfer_times_shape={transfer_times.shape}, "
                          f"machine_combinations={machine_combinations is not None}")
                
                if is_using_combos_for_transfer:
                    # Get machine combinations if not already set
                    if machine_combinations is None and workload is not None and hasattr(workload, 'get_machine_combinations'):
                        machine_combinations = workload.get_machine_combinations()
                    if machine_combinations is not None:
                        # Get first machine from each combination for transfer time lookup
                        prev_combo = machine_combinations[prev_machine]
                        curr_combo = machine_combinations[machine]
                        prev_machine_idx = actual_machines.index(prev_combo[0])
                        curr_machine_idx = actual_machines.index(curr_combo[0])
                        transfer_time = transfer_times[prev_machine_idx][curr_machine_idx]
                    else:
                        # Fallback: use combination indices directly (shouldn't happen)
                        if prev_machine < len(machines) and machine < len(machines):
                            transfer_time = transfer_times[prev_machine][machine]
                        else:
                            transfer_time = 0
                else:
                    # Traditional indexing - but check bounds first using actual machines
                    if prev_machine < num_actual_machines and machine < num_actual_machines:
                        transfer_time = transfer_times[prev_machine][machine]
                    else:
                        # This shouldn't happen - we're using combinations but didn't detect it
                        print(f"ERROR TRANSFER: prev_machine={prev_machine}, machine={machine} out of bounds for {num_actual_machines} actual machines")
                        print(f"  alpha_shape={alpha_values.shape}, alpha_cols={alpha_values.shape[1]}, num_actual_machines={num_actual_machines}")
                        print(f"  is_using_combos_for_transfer={is_using_combos_for_transfer}, using_combinations={using_combinations}")
                        # Force use of combinations - this is a safety fallback
                        if workload is not None and hasattr(workload, 'get_machine_combinations'):
                            machine_combinations = workload.get_machine_combinations()
                            if machine_combinations is not None and prev_machine < len(machine_combinations) and machine < len(machine_combinations):
                                prev_combo = machine_combinations[prev_machine]
                                curr_combo = machine_combinations[machine]
                                prev_machine_idx = actual_machines.index(prev_combo[0])
                                curr_machine_idx = actual_machines.index(curr_combo[0])
                                transfer_time = transfer_times[prev_machine_idx][curr_machine_idx]
                            else:
                                transfer_time = 0
                        else:
                            transfer_time = 0
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
    # If profile_hw is provided, prefix each machine label with its hardware
    # name (e.g. "gemmini_q31 (CPU_P#0)" instead of just "CPU_P#0"). Keeps
    # the abstract role visible while making the bitstream-level identity
    # immediately readable from the plot. profile_hw is keyed by CPU_P/CPU_E
    # — match by prefix so multi-hart machines (CPU_P#0, CPU_P#1) all pick
    # up the same hw name.
    if profile_hw:
        relabeled = []
        for m in machines:
            role = m.split("#")[0] if "#" in m else m
            hw = profile_hw.get(role)
            relabeled.append(f"{hw} ({m})" if hw else m)
        ax.set_yticklabels(relabeled)
        ax.set_ylabel("Cores (hardware)")
    else:
        ax.set_yticklabels(machines)
        ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(plot_title)

    # Add legend with job labels
    ax.legend(legend_handles, legend_labels, loc='upper right', title="Jobs", 
              bbox_to_anchor=(1.15, 1), framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Save the plot with dynamic filename.
    #
    # dpi is adaptive rather than a fixed 500. A 12x6 figure at dpi=500 is
    # 6000x3000 px BEFORE bbox_inches='tight' expands the canvas to include the
    # legend anchored outside the axes; on a large schedule (measured: 2629 ops
    # at contention B=4) FreeType then fails with
    # "raster overflow; error code 0x62" while rasterising a glyph, and the
    # whole run dies. That cost a sweep 14 of 45 cells -- the schedules had been
    # solved but the fixture is written AFTER this call, so a cosmetic plot
    # failure destroyed the scientific output.
    #
    # So: scale dpi down as the schedule grows, and retry at successively lower
    # dpi if the rasteriser still refuses. A lower-resolution Gantt is a fine
    # outcome; losing the schedule is not.
    print(f"Saving plot to {save_path}...")
    n_ops = len(durations) if durations is not None else 0
    dpi = 500 if n_ops < 800 else max(120, int(500 * (800.0 / n_ops) ** 0.5))
    last_err = None
    for attempt_dpi in (dpi, 200, 120, 80):
        try:
            plt.savefig(save_path, dpi=attempt_dpi, bbox_inches='tight')
            if attempt_dpi != dpi:
                print(f"  note: fell back to dpi={attempt_dpi} after a "
                      f"rasteriser failure at dpi={dpi}")
            last_err = None
            break
        except (RuntimeError, ValueError) as exc:
            last_err = exc
            print(f"  warning: savefig failed at dpi={attempt_dpi}: {exc}")
    if last_err is not None:
        # Re-raise so the caller can decide; run_xpurt_schedule.py treats a plot
        # failure as non-fatal precisely so the fixture still gets written.
        raise last_err
