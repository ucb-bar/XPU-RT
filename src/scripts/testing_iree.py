"""
Test script for scheduling IREE dispatch graphs (dronet and MLP) on a dual-core device.
Parses dispatch dependency JSON files and schedules them in parallel on CPU_P (performant) 
and CPU_E (efficient) cores. CPU_P is 1.5x faster than CPU_E.
"""

import sys
import os
import json
import argparse
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_dependencies
from scheduler import schedule
import plot

def load_dispatch_graph(json_path: str) -> dict:
    """Load a dispatch dependencies JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_workload_from_json(json_path: str, name_prefix: str = "") -> tuple:
    """
    Create a workload from a dispatch dependencies JSON file.
    
    Parameters:
    - json_path: Path to the dispatch_deps.json file
    - name_prefix: Optional prefix to add to dispatch names (to avoid conflicts when combining)
    
    Returns:
    - Tuple of (Workload object, job_name) where job_name is derived from filename
    """
    # Load the JSON file
    dispatch_data = load_dispatch_graph(json_path)
    
    # Get dispatches
    original_dispatches = dispatch_data.get('dispatches', {})
    
    # Update dispatch names in the data structure if prefix is provided
    if name_prefix:
        prefixed_dispatches = {}
        for dispatch_name, dispatch_info in original_dispatches.items():
            prefixed_name = f"{name_prefix}{dispatch_name}"
            prefixed_info = dispatch_info.copy()
            # Update dependencies to use prefixed names
            if 'dependencies' in prefixed_info:
                prefixed_info['dependencies'] = [
                    f"{name_prefix}{dep}" if dep in original_dispatches else dep
                    for dep in prefixed_info['dependencies']
                ]
            prefixed_dispatches[prefixed_name] = prefixed_info
        dispatch_data = {'dispatches': prefixed_dispatches}
        dispatches = prefixed_dispatches
    else:
        dispatches = original_dispatches
    
    # Generate processing times for dual-core device
    # Map dispatch names to processing times (the function expects names)
    # Use synthetic P-core runtimes in a similar ballpark to profiled data (~5 ms)
    processing_times_by_name = {}
    for dispatch_name, dispatch_info in dispatches.items():
        # Generate random P-core time in milliseconds (2–10 ms)
        p_ms_synth = float(np.random.uniform(2.0, 10.0))
        cpu_p_time = p_ms_synth
        cpu_e_time = p_ms_synth * 1.5  # CPU_P is 1.5x faster than CPU_E
        processing_times_by_name[dispatch_name] = [cpu_p_time, cpu_e_time]
    
    # Define machines (dual-core device)
    machines = ['CPU_P', 'CPU_E']
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))
    
    # Create workload from dependencies
    workload = create_workload_from_dependencies(
        dispatch_data=dispatch_data,
        processing_times=processing_times_by_name,
        machines=machines,
        transfer_times=transfer_times
    )
    
    # Extract job name from filename (e.g., "dronet_dispatch_deps.json" -> "dronet")
    filename = os.path.basename(json_path)
    job_name = filename.replace('_dispatch_deps.json', '').replace('.json', '')
    if name_prefix:
        # If prefix was added, use it as job name
        job_name = name_prefix.rstrip('_')
    
    return workload, job_name

def combine_workloads(workloads: list, job_names: list = None, job_id_mapping: list = None) -> Workload:
    """
    Combine multiple workloads into a single workload.
    Each workload becomes a separate job, preserving job_id assignments.
    
    Parameters:
    - workloads: List of Workload objects to combine
    - job_names: Optional list of job names corresponding to workloads
    - job_id_mapping: Optional list mapping workload index to job_id. 
                      If None, each workload gets a unique job_id based on its index.
                      If provided, workloads with the same job_id will share the same color.
    
    Returns:
    - Combined Workload object with job names
    """
    if not workloads:
        raise ValueError("At least one workload must be provided")
    
    # All workloads should have the same machines and transfer times
    machines = workloads[0].machines
    transfer_times = workloads[0].get_transfer_times()
    
    # Combine all operations
    all_operations = []
    combined_job_names = []
    
    for i, workload in enumerate(workloads):
        # Get job name for this workload
        workload_job_name = None
        if job_names and i < len(job_names):
            workload_job_name = job_names[i]
        elif hasattr(workload, 'job_names') and workload.job_names:
            # Use first job name from workload if available
            workload_job_name = workload.job_names[0] if workload.job_names else None
        
        # Assign job_id to all operations in this workload
        # Use job_id_mapping if provided, otherwise use index as job_id
        if job_id_mapping and i < len(job_id_mapping):
            job_id = job_id_mapping[i]
        else:
            job_id = i  # Use index as job_id
        
        for op in workload.operations:
            # Set explicit job_id for this operation
            op.job_id = job_id
            all_operations.append(op)
            
            # Track job names: add name when we encounter first operation of this job
            # If multiple workloads share the same job_id, use the first name encountered
            if op.job_id == job_id:
                # Extend list if needed
                while len(combined_job_names) <= job_id:
                    combined_job_names.append(None)
                # Only set name if not already set (first workload with this job_id wins)
                if combined_job_names[job_id] is None:
                    if workload_job_name:
                        combined_job_names[job_id] = workload_job_name
                    else:
                        combined_job_names[job_id] = f"Job {job_id}"
    
    # Remove None entries and ensure we have names for all jobs
    final_job_names = []
    for j in range(len(combined_job_names)):
        if j < len(combined_job_names) and combined_job_names[j]:
            final_job_names.append(combined_job_names[j])
        else:
            final_job_names.append(f"Job {j}")
    
    # Create combined workload
    combined_workload = Workload(all_operations, machines, transfer_times, job_names=final_job_names)
    
    return combined_workload

def add_dependency(source_workload: Workload, target_workload: Workload):
    """
    Make target workload's first operations depend on source workload's last operations.
    Source workload's last operations are those that are not predecessors of any other operation.
    
    Parameters:
    - source_workload: Workload that should complete first
    - target_workload: Workload that should start after source completes
    """
    # Find last operations in source (operations that are not predecessors of any other operation in source)
    source_last_ops = []
    source_ops_set = set(source_workload.operations)
    
    for op in source_workload.operations:
        # Check if this operation is a predecessor of any other operation in source
        is_predecessor = False
        for other_op in source_workload.operations:
            if op in other_op.predecessors:
                is_predecessor = True
                break
        if not is_predecessor:
            source_last_ops.append(op)
    
    # If no explicit last operations found, use operations with no successors
    # (operations that aren't predecessors of anything)
    if not source_last_ops:
        # Use all operations as potential last ops (fallback)
        source_last_ops = source_workload.operations
    
    # Find first operations in target (operations with no predecessors)
    target_first_ops = [op for op in target_workload.operations if not op.predecessors]
    
    # If no first operations found, use the first operation
    if not target_first_ops:
        target_first_ops = [target_workload.operations[0]] if target_workload.operations else []
    
    # Add dependencies: each target first operation depends on all source last operations
    for target_op in target_first_ops:
        for source_op in source_last_ops:
            target_op.add_predecessor(source_op)

def schedule_iree_networks(use_glpdepth=False):
    """
    Main function to schedule fast/glpdepth, dronet (depends on fast/glpdepth), and 5 MLP instances on dual-core device.
    Each MLP instance depends on the previous one (MLP0 → MLP1 → MLP2 → MLP3 → MLP4).
    
    Parameters:
    - use_glpdepth: If True, use glpdepth instead of fast as the first network
    """
    # Path to JSON files (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(
        script_dir, 
        '..', 
        '..', 
        'merlin', 
        'samples', 
        'robotic-NN', 
        'pytorch_workload', 
        'computation_graph'
    )
    
    # Select first network based on argument
    if use_glpdepth:
        first_network_name = "glpdepth"
        first_network_file = "glpdepth_dispatch_deps.json"
        first_network_prefix = "glpdepth_"
    else:
        first_network_name = "fast"
        first_network_file = "fast_dispatch_deps.json"
        first_network_prefix = "fast_"
    
    first_network_path = os.path.join(base_path, first_network_file)
    dronet_path = os.path.join(base_path, 'dronet_dispatch_deps.json')
    mlp_path = os.path.join(base_path, 'mlp_dispatch_deps.json')
    
    print("=" * 60)
    print("Loading dispatch graphs...")
    print("=" * 60)
    
    # Create workloads from JSON files
    print(f"\n1. Loading {first_network_name} dispatch graph from: {first_network_path}")
    first_network_workload, first_network_job_name = create_workload_from_json(
        first_network_path, name_prefix=first_network_prefix
    )
    print(f"   Created {first_network_job_name} workload with {len(first_network_workload.operations)} operations")
    
    print(f"\n2. Loading dronet dispatch graph from: {dronet_path}")
    dronet_workload, dronet_job_name = create_workload_from_json(dronet_path, name_prefix="dronet_")
    print(f"   Created {dronet_job_name} workload with {len(dronet_workload.operations)} operations")
    
    # Create 5 MLP workloads, each with a unique prefix
    print(f"\n3. Loading MLP dispatch graph (5 instances)...")
    mlp_workloads = []
    mlp_job_names = []
    for i in range(5):
        mlp_prefix = f"mlp{i}_"
        mlp_workload, mlp_job_name = create_workload_from_json(mlp_path, name_prefix=mlp_prefix)
        mlp_job_name = f"mlp{i}"  # Use numbered name
        mlp_workloads.append(mlp_workload)
        mlp_job_names.append(mlp_job_name)
        print(f"   Created {mlp_job_name} workload with {len(mlp_workload.operations)} operations")
    
    # Make dronet depend on first network
    print(f"\n4. Adding dependency: {dronet_job_name} depends on {first_network_job_name}...")
    add_dependency(first_network_workload, dronet_workload)
    
    # Make each MLP instance depend on the previous one (MLP instances are independent of first network/Dronet)
    first_network_label = first_network_name.capitalize()
    print(f"\n5. Adding dependencies between MLP instances (MLPs are independent of {first_network_label}/Dronet)...")
    for i in range(1, 5):
        print(f"   {mlp_job_names[i]} depends on {mlp_job_names[i-1]}...")
        add_dependency(mlp_workloads[i-1], mlp_workloads[i])
    
    # Combine workloads
    # First network and Dronet form one dependency chain: First → Dronet
    # MLP instances form their own independent chain: MLP0 → MLP1 → MLP2 → MLP3 → MLP4
    # These two chains can run in parallel (after first network completes, Dronet and MLP0 can both start)
    print(f"\n6. Combining workloads...")
    print(f"   Dependency chains:")
    print(f"     Chain 1: {first_network_job_name} → {dronet_job_name}")
    print(f"     Chain 2: {' → '.join(mlp_job_names)}")
    print(f"   (Chains are independent and can run in parallel)")
    all_workloads = [first_network_workload, dronet_workload] + mlp_workloads
    all_job_names = [first_network_job_name, dronet_job_name] + mlp_job_names
    
    # Create job_id mapping: First network=0, Dronet=1, all MLPs=2 (same color for all MLPs)
    job_id_mapping = [0, 1] + [2] * len(mlp_workloads)  # All MLPs get job_id=2
    
    # Use a single name "MLP" for all MLP instances in the legend
    all_job_names_for_legend = [first_network_job_name, dronet_job_name] + ["MLP"] * len(mlp_workloads)
    
    combined_workload = combine_workloads(all_workloads, job_names=all_job_names_for_legend, job_id_mapping=job_id_mapping)
    
    print(f"\nCombined workload statistics:")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")
    
    # Print some statistics
    operations_with_multiple_predecessors = [
        op for op in combined_workload.operations if len(op.predecessors) > 1
    ]
    print(f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")
    
    # Count independent jobs (operations with no predecessors)
    independent_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    print(f"  Independent jobs (can run in parallel): {independent_jobs}")
    
    # Schedule the combined workload
    print("\n" + "=" * 60)
    print("Scheduling combined workload...")
    print("=" * 60)
    result = schedule(combined_workload)
    t, alpha, _, _ = result  # Always returns 4 values now
    
    # Calculate makespan
    makespan = max(t[i] + combined_workload.operations[i].get_durations()[np.argmax(alpha[i])] 
                   for i in range(len(combined_workload.operations)))
    
    print(f"\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each core
    cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
    cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)
    
    print(f"\nCore assignments:")
    print(f"  CPU_P (performant): {cpu_p_count} operations")
    print(f"  CPU_E (efficient): {cpu_e_count} operations")
    
    # Count operations per network (based on order: first network, then dronet, then MLP instances)
    current_idx = 0
    first_network_indices = list(range(current_idx, current_idx + len(first_network_workload.operations)))
    current_idx += len(first_network_workload.operations)
    dronet_indices = list(range(current_idx, current_idx + len(dronet_workload.operations)))
    current_idx += len(dronet_workload.operations)
    
    # Calculate indices for each MLP instance
    mlp_indices_list = []
    for mlp_workload in mlp_workloads:
        mlp_indices = list(range(current_idx, current_idx + len(mlp_workload.operations)))
        mlp_indices_list.append(mlp_indices)
        current_idx += len(mlp_workload.operations)
    
    first_network_cpu_p = sum(1 for i in first_network_indices if np.argmax(alpha[i]) == 0)
    first_network_cpu_e = sum(1 for i in first_network_indices if np.argmax(alpha[i]) == 1)
    dronet_cpu_p = sum(1 for i in dronet_indices if np.argmax(alpha[i]) == 0)
    dronet_cpu_e = sum(1 for i in dronet_indices if np.argmax(alpha[i]) == 1)
    
    print(f"\nPer-network core assignments:")
    print(f"  {first_network_job_name.capitalize()}: CPU_P={first_network_cpu_p}, CPU_E={first_network_cpu_e}")
    print(f"  {dronet_job_name.capitalize()}: CPU_P={dronet_cpu_p}, CPU_E={dronet_cpu_e}")
    for i, mlp_indices in enumerate(mlp_indices_list):
        mlp_cpu_p = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 0)
        mlp_cpu_e = sum(1 for idx in mlp_indices if np.argmax(alpha[idx]) == 1)
        print(f"  {mlp_job_names[i].capitalize()}: CPU_P={mlp_cpu_p}, CPU_E={mlp_cpu_e}")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create title showing the dependency chain
    mlp_chain = " → ".join([name.capitalize() for name in mlp_job_names])
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path="plots/iree_combined_schedule.png",
        plot_title=f"{first_network_job_name.capitalize()} → {dronet_job_name.capitalize()} + {mlp_chain} Schedule on Dual-Core Device",
        workload=combined_workload
    )
    
    print(f"\nPlot saved to plots/iree_combined_schedule.png")
    
    return combined_workload, t, alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE dispatch graphs (fast/glpdepth, dronet, and MLP) on a dual-core device"
    )
    parser.add_argument(
        "--use-glpdepth",
        action="store_true",
        help="Use glpdepth instead of fast as the first network (default: use fast)"
    )
    args = parser.parse_args()
    
    schedule_iree_networks(use_glpdepth=args.use_glpdepth)

