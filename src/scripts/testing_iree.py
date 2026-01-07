src/scripts/testing.py"""
Test script for scheduling IREE dispatch graph (dronet) on a dual-core device.
Parses dronet_dispatch_deps.json and schedules it on CPU_P (performant) and CPU_E (efficient) cores.
CPU_P is 1.5x faster than CPU_E.
"""

import sys
import os
import json
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_dependencies
from scheduler import schedule
import plot

def load_dronet_dispatch_graph(json_path: str) -> dict:
    """Load the dronet dispatch dependencies JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_dronet_workload(json_path: str) -> Workload:
    """
    Create a workload from the dronet dispatch dependencies JSON file.
    
    Parameters:
    - json_path: Path to the dronet_dispatch_deps.json file
    
    Returns:
    - Workload object with operations linked according to dependencies
    """
    # Load the JSON file
    dispatch_data = load_dronet_dispatch_graph(json_path)
    
    # Get dispatches
    dispatches = dispatch_data.get('dispatches', {})
    num_dispatches = len(dispatches)
    
    # Generate processing times for dual-core device
    # Map dispatch names to processing times (the function expects names)
    processing_times_by_name = {}
    for dispatch_name, dispatch_info in dispatches.items():
        # Generate random processing time for this dispatch
        base_time = np.random.randint(50, 200)
        cpu_e_time = float(base_time)
        cpu_p_time = float(base_time / 1.5)  # CPU_P is 1.5x faster
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
    
    return workload

def schedule_dronet():
    """
    Main function to schedule the dronet network on dual-core device.
    """
    # Path to the JSON file (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(
        script_dir, 
        '..', 
        '..', 
        'merlin', 
        'samples', 
        'robotic-NN', 
        'pytorch_workload', 
        'computation_graph', 
        'dronet_dispatch_deps.json'
    )
    
    print(f"Loading dronet dispatch graph from: {json_path}")
    
    # Create workload from JSON
    workload = create_dronet_workload(json_path)
    
    print(f"Created workload with {len(workload.operations)} operations")
    print(f"Machines: {workload.machines}")
    
    # Print some statistics
    operations_with_multiple_predecessors = [
        op for op in workload.operations if len(op.predecessors) > 1
    ]
    print(f"Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")
    for op in operations_with_multiple_predecessors[:5]:  # Show first 5
        print(f"  Operation has {len(op.predecessors)} predecessors")
    
    # Schedule the workload
    print("\nScheduling workload...")
    t, alpha = schedule(workload)
    
    # Calculate makespan
    makespan = max(t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] 
                   for i in range(len(workload.operations)))
    
    print(f"\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each core
    cpu_p_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 0)
    cpu_e_count = sum(1 for i in range(len(alpha)) if np.argmax(alpha[i]) == 1)
    
    print(f"\nCore assignments:")
    print(f"  CPU_P (performant): {cpu_p_count} operations")
    print(f"  CPU_E (efficient): {cpu_e_count} operations")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in workload.operations if not op.predecessors)
    
    plot.plot_optimization_schedule(
        workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(workload.machines),
        workload.machines,
        workload.get_transfer_times(),
        save_path="plots/dronet_dual_core_schedule.png",
        plot_title="Dronet Network Schedule on Dual-Core Device"
    )
    
    print(f"\nPlot saved to plots/dronet_dual_core_schedule.png")
    
    return workload, t, alpha

if __name__ == "__main__":
    schedule_dronet()

