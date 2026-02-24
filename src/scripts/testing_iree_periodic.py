"""
Test script for scheduling IREE dispatch graphs from hierarchical network dependencies.
Parses top-level network dependency JSON files and schedules them on CPU_P (performant) 
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
from workload_factory import create_workload_from_network_hierarchy
from scheduler import schedule
import plot

def load_networks_graph(json_path: str) -> dict:
    """Load a network dependencies JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def schedule_iree_networks(networks_json_path: str = None, solver_verbosity: int = 0, time_limit: float = None, random_seed: int | None = 0):
    """
    Main function to schedule networks from a hierarchical network dependencies JSON file.
    
    Parameters:
    - networks_json_path: Path to the top-level networks dependencies JSON file.
                          If None, uses the default networks_deps.json file.
    - solver_verbosity: MOSEK solver verbosity level (0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress).
    - time_limit: Maximum optimization time in seconds. MOSEK will return the best solution found within this time limit.
    """
    # Get script directory and repo base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_base_path = os.path.join(script_dir, '..', '..')
    
    # Use default networks JSON if not provided
    if networks_json_path is None:
        networks_json_path = os.path.join(
            script_dir,
            '..',
            'data',
            'toplevel',
            'networks_periodic.json'
        )
    
    # Resolve to absolute path
    if not os.path.isabs(networks_json_path):
        networks_json_path = os.path.join(repo_base_path, networks_json_path)
    
    print("=" * 60)
    print("Loading network hierarchy from JSON...")
    print("=" * 60)
    print(f"\nLoading networks from: {networks_json_path}")
    
    # Load network dependencies JSON
    networks_data = load_networks_graph(networks_json_path)
    
    # Print network information
    networks = networks_data.get('networks', {})
    edges = networks_data.get('edges', [])
    print(f"\nFound {len(networks)} networks:")
    for network_id, network_info in networks.items():
        print(f"  - {network_id} (id: {network_info.get('id')}, identifier: {network_info.get('identifier')})")
    
    print(f"\nFound {len(edges)} network-level dependencies:")
    for edge in edges:
        print(f"  - {edge.get('from')} → {edge.get('to')}")
    
    # Define machines (dual-core device)
    machines = ['CPU_P', 'CPU_E']
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((2, 2))
    
    # Create workload from network hierarchy
    print(f"\nCreating workload from network hierarchy...")
    combined_workload = create_workload_from_network_hierarchy(
        networks_data=networks_data,
        repo_base_path=repo_base_path,
        machines=machines,
        transfer_times=transfer_times,
        p_core_speedup=1.5,
        random_seed=random_seed,
    )
    
    print(f"\nWorkload created successfully!")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")
    print(f"  Job names: {combined_workload.job_names}")
    
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
    result = schedule(combined_workload, solver_verbosity=solver_verbosity, time_limit=time_limit)
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
    
    # Count operations per network (group by job_id)
    network_stats = {}
    for op_idx, op in enumerate(combined_workload.operations):
        job_id = op.job_id
        if job_id not in network_stats:
            network_stats[job_id] = {'cpu_p': 0, 'cpu_e': 0, 'name': combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}"}
        
        # Find which machine this operation is assigned to
        assigned_machine = np.argmax(alpha[op_idx])
        if assigned_machine == 0:
            network_stats[job_id]['cpu_p'] += 1
        else:
            network_stats[job_id]['cpu_e'] += 1
    
    print(f"\nPer-network core assignments:")
    for job_id in sorted(network_stats.keys()):
        stats = network_stats[job_id]
        print(f"  {stats['name'].capitalize()}: CPU_P={stats['cpu_p']}, CPU_E={stats['cpu_e']}")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create title showing the networks
    network_names = [combined_workload.job_names[i] if i < len(combined_workload.job_names) else f"Job {i}" 
                     for i in sorted(set(op.job_id for op in combined_workload.operations))]
    title_networks = " + ".join([name.capitalize() for name in network_names])
    
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path="plots/iree_combined_schedule_period.png",
        plot_title=f"{title_networks} Schedule on Dual-Core Device (Periodic)",
        workload=combined_workload
    )
    
    print(f"\nPlot saved to plots/iree_combined_schedule_period.png")
    
    return combined_workload, t, alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE networks from hierarchical network dependencies JSON file"
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default=None,
        help="Path to the top-level networks dependencies JSON file (default: src/data/toplevel/networks_periodic.json)"
    )
    parser.add_argument(
        "--solver-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="MOSEK solver verbosity level: 0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Maximum optimization time in seconds. MOSEK will return the best solution found within this time limit.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for synthetic runtime generation (default: 0 for reproducible results). Use -1 for nondeterministic.",
    )
    args = parser.parse_args()

    seed = None if args.random_seed is not None and args.random_seed < 0 else args.random_seed
    
    schedule_iree_networks(
        networks_json_path=args.networks_json,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
    )

