from workload import Workload, Job, Operation, Window
import numpy as np
from typing import Tuple, Dict, List
from constants import NOT_SUPPORTED

def generate_syn_transfer_times(n_machines: int, max_transfer_time: int=500) -> np.ndarray:
    """
    Generates a symmetric matrix of transfer times between machines.
    """
    
    transfer_times = np.random.randint(0, max_transfer_time, (n_machines, n_machines))
    transfer_times = (transfer_times + transfer_times.T) / 2
    np.fill_diagonal(transfer_times, 0)
    return transfer_times

def create_sequential_job(operations: list[Operation]) -> Job:
    """
    From the list of operations, creates a job of sequentially dependent operations.
    Each operation depends on the previous one (single predecessor chain for backward compatibility).
    """
    
    for i in range(len(operations) - 1):
        operations[i].successor = operations[i+1]
        # Use add_predecessor to maintain list structure
        operations[i+1].add_predecessor(operations[i])
    
    return Job(operations)

def generate_syn_workload(n_operations: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Workload:
    """
    Generates a synthetic workload for the scheduling problem.

    Parameters:
    - n_operations: number of operations
    - n_machines: number of machines available
    - transfer_times: a matrix of transfer times between machines
    - processing_time_range: a tuple of the minimum and maximum processing times for each operation

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(processing_time_range[0], processing_time_range[1]) for _ in range(n_machines)]
        operations.append(Operation(processing_times))
    machines = [f'machine_{i}' for i in range(n_machines)]
    workload = Workload(operations, machines, transfer_times)
    return workload

def generate_syn_window(n_operations: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Window:
    """
    Generates a synthetic window for the scheduling problem.

    Parameters:
    - n_operations: number of operations
    - n_machines: number of machines available
    - transfer_times: a matrix of transfer times between machines

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(processing_time_range[0], processing_time_range[1]) for _ in range(n_machines)]
        operations.append(Operation(processing_times))
    machines = [f'machine_{i}' for i in range(n_machines)]
    expected_time = sum([np.mean(operation.get_durations()) for operation in operations])
    window = Window(expected_time, operations, machines, transfer_times)
    return window

def create_syn_sequential_workload(n_jobs: int, n_operations_per_job: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Workload:
    # Create a workload
    machines = [f'machine_{i}' for i in range(n_machines)]

    operations = [[] for _ in range(n_jobs)]
    for i in range(n_jobs):
        for _ in range(n_operations_per_job):
            processing_times = [np.random.randint(50, 150) for _ in range(n_operations_per_job)]
            operations[i].append(Operation(processing_times))

    jobs = [create_sequential_job(ops) for ops in operations]

    workload_operations = []
    for job in jobs:
        workload_operations.extend(job.get_operations())
    workload = Workload(workload_operations, machines, transfer_times)
    return workload

def create_workload_from_dependencies(
    dispatch_data: Dict,
    processing_times: Dict[str, List[float]],
    machines: List[str],
    transfer_times: np.ndarray
) -> Workload:
    """
    Creates a workload from a dependency graph structure (like dronet_dispatch_deps.json).
    
    Parameters:
    - dispatch_data: Dictionary with 'dispatches' key containing dispatch information
    - processing_times: Dictionary mapping dispatch IDs to list of processing times for each machine
    - machines: List of machine names
    - transfer_times: Matrix of transfer times between machines
    
    Returns:
    - Workload object with operations linked according to dependencies
    """
    dispatches = dispatch_data.get('dispatches', {})
    
    # Create Operation objects for each dispatch
    operations_map = {}
    # Determine job_id: operations with no dependencies within this workload start a new job
    # For a single workload, all operations belong to job_id 0
    job_id = 0
    
    for dispatch_name, dispatch_info in dispatches.items():
        # Get processing times for this dispatch (check by name first, then by ID for backward compatibility)
        if dispatch_name in processing_times:
            proc_times = processing_times[dispatch_name]
        else:
            dispatch_id = dispatch_info.get('id', dispatch_name)
            if dispatch_id in processing_times:
                proc_times = processing_times[dispatch_id]
            else:
                # Generate random processing times if not provided
                proc_times = [np.random.randint(50, 150) for _ in machines]
        
        # Extract ID and name from dispatch info
        operation_id = dispatch_info.get('id', None)
        operation_name = dispatch_name
        
        operations_map[dispatch_name] = Operation(
            proc_times, 
            operation_id=operation_id,
            operation_name=operation_name,
            job_id=job_id  # All operations in a single workload belong to the same job
        )
    
    # Set up predecessor relationships
    for dispatch_name, dispatch_info in dispatches.items():
        dependencies = dispatch_info.get('dependencies', [])
        operation = operations_map[dispatch_name]
        
        for dep_name in dependencies:
            if dep_name in operations_map:
                operation.add_predecessor(operations_map[dep_name])
    
    # Create list of operations in order (operations with no dependencies first)
    operations = list(operations_map.values())
    
    # Determine job names: operations with no predecessors start a new job
    job_names = []
    seen_prefixes = set()
    
    for operation in operations:
        if not operation.predecessors:
            # Use operation name to extract job name
            op_name = operation.operation_name or f"dispatch_{len(job_names)}"
            # Extract prefix if it exists (e.g., "dronet_dispatch_0" -> "dronet")
            if '_' in op_name:
                parts = op_name.split('_')
                if len(parts) >= 2:
                    # Take the prefix part (e.g., "dronet" from "dronet_dispatch_0")
                    prefix = parts[0]
                    # Capitalize for better display
                    job_name = prefix.capitalize()
                    # Only add if we haven't seen this prefix yet (avoid duplicates)
                    if job_name not in seen_prefixes:
                        job_names.append(job_name)
                        seen_prefixes.add(job_name)
                else:
                    if op_name not in seen_prefixes:
                        job_names.append(op_name.capitalize())
                        seen_prefixes.add(op_name)
            else:
                if op_name not in seen_prefixes:
                    job_names.append(op_name.capitalize())
                    seen_prefixes.add(op_name)
    
    # If no job names found, create default ones
    if not job_names:
        num_jobs = sum(1 for op in operations if not op.predecessors)
        job_names = [f"Job {i}" for i in range(num_jobs)]
    
    return Workload(operations, machines, transfer_times, job_names=job_names)
