from workload import Workload, Job, Operation, Window
import numpy as np
from typing import Tuple
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
    From the list of operations, creates a job of sequentially dependent operations
    """
    
    for i in range(len(operations) - 1):
        operations[i].successor = operations[i+1]
        operations[i+1].predecessor = operations[i]
    
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