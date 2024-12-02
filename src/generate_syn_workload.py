from workload import Workload, Operation
import numpy as np

def generate_syn_workload(n_operations: int, n_machines: int) -> Workload:
    """
    Generates a synthetic workload for the scheduling problem.

    Parameters:
    - n_jobs: number of jobs in the workload
    - n_machines: number of machines available
    - n_operations_per_job: number of operations per job

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(50, 1000) for _ in range(n_machines)]
        operations.append(Operation(processing_times))

    workload = Workload(operations, np.zeros(n_machines))
    return workload