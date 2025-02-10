from typing import List, Dict, Tuple
import numpy as np
from workload import Job, Operation



def greedy(jobs: List[Job], machines: List[str], job_durations: Dict[Tuple[int, str], float], transfer_times: Dict[str, Dict[str, float]]) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    """
    Greedy scheduling algorithm that minimizes the earliest end time for each workload,
    while respecting dependencies between operations within a job.
    
    Parameters:
    - jobs: List of jobs (each job consists of operations)
    - machines: List of available machines
    - job_durations: Dictionary mapping (job_index, machine) to processing time
    
    Returns:
    - start_times: Dictionary mapping (job_idx, op_idx) to its start time
    - machine_allocations: Dictionary mapping (job_idx, op_idx) to its assigned machine
    """
    
    available_jobs = set((job_idx, op_idx, op) for job_idx, job in enumerate(jobs) for op_idx, op in enumerate(job.get_operations()))  # Jobs to be scheduled
    scheduled_jobs = {}  # Operations with assigned start times
    
    machine_available_time = {machine: 0 for machine in machines}  # Track machine availability
    operation_start_times = {}
    operation_allocations = {}
    
    while available_jobs:
        # Find an operation that has no unfinished predecessors
        for job_idx, op_idx, operation in list(available_jobs):
            predecessor = operation.get_predecessor()
            if predecessor:
                pred_op_idx = op_idx - 1  # Since jobs are sequential
                if (job_idx, pred_op_idx) not in scheduled_jobs:
                    continue  # Can't schedule yet
                predecessor_end_time = scheduled_jobs[(job_idx, pred_op_idx)] + job_durations.get((job_idx, operation_allocations[(job_idx, pred_op_idx)]), 0)
            else:
                predecessor_end_time = 0  # No predecessor, can start at time 0
            
            # Find the machine with the minimum end time
            best_machine = None
            best_end_time = float('inf')
            best_start_time = 0
            
            for machine in machines:
                start_time = max(machine_available_time[machine], predecessor_end_time)
                operation_time = job_durations.get((job_idx, machine), float('inf'))
                end_time = start_time + operation_time
                
                if end_time < best_end_time:
                    best_end_time = end_time
                    best_start_time = start_time
                    best_machine = machine
            
            # Assign operation to the chosen machine
            operation_start_times[(job_idx, op_idx)] = best_start_time
            operation_allocations[(job_idx, op_idx)] = best_machine
            machine_available_time[best_machine] = best_end_time
            
            # Move operation to scheduled
            scheduled_jobs[(job_idx, op_idx)] = best_end_time
            available_jobs.remove((job_idx, op_idx, operation))
            break  # Restart loop to reevaluate available jobs
    
    return operation_start_times, operation_allocations

# def greedy(jobs: List[Job], machines: List[str], job_durations: Dict[Tuple[int, str], float], transfer_times: np.ndarray) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
#     """
#     Greedy scheduling algorithm that minimizes the earliest end time for each workload,
#     while respecting dependencies between operations within a job and considering transfer times.
#     """
#     available_jobs = set((job_idx, op_idx, op) for job_idx, job in enumerate(jobs) for op_idx, op in enumerate(job.get_operations()))
#     scheduled_jobs = {}
#     machine_available_time = {machine: 0 for machine in machines}
#     operation_start_times = {}
#     operation_allocations = {}
    
#     while available_jobs:
#         for job_idx, op_idx, operation in list(available_jobs):
#             predecessor = operation.get_predecessor()
#             if predecessor:
#                 pred_op_idx = op_idx - 1
#                 if (job_idx, pred_op_idx) not in scheduled_jobs:
#                     continue
#                 prev_machine = operation_allocations[(job_idx, pred_op_idx)]
#                 transfer_time = transfer_times[machines.index(prev_machine)][machines.index(machines[0])]
#                 predecessor_end_time = scheduled_jobs[(job_idx, pred_op_idx)] + job_durations.get((job_idx, prev_machine), 0) + transfer_time
#             else:
#                 predecessor_end_time = 0
#             best_machine = None
#             best_end_time = float('inf')
#             best_start_time = 0
#             for machine in machines:
#                 start_time = max(machine_available_time[machine], predecessor_end_time)
#                 operation_time = job_durations.get((job_idx, machine), float('inf'))
#                 end_time = start_time + operation_time
#                 if end_time < best_end_time:
#                     best_end_time = end_time
#                     best_start_time = start_time
#                     best_machine = machine
#             if predecessor:
#                 transfer_time = transfer_times[machines.index(prev_machine)][machines.index(best_machine)]
#                 best_start_time += transfer_time
#                 best_end_time += transfer_time
#             operation_start_times[(job_idx, op_idx)] = best_start_time
#             operation_allocations[(job_idx, op_idx)] = best_machine
#             machine_available_time[best_machine] = best_end_time
#             scheduled_jobs[(job_idx, op_idx)] = best_end_time
#             available_jobs.remove((job_idx, op_idx, operation))
#             break
#     return operation_start_times, operation_allocations
