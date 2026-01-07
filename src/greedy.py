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
            predecessors = operation.get_predecessors()
            if predecessors:
                # Check if all predecessors are scheduled
                all_predecessors_scheduled = True
                max_predecessor_end_time = 0
                for pred in predecessors:
                    # Find predecessor's index (assuming sequential jobs)
                    pred_op_idx = None
                    for check_job_idx, check_job in enumerate(jobs):
                        if pred in check_job.get_operations():
                            pred_op_idx = check_job.get_operations().index(pred)
                            if (check_job_idx, pred_op_idx) not in scheduled_jobs:
                                all_predecessors_scheduled = False
                                break
                            else:
                                pred_end_time = scheduled_jobs[(check_job_idx, pred_op_idx)] + job_durations.get((check_job_idx, operation_allocations.get((check_job_idx, pred_op_idx), machines[0])), 0)
                                max_predecessor_end_time = max(max_predecessor_end_time, pred_end_time)
                    if not all_predecessors_scheduled:
                        break
                
                if not all_predecessors_scheduled:
                    continue  # Can't schedule yet - waiting for predecessors
                predecessor_end_time = max_predecessor_end_time
            else:
                predecessor_end_time = 0  # No predecessors, can start at time 0
            
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
