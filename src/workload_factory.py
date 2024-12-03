from workload import Workload, Job, Operation, Window
import numpy as np
from constants import NOT_SUPPORTED

def create_sequential_job(operations: list[Operation]) -> Job:
    """
    From the list of operations, creates a job of sequentially dependent operations
    """
    
    for i in range(len(operations) - 1):
        operations[i].successor = operations[i+1]
        operations[i+1].predecessor = operations[i]
    
    return Job(operations)

def generate_syn_workload(n_operations: int, n_machines: int) -> Workload:
    """
    Generates a synthetic workload for the scheduling problem.

    Parameters:
    - n_operations: number of operations
    - n_machines: number of machines available

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(50, 150) for _ in range(n_machines)]
        operations.append(Operation(processing_times))

    workload = Workload(operations, machines=[f'machine_{i}' for i in range(n_machines)])
    return workload

def generate_syn_window():
    """
    Creates a synthetic window.
    Contains 2 jobs with 3 operations each.
    """

    operations1 = []
    operations2 = []

    for _ in range(3):
        processing_times = [np.random.randint(50, 1000) for _ in range(3)]
        operations1.append(Operation(processing_times))

    for _ in range(3):
        processing_times = [np.random.randint(50, 1000) for _ in range(3)]
        operations2.append(Operation(processing_times))
    
    job1 = create_sequential_job(operations1)
    job2 = create_sequential_job(operations2)

    window = Window(1000)
    window.add_jobs([job1, job2])
    window.machines = ['cpu', 'gpu', 'fpga']

    return window

def generate_test_window() -> Window:
    """
    An example robotics workload of a periodic EKF, a SLAM, and a PID controller.
    The system has 3 machines: CPU, GPU, and FPGA.
    EKF is supported on CPU and FPGA.
    SLAM is supported on CPU and GPU.
    PID is supported on CPU and FPGA.

    EKF has 3 operations, SLAM has 5 operations, and PID has 2 operations.
    """

    machines = ['cpu', 'gpu', 'fpga']
    ekf1 = Operation([10, NOT_SUPPORTED, 5])
    ekf2 = Operation([20, NOT_SUPPORTED, 5], predecessor=ekf1)
    ekf3 = Operation([10, NOT_SUPPORTED, 10], predecessor=ekf2)
    ekf = Job([ekf1, ekf2, ekf3])

    slam1 = Operation([100, 10, NOT_SUPPORTED])
    slam2 = Operation([200, 40, NOT_SUPPORTED], predecessor=slam1)
    slam3 = Operation([100, 80, NOT_SUPPORTED], predecessor=slam2)
    slam4 = Operation([150, 20, NOT_SUPPORTED], predecessor=slam3)
    slam5 = Operation([300, 400, NOT_SUPPORTED], predecessor=slam4)
    slam = Job([slam1, slam2, slam3, slam4, slam5])

    pid1 = Operation([50, NOT_SUPPORTED, 25])
    pid2 = Operation([50, NOT_SUPPORTED, 25], predecessor=pid1)
    pid = Job([pid1, pid2])

    window = Window(1000, machines=machines)
    window.add_jobs([ekf, slam, pid])

    return window

def create_sequential_workload(n_jobs, n_operations_per_job) -> Workload:
    # Create a workload
    machines = ['cpu', 'gpu', 'fpga']
    operations = [[] for _ in range(n_jobs)]

    for i in range(n_jobs):
        for _ in range(n_operations_per_job):
            processing_times = [np.random.randint(50, 150) for _ in range(n_operations_per_job)]
            operations[i].append(Operation(processing_times))

    jobs = [create_sequential_job(ops) for ops in operations]

    workload_operations = []
    for job in jobs:
        workload_operations.extend(job.get_operations())
    workload = Workload(workload_operations, machines)
    return workload