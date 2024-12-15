import numpy as np

class Operation:
    """
    Lowest level of a schedulable instance. An operation has a processing time and potentially a predecessor.
    Each operation has a must havea   processing time for each machine in the workload.
    """
    def __init__(self, processing_times: list[float], predecessor=None):
        self.processing_times = processing_times
        self.predecessor = predecessor
    
    def get_predecessor(self):
        return self.predecessor
    
    def get_durations(self) -> list[float]:
        return self.processing_times
    
class Job:
    """
    A set of operations that are linked together
    """
    def __init__(self, operations: list[Operation]):
        self.operations = operations

    def add_operation(self, operation: Operation):
        self.operations.append(operation)

    def get_operations(self) -> list[Operation]:
        return self.operations

class Workload:
    """
    High level representation of a schedulable workload that contains operations that are potentially
    dependent as part of a job, machines that can process the operations, and transfer times between
    machines.

    @param operations: list of operations that are part of the workload. Potentially dependent on each other.
    @param machines: list of machines that can process the operations.
    @param transfer_times: matrix of transfer times between machines. transfer_times[i][j] is the time to transfer from machine i to machine j.
    """
    def __init__(self, operations: list[Operation], machines: list[str], transfer_times: np.ndarray):
        self.operations = operations
        self.machines = machines
        self.transfer_times = transfer_times

    def get_machines(self) -> list[str]:
        return self.machines
    
    def get_operations(self) -> list[Operation]:
        return self.operations
    
    def get_durations(self) -> list[list[float]]:
        """
        Get the durations of the operations in the workload. The durations are grouped by job.
        """
        durations = []
        for i in range(len(self.operations)):
            operation = self.operations[i]
            if operation.predecessor is None:
                durations.append([operation.get_durations()])
            else:
                durations[-1].append(operation.get_durations())
        return durations
    
    def set_transfer_times(self, transfer_times: np.ndarray):
        self.transfer_times = transfer_times
    
    def get_transfer_times(self) -> np.ndarray:
        return self.transfer_times
    
class Window:
    """
    A time slice in a workload
    """
    def __init__(self, time_frame: float, operations: list[Operation], machines: list[str], transfer_times: np.ndarray):
        self.operations = operations
        self.machines = machines
        self.time_frame = time_frame
        self.transfer_times = transfer_times

    def add_operations(self, operations: list[Operation]):
        self.operations.extend(operations)
    
    def add_jobs(self, jobs: list[Job]):
        for job in jobs:
            self.operations.extend(job.get_operations())

    def get_transfer_times(self) -> np.ndarray:
        return self.transfer_times
    
    def get_durations(self) -> list[list[float]]:
        """
        Get the durations of the operations in the workload. The durations are grouped by job.
        """
        durations = []
        for i in range(len(self.operations)):
            operation = self.operations[i]
            if operation.predecessor is None:
                durations.append([operation.get_durations()])
            else:
                if len(durations) == 0:
                    durations.append([operation.get_durations()])
                else:
                    durations[-1].append(operation.get_durations())
        return durations