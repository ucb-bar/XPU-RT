import numpy as np

class Operation:
    """
    Lowest level of a schedulable instance. An operation has a processing time and potentially multiple predecessors.
    Each operation has a must havea   processing time for each machine in the workload.
    """
    def __init__(self, processing_times: list[float], predecessors=None, operation_id=None, operation_name=None, job_id=None):
        self.processing_times = processing_times
        # Support both single predecessor (backward compatibility) and list of predecessors
        if predecessors is None:
            self.predecessors = []
        elif isinstance(predecessors, list):
            self.predecessors = predecessors
        else:
            # Single predecessor provided for backward compatibility
            self.predecessors = [predecessors]
        # Keep backward compatibility: predecessor property returns first predecessor or None
        self.predecessor = self.predecessors[0] if self.predecessors else None
        # Operation identifier and name
        self.operation_id = operation_id
        self.operation_name = operation_name
        # Job identifier - explicitly tracks which job this operation belongs to
        self.job_id = job_id
    
    def get_predecessors(self):
        """Returns list of all predecessors"""
        return self.predecessors
    
    def get_predecessor(self):
        """Returns first predecessor for backward compatibility, or None if no predecessors"""
        return self.predecessors[0] if self.predecessors else None
    
    def add_predecessor(self, predecessor):
        """Add a predecessor to the list"""
        if predecessor not in self.predecessors:
            self.predecessors.append(predecessor)
            # Update backward compatibility property (first predecessor)
            self.predecessor = self.predecessors[0] if self.predecessors else None
    
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
    def __init__(self, operations: list[Operation], machines: list[str], transfer_times: np.ndarray, job_names: list[str] = None):
        self.operations = operations
        self.machines = machines
        self.transfer_times = transfer_times
        self.job_names = job_names if job_names is not None else []

    def get_machines(self) -> list[str]:
        return self.machines
    
    def get_operations(self) -> list[Operation]:
        return self.operations
    
    def get_durations(self) -> list:
        """
        Get the durations of the operations in the workload. The durations are grouped by job.
        Uses explicit job_id if available, otherwise falls back to predecessor-based grouping.
        """
        durations = []
        current_job_id = None
        
        for i in range(len(self.operations)):
            operation = self.operations[i]
            
            # Use explicit job_id if available
            if operation.job_id is not None:
                if operation.job_id != current_job_id:
                    # Start of a new job
                    durations.append([operation.get_durations()])
                    current_job_id = operation.job_id
                else:
                    # Same job, append to current job's operations
                    durations[-1].append(operation.get_durations())
            else:
                # Fallback to predecessor-based grouping for backward compatibility
                if not operation.predecessors:  # No predecessors means start of a new job
                    durations.append([operation.get_durations()])
                    current_job_id = None  # Reset for fallback mode
                else:
                    if len(durations) == 0:
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
            if not operation.predecessors:  # No predecessors means start of a new job
                durations.append([operation.get_durations()])
            else:
                if len(durations) == 0:
                    durations.append([operation.get_durations()])
                else:
                    durations[-1].append(operation.get_durations())
        return durations