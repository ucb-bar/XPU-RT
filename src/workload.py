import random

class Workload:
    """
    A workload that contains jobs and machines to schedule
    """

    operations = []
    machines = []

    def __init__(self, operations, machines):
        self.operations = operations
        self.machines = machines

    def add_job(self, job):
        self.jobs.append(job)

    def add_machine(self, machine):
        self.machines.append(machine)

    def get_jobs(self):
        return self.jobs

    def get_machines(self):
        return self.machines
    
    def get_num_machines(self):
        return len(self.machines)
    
    def get_operations(self):
        return self.operations
    

class Job:
    operations = []

    def __init__(self, operations):
        self.operations = operations

    def add_operation(self, operation):
        self.operations.append(operation)

    def get_operations(self):
        return self.operations

class VariableRuntimeJob (Job):
    """
    Model as a job that has two versions of operations with different runtimes
    """

    def __init__(self, operations, long_operations, prob=0.5):
        """
        Job that takes the long operations with probability prob
        """
        self.operations = operations
        self.long_operations = long_operations
        self.prob = prob

    def get_operations(self):
        """
        Returns the operations with shorter runtime with probability prob
        """
        if random.random() < self.prob:
            return self.long_operations
        else:
            return self.operations

class Operation:
    processing_times = []

    predecessor = None
    successor = None

    # TODO don't need sucessor
    def __init__(self, processing_times, predecessor=None, successor=None):
        self.processing_times = processing_times
        self.predecessor = predecessor
        self.successor = successor
    
    def get_predecessor(self):
        return self.predecessor

    def get_successor(self):
        return self.successor
    
    def get_durations(self):
        return self.processing_times
    
class Window:
    """
    A window of the workload that contains operations, alloted machines, and a time frame.
    """

    def __init__(self, time_frame: float, operations = [], machines = []):
        self.operations = operations
        self.machines = machines
        self.time_frame = time_frame

    def add_operations(self, operations: list[Operation]):
        self.operations.extend(operations)
    
    def add_jobs(self, jobs: list[Job]):
        for job in jobs:
            self.operations.extend(job.get_operations())