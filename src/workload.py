class Workload:
    """
    A workload that contains jobs and machines to schedule
    """

    jobs = []
    machines = []

    def __init__(self, jobs, machines):
        self.jobs = jobs
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
    

class Job:
    operations = []

    def __init__(self, operations):
        self.operations = operations

    def add_operation(self, operation):
        self.operations.append(operation)

    def get_operations(self):
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