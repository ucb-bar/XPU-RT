class Workload:

    num_operations = 0
    num_machines = 0
    operations = []

    def __init__(self, operations):
        self.operations = operations
        self.num_operations = len(operations)
        self.num_machines = len(operations[0].processing_times)

    def add_operation(self, operation):
        self.operations.append(operation)
        self.num_operations += 1

class Operation:
    processing_times = []

    predecessor = None
    successor = None

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