import json
from workload import Workload, Operation

def parse_graph_to_workload(file_path):
    # Load the .graph file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Dictionary to hold operations by name for setting dependencies
    operations_dict = {}
    
    # Iterate through each node in the graph and create Operation instances
    for node in data.get("nodes", []):
        # Placeholder processing times for each machine; these could be populated based on actual data if available
        processing_times = [1] * 3  # Assuming 3 machines, adjust as necessary

        # Create an operation with dummy predecessors and successors
        operation = Operation(processing_times)
        operations_dict[node["name"]] = operation

    # Now set predecessors and successors based on inputs
    for node in data.get("nodes", []):
        operation = operations_dict[node["name"]]
        inputs = node.get("inputs", [])
        
        # If inputs exist, assume they represent predecessors
        if inputs:
            # Assuming single predecessor for simplicity, modify if multiple predecessors allowed
            predecessor_name = inputs[0][0]  # Reference to the first input node's name
            if predecessor_name in operations_dict:
                predecessor = operations_dict[predecessor_name]
                operation.predecessor = predecessor
                predecessor.successor = operation

    # Collect all operations into a list
    operations = list(operations_dict.values())
    
    # Create a Workload instance with these operations
    workload = Workload(operations)
    
    return workload

# Example usage:
file_path = 'data/AlexNetTVM.graph'
workload = parse_graph_to_workload(file_path)

# Display some details to verify
print(f"Number of operations: {workload.num_operations}")
print(f"Number of machines: {workload.num_machines}")
for idx, op in enumerate(workload.operations[:5]):
    print(f"Operation {idx+1} durations: {op.get_durations()}")
    print(f"Predecessor: {op.get_predecessor()}")
    print(f"Successor: {op.get_successor()}")