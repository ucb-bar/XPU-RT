"""
Fusion module for combining small operations into larger blocks for scheduling.

Fusion combines operations with runtime below a threshold together with their
neighboring operations (connected via dependencies) that also fall under the threshold.
These fused operations are scheduled as one contiguous block on the same hardware.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from xpu_rt.scheduler.workload import Workload, Operation


class FusedOperation(Operation):
    """
    A fused operation that represents multiple original operations combined together.
    """
    def __init__(self, original_operations: List[Operation], processing_times: List[float], 
                 predecessors=None, operation_id=None, operation_name=None, job_id=None):
        super().__init__(processing_times, predecessors, operation_id, operation_name, job_id)
        self.original_operations = original_operations  # List of original operations that were fused
        self.fused_operation_ids = [op.operation_id for op in original_operations if op.operation_id is not None]
    
    def get_original_operations(self) -> List[Operation]:
        """Returns the list of original operations that were fused together."""
        return self.original_operations


def get_min_duration(operation: Operation, machine_combinations: List[List[str]], machines: List[str]) -> float:
    """
    Get the minimum duration of an operation across all machine combinations.
    This is used to determine if an operation is below the fusion threshold.
    """
    min_dur = float('inf')
    for k in range(len(machine_combinations)):
        dur = operation.get_duration_for_combination(k, machine_combinations, machines)
        min_dur = min(min_dur, dur)
    return min_dur


def find_neighbors(operation: Operation, workload: Workload) -> Tuple[List[Operation], List[Operation]]:
    """
    Find neighboring operations (predecessors and successors) of an operation.
    
    Returns:
        (predecessors, successors) where:
        - predecessors: operations that this operation depends on
        - successors: operations that depend on this operation
    """
    predecessors = operation.get_predecessors()
    
    # Find successors: operations that have this operation as a predecessor
    successors = []
    for op in workload.operations:
        if operation in op.get_predecessors():
            successors.append(op)
    
    return predecessors, successors


def would_create_cycle(group: Set[int], workload: Workload, existing_fusion_groups: List[Set[int]]) -> bool:
    """
    Check if fusing the operations in 'group' would create a cycle in the dependency graph.
    
    We simulate the fusion and check if it creates a cycle by:
    1. Building a dependency graph of what the fused operations would be
    2. Checking for cycles in that graph
    
    Args:
        group: Set of operation indices to be fused
        workload: The original workload
        existing_fusion_groups: List of already-formed fusion groups
        
    Returns:
        True if fusion would create a cycle, False otherwise
    """
    # Build a map of original op index -> which fusion group it belongs to
    orig_to_group = {}
    for i, existing_group in enumerate(existing_fusion_groups):
        for orig_idx in existing_group:
            orig_to_group[orig_idx] = i
    
    # Create a temporary fusion group for the new group
    new_group_idx = len(existing_fusion_groups)
    for orig_idx in group:
        orig_to_group[orig_idx] = new_group_idx
    
    # Build dependency graph of fused operations
    # Each fusion group becomes a node
    num_groups = len(existing_fusion_groups) + 1
    dependency_graph = {i: [] for i in range(num_groups)}
    
    # For each fusion group, find its predecessors (other fusion groups)
    for group_idx, fusion_group in enumerate(existing_fusion_groups + [group]):
        group_predecessors = set()
        for orig_idx in fusion_group:
            orig_op = workload.operations[orig_idx]
            for pred in orig_op.get_predecessors():
                try:
                    pred_idx = workload.operations.index(pred)
                    if pred_idx not in fusion_group:  # External predecessor
                        pred_group = orig_to_group.get(pred_idx)
                        if pred_group is not None and pred_group != group_idx:
                            group_predecessors.add(pred_group)
                except ValueError:
                    pass
        
        # Add edges to dependency graph (avoid duplicates)
        for pred_group in group_predecessors:
            if pred_group not in dependency_graph[group_idx]:
                dependency_graph[group_idx].append(pred_group)
    
    # Also check successors - if group A depends on group B, then group B has group A as a successor
    # But we only need predecessors for cycle detection, so this is fine as-is
    
    # Check for cycles using DFS
    def has_cycle_dfs(node_idx, visited, rec_stack, graph):
        visited[node_idx] = True
        rec_stack[node_idx] = True
        for neighbor_idx in graph.get(node_idx, []):
            if not visited.get(neighbor_idx, False):
                if has_cycle_dfs(neighbor_idx, visited, rec_stack, graph):
                    return True
            elif rec_stack.get(neighbor_idx, False):
                # Found a cycle!
                return True
        rec_stack[node_idx] = False
        return False
    
    visited = {}
    rec_stack = {}
    for node_idx in range(num_groups):
        if not visited.get(node_idx, False):
            if has_cycle_dfs(node_idx, visited, rec_stack, dependency_graph):
                return True
    
    return False


def fuse_operations(workload: Workload, threshold: float) -> Tuple[Workload, Dict[int, List[int]]]:
    """
    Fuse operations below the threshold into linear chains.
    Only fuses operations that form linear dependencies where each operation has at most one predecessor.
    This guarantees no cycles can be created.
    
    Args:
        workload: The original workload to fuse
        threshold: Maximum duration (in time units) for operations to be considered for fusion
        
    Returns:
        (fused_workload, fusion_map) where:
        - fused_workload: New workload with fused operations
        - fusion_map: Dictionary mapping fused operation index -> list of original operation indices
    """
    if threshold <= 0:
        # No fusion requested
        return workload, {i: [i] for i in range(len(workload.operations))}
    
    machine_combinations = workload.get_machine_combinations()
    machines = workload.machines
    
    # Step 1: Identify operations below threshold
    below_threshold = set()
    for i, op in enumerate(workload.operations):
        min_dur = get_min_duration(op, machine_combinations, machines)
        if min_dur <= threshold:
            below_threshold.add(i)
    
    if not below_threshold:
        # No operations to fuse
        return workload, {i: [i] for i in range(len(workload.operations))}
    
    # Step 2: Find linear chains of operations below threshold
    # A linear chain is a sequence where:
    # - Each operation (except the first) has exactly one predecessor
    # - That predecessor is also in the chain and below threshold
    # - Each operation (except the last) has at least one successor that is also in the chain and below threshold
    
    fusion_groups: List[List[int]] = []  # Use List instead of Set to preserve order
    processed = set()
    
    def find_linear_chain(start_idx: int) -> List[int]:
        """
        Find a linear chain starting from start_idx.
        A linear chain is a sequence where each operation has exactly one predecessor in the chain.
        """
        chain = []
        current_idx = start_idx
        
        # First, go backwards to find the start of the chain
        while current_idx is not None and current_idx in below_threshold and current_idx not in processed:
            current_op = workload.operations[current_idx]
            predecessors = current_op.get_predecessors()
            
            # Filter to only predecessors that are below threshold
            pred_below_threshold = []
            pred_not_below_threshold = []
            for pred in predecessors:
                try:
                    pred_idx = workload.operations.index(pred)
                    if pred_idx in below_threshold:
                        pred_below_threshold.append(pred_idx)
                    else:
                        pred_not_below_threshold.append(pred_idx)
                except ValueError:
                    pass
            
            # For a linear chain, we can only have at most one predecessor below threshold
            if len(pred_below_threshold) > 1:
                # Multiple predecessors below threshold - can't form a linear chain from here
                break
            
            # If there are any predecessors NOT below threshold, we must stop here
            # because this operation depends on something outside the chain (inter-network dependency)
            if len(pred_not_below_threshold) > 0:
                # Has inter-network or other external dependencies - this is the start of the chain
                break
            
            if len(pred_below_threshold) == 0:
                # No predecessors below threshold - this is the start of the chain
                break
            
            # Exactly one predecessor below threshold and no external dependencies - continue backwards
            current_idx = pred_below_threshold[0]
        
        # Now build the chain forward from the start
        chain_start = current_idx
        current_idx = chain_start
        
        while current_idx is not None and current_idx in below_threshold and current_idx not in processed:
            chain.append(current_idx)
            current_op = workload.operations[current_idx]
            
            # Find successors that are below threshold
            successors = []
            for op in workload.operations:
                if current_op in op.get_predecessors():
                    try:
                        succ_idx = workload.operations.index(op)
                        if succ_idx in below_threshold:
                            successors.append(succ_idx)
                    except ValueError:
                        pass
            
            # For a linear chain, we can only have at most one successor below threshold
            if len(successors) > 1:
                # Multiple successors below threshold - end the chain here
                break
            
            if len(successors) == 0:
                # No successors below threshold - end of chain
                break
            
            # Exactly one successor below threshold - continue forward
            # But check that this successor has only one predecessor below threshold (us)
            # AND no predecessors NOT below threshold (inter-network dependencies)
            succ_idx = successors[0]
            succ_op = workload.operations[succ_idx]
            succ_predecessors = succ_op.get_predecessors()
            pred_below_threshold = []
            pred_not_below_threshold = []
            for pred in succ_predecessors:
                try:
                    pred_idx = workload.operations.index(pred)
                    if pred_idx in below_threshold:
                        pred_below_threshold.append(pred_idx)
                    else:
                        pred_not_below_threshold.append(pred_idx)
                except ValueError:
                    pass
            
            # The successor must have exactly one predecessor below threshold (which is us)
            if len(pred_below_threshold) != 1 or pred_below_threshold[0] != current_idx:
                # Either multiple predecessors or not us - end the chain
                break
            
            # If the successor has any predecessors NOT below threshold, stop here
            # to preserve inter-network dependencies
            if len(pred_not_below_threshold) > 0:
                # Successor has external dependencies (inter-network) - end the chain here
                break
            
            current_idx = succ_idx
        
        return chain
    
    # Step 3: Find all linear chains
    for idx in below_threshold:
        if idx not in processed:
            chain = find_linear_chain(idx)
            if len(chain) > 1:  # Only fuse chains with at least 2 operations
                fusion_groups.append(chain)
                processed.update(chain)
            else:
                processed.add(idx)
    
    # Step 4: Create fused operations
    fused_operations: List[Operation] = []
    fusion_map: Dict[int, List[int]] = {}
    original_to_fused: Dict[int, int] = {}  # Maps original op index to fused op index
    
    # First, add all non-fused operations (they keep their original predecessors for now)
    for i, op in enumerate(workload.operations):
        in_any_group = any(i in group for group in fusion_groups)
        if not in_any_group:
            fused_operations.append(op)
            fused_idx = len(fused_operations) - 1
            fusion_map[fused_idx] = [i]
            original_to_fused[i] = fused_idx
    
    # Then, create fused operations for each linear chain
    for chain in fusion_groups:
        # Chain is already ordered (linear sequence)
        original_ops = [workload.operations[i] for i in chain]
        
        # Combine processing times: sum for each machine
        # For each machine, sum the durations
        # IMPORTANT: We sum processing_times[machine_idx] which is correct for singleton combinations
        # The scheduler will use get_duration_for_combination() which for singletons returns processing_times[machine_idx]
        combined_processing_times = []
        for machine_idx in range(len(machines)):
            total_dur = sum(op.processing_times[machine_idx] for op in original_ops 
                          if machine_idx < len(op.processing_times))
            combined_processing_times.append(total_dur)
        
        # Combine predecessors: for a linear chain, only the first operation's predecessors matter
        # (since each operation in the chain only has the previous one as a predecessor)
        # Store as original operation indices for now, we'll convert to fused operations later
        combined_predecessor_indices = set()
        first_op = original_ops[0]
        for pred in first_op.get_predecessors():
            try:
                pred_idx = workload.operations.index(pred)
                if pred_idx not in chain:
                    combined_predecessor_indices.add(pred_idx)
            except ValueError:
                # Predecessor not in workload (shouldn't happen, but be safe)
                pass
        
        # Create fused operation with empty predecessors for now (will be updated in Step 5)
        fused_op = FusedOperation(
            original_operations=original_ops,
            processing_times=combined_processing_times,
            predecessors=[],  # Will be set in Step 5
            operation_id=f"fused_{len(fused_operations)}",
            operation_name=f"Fused({','.join([op.operation_name or f'op{i}' for i, op in enumerate(original_ops)])})",
            job_id=original_ops[0].job_id if original_ops else None
        )
        
        fused_idx = len(fused_operations)
        fused_operations.append(fused_op)
        fusion_map[fused_idx] = chain
        for orig_idx in chain:
            original_to_fused[orig_idx] = fused_idx
        
        # Store predecessor indices for this fused operation (will use in Step 5)
        fused_op._temp_predecessor_indices = combined_predecessor_indices
    
    # Step 5: Update all operations to point to fused operations when their predecessors were fused
    for fused_idx, fused_op in enumerate(fused_operations):
        updated_predecessors = []
        
        if isinstance(fused_op, FusedOperation):
            # For fused operations, use the stored predecessor indices
            pred_indices = getattr(fused_op, '_temp_predecessor_indices', set())
            for pred_idx in pred_indices:
                if pred_idx in original_to_fused:
                    # This predecessor was fused - point to the fused operation
                    fused_pred_idx = original_to_fused[pred_idx]
                    fused_pred_op = fused_operations[fused_pred_idx]
                    if fused_pred_op not in updated_predecessors:
                        updated_predecessors.append(fused_pred_op)
                else:
                    # This predecessor was not fused - use the original operation
                    pred_op = workload.operations[pred_idx]
                    if pred_op not in updated_predecessors:
                        updated_predecessors.append(pred_op)
            # Clean up temporary attribute
            if hasattr(fused_op, '_temp_predecessor_indices'):
                delattr(fused_op, '_temp_predecessor_indices')
        else:
            # For non-fused operations, update their predecessors
            for pred in fused_op.get_predecessors():
                try:
                    pred_idx = workload.operations.index(pred)
                    if pred_idx in original_to_fused:
                        # This predecessor was fused - point to the fused operation
                        fused_pred_idx = original_to_fused[pred_idx]
                        fused_pred_op = fused_operations[fused_pred_idx]
                        if fused_pred_op not in updated_predecessors:
                            updated_predecessors.append(fused_pred_op)
                    else:
                        # This predecessor was not fused - keep it as is
                        if pred not in updated_predecessors:
                            updated_predecessors.append(pred)
                except ValueError:
                    # Predecessor not found (shouldn't happen, but be safe)
                    if pred not in updated_predecessors:
                        updated_predecessors.append(pred)
        
        fused_op.predecessors = updated_predecessors
        fused_op.predecessor = updated_predecessors[0] if updated_predecessors else None
    
    # Step 7: Create new workload with fused operations
    fused_workload = Workload(
        operations=fused_operations,
        machines=workload.machines,
        transfer_times=workload.transfer_times,
        job_names=workload.job_names,
        machine_combinations=workload.machine_combinations
    )
    
    return fused_workload, fusion_map


def expand_schedule(fused_workload: Workload, fusion_map: Dict[int, List[int]], 
                   original_workload: Workload, t_fused: np.ndarray, alpha_fused: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand a schedule from fused operations back to original operations.
    
    Args:
        fused_workload: The fused workload that was scheduled
        fusion_map: Mapping from fused operation index to list of original operation indices
        original_workload: The original workload
        t_fused: Start times for fused operations
        alpha_fused: Machine assignments for fused operations
        
    Returns:
        (t_original, alpha_original) where:
        - t_original: Start times for original operations
        - alpha_original: Machine assignments for original operations
    """
    if t_fused is None or alpha_fused is None:
        raise ValueError("Cannot expand schedule: fused schedule is None (optimization failed)")
    
    num_original = len(original_workload.operations)
    num_combinations = len(original_workload.machine_combinations)
    
    t_original = np.zeros(num_original)
    alpha_original = np.zeros((num_original, num_combinations))
    
    # Track which original operations have been scheduled
    scheduled_ops = set()
    
    # For each fused operation, assign its schedule to all original operations in the fusion group
    for fused_idx, fused_op in enumerate(fused_workload.operations):
        if isinstance(fused_op, FusedOperation):
            original_indices = fusion_map[fused_idx]
            fused_start_time = t_fused[fused_idx]
            fused_machine_assignment = alpha_fused[fused_idx]
            
            # Assign same start time and machine to all original operations
            # Operations within a fused block execute sequentially
            cumulative_time = fused_start_time
            for i, orig_idx in enumerate(original_indices):
                if orig_idx >= num_original:
                    raise ValueError(f"Invalid original index {orig_idx} (max: {num_original-1})")
                t_original[orig_idx] = cumulative_time
                alpha_original[orig_idx] = fused_machine_assignment
                scheduled_ops.add(orig_idx)
                
                # Next operation in the fused block starts after this one finishes
                # Use the minimum duration for this operation on the assigned machine
                assigned_combo = np.argmax(fused_machine_assignment)
                orig_op = original_workload.operations[orig_idx]
                dur = orig_op.get_duration_for_combination(
                    assigned_combo, 
                    original_workload.machine_combinations, 
                    original_workload.machines
                )
                cumulative_time += dur
        else:
            # Non-fused operation: map directly
            orig_idx = fusion_map[fused_idx][0]
            if orig_idx >= num_original:
                raise ValueError(f"Invalid original index {orig_idx} (max: {num_original-1})")
            t_original[orig_idx] = t_fused[fused_idx]
            alpha_original[orig_idx] = alpha_fused[fused_idx]
            scheduled_ops.add(orig_idx)
    
    # Validate that all original operations were scheduled
    if len(scheduled_ops) != num_original:
        missing = set(range(num_original)) - scheduled_ops
        raise ValueError(f"Not all operations were scheduled! Missing {len(missing)} operations: {sorted(missing)[:10]}...")
    
    return t_original, alpha_original


def print_fusion_report(original_workload: Workload, fused_workload: Workload, fusion_map: Dict[int, List[int]], output_file: str = None) -> None:
    """
    Print a detailed report of the fusion transformation for debugging.
    
    Args:
        original_workload: The original workload before fusion
        fused_workload: The fused workload
        fusion_map: Mapping from fused operation index to list of original operation indices
        output_file: Optional file path to write the report to. If None, prints to stdout.
    """
    import sys
    from io import StringIO
    
    # Redirect output to file or stdout
    if output_file:
        f = open(output_file, 'w')
        original_stdout = sys.stdout
        sys.stdout = f
    else:
        f = None
    
    try:
        print("\n" + "=" * 80)
        print("FUSION REPORT")
        print("=" * 80)
        
        print(f"\nOriginal workload: {len(original_workload.operations)} operations")
        print(f"Fused workload: {len(fused_workload.operations)} operations")
        print(f"Reduction: {len(original_workload.operations) - len(fused_workload.operations)} operations fused")
        
        # Analyze fusion groups
        fusion_groups = []
        non_fused = []
        for fused_idx, original_indices in fusion_map.items():
            if len(original_indices) > 1:
                fusion_groups.append((fused_idx, original_indices))
            else:
                non_fused.append((fused_idx, original_indices[0]))
        
        print(f"\nFusion groups: {len(fusion_groups)}")
        print(f"Non-fused operations: {len(non_fused)}")
        
        # Print details of each fusion group
        print("\n" + "-" * 80)
        print("FUSION GROUPS:")
        print("-" * 80)
        for fused_idx, original_indices in fusion_groups:
            fused_op = fused_workload.operations[fused_idx]
            print(f"\nFused Operation {fused_idx}:")
            print(f"  Contains {len(original_indices)} original operations: {original_indices}")
            print(f"  Name: {fused_op.operation_name}")
            print(f"  Processing times: {fused_op.processing_times}")
            
            # Check predecessors
            preds = fused_op.get_predecessors()
            print(f"  Predecessors ({len(preds)}):")
            for pred in preds:
                # Find which fused operation this predecessor is
                pred_fused_idx = None
                for fidx, op in enumerate(fused_workload.operations):
                    if op == pred:
                        pred_fused_idx = fidx
                        break
                if pred_fused_idx is not None:
                    pred_orig_indices = fusion_map.get(pred_fused_idx, [])
                    print(f"    - Fused op {pred_fused_idx} (original ops: {pred_orig_indices})")
                else:
                    print(f"    - {pred.operation_name if hasattr(pred, 'operation_name') else 'Unknown'}")
            
            # Check for operations within the fusion group that depend on each other
            print(f"  Internal dependencies:")
            for i, orig_idx_i in enumerate(original_indices):
                orig_op_i = original_workload.operations[orig_idx_i]
                for j, orig_idx_j in enumerate(original_indices):
                    if i != j:
                        orig_op_j = original_workload.operations[orig_idx_j]
                        if orig_op_j in orig_op_i.get_predecessors():
                            print(f"    - Original op {orig_idx_i} depends on original op {orig_idx_j} (within fusion group)")
        
        # Check for potential issues
        print("\n" + "-" * 80)
        print("DEPENDENCY ANALYSIS:")
        print("-" * 80)
        
        # Build dependency graph with operation names for reporting
        dependency_graph = {}
        operation_names = {}  # Map fused_idx -> list of operation names
        for fused_idx, fused_op in enumerate(fused_workload.operations):
            dependency_graph[fused_idx] = []
            # Get operation names
            if isinstance(fused_op, FusedOperation):
                orig_indices = fusion_map.get(fused_idx, [])
                names = []
                for orig_idx in orig_indices:
                    orig_op = original_workload.operations[orig_idx]
                    name = orig_op.operation_name or orig_op.operation_id or f"op{orig_idx}"
                    names.append(name)
                operation_names[fused_idx] = names
            else:
                orig_idx = fusion_map.get(fused_idx, [fused_idx])[0]
                orig_op = original_workload.operations[orig_idx]
                name = orig_op.operation_name or orig_op.operation_id or f"op{orig_idx}"
                operation_names[fused_idx] = [name]
            
            for pred in fused_op.get_predecessors():
                for fidx, op in enumerate(fused_workload.operations):
                    if op == pred:
                        dependency_graph[fused_idx].append(fidx)
                        break
        
        # Find all cycles with detailed paths
        def find_cycles_dfs(node_idx, path, visited, rec_stack, graph, cycles, names_map):
            visited[node_idx] = True
            rec_stack[node_idx] = True
            path.append(node_idx)
            
            for neighbor_idx in graph.get(node_idx, []):
                if neighbor_idx in path:
                    # Found a cycle! Extract the cycle path
                    cycle_start = path.index(neighbor_idx)
                    cycle_path = path[cycle_start:] + [neighbor_idx]
                    cycles.append(cycle_path)
                elif not visited.get(neighbor_idx, False):
                    find_cycles_dfs(neighbor_idx, path, visited, rec_stack, graph, cycles, names_map)
            
            path.pop()
            rec_stack[node_idx] = False
        
        # Find all cycles
        visited = {}
        rec_stack = {}
        cycles = []
        for node_idx in range(len(fused_workload.operations)):
            if not visited.get(node_idx, False):
                find_cycles_dfs(node_idx, [], visited, rec_stack, dependency_graph, cycles, operation_names)
        
        if cycles:
            print(f"  WARNING: {len(cycles)} cycle(s) detected in fused dependency graph!")
            print("\n  CYCLE DETAILS:")
            for cycle_idx, cycle in enumerate(cycles, 1):
                print(f"\n    Cycle {cycle_idx}:")
                for i, fused_idx in enumerate(cycle[:-1]):  # Don't repeat the last node
                    next_idx = cycle[i + 1]
                    names = operation_names.get(fused_idx, [f"fused_op{fused_idx}"])
                    next_names = operation_names.get(next_idx, [f"fused_op{next_idx}"])
                    print(f"      FusedOp{fused_idx} ({', '.join(names)}) -> FusedOp{next_idx} ({', '.join(next_names)})")
                # Show the closing edge
                first_idx = cycle[0]
                last_idx = cycle[-1]
                first_names = operation_names.get(first_idx, [f"fused_op{first_idx}"])
                last_names = operation_names.get(last_idx, [f"fused_op{last_idx}"])
                print(f"      FusedOp{last_idx} ({', '.join(last_names)}) -> FusedOp{first_idx} ({', '.join(first_names)}) [CLOSES CYCLE]")
        else:
            print("  No cycles detected in fused dependency graph.")
        
        # Check for operations with no predecessors (entry points)
        entry_points = [i for i, op in enumerate(fused_workload.operations) if not op.get_predecessors()]
        print(f"  Entry points (no predecessors): {len(entry_points)} operations")
        
        # Check processing times
        print("\n" + "-" * 80)
        print("PROCESSING TIME ANALYSIS:")
        print("-" * 80)
        for fused_idx, fused_op in enumerate(fused_workload.operations):
            if isinstance(fused_op, FusedOperation):
                original_indices = fusion_map[fused_idx]
                print(f"\nFused op {fused_idx} (original ops: {original_indices}):")
                print(f"  Combined processing times: {fused_op.processing_times}")
                # Show individual operation times
                for orig_idx in original_indices:
                    orig_op = original_workload.operations[orig_idx]
                    print(f"    Original op {orig_idx}: {orig_op.processing_times}")
        
        print("\n" + "=" * 80)
    finally:
        if f:
            sys.stdout = original_stdout
            f.close()
            print(f"\nFusion report written to: {output_file}")
