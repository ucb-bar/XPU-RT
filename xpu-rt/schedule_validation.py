import numpy as np
from typing import Dict, List, Tuple, Optional
from workload import Workload, Operation
import json
try:
    from fusion import FusedOperation
except ImportError:
    # FusedOperation might not be available
    FusedOperation = None

def overlap_fixer(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    Resolves overlaps by pushing them forward in time
    @return: updated t that is free of overlaps and respects the precedence constraints
    """
    transfer_times = workload.get_transfer_times()
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[j] in workload.operations[i].predecessors:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[i] in workload.operations[j].predecessors:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    t[j] = t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    t[i] = t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time

    return t

def count_overlaps(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    @return: number of overlaps in the schedule
    """
    transfer_times = workload.get_transfer_times()
    count = 0

    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[j] in workload.operations[i].predecessors:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[i] in workload.operations[j].predecessors:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    count += 1
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    count += 1
    return count



def validate_schedule(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    original_json_data: Dict,
    profiled_times_p: Optional[Dict[int, Dict]] = None,
    profiled_times_e: Optional[Dict[int, Dict]] = None,
    profiled_times_by_network: Optional[Dict[str, Dict[str, Dict[int, Dict]]]] = None,
    output_file: str = "validation_report.txt",
) -> Tuple[bool, Dict]:
    """
    Validate a schedule against original JSON dependencies and profiled data.
    
    Args:
        workload: The workload that was scheduled
        t: Start times for operations
        alpha: Machine assignments for operations
        original_json_data: Original JSON dispatch data (with 'dispatches' key)
        profiled_times_p: Optional dict mapping dispatch_id -> profiled P-core runtime
        profiled_times_e: Optional dict mapping dispatch_id -> profiled E-core runtime
        output_file: Path to write validation report
        
    Returns:
        (is_valid, validation_results) where:
        - is_valid: True if all validations pass
        - validation_results: Dict with detailed validation results
    """
    results = {
        "all_operations_scheduled": True,
        "predecessor_constraints_satisfied": True,
        "durations_match_profiled": True,
        "machine_assignments_valid": True,
        "no_overlaps": True,
        "errors": [],
        "warnings": [],
        "operation_details": [],
    }
    
    dispatches = original_json_data.get("dispatches", {})
    num_operations = len(workload.operations)
    machine_combinations = workload.machine_combinations
    machines = workload.machines
    
    # Create mapping from operation name to index
    # Note: workload.operations should be the original operations (after expansion from fusion)
    op_name_to_idx = {}
    for i, op in enumerate(workload.operations):
        # Try to extract dispatch name from operation name
        # Operation names might be like "mobilenet_dispatch_12" or "fast_dispatch_0"
        op_name = op.operation_name or op.operation_id or f"op{i}"
        op_name_to_idx[op_name] = i
        
        # Also try matching without prefix if needed (for cross-network dependencies)
        # But for now, use the full name
    
    # Validation 1: Check all operations from JSON are scheduled
    json_dispatch_names = set(dispatches.keys())
    scheduled_op_names = set(op_name_to_idx.keys())
    missing_ops = json_dispatch_names - scheduled_op_names
    extra_ops = scheduled_op_names - json_dispatch_names
    
    if missing_ops:
        results["all_operations_scheduled"] = False
        results["errors"].append(f"Missing {len(missing_ops)} operations from JSON: {sorted(list(missing_ops))[:10]}")
    
    if extra_ops:
        results["warnings"].append(f"Extra {len(extra_ops)} operations in schedule not in JSON: {sorted(list(extra_ops))[:10]}")
    
    # Validation 2: Check predecessor constraints
    predecessor_violations = []
    for i, op in enumerate(workload.operations):
        op_name = op.operation_name or op.operation_id or f"op{i}"
        if op_name not in dispatches:
            continue
            
        dispatch_info = dispatches[op_name]
        json_deps = dispatch_info.get("dependencies", [])
        
        # Get actual predecessors from operation
        actual_predecessors = op.get_predecessors()
        actual_pred_names = []
        for pred in actual_predecessors:
            pred_name = pred.operation_name or pred.operation_id
            if pred_name:
                actual_pred_names.append(pred_name)
        
        # Check that each JSON dependency is satisfied
        for json_dep in json_deps:
            if json_dep not in actual_pred_names:
                # Check if it's in the schedule (might be from different network)
                if json_dep in op_name_to_idx:
                    pred_idx = op_name_to_idx[json_dep]
                    # Check timing constraint
                    pred_finish_time = t[pred_idx] + workload.operations[pred_idx].get_duration_for_combination(
                        np.argmax(alpha[pred_idx]),
                        machine_combinations,
                        machines
                    )
                    if t[i] < pred_finish_time:
                        predecessor_violations.append({
                            "operation": op_name,
                            "predecessor": json_dep,
                            "op_start": t[i],
                            "pred_finish": pred_finish_time,
                            "violation": pred_finish_time - t[i],
                        })
                else:
                    results["warnings"].append(f"Operation {op_name} depends on {json_dep} (from JSON) but it's not in the schedule")
        
        # Check timing constraints for actual predecessors
        for pred in actual_predecessors:
            try:
                pred_idx = workload.operations.index(pred)
                pred_finish_time = t[pred_idx] + workload.operations[pred_idx].get_duration_for_combination(
                    np.argmax(alpha[pred_idx]),
                    machine_combinations,
                    machines
                )
                if t[i] < pred_finish_time:
                    pred_name = pred.operation_name or pred.operation_id or f"op{pred_idx}"
                    predecessor_violations.append({
                        "operation": op_name,
                        "predecessor": pred_name,
                        "op_start": t[i],
                        "pred_finish": pred_finish_time,
                        "violation": pred_finish_time - t[i],
                    })
            except ValueError:
                # Predecessor not in workload (external dependency)
                pass
    
    if predecessor_violations:
        results["predecessor_constraints_satisfied"] = False
        results["errors"].append(f"Found {len(predecessor_violations)} predecessor constraint violations")
        results["predecessor_violation_details"] = predecessor_violations
    
    # Validation 3: Check durations match profiled data
    duration_mismatches = []
    for i, op in enumerate(workload.operations):
        op_name = op.operation_name or op.operation_id or f"op{i}"
        if op_name not in dispatches:
            continue
        
        dispatch_info = dispatches[op_name]
        dispatch_id = dispatch_info.get("id")
        
        if dispatch_id is None:
            continue
        
        # Get assigned machine
        assigned_combo_idx = np.argmax(alpha[i])
        assigned_combo = machine_combinations[assigned_combo_idx]
        
        # Get scheduled duration (what the scheduler used)
        scheduled_duration = op.get_duration_for_combination(
            assigned_combo_idx,
            machine_combinations,
            machines
        )
        
        # Also get the processing_times to see what's stored
        processing_times = op.processing_times if hasattr(op, 'processing_times') else []
        
        # Determine which network this operation belongs to from its name prefix
        network_prefix = None
        if op_name.startswith("mobilenet_"):
            network_prefix = "mobilenet"
        elif op_name.startswith("dronet_"):
            network_prefix = "dronet"
        elif op_name.startswith("diffusion_"):
            network_prefix = "diffusion"
        elif op_name.startswith("fast_"):
            network_prefix = "fast"
        elif op_name.startswith("mlp"):
            network_prefix = "mlp"
        
        # Get the correct profiled data for this network
        # If profiled_times_by_network is provided, use network-specific data
        # Otherwise fall back to combined profiled_times_p/e (which may have collisions)
        network_profiled_p = None
        network_profiled_e = None
        if profiled_times_by_network and network_prefix:
            if network_prefix in profiled_times_by_network:
                network_profiled_p = profiled_times_by_network[network_prefix].get("p")
                network_profiled_e = profiled_times_by_network[network_prefix].get("e")
        
        # Fall back to combined profiled times if network-specific not available
        if network_profiled_p is None:
            network_profiled_p = profiled_times_p
        if network_profiled_e is None:
            network_profiled_e = profiled_times_e
        
        # Check against profiled data using network-specific data to avoid dispatch_id collisions
        if "CPU_P" in assigned_combo and network_profiled_p and dispatch_id in network_profiled_p:
            profiled_duration = network_profiled_p[dispatch_id]["time_ms"]
            # Get what processing_times[0] (CPU_P) should be
            cpu_p_stored = processing_times[0] if len(processing_times) > 0 else None
            # Only flag mismatch if scheduled matches stored (ensures operation is correct)
            if cpu_p_stored is not None and abs(scheduled_duration - cpu_p_stored) < 0.01:
                if abs(scheduled_duration - profiled_duration) > 0.01:  # Allow 0.01ms tolerance
                    duration_mismatches.append({
                        "operation": op_name,
                        "dispatch_id": dispatch_id,
                        "machine": "CPU_P",
                        "scheduled": scheduled_duration,
                        "profiled": profiled_duration,
                        "difference": abs(scheduled_duration - profiled_duration),
                        "processing_times_cpu_p": cpu_p_stored,
                        "network": network_prefix,
                    })
        
        if "CPU_E" in assigned_combo and network_profiled_e and dispatch_id in network_profiled_e:
            profiled_duration = network_profiled_e[dispatch_id]["time_ms"]
            # Get what processing_times[1] (CPU_E) should be
            cpu_e_stored = processing_times[1] if len(processing_times) > 1 else None
            # Only flag mismatch if scheduled matches stored (ensures operation is correct)
            if cpu_e_stored is not None and abs(scheduled_duration - cpu_e_stored) < 0.01:
                if abs(scheduled_duration - profiled_duration) > 0.01:  # Allow 0.01ms tolerance
                    duration_mismatches.append({
                        "operation": op_name,
                        "dispatch_id": dispatch_id,
                        "machine": "CPU_E",
                        "scheduled": scheduled_duration,
                        "profiled": profiled_duration,
                        "difference": abs(scheduled_duration - profiled_duration),
                        "processing_times_cpu_e": cpu_e_stored,
                        "network": network_prefix,
                    })
    
    if duration_mismatches:
        results["durations_match_profiled"] = False
        results["warnings"].append(f"Found {len(duration_mismatches)} duration mismatches with profiled data")
        results["duration_mismatch_details"] = duration_mismatches
    
    # Validation 4: Check machine assignments are valid
    invalid_assignments = []
    for i, op in enumerate(workload.operations):
        assigned_combo_idx = np.argmax(alpha[i])
        if assigned_combo_idx >= len(machine_combinations):
            invalid_assignments.append({
                "operation": op.operation_name or op.operation_id or f"op{i}",
                "index": i,
                "assigned_combo_idx": assigned_combo_idx,
                "max_combo_idx": len(machine_combinations) - 1,
            })
        elif np.sum(alpha[i]) != 1.0:
            invalid_assignments.append({
                "operation": op.operation_name or op.operation_id or f"op{i}",
                "index": i,
                "alpha_sum": np.sum(alpha[i]),
                "expected": 1.0,
            })
    
    if invalid_assignments:
        results["machine_assignments_valid"] = False
        results["errors"].append(f"Found {len(invalid_assignments)} invalid machine assignments")
    
    # Validation 5: Check for overlaps on same machine
    overlaps = []
    for i in range(num_operations):
        for j in range(i + 1, num_operations):
            # Check if operations use overlapping machines
            combo_i = machine_combinations[np.argmax(alpha[i])]
            combo_j = machine_combinations[np.argmax(alpha[j])]
            
            if set(combo_i) & set(combo_j):  # Overlapping machines
                # Check if they overlap in time
                dur_i = workload.operations[i].get_duration_for_combination(
                    np.argmax(alpha[i]),
                    machine_combinations,
                    machines
                )
                dur_j = workload.operations[j].get_duration_for_combination(
                    np.argmax(alpha[j]),
                    machine_combinations,
                    machines
                )
                
                finish_i = t[i] + dur_i
                finish_j = t[j] + dur_j
                
                # Check for overlap
                if not (finish_i <= t[j] or finish_j <= t[i]):
                    overlaps.append({
                        "operation1": workload.operations[i].operation_name or workload.operations[i].operation_id or f"op{i}",
                        "operation2": workload.operations[j].operation_name or workload.operations[j].operation_id or f"op{j}",
                        "machine": list(set(combo_i) & set(combo_j))[0],
                        "op1_time": (t[i], finish_i),
                        "op2_time": (t[j], finish_j),
                    })
    
    if overlaps:
        results["no_overlaps"] = False
        results["errors"].append(f"Found {len(overlaps)} overlaps on same machines")
        results["overlap_details"] = overlaps
    
    # Collect operation details for report
    for i, op in enumerate(workload.operations):
        op_name = op.operation_name or op.operation_id or f"op{i}"
        assigned_combo_idx = np.argmax(alpha[i])
        assigned_combo = machine_combinations[assigned_combo_idx]
        duration = op.get_duration_for_combination(
            assigned_combo_idx,
            machine_combinations,
            machines
        )
        
        dispatch_id = None
        if op_name in dispatches:
            dispatch_id = dispatches[op_name].get("id")
        
        results["operation_details"].append({
            "index": i,
            "name": op_name,
            "dispatch_id": dispatch_id,
            "start_time": float(t[i]),
            "duration": float(duration),
            "finish_time": float(t[i] + duration),
            "machine": assigned_combo[0] if len(assigned_combo) == 1 else "+".join(assigned_combo),
        })
    
    # Write report to file (pass workload data for detailed reporting)
    write_validation_report(results, output_file, workload, t, alpha, dispatches, profiled_times_p, profiled_times_e, machine_combinations, machines)
    
    # Determine overall validity
    is_valid = (
        results["all_operations_scheduled"]
        and results["predecessor_constraints_satisfied"]
        and results["machine_assignments_valid"]
        and results["no_overlaps"]
    )
    
    return is_valid, results


def write_validation_report(results: Dict, output_file: str, workload: Workload = None, t: np.ndarray = None, alpha: np.ndarray = None, dispatches: Dict = None, profiled_times_p: Dict = None, profiled_times_e: Dict = None, machine_combinations: List = None, machines: List = None):
    """Write validation results to a file."""
    import os
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SCHEDULE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary
        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"All operations scheduled: {'PASS' if results['all_operations_scheduled'] else 'FAIL'}\n")
        f.write(f"Predecessor constraints satisfied: {'PASS' if results['predecessor_constraints_satisfied'] else 'FAIL'}\n")
        f.write(f"Durations match profiled data: {'PASS' if results['durations_match_profiled'] else 'FAIL'}\n")
        f.write(f"Machine assignments valid: {'PASS' if results['machine_assignments_valid'] else 'FAIL'}\n")
        f.write(f"No overlaps: {'PASS' if results['no_overlaps'] else 'FAIL'}\n")
        f.write("\n")
        
        # Errors
        if results["errors"]:
            f.write("ERRORS\n")
            f.write("-" * 80 + "\n")
            for error in results["errors"]:
                f.write(f"  - {error}\n")
            f.write("\n")
        
        # Warnings
        if results["warnings"]:
            f.write("WARNINGS\n")
            f.write("-" * 80 + "\n")
            for warning in results["warnings"]:
                f.write(f"  - {warning}\n")
            f.write("\n")
        
        # Predecessor violations
        if not results["predecessor_constraints_satisfied"]:
            f.write("PREDECESSOR CONSTRAINT VIOLATIONS\n")
            f.write("-" * 80 + "\n")
            # Find violations in operation_details or errors
            f.write("  (See errors section above)\n")
            f.write("\n")
        
        # Duration mismatches (detailed)
        if "duration_mismatch_details" in results and results["duration_mismatch_details"]:
            f.write("DURATION MISMATCHES (Detailed)\n")
            f.write("-" * 80 + "\n")
            f.write("NOTE: Mismatches may occur when the same dispatch_id appears in multiple networks.\n")
            f.write("The profiled data may be from a different network than the operation.\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Operation':<35} {'Dispatch ID':<12} {'Network':<12} {'Machine':<8} {'Scheduled':<12} {'Profiled':<12} {'Diff':<12} {'Stored P/E':<12}\n")
            f.write("-" * 80 + "\n")
            mismatches = results["duration_mismatch_details"]
            for mm in sorted(mismatches, key=lambda x: x["difference"], reverse=True)[:30]:  # Top 30
                stored_val = mm.get("processing_times_cpu_p") or mm.get("processing_times_cpu_e")
                stored_str = f"{stored_val:.3f}" if stored_val is not None else "N/A"
                network = mm.get("network", "unknown")
                f.write(
                    f"{mm['operation']:<35} "
                    f"{mm['dispatch_id']:<12} "
                    f"{network:<12} "
                    f"{mm['machine']:<8} "
                    f"{mm['scheduled']:<12.3f} "
                    f"{mm['profiled']:<12.3f} "
                    f"{mm['difference']:<12.3f} "
                    f"{stored_str:<12}\n"
                )
            if len(mismatches) > 30:
                f.write(f"\n... and {len(mismatches) - 30} more mismatches\n")
            f.write("\n")
        
        # Overlaps (detailed)
        if "overlap_details" in results and results["overlap_details"]:
            f.write("OVERLAPS ON SAME MACHINES (Detailed)\n")
            f.write("-" * 80 + "\n")
            f.write("These operations overlap in time on the same machine.\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Operation 1':<35} {'Operation 2':<35} {'Machine':<10} {'Op1 Time':<20} {'Op2 Time':<20}\n")
            f.write("-" * 80 + "\n")
            for overlap in results["overlap_details"]:
                op1_time_str = f"[{overlap['op1_time'][0]:.3f}, {overlap['op1_time'][1]:.3f}]"
                op2_time_str = f"[{overlap['op2_time'][0]:.3f}, {overlap['op2_time'][1]:.3f}]"
                f.write(
                    f"{overlap['operation1']:<35} "
                    f"{overlap['operation2']:<35} "
                    f"{overlap['machine']:<10} "
                    f"{op1_time_str:<20} "
                    f"{op2_time_str:<20}\n"
                )
            f.write("\n")
        
        # Operation details
        f.write("OPERATION DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Name':<30} {'Dispatch ID':<12} {'Start':<10} {'Duration':<10} {'Finish':<10} {'Machine':<10}\n")
        f.write("-" * 80 + "\n")
        for op_detail in results["operation_details"]:
            dispatch_id_str = str(op_detail["dispatch_id"]) if op_detail["dispatch_id"] is not None else "N/A"
            f.write(
                f"{op_detail['index']:<8} "
                f"{op_detail['name']:<30} "
                f"{dispatch_id_str:<12} "
                f"{op_detail['start_time']:<10.3f} "
                f"{op_detail['duration']:<10.3f} "
                f"{op_detail['finish_time']:<10.3f} "
                f"{op_detail['machine']:<10}\n"
            )
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Report generated at: {__import__('datetime').datetime.now()}\n")
        f.write("=" * 80 + "\n")

