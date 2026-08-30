from workload import Workload, Job, Operation, Window
import numpy as np
import json
import os
from typing import Tuple, Dict, List, Optional

from profile_metrics import profile_based_horizon_ms

# Hardware constants — SpacemiT x60
CPU_P = "CPU_P"
CPU_E = "CPU_E"


def machine_type_prefix(machine_name: str) -> str:
    """Return the type prefix of a per-core machine name, e.g. 'CPU_P#2' -> 'CPU_P'."""
    return machine_name.split("#")[0] if "#" in machine_name else machine_name


def expand_machine_core_counts_to_list(machine_core_counts: dict[str, int]) -> list[str]:
    """Expand {'CPU_P': 4, 'CPU_E': 5} into ['CPU_P#0', ..., 'CPU_P#3', 'CPU_E#0', ..., 'CPU_E#4']."""
    machines = []
    for machine_type, count in machine_core_counts.items():
        for i in range(count):
            machines.append(f"{machine_type}#{i}")
    return machines


def build_machine_combinations(machine_core_counts: dict[str, int]) -> tuple[list[str], list[list[str]]]:
    """
    Build the full machines list and cumulative core-group combinations.

    For {'CPU_P': 4, 'CPU_E': 4} returns:
      machines = ['CPU_P#0', ..., 'CPU_P#3', 'CPU_E#0', ..., 'CPU_E#3']
      combinations = [
          ['CPU_P#0'],                                  # 1 P-core  → topo_0
          ['CPU_P#0', 'CPU_P#1'],                       # 2 P-cores → topo_0_1
          ['CPU_P#0', 'CPU_P#1', 'CPU_P#2'],            # 3 P-cores → topo_0_1_2
          ['CPU_P#0', 'CPU_P#1', 'CPU_P#2', 'CPU_P#3'], # 4 P-cores → topo_0_1_2_3
          ['CPU_E#0'],                                  # 1 E-core  → topo_0
          ...
      ]

    Each combination only contains cores from the same processor type.
    """
    machines = expand_machine_core_counts_to_list(machine_core_counts)
    combinations = []
    for machine_type, count in machine_core_counts.items():
        cores = [f"{machine_type}#{i}" for i in range(count)]
        for n in range(1, count + 1):
            combinations.append(cores[:n])
    return machines, combinations


def topo_tag_for_combination(combo: list[str]) -> str:
    """Return the topo tag that corresponds to a combination size, e.g. 3 cores → 'topo_0_1_2'."""
    n = len(combo)
    return "topo_" + "_".join(str(i) for i in range(n))


def resolve_dispatch_deps_path(repo_base_path: str, dispatch_deps_path: str) -> str:
    """
    Resolve a dispatch dependency JSON path across current and legacy layouts.
    """
    if not dispatch_deps_path:
        return ""

    raw_path = dispatch_deps_path.strip()
    if os.path.isabs(raw_path):
        return raw_path if os.path.exists(raw_path) else ""

    normalized = raw_path.lstrip("./")
    candidates: List[str] = [
        os.path.join(repo_base_path, normalized),
    ]

    legacy_prefix = "src/pytorch_workload/samples/"
    if normalized.startswith(legacy_prefix):
        candidates.append(
            os.path.join(
                repo_base_path,
                "xpu-rt",
                "pytorch_workload",
                "samples",
                normalized[len(legacy_prefix):],
            )
        )

    old_merlin_prefix = "merlin/samples/robotic-NN/pytorch_workload/computation_graph/"
    if normalized.startswith(old_merlin_prefix):
        candidates.append(
            os.path.join(
                repo_base_path,
                "xpu-rt",
                "pytorch_workload",
                "samples",
                normalized[len(old_merlin_prefix):],
            )
        )

    # Broad fallback for stale "src/" prefix.
    if normalized.startswith("src/"):
        candidates.append(os.path.join(repo_base_path, normalized[len("src/"):]))

    # Final fallback: same filename under canonical samples directory.
    candidates.append(
        os.path.join(
            repo_base_path,
            "xpu-rt",
            "pytorch_workload",
            "samples",
            os.path.basename(normalized),
        )
    )

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate

    return ""

def generate_syn_transfer_times(n_machines: int, max_transfer_time: int=500) -> np.ndarray:
    """
    Generates a symmetric matrix of transfer times between machines.
    """
    
    transfer_times = np.random.randint(0, max_transfer_time, (n_machines, n_machines))
    transfer_times = (transfer_times + transfer_times.T) / 2
    np.fill_diagonal(transfer_times, 0)
    return transfer_times

def create_sequential_job(operations: list[Operation]) -> Job:
    """
    From the list of operations, creates a job of sequentially dependent operations.
    Each operation depends on the previous one (single predecessor chain for backward compatibility).
    """
    
    for i in range(len(operations) - 1):
        operations[i+1].add_predecessor(operations[i])
    
    return Job(operations)

def generate_syn_workload(n_operations: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Workload:
    """
    Generates a synthetic workload for the scheduling problem.

    Parameters:
    - n_operations: number of operations
    - n_machines: number of machines available
    - transfer_times: a matrix of transfer times between machines
    - processing_time_range: a tuple of the minimum and maximum processing times for each operation

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(processing_time_range[0], processing_time_range[1]) for _ in range(n_machines)]
        operations.append(Operation(processing_times))
    machines = [f'machine_{i}' for i in range(n_machines)]
    workload = Workload(operations, machines, transfer_times)
    return workload

def generate_syn_window(n_operations: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Window:
    """
    Generates a synthetic window for the scheduling problem.

    Parameters:
    - n_operations: number of operations
    - n_machines: number of machines available
    - transfer_times: a matrix of transfer times between machines

    Returns:
    - workload: a Workload object containing the synthetic operations
    """
    
    operations = []
    for _ in range(n_operations):
        processing_times = [np.random.randint(processing_time_range[0], processing_time_range[1]) for _ in range(n_machines)]
        operations.append(Operation(processing_times))
    machines = [f'machine_{i}' for i in range(n_machines)]
    expected_time = sum([np.mean(operation.get_durations()) for operation in operations])
    window = Window(expected_time, operations, machines, transfer_times)
    return window

def create_syn_sequential_workload(n_jobs: int, n_operations_per_job: int, n_machines: int, transfer_times: np.ndarray, processing_time_range: Tuple[float, float]=(50, 150)) -> Workload:
    # Create a workload
    machines = [f'machine_{i}' for i in range(n_machines)]

    operations = [[] for _ in range(n_jobs)]
    for i in range(n_jobs):
        for _ in range(n_operations_per_job):
            processing_times = [np.random.randint(50, 150) for _ in range(n_operations_per_job)]
            operations[i].append(Operation(processing_times))

    jobs = [create_sequential_job(ops) for ops in operations]

    workload_operations = []
    for job in jobs:
        workload_operations.extend(job.get_operations())
    workload = Workload(workload_operations, machines, transfer_times)
    return workload

def create_workload_from_dependencies(
    dispatch_data: Dict,
    processing_times: Dict[str, List[float]],
    machines: List[str],
    transfer_times: np.ndarray
) -> Workload:
    """
    Creates a workload from a dependency graph structure (like dronet_dispatch_deps.json).
    
    Parameters:
    - dispatch_data: Dictionary with 'dispatches' key containing dispatch information
    - processing_times: Dictionary mapping dispatch IDs to list of processing times for each machine
    - machines: List of machine names
    - transfer_times: Matrix of transfer times between machines
    
    Returns:
    - Workload object with operations linked according to dependencies
    """
    dispatches = dispatch_data.get('dispatches', {})
    
    # Create Operation objects for each dispatch
    operations_map = {}
    # Determine job_id: operations with no dependencies within this workload start a new job
    # For a single workload, all operations belong to job_id 0
    job_id = 0
    
    for dispatch_name, dispatch_info in dispatches.items():
        # Get processing times for this dispatch (check by name first, then by ID for backward compatibility)
        if dispatch_name in processing_times:
            proc_times = processing_times[dispatch_name]
        else:
            dispatch_id = dispatch_info.get('id', dispatch_name)
            if dispatch_id in processing_times:
                proc_times = processing_times[dispatch_id]
            else:
                # Generate random processing times if not provided
                proc_times = [np.random.randint(50, 150) for _ in machines]
        
        # Extract ID and name from dispatch info
        operation_id = dispatch_info.get('id', None)
        operation_name = dispatch_name
        
        operations_map[dispatch_name] = Operation(
            proc_times, 
            operation_id=operation_id,
            operation_name=operation_name,
            job_id=job_id  # All operations in a single workload belong to the same job
        )
    
    # Set up predecessor relationships
    for dispatch_name, dispatch_info in dispatches.items():
        dependencies = dispatch_info.get('dependencies', [])
        operation = operations_map[dispatch_name]
        
        for dep_name in dependencies:
            if dep_name in operations_map:
                operation.add_predecessor(operations_map[dep_name])
    
    # Create list of operations in order (operations with no dependencies first)
    operations = list(operations_map.values())
    
    # Determine job names: operations with no predecessors start a new job
    job_names = []
    seen_prefixes = set()
    
    for operation in operations:
        if not operation.predecessors:
            # Use operation name to extract job name
            op_name = operation.operation_name or f"dispatch_{len(job_names)}"
            # Extract prefix if it exists (e.g., "dronet_dispatch_0" -> "dronet")
            if '_' in op_name:
                parts = op_name.split('_')
                if len(parts) >= 2:
                    # Take the prefix part (e.g., "dronet" from "dronet_dispatch_0")
                    prefix = parts[0]
                    # Capitalize for better display
                    job_name = prefix.capitalize()
                    # Only add if we haven't seen this prefix yet (avoid duplicates)
                    if job_name not in seen_prefixes:
                        job_names.append(job_name)
                        seen_prefixes.add(job_name)
                else:
                    if op_name not in seen_prefixes:
                        job_names.append(op_name.capitalize())
                        seen_prefixes.add(op_name)
            else:
                if op_name not in seen_prefixes:
                    job_names.append(op_name.capitalize())
                    seen_prefixes.add(op_name)
    
    # If no job names found, create default ones
    if not job_names:
        num_jobs = sum(1 for op in operations if not op.predecessors)
        job_names = [f"Job {i}" for i in range(num_jobs)]
    
    return Workload(operations, machines, transfer_times, job_names=job_names)

def create_workload_from_network_hierarchy(
    networks_data: Dict,
    repo_base_path: str,
    machines: List[str],
    transfer_times: np.ndarray,
    processing_times: Optional[Dict[str, List[float]]] = None,
    p_core_speedup: float = 1.5,
    random_seed: Optional[int] = 0,
    machine_combinations: Optional[List[List[str]]] = None,
) -> Workload:
    """
    Creates a workload from a hierarchical network dependencies structure.

    Handles two levels:
    1. Top level: Networks with dependencies between networks
    2. Sub level: Dispatches within each network with dependencies between dispatches

    Periodic networks (those with 'period' and 'window_duration') are expanded into
    multiple instances with calculated time windows.

    Parameters:
    - networks_data: Dictionary with 'networks' and 'edges' keys
    - repo_base_path: Base path of the repository
    - machines: List of all physical core names
    - transfer_times: Matrix of transfer times between machines
    - processing_times: Optional dict mapping prefixed dispatch names to processing times.
                       Values are per-combination (one entry per machine_combination).
                       If None, generates synthetic times using p_core_speedup.
    - p_core_speedup: Speedup factor for P-core vs E-core (synthetic generation)
    - random_seed: Seed for synthetic runtime generation. None = nondeterministic.
    - machine_combinations: List of core groupings (e.g. [[CPU_P#0], [CPU_P#0,CPU_P#1], ...]).
                           If None, each machine becomes a singleton combination.
    """
    networks = networks_data.get('networks', {})
    network_edges = networks_data.get('edges', [])

    # Random number generator for synthetic processing times
    rng = np.random.default_rng(random_seed)

    effective_combos = machine_combinations if machine_combinations is not None else [[m] for m in machines]

    def _synthetic_proc_times() -> List[float]:
        """Generate synthetic per-combination processing times."""
        p_ms = float(rng.uniform(2.0, 10.0))
        times = []
        for combo in effective_combos:
            core_type = machine_type_prefix(combo[0])
            base = p_ms if core_type == CPU_P else p_ms * p_core_speedup
            # Rough scaling: more cores = faster (but not perfectly linear)
            times.append(base / len(combo))
        return times

    def _estimate_num_periodic_instances() -> Dict[str, int]:
        """
        Heuristic to estimate how many instances to create for each periodic network.

        For now:
          1) Compute a worst-case horizon H (ms):
               - If hardware.profile + profiled results.csv can be resolved, prefer
                 H = S_np / (1 - F_p)  where
                   S_np = sum of worst-case layer times (max over CPU_P/CPU_E per
                          dispatch node) for all non-periodic, non-window-slice networks
                   F_p  = max over periodic workloads of (S_p / T) — utilization
                          fraction over the period (NOT window). Periodic tasks
                          consume F_p of CPU per unit time on average; nonperiodic
                          gets (1 - F_p), so its wall time is S_np / (1 - F_p).
               - Else fall back to:
                   H = 2.0 * sum_over_nonperiodic_ops( worst_machine_duration(op) )
             where worst_machine_duration(op) is the max duration across machines.
          2) For each periodic network with period T, set:
               num_instances = ceil(H / T), at least 1.
        """
        total_worst_nonperiodic = 0.0

        for net_id, net_info in networks.items():
            period = net_info.get("period", None)
            window_duration = net_info.get("window_duration", None)
            # Skip periodic networks when computing the non-periodic horizon
            if period is not None and window_duration is not None:
                continue

            dispatch_deps_path = net_info.get("dispatch_deps_path", "")
            full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
            if not os.path.exists(full_dispatch_path):
                continue

            try:
                with open(full_dispatch_path, "r") as f:
                    dispatch_data = json.load(f)
            except Exception:
                continue

            dispatches = dispatch_data.get("dispatches", {})
            base_prefix = f"{net_id}_"

            for dispatch_name, dispatch_info in dispatches.items():
                prefixed_name = f"{base_prefix}{dispatch_name}"

                # Determine processing times for this dispatch
                if processing_times and prefixed_name in processing_times:
                    proc_times = processing_times[prefixed_name]
                else:
                    proc_times = _synthetic_proc_times()

                if not proc_times:
                    continue

                worst_dur = max(proc_times)
                total_worst_nonperiodic += float(worst_dur)

        # If there are no non-periodic operations there is no horizon to derive
        # instance counts from, so fall back to 1 instance per periodic network.
        # An explicit per-network `num_instances` still wins: it is the caller
        # pinning the count directly (same override as below), and silently
        # collapsing a workload that asks for 6 instances down to 1 makes a
        # purely-periodic workload unschedulable rather than merely unbounded.
        if total_worst_nonperiodic <= 0.0:
            periodic_counts: Dict[str, int] = {}
            for net_id, net_info in networks.items():
                period = net_info.get("period", None)
                window_duration = net_info.get("window_duration", None)
                if period is not None and window_duration is not None:
                    forced = net_info.get("num_instances", None)
                    periodic_counts[net_id] = (
                        forced if isinstance(forced, int) and forced > 0 else 1
                    )
            return periodic_counts

        profile_horizon = profile_based_horizon_ms(networks_data, repo_base_path)
        # The min-per-op based horizon (profile_horizon) is a LOWER bound
        # on the makespan — it assumes every op runs on its best HW with
        # ideal parallelism.  In practice the scheduler can't always reach
        # that bound: dependency chains serialise large fractions of the
        # DAG, and ops without per-HW alternatives (e.g. yolov8 silu_s8
        # falling back to scalar on every backend) cap concurrency.
        # `total_worst_nonperiodic` (sum of max-per-op for all non-periodic
        # ops) is the corresponding UPPER bound — it's what you'd see if
        # every op landed on its slowest available HW with no overlap.
        # Take the max so periodic-instance counts always cover at least
        # the worst-case makespan.  Without this, periodic networks are
        # truncated short of the actual schedule (caller sees a periodic
        # workload that "ends" before the makespan does).
        if profile_horizon is not None and profile_horizon > 0.0:
            horizon = max(float(profile_horizon), total_worst_nonperiodic)
        else:
            horizon = 2.0 * total_worst_nonperiodic
        periodic_counts: Dict[str, int] = {}
        for net_id, net_info in networks.items():
            period = net_info.get("period", None)
            window_duration = net_info.get("window_duration", None)
            if period is None or window_duration is None:
                continue
            try:
                T = float(period)
            except (TypeError, ValueError):
                continue
            if T <= 0:
                continue
            num_instances = int(np.ceil(horizon / T))
            # Per-network override: lets the toplevel JSON cap periodic
            # instance count directly. Useful when the horizon heuristic
            # over-estimates (e.g. profile data is being pinned per-network
            # so the cross-product worst-case bloats).
            forced = net_info.get("num_instances", None)
            if isinstance(forced, int) and forced > 0:
                num_instances = forced
            periodic_counts[net_id] = max(1, num_instances)

        return periodic_counts

    # Estimate number of instances for each periodic network
    periodic_num_instances = _estimate_num_periodic_instances()
    
    expanded_networks: Dict[str, Dict] = {}
    periodic_network_to_instances: Dict[str, List[str]] = {}  # Maps periodic network -> list of instance identifiers
    periodic_base_to_instances: Dict[str, str] = {}  # Maps instance identifier -> base periodic network identifier

    # Pre-generate processing times for periodic networks to ensure consistency across instances
    periodic_processing_times_cache: Dict[Tuple[str, str], List[float]] = {}  # (base_network_id, dispatch_name) -> proc_times

    # Track all user-provided base IDs so periodic instance IDs don't
    # collide with another network's id.  Previously we used `base_id + i`
    # which overlaps when a non-periodic network has id == base_id+i;
    # the resulting shared job_id silently merged operations under one
    # job (and confused downstream printers / plotters that key on
    # job_names[job_id]).  Use a counter past the max user-provided id.
    _user_ids = [int(n.get('id', 0)) for n in networks.values()]
    next_periodic_instance_id = (max(_user_ids) + 1) if _user_ids else 0

    for network_identifier, network_info in networks.items():
        # Check if this network is periodic
        period = network_info.get('period', None)
        window_duration = network_info.get('window_duration', None)
        start_time = network_info.get('start_time', 0)
        
        if period is not None and window_duration is not None:
            # This is a periodic network - pre-generate processing times for consistency
            dispatch_deps_path = network_info.get('dispatch_deps_path', '')
            full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
            
            if os.path.exists(full_dispatch_path):
                with open(full_dispatch_path, 'r') as f:
                    dispatch_data = json.load(f)
                dispatches = dispatch_data.get('dispatches', {})
                
                # Generate processing times once for this periodic network
                for dispatch_name, dispatch_info in dispatches.items():
                    cache_key = (network_identifier, dispatch_name)
                    
                    # Check if we should use provided processing times or generate synthetic
                    base_prefix = f"{network_identifier}_"
                    prefixed_dispatch_name = f"{base_prefix}{dispatch_name}"
                    
                    if processing_times and prefixed_dispatch_name in processing_times:
                        periodic_processing_times_cache[cache_key] = processing_times[prefixed_dispatch_name]
                    else:
                        periodic_processing_times_cache[cache_key] = _synthetic_proc_times()
            
            # Expand periodic network into multiple instances
            base_id = network_info.get('id', 0)
            base_identifier = network_info.get('identifier', network_identifier)
            instance_identifiers = []

            num_instances = periodic_num_instances.get(network_identifier, 1)
            for i in range(num_instances):
                instance_identifier = f"{network_identifier}{i}"
                instance_min_start_t = start_time + i * period
                instance_max_end_t = start_time + i * period + window_duration

                # Create instance network info.  The first instance keeps
                # the base network id so the user-facing legend still says
                # "<network>0".  Later instances get fresh ids past the
                # max user-provided id, avoiding collisions with another
                # network whose id happens to equal base_id+i.
                instance_info = network_info.copy()
                if i == 0:
                    instance_info['id'] = base_id
                else:
                    instance_info['id'] = next_periodic_instance_id
                    next_periodic_instance_id += 1
                instance_info['identifier'] = f"{base_identifier}{i}"
                instance_info['min_start_t'] = instance_min_start_t
                instance_info['max_end_t'] = instance_max_end_t
                # Remove periodic fields as they're now expanded
                instance_info.pop('period', None)
                instance_info.pop('window_duration', None)
                instance_info.pop('start_time', None)
                
                expanded_networks[instance_identifier] = instance_info
                instance_identifiers.append(instance_identifier)
                periodic_base_to_instances[instance_identifier] = network_identifier
            
            periodic_network_to_instances[network_identifier] = instance_identifiers
        else:
            # Regular network - add as-is
            expanded_networks[network_identifier] = network_info
    
    # Expand edges: if an edge references a periodic network, expand it to all instances
    expanded_edges = []
    for edge in network_edges:
        from_network = edge.get('from')
        to_network = edge.get('to')
        
        # Check if networks are periodic and expand them
        from_instances = periodic_network_to_instances.get(from_network, [from_network])
        to_instances = periodic_network_to_instances.get(to_network, [to_network])
        
        # For now, if both are periodic, create edges between corresponding instances
        # (instance 0 -> instance 0, instance 1 -> instance 1, etc.)
        # This can be customized later
        if len(from_instances) > 1 and len(to_instances) > 1:
            # Both are periodic - create edges between corresponding instances
            for i in range(min(len(from_instances), len(to_instances))):
                expanded_edges.append({
                    'from': from_instances[i],
                    'to': to_instances[i]
                })
        elif len(from_instances) > 1:
            # Only from_network is periodic - create edges from all instances to to_network
            for from_inst in from_instances:
                expanded_edges.append({
                    'from': from_inst,
                    'to': to_network
                })
        elif len(to_instances) > 1:
            # Only to_network is periodic - create edges from from_network to all instances
            for to_inst in to_instances:
                expanded_edges.append({
                    'from': from_network,
                    'to': to_inst
                })
        else:
            # Neither is periodic - keep original edge
            expanded_edges.append(edge)
    
    # Map to store operations for each network
    network_operations_map: Dict[str, List[Operation]] = {}
    # Map to store all operations by their prefixed names
    all_operations_map: Dict[str, Operation] = {}
    # Map network identifier to its job_id
    network_job_ids: Dict[str, int] = {}
    # Map network identifier to its job name
    network_job_names: Dict[str, str] = {}
    
    # First pass: Load each network's dispatch graph and create operations
    # Now iterate over expanded_networks instead of networks
    for network_identifier, network_info in expanded_networks.items():
        network_id = network_info.get('id', 0)
        dispatch_deps_path = network_info.get('dispatch_deps_path', '')
        
        # Resolve path relative to repo base
        full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
        
        if not os.path.exists(full_dispatch_path):
            raise FileNotFoundError(
                "Dispatch dependencies file not found for "
                f"'{dispatch_deps_path}' (repo base: {repo_base_path})"
            )
        
        # Load dispatch dependencies JSON
        with open(full_dispatch_path, 'r') as f:
            dispatch_data = json.load(f)
        
        dispatches = dispatch_data.get('dispatches', {})
        network_prefix = f"{network_identifier}_"
        
        # Store job_id and job_name for this network
        network_job_ids[network_identifier] = network_id
        network_job_names[network_identifier] = network_info.get('identifier', network_identifier)
        
        # Extract time constraints from network info (if present)
        network_min_start_t = network_info.get('min_start_t', None)
        network_max_end_t = network_info.get('max_end_t', None)
        
        # Create operations for dispatches in this network
        network_ops_map: Dict[str, Operation] = {}
        
        for dispatch_name, dispatch_info in dispatches.items():
            # Create prefixed dispatch name to avoid conflicts
            prefixed_dispatch_name = f"{network_prefix}{dispatch_name}"
            
            # Get processing times
            # Check if this is an instance of a periodic network - if so, use cached times
            base_periodic_network = periodic_base_to_instances.get(network_identifier)
            if base_periodic_network is not None:
                # This is a periodic instance - use pre-generated processing times
                cache_key = (base_periodic_network, dispatch_name)
                if cache_key in periodic_processing_times_cache:
                    proc_times = periodic_processing_times_cache[cache_key]
                else:
                    proc_times = _synthetic_proc_times()
            elif processing_times and prefixed_dispatch_name in processing_times:
                proc_times = processing_times[prefixed_dispatch_name]
            else:
                proc_times = _synthetic_proc_times()
            
            # Extract dispatch ID and create operation
            # Inherit time constraints from network if present
            dispatch_id = dispatch_info.get('id', None)
            operation = Operation(
                proc_times,
                operation_id=dispatch_id,
                operation_name=prefixed_dispatch_name,
                job_id=network_id,
                min_start_t=network_min_start_t,
                max_end_t=network_max_end_t
            )
            
            network_ops_map[dispatch_name] = operation
            all_operations_map[prefixed_dispatch_name] = operation
        
        # Set up dispatch-level dependencies within this network
        for dispatch_name, dispatch_info in dispatches.items():
            dependencies = dispatch_info.get('dependencies', [])
            operation = network_ops_map[dispatch_name]
            
            for dep_name in dependencies:
                if dep_name in network_ops_map:
                    # Dependency is within the same network
                    operation.add_predecessor(network_ops_map[dep_name])
                elif f"{network_prefix}{dep_name}" in all_operations_map:
                    # Dependency might be from a prefixed name (shouldn't happen in normal case)
                    operation.add_predecessor(all_operations_map[f"{network_prefix}{dep_name}"])
        
        # Store operations for this network
        network_operations_map[network_identifier] = list(network_ops_map.values())
    
    # Second pass: Set up network-level dependencies
    # For each edge (from_network -> to_network):
    #   - Find last operations in from_network (operations with no successors within that network)
    #   - Find first operations in to_network (operations with no predecessors within that network)
    #   - Make first operations of to_network depend on last operations of from_network
    
    # Use expanded_edges instead of network_edges
    for edge in expanded_edges:
        from_network = edge.get('from')
        to_network = edge.get('to')
        
        if from_network not in network_operations_map or to_network not in network_operations_map:
            continue
        
        from_ops = network_operations_map[from_network]
        to_ops = network_operations_map[to_network]
        
        # Find last operations in from_network (not predecessors of any other operation in that network)
        from_last_ops = []
        from_ops_set = set(from_ops)
        for op in from_ops:
            is_predecessor = False
            for other_op in from_ops:
                if op in other_op.predecessors:
                    is_predecessor = True
                    break
            if not is_predecessor:
                from_last_ops.append(op)
        
        # If no explicit last operations found, use all operations (fallback)
        if not from_last_ops:
            from_last_ops = from_ops
        
        # Find first operations in to_network (operations with no predecessors within that network)
        # Check only within-network predecessors by comparing job_id
        to_network_job_id = network_job_ids.get(to_network, 0)
        to_first_ops = [
            op for op in to_ops 
            if not any(pred.job_id == to_network_job_id for pred in op.predecessors)
        ]
        
        # If no first operations found, use the first operation (fallback)
        if not to_first_ops:
            to_first_ops = [to_ops[0]] if to_ops else []
        
        # Add dependencies: each first operation of to_network depends on all last operations of from_network
        for to_op in to_first_ops:
            for from_op in from_last_ops:
                to_op.add_predecessor(from_op)
    
    # Collect all operations
    all_operations = []
    for network_ops in network_operations_map.values():
        all_operations.extend(network_ops)
    
    # Create job names list (ordered by job_id)
    # Group networks by job_id (in case multiple networks share the same job_id)
    job_id_to_names: Dict[int, List[str]] = {}
    for network_identifier, job_id in network_job_ids.items():
        if job_id not in job_id_to_names:
            job_id_to_names[job_id] = []
        job_name = network_job_names[network_identifier]
        if job_name not in job_id_to_names[job_id]:
            job_id_to_names[job_id].append(job_name)
    
    # Create ordered job names list
    max_job_id = max(network_job_ids.values()) if network_job_ids else 0
    job_names = []
    for job_id in range(max_job_id + 1):
        if job_id in job_id_to_names:
            # Use first name if multiple networks share same job_id
            job_names.append(job_id_to_names[job_id][0])
        else:
            job_names.append(f"Job {job_id}")
    
    return Workload(all_operations, machines, transfer_times, job_names=job_names, machine_combinations=machine_combinations)
