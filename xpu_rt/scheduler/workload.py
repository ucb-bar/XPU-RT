import numpy as np

class Operation:
    """
    Lowest level of a schedulable instance. An operation has a processing time and potentially multiple predecessors.
    Each operation has a must havea   processing time for each machine in the workload.
    """
    def __init__(self, processing_times: list[float], predecessors=None, operation_id=None, operation_name=None, job_id=None, min_start_t=None, max_end_t=None, deadline_us=None, skip_allowed=False, processing_times_by_pred=None, infeasible_combinations=None):
        self.processing_times = processing_times
        # Per-(predecessor combination, current combination) cost tensor.
        # Shape: dict[(k_pred_idx, k_curr_idx) -> duration_us]. When set, the
        # MOSEK MILP linearises alpha[i_pred, k_pred] * alpha[i, k_curr] into
        # gamma[i, k_pred, k_curr] and uses cost[k_pred, k_curr] for the
        # effective duration of op i, capturing cross-cluster cache-state
        # penalties (e.g. CPU_P→CPU_E coming hot vs cold). When unset the
        # solver falls back to the 2D processing_times[k_curr] cost.
        self.processing_times_by_pred = processing_times_by_pred or {}
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
        # Time constraints - minimum start time and maximum end time
        self.min_start_t = min_start_t
        self.max_end_t = max_end_t
        # Robotics-deadline support (added by PR5 of the rosy-sundae plan):
        # `deadline_us` is a hard "must finish by" time in the same units as
        # processing_times. `skip_allowed` opts this op into the binary
        # skip-indicator path: if the deadline can't be met, the solver may
        # set s[i]=1, dropping the op from the schedule. If skip_allowed is
        # False, the deadline becomes a hard constraint and infeasible
        # instances surface as MOSEK status=infeasible.
        self.deadline_us = deadline_us
        self.skip_allowed = bool(skip_allowed)
        # Hard-exclusion set of machine combination indices the solver
        # MUST NOT pick for this op. Used by the heterogeneous QNN
        # workflow to forbid (op, backend) cells where the on-board
        # build/run failed — there's no measured cost for those, and we
        # don't want the MILP guessing via large coefficients. The
        # scheduler injects `alpha[i, k] = 0` for each k in this set
        # (see scheduler.py "(2b) infeasibility hard exclusion").
        self.infeasible_combinations = set(infeasible_combinations or ())
    
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
        """
        Returns processing times. For backward compatibility, this returns one duration per machine.
        When used with machine combinations, if combinations are singletons, this maps correctly.
        For non-singleton combinations, use get_duration_for_combination().
        """
        return self.processing_times
    
    def get_duration_for_combination(self, combination_idx: int, machine_combinations: list[list[str]], machines: list[str]) -> float:
        """
        Get the duration for a specific machine combination.

        Processing times are indexed per-combination (one entry per machine combination),
        so this is a direct index lookup.

        @param combination_idx: index of the machine combination
        @param machine_combinations: list of machine combinations (kept for call-site compatibility)
        @param machines: list of all machines (kept for call-site compatibility)
        @return: duration for this combination
        """
        if combination_idx < 0 or combination_idx >= len(self.processing_times):
            raise ValueError(f"Invalid combination index: {combination_idx}")
        return self.processing_times[combination_idx]
    
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
    @param machines: list of machines that can process the operations (for backward compatibility).
    @param transfer_times: matrix of transfer times between machines. transfer_times[i][j] is the time to transfer from machine i to machine j.
    @param job_names: optional list of job names for plotting/display.
    @param machine_combinations: optional list of machine combinations (list of lists). 
                                 If None, each machine in 'machines' becomes a singleton combination for backward compatibility.
                                 Example: [['CPU_P'], ['CPU_E'], ['CPU_P', 'CPU_E']]
    """
    def __init__(self, operations: list[Operation], machines: list[str], transfer_times: np.ndarray, job_names: list[str] = None, machine_combinations: list[list[str]] = None):
        self.operations = operations
        self.machines = machines  # Keep for backward compatibility
        
        # Set up machine_combinations: if not provided, create singleton combinations from machines
        if machine_combinations is None:
            # Backward compatibility: each machine becomes its own combination
            self.machine_combinations = [[m] for m in machines]
        else:
            self.machine_combinations = machine_combinations
        
        # Validate that all machines in combinations exist in the original machines list
        all_machines_in_combinations = set()
        for combo in self.machine_combinations:
            all_machines_in_combinations.update(combo)
        if not all_machines_in_combinations.issubset(set(machines)):
            raise ValueError(f"Machine combinations contain machines not in the machines list: {all_machines_in_combinations - set(machines)}")
        
        self.transfer_times = transfer_times
        self.job_names = job_names if job_names is not None else []
    
    def get_machine_combinations(self) -> list[list[str]]:
        """Returns the list of machine combinations."""
        return self.machine_combinations
    
    def combinations_overlap(self, combo_idx1: int, combo_idx2: int) -> bool:
        """
        Check if two machine combinations overlap (share any machines).
        
        @param combo_idx1: index of first combination
        @param combo_idx2: index of second combination
        @return: True if combinations share at least one machine, False otherwise
        """
        if combo_idx1 < 0 or combo_idx1 >= len(self.machine_combinations):
            return False
        if combo_idx2 < 0 or combo_idx2 >= len(self.machine_combinations):
            return False
        
        set1 = set(self.machine_combinations[combo_idx1])
        set2 = set(self.machine_combinations[combo_idx2])
        return bool(set1 & set2)  # True if intersection is non-empty

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
    def __init__(self, time_frame: float, operations: list[Operation], machines: list[str], transfer_times: np.ndarray, machine_combinations: list[list[str]] = None):
        self.operations = operations
        self.machines = machines  # Keep for backward compatibility
        
        # Set up machine_combinations: if not provided, create singleton combinations from machines
        if machine_combinations is None:
            # Backward compatibility: each machine becomes its own combination
            self.machine_combinations = [[m] for m in machines]
        else:
            self.machine_combinations = machine_combinations
        
        # Validate that all machines in combinations exist in the original machines list
        all_machines_in_combinations = set()
        for combo in self.machine_combinations:
            all_machines_in_combinations.update(combo)
        if not all_machines_in_combinations.issubset(set(machines)):
            raise ValueError(f"Machine combinations contain machines not in the machines list: {all_machines_in_combinations - set(machines)}")
        
        self.time_frame = time_frame
        self.transfer_times = transfer_times
    
    def get_machine_combinations(self) -> list[list[str]]:
        """Returns the list of machine combinations."""
        return self.machine_combinations
    
    def combinations_overlap(self, combo_idx1: int, combo_idx2: int) -> bool:
        """
        Check if two machine combinations overlap (share any machines).
        
        @param combo_idx1: index of first combination
        @param combo_idx2: index of second combination
        @return: True if combinations share at least one machine, False otherwise
        """
        if combo_idx1 < 0 or combo_idx1 >= len(self.machine_combinations):
            return False
        if combo_idx2 < 0 or combo_idx2 >= len(self.machine_combinations):
            return False
        
        set1 = set(self.machine_combinations[combo_idx1])
        set2 = set(self.machine_combinations[combo_idx2])
        return bool(set1 & set2)  # True if intersection is non-empty

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