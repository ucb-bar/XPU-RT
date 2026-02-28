import numpy as np
from workload import Workload

def overlap_fixer(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    Resolves overlaps by pushing them forward in time
    @return: updated t that is free of overlaps and respects the precedence constraints
    """
    # Respect per-machine concurrency limits from workload.get_machines() (dict: name -> limit)
    machine_limits = workload.get_machines()
    machine_combinations = workload.get_machine_combinations()
    # map machine name -> index in workload.machines list (for any transfer-time lookups elsewhere)
    machine_name_to_idx = {name: idx for idx, name in enumerate(workload.machines)}

    num_ops = len(t)

    # Repeat a few passes to resolve cascading adjustments
    for _ in range(num_ops):
        # iterate operations in increasing start-time order
        order = sorted(range(num_ops), key=lambda x: t[x])
        for i in order:
            combo_i = int(np.argmax(alpha[i]))
            dur_i = workload.operations[i].get_duration_for_combination(combo_i, machine_combinations, workload.machines)
            start_i = t[i]
            end_i = start_i + dur_i

            new_start = start_i

            # For each machine used by operation i, check current overlapping operations
            for m in machine_combinations[combo_i]:
                limit = int(machine_limits.get(m, 1))
                # collect end times of ops that overlap with i on machine m
                overlapping_ends = []
                for j in range(num_ops):
                    if j == i:
                        continue
                    combo_j = int(np.argmax(alpha[j]))
                    if m not in machine_combinations[combo_j]:
                        continue
                    dur_j = workload.operations[j].get_duration_for_combination(combo_j, machine_combinations, workload.machines)
                    start_j = t[j]
                    end_j = start_j + dur_j
                    # intervals overlap?
                    if start_i < end_j and start_j < end_i:
                        overlapping_ends.append(end_j)

                # If number of overlapping operations (excluding i) is already >= limit, push i
                if len(overlapping_ends) >= limit:
                    # push i to after the earliest-finishing overlapping operation
                    earliest_end = min(overlapping_ends)
                    new_start = max(new_start, earliest_end)

            if new_start > start_i:
                t[i] = new_start

    return t

def count_overlaps(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    @return: number of overlaps in the schedule
    """
    machine_limits = workload.get_machines()
    machine_combinations = workload.get_machine_combinations()
    num_ops = len(t)
    violations = 0

    # For each machine, count operations that exceed concurrency limit
    for m, limit in machine_limits.items():
        limit = int(limit)
        # build intervals for ops that use machine m
        intervals = []  # (start, end, idx)
        for i in range(num_ops):
            combo_i = int(np.argmax(alpha[i]))
            if m not in machine_combinations[combo_i]:
                continue
            dur_i = workload.operations[i].get_duration_for_combination(combo_i, machine_combinations, workload.machines)
            intervals.append((t[i], t[i] + dur_i, i))

        # for each operation, count overlaps and increment violations when exceeding limit
        for start_i, end_i, i in intervals:
            overlap_count = 0
            for start_j, end_j, j in intervals:
                if i == j:
                    continue
                if start_i < end_j and start_j < end_i:
                    overlap_count += 1
            if overlap_count >= limit:
                violations += 1

    return violations