"""
List-scheduling greedy heuristic for the XPU-RT scheduler.

Companion to `scheduler.py` (the MILP-based optimal scheduler). Given a
:class:`Workload`, picks the (op, machine_combination) pair that gives
the earliest completion time, respecting:

  - intra-job data deps (op.predecessors + transfer_times),
  - machine-combination conflicts (combinations_overlap),
  - periodic / windowed time-bounds (op.min_start_t / op.max_end_t).

This used to live inline in `scripts/run_greedy_schedule.py`. It moved
here when run_greedy was folded into `run_xpurt_schedule.py --solver
greedy`, so both schedulers ship as siblings under `xpu-rt/`.
"""

from __future__ import annotations

import numpy as np

from workload import Workload


def greedy_schedule(workload: Workload) -> tuple[np.ndarray, np.ndarray]:
    """List-scheduling greedy. Returns ``(t, alpha)`` matching scheduler.schedule().

    Each iteration picks an op whose predecessors are scheduled, then
    picks the machine combination that gives the earliest completion
    time. Ties broken by the order operations appear in
    ``workload.operations``.
    """
    num_operations = len(workload.operations)
    machines = workload.machines
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()

    t = np.zeros(num_operations)
    alpha = np.zeros((num_operations, num_combinations))

    # When does each combination become free? For overlapping combinations
    # we propagate the busy-until time when one of them schedules an op.
    combination_available_time = np.zeros(num_combinations)
    scheduled = [False] * num_operations

    while not all(scheduled):
        best_op_idx = None
        best_completion_time = float("inf")
        best_combination_idx = None
        best_start_time = 0.0

        for i in range(num_operations):
            if scheduled[i]:
                continue
            op = workload.operations[i]

            # Skip if any predecessor isn't scheduled yet.
            can_schedule = True
            for pred in op.predecessors:
                pred_idx = workload.operations.index(pred)
                if not scheduled[pred_idx]:
                    can_schedule = False
                    break
            if not can_schedule:
                continue

            # Try every combination, keep the earliest completion.
            for combo_idx in range(num_combinations):
                earliest_start = combination_available_time[combo_idx]

                # Wait for any overlapping combinations that hold a
                # currently-running op.
                for j in range(num_operations):
                    if not scheduled[j]:
                        continue
                    other_combo_idx = int(np.argmax(alpha[j, :]))
                    if workload.combinations_overlap(combo_idx, other_combo_idx):
                        other_dur = workload.operations[j].get_duration_for_combination(
                            other_combo_idx, machine_combinations, machines
                        )
                        other_end = t[j] + other_dur
                        earliest_start = max(earliest_start, other_end)

                # Wait for predecessors + their transfer cost into this
                # combination's first machine.
                for pred in op.predecessors:
                    pred_idx = workload.operations.index(pred)
                    pred_combo_idx = int(np.argmax(alpha[pred_idx, :]))
                    pred_dur = workload.operations[pred_idx].get_duration_for_combination(
                        pred_combo_idx, machine_combinations, machines
                    )
                    pred_end = t[pred_idx] + pred_dur
                    pred_combo = machine_combinations[pred_combo_idx]
                    cand_combo = machine_combinations[combo_idx]
                    pred_machine_idx = machines.index(pred_combo[0])
                    cand_machine_idx = machines.index(cand_combo[0])
                    transfer = transfer_times[pred_machine_idx, cand_machine_idx]
                    earliest_start = max(earliest_start, pred_end + transfer)

                # Honor periodic / windowed time-bounds carried by the
                # Operation (set by create_workload_from_network_hierarchy
                # when expanding periodic networks: instance i gets
                # min_start_t = start + i*period, max_end_t = min_start +
                # window_duration). Without this, all periodic instances
                # collapse to t=0 since they have no predecessors and the
                # only nonzero floor was the prior op on the machine.
                if op.min_start_t is not None:
                    earliest_start = max(earliest_start, float(op.min_start_t))

                duration = workload.operations[i].get_duration_for_combination(
                    combo_idx, machine_combinations, machines
                )
                completion_time = earliest_start + duration

                # If the candidate combo would miss the periodic window,
                # only keep it as a "least bad" fallback — let validation
                # flag it. A non-backtracking greedy can't do better.
                if op.max_end_t is not None and completion_time > float(op.max_end_t):
                    if completion_time < best_completion_time:
                        best_completion_time = completion_time
                        best_op_idx = i
                        best_combination_idx = combo_idx
                        best_start_time = earliest_start
                    continue

                if completion_time < best_completion_time:
                    best_completion_time = completion_time
                    best_op_idx = i
                    best_combination_idx = combo_idx
                    best_start_time = earliest_start

        if best_op_idx is None:
            # Cycle in the dep DAG (shouldn't happen if validation runs).
            # Force-place the first unscheduled op onto combo 0 so we make
            # progress instead of looping forever.
            for i in range(num_operations):
                if not scheduled[i]:
                    best_op_idx = i
                    best_combination_idx = 0
                    best_start_time = combination_available_time[0]
                    break

        # Commit the chosen (op, combo).
        t[best_op_idx] = best_start_time
        alpha[best_op_idx, best_combination_idx] = 1.0
        scheduled[best_op_idx] = True

        duration = workload.operations[best_op_idx].get_duration_for_combination(
            best_combination_idx, machine_combinations, machines
        )
        op_end = best_start_time + duration

        combination_available_time[best_combination_idx] = op_end
        # Propagate busy-until to overlapping combos.
        for combo_idx in range(num_combinations):
            if combo_idx == best_combination_idx:
                continue
            if workload.combinations_overlap(best_combination_idx, combo_idx):
                combination_available_time[combo_idx] = max(
                    combination_available_time[combo_idx], op_end
                )

    return t, alpha
