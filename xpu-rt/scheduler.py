#
# Problem formulation from https://www.sciencedirect.com/science/article/pii/S037722172300382X#sec0014 Section 2.1.
#

import cvxpy as cp
import numpy as np
import time
import os
import sys

# Ensure local modules are imported correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload, Window
from packing import greedy_packing, convex_packing, combine_solved_windows
from typing import Tuple, Optional

# Import from local fusion module (not the system package)
try:
    from .fusion import fuse_operations, expand_schedule, print_fusion_report
except ImportError:
    from fusion import fuse_operations, expand_schedule, print_fusion_report


def _constraints_section_logger(enabled: bool, constraints: list):
    """
    Lightweight logger for timing constraint-generation sections.

    Usage:
        end = log("name")   # starts section
        ... add constraints ...
        end()               # prints timing (+count, seconds) if enabled
    """
    def _start(name: str):
        if not enabled:
            return lambda: None
        start_t = time.perf_counter()
        start_n = len(constraints)
        print(f"[constraints] start: {name} (n={start_n})")

        def _end():
            end_t = time.perf_counter()
            end_n = len(constraints)
            print(
                f"[constraints] done : {name} (+{end_n - start_n}, n={end_n}, {end_t - start_t:.3f}s)"
            )

        return _end

    return _start


def _compute_dependency_descendants_bitset(operations: list) -> Optional[list[int]]:
    """
    Compute transitive reachability (descendants) for the precedence graph induced by
    op.get_predecessors() edges, restricted to the given `operations` list.

    Returns:
        descendants: list[int] where bit j of descendants[i] is 1 iff i ->* j
        or None if the graph appears cyclic (no valid topological order).
    """
    n = len(operations)
    if n == 0:
        return []

    # Fast op->index lookup. Prefer hashing the object; fall back to id().
    op_to_idx = None
    try:
        op_to_idx = {op: i for i, op in enumerate(operations)}
    except TypeError:
        op_to_idx = None
    opid_to_idx = {id(op): i for i, op in enumerate(operations)}

    def _idx(op):
        if op_to_idx is not None:
            try:
                return op_to_idx[op]
            except KeyError:
                pass
        return opid_to_idx.get(id(op), None)

    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n

    for i, op in enumerate(operations):
        for pred in op.get_predecessors():
            p = _idx(pred)
            if p is None:
                continue
            succ[p].append(i)
            indeg[i] += 1

    # Kahn topological sort
    from collections import deque

    q = deque([u for u in range(n) if indeg[u] == 0])
    topo: list[int] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        # Cycle (or inconsistent predecessor lists); skip pruning in this case.
        return None

    descendants = [0] * n
    for u in reversed(topo):
        bits = 0
        for v in succ[u]:
            bits |= descendants[v] | (1 << v)
        descendants[u] = bits
    return descendants


def _auto_big_m(operations, machine_combinations, machines, transfer_times,
                num_combinations) -> float:
    """Compute a big-M that strictly upper-bounds any feasible makespan.

    The MILP non-overlap constraints use big-M relaxation: when the alpha
    indicators are off the constraint should be slack. If H is too small
    relative to op durations + transfers, the relaxation becomes binding
    and the problem reports infeasible even though valid schedules exist.

    H = 2 * (sum(max op duration across combinations) + N * max transfer)
    """
    max_durs = []
    for op in operations:
        durs = [op.get_duration_for_combination(k, machine_combinations,
                                                machines)
                for k in range(num_combinations)]
        max_durs.append(max(durs) if durs else 0.0)
    max_transfer = 0.0
    for row in transfer_times:
        for v in row:
            if v > max_transfer:
                max_transfer = v
    H = float(2 * (sum(max_durs) + len(operations) * max_transfer + 1.0))
    return max(H, 5000.0)


def schedule_window(window: Window, debug_constraints: bool = False,
                    target_diversity_weight: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    num_operations = len(window.operations)
    machine_combinations = window.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = window.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters: auto-size big-M so the non-overlap relaxation is
    # non-binding when alpha-pair is off. Hard-coded 5000 was undersized
    # for >5ms-makespan workloads in microseconds (would yield infeasible
    # on dronet's 7ms and the 130ms multi-model run).
    H = _auto_big_m(window.operations, machine_combinations, window.machines,
                    transfer_times, num_combinations)

    # Constraints
    constraints = []
    log = _constraints_section_logger(enabled=debug_constraints, constraints=constraints)
    build_all_start = time.perf_counter() if debug_constraints else None

    # (2) Each operation must be assigned to exactly one machine combination
    end = log("(2) assignment (alpha row-sum == 1)")
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    end()

    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    end = log("(3) precedence")
    for i in range(num_operations):
        predecessors = window.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = None
            try:
                i_pred = window.operations.index(pred)
            except ValueError:
                # happens when the predecessor is not in the window
                i_pred = None

            # check if there is a required predecessor in this window
            if i_pred is not None:
                # Build duration vector for predecessor
                dur_vec_pred = [window.operations[i_pred].get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
                
                # For transfer time, use maximum as upper bound (DCP-compliant)
                max_transfer_time = 0
                for k_pred in range(num_combinations):
                    for k_curr in range(num_combinations):
                        machine_pred = window.machines.index(machine_combinations[k_pred][0])
                        machine_curr = window.machines.index(machine_combinations[k_curr][0])
                        transfer_time_val = transfer_times[machine_pred][machine_curr]
                        max_transfer_time = max(max_transfer_time, transfer_time_val)
                
                transfer_time_weighted = max_transfer_time
                
                constraints.append(
                    t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
                )
    end()

    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    end = log("(4)(5) non-overlap (pairwise, overlapping combinations)")
    dep_desc = _compute_dependency_descendants_bitset(window.operations)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            # Optimization: If i and j are already ordered by a dependency chain (i ->* j or j ->* i),
            # precedence constraints already enforce a non-overlap in time, so skip overlap constraints.
            if dep_desc is not None:
                if ((dep_desc[i] >> j) & 1) or ((dep_desc[j] >> i) & 1):
                    continue
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if window.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        # Get duration for combination k2
                        dur_j_k2 = window.operations[j].get_duration_for_combination(k2, machine_combinations, window.machines)
                        constraints.append(
                            t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        # Get duration for combination k1
                        dur_i_k1 = window.operations[i].get_duration_for_combination(k1, machine_combinations, window.machines)
                        constraints.append(
                            t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    end()

    # (6) Makespan lower bound. When operations[i].processing_times_by_pred
    # is set we use the predecessor-aware tensor: linearise the bilinear
    # alpha[pred,k_pred] * alpha[i,k_curr] into gamma and contribute
    # cost[k_pred,k_curr] * gamma. Falls back to the 2D dur_vec path when
    # the per-pred map is empty so the existing workloads are unaffected.
    end = log("(6) makespan lower bound (C_max)")
    pred_aware_gammas = {}  # i -> cp.Variable (num_combinations, num_combinations) binary
    for i in range(num_operations):
        op = window.operations[i]
        pmap = getattr(op, "processing_times_by_pred", None) or {}
        preds = op.get_predecessors()
        # Pick the dominant predecessor (longest critical path proxy: first in
        # list; future work could pick by index of largest cost). With a single
        # predecessor this is exact; with multiple, tightening would require
        # max-style aggregation which is non-DCP — punt to dominant.
        dom_pred = preds[0] if preds and pmap else None
        if dom_pred is not None:
            try:
                i_pred = window.operations.index(dom_pred)
            except ValueError:
                i_pred = None
        else:
            i_pred = None
        if i_pred is not None:
            gamma = cp.Variable((num_combinations, num_combinations), boolean=True)
            pred_aware_gammas[i] = (i_pred, gamma)
            # Linearisation of gamma[k_pred, k_curr] = alpha[i_pred,k_pred] * alpha[i,k_curr]
            constraints.append(cp.sum(gamma) == 1)
            for k_pred in range(num_combinations):
                constraints.append(cp.sum(gamma[k_pred, :]) <= alpha[i_pred, k_pred])
            for k_curr in range(num_combinations):
                constraints.append(cp.sum(gamma[:, k_curr]) <= alpha[i, k_curr])
            # Effective duration is sum cost[k_pred,k_curr] * gamma[k_pred,k_curr];
            # default to the 2D dur for missing entries (e.g. cold-start).
            base_dur_vec = [op.get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
            cost_matrix = np.zeros((num_combinations, num_combinations))
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    if (k_pred, k_curr) in pmap:
                        cost_matrix[k_pred, k_curr] = pmap[(k_pred, k_curr)]
                    else:
                        cost_matrix[k_pred, k_curr] = base_dur_vec[k_curr]
            constraints.append(
                C_max >= t[i] + cp.sum(cp.multiply(cost_matrix, gamma))
            )
        else:
            dur_vec = [op.get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
            constraints.append(
                C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
            )
    end()

    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    end = log("t >= 0")
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )
    end()

    # term to maximize consecutive empty space on each machine combination
    end = log("empty_space (aux objective constraints)")
    empty_space = cp.Variable(num_combinations)
    for k in range(num_combinations):
        for i in range(num_operations):
            for j in range(i+1, num_operations):
                # Only consider if both operations could be on this combination (though they can't overlap)
                dur_j_k = window.operations[j].get_duration_for_combination(k, machine_combinations, window.machines)
                constraints.append(
                    empty_space[k] >= t[i] - (t[j] + dur_j_k - (2 - alpha[i, k] - alpha[j, k] + beta[i, j]) * H)
                )
    end()

    if debug_constraints:
        print(f"[constraints] total build time: {time.perf_counter() - build_all_start:.3f}s (n={len(constraints)})")


    # objective_func = 150*C_max + cp.sum(empty_space)
    objective_func = C_max

    # Optional: --target-diversity-weight encourages using more distinct
    # primary machines. For each machine M we introduce a binary `used[M]`
    # forced to 1 if any operation is assigned to a combination whose primary
    # machine is M. We then SUBTRACT λ × Σ used[M] from the objective so the
    # solver prefers schedules that touch more machines, all else equal.
    if target_diversity_weight and target_diversity_weight > 0.0:
        primary_machines = list(window.machines)
        used = cp.Variable(len(primary_machines), boolean=True)
        for m_idx, m_name in enumerate(primary_machines):
            matching_combos = [
                k for k, c in enumerate(machine_combinations)
                if c[0] == m_name
            ]
            if not matching_combos:
                constraints.append(used[m_idx] == 0)
                continue
            for i in range(num_operations):
                for k in matching_combos:
                    constraints.append(used[m_idx] >= alpha[i, k])
        objective_func = objective_func - target_diversity_weight * cp.sum(used)

    # Optimization problem
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value

def schedule(
    workload: Workload,
    fusion_threshold: Optional[float] = None,
    verbose: bool = False,
    solver_verbosity: int = 0,
    time_limit: Optional[float] = None,
    restrict_makespan_to_nonperiodic: bool = True,
    prune_cross_period_constraints: bool = True,
    debug_constraints: bool = False,
    prune_overlap_constraints_for_dependency_chain: bool = True,
    target_diversity_weight: float = 0.0,
    cvxpy_solver: str = "MOSEK",
    emit_report_to: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Workload], Optional[dict]]:
    """
    Schedule a workload, optionally with operation fusion.
    
    Args:
        workload: The workload to schedule
        fusion_threshold: If provided, fuse operations with duration <= threshold (in time units).
                        If None, no fusion is performed.
        verbose: If True, print problem statistics and timing information.
        solver_verbosity: MOSEK solver verbosity level (0=silent, >0=enables verbose output).
        time_limit: Maximum optimization time in seconds. If None, no time limit is set.
                   MOSEK will return the best solution found within the time limit.
        restrict_makespan_to_nonperiodic: If True, C_max only tracks non-periodic operations
                   (those without min_start_t / max_end_t). Periodic/background operations
                   still obey all constraints but do not affect the makespan objective.
        prune_cross_period_constraints: If True, skip precedence and non-overlap constraints
                   between operations whose time windows provably do not overlap in time.
                   This reduces redundant constraints for periodic tasks in disjoint periods.
        debug_constraints: If True (or verbose=True), print timing and counts for each major
                   constraint family during model construction.
        prune_overlap_constraints_for_dependency_chain: If True, skip generating (4)(5) non-overlap
                   constraints for operation pairs that are already ordered by the precedence graph
                   via a transitive dependency chain (i ->* j or j ->* i).
    
    Returns:
        (t, alpha, fused_workload, fusion_map) where:
        - t: Start times for operations (original operations if fusion was used)
        - alpha: Machine assignments for operations (original operations if fusion was used)
        - fused_workload: The fused workload (None if no fusion)
        - fusion_map: Mapping from fused op index to original op indices (None if no fusion)
    """
    original_workload = workload
    fusion_map = None
    
    # Apply fusion only if threshold is explicitly provided and positive
    # If fusion_threshold is None (flag not passed) or <= 0, skip fusion
    if fusion_threshold is not None and fusion_threshold > 0:
        workload, fusion_map = fuse_operations(workload, fusion_threshold)
        print(f"Fusion applied: {len(original_workload.operations)} operations -> {len(workload.operations)} fused operations")
        # Print detailed fusion report for debugging (to file)
        os.makedirs("fusion_reports", exist_ok=True)
        report_file = f"fusion_reports/fusion_report_{len(original_workload.operations)}to{len(workload.operations)}.txt"
        print_fusion_report(original_workload, workload, fusion_map, output_file=report_file)
    else:
        # No fusion requested - use original workload as-is
        if verbose:
            print("Fusion skipped: fusion_threshold not provided or <= 0")
    
    num_operations = len(workload.get_operations())
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters: auto-size big-M (see top of schedule_window for
    # rationale).
    H = _auto_big_m(workload.get_operations(), machine_combinations,
                    workload.machines, transfer_times, num_combinations)

    # Constraints
    constraints = []
    constraints_debug_enabled = debug_constraints or verbose
    log = _constraints_section_logger(enabled=constraints_debug_enabled, constraints=constraints)
    build_all_start = time.perf_counter()

    # (2) Each operation must be assigned to exactly one machine combination
    end = log("(2) assignment (alpha row-sum == 1)")
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    end()

    # (2b) Hard infeasibility exclusion. When an op carries
    # `infeasible_combinations`, force alpha[i, k] = 0 for those k. This
    # is the no-extrapolation guard for the heterogeneous QNN workflow:
    # if we measured that (dispatch i, backend k) cannot build/run on
    # the board, the MILP is forbidden from assigning it there — no
    # large coefficient as a soft penalty, no silent fallback. If every
    # combination of some op is infeasible the model is infeasible and
    # the solver returns no solution, which is the right loud failure
    # ("profile that op on at least one backend before scheduling").
    n_infeasible = 0
    end = log("(2b) infeasibility hard exclusion")
    ops = workload.get_operations()
    for i in range(num_operations):
        infe = getattr(ops[i], "infeasible_combinations", set()) or set()
        for k in infe:
            if 0 <= k < num_combinations:
                constraints.append(alpha[i, k] == 0)
                n_infeasible += 1
    end()
    if constraints_debug_enabled and n_infeasible:
        print(f"  (2b) added {n_infeasible} hard exclusions across "
              f"{sum(1 for o in ops if getattr(o, 'infeasible_combinations', None))}"
              f" ops")

    # (2c) Phase F2b — singleton-feasible pre-fix (MOSEK convergence aid).
    # When an op has exactly one feasible combination (either because
    # `infeasible_combinations` rules out all others or because all
    # other combos report >= 1e8 in processing_times, indicating "no
    # feasible measurement"), force alpha[i, single_k] = 1 directly.
    # Combined with the row-sum-equals-1 constraint, this lets MOSEK
    # presolve eliminate the binary variable entirely. Phase F1's
    # diagnosis pointed at canonicalization wall — singleton pre-fix
    # reduces that wall in proportion to how many ops are pinned.
    INFEASIBLE_COST = 1e8  # processing_times sentinel for "no profile data"
    n_singletons = 0
    end = log("(2c) F2b singleton-feasible pre-fix")
    for i in range(num_operations):
        infe = set(getattr(ops[i], "infeasible_combinations", set()) or set())
        proc_times = list(getattr(ops[i], "processing_times", []) or [])
        feasible_ks = [k for k in range(num_combinations)
                       if k not in infe and
                       (k >= len(proc_times) or proc_times[k] < INFEASIBLE_COST)]
        if len(feasible_ks) == 1:
            constraints.append(alpha[i, feasible_ks[0]] == 1)
            n_singletons += 1
    end()
    if constraints_debug_enabled and n_singletons:
        print(f"  (2c) F2b: pre-fixed {n_singletons} singleton-feasible ops "
              f"({100.0*n_singletons/num_operations:.1f}% of {num_operations} total)")

    # (2d) Phase F2c — symmetry-breaking constraints for periodic instances.
    # When the workload contains multiple periodic INSTANCES of the same
    # base network (e.g. mlp_control0, mlp_control1, mlp_control2,
    # mlp_control3), their dispatch sets are structurally identical. MOSEK
    # without symmetry-breaking explores N! permutations of which instance
    # goes where. We add ordering constraints that force instance k's
    # first op to start no later than instance (k+1)'s first op. Since
    # the periodic min_start_t already enforces this in the data, the
    # explicit constraint just helps MOSEK's branch-and-bound prune
    # symmetric subtrees up front.
    n_symmetry = 0
    end = log("(2d) F2c symmetry-breaking for periodic instances")
    # Group ops by (job_name_base, dispatch_id) to find instances.
    # job_name = e.g. "mlp_control0", "mlp_control1"; the base is the
    # leading non-digit prefix.
    import re as _re_sym
    instances_by_base: dict[str, dict[int, list[int]]] = {}
    for i in range(num_operations):
        op = ops[i]
        job_name = getattr(op, "operation_name", "") or ""
        # dispatch_name format: "<job_name>_dispatch_<id>"; the parent
        # network's instance index is parseable from job_name.
        m = _re_sym.match(r"^(.+?)(\d+)_dispatch_", job_name)
        if not m:
            continue
        base = m.group(1).rstrip("_")
        try:
            inst_k = int(m.group(2))
        except ValueError:
            continue
        # Parse dispatch index for matching identical ops across instances.
        m2 = _re_sym.search(r"_dispatch_(\d+)", job_name)
        if not m2:
            continue
        try:
            disp_id = int(m2.group(1))
        except ValueError:
            continue
        instances_by_base.setdefault(base, {}).setdefault(disp_id, []).append(
            (inst_k, i)
        )
    # For each (base, disp_id), if we have ≥ 2 instances, add a chain
    # t[inst_0] ≤ t[inst_1] ≤ ... — they're already imposed by
    # min_start_t for periodic data but the explicit chain breaks
    # symmetric exploration in the optimizer.
    for base, by_disp in instances_by_base.items():
        for disp_id, instance_list in by_disp.items():
            if len(instance_list) < 2:
                continue
            instance_list.sort()  # by inst_k ascending
            for a, b in zip(instance_list, instance_list[1:]):
                _, ia = a
                _, ib = b
                # Allow ties — same-instant scheduling of symmetric ops
                # is harmless and may even be the periodic solution.
                constraints.append(t[ia] <= t[ib])
                n_symmetry += 1
    end()
    if constraints_debug_enabled and n_symmetry:
        print(f"  (2d) F2c: added {n_symmetry} symmetry-breaking constraints "
              f"across {len(instances_by_base)} periodic networks")

    def _periods_overlap(op_a, op_b) -> bool:
        """Return True if the time windows of two operations can overlap."""
        a_start = getattr(op_a, "min_start_t", None)
        a_end = getattr(op_a, "max_end_t", None)
        b_start = getattr(op_b, "min_start_t", None)
        b_end = getattr(op_b, "max_end_t", None)
        # If any bound is missing, conservatively assume they may overlap
        if a_start is None or a_end is None or b_start is None or b_end is None:
            return True
        # Intervals [a_start, a_end) and [b_start, b_end) overlap iff both:
        # a_start < b_end and b_start < a_end
        return (a_start < b_end) and (b_start < a_end)

    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    end = log("(3) precedence")
    for i in range(num_operations):
        op_i = workload.operations[i]
        predecessors = op_i.get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Optionally prune precedence constraints between non-overlapping periods
            if prune_cross_period_constraints:
                pred_start = getattr(pred, "min_start_t", None)
                pred_end = getattr(pred, "max_end_t", None)
                succ_start = getattr(op_i, "min_start_t", None)
                succ_end = getattr(op_i, "max_end_t", None)
                if pred_start is not None and pred_end is not None and succ_start is not None and succ_end is not None:
                    # If predecessor's window ends before successor's window starts,
                    # precedence is automatically satisfied by time-window constraints.
                    if pred_end <= succ_start:
                        continue
                    # If successor's window ends before predecessor's window starts,
                    # the precedence is impossible under the windows; raise early.
                    if succ_end <= pred_start:
                        raise ValueError(
                            f"Infeasible precedence: successor window [{succ_start}, {succ_end}) "
                            f"before predecessor window [{pred_start}, {pred_end})."
                        )

            # Build duration vector for predecessor
            dur_vec_pred = [
                workload.operations[i_pred].get_duration_for_combination(k, machine_combinations, workload.machines)
                for k in range(num_combinations)
            ]
            
            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            max_transfer_time = 0
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    machine_pred = workload.machines.index(machine_combinations[k_pred][0])
                    machine_curr = workload.machines.index(machine_combinations[k_curr][0])
                    transfer_time_val = transfer_times[machine_pred][machine_curr]
                    max_transfer_time = max(max_transfer_time, transfer_time_val)
            
            transfer_time_weighted = max_transfer_time
            
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
            )
    end()

    # Time window constraints: operations must respect min_start_t and max_end_t if specified
    end = log("time windows (min_start_t/max_end_t)")
    for i in range(num_operations):
        op = workload.operations[i]
        # Constraint: operation must start after min_start_t (if specified)
        if op.min_start_t is not None:
            constraints.append(
                t[i] >= op.min_start_t
            )
        # Constraint: operation must end before max_end_t (if specified)
        if op.max_end_t is not None:
            # Build duration vector for all combinations
            dur_vec = [op.get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
            # Operation completion time = start_time + duration_for_chosen_combination
            # Must be <= max_end_t
            constraints.append(
                t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])) <= op.max_end_t
            )
            # constraints.append(
            #     t[i] <= op.max_end_t
            # )
    end()

    # (7)+(8) Robotics-deadline support: per-op `deadline_us` constraint
    # with optional binary skip indicator `s[i]` for ops that opted into
    # `skip_allowed`. Mirrors the relaxation pattern used for the
    # non-overlap big-M (lines ~496) so the constraint becomes slack only
    # when s[i]=1.
    skip_vars: dict[int, "cp.Variable"] = {}
    end = log("(7)+(8) robotics deadlines + skip indicator")
    for i in range(num_operations):
        op = workload.operations[i]
        deadline = getattr(op, "deadline_us", None)
        if deadline is None:
            continue
        skip_allowed = getattr(op, "skip_allowed", False)
        dur_vec = [
            op.get_duration_for_combination(k, machine_combinations, workload.machines)
            for k in range(num_combinations)
        ]
        if skip_allowed:
            s_i = cp.Variable(boolean=True)
            skip_vars[i] = s_i
            # When s_i=0, full deadline applies. When s_i=1, the deadline
            # is relaxed by H, effectively disabling the bound.
            constraints.append(
                t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
                <= deadline + H * s_i
            )
        else:
            constraints.append(
                t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
                <= deadline
            )
    end()

    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    end = log("(4)(5) non-overlap (pairwise, overlapping combinations)")
    dep_desc = None
    if prune_overlap_constraints_for_dependency_chain:
        dep_desc = _compute_dependency_descendants_bitset(workload.operations)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            op_i = workload.operations[i]
            op_j = workload.operations[j]

            # Optionally skip pairs whose time windows cannot overlap
            if prune_cross_period_constraints and not _periods_overlap(op_i, op_j):
                continue

            # Optimization: If i and j are already ordered by a dependency chain (i ->* j or j ->* i),
            # precedence constraints already enforce a non-overlap in time, so skip overlap constraints.
            if dep_desc is not None:
                if ((dep_desc[i] >> j) & 1) or ((dep_desc[j] >> i) & 1):
                    continue

            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(
                            k2, machine_combinations, workload.machines
                        )
                        constraints.append(
                            t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(
                            k1, machine_combinations, workload.machines
                        )
                        constraints.append(
                            t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    end()

    # (6) Makespan constraints:
    if restrict_makespan_to_nonperiodic:
        # C_max tracks only NON-periodic operations (operations without explicit time-window bounds).
        # Periodic/background operations (with min_start_t or max_end_t set) do NOT constrain C_max.
        end = log("(6) makespan lower bound (C_max) - non-periodic only")
        non_periodic_ops_exist = False
        for i in range(num_operations):
            op = workload.operations[i]
            # Treat an operation as periodic/background if it has any time-window bound
            is_periodic = getattr(op, "min_start_t", None) is not None or getattr(op, "max_end_t", None) is not None
            if is_periodic:
                continue
            non_periodic_ops_exist = True
            # Build duration vector for all combinations
            dur_vec = [
                workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines)
                for k in range(num_combinations)
            ]
            constraints.append(
                C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
            )
        # If there are no non-periodic operations, C_max is unconstrained from below
        # (objective will be trivial), which is acceptable: only periodic tasks exist.
        end()
    else:
        # Original behavior: C_max covers all operations (including periodic ones)
        end = log("(6) makespan lower bound (C_max) - all operations")
        for i in range(num_operations):
            dur_vec = [
                workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines)
                for k in range(num_combinations)
            ]
            constraints.append(
                C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
            )
        end()
    
    # Debug: Print durations for first operation to verify they're correct
    if (verbose or debug_constraints) and num_operations > 0:
        print(f"\nDEBUG DURATIONS for first operation:")
        for k in range(num_combinations):
            combo = machine_combinations[k]
            combo_str = "+".join(combo) if len(combo) > 1 else combo[0]
            dur = workload.operations[0].get_duration_for_combination(k, machine_combinations, workload.machines)
            print(f"  Combination {k} ({combo_str}): {dur:.3f} ms")
    
    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    end = log("t >= 0")
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )
    end()

    if constraints_debug_enabled:
        print(
            f"[constraints] total build time: {time.perf_counter() - build_all_start:.3f}s "
            f"(n={len(constraints)})"
        )

    # Optimization problem. C_max is the primary objective; skip indicators
    # contribute a heavy penalty so the solver only sets s_i=1 when the
    # deadline forces it. The penalty must dominate any conceivable C_max
    # win from skipping, so we weight by H (the auto-sized big-M, which is
    # already an upper bound on C_max).
    objective_func = C_max
    if skip_vars:
        skip_penalty = float(H)
        objective_func = objective_func + skip_penalty * cp.sum(
            [s for s in skip_vars.values()]
        )

    # Optional target-diversity penalty: subtract λ × (number of distinct
    # primary machines used) so the solver prefers placements that touch
    # more devices, all else equal.
    if target_diversity_weight and target_diversity_weight > 0.0:
        primary_machines = list(workload.machines)
        used = cp.Variable(len(primary_machines), boolean=True)
        for m_idx, m_name in enumerate(primary_machines):
            matching_combos = [
                k for k, c in enumerate(machine_combinations)
                if c[0] == m_name
            ]
            if not matching_combos:
                constraints.append(used[m_idx] == 0)
                continue
            for i in range(num_operations):
                for k in matching_combos:
                    constraints.append(used[m_idx] >= alpha[i, k])
        objective_func = objective_func - target_diversity_weight * cp.sum(used)

    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    
    # Print problem statistics
    if verbose:
        print(f"\n{'='*60}")
        print("OPTIMIZATION PROBLEM STATISTICS")
        print(f"{'='*60}")
        print(f"Number of operations: {num_operations}")
        print(f"Number of machine combinations: {num_combinations}")
        num_vars = num_operations * num_combinations + num_operations * num_operations + num_operations + 1
        print(f"Number of variables: {num_vars}")
        print(f"  - alpha (operation->combination): {num_operations * num_combinations}")
        print(f"  - beta (operation ordering): {num_operations * num_operations}")
        print(f"  - t (start times): {num_operations}")
        print(f"  - C_max (makespan): 1")
        print(f"Number of constraints: {len(constraints)}")
        print(f"{'='*60}\n")
        print("Starting optimization...")
    # Always time the solve regardless of verbosity — SchedulerReport needs it.
    start_time = time.time()
    
    # Configure solver backend. Default is MOSEK (original behaviour); any CVXPY
    # MILP-capable backend can be selected via cvxpy_solver.
    solver_verbose = verbose or (solver_verbosity > 0)
    solver_attr = getattr(cp, cvxpy_solver, None)
    if solver_attr is None:
        raise ValueError(
            f"Unknown CVXPY solver '{cvxpy_solver}'. "
            f"Available: {cp.installed_solvers()}"
        )
    solver_kwargs: dict = {}

    if cvxpy_solver == "MOSEK":
        mosek_params: dict = {}
        if time_limit is not None and time_limit > 0:
            mosek_params['MSK_DPAR_OPTIMIZER_MAX_TIME'] = time_limit
            if verbose:
                print(f"Time limit set to {time_limit:.1f} seconds ({time_limit/60:.1f} minutes)")
        # F2d/F2e: env-injected MOSEK params for convergence-aid sweeps.
        # XPURT_MOSEK_MIO_GAP=0.05 sets MSK_DPAR_MIO_TOL_REL_GAP=0.05.
        # XPURT_MOSEK_PARAMS=key=val;key=val sets arbitrary MOSEK keys.
        _mio_gap = os.environ.get("XPURT_MOSEK_MIO_GAP", "")
        if _mio_gap:
            try:
                mosek_params['MSK_DPAR_MIO_TOL_REL_GAP'] = float(_mio_gap)
            except ValueError:
                pass
        _extra = os.environ.get("XPURT_MOSEK_PARAMS", "")
        if _extra:
            for kv in _extra.split(";"):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k = k.strip(); v = v.strip()
                # Try float first, then string.
                try:
                    mosek_params[k] = float(v)
                except ValueError:
                    mosek_params[k] = v
        if mosek_params:
            solver_kwargs["mosek_params"] = mosek_params
    elif cvxpy_solver == "GUROBI":
        if time_limit is not None and time_limit > 0:
            solver_kwargs["TimeLimit"] = time_limit
    elif cvxpy_solver == "HIGHS":
        if time_limit is not None and time_limit > 0:
            solver_kwargs["time_limit"] = time_limit
    elif cvxpy_solver == "SCIP":
        if time_limit is not None and time_limit > 0:
            solver_kwargs["scip_params"] = {"limits/time": time_limit}
    elif cvxpy_solver == "CBC":
        if time_limit is not None and time_limit > 0:
            solver_kwargs["maximumSeconds"] = time_limit

    problem.solve(solver=solver_attr, verbose=solver_verbose, **solver_kwargs)
    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"{'='*60}")

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)

    t_result = t.value
    alpha_result = alpha.value
    # Stash skip-indicator results onto the original workload so callers
    # know which ops were dropped. Use indices into `original_workload`
    # (= workload pre-fusion) so the schema remains stable for callers
    # that don't enable fusion.
    skipped_indices: list[int] = []
    if skip_vars:
        for i, sv in skip_vars.items():
            if sv.value is not None and sv.value > 0.5:
                skipped_indices.append(i)
        original_workload.skipped_op_indices = skipped_indices
    else:
        original_workload.skipped_op_indices = []

    # Stash solver state for downstream feedback derivation (xpu-rt/feedback.py).
    # Sidecar on the workload keeps the schedule() return signature stable.
    original_workload.solver_state = {
        "problem_status": str(problem.status),
        "makespan": float(C_max.value) if C_max.value is not None else None,
        "objective_value": float(problem.value) if problem.value is not None else None,
        "num_operations": num_operations,
        "num_combinations": num_combinations,
        "fusion_applied": fusion_threshold is not None and fusion_threshold > 0,
        "solve_wall_s": float(elapsed_time),
        "solver_name": cvxpy_solver,
    }

    # Build SchedulerReport and stash + optionally write to disk. Non-breaking:
    # the schedule() return tuple stays 4-element; callers access the report
    # via workload.solver_state["report"] or read the JSON file.
    if t_result is not None and alpha_result is not None:
        try:
            from profiling import SchedulerReport
            report = SchedulerReport.from_solver_state(
                original_workload, t_result, alpha_result,
                solver_name=cvxpy_solver,
                solve_wall_s=float(elapsed_time),
                solver_status=str(problem.status),
                fusion_map=fusion_map,
            )
            original_workload.solver_state["report"] = report
            if emit_report_to:
                report.write_json(emit_report_to)
        except Exception as exc:
            print(f"warning: SchedulerReport build failed: {exc}")
    
    # Check if optimization was successful
    if t_result is None or alpha_result is None:
        print("Warning: Optimization failed (infeasible or error). Cannot expand schedule.")
        return None, None, workload if fusion_threshold else None, fusion_map
    
    # Check problem status - warn if not optimal
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: Problem status is '{problem.status}' (not optimal).")
        if problem.status == cp.SOLVER_ERROR:
            print("  Solver encountered an error. Solution may be invalid.")
        elif problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            print("  Problem is infeasible. Solution is invalid.")
            return None, None, workload if fusion_threshold else None, fusion_map
        elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
            print("  Problem is unbounded. Solution may be invalid.")
        else:
            print("  Solution may not be optimal but should be feasible.")
    
    # Validate solution dimensions
    if len(t_result) != num_operations or alpha_result.shape[0] != num_operations:
        print(f"Error: Solution dimensions don't match. Expected {num_operations} operations, got {len(t_result)} start times and {alpha_result.shape[0]} assignments.")
        return None, None, workload if fusion_threshold else None, fusion_map
    
    # Expand schedule back to original operations if fusion was used
    if fusion_threshold is not None and fusion_threshold > 0 and fusion_map is not None:
        t_result, alpha_result = expand_schedule(workload, fusion_map, original_workload, t_result, alpha_result)
        print(f"Schedule expanded: {len(workload.operations)} fused operations -> {len(original_workload.operations)} original operations")
        
        # Validate expansion
        if len(t_result) != len(original_workload.operations):
            print(f"ERROR: Expanded schedule has {len(t_result)} operations but original workload has {len(original_workload.operations)} operations!")
        else:
            print(f"Validation: All {len(original_workload.operations)} original operations have been scheduled.")
        
        return t_result, alpha_result, workload, fusion_map
    
    # Validate non-fused schedule
    if len(t_result) != len(workload.operations):
        print(f"ERROR: Schedule has {len(t_result)} operations but workload has {len(workload.operations)} operations!")
    else:
        print(f"Validation: All {len(workload.operations)} operations have been scheduled.")
    
    return t_result, alpha_result, None, None

def schedule_additional_objectives(
    workload: Workload,
    nominal_start_times: list[float],
    gap_bound: float,
    debug_constraints: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    @param nominal_start_times: list of nominal start times for each operation. If there is no desired start time, set index to -1
    @param gap_bound: the maximum maximum allowable gap between operations to bound the optimization problem
    """
    num_operations = len(workload.get_operations())
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # desired frequency
    z = cp.Variable(num_operations)

    # interrupt tolerance
    G_max = cp.Variable() # TODO have a G_max for each machine
    g = cp.Variable(num_operations)

    # Hyperparameters: auto-size big-M (see top of schedule_window).
    H = _auto_big_m(workload.get_operations(), machine_combinations,
                    workload.machines, transfer_times, num_combinations)

    # Constraints
    constraints = []
    log = _constraints_section_logger(enabled=debug_constraints, constraints=constraints)
    build_all_start = time.perf_counter() if debug_constraints else None

    # (2) Each operation must be assigned to exactly one machine combination
    end = log("(2) assignment (alpha row-sum == 1)")
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    end()

    # (3) Precedence constraints: operation i must start after ALL its predecessors complete
    end = log("(3) precedence")
    for i in range(num_operations):
        predecessors = workload.operations[i].get_predecessors()
        for pred in predecessors:
            i_pred = workload.operations.index(pred)

            # Build duration vector for predecessor
            dur_vec_pred = [workload.operations[i_pred].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
            
            # For transfer time, we need to handle the product of two binary variables (alpha[i_pred, k_pred] * alpha[i, k_curr])
            # Since this is non-convex, we use an upper bound approach: use the maximum transfer time
            # This is conservative but ensures correctness (actual transfer time will be <= max)
            # For backward compatibility with singleton combinations, this works correctly
            max_transfer_time = 0
            for k_pred in range(num_combinations):
                for k_curr in range(num_combinations):
                    machine_pred = workload.machines.index(machine_combinations[k_pred][0])
                    machine_curr = workload.machines.index(machine_combinations[k_curr][0])
                    transfer_time_val = transfer_times[machine_pred][machine_curr]
                    max_transfer_time = max(max_transfer_time, transfer_time_val)
            
            # For transfer time, we use the maximum transfer time as an upper bound
            # This is conservative but ensures correctness and is DCP-compliant
            # The actual transfer time will be <= max_transfer_time, which is safe for scheduling
            # For backward compatibility (singleton combinations), this still works correctly
            # since the maximum is just the max over all machine pairs
            transfer_time_weighted = max_transfer_time
            
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + transfer_time_weighted
            )
    end()

    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    end = log("(4)(5) non-overlap (pairwise, overlapping combinations)")
    dep_desc = _compute_dependency_descendants_bitset(workload.operations)
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            if dep_desc is not None:
                if ((dep_desc[i] >> j) & 1) or ((dep_desc[j] >> i) & 1):
                    continue
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        constraints.append(
                            t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        constraints.append(
                            t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )
    end()

    # (6)
    end = log("(6) makespan lower bound (C_max)")
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines) for k in range(num_combinations)]
        constraints.append(
            C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
        )
    end()

    # (7) and (8) are covered by boolean argument of alpha and beta variables
    # all operations start at 0
    end = log("t >= 0")
    for i in range(num_operations):
        constraints.append(
            t[i] >= 0
        )
    end()

    # desired frequency
    end = log("desired frequency (z)")
    for i in range(num_operations):
        if nominal_start_times[i] >= 0:
            constraints.append(
                z[i] >= t[i] - nominal_start_times[i]
            )
            constraints.append(
                z[i] >= -(t[i] - nominal_start_times[i])
            )
        constraints.append(
            z[i] >= 0
        )
    end()

    # interrupt tolerance
    end = log("interrupt tolerance (g/G_max)")
    for i in range(num_operations):
        for j in range(i+1, num_operations):
            for k1 in range(num_combinations):
                for k2 in range(num_combinations):
                    # Only add constraint if combinations overlap
                    if workload.combinations_overlap(k1, k2):
                        dur_j_k2 = workload.operations[j].get_duration_for_combination(k2, machine_combinations, workload.machines)
                        dur_i_k1 = workload.operations[i].get_duration_for_combination(k1, machine_combinations, workload.machines)
                        constraints.append(
                            g[i] >= (t[i] - t[j] - dur_j_k2) - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                        )
                        constraints.append(
                            g[j] >= (t[j] - t[i] - dur_i_k1) - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                        )

    for i in range(num_operations):
        constraints.append(
            G_max <= g[i]
        )
        constraints.append(
            g[i] >= 0
        )
        constraints.append(
            g[i] <= gap_bound
        )
    end()

    if debug_constraints:
        print(f"[constraints] total build time: {time.perf_counter() - build_all_start:.3f}s (n={len(constraints)})")

    # Optimization problem
    objective_func = C_max + cp.sum(z) - 0.1*G_max
    # objective_func = C_max + cp.sum(z)
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value

def schedule_with_greedy_packing(workload: Workload, n_splits: int,
                                 target_diversity_weight: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    windows = greedy_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(
            window, target_diversity_weight=target_diversity_weight)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha

def schedule_with_convex_packing(workload: Workload, n_splits: int,
                                 target_diversity_weight: float = 0.0) -> Tuple[int, int]:
    windows = convex_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(
            window, target_diversity_weight=target_diversity_weight)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha