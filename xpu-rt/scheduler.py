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

# Feedback-driven compilation: post-schedule dispatch-granularity advisor.
# Re-exported here so `scheduler.analyze_granularity(...)` works without
# changing schedule()'s own (t, alpha, fused_workload, fusion_map) return
# contract, which ModelBlaster's XPU-RT scheduler bridge scripts depend on.
try:
    from .granularity_advisor import analyze_granularity, from_workload
except ImportError:
    from granularity_advisor import analyze_granularity, from_workload


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


def _compute_big_m(workload_or_window, machine_combinations: list, machines: list,
                   transfer_times, *, min_value: float = 5000.0, slack: float = 2.0) -> float:
    """
    Compute a Big-M for non-overlap / precedence constraints.

    Tight upper bound: every op runs serially using its slowest combination,
    plus the worst-case transfer time per op-pair. Multiplied by `slack` and
    floored at `min_value` so simple workloads behave as before.

    Returns a scalar suitable for use as the H constant in disjunctive
    scheduling constraints.
    """
    ops = list(workload_or_window.operations)
    if not ops:
        return float(min_value)
    sum_max_dur = 0.0
    for op in ops:
        max_dur = 0.0
        for k in range(len(machine_combinations)):
            d = op.get_duration_for_combination(k, machine_combinations, machines)
            if d > max_dur:
                max_dur = float(d)
        sum_max_dur += max_dur
    try:
        max_transfer = float(np.max(transfer_times)) if transfer_times is not None else 0.0
    except (TypeError, ValueError):
        max_transfer = 0.0
    bound = (sum_max_dur + max_transfer * len(ops)) * slack
    return max(float(min_value), bound)


def schedule_window(window: Window, debug_constraints: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    num_operations = len(window.operations)
    machine_combinations = window.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = window.get_transfer_times()

    alpha = cp.Variable((num_operations, num_combinations), boolean=True)
    beta = cp.Variable((num_operations, num_operations), boolean=True)
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters
    H = _compute_big_m(window, machine_combinations, window.machines, transfer_times)

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

    # (6)
    end = log("(6) makespan lower bound (C_max)")
    for i in range(num_operations):
        # Build duration vector for all combinations
        dur_vec = [window.operations[i].get_duration_for_combination(k, machine_combinations, window.machines) for k in range(num_combinations)]
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

    # Optimization problem
    objective = cp.Minimize(objective_func)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)
    return t.value, alpha.value

# cvxpy solvers that can handle the boolean variables this model needs.
# CLARABEL/SCS/OSQP/DAQP/ECOS are continuous-only and reject it outright.
MIP_CAPABLE_SOLVERS = ("MOSEK", "HIGHS", "SCIPY", "GUROBI", "CPLEX", "SCIP",
                       "CBC", "GLPK_MI", "XPRESS")


def _solver_options(solver_name: str, time_limit: Optional[float],
                    verbose: bool) -> dict:
    """Per-backend spelling of "stop after this many seconds".

    Every solver names its own time limit, and cvxpy passes these through
    untouched, so a wrong key is silently ignored rather than rejected —
    which reads as "the time limit does nothing" at the call site.
    """
    if time_limit is None or time_limit <= 0:
        return {}
    name = solver_name.upper()
    if name == "MOSEK":
        return {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit}}
    if name == "HIGHS":
        return {"time_limit": float(time_limit)}
    if name == "SCIPY":
        return {"scipy_options": {"time_limit": float(time_limit)}}
    if name in ("GUROBI", "CPLEX", "XPRESS"):
        return {"TimeLimit": float(time_limit)}
    if name == "SCIP":
        return {"scip_params": {"limits/time": float(time_limit)}}
    if verbose:
        print(f"warning: no time-limit parameter known for solver {solver_name!r}; "
              f"running it unbounded")
    return {}


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
    cvxpy_solver: str = "MOSEK",
    warm_start=None,
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
        cvxpy_solver: which cvxpy backend to solve with. Must handle boolean
                   variables — see MIP_CAPABLE_SOLVERS. Defaults to MOSEK,
                   the historical hardcoded choice.
        warm_start: an existing (t, alpha) to hand the solver as its starting
                   integer solution (a "MIP start"). For MOSEK this sets the
                   variables' .value and turns on MSK_IPAR_MIO_CONSTRUCT_SOL,
                   which makes it try to build a feasible incumbent from them
                   rather than searching for one from scratch.
    
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
        import os
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
    t = cp.Variable(num_operations)
    C_max = cp.Variable()

    # Hyperparameters
    H = _compute_big_m(workload, machine_combinations, workload.machines, transfer_times)

    # Constraints
    constraints = []
    constraints_debug_enabled = debug_constraints or verbose
    log = _constraints_section_logger(enabled=constraints_debug_enabled, constraints=constraints)
    build_all_start = time.perf_counter()

    # `time_limit` used to bound only the solver call --
    # MSK_DPAR_OPTIMIZER_MAX_TIME and friends, set in _solver_options -- while
    # everything below it ran unbounded. Building the model is not a
    # formality: the (4)(5) rows are O(surviving pairs x combinations^2) cvxpy
    # expressions, and on a 295-operation workload that is ~18 minutes of
    # construction against a 120 s budget. The flag therefore did not mean what
    # it says, and the overrun was silent -- the run simply appeared to hang.
    #
    # Construction now shares the budget: it is checked against a deadline as
    # it goes, and whatever it consumes is deducted from what the solver is
    # given, so `time_limit` bounds total wall time. Exhausting it during
    # construction raises rather than continuing, because a model that cannot
    # be built inside the budget cannot be solved inside it either.
    _budget = float(time_limit) if time_limit and time_limit > 0 else None
    _deadline = (build_all_start + _budget) if _budget else None

    def _check_budget(where: str, done: int = 0, total: int = 0):
        if _deadline is None or time.perf_counter() <= _deadline:
            return
        spent = time.perf_counter() - build_all_start
        progress = f", {done}/{total} of them built" if total else ""
        raise TimeoutError(
            f"MILP model construction hit the {_budget:g} s time limit after "
            f"{spent:.1f} s, while building {where}{progress}; "
            f"{len(constraints)} constraints so far for {num_operations} "
            f"operations x {num_combinations} combinations. cvxpy builds every "
            f"row as a symbolic expression and the count grows quadratically "
            f"in operations, so this does not finish by waiting a little "
            f"longer. Raise time_limit, shrink the workload, or use a backend "
            f"that skips cvxpy: --solver milp_native (MOSEK's own API) or "
            f"--solver cpsat.")

    # (2) Each operation must be assigned to exactly one machine combination
    end = log("(2) assignment (alpha row-sum == 1)")
    for i in range(num_operations):
        constraints.append(
            cp.sum(alpha[i, :]) == 1
        )
    end()

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

    # (4) and (5) Non-overlap constraints: if two operations are assigned to overlapping combinations, enforce ordering
    #
    # `beta` used to be a dense (num_operations x num_operations) boolean
    # matrix, allocated before any of the pruning below ran. The pruning is
    # very effective on the *constraints* — on the QRB5165 3-way workload it
    # drops 94% of the (4)(5) pairs — but nothing shrank the variable, so
    # MOSEK still had to presolve an ordering bit for every pair, including
    # the pruned ones: 36,481 of that model's 37,246 variables, ~95% of them
    # never referenced by a constraint. Deciding the surviving pairs first
    # and allocating one ordering bit per surviving pair keeps the model
    # identical and makes the variable count track the pruning.
    end = log("(4)(5) non-overlap (pairwise, overlapping combinations)")
    dep_desc = None
    if prune_overlap_constraints_for_dependency_chain:
        dep_desc = _compute_dependency_descendants_bitset(workload.operations)

    def _pair_needs_ordering(i: int, j: int) -> bool:
        op_i = workload.operations[i]
        op_j = workload.operations[j]
        # Skip pairs whose time windows cannot overlap.
        if prune_cross_period_constraints and not _periods_overlap(op_i, op_j):
            return False
        # Skip pairs already ordered by a dependency chain (i ->* j or
        # j ->* i): the precedence constraints already separate them in time.
        if dep_desc is not None:
            if ((dep_desc[i] >> j) & 1) or ((dep_desc[j] >> i) & 1):
                return False
        return True

    ordering_pairs = [(i, j)
                      for i in range(num_operations)
                      for j in range(i + 1, num_operations)
                      if _pair_needs_ordering(i, j)]
    beta_index = {pair: n for n, pair in enumerate(ordering_pairs)}
    # cvxpy rejects a zero-length variable, and a workload with no competing
    # pairs (a single serial chain, say) legitimately produces none.
    beta = cp.Variable(max(1, len(ordering_pairs)), boolean=True)

    for _pair_n, (i, j) in enumerate(ordering_pairs):
        # This loop dominates construction; check often enough to stop
        # promptly, rarely enough not to matter.
        if not _pair_n & 0x1FF:
            _check_budget("(4)(5) pairwise non-overlap rows",
                          _pair_n, len(ordering_pairs))
        beta_ij = beta[beta_index[(i, j)]]
        for k1 in range(num_combinations):
            for k2 in range(num_combinations):
                # Only add constraint if combinations overlap
                if workload.combinations_overlap(k1, k2):
                    # (4) Operation i starts after j finishes (if i is on k1 and j is on k2)
                    dur_j_k2 = workload.operations[j].get_duration_for_combination(
                        k2, machine_combinations, workload.machines
                    )
                    constraints.append(
                        t[i] >= t[j] + dur_j_k2 - (2 - alpha[i, k1] - alpha[j, k2] + beta_ij) * H
                    )
                    # (5) Operation j starts after i finishes (if j is on k2 and i is on k1)
                    dur_i_k1 = workload.operations[i].get_duration_for_combination(
                        k1, machine_combinations, workload.machines
                    )
                    constraints.append(
                        t[j] >= t[i] + dur_i_k1 - (3 - alpha[i, k1] - alpha[j, k2] - beta_ij) * H
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
        # If there are no non-periodic operations, C_max would be unconstrained
        # from below and the objective trivial -- MOSEK then fails outright with
        # a SolverError rather than returning a degenerate answer. A purely
        # periodic workload is a legitimate input (it just has no non-periodic
        # work to pack against), so fall back to bounding C_max over ALL
        # operations. That is exactly the `else` branch's semantics, and it
        # keeps the intended behaviour whenever non-periodic ops do exist.
        if not non_periodic_ops_exist:
            for i in range(num_operations):
                dur_vec = [
                    workload.operations[i].get_duration_for_combination(k, machine_combinations, workload.machines)
                    for k in range(num_combinations)
                ]
                constraints.append(
                    C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :]))
                )
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

    # Optimization problem
    objective = cp.Minimize(C_max)
    problem = cp.Problem(objective, constraints)
    
    # Print problem statistics
    if verbose:
        print(f"\n{'='*60}")
        print("OPTIMIZATION PROBLEM STATISTICS")
        print(f"{'='*60}")
        print(f"Number of operations: {num_operations}")
        print(f"Number of machine combinations: {num_combinations}")
        num_vars = (num_operations * num_combinations + len(ordering_pairs)
                    + num_operations + 1)
        print(f"Number of variables: {num_vars}")
        print(f"  - alpha (operation->combination): {num_operations * num_combinations}")
        print(f"  - beta (operation ordering): {len(ordering_pairs)} "
              f"(dense would be {num_operations * num_operations})")
        print(f"  - t (start times): {num_operations}")
        print(f"  - C_max (makespan): 1")
        print(f"Number of constraints: {len(constraints)}")
        print(f"{'='*60}\n")
        print("Starting optimization...")
        start_time = time.time()
    
    solver_name = (cvxpy_solver or "MOSEK").upper()
    if solver_name not in MIP_CAPABLE_SOLVERS:
        raise ValueError(
            f"cvxpy_solver {solver_name!r} cannot solve a model with boolean "
            f"variables. MIP-capable backends: {', '.join(MIP_CAPABLE_SOLVERS)}."
        )
    if solver_name not in cp.installed_solvers():
        raise ValueError(
            f"cvxpy_solver {solver_name!r} is not installed in this environment. "
            f"Installed: {', '.join(cp.installed_solvers())}."
        )
    solver_verbose = verbose or (solver_verbosity > 0)
    # Whatever construction spent is no longer available to the solver, or
    # `time_limit` would be a per-phase allowance rather than a wall-clock one.
    _solver_budget = time_limit
    if _budget is not None:
        _build_s = time.perf_counter() - build_all_start
        _check_budget("the model", 0, 0)
        _solver_budget = max(_budget - _build_s, 1e-3)
        # `problem.solve` does not go straight to MOSEK: cvxpy first
        # canonicalises the whole model, and that pass is neither
        # interruptible nor covered by MSK_DPAR_OPTIMIZER_MAX_TIME. It is the
        # real cost -- on a 295-op workload the rows take ~114 s to build and
        # canonicalisation then runs for many minutes, which is where the
        # 18-minute overrun against a 120 s budget came from.
        #
        # Since it cannot be bounded from in here, refuse instead of entering
        # it with a budget that cannot cover it. Canonicalising N rows costs at
        # least as much as building them did, so "less time left than building
        # took" is a conservative test for "this cannot finish in budget".
        if _solver_budget < _build_s:
            raise TimeoutError(
                f"MILP model built in {_build_s:.1f} s of a {_budget:g} s time "
                f"limit, leaving {_solver_budget:.1f} s -- not enough to "
                f"canonicalise it, let alone solve it. cvxpy canonicalisation "
                f"runs inside problem.solve(), is not interruptible, and is "
                f"not covered by the solver's own time limit, so continuing "
                f"here would overrun the budget by minutes with no way to stop "
                f"({len(constraints)} constraints, {len(ordering_pairs)} "
                f"ordering pairs, {num_operations} operations). Raise "
                f"time_limit, shrink the workload, or use a backend that "
                f"skips cvxpy: --solver milp_native (MOSEK's own API) or "
                f"--solver cpsat.")
        if verbose:
            print(f"model built in {_build_s:.1f} s; "
                  f"{_solver_budget:.1f} s of the {_budget:g} s budget left "
                  f"for {solver_name}")
    opts = _solver_options(solver_name, _solver_budget, verbose)
    if warm_start is not None:
        ws_t, ws_alpha = warm_start
        try:
            ws_t = np.asarray(ws_t, dtype=float)
            chosen = [int(np.argmax(row)) for row in np.asarray(ws_alpha)]
            t.value = ws_t
            a = np.zeros((num_operations, num_combinations))
            for i, c in enumerate(chosen):
                a[i, c] = 1.0
            alpha.value = a

            # `beta` is the bulk of the integer solution — 8,012 of the 8,496
            # booleans on a 242-op instance — and leaving it unset makes the
            # supplied point 94% unspecified, which no MIP start can use.
            # Constraints (4)(5) define it as: beta[i,j] == 1 means i runs
            # before j, 0 means j runs before i.
            finish = np.array([ws_t[i] + workload.operations[i]
                               .get_duration_for_combination(
                                   chosen[i], machine_combinations, workload.machines)
                               for i in range(num_operations)])
            b = np.zeros(max(1, len(ordering_pairs)))
            for idx, (i, j) in enumerate(ordering_pairs):
                b[idx] = 1.0 if finish[i] <= ws_t[j] + 1e-9 else 0.0
            beta.value = b

            # C_max bounds the *finish* of the objective set, not the start.
            targets = [i for i in range(num_operations)
                       if not (restrict_makespan_to_nonperiodic
                               and (workload.operations[i].min_start_t is not None
                                    or workload.operations[i].max_end_t is not None))]
            C_max.value = float(finish[targets].max() if targets else finish.max())
            opts["warm_start"] = True
            if solver_name == "MOSEK":
                # Without CONSTRUCT_SOL MOSEK ignores the supplied integer
                # values entirely; with it, it tries to repair them into a
                # feasible incumbent before branching.
                opts.setdefault("mosek_params", {})
                opts["mosek_params"]["MSK_IPAR_MIO_CONSTRUCT_SOL"] = 1
            if verbose:
                print(f"warm start supplied to {solver_name}: "
                      f"{num_operations} starts, {num_operations * num_combinations} "
                      f"alpha, {len(ordering_pairs)} beta, C_max={C_max.value:.2f}")
        except Exception as exc:
            print(f"warning: could not apply warm start ({exc}); solving cold")
    if verbose and time_limit:
        print(f"Solving with {solver_name}, time limit {_solver_budget:.1f} s")
    problem.solve(solver=solver_name, verbose=solver_verbose, **opts)
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"{'='*60}")

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)

    t_result = t.value
    alpha_result = alpha.value
    
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

    # Hyperparameters
    H = _compute_big_m(workload, machine_combinations, workload.machines, transfer_times)

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

def schedule_with_greedy_packing(workload: Workload, n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    windows = greedy_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha

def schedule_with_convex_packing(workload: Workload, n_splits: int) -> Tuple[int, int]:
    windows = convex_packing(workload, n_splits)

    solutions = []
    for i, window in enumerate(windows):
        t, alpha = schedule_window(window)
        solutions.append((t, alpha))

    t, alpha = combine_solved_windows(workload, windows, solutions)

    return t, alpha