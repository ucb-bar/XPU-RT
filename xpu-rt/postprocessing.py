"""
Post-processing utilities for the XPU-RT scheduler.

Handles trimming periodic operations and serializing the solved schedule to JSON.
"""

from __future__ import annotations

import json
import os
import numpy as np

from workload import Workload, Operation
from typing import Dict, List, Tuple, Optional

try:
    from fusion import FusedOperation
except ImportError:
    # FusedOperation might not be available
    FusedOperation = None

from granularity_advisor import analyze_granularity, from_workload, group_by_periodicity


# `validate_schedule` and `write_validation_report` are NOT here. They used to
# be, and were simultaneously present in schedule_validation.py -- 448 lines
# byte-identical in both files, which is two places to fix when one of them is
# wrong. They live in `schedule_validation` now, which is where `schedulers.py`
# already documented them as living.
#
# Deliberately NOT re-exported from here. `schedule_validation` imports
# `count_overlaps` and `overlap_fixer` from this module, so the dependency runs
# base-helpers <- validation; a re-export would reverse it and make the pair
# circular. Nothing in the repo reached them through this module.


def automerge_enabled() -> bool:
    """Whether output_scheduled_json applies the adjacent auto-merge post-pass.

    Opt-in (XPURT_AUTOMERGE=1). This pass rewrites the emitted fixture, so
    leaving it on by default silently turns every cross-policy comparison into
    a comparison of policy+automerge. Callers should record this flag in their
    manifest alongside schedulers.compaction_enabled().

    XPURT_NO_AUTOMERGE=1 is honoured as an explicit force-off.
    """
    if os.environ.get("XPURT_NO_AUTOMERGE", "0") in ("1", "true", "True"):
        return False
    return os.environ.get("XPURT_AUTOMERGE", "0") in ("1", "true", "True")


def output_scheduled_json(
    combined_workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    output_path: str,
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None,
    profile_hw: dict | None = None,
    profiled_times_by_network: dict[str, dict[str, dict[int, dict]]] | None = None,
    pdb_hash: str | None = None,
    pdb_files: list[str] | None = None,
    combo_impls: list[str] | None = None,
):
    """
    Output a combined JSON file with all dispatches, their hardware targets, and start times.

    Args:
        combined_workload: Combined workload after scheduling
        t: Start times array from scheduling
        alpha: Assignment matrix from scheduling
        output_path: Path to save the output JSON file
        profiled_times_p: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for P-core
        profiled_times_e: Optional dict mapping dispatch_id -> {"time_ms": float, "module_name": str} for E-core
        combo_impls: Optional per-combination implementation tag ("rvv" /
            "ime"), parallel to machine_combinations. WITHOUT IT A
            HETEROGENEOUS SCHEDULE IS UNREADABLE: with `enable_impls` on, the
            same core appears in several combinations -- one per legal
            implementation -- so `hardware_target: CPU_P#1` says which core ran
            a dispatch and NOT whether it used the MAC unit. The two facts that
            make a placement heterogeneous are then both absent from the
            artifact, and a reader can only recover them by matching durations
            against the profile CSVs and hoping the costs differ.
    """
    machine_combinations = combined_workload.get_machine_combinations()

    # Network-keyed lookup helpers. The combined_profiled_p/e dicts are
    # keyed by dispatch_id alone, so when two networks share dispatch_ids
    # (dronet has 0..29, yolov8_nano has 0..N — they all overlap on
    # 0..29) the second network's `update()` overwrites the first's
    # entries. That made every dronet dispatch's `module_name` come out
    # as a yolov8_nano string in the schedule JSON.
    #
    # Fix: when `profiled_times_by_network` is supplied, route the
    # module_name lookup through the per-network bucket. Match each op's
    # operation_name against the longest base-network prefix from the
    # bucket — periodic instances like `dronet0_dispatch_22` resolve to
    # the base `dronet` (whose profile data covers all instances).
    base_network_prefixes = (
        sorted(profiled_times_by_network.keys(), key=len, reverse=True)
        if profiled_times_by_network else []
    )

    def _network_for_op(op_name: str) -> str | None:
        for base in base_network_prefixes:
            # Non-periodic: <base>_<dispatch_name>
            if op_name.startswith(base + "_"):
                return base
            # Periodic instance: <base><digits>_<dispatch_name>
            if op_name.startswith(base):
                rest = op_name[len(base):]
                i = 0
                while i < len(rest) and rest[i].isdigit():
                    i += 1
                if i > 0 and i < len(rest) and rest[i] == "_":
                    return base
        return None

    # First pass: collect all dispatch info with completion times
    dispatch_info_list = []

    for op_idx in range(len(combined_workload.operations)):
        op = combined_workload.operations[op_idx]

        # Get dispatch name from operation
        dispatch_name = op.operation_name if hasattr(op, 'operation_name') and op.operation_name else f"op_{op_idx}"

        # Get hardware target (which combination was assigned)
        combo_idx = int(np.argmax(alpha[op_idx]))
        hardware_target = "+".join(machine_combinations[combo_idx]) if len(machine_combinations[combo_idx]) > 1 else machine_combinations[combo_idx][0]

        # Get start time
        start_time = float(t[op_idx])

        # Get duration for the assigned combination
        duration = op.get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )

        # Get dispatch ID
        dispatch_id = op.operation_id if hasattr(op, 'operation_id') and op.operation_id is not None else op_idx

        # Get job name
        job_id = op.job_id if hasattr(op, 'job_id') and op.job_id is not None else 0
        job_name = combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}"

        # Get module name from profiled data if available. Prefer the
        # per-network bucket (no dispatch_id collisions across networks).
        # If we resolved a network but the bucket has no entry for this
        # dispatch_id (e.g. zero-cost IR ops like view/reshape that the
        # profile CSV skips), leave module_name as None — falling through
        # to the combined dicts here would pick up a *different*
        # network's entry by accident, which is the bug this routing was
        # introduced to fix.
        module_name = None
        net_id = _network_for_op(dispatch_name) if profiled_times_by_network else None
        if net_id and isinstance(dispatch_id, int):
            net_p = profiled_times_by_network[net_id].get("p", {})
            net_e = profiled_times_by_network[net_id].get("e", {})
            if dispatch_id in net_p:
                module_name = net_p[dispatch_id].get("module_name")
            elif dispatch_id in net_e:
                module_name = net_e[dispatch_id].get("module_name")
        elif not profiled_times_by_network:
            # No per-network data available — best-effort fallback to the
            # combined dicts (still collision-prone for multi-network
            # workloads, but better than nothing for single-network).
            if profiled_times_p and isinstance(dispatch_id, int) and dispatch_id in profiled_times_p:
                module_name = profiled_times_p[dispatch_id].get("module_name")
            elif profiled_times_e and isinstance(dispatch_id, int) and dispatch_id in profiled_times_e:
                module_name = profiled_times_e[dispatch_id].get("module_name")

        completion_time = start_time + float(duration)

        dispatch_info_list.append({
            'op_idx': op_idx,
            # Carried explicitly. The dispatch dict is built in a SECOND pass
            # below, where `combo_idx` from this loop is stale -- it holds
            # whatever the last operation happened to get. Reading it there
            # tagged every dispatch with the final op's implementation.
            'combo_idx': combo_idx,
            'dispatch_name': dispatch_name,
            'dispatch_id': dispatch_id,
            'hardware_target': hardware_target,
            'start_time': start_time,
            'duration': float(duration),
            'completion_time': completion_time,
            'job_name': job_name,
            'module_name': module_name,
            'op': op,
        })

    # Build time dependency mapping: for each hardware target, track dispatches sorted by completion time
    hardware_dispatch_map = {}  # hardware_target -> list of (completion_time, dispatch_name, start_time)

    for info in dispatch_info_list:
        hw_target = info['hardware_target']
        if hw_target not in hardware_dispatch_map:
            hardware_dispatch_map[hw_target] = []
        hardware_dispatch_map[hw_target].append((
            info['completion_time'],
            info['dispatch_name'],
            info['start_time']
        ))

    # Sort each hardware target's dispatches by completion time
    for hw_target in hardware_dispatch_map:
        hardware_dispatch_map[hw_target].sort(key=lambda x: x[0])  # Sort by completion_time

    # Build combined dispatches dictionary
    combined_dispatches = {}

    for info in dispatch_info_list:
        dispatch_name = info['dispatch_name']
        hardware_target = info['hardware_target']
        start_time = info['start_time']
        op = info['op']

        # Get dependencies (from operation predecessors)
        dependencies = []
        for pred_op in op.predecessors:
            # Find the index of this predecessor in the combined workload
            pred_idx = None
            for idx, combined_operation in enumerate(combined_workload.operations):
                if combined_operation == pred_op:
                    pred_idx = idx
                    break
            if pred_idx is not None:
                pred_dispatch_name = combined_workload.operations[pred_idx].operation_name if hasattr(combined_workload.operations[pred_idx], 'operation_name') and combined_workload.operations[pred_idx].operation_name else f"op_{pred_idx}"
                dependencies.append(pred_dispatch_name)

        # Find time dependency: previous dispatch on same hardware target
        time_dependency = None
        if hardware_target in hardware_dispatch_map:
            hw_dispatches = hardware_dispatch_map[hardware_target]
            # Find the dispatch that finished most recently before this one starts.
            # Strict `<` is required, not `<=`: two independent dispatches sharing
            # a core (e.g. parallel branches after a fork) can legitimately tie on
            # completion_time/start_time. Since the list is only sorted by
            # completion_time, ties break on incidental dispatch_info_list
            # insertion order, not true precedence -- with `<=` this let a later
            # dispatch's own successor be picked as its "predecessor" (observed:
            # dispatch_28 and dispatch_29 each pointed at the other), producing a
            # 2-cycle that ingest_xpurt_schedule.py's topological sort correctly
            # rejects. Ties are dropped entirely rather than guessed at; real data
            # dependencies (above) already order anything that must be ordered.
            for completion_time, prev_dispatch_name, prev_start_time in hw_dispatches:
                if completion_time < start_time and prev_dispatch_name != dispatch_name:
                    time_dependency = prev_dispatch_name
                elif completion_time >= start_time:
                    break  # No need to check further (sorted by completion time)

        # Create dispatch entry
        dispatch_entry = {
            "id": info['dispatch_id'],
            "ordinal": 1,  # Keep original structure
            "total": 1,
            "dependencies": dependencies,
            "hardware_target": hardware_target,
            "start_time": start_time,
            "duration": info['duration'],
            "job_name": info['job_name']
        }

        # Add module_name if available
        if info['module_name']:
            dispatch_entry["module_name"] = info['module_name']

        # Add time_dependency if found
        if time_dependency:
            dispatch_entry["time_dependency"] = time_dependency

        # Release semantics, stated rather than guessed.
        #
        # The runner infers these when the JSON omits them, and its heuristic is
        # a literal name match: only jobs whose name starts with "mlp" are
        # phase-locked, everything else is released the moment its predecessor
        # finishes (dispatch_types.h InferSchedulingPolicies). So DroNet -- and
        # any future model -- was chained instance-to-instance and started
        # *early*, which makes its measured lateness a comparison against a
        # release the runtime never enforced. Emitting the field explicitly sets
        # policies_from_json and retires the heuristic for every model at once.
        #
        # min_start_t is the periodic release k*T that workload_factory computed;
        # a root dispatch is one with no predecessors inside its instance.
        release_t = getattr(op, "min_start_t", None)
        if release_t is not None and not dependencies:
            dispatch_entry["release_policy"] = "phase_locked"
            dispatch_entry["release_us"] = float(release_t) * 1000.0
            dispatch_entry["time_dep_mode"] = "soft"
        else:
            dispatch_entry["release_policy"] = "immediate"
            dispatch_entry["time_dep_mode"] = "hard"

        # Propagate honest deadline-miss flag set by heuristic schedulers
        # (Phase A2). When present, downstream readers (Gantt overlay,
        # band-compliance audit) can mark the overrun directly without
        # re-deriving from workload metadata.
        if getattr(op, "deadline_miss", False):
            dispatch_entry["deadline_miss"] = True
            overrun = getattr(op, "deadline_overrun_us", None)
            if overrun is not None:
                dispatch_entry["deadline_overrun_us"] = float(overrun)

        # Which IMPLEMENTATION ran this dispatch, when the solve had more than
        # one to choose from. `hardware_target` names the core; with
        # `enable_impls` on the same core appears in several combinations and
        # the core alone does not say whether the MAC unit was used. Recorded
        # only when there is a choice to record, so a single-impl schedule is
        # byte-identical to before.
        _ci = info['combo_idx']
        if combo_impls is not None and _ci < len(combo_impls):
            dispatch_entry["impl"] = combo_impls[_ci]

        combined_dispatches[dispatch_name] = dispatch_entry

    # Feedback-driven compilation: derive periodic-network periods and
    # non-periodic granularity advice from the same live Operations used
    # above (precise -- real min_start_t/max_end_t, not inferred from
    # naming). Additive metadata only; never fails JSON output itself if
    # something here is unexpectedly malformed -- this signal is advisory,
    # not load-bearing. See granularity_advisor.py.
    periodic_networks = {}
    granularity_advice = []
    try:
        # The real network names, so `periodic_networks` is keyed by what the
        # networks are actually called. Written stripped, this metadata makes
        # every downstream consumer -- including the deadline scorer -- split
        # job names in the wrong place for any network ending in a digit.
        known = set(profiled_times_by_network or ()) or None
        records = from_workload(combined_workload, t, alpha, known)
        periodic_periods, _ = group_by_periodicity(records)
        periodic_networks = {base: round(period, 3) for base, period in periodic_periods.items()}
        granularity_advice = [advice.as_dict() for advice in analyze_granularity(records)]
    except Exception as e:
        print(f"warning: granularity advisor failed, omitting from output ({e})")

    # Create output JSON structure
    output_data = {
        "dot_file": "combined_schedule_periodic.json",
        "dispatches": combined_dispatches,
        "metadata": {
            "makespan": float(max(
                t[i] + combined_workload.operations[i].get_duration_for_combination(
                    np.argmax(alpha[i]), machine_combinations, combined_workload.machines
                )
                for i in range(len(combined_workload.operations))
            )),
            "num_operations": len(combined_workload.operations),
            "machines": combined_workload.machines,
            "machine_combinations": [combo if isinstance(combo, list) else [combo] for combo in machine_combinations],
            # Parallel to machine_combinations: which implementation each one
            # denotes. Absent when the solve had a single implementation.
            **({"combo_impls": list(combo_impls)} if combo_impls is not None else {}),
            # profile_hw persists the bitstream-level identity of each
            # CPU role (e.g. CPU_P → "gemmini_q31", CPU_E → "RVV") so that
            # downstream re-plotting (scripts/plot_scheduled_json.py) can
            # show real hardware names on the y-axis instead of just the
            # abstract CPU_P / CPU_E roles. Optional — older schedules
            # without this field still load fine.
            **({"profile_hw": profile_hw} if profile_hw else {}),
            # periodic_networks: {base_id -> inferred period_ms}, ground
            # truth for granularity_advisor.from_schedule_json() to prefer
            # over its naming-based fallback when analyzing this file later.
            # granularity_advice: feedback-driven-compilation signal -- see
            # granularity_advisor.py and README "Feedback-driven compilation".
            **({"periodic_networks": periodic_networks} if periodic_networks else {}),
            **({"granularity_advice": granularity_advice} if granularity_advice else {}),
            # pdb_hash / pdb_files persist the content fingerprint of
            # the profile CSVs that the solver consumed. The runtime
            # loader (pipeline/ingest_xpurt_schedule.py) recomputes the
            # hash over the SAME paths and refuses (or warns, depending
            # on MB_INGEST_STRICT_PDB_CHECK) when it differs — defends
            # against the "predicted 70 ms / measured 638 ms" trap that
            # killed v8 (the fixture was solved against a pre-bit-exact
            # PDB; the runtime ran bit-exact kernels against it).
            **({"pdb_hash": pdb_hash} if pdb_hash else {}),
            **({"pdb_files": pdb_files} if pdb_files else {}),
            # Exact CP-SAT runs attach their sequential optimality proof to the
            # workload. Persist it with the schedule so a figure cannot claim
            # "best baseline" without carrying the certificate that supports
            # the statement.
            **({"solver_certificate": combined_workload.solver_certificate}
               if getattr(combined_workload, "solver_certificate", None)
               else {}),
            **({"analytic_response_lower_bounds":
                combined_workload.analytic_response_lower_bounds}
               if getattr(combined_workload,
                          "analytic_response_lower_bounds", None)
               else {}),
        }
    }

    # Apply same-network adjacent auto-merge (schedule-time fusion of
    # back-to-back dispatches on the same core that have no external readers).
    #
    # OPT-IN via XPURT_AUTOMERGE=1. It used to be opt-out, but this pass
    # rewrites the emitted fixture -- collapsing dispatches and shifting start
    # times -- so leaving it on by default makes every cross-policy comparison
    # a comparison of policy+automerge, and makes per-instance intervals
    # derived from the fixture reflect the post-pass rather than the schedule.
    # XPURT_NO_AUTOMERGE=1 is still honoured as an explicit force-off.
    if automerge_enabled():
        try:
            from automerge import automerge_adjacent, automerge_savings
            before = output_data
            output_data = automerge_adjacent(output_data, max_gap_us=50.0,
                                             saved_handshake_us=5.0)
            savings = automerge_savings(before, output_data)
            if savings["pairs_merged"] > 0:
                print(f"automerge: collapsed {savings['pairs_merged']} "
                      f"adjacent same-network pair(s) → "
                      f"{savings['dispatches_after']} dispatches, "
                      f"makespan {savings['makespan_before']:.1f}µs "
                      f"→ {savings['makespan_after']:.1f}µs")
        except Exception as exc:
            print(f"warning: automerge pass skipped ({exc})")

    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nScheduled JSON saved to: {output_path}")


def trim_periodic_after_nonperiodic_makespan(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    horizon_ms: float | None = None,
) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Post-process the schedule to discard periodic/background operations that occur
    entirely after the last non-periodic operation completes.

    An operation is considered periodic/background if it has a time-window bound
    (min_start_t or max_end_t set). Non-periodic operations have both as None.

    We:
      1) Compute the cut point: the makespan over non-periodic operations
         only, or `horizon_ms` when the workload declares a longer span.
      2) Drop any periodic operation whose window starts at or after the cut.
         (i.e., its period does not overlap the interval of interest).

    `horizon_ms` is how long the workload says it runs.  Without it the cut
    is the non-periodic makespan alone, which treats every periodic task as
    background filler for the "real" work — right for a workload built
    around a yolov8 pass, wrong for a control loop that is the point of the
    workload.  A 40 ms yolo window would leave a 16 ms mlp_control with two
    instances on the plot however many the workload asked for.
    """
    if t is None or alpha is None or len(workload.operations) == 0:
        return workload, t, alpha

    # 1) Compute makespan over non-periodic operations
    nonperiodic_completion_times: list[float] = []
    for i, op in enumerate(workload.operations):
        is_periodic = (getattr(op, "min_start_t", None) is not None) or (getattr(op, "max_end_t", None) is not None)
        if is_periodic:
            continue
        # Completion time based on chosen machine
        combo_idx = int(np.argmax(alpha[i]))
        dur = op.get_duration_for_combination(combo_idx, workload.get_machine_combinations(), workload.machines)
        nonperiodic_completion_times.append(float(t[i] + dur))

    try:
        declared_horizon = max(0.0, float(horizon_ms or 0.0))
    except (TypeError, ValueError):
        declared_horizon = 0.0

    if not nonperiodic_completion_times and declared_horizon <= 0.0:
        # No non-periodic ops and no declared span: nothing to trim against.
        return workload, t, alpha

    nonperiodic_makespan = max(
        [declared_horizon] + nonperiodic_completion_times)

    # 2) Build keep mask: always keep non-periodic ops; for periodic, keep only
    #    those whose window overlaps [0, nonperiodic_makespan).
    keep_indices: list[int] = []
    for i, op in enumerate(workload.operations):
        min_start_t = getattr(op, "min_start_t", None)
        max_end_t = getattr(op, "max_end_t", None)
        is_periodic = (min_start_t is not None) or (max_end_t is not None)

        if not is_periodic:
            keep_indices.append(i)
            continue

        # If no explicit window, treat as non-periodic (already handled above).
        if min_start_t is None or max_end_t is None:
            keep_indices.append(i)
            continue

        # Period window [min_start_t, max_end_t) overlaps [0, nonperiodic_makespan) iff:
        #   min_start_t < nonperiodic_makespan and max_end_t > 0
        if (min_start_t < nonperiodic_makespan) and (max_end_t > 0):
            keep_indices.append(i)
        # else: drop this periodic op (it is entirely after the relevant horizon)

    if len(keep_indices) == len(workload.operations):
        # Nothing trimmed
        return workload, t, alpha

    # Build trimmed workload and schedule arrays
    trimmed_ops = [workload.operations[i] for i in keep_indices]
    trimmed_t = np.array([t[i] for i in keep_indices])
    trimmed_alpha = np.array([alpha[i] for i in keep_indices])

    trimmed_workload = Workload(
        trimmed_ops,
        workload.machines,
        workload.transfer_times,
        job_names=workload.job_names,
        machine_combinations=workload.machine_combinations,
    )

    return trimmed_workload, trimmed_t, trimmed_alpha

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




