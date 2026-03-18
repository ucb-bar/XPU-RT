"""
Test script for scheduling IREE dispatch graphs from hierarchical network dependencies.
Parses top-level network dependency JSON files and schedules them on configured
performant/efficient cores (defaults: CPU_P/CPU_E).
"""

import sys
import os
import json
import argparse
import csv
import glob
import numpy as np

# Add parent path to sys path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload import Workload, Operation
from workload_factory import create_workload_from_network_hierarchy, resolve_dispatch_deps_path
from scheduler import schedule
import plot

def load_networks_graph(json_path: str) -> dict:
    """Load a network dependencies JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _first_nonempty_string(candidates: list[object], default: str) -> str:
    for candidate in candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
    return default


def _coerce_positive_float(value: object, default: float, field_name: str) -> float:
    if value is None:
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        print(f"  (warning) invalid {field_name}={value!r}; using default {default}")
        return default
    if converted <= 0:
        print(f"  (warning) invalid {field_name}={value!r}; must be > 0. Using default {default}")
        return default
    return converted


def _coerce_nonnegative_float_or_none(value: object, default: float | None, field_name: str) -> float | None:
    if value is None:
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        print(f"  (warning) invalid {field_name}={value!r}; using default {default}")
        return default
    if converted < 0:
        print(f"  (warning) invalid {field_name}={value!r}; must be >= 0. Using default {default}")
        return default
    return converted


def _coerce_bool(value: object, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    print(f"  (warning) invalid {field_name}={value!r}; using default {default}")
    return default


def _coerce_int_in_range(value: object, default: int, field_name: str, *, min_value: int, max_value: int) -> int:
    if value is None:
        return default
    try:
        converted = int(value)
    except (TypeError, ValueError):
        print(f"  (warning) invalid {field_name}={value!r}; using default {default}")
        return default
    if converted < min_value or converted > max_value:
        print(
            f"  (warning) invalid {field_name}={value!r}; must be in [{min_value}, {max_value}]. "
            f"Using default {default}"
        )
        return default
    return converted


def _coerce_seed(value: object, default: int | None = 0) -> int | None:
    """
    Parse seed values:
      - negative => nondeterministic (None)
      - None => default
    """
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        print(f"  (warning) invalid random_seed={value!r}; using default {default}")
        return default
    return None if parsed < 0 else parsed


def _resolve_aux_json_path(path: str, repo_base_path: str, networks_json_path: str) -> str:
    """Resolve relative paths first from networks JSON dir, then from repo root."""
    if os.path.isabs(path):
        return path
    from_networks_dir = os.path.join(os.path.dirname(networks_json_path), path)
    if os.path.exists(from_networks_dir):
        return from_networks_dir
    return os.path.join(repo_base_path, path)


def _load_hardware_runtime_config(
    networks_data: dict,
    repo_base_path: str,
    networks_json_path: str,
) -> dict:
    """
    Load hardware/runtime config from:
      1) optional external JSON referenced by 'hardware_json_path'
      2) inline top-level 'hardware' object in networks JSON
      3) optional top-level 'scheduler' object for generic hyperparameters

    Supported keys (inline or in hardware JSON):
      hardware:
        machines: {cpu_p, cpu_e} OR [cpu_p_name, cpu_e_name]
        cpu_p: {name, profile_hw}
        cpu_e: {name, profile_hw}
        profile: {target, topo_tag}
        p_core_speedup: float
      scheduler:
        random_seed: int (or -1 for nondeterministic)
        solver_verbosity: int in [0, 4]
        time_limit: float seconds (>= 0)
        use_profiled: bool
        prune_periodic: bool
        restrict_makespan_to_nonperiodic: bool
    """
    defaults = {
        "cpu_p_name": "CPU_P",
        "cpu_e_name": "CPU_E",
        "cpu_p_profile_hw": "RVV",
        "cpu_e_profile_hw": "scalar",
        "profile_target": "spacemit_x60",
        "profile_topo_tag": "topo_0_1_2_3",
        "p_core_speedup": 1.5,
        "random_seed": 0,
        "solver_verbosity": 0,
        "time_limit": None,
        "use_profiled": False,
        "prune_periodic": True,
        "restrict_makespan_to_nonperiodic": True,
    }

    external_data: dict = {}
    hardware_json_path = networks_data.get("hardware_json_path")
    if isinstance(hardware_json_path, str) and hardware_json_path.strip():
        resolved_hardware_json_path = _resolve_aux_json_path(
            hardware_json_path.strip(),
            repo_base_path=repo_base_path,
            networks_json_path=networks_json_path,
        )
        if os.path.exists(resolved_hardware_json_path):
            try:
                with open(resolved_hardware_json_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    external_data = loaded
                else:
                    print(
                        "  (warning) hardware_json_path did not contain a JSON object: "
                        f"{resolved_hardware_json_path}. Ignoring."
                    )
            except Exception as e:
                print(
                    f"  (warning) failed to read hardware_json_path={resolved_hardware_json_path}: {e}. "
                    "Ignoring external hardware config."
                )
        else:
            print(f"  (warning) hardware_json_path not found: {resolved_hardware_json_path}. Using defaults/inline config.")

    external_hardware_cfg = external_data.get("hardware", external_data) if isinstance(external_data, dict) else {}
    if not isinstance(external_hardware_cfg, dict):
        external_hardware_cfg = {}

    inline_hardware_cfg = networks_data.get("hardware", {})
    if not isinstance(inline_hardware_cfg, dict):
        inline_hardware_cfg = {}

    scheduler_cfg = networks_data.get("scheduler", {})
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}

    hardware_cfg = _deep_merge_dict(external_hardware_cfg, inline_hardware_cfg)

    machines_cfg = hardware_cfg.get("machines", {})
    if not isinstance(machines_cfg, dict) and not isinstance(machines_cfg, list):
        machines_cfg = {}
    cpu_p_cfg = hardware_cfg.get("cpu_p", {})
    if not isinstance(cpu_p_cfg, dict):
        cpu_p_cfg = {}
    cpu_e_cfg = hardware_cfg.get("cpu_e", {})
    if not isinstance(cpu_e_cfg, dict):
        cpu_e_cfg = {}
    profile_cfg = hardware_cfg.get("profile", {})
    if not isinstance(profile_cfg, dict):
        profile_cfg = {}
    profile_hw_cfg = hardware_cfg.get("profile_hw", {})
    if not isinstance(profile_hw_cfg, dict):
        profile_hw_cfg = {}

    list_machine_p = machines_cfg[0] if isinstance(machines_cfg, list) and len(machines_cfg) > 0 else None
    list_machine_e = machines_cfg[1] if isinstance(machines_cfg, list) and len(machines_cfg) > 1 else None

    cpu_p_name = _first_nonempty_string(
        [
            cpu_p_cfg.get("name"),
            machines_cfg.get("cpu_p") if isinstance(machines_cfg, dict) else None,
            machines_cfg.get("CPU_p") if isinstance(machines_cfg, dict) else None,
            hardware_cfg.get("cpu_p_name"),
            hardware_cfg.get("cpu_p_machine"),
            networks_data.get("cpu_p_name"),
            networks_data.get("cpu_p_machine"),
            list_machine_p,
        ],
        default=defaults["cpu_p_name"],
    )
    cpu_e_name = _first_nonempty_string(
        [
            cpu_e_cfg.get("name"),
            machines_cfg.get("cpu_e") if isinstance(machines_cfg, dict) else None,
            machines_cfg.get("CPU_e") if isinstance(machines_cfg, dict) else None,
            hardware_cfg.get("cpu_e_name"),
            hardware_cfg.get("cpu_e_machine"),
            networks_data.get("cpu_e_name"),
            networks_data.get("cpu_e_machine"),
            list_machine_e,
        ],
        default=defaults["cpu_e_name"],
    )
    if cpu_p_name == cpu_e_name:
        raise ValueError(f"Invalid hardware config: cpu_p_name and cpu_e_name are both '{cpu_p_name}'. They must be distinct.")

    cpu_p_profile_hw = _first_nonempty_string(
        [
            cpu_p_cfg.get("profile_hw"),
            profile_hw_cfg.get("cpu_p"),
            hardware_cfg.get("cpu_p_profile_hw"),
            networks_data.get("cpu_p_profile_hw"),
        ],
        default=defaults["cpu_p_profile_hw"],
    )
    cpu_e_profile_hw = _first_nonempty_string(
        [
            cpu_e_cfg.get("profile_hw"),
            profile_hw_cfg.get("cpu_e"),
            hardware_cfg.get("cpu_e_profile_hw"),
            networks_data.get("cpu_e_profile_hw"),
        ],
        default=defaults["cpu_e_profile_hw"],
    )

    profile_target = _first_nonempty_string(
        [
            profile_cfg.get("target"),
            hardware_cfg.get("profile_target"),
            hardware_cfg.get("target"),
            networks_data.get("profile_target"),
            networks_data.get("target"),
        ],
        default=defaults["profile_target"],
    )
    profile_topo_tag = _first_nonempty_string(
        [
            profile_cfg.get("topo_tag"),
            hardware_cfg.get("topo_tag"),
            hardware_cfg.get("profile_topo_tag"),
            networks_data.get("profile_topo_tag"),
            networks_data.get("topo_tag"),
        ],
        default=defaults["profile_topo_tag"],
    )

    raw_speedup = (
        hardware_cfg.get("p_core_speedup")
        if hardware_cfg.get("p_core_speedup") is not None
        else scheduler_cfg.get("p_core_speedup", networks_data.get("p_core_speedup"))
    )
    p_core_speedup = _coerce_positive_float(
        raw_speedup,
        default=defaults["p_core_speedup"],
        field_name="p_core_speedup",
    )

    raw_seed = (
        scheduler_cfg.get("random_seed")
        if scheduler_cfg.get("random_seed") is not None
        else scheduler_cfg.get("seed")
    )
    if raw_seed is None:
        raw_seed = networks_data.get("random_seed", networks_data.get("seed"))
    random_seed = _coerce_seed(raw_seed, default=defaults["random_seed"])

    solver_verbosity = _coerce_int_in_range(
        scheduler_cfg.get("solver_verbosity", networks_data.get("solver_verbosity")),
        default=defaults["solver_verbosity"],
        field_name="solver_verbosity",
        min_value=0,
        max_value=4,
    )
    time_limit = _coerce_nonnegative_float_or_none(
        scheduler_cfg.get("time_limit", networks_data.get("time_limit")),
        default=defaults["time_limit"],
        field_name="time_limit",
    )
    use_profiled = _coerce_bool(
        scheduler_cfg.get("use_profiled", networks_data.get("use_profiled")),
        default=defaults["use_profiled"],
        field_name="use_profiled",
    )
    prune_periodic = _coerce_bool(
        scheduler_cfg.get("prune_periodic", networks_data.get("prune_periodic")),
        default=defaults["prune_periodic"],
        field_name="prune_periodic",
    )
    restrict_makespan_to_nonperiodic = _coerce_bool(
        scheduler_cfg.get(
            "restrict_makespan_to_nonperiodic",
            networks_data.get("restrict_makespan_to_nonperiodic"),
        ),
        default=defaults["restrict_makespan_to_nonperiodic"],
        field_name="restrict_makespan_to_nonperiodic",
    )

    return {
        "cpu_p_name": cpu_p_name,
        "cpu_e_name": cpu_e_name,
        "cpu_p_profile_hw": cpu_p_profile_hw,
        "cpu_e_profile_hw": cpu_e_profile_hw,
        "profile_target": profile_target,
        "profile_topo_tag": profile_topo_tag,
        "p_core_speedup": p_core_speedup,
        "random_seed": random_seed,
        "solver_verbosity": solver_verbosity,
        "time_limit": time_limit,
        "use_profiled": use_profiled,
        "prune_periodic": prune_periodic,
        "restrict_makespan_to_nonperiodic": restrict_makespan_to_nonperiodic,
    }


def load_profiled_times(csv_path: str) -> dict[int, dict]:
    """
    Load profiled runtimes from a CSV file.

    Expected columns:
      - dispatch_id
      - module_name (optional)
      - mean_time
      - mean_unit (assumed 'ms' if missing)
    
    Returns:
      dict mapping dispatch_id (int) -> {"time_ms": float, "module_name": str}
    """
    profiled: dict[int, dict] = {}
    if not os.path.exists(csv_path):
        return profiled
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dispatch_id_str = row.get("dispatch_id")
            if not dispatch_id_str:
                continue
            try:
                dispatch_id = int(dispatch_id_str)
            except ValueError:
                continue
            
            module_name = row.get("module_name", "")
            try:
                mean_time = float(row.get("mean_time", 0.0))
            except ValueError:
                continue
            unit = row.get("mean_unit", "ms")
            if unit == "us":
                mean_time_ms = mean_time / 1000.0
            elif unit == "s":
                mean_time_ms = mean_time * 1000.0
            else:
                mean_time_ms = mean_time
            profiled[dispatch_id] = {
                "time_ms": mean_time_ms,
                "module_name": module_name,
            }
    return profiled

def _find_profile_csvs_in_gen(
    repo_base_path: str,
    *,
    model: str,
    target: str,
    hw: str,
    basename: str,
    num_cores: int,
) -> dict[str, str]:
    """
    Find profiling results.csv files for each core topology.

    For num_cores=4, generates topo tags: topo_0, topo_0_1, topo_0_1_2, topo_0_1_2_3
    Returns dict mapping topo_tag -> csv_path for all found CSVs.
    """
    profile_root = os.path.join(repo_base_path, "gen", "profile")
    topo_tags = ["topo_" + "_".join(str(i) for i in range(n)) for n in range(1, num_cores + 1)]

    found: dict[str, str] = {}
    for topo_tag in topo_tags:
        # New layout (with input_tag subdir)
        pat1 = os.path.join(profile_root, hw, target, model, basename, "*", topo_tag, "results.csv")
        matches = glob.glob(pat1)

        # Back-compat layout (no input_tag subdir)
        if not matches:
            pat2 = os.path.join(profile_root, hw, target, model, basename, topo_tag, "results.csv")
            matches = glob.glob(pat2)

        if matches:
            found[topo_tag] = max(matches, key=lambda p: os.path.getmtime(p))

    return found

def output_scheduled_json(
    combined_workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    output_path: str,
    profiled_times_p: dict | None = None,
    profiled_times_e: dict | None = None
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
    """
    machine_combinations = combined_workload.get_machine_combinations()
    
    # First pass: collect all dispatch info with completion times
    dispatch_info_list = []
    
    for op_idx in range(len(combined_workload.operations)):
        op = combined_workload.operations[op_idx]
        
        # Get dispatch name from operation
        dispatch_name = op.operation_name if hasattr(op, 'operation_name') and op.operation_name else f"op_{op_idx}"
        
        # Get hardware target (which combination was assigned)
        combo_idx = np.argmax(alpha[op_idx])
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
        
        # Get module name from profiled data if available
        module_name = None
        if profiled_times_p and isinstance(dispatch_id, int) and dispatch_id in profiled_times_p:
            module_name = profiled_times_p[dispatch_id].get("module_name")
        elif profiled_times_e and isinstance(dispatch_id, int) and dispatch_id in profiled_times_e:
            module_name = profiled_times_e[dispatch_id].get("module_name")
        
        completion_time = start_time + float(duration)
        
        dispatch_info_list.append({
            'op_idx': op_idx,
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
            # Find the dispatch that finished most recently before this one starts
            for completion_time, prev_dispatch_name, prev_start_time in hw_dispatches:
                if completion_time <= start_time and prev_dispatch_name != dispatch_name:
                    time_dependency = prev_dispatch_name
                elif completion_time > start_time:
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
        
        combined_dispatches[dispatch_name] = dispatch_entry
    
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
            "machine_combinations": [combo if isinstance(combo, list) else [combo] for combo in machine_combinations]
        }
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nScheduled JSON saved to: {output_path}")


def _trim_periodic_after_nonperiodic_makespan(workload: Workload, t: np.ndarray, alpha: np.ndarray) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Post-process the schedule to discard periodic/background operations that occur
    entirely after the last non-periodic operation completes.
    
    An operation is considered periodic/background if it has a time-window bound
    (min_start_t or max_end_t set). Non-periodic operations have both as None.
    
    We:
      1) Compute the makespan over non-periodic operations only.
      2) Drop any periodic operation whose window starts at or after this makespan.
         (i.e., its period does not overlap the non-periodic makespan interval).
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

    if not nonperiodic_completion_times:
        # No non-periodic ops: nothing to trim
        return workload, t, alpha

    nonperiodic_makespan = max(nonperiodic_completion_times)

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

def schedule_iree_networks(
    networks_json_path: str = None,
    solver_verbosity: int | None = None,
    time_limit: float | None = None,
    random_seed: int | None = None,
    p_core_speedup: float | None = None,
    use_profiled: bool | None = None,
    prune_periodic: bool | None = None,
    restrict_makespan_to_nonperiodic: bool | None = None,
) -> tuple[Workload, np.ndarray, np.ndarray]:
    """
    Main function to schedule networks from a hierarchical network dependencies JSON file.
    
    Parameters:
    - networks_json_path: Path to the top-level networks dependencies JSON file.
                          If None, uses data/toplevel/networks_periodic_profile.json.
    - solver_verbosity: Optional MOSEK verbosity override. If None, use JSON/default.
    - time_limit: Optional solver time-limit override. If None, use JSON/default.
    - random_seed: Optional runtime seed override. None means use JSON config/default.
    - p_core_speedup: Optional P-core speedup override. None means use JSON config/default.
    - use_profiled: Optional profiled-runtimes override. None means use JSON/default.
    - prune_periodic: Optional periodic-pruning override. None means use JSON/default.
    - restrict_makespan_to_nonperiodic: Optional makespan-objective override. None means use JSON/default.
    """
    # Get script directory and repo base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_base_path = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Use default networks JSON if not provided
    if networks_json_path is None:
        networks_json_path = os.path.join(
        script_dir, 
        '..', 
            'data',
            'toplevel',
            'networks_periodic_profile.json'
        )
    
    # Resolve to absolute path
    if not os.path.isabs(networks_json_path):
        networks_json_path = os.path.join(repo_base_path, networks_json_path)
    
    print("=" * 60)
    print("Loading network hierarchy from JSON...")
    print("=" * 60)
    print(f"\nLoading networks from: {networks_json_path}")
    
    # Load network dependencies JSON
    networks_data = load_networks_graph(networks_json_path)

    # Resolve hardware and runtime settings from JSON (+ optional hardware JSON file).
    runtime_cfg = _load_hardware_runtime_config(
        networks_data,
        repo_base_path=repo_base_path,
        networks_json_path=networks_json_path,
    )
    cpu_p_name = runtime_cfg["cpu_p_name"]
    cpu_e_name = runtime_cfg["cpu_e_name"]
    cpu_p_profile_hw = runtime_cfg["cpu_p_profile_hw"]
    cpu_e_profile_hw = runtime_cfg["cpu_e_profile_hw"]
    profile_target = runtime_cfg["profile_target"]
    profile_topo_tag = runtime_cfg["profile_topo_tag"]
    effective_p_core_speedup = _coerce_positive_float(
        p_core_speedup,
        default=runtime_cfg["p_core_speedup"],
        field_name="p_core_speedup",
    )
    effective_random_seed = _coerce_seed(
        random_seed,
        default=runtime_cfg["random_seed"],
    )
    effective_solver_verbosity = _coerce_int_in_range(
        solver_verbosity,
        default=runtime_cfg["solver_verbosity"],
        field_name="solver_verbosity",
        min_value=0,
        max_value=4,
    )
    effective_time_limit = _coerce_nonnegative_float_or_none(
        time_limit,
        default=runtime_cfg["time_limit"],
        field_name="time_limit",
    )
    effective_use_profiled = _coerce_bool(
        use_profiled,
        default=runtime_cfg["use_profiled"],
        field_name="use_profiled",
    )
    effective_prune_periodic = _coerce_bool(
        prune_periodic,
        default=runtime_cfg["prune_periodic"],
        field_name="prune_periodic",
    )
    effective_restrict_makespan_to_nonperiodic = _coerce_bool(
        restrict_makespan_to_nonperiodic,
        default=runtime_cfg["restrict_makespan_to_nonperiodic"],
        field_name="restrict_makespan_to_nonperiodic",
    )
    rng = np.random.default_rng(effective_random_seed)
    
    # Print network information
    networks = networks_data.get('networks', {})
    edges = networks_data.get('edges', [])
    print(f"\nFound {len(networks)} networks:")
    for network_id, network_info in networks.items():
        print(f"  - {network_id} (id: {network_info.get('id')}, identifier: {network_info.get('identifier')})")
    
    print(f"\nFound {len(edges)} network-level dependencies:")
    for edge in edges:
        print(f"  - {edge.get('from')} → {edge.get('to')}")
    
    print("\nResolved runtime configuration:")
    print(f"  Machines: [{cpu_p_name}, {cpu_e_name}]")
    print(f"  Profile HW mapping: {cpu_p_name}->{cpu_p_profile_hw}, {cpu_e_name}->{cpu_e_profile_hw}")
    print(f"  Profile target/topology: target={profile_target}, topo_tag={profile_topo_tag}")
    print(f"  p_core_speedup: {effective_p_core_speedup}")
    print(f"  random_seed: {'nondeterministic' if effective_random_seed is None else effective_random_seed}")
    print(f"  solver_verbosity: {effective_solver_verbosity}")
    print(f"  time_limit: {effective_time_limit}")
    print(f"  use_profiled: {effective_use_profiled}")
    print(f"  prune_periodic: {effective_prune_periodic}")
    print(f"  restrict_makespan_to_nonperiodic: {effective_restrict_makespan_to_nonperiodic}")

    # Define machines (dual-core device)
    machines = {cpu_p_name : 4, cpu_e_name : 4}
    
    # Create transfer times matrix (zero transfer time between cores on same device)
    transfer_times = np.zeros((len(machines), len(machines)))

    # Optional: build profiled processing times if requested
    processing_times: dict[str, list[float]] | None = None
    combined_profiled_p: dict[int, dict] | None = None
    combined_profiled_e: dict[int, dict] | None = None
    if effective_use_profiled:
        print("\nUsing profiled runtimes where available...")
        processing_times = {}
        combined_profiled_p = {}
        combined_profiled_e = {}

        # Prefer profiles generated under gen/profile/... (produced by runtime/scripts/profile_remote.sh).
        # Mapping of machine -> profile-hw is JSON configurable.
        DEFAULT_TARGET = profile_target
        TOPO_TAG = profile_topo_tag

        def _basename_from_dispatch_deps_path(path: str) -> str:
            # Example:
            #   gen/vmfb/dronet/spacemit_x60/scalar/dronet.q.int8/dronet.q.int8_dispatch_graph.json
            # basename directory is ".../<basename>/file.json"
            return os.path.basename(os.path.dirname(path)) if path else ""

        def _profiles_for_network(
            net_id: str,
            net_info: dict,
            dispatch_deps_path: str,
        ) -> tuple[dict[int, dict] | None, dict[int, dict] | None]:
            target = DEFAULT_TARGET
            basename = _basename_from_dispatch_deps_path(dispatch_deps_path) or f"{net_id}.q.int8"

            # Profiles are usually stored under gen/profile/<hw>/<target>/<model>/<basename>/...
            # Some net IDs include instance suffixes (e.g. mlp0, mlp1), while profile model
            # directories use the base model name (e.g. mlp). Try several model candidates.
            basename_model = os.path.basename(basename).split(".")[0]
            model_candidates: list[str] = []
            for c in (
                net_id,
                net_info.get("identifier") if isinstance(net_info, dict) else None,
                basename_model,
            ):
                if isinstance(c, str) and c and c not in model_candidates:
                    model_candidates.append(c)
            if not model_candidates:
                model_candidates = [net_id]

            csv_p = None
            csv_e = None
            selected_model = model_candidates[0]
            for model_candidate in model_candidates:
                candidate_csv_p = _find_profile_csvs_in_gen(
                    repo_base_path,
                    model=model_candidate,
                    target=target,
                    hw=cpu_p_profile_hw,
                    basename=basename,
                    num_cores=machines[cpu_p_name],
                )
                candidate_csv_e = _find_profile_csvs_in_gen(
                    repo_base_path,
                    model=model_candidate,
                    target=target,
                    hw=cpu_e_profile_hw,
                    basename=basename,
                    num_cores=machines[cpu_e_name],
                )
                if candidate_csv_p or candidate_csv_e:
                    csv_p = candidate_csv_p
                    csv_e = candidate_csv_e
                    selected_model = model_candidate
                    break

            if selected_model != net_id:
                print(f"  (info) profile model fallback: net_id={net_id} -> model={selected_model}")

            # Merge all topo CSVs into a single profile dict, keyed by topo_tag
            prof_p: dict[str, dict[int, dict]] = {}
            for topo_tag, csv_path in csvs_p.items():
                loaded = load_profiled_times(csv_path)
                if not loaded:
                    print(f"  (warning) profile CSV had no usable rows: {csv_path}")
                else:
                    prof_p[topo_tag] = loaded

            prof_e: dict[str, dict[int, dict]] = {}
            for topo_tag, csv_path in csvs_e.items():
                loaded = load_profiled_times(csv_path)
                if not loaded:
                    print(f"  (warning) profile CSV had no usable rows: {csv_path}")
                else:
                    prof_e[topo_tag] = loaded

            return (prof_p or None), (prof_e or None)   

        for net_id, net_info in networks.items():
            dispatch_deps_path = net_info.get("dispatch_deps_path", "")
            full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
            if not os.path.exists(full_dispatch_path):
                continue

            prof_p, prof_e = _profiles_for_network(net_id, net_info, dispatch_deps_path)
            if prof_p is None and prof_e is None:
                continue

            # Combine profiled data for JSON output
            for topo_tag, topo_data in prof_p.items():
                if topo_tag not in combined_profiled_p:
                    combined_profiled_p[topo_tag] = {}
                combined_profiled_p[topo_tag].update(topo_data)
            if prof_e:
                for topo_tag, topo_data in prof_e.items():
                    if topo_tag not in combined_profiled_e:
                        combined_profiled_e[topo_tag] = {}
                    combined_profiled_e[topo_tag].update(topo_data)

            with open(full_dispatch_path, "r") as f:
                dispatch_data = json.load(f)
            dispatches = dispatch_data.get("dispatches", {})

            net_prefix = f"{net_id}_"

            for dispatch_name, dispatch_info in dispatches.items():
                dispatch_id = dispatch_info.get("id", None)
                prefixed_name = f"{net_prefix}{dispatch_name}"

                cpu_p_time: float
                cpu_e_time: float

                p_ms = None
                e_ms = None
                p_ms_by_topo: dict[str, float] = {}

                if isinstance(dispatch_id, int) and prof_p:
                    for topo_tag, topo_data in prof_p.items():
                        if dispatch_id in topo_data:
                            p_ms_by_topo[topo_tag] = topo_data[dispatch_id]["time_ms"]
               
                e_ms_by_topo: dict[str, float] = {}

                if isinstance(dispatch_id, int) and prof_e:
                    for topo_tag, topo_data in prof_e.items():
                        if dispatch_id in topo_data:
                            e_ms_by_topo[topo_tag] = topo_data[dispatch_id]["time_ms"]
                
                all_topo_tags = sorted(set(p_ms_by_topo.keys()) | set(e_ms_by_topo.keys()))
                if all_topo_tags:
                    times_by_topo: dict[str, list[float]] = {}
                    for topo_tag in all_topo_tags:
                        p_ms = p_ms_by_topo.get(topo_tag)
                        e_ms = e_ms_by_topo.get(topo_tag)

                        if p_ms is not None:
                            cpu_p_time = float(p_ms)
                            cpu_e_time = float(e_ms) if e_ms is not None else float(p_ms * effective_p_core_speedup)
                        elif e_ms is not None:
                            cpu_e_time = float(e_ms)
                            cpu_p_time = float(e_ms / effective_p_core_speedup)
                        else:
                            p_ms_synth = float(rng.uniform(2.0, 10.0))
                            cpu_p_time = p_ms_synth
                            cpu_e_time = p_ms_synth * effective_p_core_speedup

                        times_by_topo[topo_tag] = [cpu_p_time, cpu_e_time]
                    processing_times[prefixed_name] = times_by_topo
                else:
                    # No profiled data at all - use synthetic
                    p_ms_synth = float(rng.uniform(2.0, 10.0))
                    processing_times[prefixed_name] = {
                        "topo_0": [p_ms_synth, p_ms_synth * effective_p_core_speedup]
                    }    # Create workload from network hierarchy
    print(f"\nCreating workload from network hierarchy...")
    combined_workload = create_workload_from_network_hierarchy(
        networks_data=networks_data,
        repo_base_path=repo_base_path,
        machines=machines,
        transfer_times=transfer_times,
        p_core_speedup=effective_p_core_speedup,
        random_seed=effective_random_seed,
        processing_times=processing_times,
    )
    
    print(f"\nWorkload created successfully!")
    print(f"  Total operations: {len(combined_workload.operations)}")
    print(f"  Machines: {combined_workload.machines}")
    print(f"  Job names: {combined_workload.job_names}")
    
    # Print some statistics
    operations_with_multiple_predecessors = [
        op for op in combined_workload.operations if len(op.predecessors) > 1
    ]
    print(f"  Operations with multiple predecessors: {len(operations_with_multiple_predecessors)}")
    
    # Count independent jobs (operations with no predecessors)
    independent_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    print(f"  Independent jobs (can run in parallel): {independent_jobs}")
    
    # Schedule the combined workload
    print("\n" + "=" * 60)
    print("Scheduling combined workload...")
    print("=" * 60)
    result = schedule(
        combined_workload,
        solver_verbosity=effective_solver_verbosity,
        time_limit=effective_time_limit,
        restrict_makespan_to_nonperiodic=effective_restrict_makespan_to_nonperiodic,
        prune_cross_period_constraints=effective_prune_periodic,
    )
    t, alpha, _, _ = result  # Always returns 4 values now
    
    # Post-process schedule: optionally trim periodic/background operations that
    # occur entirely after the non-periodic makespan.
    if effective_prune_periodic:
        combined_workload, t, alpha = _trim_periodic_after_nonperiodic_makespan(combined_workload, t, alpha)
    
    # Calculate makespan
    makespan = max(t[i] + combined_workload.operations[i].get_durations()[np.argmax(alpha[i])] 
                   for i in range(len(combined_workload.operations)))
    
    print(f"\nScheduling completed!")
    print(f"Makespan: {makespan:.2f} time units")
    
    # Count operations assigned to each machine
    machine_names = list(combined_workload.machines)
    machine_counts = {name: 0 for name in machine_names}
    for i in range(len(alpha)):
        assigned_machine_idx = int(np.argmax(alpha[i]))
        if 0 <= assigned_machine_idx < len(machine_names):
            machine_counts[machine_names[assigned_machine_idx]] += 1
    
    print("\nCore assignments:")
    for machine_name in machine_names:
        if machine_name == cpu_p_name:
            role = "performant"
        elif machine_name == cpu_e_name:
            role = "efficient"
        else:
            role = "machine"
        print(f"  {machine_name} ({role}): {machine_counts[machine_name]} operations")
    
    # Count operations per network (group by job_id)
    network_stats = {}
    for op_idx, op in enumerate(combined_workload.operations):
        job_id = op.job_id
        if job_id not in network_stats:
            network_stats[job_id] = {
                "name": combined_workload.job_names[job_id] if job_id < len(combined_workload.job_names) else f"Job {job_id}",
                "machine_counts": {name: 0 for name in machine_names},
            }
        
        # Find which machine this operation is assigned to
        assigned_machine_idx = int(np.argmax(alpha[op_idx]))
        if 0 <= assigned_machine_idx < len(machine_names):
            assigned_machine_name = machine_names[assigned_machine_idx]
            network_stats[job_id]["machine_counts"][assigned_machine_name] += 1
    
    print(f"\nPer-network core assignments:")
    for job_id in sorted(network_stats.keys()):
        stats = network_stats[job_id]
        machine_counts_text = ", ".join(
            f"{machine_name}={stats['machine_counts'][machine_name]}"
            for machine_name in machine_names
        )
        print(f"  {stats['name'].capitalize()}: {machine_counts_text}")
    
    # Create plot
    os.makedirs("plots", exist_ok=True)
    
    # Count number of jobs (operations with no predecessors)
    num_jobs = sum(1 for op in combined_workload.operations if not op.predecessors)
    
    # Create title showing the networks
    network_names = [combined_workload.job_names[i] if i < len(combined_workload.job_names) else f"Job {i}" 
                     for i in sorted(set(op.job_id for op in combined_workload.operations))]
    title_networks = " + ".join([name.capitalize() for name in network_names])
    
    plot.plot_optimization_schedule(
        combined_workload.get_durations(),
        t,
        alpha,
        num_jobs,
        len(combined_workload.machines),
        combined_workload.machines,
        combined_workload.get_transfer_times(),
        save_path="plots/iree_combined_schedule_period.png",
        plot_title=f"{title_networks} Schedule on Dual-Core Device (Periodic)",
        workload=combined_workload
    )
    
    print(f"\nPlot saved to plots/iree_combined_schedule_period.png")
    
    # Output combined JSON file with scheduling information
    os.makedirs("schedules", exist_ok=True)
    # Extract base filename from input JSON path
    input_json_basename = os.path.basename(networks_json_path)
    input_json_name = os.path.splitext(input_json_basename)[0]  # Remove .json extension
    json_output_path = f"schedules/scheduled_{input_json_name}.json"
    if effective_use_profiled:
        json_output_path = json_output_path.replace(".json", "_profiled.json")
    
    print(f"\nOutputting scheduled JSON...")
    output_scheduled_json(
        combined_workload=combined_workload,
        t=t,
        alpha=alpha,
        output_path=json_output_path,
        profiled_times_p=combined_profiled_p,
        profiled_times_e=combined_profiled_e
    )
    
    return combined_workload, t, alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule IREE networks from hierarchical network dependencies JSON file"
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default="data/toplevel/networks_periodic_profile.json",
        help="Path to the top-level networks dependencies JSON file (default: data/toplevel/networks_periodic_profile.json)"
    )
    parser.add_argument(
        "--solver-verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="MOSEK solver verbosity level override: 0=silent, 1=errors, 2=warnings, 3=info, 4=detailed progress. If omitted, use JSON/default.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20,
        help="Maximum optimization time in seconds (override). If omitted, use JSON/default.",
    )
    parser.add_argument(
        "--profiled",
        dest="profiled",
        action="store_true",
        default=True,
        help="Enable profiled runtimes (default: enabled).",
    )
    parser.add_argument(
        "--no-profiled",
        dest="profiled",
        action="store_false",
        help="Disable profiled runtimes.",
    )
    parser.add_argument(
        "--prune-periods",
        dest="prune_periodic",
        action="store_true",
        default=None,
        help="Enable pruning of periodic operations (override).",
    )
    parser.add_argument(
        "--no-prune-periods",
        dest="prune_periodic",
        action="store_false",
        help="Disable pruning of periodic operations (override).",
    )
    parser.add_argument(
        "--restrict-makespan-to-nonperiodic",
        dest="restrict_makespan_to_nonperiodic",
        action="store_true",
        default=None,
        help="Restrict makespan objective to non-periodic operations (override).",
    )
    parser.add_argument(
        "--include-periodic-in-makespan",
        dest="restrict_makespan_to_nonperiodic",
        action="store_false",
        help="Include periodic operations in makespan objective (override).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for synthetic runtime generation. If omitted, uses JSON config (scheduler.random_seed/seed) or 0. Use -1 for nondeterministic.",
    )
    parser.add_argument(
        "--p-core-speedup",
        type=float,
        default=None,
        help="Override P-core speedup factor. If omitted, uses JSON config (hardware/scheduler p_core_speedup) or 1.5.",
    )
    args = parser.parse_args()
    
    seed = None if args.random_seed is not None and args.random_seed < 0 else args.random_seed
    
    schedule_iree_networks(
        networks_json_path=args.networks_json,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
        p_core_speedup=args.p_core_speedup,
        use_profiled=args.profiled,
        prune_periodic=args.prune_periodic,
        restrict_makespan_to_nonperiodic=args.restrict_makespan_to_nonperiodic,
    )
