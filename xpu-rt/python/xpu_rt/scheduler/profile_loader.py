"""
Profile data loading utilities for the XPU-RT scheduler.

Handles discovering, parsing, and assembling profiled dispatch runtimes
from CSV files produced by runtime/scripts/profile_remote.sh.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from xpu_rt.scheduler.workload_factory import (
    resolve_dispatch_deps_path,
    topo_tag_for_combination,
    machine_type_prefix,
)


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


def find_profile_csv(
    repo_base_path: str,
    *,
    model: str,
    target: str,
    hw: str,
    basename: str,
    topo_tag: str = "topo_0_1_2_3",
) -> str | None:
    """
    Find a profiling results.csv produced by runtime/scripts/profile_remote.sh.

    Expected layout:
      gen/profile/<hw>/<target>/<model>/<basename>/<input_tag>/<topo_tag>/results.csv

    We pick the most recently modified match.
    """
    profile_root = os.path.join(repo_base_path, "gen", "profile")

    # New layout (with input_tag subdir).
    pat1 = os.path.join(profile_root, hw, target, model, basename, "*", topo_tag, "results.csv")
    matches = glob.glob(pat1)

    # Back-compat layout (no input_tag subdir).
    if not matches:
        pat2 = os.path.join(profile_root, hw, target, model, basename, topo_tag, "results.csv")
        matches = glob.glob(pat2)

    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))


def _basename_from_dispatch_deps_path(path: str) -> str:
    """Extract the parent directory name from a dispatch deps path."""
    return os.path.basename(os.path.dirname(path)) if path else ""


def _model_candidates(net_id: str, net_info: dict, dispatch_deps_path: str) -> list[str]:
    """Return candidate model names to try when searching for profiling CSVs."""
    basename = _basename_from_dispatch_deps_path(dispatch_deps_path) or f"{net_id}.q.int8"
    basename_model = os.path.basename(basename).split(".")[0]
    candidates = []
    for c in (net_id, net_info.get("identifier"), basename_model):
        if isinstance(c, str) and c and c not in candidates:
            candidates.append(c)
    return candidates or [net_id]


def _load_all_topo_profiles(
    net_id: str,
    net_info: dict,
    dispatch_deps_path: str,
    repo_base_path: str,
    profile_target: str,
    combo_hw: list[str],
    machine_combinations: list[list[str]],
) -> dict[tuple[str, str], dict[int, dict]]:
    """
    Load profiles for all (hw, topo) combinations for a network.

    Returns {(hw, topo): {dispatch_id: {"time_ms", "module_name"}}}.
    """
    basename = _basename_from_dispatch_deps_path(dispatch_deps_path) or f"{net_id}.q.int8"
    candidates = _model_candidates(net_id, net_info, dispatch_deps_path)
    profiles: dict[tuple[str, str], dict[int, dict]] = {}

    hw_types = set(combo_hw)
    topo_tags = set(topo_tag_for_combination(c) for c in machine_combinations)

    for hw in hw_types:
        for topo in topo_tags:
            for model_candidate in candidates:
                csv_path = find_profile_csv(
                    repo_base_path, model=model_candidate,
                    target=profile_target, hw=hw, basename=basename, topo_tag=topo,
                )
                if csv_path:
                    prof = load_profiled_times(csv_path)
                    if prof:
                        profiles[(hw, topo)] = prof
                        if model_candidate != net_id:
                            print(f"  (info) profile fallback: {net_id}/{hw}/{topo} -> model={model_candidate}")
                    break
    return profiles


def load_profiled_processing_times(
    networks: dict,
    repo_base_path: str,
    machine_combinations: list[list[str]],
    combo_hw: list[str],
    profile_target: str,
    cpu_p_profile_hw: str,
    cpu_e_profile_hw: str,
    rng: np.random.Generator,
    p_core_speedup: float,
) -> tuple[dict[str, list[float]], dict[int, dict], dict[int, dict]]:
    """
    Load profiled processing times for all networks and dispatches.

    For each dispatch in each network, builds a list of processing times
    (one per machine combination) from profiled CSVs. Falls back to
    synthetic times when profiled data is missing.

    Returns:
      (processing_times, combined_profiled_p, combined_profiled_e) where:
      - processing_times: {prefixed_dispatch_name: [time_per_combination]}
      - combined_profiled_p: {dispatch_id: {"time_ms", "module_name"}} for P-cores
      - combined_profiled_e: {dispatch_id: {"time_ms", "module_name"}} for E-cores
    """
    processing_times: dict[str, list[float]] = {}
    combined_profiled_p: dict[int, dict] = {}
    combined_profiled_e: dict[int, dict] = {}

    for net_id, net_info in networks.items():
        dispatch_deps_path = net_info.get("dispatch_deps_path", "")
        full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
        if not os.path.exists(full_dispatch_path):
            continue

        all_profiles = _load_all_topo_profiles(
            net_id, net_info, dispatch_deps_path,
            repo_base_path, profile_target, combo_hw, machine_combinations,
        )
        if not all_profiles:
            continue

        with open(full_dispatch_path, "r") as f:
            dispatch_data = json.load(f)
        dispatches = dispatch_data.get("dispatches", {})

        net_prefix = f"{net_id}_"
        for dispatch_name, dispatch_info in dispatches.items():
            dispatch_id = dispatch_info.get("id", None)
            prefixed_name = f"{net_prefix}{dispatch_name}"

            combo_times = []
            for ci, combo in enumerate(machine_combinations):
                hw = combo_hw[ci]
                topo = topo_tag_for_combination(combo)
                prof = all_profiles.get((hw, topo))

                t_ms = None
                if prof and isinstance(dispatch_id, int) and dispatch_id in prof:
                    t_ms = prof[dispatch_id]["time_ms"]

                if t_ms is not None:
                    combo_times.append(float(t_ms))
                else:
                    p_ms_synth = float(rng.uniform(2.0, 10.0))
                    core_type = machine_type_prefix(combo[0])
                    if core_type == "CPU_P":
                        combo_times.append(p_ms_synth)
                    else:
                        combo_times.append(p_ms_synth * p_core_speedup)

            processing_times[prefixed_name] = combo_times

        for (hw, topo), prof in all_profiles.items():
            if hw == cpu_p_profile_hw:
                combined_profiled_p.update(prof)
            else:
                combined_profiled_e.update(prof)

    return processing_times, combined_profiled_p, combined_profiled_e
