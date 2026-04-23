"""
CLI entry point for the core-count flow pipeline.

Mirrors scheduler_lite.py but routes through core_count_flow.schedule_with_core_count_flow
(convex_packing -> greedy count pick -> fixed-count MILP), leaving scheduler.py unchanged.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload
from workload_factory import (
    build_machine_combinations,
    create_workload_from_network_hierarchy,
    machine_type_prefix,
)
from core_count_flow import schedule_with_core_count_flow
from profile_loader import load_profiled_processing_times
from postprocessing import output_scheduled_json, trim_periodic_after_nonperiodic_makespan
from scheduler_lite import load_networks_config
import plot

CPU_P = "CPU_P"
CPU_E = "CPU_E"


def run(
    networks_json_path: str,
    n_splits: int = 3,
    solver_verbosity: int = 0,
    time_limit: float | None = None,
    random_seed: int | None = None,
    p_core_speedup: float | None = None,
    use_profiled: bool | None = None,
    prune_periodic: bool | None = None,
    restrict_makespan_to_nonperiodic: bool | None = None,
    save_data: bool = False,
) -> Tuple[Workload, np.ndarray, np.ndarray]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_base_path = os.path.abspath(os.path.join(script_dir, ".."))
    if not os.path.isabs(networks_json_path):
        networks_json_path = os.path.join(repo_base_path, networks_json_path)

    print("=" * 60)
    print("Core-count flow: loading network hierarchy from JSON")
    print("=" * 60)
    print(f"Config: {networks_json_path}")

    networks_data, cfg = load_networks_config(networks_json_path)
    if p_core_speedup is not None:
        cfg["p_core_speedup"] = p_core_speedup
    if random_seed is not None:
        cfg["random_seed"] = random_seed
    if solver_verbosity is not None:
        cfg["solver_verbosity"] = solver_verbosity
    if time_limit is not None:
        cfg["time_limit"] = time_limit
    if use_profiled is not None:
        cfg["use_profiled"] = use_profiled
    if prune_periodic is not None:
        cfg["prune_periodic"] = prune_periodic
    if restrict_makespan_to_nonperiodic is not None:
        cfg["restrict_makespan_to_nonperiodic"] = restrict_makespan_to_nonperiodic

    machine_core_counts = cfg["machine_core_counts"]
    machines, machine_combinations = build_machine_combinations(machine_core_counts)
    n_cores = len(machines)
    transfer_times = np.zeros((n_cores, n_cores))

    combo_hw = []
    for combo in machine_combinations:
        core_type = machine_type_prefix(combo[0])
        if core_type == CPU_P:
            combo_hw.append(cfg["cpu_p_profile_hw"])
        else:
            combo_hw.append(cfg["cpu_e_profile_hw"])

    rng = np.random.default_rng(cfg["random_seed"])

    processing_times = None
    combined_profiled_p = None
    combined_profiled_e = None
    if cfg["use_profiled"]:
        print("Loading profiled processing times...")
        processing_times, combined_profiled_p, combined_profiled_e = load_profiled_processing_times(
            networks=networks_data.get("networks", {}),
            repo_base_path=repo_base_path,
            machine_combinations=machine_combinations,
            combo_hw=combo_hw,
            profile_target=cfg["profile_target"],
            cpu_p_profile_hw=cfg["cpu_p_profile_hw"],
            cpu_e_profile_hw=cfg["cpu_e_profile_hw"],
            rng=rng,
            p_core_speedup=cfg["p_core_speedup"],
        )

    workload = create_workload_from_network_hierarchy(
        networks_data=networks_data,
        repo_base_path=repo_base_path,
        machines=machines,
        transfer_times=transfer_times,
        p_core_speedup=cfg["p_core_speedup"],
        random_seed=cfg["random_seed"],
        processing_times=processing_times,
        machine_combinations=machine_combinations,
    )
    print(f"Workload: {len(workload.operations)} operations across "
          f"{len(workload.machines)} cores, {len(workload.get_machine_combinations())} "
          f"cumulative combinations")

    expanded_workload, t, alpha = schedule_with_core_count_flow(
        workload=workload,
        machine_core_counts=machine_core_counts,
        n_splits=n_splits,
        solver_verbosity=cfg["solver_verbosity"],
        time_limit=cfg["time_limit"],
        restrict_makespan_to_nonperiodic=cfg["restrict_makespan_to_nonperiodic"],
        prune_cross_period_constraints=cfg["prune_periodic"],
    )

    if cfg["prune_periodic"]:
        expanded_workload, t, alpha = trim_periodic_after_nonperiodic_makespan(
            expanded_workload, t, alpha,
        )

    combinations = expanded_workload.get_machine_combinations()
    completion_times = []
    for i, op in enumerate(expanded_workload.operations):
        k = int(np.argmax(alpha[i]))
        d = op.get_duration_for_combination(k, combinations, expanded_workload.machines)
        if op.min_start_t is None and op.max_end_t is None:
            completion_times.append(float(t[i]) + float(d))
    makespan = max(completion_times) if completion_times else 0.0
    print(f"Makespan (non-periodic): {makespan:.2f} ms")

    os.makedirs("plots", exist_ok=True)
    num_jobs = sum(1 for op in expanded_workload.operations if not op.predecessors)
    network_names = [
        expanded_workload.job_names[i] if i < len(expanded_workload.job_names) else f"Job {i}"
        for i in sorted(set(op.job_id for op in expanded_workload.operations))
    ]
    title = " + ".join(name.capitalize() for name in network_names)
    plot.plot_optimization_schedule(
        expanded_workload.get_durations(), t, alpha,
        num_jobs, len(expanded_workload.machines),
        expanded_workload.machines, expanded_workload.get_transfer_times(),
        save_path="plots/core_count_flow_schedule.png",
        plot_title=f"{title} Core-Count-Flow Schedule ({n_cores} cores)",
        workload=expanded_workload,
    )
    print("Plot -> plots/core_count_flow_schedule.png")

    os.makedirs("schedules", exist_ok=True)
    base = os.path.splitext(os.path.basename(networks_json_path))[0]
    out = f"schedules/core_count_flow_{base}.json"
    if cfg["use_profiled"]:
        out = out.replace(".json", "_profiled.json")
    output_scheduled_json(
        combined_workload=expanded_workload, t=t, alpha=alpha,
        output_path=out,
        profiled_times_p=combined_profiled_p,
        profiled_times_e=combined_profiled_e,
    )
    print(f"Schedule -> {out}")

    if save_data:
        unsorted_dir = os.path.join("schedules", "unsorted")
        os.makedirs(unsorted_dir, exist_ok=True)
        stamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
        dest = os.path.join(unsorted_dir, f"{os.path.splitext(os.path.basename(out))[0]}_{stamp}.json")
        shutil.copy2(out, dest)
        print(f"Copied -> {dest}")

    return expanded_workload, t, alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Core-count flow scheduler pipeline")
    parser.add_argument("--networks-json", type=str,
                        default="data/toplevel/networks_periodic_profile.json")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="Number of window splits for convex_packing (default: 3)")
    parser.add_argument("--solver-verbosity", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--time-limit", type=float, default=20)
    parser.add_argument("--profiled", dest="profiled", action="store_true", default=True)
    parser.add_argument("--no-profiled", dest="profiled", action="store_false")
    parser.add_argument("--prune-periods", dest="prune_periodic", action="store_true", default=None)
    parser.add_argument("--no-prune-periods", dest="prune_periodic", action="store_false")
    parser.add_argument("--restrict-makespan-to-nonperiodic",
                        dest="restrict_makespan_to_nonperiodic", action="store_true", default=None)
    parser.add_argument("--include-periodic-in-makespan",
                        dest="restrict_makespan_to_nonperiodic", action="store_false")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--p-core-speedup", type=float, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    args = parser.parse_args()

    seed = None if (args.random_seed is not None and args.random_seed < 0) else args.random_seed
    run(
        networks_json_path=args.networks_json,
        n_splits=args.n_splits,
        solver_verbosity=args.solver_verbosity,
        time_limit=args.time_limit,
        random_seed=seed,
        p_core_speedup=args.p_core_speedup,
        use_profiled=args.profiled,
        prune_periodic=args.prune_periodic,
        restrict_makespan_to_nonperiodic=args.restrict_makespan_to_nonperiodic,
        save_data=args.save_data,
    )