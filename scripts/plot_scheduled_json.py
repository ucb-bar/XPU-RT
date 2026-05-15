#!/usr/bin/env python3
"""
Plot a previously-scheduled JSON file using plot_optimization_schedule,
without running the scheduler or profiling pipeline.

Usage:
    python scripts/plot_scheduled_json.py <scheduled_json_path> [--save <output_png>]
"""

import argparse
import json
import sys
import os

# Add xpu-rt to path so we can import plot, workload, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xpu-rt'))

import numpy as np
from xpu_rt.scheduler.workload import Workload, Operation
from xpu_rt.scheduler import plot


def load_and_plot(json_path: str, save_path: str | None = None):
    with open(json_path) as f:
        data = json.load(f)

    dispatches = data["dispatches"]
    metadata = data["metadata"]
    machines = metadata["machines"]
    machine_combinations = metadata["machine_combinations"]
    num_combos = len(machine_combinations)

    # Build a hardware_target string -> combo index map
    combo_labels = ["+".join(combo) for combo in machine_combinations]
    hw_target_to_combo = {label: idx for idx, label in enumerate(combo_labels)}

    # Collect unique job names in order of first appearance
    job_name_order = []
    for d in dispatches.values():
        jn = d["job_name"]
        if jn not in job_name_order:
            job_name_order.append(jn)
    job_name_to_id = {name: idx for idx, name in enumerate(job_name_order)}

    # Sort dispatches by start_time so operations are in chronological order
    sorted_names = sorted(dispatches.keys(), key=lambda n: dispatches[n]["start_time"])

    # First pass: create Operation stubs (no predecessors yet)
    op_by_name: dict[str, Operation] = {}
    for name in sorted_names:
        d = dispatches[name]
        combo_idx = hw_target_to_combo[d["hardware_target"]]
        # Build processing_times: one entry per combination, assigned combo gets real duration
        processing_times = [1e6] * num_combos
        processing_times[combo_idx] = d["duration"]

        op = Operation(
            processing_times=processing_times,
            predecessors=None,
            operation_id=d.get("id"),
            operation_name=name,
            job_id=job_name_to_id[d["job_name"]],
        )
        op_by_name[name] = op

    # Second pass: wire up predecessors
    for name in sorted_names:
        d = dispatches[name]
        for dep_name in d.get("dependencies", []):
            if dep_name in op_by_name:
                op_by_name[name].add_predecessor(op_by_name[dep_name])

    # Build ordered operation list, start-time array, and alpha matrix
    operations = [op_by_name[name] for name in sorted_names]
    t = np.array([dispatches[name]["start_time"] for name in sorted_names])
    alpha = np.zeros((len(operations), num_combos))
    for i, name in enumerate(sorted_names):
        combo_idx = hw_target_to_combo[dispatches[name]["hardware_target"]]
        alpha[i, combo_idx] = 1.0

    # Transfer times: zeros (no transfer info in the JSON)
    transfer_times = np.zeros((len(machines), len(machines)))

    workload = Workload(
        operations=operations,
        machines=machines,
        transfer_times=transfer_times,
        job_names=job_name_order,
        machine_combinations=machine_combinations,
    )

    num_jobs = len(job_name_order)

    # Default save path
    if save_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        save_path = f"plots/{base}.png"

    title = os.path.splitext(os.path.basename(json_path))[0]

    plot.plot_optimization_schedule(
        durations=workload.get_durations(),
        t=t,
        alpha=alpha,
        num_jobs=num_jobs,
        num_machines=num_combos,
        machines=combo_labels,
        transfer_times=transfer_times,
        save_path=save_path,
        plot_title=title,
        workload=workload,
    )

    print(f"Done. Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a scheduled JSON file")
    parser.add_argument("json_path", help="Path to the scheduled JSON file")
    parser.add_argument("--save", default=None, help="Output PNG path (default: plots/<basename>.png)")
    args = parser.parse_args()
    load_and_plot(args.json_path, args.save)
