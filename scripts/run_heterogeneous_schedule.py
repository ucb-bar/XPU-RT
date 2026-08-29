#!/usr/bin/env python3
"""Run XPU-RT's MOSEK MILP on the YOLOv8 heterogeneous workload.

Inputs:
  - build/het/workload.json + processing_times.json + transfer_times.json
    produced by build_workload_from_graph.py (each row a real on-board
    measurement or an infeasibility marker).

Outputs:
  - build/het/schedule.json — per-dispatch (machine, start_us, finish_us).
  - build/het/schedule_gantt.png — per-machine-lane Gantt (via
    xpu-rt/plot.py:plot_optimization_schedule).
  - build/het/schedule_dag.dot — coloured-by-backend DAG (via
    qnn_scheduler/plot.py:dot_graph). Render with `dot -Tpng`.

The MILP variables / constraints are unchanged; we just feed it
measured numbers + the (2b) hard exclusion constraints for
infeasible (dispatch, backend) cells. Returns non-zero if MOSEK
reports infeasible (which means at least one dispatch has no
measured backend — fix that before calling again).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "xpu-rt"))

from workload_factory import create_workload_from_dependencies  # noqa
from scheduler import schedule  # noqa
import plot as xpu_plot  # noqa


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default=_ROOT / "build" / "het",
                    type=pathlib.Path,
                    help="Directory containing workload.json, "
                         "processing_times.json, transfer_times.json")
    ap.add_argument("--time-limit", default=120.0, type=float,
                    help="MOSEK time limit (s)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    workload_data = json.loads((args.in_dir / "workload.json").read_text())
    processing_times = json.loads((args.in_dir / "processing_times.json").read_text())
    transfer_data = json.loads((args.in_dir / "transfer_times.json").read_text())
    machines = transfer_data["machines"]
    transfer_times = np.array(transfer_data["matrix"], dtype=float)

    workload = create_workload_from_dependencies(
        workload_data, processing_times, machines, transfer_times,
    )
    n_ops = len(workload.operations)
    n_inf = sum(1 for op in workload.operations
                if getattr(op, "infeasible_combinations", None))
    print(f"workload: {n_ops} ops, "
          f"{n_inf} have infeasible-combination constraints, "
          f"{len(machines)} machines: {machines}")

    t0 = time.time()
    t, alpha, _, _ = schedule(
        workload, time_limit=args.time_limit, verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print(f"MOSEK solve: {elapsed:.1f}s")

    if t is None or alpha is None:
        print("ERROR: MOSEK returned no solution. Probable cause: at least "
              "one dispatch has every backend marked infeasible. Profile "
              "those before scheduling. Use:")
        for op in workload.operations:
            infe = getattr(op, "infeasible_combinations", set())
            if len(infe) == len(machines):
                print(f"  unschedulable: {op.operation_name}")
        return 2

    # Emit per-dispatch schedule. Two requirements:
    # 1. Self-describing: include start/finish/machine for diagnostics.
    # 2. Conform to the schema iree-merlin expects from
    #    `--iree-merlin-schedule-spec=...`:
    #       { "schema_version": 1, "machines": [...],
    #         "dispatches": { "<name>": { "id": N,
    #                                     "hardware_target": "<machine>" }} }
    #    The DispatchCreation pass auto-derives @device_a/@device_b/@device_c
    #    from the machines list and stamps stream.affinity per dispatch id.
    sched: dict[str, dict] = {}
    for i, op in enumerate(workload.operations):
        k = int(np.argmax(alpha[i]))
        m = machines[k]
        start = float(t[i])
        finish = start + float(op.processing_times[k])
        sched[op.operation_name] = {
            "id": op.operation_id,
            "hardware_target": m,
            # Diagnostic-only fields; ignored by ScheduleSpec.cpp parser.
            "start_us": start, "finish_us": finish,
            "duration_us": float(op.processing_times[k]),
        }
    makespan = max(s["finish_us"] for s in sched.values())
    out_sched = {
        "schema_version": 1,
        "machines": list(machines),
        "makespan_us": makespan,
        "dispatches": sched,
    }
    (args.in_dir / "schedule.json").write_text(json.dumps(out_sched, indent=2))
    print(f"schedule -> {args.in_dir / 'schedule.json'}")
    print(f"  makespan: {makespan:.0f} us  ({makespan/1000:.2f} ms)")
    counts = {m: 0 for m in machines}
    for s in sched.values():
        counts[s["hardware_target"]] += 1
    print(f"  per-machine assignments: {counts}")

    # Render Gantt (xpu-rt/plot.py).
    durations = workload.get_durations()
    gantt_path = args.in_dir / "schedule_gantt.png"
    try:
        xpu_plot.plot_optimization_schedule(
            durations, t, alpha,
            num_jobs=1, num_machines=len(machines),
            machines=machines, transfer_times=transfer_times,
            save_path=str(gantt_path),
            plot_title=f"YOLOv8 heterogeneous (makespan {makespan/1000:.1f} ms)",
            workload=workload,
        )
        print(f"gantt    -> {gantt_path}")
    except Exception as e:
        print(f"gantt    -> SKIPPED ({e})")

    # Emit a .dot DAG (Graphviz). Coloured by backend.
    dot_path = args.in_dir / "schedule_dag.dot"
    backend_colour = {"HTA": "#fef3c7", "GPU": "#dbeafe", "CPU": "#f3f4f6"}
    lines = ["digraph G {", "  rankdir=LR;",
             "  node [shape=box, style=filled, fontname=\"Helvetica\"];"]
    name_to_op = {op.operation_name: op for op in workload.operations}
    for name, s in sched.items():
        col = backend_colour.get(s["hardware_target"], "#cbd5e1")
        lines.append(
            f'  "{name}" [label="{name}\\n{s["hardware_target"]}  '
            f'{s["duration_us"]/1000:.2f} ms", fillcolor="{col}"];'
        )
    for name, op in name_to_op.items():
        m_self = sched[name]["hardware_target"]
        for pred in op.predecessors or []:
            pn = pred.operation_name
            if pn not in sched:
                continue
            m_pred = sched[pn]["hardware_target"]
            edge_attrs = []
            if m_self != m_pred:
                edge_attrs.append("style=dashed")
                edge_attrs.append("color=red")
                bridge = sched[name]["start_us"] - sched[pn]["finish_us"]
                edge_attrs.append(f'label="{bridge:.0f}us"')
            attr = "[" + ",".join(edge_attrs) + "]" if edge_attrs else ""
            lines.append(f'  "{pn}" -> "{name}" {attr};')
    lines.append("}")
    dot_path.write_text("\n".join(lines))
    print(f"dag      -> {dot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
