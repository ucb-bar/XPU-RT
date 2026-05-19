#!/usr/bin/env python3
"""Merlin → XPU-RT adapter.

Bridges merlin's per-dispatch profile output (produced by
`third_party/merlin/tools/board_roundtrip.py`) into XPU-RT's scheduler. Replaces three
merlin-side prototypes (xpu_rt_schedule.py, multi_model_workload.py,
profiled_to_xpu_rt.py) with a single tool that lives next to the rest of
XPU-RT's scheduling driver scripts.

Subcommands:

  to-csv <merlin_output_dir>
      Emits XPU-RT's `dispatch_id, module_name, mean_time, mean_unit` CSV
      from a merlin profiled_manifest.json.

  schedule <merlin_output_dir>
      Builds a single-model workload from one merlin output dir, runs the
      scheduler (`--solver greedy` by default; `--solver mosek` calls into
      XPU-RT's MILP), emits a merlin-format `schedule.json` ready to feed
      back through `./merlin compile --with-schedule`.

  multi <spec> [<spec> ...]
      Composes a multi-model workload from several merlin output dirs (each
      already profiled), schedules the combined graph, and emits both a
      merlin schedule.json and a Gantt PNG. Each spec is
      `<merlin_output_dir>:<count>[:<key>]`. Multiple granularities of the
      same model can be passed as separate specs with different keys.

This is the only file that needs to know about both:
  * merlin's manifest schema (per-dispatch shapes, deps, profiles by
    machine name)
  * XPU-RT's scheduler API (workload_factory, scheduler.schedule, plot)

Run from the XPU-RT repo root:
    python scripts/merlin_adapter.py schedule <dir> --solver greedy
    python scripts/merlin_adapter.py multi a:3:dronet b:1:mn2 --output out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np

# Make XPU-RT's scheduling modules importable when run directly.
_XPU_RT_PYDIR = pathlib.Path(__file__).resolve().parent.parent / "xpu-rt"
if str(_XPU_RT_PYDIR) not in sys.path:
    sys.path.insert(0, str(_XPU_RT_PYDIR))


def _load_manifest(merlin_dir: pathlib.Path) -> dict:
    p = merlin_dir / "breakdowns" / "profiled_manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p} — run third_party/merlin/tools/board_roundtrip.py first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# to-csv: tiny, pre-existing format (XPU-RT's profile_loader expects it).
# ---------------------------------------------------------------------------
def cmd_to_csv(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.merlin_dir)
    out_path = (args.merlin_dir / "breakdowns" /
                (args.csv_name or "profiled_times.csv"))
    rows = []
    for name, e in manifest["dispatches"].items():
        if e.get("mean_time_us") is None:
            continue
        rows.append({
            "dispatch_id": e["id"],
            "module_name": name,
            "mean_time": f'{e["mean_time_us"]:.3f}',
            "mean_unit": "us",
            "op_summary": e.get("op_summary", ""),
        })
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dispatch_id", "module_name", "mean_time",
                           "mean_unit", "op_summary"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {out_path}  ({len(rows)} rows)")
    return 0


# ---------------------------------------------------------------------------
# Workload builder shared by `schedule` and `multi`. Returns a populated
# XPU-RT Workload object plus the namespaced dispatch dict (for emitting
# the merlin schedule.json afterwards).
# ---------------------------------------------------------------------------
def _skipped_set(workload) -> set[str]:
    """Read back the skipped-op indices stashed by xpu-rt/scheduler.py
    onto the workload after MOSEK ran. Returns the set of operation_name
    strings that the solver chose to drop (set s_i=1)."""
    indices = getattr(workload, "skipped_op_indices", []) or []
    out: set[str] = set()
    for i in indices:
        if 0 <= i < len(workload.operations):
            name = workload.operations[i].operation_name
            if name:
                out.add(name)
    return out


def _apply_robotics_annotations(
    dispatches: dict,
    deadlines: list[str],
    skips: list[str],
) -> None:
    """Stamp deadline_us / skip_allowed onto dispatch entries based on
    --deadline-ms / --skip-allowed CLI flags. Each flag value is a
    `<job-prefix>=<value>` (deadline) or `<job-prefix>` (skip) that
    applies to every dispatch whose key starts with `<job-prefix>_` (the
    composer's replica naming convention)."""
    deadline_map: dict[str, float] = {}
    for s in deadlines:
        if "=" not in s:
            continue
        prefix, val = s.split("=", 1)
        try:
            deadline_map[prefix.strip()] = float(val) * 1000.0  # ms -> us
        except ValueError:
            pass
    skip_set = {s.strip() for s in skips if s.strip()}
    # Special prefix "ALL" matches every dispatch (handy for single-model
    # schedules where the dispatch keys don't carry a job prefix).
    for name, e in dispatches.items():
        match_prefix = None
        for prefix in deadline_map:
            if prefix == "ALL" or name.startswith(prefix):
                if match_prefix is None or len(prefix) > len(match_prefix or ""):
                    match_prefix = prefix
        if match_prefix is not None:
            e["deadline_us"] = deadline_map[match_prefix]
        for prefix in skip_set:
            if prefix == "ALL" or name.startswith(prefix):
                e["skip_allowed"] = True
                break


def _build_workload(
    dispatches: dict,
    machines: list[str],
    transfer_us: float,
):
    from workload_factory import create_workload_from_dependencies

    processing_times: dict[str, list[float]] = {}
    dropped: list[str] = []
    for name, e in dispatches.items():
        profiles = e.get("profiles", {}) or {}
        row = []
        for m in machines:
            t = profiles.get(m, {}).get("mean_time_us")
            row.append(t if t is not None else float("inf"))
        if all(t == float("inf") for t in row):
            dropped.append(name)
            continue
        worst = max(t for t in row if t != float("inf"))
        processing_times[name] = [worst * 10 if t == float("inf") else t
                                  for t in row]
        # Pass through cost_by_pred (when populated by derive_pred_aware_costs)
        # so the workload_factory can build the per-(predecessor, current)
        # cost map that the MOSEK MILP linearises.
    if dropped:
        print(f"WARNING: dropping {len(dropped)} dispatches without profile "
              f"data (benchmark failures)", file=sys.stderr)
        dropped_set = set(dropped)
        for name in dropped:
            del dispatches[name]
        for e in dispatches.values():
            e["dependencies"] = [d for d in e.get("dependencies", [])
                                 if d not in dropped_set]

    n = len(machines)
    transfer = np.full((n, n), transfer_us, dtype=float)
    np.fill_diagonal(transfer, 0.0)
    workload = create_workload_from_dependencies(
        {"dispatches": dispatches}, processing_times, machines, transfer,
    )
    return workload, processing_times


def _run_scheduler(workload, machines, solver: str, time_limit_s: float,
                   transfer_us_matrix=None,
                   critical_path_bias_us: float = 0.0,
                   target_diversity_weight: float = 0.0):
    """Run either the in-tree greedy scheduler (no MOSEK needed) or
    XPU-RT's MILP. Returns (start_times, machine_assignments).

    `transfer_us_matrix`: optional NxN ndarray of inter-machine transfer
        costs. When provided AND a predecessor ran on a different machine,
        the greedy picker adds `transfer_us_matrix[i_pred, i_self]` to the
        effective ready time. This is what penalises cross-cluster bounces
        the previous default (10us flat) was too small to discourage.
    `critical_path_bias_us`: for greedy. If a predecessor's finish time
        equals the current ready time (i.e. that predecessor IS the
        critical path into this op), the picker adds this bias to the
        finish time of any candidate machine != the predecessor's
        machine. Defaults to 0 (off). Setting it to roughly the size of
        a small dispatch nudges the picker to keep the critical chain on
        one cluster, eliminating most stall-on-the-other-cluster gaps.
    """
    if solver == "mosek":
        from scheduler import schedule
        t, alpha, _fused, _fmap = schedule(
            workload, time_limit=time_limit_s, verbose=False,
            target_diversity_weight=target_diversity_weight,
        )
        if t is None or alpha is None:
            raise RuntimeError("MOSEK scheduler returned no solution")
        starts = {op.operation_name: float(t[i])
                  for i, op in enumerate(workload.operations)}
        assigns = {op.operation_name: machines[int(np.argmax(alpha[i]))]
                   for i, op in enumerate(workload.operations)}
        return starts, assigns

    # Greedy. List scheduler with two improvements over the naive
    # "earliest-finish" picker:
    #   1. Cross-cluster transfer cost: if predecessor ran on a different
    #      machine, the data has to land on this op's machine before it
    #      can start. transfer_us_matrix[i_pred, i_self] is added to the
    #      ready time.
    #   2. Critical-path bias: when a predecessor would directly bottleneck
    #      this op's start (i.e. its finish equals the current ready
    #      time), penalise machines != that predecessor's machine by
    #      `critical_path_bias_us`. This is what eliminates the
    #      "CPU_P stalls waiting for CPU_E to finish dispatch_3" gaps.
    deps = {op.operation_name: list(op.predecessors or [])
            for op in workload.operations}
    by_name = {op.operation_name: op for op in workload.operations}
    pending = {n: [p.operation_name for p in d] for n, d in deps.items()}
    order: list[str] = []
    while pending:
        ready = sorted(n for n, d in pending.items() if not d)
        if not ready:
            raise RuntimeError("dependency cycle")
        for n in ready:
            order.append(n)
            del pending[n]
            for rem in pending.values():
                if n in rem:
                    rem.remove(n)

    n_machines = len(machines)
    machine_idx = {m: i for i, m in enumerate(machines)}

    machine_free = {m: 0.0 for m in machines}
    starts: dict[str, float] = {}
    finishes: dict[str, tuple[float, str]] = {}
    assigns: dict[str, str] = {}
    for name in order:
        op = by_name[name]
        best = None
        for mi, m in enumerate(machines):
            # Compute ready time on machine `m` taking transfer costs into
            # account.
            ready_t = 0.0
            critical_pred_mi: int | None = None
            for pred in op.predecessors or []:
                f, fm = finishes[pred.operation_name]
                t_us = 0.0
                if transfer_us_matrix is not None and fm != m:
                    fmi = machine_idx[fm]
                    t_us = float(transfer_us_matrix[fmi, mi])
                pred_arrival = f + t_us
                if pred_arrival > ready_t:
                    ready_t = pred_arrival
                    critical_pred_mi = machine_idx[fm]

            start = max(ready_t, machine_free[m])
            finish = start + op.processing_times[mi]

            # Critical-path bias: if this op is gated by a single
            # predecessor on a different machine, prefer that machine.
            if (critical_path_bias_us > 0.0
                    and critical_pred_mi is not None
                    and critical_pred_mi != mi
                    and start == ready_t):
                finish += critical_path_bias_us

            if best is None or finish < best[0]:
                best = (finish, m, start, mi)
        finish, m, start, mi = best
        # Strip the bias before recording machine_free / actual finish.
        actual_finish = start + op.processing_times[mi]
        machine_free[m] = actual_finish
        starts[name] = start
        assigns[name] = m
        finishes[name] = (actual_finish, m)
    return starts, assigns


def _solver_state_arrays(workload, machines: list[str],
                         starts: dict[str, float],
                         assigns: dict[str, str]):
    """Rebuild (t, alpha) numpy arrays from the dict form that
    `_run_scheduler` emits, so derive_dispatch_hints (which works in
    workload-index space) can consume the result of either solver path.

    Assumes singleton machine combinations (one combination per machine),
    which is what `_build_workload` constructs by default.
    """
    n_ops = len(workload.operations)
    n_k = len(workload.machine_combinations)
    t_np = np.zeros(n_ops, dtype=float)
    alpha_np = np.zeros((n_ops, n_k), dtype=float)
    # Build machine -> combination index, only well-defined when each
    # combination is a single machine.
    machine_to_k: dict[str, int] = {}
    for k, combo in enumerate(workload.machine_combinations):
        if len(combo) == 1:
            machine_to_k[combo[0]] = k
    for i, op in enumerate(workload.operations):
        name = op.operation_name
        t_np[i] = float(starts.get(name, 0.0))
        m = assigns.get(name)
        if m is not None and m in machine_to_k:
            alpha_np[i, machine_to_k[m]] = 1.0
    return t_np, alpha_np


def _emit_feedback(args: argparse.Namespace,
                   workload,
                   machines: list[str],
                   starts: dict[str, float],
                   assigns: dict[str, str],
                   schedule_path: pathlib.Path) -> None:
    """Write xpurt_feedback.json next to schedule.json. No-op if --emit-feedback
    was not passed, so existing flows are byte-identical."""
    if not getattr(args, "emit_feedback", False):
        return
    from feedback import derive_dispatch_hints, write_feedback_json

    t_np, alpha_np = _solver_state_arrays(workload, machines, starts, assigns)
    payload = derive_dispatch_hints(
        workload, t_np, alpha_np,
        run_id=getattr(args, "feedback_run_id", None),
        source_schedule=schedule_path.name,
    )
    fb_path = schedule_path.with_name("xpurt_feedback.json")
    write_feedback_json(payload, fb_path)
    n_hints = len(payload.get("dispatches", {}))
    print(f"feedback -> {fb_path}  ({n_hints} dispatches with hints)")


def _write_merlin_schedule(
    out_path: pathlib.Path,
    dispatches: dict,
    machines: list[str],
    starts: dict[str, float],
    assigns: dict[str, str],
    device_map: dict[str, str],
    proc_us: dict[str, list[float]] | None = None,
    skipped_names: set[str] | None = None,
) -> None:
    """Emit a schedule.json that satisfies BOTH consumers:

      * merlin's `--iree-merlin-schedule-spec` (compile-time apply-affinity
        pass) — keys off `id` + `hardware_target`.
      * the on-board C dispatch runner at samples/common/xpu-rt/
        scheduler_runner.cc — keys off `start_time` (ms), `duration` (ms),
        `module_name` (IREE export name within the per-dispatch VMFB).

    We emit both the modern `start_time_us` and the legacy `start_time`
    (ms) fields plus per-dispatch `duration` (ms, derived from the
    machine the scheduler picked). `module_name` is taken from the
    breakdown manifest entry produced by tools/breakdown_vmfb.py.
    """
    out_dispatches: dict[str, dict] = {}
    for name, e in dispatches.items():
        target = assigns.get(name, machines[0])
        start_us = starts.get(name, 0.0)
        duration_us = 0.0
        if proc_us and name in proc_us:
            mi = machines.index(target) if target in machines else 0
            row = proc_us[name]
            if mi < len(row):
                duration_us = row[mi]
        out_dispatches[name] = {
            "id": e["id"],
            "subid": e.get("subid"),
            "ordinal": e.get("ordinal", 1),
            "total": e.get("total", 1),
            "hardware_target": target,
            # Microseconds for merlin compile, milliseconds for the C
            # runner. Both keys are present so neither consumer cares.
            "start_time_us": start_us,
            "start_time": start_us / 1000.0,
            "duration": duration_us / 1000.0,
            "dependencies": e.get("dependencies", []),
            "op_summary": e.get("op_summary", ""),
            "job_name": e.get("job_name", ""),
            # Required by the C runner to look up the VMFB export. Derived
            # from breakdown_vmfb.py's parse of the per-dispatch benchmark
            # MLIR's hal.executable.export name.
            "module_name": e.get("module_name", ""),
            # Explicit relative VMFB path so the C runner skips its
            # SpacemiTX60-style auto-resolution (which assumes a specific
            # `gen/vmfb/<model>/spacemit_x60/...` layout). Relative paths
            # are joined onto --vmfb_dir at runtime, absolute paths are
            # used as-is.
            "vmfb_path": f"{name}.vmfb",
            # Robotics-deadline support (PR5 of the rosy-sundae plan):
            # carried through so the on-board scheduler runner can stop
            # the dominant job at deadline and skip background instances.
            **({"deadline_us": e["deadline_us"]} if "deadline_us" in e else {}),
            **({"skip_allowed": True} if e.get("skip_allowed") else {}),
            **({"skipped": True} if (skipped_names and name in skipped_names) else {}),
        }
    # `machines` is the canonical source of truth for which devices the
    # model can run on; the merlin compiler auto-derives the deviceMap
    # (machines[i] -> @device_<letter>) from this list. `device_map` stays
    # for one release as an explicit override + back-compat for older
    # consumers, but is strictly optional.
    payload = {
        "schema_version": 1,
        "machines": list(machines),
        "device_map": device_map,
        "source": "XPU-RT/scripts/merlin_adapter.py",
        "dispatches": out_dispatches,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# schedule: single-model.
# ---------------------------------------------------------------------------
def cmd_schedule(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.merlin_dir)
    dispatches = dict(manifest["dispatches"])

    _apply_robotics_annotations(dispatches, args.deadline_ms, args.skip_allowed)
    workload, proc_us = _build_workload(dispatches, args.machines,
                                        args.transfer_time_us)
    print(f"workload: {len(workload.operations)} ops, "
          f"{len(args.machines)} machines ({args.solver})")
    n = len(args.machines)
    transfer = np.full((n, n), args.transfer_time_us, dtype=float)
    np.fill_diagonal(transfer, 0.0)
    starts, assigns = _run_scheduler(
        workload, args.machines, args.solver, args.time_limit_s,
        transfer_us_matrix=transfer,
        critical_path_bias_us=args.critical_path_bias_us,
        target_diversity_weight=getattr(args, "target_diversity_weight", 0.0),
    )

    skipped_names = _skipped_set(workload)
    device_map = dict(p.split(":", 1) for p in args.device_map)
    out_path = args.merlin_dir / "breakdowns" / "schedule.json"
    _write_merlin_schedule(out_path, dispatches, args.machines, starts,
                           assigns, device_map, proc_us=proc_us,
                           skipped_names=skipped_names)
    _emit_feedback(args, workload, args.machines, starts, assigns, out_path)
    summary: dict[str, int] = {}
    for m in assigns.values():
        summary[m] = summary.get(m, 0) + 1
    print(f"schedule -> {out_path}")
    for m, n in summary.items():
        print(f"  {m}: {n} dispatches")
    return 0


# ---------------------------------------------------------------------------
# multi: multi-model composition.
# ---------------------------------------------------------------------------
def _compose(specs: list[tuple[pathlib.Path, int, str]]) -> dict:
    combined: dict[str, dict] = {}
    for merlin_dir, count, key in specs:
        manifest = _load_manifest(merlin_dir)
        per_dispatch = manifest["dispatches"]
        for inst in range(count):
            prefix = f"{key}{inst}_"
            for name, e in per_dispatch.items():
                new = dict(e)
                new["dependencies"] = [
                    f"{prefix}{d}" for d in e.get("dependencies", [])
                ]
                new["job_name"] = key
                new["instance"] = inst
                combined[f"{prefix}{name}"] = new
    return combined


def _plot_schedule(
    dispatches: dict,
    machines: list[str],
    starts: dict[str, float],
    assigns: dict[str, str],
    proc_us: dict[str, list[float]],
    out_path: pathlib.Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plot", file=sys.stderr)
        return
    cmap = plt.get_cmap("tab20")
    job_color: dict[str, object] = {}
    for e in dispatches.values():
        j = e.get("job_name") or "unknown"
        if j not in job_color:
            job_color[j] = cmap(len(job_color) % cmap.N)
    fig, ax = plt.subplots(figsize=(14, 1.2 * len(machines) + 0.5))
    machine_to_y = {m: i for i, m in enumerate(machines)}
    for name, e in dispatches.items():
        m = assigns.get(name)
        if m not in machine_to_y:
            continue
        mi = machines.index(m)
        dur_ms = proc_us[name][mi] / 1000.0
        start_ms = starts.get(name, 0.0) / 1000.0
        ax.broken_barh(
            [(start_ms, dur_ms)], (machine_to_y[m] - 0.4, 0.8),
            facecolors=job_color[e.get("job_name") or "unknown"],
            edgecolors="black", linewidth=0.4,
        )
    ax.set_yticks(list(machine_to_y.values()))
    ax.set_yticklabels(machines)
    ax.set_xlabel("time (ms)")
    ax.set_title(f"multi-model schedule ({len(dispatches)} dispatches)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    handles = [mpatches.Patch(color=c, label=j) for j, c in job_color.items()]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def cmd_multi(args: argparse.Namespace) -> int:
    specs: list[tuple[pathlib.Path, int, str]] = []
    for s in args.workload:
        parts = s.split(":")
        if not 2 <= len(parts) <= 3:
            raise ValueError(f"bad workload spec '{s}'")
        d = pathlib.Path(parts[0]).resolve()
        count = int(parts[1])
        key = parts[2] if len(parts) == 3 else d.name
        specs.append((d, count, key))

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    breakdowns = out_dir / "breakdowns"
    breakdowns.mkdir(exist_ok=True)

    dispatches = _compose(specs)
    print(f"composed: {sum(c for _, c, _ in specs)} instances, "
          f"{len(dispatches)} dispatches across {len(specs)} models")
    _apply_robotics_annotations(dispatches, args.deadline_ms, args.skip_allowed)
    (breakdowns / "combined_manifest.json").write_text(
        json.dumps({"schema_version": 1, "dispatches": dispatches}, indent=2)
        + "\n")

    workload, proc_us = _build_workload(dispatches, args.machines,
                                        args.transfer_time_us)
    n = len(args.machines)
    transfer = np.full((n, n), args.transfer_time_us, dtype=float)
    np.fill_diagonal(transfer, 0.0)
    starts, assigns = _run_scheduler(
        workload, args.machines, args.solver, args.time_limit_s,
        transfer_us_matrix=transfer,
        critical_path_bias_us=args.critical_path_bias_us,
        target_diversity_weight=getattr(args, "target_diversity_weight", 0.0),
    )

    skipped_names = _skipped_set(workload)
    if skipped_names:
        print(f"  skipped {len(skipped_names)} dispatches under deadline "
              f"pressure (e.g. {sorted(skipped_names)[:3]}…)")
    device_map = dict(p.split(":", 1) for p in args.device_map)
    schedule_path = breakdowns / "combined_schedule.json"
    _write_merlin_schedule(schedule_path, dispatches, args.machines,
                           starts, assigns, device_map, proc_us=proc_us,
                           skipped_names=skipped_names)
    _emit_feedback(args, workload, args.machines, starts, assigns,
                   schedule_path)

    summary: dict[str, list[float]] = {m: [] for m in args.machines}
    for name, m in assigns.items():
        mi = args.machines.index(m)
        summary[m].append(proc_us[name][mi])
    for m, durs in summary.items():
        if durs:
            print(f"  {m}: {len(durs)} dispatches, sum={sum(durs)/1000:.2f} ms")
    finishes = [starts[name] + proc_us[name][args.machines.index(assigns[name])]
                for name in dispatches if name in proc_us and name in assigns]
    if finishes:
        print(f"  makespan ~= {max(finishes)/1000:.2f} ms")

    if args.plot:
        plot_path = breakdowns / "combined_schedule.png"
        _plot_schedule(dispatches, args.machines, starts, assigns, proc_us,
                       plot_path)
        print(f"plot -> {plot_path}")
    return 0


# ---------------------------------------------------------------------------
# main + argparse plumbing.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--machines", nargs="+", default=["CPU_P", "CPU_E"])
    common.add_argument("--transfer-time-us", type=float, default=0.0)
    common.add_argument("--solver", choices=["greedy", "mosek"],
                        default="greedy")
    common.add_argument("--time-limit-s", type=float, default=30.0)
    common.add_argument("--device-map", nargs="+",
                        default=["CPU_P:@device_a", "CPU_E:@device_b"])
    common.add_argument(
        "--target-diversity-weight", type=float, default=0.0,
        help="MOSEK only: subtract λ × (number of distinct primary machines "
             "used) from the objective. Encourages the solver to spread "
             "dispatches across more devices when makespan-equivalent "
             "schedules exist. λ should be small relative to the typical "
             "makespan in microseconds (e.g. 50-200 for a 5-20ms model). "
             "Default 0.0 (off) preserves makespan-only behavior.")
    common.add_argument(
        "--critical-path-bias-us", type=float, default=0.0,
        help="Greedy bias: when an op is gated by a single predecessor on "
             "machine X, charge candidate machines != X by this many "
             "microseconds. Eliminates 'CPU_P stalls waiting for CPU_E "
             "to finish' gaps. Recommended starting value: a small-conv "
             "duration (e.g. 200-500us for QRB5165 dronet).")
    common.add_argument(
        "--deadline-ms", action="append", default=[],
        help="Robotics-deadline support. Format: <job-prefix>=<deadline-ms>. "
             "All ops whose key starts with <job-prefix>_ get their "
             "deadline_us field set so the MOSEK MILP enforces the bound. "
             "Repeatable. Example: --deadline-ms dronet=33 "
             "--deadline-ms perception=10. Greedy ignores this flag.")
    common.add_argument(
        "--skip-allowed", action="append", default=[],
        help="Robotics-deadline support. Marks every op whose key starts "
             "with <job-prefix>_ as `skip_allowed=True`. The MOSEK MILP "
             "may then drop these ops (set s_i=1) when their job's "
             "deadline can't be met. Repeatable. Example: "
             "--skip-allowed mlp --skip-allowed dcoarse.")
    common.add_argument(
        "--emit-feedback", action="store_true",
        help="After scheduling, derive per-dispatch granularity hints from "
             "the solver state and write `xpurt_feedback.json` next to "
             "schedule.json. Consumed by Merlin's targetgen_mcp "
             "ingest_xpurt_feedback tool. Inert when omitted — Merlin's "
             "standalone path is unchanged.")
    common.add_argument(
        "--feedback-run-id", default=None,
        help="Optional run identifier stamped into xpurt_feedback.json. "
             "Use the same id across iterations to accumulate hints "
             "(MCP merge semantics). Defaults to a UTC timestamp.")

    s_csv = sub.add_parser("to-csv",
                           help="emit XPU-RT's profile CSV from a merlin "
                                "profiled_manifest.json")
    s_csv.add_argument("merlin_dir", type=pathlib.Path)
    s_csv.add_argument("--csv-name", default=None)
    s_csv.set_defaults(func=cmd_to_csv)

    s_sched = sub.add_parser("schedule",
                             help="schedule a single merlin output dir",
                             parents=[common])
    s_sched.add_argument("merlin_dir", type=pathlib.Path)
    s_sched.set_defaults(func=cmd_schedule)

    s_multi = sub.add_parser("multi",
                             help="compose multi-model schedule + Gantt plot",
                             parents=[common])
    s_multi.add_argument("--workload", nargs="+", required=True)
    s_multi.add_argument("--output", required=True, type=pathlib.Path)
    s_multi.add_argument("--plot", action="store_true")
    s_multi.set_defaults(func=cmd_multi)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
