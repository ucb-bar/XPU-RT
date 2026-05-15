#!/usr/bin/env python3
"""End-to-end iterative compile-schedule-requant orchestrator (XPU-RT).

Phase H of the heterogeneous-scheduling pipeline. Lives in XPU-RT (the
layer above merlin) because it orchestrates *across* merlin's compile and
runtime building blocks: it invokes merlin's compile_dispatch_matrix to
produce per-target dispatch dumps, merlin's profile_dispatch_matrix to
time them on-board, then drives XPU-RT's MOSEK scheduler with our
target-diversity weight, and (when Phase A2 is enabled) re-applies
schedule-driven re-quantization to the source MLIR before re-iterating.

Each round:
  1. third_party/merlin/tools/compile_dispatch_matrix.py → per-target dispatch dumps
     + matrix.json (canonical dispatch list)
  2. third_party/merlin/tools/profile_dispatch_matrix.py → profiled_manifest.json
     (on-board mean_us per (dispatch, target))
  3. build workload + processing_times + transfer_times JSON, with
     per-edge transfer costs derived from XPU-RT's qrb5165_costs.json
     when --use-cost-table is given (Phase D)
  4. MOSEK scheduler (xpu-rt/scheduler.py) with target_diversity_weight
     → schedule.json
  5. apply_placement_requantization (Phase A2; via merlin iree-compile
     plugin flag --merlin-placement-requant-json when enabled)
  6. terminate when placement-set stable across two rounds OR k == max

Usage:
  python3 XPU-RT/scripts/heterogeneous_loop.py \\
      --merlin-root /scratch2/agustin/merlin \\
      --source <model.mlir> \\
      --out-dir /tmp/het_loop_<model> \\
      --targets cpu,qnn_gpu,qnn_hta \\
      --diversity-weight 100 --max-rounds 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys

# This file lives in XPU-RT/scripts/. The merlin repo location is supplied
# via --merlin-root or $MERLIN_ROOT (default: /scratch2/agustin/merlin).
XPURT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MERLIN_ROOT = pathlib.Path(
    os.environ.get("MERLIN_ROOT", "/scratch2/agustin/merlin"))


@dataclasses.dataclass
class LoopConfig:
    source: pathlib.Path
    out_dir: pathlib.Path
    targets: list[str]
    machines: list[str]            # MOSEK machine names corresponding to targets
    target_to_machine: dict[str, str]
    machine_to_target: dict[str, str]
    diversity_weight: float
    max_rounds: int
    transfer_us: float
    iterations: int
    warmup: int
    iree_compile: pathlib.Path
    skip_profile: bool
    merlin_root: pathlib.Path
    cost_table: pathlib.Path | None  # qrb5165_costs.json for Phase D edges
    ssh_host: str
    ssh_identity: pathlib.Path | None
    profile_input_mode: str
    capture_dir: pathlib.Path | None
    dispatch_graph_json: pathlib.Path | None


def _run(cmd: list[str], cwd: pathlib.Path | None = None) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          check=False).returncode


def _round_dir(out: pathlib.Path, k: int) -> pathlib.Path:
    d = out / f"round_{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_compile_matrix(cfg: LoopConfig, src: pathlib.Path,
                        round_dir: pathlib.Path) -> pathlib.Path:
    matrix_dir = round_dir / "matrix"
    rc = _run([
        "conda", "run", "-n", "merlin-dev", "uv", "run", "python",
        str(cfg.merlin_root / "tools" / "compile_dispatch_matrix.py"),
        "--source", str(src),
        "--targets", ",".join(cfg.targets),
        "--out-dir", str(matrix_dir),
        "--clean",
        "--iree-compile", str(cfg.iree_compile),
        *(["--dispatch-graph-json", str(cfg.dispatch_graph_json)]
          if cfg.dispatch_graph_json else []),
    ])
    if rc != 0:
        raise RuntimeError(f"compile_dispatch_matrix failed (rc={rc})")
    return matrix_dir / "matrix.json"


def step_profile(cfg: LoopConfig, matrix_path: pathlib.Path,
                 round_dir: pathlib.Path) -> pathlib.Path:
    out = round_dir / "profiled_manifest.json"
    if cfg.skip_profile:
        # synth: copy matrix as profile with mean_us=1000 placeholder
        m = json.loads(matrix_path.read_text())
        for d in m["dispatches"].values():
            for t in cfg.targets:
                if d.get(t, {}).get("feasible"):
                    d[t]["profile"] = {"mean_us": 1000.0, "setup_us": 0.0,
                                        "median_us": 1000.0, "p99_us": 1000.0}
        out.write_text(json.dumps(m, indent=2))
        print(f"[skip-profile] synthetic profile written: {out}")
        return out
    cmd = [
        "conda", "run", "-n", "merlin-dev", "uv", "run", "python",
        str(cfg.merlin_root / "tools" / "profile_dispatch_matrix.py"),
        "--matrix", str(matrix_path),
        "--out", str(out),
        "--targets", ",".join(cfg.targets),
        "--iterations", str(cfg.iterations),
        "--warmup", str(cfg.warmup),
        "--ssh-host", cfg.ssh_host,
        "--input-mode", cfg.profile_input_mode,
    ]
    if cfg.ssh_identity:
        cmd.extend(["--ssh-identity", str(cfg.ssh_identity)])
    if cfg.capture_dir:
        cmd.extend(["--capture-dir", str(cfg.capture_dir)])
    rc = _run(cmd)
    if rc != 0:
        print(f"warning: profile_dispatch_matrix returned rc={rc}; "
              f"falling back to compile-matrix-only feasibility")
    return out


def _load_cost_table(cost_table_path: pathlib.Path) -> dict | None:
    """Load XPU-RT's qrb5165_costs.json (linear-fit transfer model).

    Returns the parsed JSON or None if missing. Used by Phase D to derive
    per-machine transfer costs from `bytes_per_us_mean` + `fixed_overhead_us`
    coefficients fitted on-board by profile_transfers_on_board.py.
    """
    if cost_table_path is None or not cost_table_path.is_file():
        return None
    try:
        return json.loads(cost_table_path.read_text())
    except json.JSONDecodeError:
        return None


def _machine_pair_transfer_us(cost_table: dict, src_m: str, dst_m: str,
                              bytes_: int, dtype: str = "uint8") -> float | None:
    """Look up linear-fit transfer cost for a (src, dst, bytes, dtype) tuple.

    The on-board profiler emits keys in two flavors:
      - "<SRC>__<DST>" (same-device memcpy, e.g. "CPU__CPU")
      - "<SRC>->...::<dtype>" (legacy variant; rare)

    For cross-device pairs that the profiler doesn't directly cover, we
    fall back to the worst same-device memcpy in the table — a deliberate
    overestimate that gives the scheduler a defensible upper bound on
    inter-cluster bridge cost. Returns None if the table has no memcpy
    entries at all.
    """
    if src_m == dst_m:
        return 0.0
    memcpy = (cost_table or {}).get("memcpy", {})
    if not memcpy:
        return None

    candidates: list[dict] = []
    # Direct cross-device key.
    direct = memcpy.get(f"{src_m}__{dst_m}")
    if direct:
        candidates.append(direct)
    else:
        for k, v in memcpy.items():
            if k.startswith(f"{src_m}->{dst_m}::"):
                candidates.append(v)
                break

    if not candidates:
        # Cross-device entry not directly profiled. Use the slowest
        # same-device entry in the table as a conservative bound.
        for v in memcpy.values():
            candidates.append(v)

    if not candidates:
        return None
    # Pick the most pessimistic (lowest bytes_per_us_mean / highest cost).
    def _est(row: dict) -> float:
        bpus = float(row.get("bytes_per_us_mean", 1.0))
        fixed = float(row.get("fixed_overhead_us", 0.0))
        if bpus <= 0:
            return fixed if fixed > 0 else float("inf")
        return fixed + bytes_ / bpus
    return max(_est(c) for c in candidates)


def build_workload(cfg: LoopConfig, profile_path: pathlib.Path,
                   round_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Convert profiled_manifest.json to MOSEK workload + processing_times + transfer_times.

    Phase D integration: when cfg.cost_table is set, we load XPU-RT's
    qrb5165_costs.json (produced by profile_transfers_on_board.py's linear
    fit) and compute per-(src,dst) transfer costs at the *largest binding
    byte size of any feasible cell* — a deliberately conservative volume
    that overestimates fast cases but never underestimates large tensors.
    When `profiled_manifest.json` carries a top-level `dispatch_graph`, its
    dependencies drive the workload DAG and its exported edge-volume metadata
    overrides the older conservative binding-size heuristic.
    """
    profile = json.loads(profile_path.read_text())
    cost_table = _load_cost_table(cfg.cost_table) if cfg.cost_table else None
    INFEAS = 1.0e9

    workload = {"dispatches": {}}
    proc_times: dict[str, list[float]] = {}
    dispatch_graph = profile.get("dispatch_graph", {})

    # Track the largest edge payload we see — this sets the transfer-matrix
    # volume fallback when we only have a single global matrix.
    max_bytes_seen = 0

    # Setup-time bookkeeping. Each target has a one-time setup cost
    # captured as `setup_us` in the profile (device init + module load +
    # graph finalize). For an N-iteration steady-state schedule the
    # amortized setup is setup_us / N, which is what we use to inflate
    # the per-cell mean_us. Captured separately too for downstream tools.
    target_setup_us: dict[str, list[float]] = {t: [] for t in cfg.targets}

    for canonical, row in profile["dispatches"].items():
        infeasible_for: list[str] = []
        row_times: list[float] = []
        for target in cfg.targets:
            cell = row.get(target, {})
            if isinstance(cell, dict):
                for sz in cell.get("binding_byte_sizes", []):
                    max_bytes_seen = max(max_bytes_seen, int(sz))
            prof = cell.get("profile") if isinstance(cell, dict) else None
            mu = (prof.get("mean_us") if prof and "mean_us" in prof else None)
            su = (prof.get("setup_us") if prof and "setup_us" in prof else 0.0)
            if mu is None or mu <= 0:
                infeasible_for.append(cfg.target_to_machine[target])
                row_times.append(INFEAS)
            else:
                if isinstance(su, (int, float)) and su > 0:
                    target_setup_us[target].append(float(su))
                row_times.append(float(mu))
        if len(infeasible_for) == len(cfg.targets):
            print(f"  drop {canonical}: infeasible everywhere")
            continue
        graph_meta = dispatch_graph.get(canonical, {})
        deps = list(graph_meta.get("dependencies", []))
        graph_volume = graph_meta.get("transfer_volume_bytes")
        if graph_volume is not None:
            max_bytes_seen = max(max_bytes_seen, int(graph_volume))
        workload["dispatches"][canonical] = {
            "id": len(workload["dispatches"]),
            "dependencies": deps,
            "infeasible_machines": infeasible_for,
            "cost_by_pred": {},
        }
        proc_times[canonical] = row_times

    n_m = len(cfg.machines)
    transfer = [[0.0] * n_m for _ in range(n_m)]
    if cost_table and max_bytes_seen > 0:
        # Phase D: fill transfer matrix from cost-table linear fit.
        print(f"[phase-D] using cost table {cfg.cost_table} at "
              f"max_volume={max_bytes_seen} bytes")
        for i, src in enumerate(cfg.machines):
            for j, dst in enumerate(cfg.machines):
                if i == j:
                    continue
                t = _machine_pair_transfer_us(cost_table, src, dst,
                                              max_bytes_seen)
                transfer[i][j] = (t if t is not None else cfg.transfer_us)
    else:
        # Fallback: constant transfer_us between distinct machines.
        for i in range(n_m):
            for j in range(n_m):
                if i != j:
                    transfer[i][j] = cfg.transfer_us
    transfer_payload = {"machines": cfg.machines, "matrix": transfer,
                        "edge_volume_bytes": max_bytes_seen,
                        "from_cost_table": bool(cost_table)}

    # Setup-time summary: median per target. Recorded in workload.json
    # under a top-level "setup_us_by_machine" so downstream tooling can
    # surface it (and so the data is preserved across rounds for the
    # iterative loop's diff/replay logic).
    import statistics
    setup_by_machine = {}
    for target, vs in target_setup_us.items():
        m = cfg.target_to_machine[target]
        setup_by_machine[m] = (statistics.median(vs) if vs else 0.0)
    workload["setup_us_by_machine"] = setup_by_machine
    workload["transfer_volume_bytes"] = max_bytes_seen
    workload["transfer_from_cost_table"] = bool(cost_table)

    workload_path = round_dir / "workload.json"
    proc_path = round_dir / "processing_times.json"
    transfer_path = round_dir / "transfer_times.json"
    workload_path.write_text(json.dumps(workload, indent=2))
    proc_path.write_text(json.dumps(proc_times, indent=2))
    transfer_path.write_text(json.dumps(transfer_payload, indent=2))
    print(f"workload  -> {workload_path}  ({len(workload['dispatches'])} dispatches)")
    print(f"proc_times-> {proc_path}")
    print(f"transfer  -> {transfer_path}")
    return workload_path, proc_path, transfer_path


def step_schedule(cfg: LoopConfig,
                  workload_path: pathlib.Path,
                  proc_path: pathlib.Path,
                  transfer_path: pathlib.Path,
                  round_dir: pathlib.Path) -> pathlib.Path:
    """Invoke a self-contained MILP run via xpu-rt/workload_factory + scheduler.

    Uses a small inline driver because merlin_adapter.py expects a
    third_party/merlin/breakdowns/ directory layout we don't produce here.
    """
    sched_path = round_dir / "schedule.json"
    driver = round_dir / "_run_mosek.py"
    driver.write_text(_MOSEK_DRIVER_TEMPLATE.format(
        xpurt_root=str(XPURT_ROOT),
        workload=str(workload_path),
        proc_times=str(proc_path),
        transfers=str(transfer_path),
        machines=json.dumps(cfg.machines),
        diversity_weight=cfg.diversity_weight,
        out=str(sched_path),
    ))
    # Use the merlin-dev env's python directly to avoid nested-conda pitfalls.
    py = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"
    rc = _run([py, str(driver)])
    if rc != 0 or not sched_path.exists():
        raise RuntimeError(f"MOSEK schedule failed (rc={rc})")
    return sched_path


_MOSEK_DRIVER_TEMPLATE = '''
import json, sys
XPURT_ROOT = "{xpurt_root}"
# xpu_rt must be installed in the driver subprocess environment
# (e.g. via `uv sync` in the repo root before running this script).
sys.path.insert(0, XPURT_ROOT + "/xpu-rt/python")
from xpu_rt.scheduler.workload_factory import create_workload_from_dependencies  # noqa
from xpu_rt.scheduler.scheduler import schedule  # noqa
import numpy as np

workload_data = json.load(open("{workload}"))
proc_times = json.load(open("{proc_times}"))
transfers = json.load(open("{transfers}"))
machines = {machines}

# Build dependency dict the factory expects
deps = {{name: {{"id": d["id"], "dependencies": d["dependencies"],
                "infeasible_machines": d.get("infeasible_machines", [])}}
        for name, d in workload_data["dispatches"].items()}}

w = create_workload_from_dependencies(
    workload_data,                     # dispatch_data with "dispatches" key
    proc_times,
    machines,
    np.array(transfers["matrix"]),
)
t, alpha, _f, _fm = schedule(
    w, target_diversity_weight={diversity_weight},
    verbose=False, time_limit=60.0,
)
if t is None or alpha is None:
    print("MOSEK no solution"); sys.exit(2)

ops = list(w.operations)
machines_list = list(w.machines)
schedule_json = {{
    "machines": machines_list,
    "diversity_weight": {diversity_weight},
    "ops": [],
}}
for i, op in enumerate(ops):
    machine_idx = int(np.argmax(alpha[i]))
    schedule_json["ops"].append({{
        "name": op.operation_name,
        "machine": machines_list[machine_idx],
        "start_us": float(t[i]),
    }})
schedule_json["makespan_us"] = float(max(o["start_us"] for o in schedule_json["ops"]))
counts = {{m: 0 for m in machines_list}}
for o in schedule_json["ops"]:
    counts[o["machine"]] += 1
schedule_json["counts_per_machine"] = counts
json.dump(schedule_json, open("{out}", "w"), indent=2)
print(f"makespan_us={{schedule_json['makespan_us']:.1f}}  counts={{counts}}")
'''


_TARGET_PREFERRED_DTYPE = {
    "CPU": {"f32", "f16", "i8"},  # CPU accepts everything
    "GPU": {"f16", "f32"},         # QNN GPU prefers fp16
    "HTA": {"i8", "u8"},           # QNN HTA strictly int8/uint8
}


def step_apply_requant(cfg: LoopConfig, src: pathlib.Path,
                       schedule_path: pathlib.Path,
                       round_dir: pathlib.Path) -> pathlib.Path:
    """Phase A2: schedule-driven re-quantization.

    Reads the schedule.json's placements and decides whether the source
    MLIR's dtype regime matches each placement's preferred dtype set. When
    a mismatch is detected, we emit a placement_requant.json sidecar that
    maps {dispatch_name: {"from_dtype", "to_dtype"}} and pass the source
    through `iree-compile --merlin-placement-requant-json=<path>` to insert
    quant.qcast/dcast boundaries.

    Until the C++ ApplyPlacementRequantization pass is wired into the
    merlin compiler plugin, this function emits the sidecar and copies the
    source — the sidecar is NOT yet consumed by the compiler. The sidecar
    is what the C++ pass (when landed) will read. For models whose source
    dtype already matches every placement (e.g. yolov8_int8 placed on HTA),
    this step is a no-op regardless.
    """
    out = round_dir / "next_round_source.mlir"
    sidecar = round_dir / "placement_requant.json"

    schedule = json.loads(schedule_path.read_text())
    src_text = src.read_text()
    # Detect source-IR dtype regime from the input — coarse heuristic:
    #   contains 'i8>' / 'tensor<...xi8' → int8 source
    #   contains 'f16' → fp16 source
    #   else → f32 source
    if ("xi8>" in src_text) or ("xui8>" in src_text):
        source_dtype = "i8"
    elif "f16>" in src_text or "xf16," in src_text:
        source_dtype = "f16"
    else:
        source_dtype = "f32"

    requant_map: dict = {"source_dtype": source_dtype, "ops": {}}
    needs_requant = False
    for op in schedule["ops"]:
        m = op["machine"]
        prefs = _TARGET_PREFERRED_DTYPE.get(m, set())
        if source_dtype in prefs:
            continue
        # Pick the preferred dtype with the smallest precision-loss footprint.
        if "i8" in prefs:
            target_dtype = "i8"
        elif "f16" in prefs:
            target_dtype = "f16"
        else:
            target_dtype = next(iter(prefs)) if prefs else source_dtype
        requant_map["ops"][op["name"]] = {
            "machine": m,
            "from_dtype": source_dtype,
            "to_dtype": target_dtype,
        }
        needs_requant = True

    sidecar.write_text(json.dumps(requant_map, indent=2))

    if not needs_requant:
        print(f"[phase-A2] no re-quant needed (source={source_dtype}, all "
              f"placements compatible)")
        shutil.copy(src, out)
        return out

    print(f"[phase-A2] {len(requant_map['ops'])} ops need re-quant: "
          f"{list(requant_map['ops'])}")
    print(f"[phase-A2] sidecar written: {sidecar}")

    # Invoke the merlin compiler plugin's ApplyPlacementRequantization pass
    # via iree-opt to insert quant.qcast/dcast at the right anchors.
    iree_opt = cfg.merlin_root / "build/host-merlin-release-qrb/tools/iree-opt"
    if not iree_opt.is_file():
        print(f"[phase-A2] iree-opt not found at {iree_opt}; passing source through")
        shutil.copy(src, out)
        return out
    cmd = [
        str(iree_opt),
        "--iree-plugin=hal_target_qnn",  # ensures the QNN/merlin pass is registered
        f"--pass-pipeline=builtin.module(merlin-apply-placement-requant{{sidecar={sidecar}}})",
        str(src),
        "-o", str(out),
    ]
    rc = _run(cmd)
    if rc != 0:
        print(f"[phase-A2] ApplyPlacementRequantization rc={rc}; using source as-is")
        shutil.copy(src, out)
    return out


def placements_match(prev_schedule: pathlib.Path | None,
                     cur_schedule: pathlib.Path) -> bool:
    if prev_schedule is None:
        return False
    a = {o["name"]: o["machine"] for o in
         json.loads(prev_schedule.read_text())["ops"]}
    b = {o["name"]: o["machine"] for o in
         json.loads(cur_schedule.read_text())["ops"]}
    return a == b


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--targets", default="cpu,qnn_gpu,qnn_hta")
    p.add_argument("--diversity-weight", type=float, default=100.0)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--transfer-us", type=float, default=200.0)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ssh-host", default="root@10.44.120.201")
    p.add_argument("--ssh-identity", type=pathlib.Path,
                   default=pathlib.Path("/scratch2/agustin/DIMA_SLICE"))
    p.add_argument("--profile-input-mode", choices=("captured", "zero"),
                   default="captured",
                   help="Input mode for standalone per-dispatch profiling")
    p.add_argument("--capture-dir", type=pathlib.Path, default=None,
                   help="Per-dispatch captured-input root used when "
                        "--profile-input-mode=captured")
    p.add_argument("--dispatch-graph-json", type=pathlib.Path, default=None,
                   help="Optional real dependency manifest to carry into "
                        "matrix/profiled_manifest and the XPU-RT workload DAG")
    p.add_argument("--merlin-root", type=pathlib.Path, default=DEFAULT_MERLIN_ROOT,
                   help="Path to the merlin repo (default: $MERLIN_ROOT or "
                        f"{DEFAULT_MERLIN_ROOT})")
    p.add_argument("--iree-compile", type=pathlib.Path, default=None,
                   help="Path to merlin's iree-compile (default: "
                        "<merlin-root>/build/host-merlin-release-qrb/tools/iree-compile)")
    p.add_argument("--cost-table", type=pathlib.Path, default=None,
                   help="Path to qrb5165_costs.json from "
                        "profile_transfers_on_board.py (Phase D). When set, "
                        "transfer_times.json is built from the linear-fit "
                        "coefficients instead of a constant value.")
    p.add_argument("--skip-profile", action="store_true",
                   help="Synthesise per-cell timings instead of running on board")
    p.add_argument("--run-on-board", action="store_true",
                   help="After scheduling, invoke run_on_board_flow.py to "
                        "execute the schedule on the QRB5165 board with "
                        "real data flowing dispatch-to-dispatch (Phase F).")
    p.add_argument("--input-from", type=pathlib.Path, default=None,
                   help="Raw bytes for the chain's first input (with "
                        "--run-on-board)")
    p.add_argument("--output-to", type=pathlib.Path, default=None,
                   help="Where to write the chain's final output bytes "
                        "(with --run-on-board)")
    p.add_argument("--verify-against", type=pathlib.Path, default=None,
                   help="Reference output binary; after --run-on-board "
                        "completes, calls third_party/merlin/tools/verify_het_e2e.py "
                        "compare to assert numerical equivalence within "
                        "--rtol/--atol (Phase G integration).")
    p.add_argument("--verify-shape", default=None,
                   help="Shape+dtype of --output-to / --verify-against, "
                        "e.g. 1x320xi8. Required with --verify-against.")
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args(argv)
    if args.iree_compile is None:
        args.iree_compile = (args.merlin_root /
                             "build/host-merlin-release-qrb/tools/iree-compile")
    if (not args.skip_profile and args.profile_input_mode == "captured" and
            args.capture_dir is None):
        print("error: --capture-dir is required when "
              "--profile-input-mode=captured", file=sys.stderr)
        return 2

    targets = [t.strip() for t in args.targets.split(",")]
    target_to_machine = {"cpu": "CPU", "qnn_gpu": "GPU", "qnn_hta": "HTA"}
    machines = [target_to_machine[t] for t in targets]
    machine_to_target = {m: t for t, m in target_to_machine.items()}

    cfg = LoopConfig(
        source=args.source.resolve(),
        out_dir=args.out_dir.resolve(),
        targets=targets,
        machines=machines,
        target_to_machine=target_to_machine,
        machine_to_target=machine_to_target,
        diversity_weight=args.diversity_weight,
        max_rounds=args.max_rounds,
        transfer_us=args.transfer_us,
        iterations=args.iterations,
        warmup=args.warmup,
        iree_compile=args.iree_compile.resolve(),
        skip_profile=args.skip_profile,
        merlin_root=args.merlin_root.resolve(),
        cost_table=args.cost_table.resolve() if args.cost_table else None,
        ssh_host=args.ssh_host,
        ssh_identity=args.ssh_identity.resolve() if args.ssh_identity else None,
        profile_input_mode=args.profile_input_mode,
        capture_dir=args.capture_dir.resolve() if args.capture_dir else None,
        dispatch_graph_json=(
            args.dispatch_graph_json.resolve()
            if args.dispatch_graph_json else None
        ),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cur_source = cfg.source
    prev_sched: pathlib.Path | None = None
    final_sched: pathlib.Path | None = None
    for k in range(cfg.max_rounds):
        print(f"\n========== ROUND {k} ==========")
        rd = _round_dir(cfg.out_dir, k)
        matrix = step_compile_matrix(cfg, cur_source, rd)
        prof = step_profile(cfg, matrix, rd)
        wl, pt, tt = build_workload(cfg, prof, rd)
        sched = step_schedule(cfg, wl, pt, tt, rd)
        final_sched = sched
        if placements_match(prev_sched, sched):
            print(f"Placement stable after round {k}; converged.")
            break
        prev_sched = sched
        cur_source = step_apply_requant(cfg, cur_source, sched, rd)
    else:
        print(f"Hit max_rounds={cfg.max_rounds} without convergence.")

    if final_sched:
        final_link = cfg.out_dir / "schedule.json"
        if final_link.exists() or final_link.is_symlink():
            final_link.unlink()
        final_link.symlink_to(final_sched.resolve())
        print(f"\nFinal schedule: {final_link} -> {final_sched.resolve()}")
        sched_data = json.loads(final_sched.read_text())
        print(f"Makespan: {sched_data['makespan_us']:.1f} us")
        print(f"Counts:   {sched_data['counts_per_machine']}")

    # Phase F integration: optionally run the schedule on board.
    if args.run_on_board and final_sched is not None:
        print("\n========== ON-BOARD RUN (Phase F) ==========")
        # Use the latest round's profiled_manifest (it has vmfb_remote paths).
        last_round = cfg.max_rounds - 1
        # Find the actual last round dir (may have terminated earlier).
        manifest_path = None
        for k in range(cfg.max_rounds - 1, -1, -1):
            cand = cfg.out_dir / f"round_{k}" / "profiled_manifest.json"
            if cand.exists():
                manifest_path = cand
                break
        if manifest_path is None:
            print("error: no profiled_manifest.json found; run without --skip-profile.")
            return 1
        onboard_dir = cfg.out_dir / "onboard"
        cmd = [
            "python3",
            str(XPURT_ROOT / "scripts" / "run_on_board_flow.py"),
            "--schedule", str(final_sched),
            "--manifest", str(manifest_path),
            "--out-dir", str(onboard_dir),
            "--iterations", str(cfg.iterations),
            "--warmup", str(cfg.warmup),
            "--board-vmfb-dir", "/root/dispatch_profile",
            "--ssh-host", cfg.ssh_host,
        ]
        if cfg.ssh_identity:
            cmd.extend(["--ssh-identity", str(cfg.ssh_identity)])
        if args.input_from:
            cmd.extend(["--input-from", str(args.input_from)])
        if args.output_to:
            cmd.extend(["--output-to", str(args.output_to)])
        rc = _run(cmd)
        if rc != 0:
            print(f"on-board run failed (rc={rc})")

        # Phase G integration: numerical verify if a reference is provided.
        if (rc == 0 and args.verify_against and args.output_to and
                args.verify_shape):
            verify_cmd = [
                "conda", "run", "-n", "merlin-dev", "uv", "run", "python",
                str(args.merlin_root / "tools" / "verify_het_e2e.py"),
                "compare",
                "--reference", str(args.verify_against),
                "--candidate", str(args.output_to),
                "--shape", args.verify_shape,
                "--rtol", str(args.rtol),
                "--atol", str(args.atol),
            ]
            verify_rc = _run(verify_cmd)
            print(f"\nVerify rc={verify_rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
