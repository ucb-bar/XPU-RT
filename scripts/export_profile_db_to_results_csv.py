#!/usr/bin/env python3
"""Bridge ModelBlaster's FireSim profile DB into the CSV layout XPU-RT reads.

Why this exists
---------------
The two repos measure and consume per-dispatch costs in different places, with
disjoint coverage:

  ModelBlaster  benchmarks/profile_db/<model>__<backend>__int8.jsonl
                has yolov8_nano_64 and the `gemmini` / `rvv_opu` backends that
                the FireSimGemminiAndOPUShuttleConfig bitstream actually
                implements (tile 0 Shuttle+Gemmini RoCC, tile 1 Shuttle+Saturn
                OPU).

  XPU-RT        gen/profile/<hw>/<target>/<model>/<basename>/<topo>/results.csv
                (profile_loader.find_profile_csv) has only `gemmini_q31` and
                `V256D128_rvv`, and no yolov8_nano_64 at all.

So XPU-RT cannot schedule the workload we want to run on the bitstream we have.
This script closes that gap by exporting the profile DB's median cycles per
dispatch into the IREE `results.csv` schema and directory layout that
profile_loader already globs — one canonical measured source, no reimplemented
cost model.

Provenance
----------
Every row keeps raw `cycles` alongside the derived `mean_time`, and each CSV
gets a sidecar `_provenance.json` recording the clock assumption, both repo
SHAs, and the per-dispatch sample counts.

The clock matters and is not the FPGA's. Cycles are converted with
`--clock-mhz` (default 1000.0, i.e. 1 GHz), which is what ModelBlaster's
solver configs and profile pipeline assume (`cycles_per_ms: 1000000`). The
Alveo U250 bitstreams actually close timing at 25-30 MHz. At 25 MHz a DroNet
inference is 361 ms and a 10 ms control period is impossible, so the
millisecond-denominated workload only exists under the 1 GHz assumption. That
is a documented, uniform scale factor recorded in every manifest — not a
hidden one — and it cancels in relative comparisons, but no absolute
millisecond claim should be made without restating it.

Zero-cost ops
-------------
ModelBlaster treats `view` and `chunk2_c1` as zero-cost aliases: the generated
walker skips the kernel call but still posts the dependency semaphore
(pipeline/ingest_xpurt_schedule.py `_ZERO_COST_OPS`). They therefore appear in
the dispatch graph but never in a profile run. This script emits them with
cycles=0 and `source=zero_cost_by_construction`, so precedence edges survive
without inventing a duration, and reports how many it did that for.

Any OTHER unprofiled dispatch is a hard error. Silently substituting a cost is
exactly how a schedule gets built against fictional timings.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# ModelBlaster's zero-cost alias ops. Kept in sync with
# ModelBlaster/pipeline/ingest_xpurt_schedule.py::_ZERO_COST_OPS.
ZERO_COST_OPS = {"view", "chunk2_c1"}

# IREE results.csv schema (9 IREE columns + 4 ModelBlaster extensions),
# matching ModelBlaster/pipeline/profile_writer.py.
CSV_COLUMNS = [
    "dispatch_id",
    "module_name",
    "vmfb_path",
    "mlir_path",
    "mean_time",
    "mean_unit",
    "mean_time_ns",
    "returncode",
    "log_path",
    "source",
    "op",
    "shape",
    "cycles",
]


def _git_sha(repo: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _git_dirty(repo: str) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def load_profile_db(path: str) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, str]]:
    """Return (median cycles, sample count, op type) per dispatch_id."""
    samples: Dict[int, List[float]] = collections.defaultdict(list)
    op_types: Dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = int(r["dispatch_id"])
            samples[did].append(float(r["cycles"]))
            if r.get("op_type"):
                op_types[did] = str(r["op_type"])
    medians = {d: statistics.median(v) for d, v in samples.items()}
    counts = {d: len(v) for d, v in samples.items()}
    return medians, counts, op_types


def load_graph_ops(graph_json: str, ir_json: Optional[str]) -> Tuple[Set[int], Dict[int, str]]:
    """Return the dispatch ids in the emitted XPU-RT graph, plus op types.

    Op types come from the ModelBlaster IR when available, because the emitted
    XPU-RT graph deliberately carries only ids and dependencies.
    """
    with open(graph_json) as f:
        g = json.load(f)
    ids = {int(v["id"]) for v in g["dispatches"].values()}

    op_by_id: Dict[int, str] = {}
    if ir_json and os.path.exists(ir_json):
        with open(ir_json) as f:
            ir = json.load(f)
        for op in ir.get("ops", []):
            # `view` ops carry dispatch_id: null — they are pure aliases and
            # never become dispatches at all.
            did = op.get("dispatch_id")
            if did is None:
                continue
            op_by_id[int(did)] = str(op.get("op", ""))
    return ids, op_by_id


def export_one(
    *,
    model: str,
    backend: str,
    quant: str,
    target: str,
    topo_tag: str,
    mb_root: str,
    out_root: str,
    graph_root: str,
    clock_mhz: float,
    xpurt_root: str,
) -> Dict[str, object]:
    basename = f"{model}.{quant}"
    db_path = os.path.join(
        mb_root, "benchmarks", "profile_db", f"{model}__{backend}__{quant}.jsonl"
    )
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"no profile DB for ({model}, {backend}, {quant}): {db_path}\n"
            f"Available: "
            + ", ".join(sorted(os.listdir(os.path.join(mb_root, 'benchmarks', 'profile_db'))))
        )

    graph_json = os.path.join(
        graph_root, model, target, "gemmini", basename, f"{basename}_dispatch_graph.json"
    )
    if not os.path.exists(graph_json):
        raise FileNotFoundError(
            f"no emitted dispatch graph for {model}: {graph_json}\n"
            f"Generate it first with ModelBlaster's emitter:\n"
            f"  python -m pipeline.emit_dispatch_graph "
            f"--ir examples/{model}/{quant}/generated/graph.json "
            f"--out-root {graph_root} --target {target} --hw gemmini"
        )
    ir_json = os.path.join(mb_root, "examples", model, quant, "generated", "graph.json")

    medians, counts, db_ops = load_profile_db(db_path)
    graph_ids, ir_ops = load_graph_ops(graph_json, ir_json)

    def op_of(did: int) -> str:
        return ir_ops.get(did) or db_ops.get(did) or ""

    # Partition the graph's dispatches by whether we have a measurement.
    profiled = sorted(graph_ids & set(medians))
    unprofiled = sorted(graph_ids - set(medians))
    structural_zero = [d for d in unprofiled if op_of(d) in ZERO_COST_OPS]
    genuinely_missing = [d for d in unprofiled if op_of(d) not in ZERO_COST_OPS]

    if genuinely_missing:
        detail = ", ".join(f"id={d} op={op_of(d)!r}" for d in genuinely_missing[:12])
        raise RuntimeError(
            f"({model}, {backend}): {len(genuinely_missing)} dispatch(es) in the "
            f"graph have no measurement and are not zero-cost ops: {detail}"
            + ("..." if len(genuinely_missing) > 12 else "")
            + f"\nProfile them on the target before scheduling. Substituting a "
            f"cost here is how a schedule gets built against fictional timings."
        )

    cycles_per_ms = clock_mhz * 1000.0  # MHz -> cycles per millisecond
    ns_per_cycle = 1000.0 / clock_mhz

    out_dir = os.path.join(
        out_root, backend, target, model, basename, topo_tag
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    total_cycles = 0.0
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for did in sorted(graph_ids):
            is_zero = did in structural_zero
            cyc = 0.0 if is_zero else float(medians[did])
            total_cycles += cyc
            op = op_of(did)
            w.writerow({
                "dispatch_id": did,
                "module_name": f"{model}$dispatch_{did}_{op}" if op else f"{model}$dispatch_{did}",
                "vmfb_path": "",
                "mlir_path": "",
                "mean_time": f"{cyc / cycles_per_ms:.9f}",
                "mean_unit": "ms",
                "mean_time_ns": f"{cyc * ns_per_cycle:.3f}",
                "returncode": 0,
                "log_path": "",
                "source": (
                    "zero_cost_by_construction" if is_zero else "firesim_measured"
                ),
                "op": op,
                "shape": "",
                "cycles": f"{cyc:.0f}",
            })

    prov = {
        "schema": "xpurt.profile_bridge/1",
        "model": model,
        "backend": backend,
        "quant": quant,
        "target": target,
        "topo_tag": topo_tag,
        # timing provenance
        "timing_source": "firesim_measured",
        "measured_or_derived": "measured_cycles_derived_ms",
        "clock_mhz_assumed": clock_mhz,
        "scaling_factor_cycles_to_ms": 1.0 / cycles_per_ms,
        "clock_note": (
            "Cycles are FireSim-measured. mean_time is derived as "
            f"cycles/{cycles_per_ms:.0f}, i.e. an assumed {clock_mhz} MHz target "
            "clock. This is NOT the Alveo U250 bitstream frequency (25-30 MHz). "
            "Raw cycles are retained in the `cycles` column."
        ),
        "source_file": os.path.relpath(db_path, mb_root),
        "source_repo": mb_root,
        "source_repo_sha": _git_sha(mb_root),
        "source_repo_dirty": _git_dirty(mb_root),
        "xpurt_repo_sha": _git_sha(xpurt_root),
        "dispatch_graph": os.path.relpath(graph_json, xpurt_root),
        # coverage
        "n_dispatches_in_graph": len(graph_ids),
        "n_profiled": len(profiled),
        "n_zero_cost_by_construction": len(structural_zero),
        "zero_cost_dispatch_ids": structural_zero,
        "zero_cost_ops": sorted({op_of(d) for d in structural_zero}),
        "n_profiled_not_in_graph": len(set(medians) - graph_ids),
        "samples_per_dispatch_min": min(counts.values()) if counts else 0,
        "samples_per_dispatch_max": max(counts.values()) if counts else 0,
        "total_cycles": total_cycles,
        "total_ms_at_assumed_clock": total_cycles / cycles_per_ms,
    }
    with open(os.path.join(out_dir, "_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    return {
        "model": model,
        "backend": backend,
        "csv": csv_path,
        "n_graph": len(graph_ids),
        "n_profiled": len(profiled),
        "n_zero": len(structural_zero),
        "total_ms": total_cycles / cycles_per_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--modelblaster-root", default="/scratch2/agustin/ModelBlaster")
    ap.add_argument(
        "--models", default="dronet,mlp_control,yolov8_nano_64",
        help="comma-separated model names",
    )
    ap.add_argument(
        "--backends", default="gemmini,rvv_opu",
        help="comma-separated profile_db backend tags. gemmini+rvv_opu match the "
             "FireSimGemminiAndOPUShuttleConfig bitstream.",
    )
    ap.add_argument("--quant", default="int8")
    ap.add_argument(
        "--target", default="firesim_gemmini_opu",
        help="target tag used in the output path; should name the bitstream",
    )
    ap.add_argument(
        "--topo-tag", default="topo_0",
        help="single-core-per-cluster topology (topo_tag_for_combination of a "
             "singleton combination)",
    )
    ap.add_argument(
        "--clock-mhz", type=float, default=1000.0,
        help="assumed target clock for the cycles->ms conversion (default 1000.0, "
             "matching ModelBlaster's cycles_per_ms=1000000). NOT the FPGA "
             "frequency; recorded in every _provenance.json.",
    )
    ap.add_argument("--out-root", default=None, help="default <repo>/gen/profile")
    ap.add_argument("--graph-root", default=None, help="default <repo>/gen/vmfb")
    args = ap.parse_args()

    xpurt_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out_root = args.out_root or os.path.join(xpurt_root, "gen", "profile")
    graph_root = args.graph_root or os.path.join(xpurt_root, "gen", "vmfb")

    models = [m for m in args.models.split(",") if m]
    backends = [b for b in args.backends.split(",") if b]

    print(f"ModelBlaster: {args.modelblaster_root}")
    print(f"  sha {_git_sha(args.modelblaster_root)[:12]} "
          f"dirty={_git_dirty(args.modelblaster_root)}")
    print(f"assumed clock: {args.clock_mhz} MHz "
          f"(cycles/{args.clock_mhz * 1000:.0f} -> ms); NOT the FPGA frequency")
    print(f"out: {os.path.relpath(out_root, xpurt_root)}/<backend>/{args.target}/"
          f"<model>/<model>.{args.quant}/{args.topo_tag}/results.csv")
    print()

    rows = []
    for model in models:
        for backend in backends:
            rows.append(export_one(
                model=model, backend=backend, quant=args.quant,
                target=args.target, topo_tag=args.topo_tag,
                mb_root=args.modelblaster_root, out_root=out_root,
                graph_root=graph_root, clock_mhz=args.clock_mhz,
                xpurt_root=xpurt_root,
            ))

    print(f"{'model':<17}{'backend':<10}{'graph':>6}{'profiled':>10}"
          f"{'zero':>6}{'total ms':>11}")
    for r in rows:
        print(f"{r['model']:<17}{r['backend']:<10}{r['n_graph']:>6}"
              f"{r['n_profiled']:>10}{r['n_zero']:>6}{r['total_ms']:>11.3f}")

    n_zero = sum(r["n_zero"] for r in rows)
    if n_zero:
        print(f"\n{n_zero} row(s) emitted as cycles=0 "
              f"(source=zero_cost_by_construction): ModelBlaster treats "
              f"{sorted(ZERO_COST_OPS)} as zero-cost aliases whose kernel call is "
              f"skipped while the dependency semaphore is still posted, so they "
              f"appear in the graph but never in a profile run. Precedence edges "
              f"are preserved without inventing a duration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
