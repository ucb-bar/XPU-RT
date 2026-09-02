#!/usr/bin/env python3
"""Merge a per-target profiled_manifest.json into qnn_scheduler/qrb5165_costs.json.

Each compile + board_roundtrip cycle produces one
`<output_dir>/breakdowns/profiled_manifest.json` per target. This script
takes one or more such manifests (each tagged with the backend they were
measured on) and writes their `mean_time_us` per-dispatch into the
cost-table's `execute` map under the canonical key:

    <op_kind>@<input_shape>-><output_shape>@<dtype>::<backend>::0

`op_kind` is derived from the dispatch's op_summary
("elementwise_3x320x320_f32xi8" → "elementwise"). Shape signature is
the input/output tensor shapes from the breakdowns/dispatch_*.shapes.json.
dtype is the element type of the first output tensor.

No estimates, no extrapolations: only rows that have a measured
mean_time_us land in the table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from qnn_scheduler.cost_table import CostTable  # noqa: E402


# Map IREE-MLIR element types to our dtype tags.
_DTYPE_MAP = {
    "f32": "fp32", "f16": "fp16",
    "i8": "int8",  "ui8": "uint8", "i32": "int32", "ui32": "uint32",
    "i64": "int64",
}


def _parse_tensor(s: str) -> tuple[str, tuple[int, ...]]:
    """Parse "tensor<3x320x320xf32>" → ("fp32", (3, 320, 320))."""
    m = re.match(r"tensor<([0-9x?]+)x([a-z][a-z0-9_]*)>", s)
    if not m:
        return "unknown", ()
    shape_str, et = m.group(1), m.group(2)
    shape = tuple(int(d) if d.isdigit() else 0 for d in shape_str.split("x"))
    return _DTYPE_MAP.get(et, et), shape


def _shape_str(shape: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape) if shape else "?"


def _op_kind_from_summary(op_summary: str) -> str:
    """yolov8 op_summary examples:
        elementwise_3x320x320_f32xi8     → elementwise
        conv_2d_nchw_fchw_q_…            → Conv2d
        pooling_nchw_max_…               → MaxPool
        pack_…  / unpack_…               → Pack/Unpack
    Strip the trailing _<shape> + dtype suffix."""
    parts = op_summary.split("_")
    # Heuristic: drop tokens that look like shapes ("3x320x320") or
    # type signatures ("f32xi8").
    keep = []
    for tok in parts:
        if re.fullmatch(r"\d+(x\d+)*", tok):
            break
        if re.fullmatch(r"[a-z]\d+x?[a-z]?\d*", tok):
            break
        keep.append(tok)
    return "_".join(keep) or op_summary


def ingest(profiled_manifest: pathlib.Path, breakdowns_dir: pathlib.Path,
           backend: str, table: CostTable, *, infeasibility_source: str = "") -> tuple[int, int]:
    """Ingest one (profiled_manifest, backend) pair.

    Records two kinds of rows, both real measurements, never extrapolated:
      - "measured" — a real mean_time_us came from board_roundtrip.
      - "infeasible" — the dispatch's per-target VMFB exists in
        breakdowns/manifest.json (built attempt was made) but no
        mean_time_us came back (board_roundtrip got no sample). That's
        the signal the runtime rejected the placeholder/empty ctxbin.

    Both row kinds are keyed BY DISPATCH NAME (not by shape signature)
    so the scheduler can do exact per-dispatch lookups. Returns
    (n_measured, n_infeasible).
    """
    data = json.loads(profiled_manifest.read_text())
    dispatches = data.get("dispatches", {})
    # Also load the build-time manifest: any dispatch present there but
    # missing from `dispatches` (or with mean_time_us=None) is the
    # population we mark infeasible.
    bm_path = breakdowns_dir / "manifest.json"
    build_manifest = json.loads(bm_path.read_text()) if bm_path.exists() else {"dispatches": {}}
    build_dispatches = build_manifest.get("dispatches", {})

    n_meas = 0
    n_infe = 0
    for name in build_dispatches:
        e = dispatches.get(name, {}) or {}
        mean_us = e.get("mean_time_us")
        # Per-dispatch row key — unambiguous, no aggregation across shapes.
        row_key = f"dispatch::{name}::{backend}::0"
        if mean_us is None:
            table.execute[row_key] = {
                "infeasible": True,
                "reason": (infeasibility_source or
                           "no sample from board_roundtrip "
                           "(placeholder ctxbin or runtime reject)"),
                "extrapolated": False,
                "source": f"build_attempted_no_sample {profiled_manifest}",
                "dispatch_name": name,
                "backend": backend,
            }
            n_infe += 1
        else:
            table.execute[row_key] = {
                "mean_us": float(mean_us),
                "median_us": float(e.get("median_time_us") or mean_us),
                "stddev_us": float(e.get("stddev_time_us") or 0.0),
                "iters": int(e.get("repetitions") or 0),
                "infeasible": False,
                "extrapolated": False,
                "source": f"board_roundtrip iree-benchmark-module {profiled_manifest}",
                "dispatch_name": name,
                "backend": backend,
            }
            n_meas += 1
    return n_meas, n_infe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-table", required=True, type=pathlib.Path,
                        help="qrb5165_costs.json")
    parser.add_argument("--manifest", required=True, type=pathlib.Path,
                        action="append",
                        help="profiled_manifest.json (repeatable)")
    parser.add_argument("--backend", required=True, action="append",
                        choices=["CPU", "GPU", "HTA"],
                        help="One per --manifest, in matching order")
    args = parser.parse_args()

    if len(args.manifest) != len(args.backend):
        parser.error("--manifest and --backend must have same count")

    table = CostTable.load(args.cost_table) if args.cost_table.exists() else CostTable()
    if not table.device:
        table.device = "qrb5165"
    if not table.qairt_sdk:
        table.qairt_sdk = "2.45.0.260326"

    total_meas = 0
    total_infe = 0
    for manifest_path, backend in zip(args.manifest, args.backend):
        breakdowns = manifest_path.parent
        meas, infe = ingest(manifest_path, breakdowns, backend, table)
        print(f"  {backend}: {meas} measured, {infe} infeasible from {manifest_path}")
        total_meas += meas
        total_infe += infe

    table.save(args.cost_table)
    print(f"  TOTAL: {total_meas} measured, {total_infe} infeasible across "
          f"{len(args.backend)} backends")
    print(f"  cost table now has {len(table.execute)} execute rows at {args.cost_table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
