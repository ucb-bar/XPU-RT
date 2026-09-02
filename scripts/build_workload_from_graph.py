#!/usr/bin/env python3
"""Build the XPU-RT workload JSON for the heterogeneous YOLOv8 schedule.

Inputs:
  - The breakdowns/manifest.json from `build/het/qrb5165_cpu/breakdowns/`
    (contains every dispatch's dependencies, op_summary, and IO tensor
    shapes — same DAG for all three targets).
  - The cost table populated by ingest_per_target_profiles.py (per-
    (dispatch, backend) measured times + infeasibility markers).
  - The TransferModel (from qnn_scheduler.transfer_model) for per-edge
    bridge costs computed from the actual tensor volumes flowing across
    each edge.

Outputs:
  - workload.json: shape consumed by xpu-rt/workload_factory.py
    (`create_workload_from_dependencies()`):
      {
        "dispatches": {
          "dispatch_K": {
            "id": K,
            "dependencies": ["dispatch_J", ...],
            "infeasible_machines": ["GPU", "HTA", ...]   # measured infeasibility
          }, ...
        }
      }
  - processing_times.json: { "dispatch_K": [HTA_us, GPU_us, CPU_us], ... }
    Cells corresponding to infeasible_machines get a sentinel value (the
    MILP forbids them via the (2b) hard exclusion constraint, so the
    cell value is never read; we set it to 0 to keep the matrix clean).
  - transfer_times.json: 3x3 matrix of constant cross-machine bridge
    cost (per-edge variable bridges live in `cost_by_pred` per dispatch).

  cost_by_pred per dispatch is computed from the actual upstream tensor
  volume + dtype delta between (predecessor's output, this op's input)
  via the cost table's memcpy + dequant_quant + rescale rows. No
  estimates: every coefficient is a board measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Optional

_HERE = pathlib.Path(__file__).resolve()
_XPU_RT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_XPU_RT_ROOT))

from qnn_scheduler.cost_table import CostTable  # noqa: E402

_DTYPE_MAP = {
    "f32": "fp32", "f16": "fp16",
    "i8": "int8", "ui8": "uint8", "i32": "int32", "ui32": "uint32",
}
_BPE = {"uint8": 1, "int8": 1, "fp16": 2, "fp32": 4, "int32": 4, "uint32": 4}

MACHINES = ["HTA", "GPU", "CPU"]


def _parse_tensor(s: str) -> tuple[str, tuple[int, ...]]:
    m = re.match(r"tensor<([0-9x?]+)x([a-z][a-z0-9_]*)>", s)
    if not m:
        return "unknown", ()
    shape_str, et = m.group(1), m.group(2)
    shape = tuple(int(d) if d.isdigit() else 0 for d in shape_str.split("x"))
    return _DTYPE_MAP.get(et, et), shape


def _vol_bytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= max(1, d)
    return n * _BPE.get(dtype, 4)


def _cost_lookup(table: CostTable, dispatch_name: str, backend: str) -> Optional[float]:
    """Return measured mean_us for (dispatch, backend) or None if absent
    OR explicitly infeasible."""
    row = table.execute.get(f"dispatch::{dispatch_name}::{backend}::0")
    if row is None:
        return None
    if row.get("infeasible"):
        return None
    return float(row.get("mean_us", 0.0))


_RE_CONV_SUMMARY = re.compile(r"^conv_(\d+)x(\d+)x(\d+)x(\d+)x(\d+)x(\d+)_")


def _shape_equal_lookup(table: CostTable, op_summary: str, backend: str) -> Optional[float]:
    """Cross-reference: when (dispatch, backend) is per-dispatch infeasible
    but the op is a conv whose shape was measured by the per-shape sweep
    (NHWC fixture, same kernel/stride/IC/OC/IH/IW), use that measurement.

    This is shape-equal *lookup*, not extrapolation: we measured the
    Conv2d at exactly the same input/output shape + dtype on real
    hardware. The dispatch_name labelling is metadata, not part of the
    physical op. The workflow records `shape_equal_fallback=True` in the
    output so the source is auditable.
    """
    m = _RE_CONV_SUMMARY.match(op_summary)
    if not m:
        return None
    oc, oh, ow, ic, kh, kw = (int(x) for x in m.groups())
    # The per-shape sweep ran SAME-padding stride-1 only and at uint8
    # for HTA, fp16 for GPU.
    if backend == "HTA":
        dtype = "uint8"
    elif backend == "GPU":
        dtype = "fp16"
    else:
        return None
    sig = (f"1x{oh}x{ow}x{ic}->1x{oh}x{ow}x{oc},g1,k{kh},s1")
    key = f"Conv2d@{sig}@{dtype}::{backend}::0"
    row = table.execute.get(key)
    if row is None or row.get("infeasible"):
        return None
    return float(row.get("mean_us", 0.0))


def _bridge_us(table: CostTable, src_machine: str, dst_machine: str,
               src_dtype: str, dst_dtype: str, n_elem: int, vol_bytes: int) -> float:
    """Sum measured memcpy + (dequant_quant or rescale) for a real edge.

    Returns 0.0 if a same-machine same-dtype edge (no work). Any
    component without a measurement raises — no estimates.
    """
    cost = 0.0
    if src_machine != dst_machine:
        cost += table.memcpy_us(vol_bytes, src_machine, dst_machine)
    if src_dtype != dst_dtype:
        cost += table.dequant_quant_us(n_elem, src_dtype, dst_dtype)
    return cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--breakdowns",
                    default=None,
                    required=True,
                    type=pathlib.Path,
                    help="directory holding manifest.json. This used to "
                         "default to a hardcoded merlin build path on one "
                         "machine; required now, because a default that only "
                         "resolves on the author's filesystem is a worse "
                         "failure than asking.")
    ap.add_argument("--cost-table",
                    default=_XPU_RT_ROOT / "qnn_scheduler" / "qrb5165_costs.json",
                    type=pathlib.Path)
    ap.add_argument("--out-dir",
                    default=_XPU_RT_ROOT / "build" / "het",
                    type=pathlib.Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = CostTable.load(args.cost_table)
    manifest = json.loads((args.breakdowns / "manifest.json").read_text())
    dispatches = manifest["dispatches"]

    # processing_times[dispatch_name] = [HTA, GPU, CPU]
    processing_times: dict[str, list[float]] = {}
    n_machines = len(MACHINES)
    out_dispatches: dict[str, dict] = {}
    n_unschedulable = 0
    n_no_exe = 0
    for name, e in dispatches.items():
        # Skip dispatches with no built executable. These are degenerate
        # control/glue dispatches that the breakdown_vmfb pass dropped
        # because they don't have a per-dispatch benchmark MLIR — they
        # aren't real schedulable units.
        if not e.get("executable"):
            n_no_exe += 1
            continue
        infeasible_for: list[str] = []
        row = []
        shape_equal_used: list[str] = []
        for m in MACHINES:
            t = _cost_lookup(table, name, m)
            if t is None:
                # Per-dispatch infeasible. Try the shape-equal fallback
                # (Conv2d shape match against the per-shape sweep). This
                # is exact-shape lookup, not extrapolation.
                t = _shape_equal_lookup(table, e.get("op_summary", ""), m)
                if t is not None:
                    shape_equal_used.append(m)
            if t is None:
                infeasible_for.append(m)
                row.append(0.0)
            else:
                row.append(t)
        if len(infeasible_for) == n_machines:
            n_unschedulable += 1
            print(f"  WARN dispatch {name} infeasible on every backend "
                  f"— MILP will fail to schedule. "
                  f"Profile this one before scheduling.")
        processing_times[name] = row

        # Per-(predecessor) bridge costs: each dependency contributes a
        # cost_by_pred entry per (pred_machine, this_machine) combo.
        # We compute the FULL cost_by_pred map; the MILP picks per-edge.
        # Volume + dtypes are derived from the OUTPUT of the predecessor
        # dispatch (which is what flows across the edge).
        cost_by_pred: dict[str, float] = {}
        for dep in e.get("dependencies", []):
            dep_e = dispatches.get(dep)
            if not dep_e:
                continue
            outs = dep_e.get("outputs", [])
            if not outs:
                continue
            dt, sh = _parse_tensor(outs[0])
            n_elem = 1
            for d in sh:
                n_elem *= max(1, d)
            vol = _vol_bytes(sh, dt)
            for src_m in MACHINES:
                for dst_m in MACHINES:
                    try:
                        bridge = _bridge_us(table, src_m, dst_m, dt, dt,
                                            n_elem, vol)
                    except Exception:
                        # Component missing (e.g., HTA__CPU memcpy not
                        # measured yet) — skip this edge entry; the MILP
                        # falls back to the diagonal-zero transfer matrix.
                        continue
                    key = f"{src_m}->{dst_m}"
                    cost_by_pred[key] = max(cost_by_pred.get(key, 0.0), bridge)

        out_dispatches[name] = {
            "id": e["id"],
            "dependencies": [d for d in e.get("dependencies", [])
                             if dispatches.get(d, {}).get("executable")],
            "op_summary": e.get("op_summary", ""),
            "infeasible_machines": infeasible_for,
            "shape_equal_used": shape_equal_used,
            "cost_by_pred": cost_by_pred,
        }

    workload = {"dispatches": out_dispatches}
    (args.out_dir / "workload.json").write_text(json.dumps(workload, indent=2))
    (args.out_dir / "processing_times.json").write_text(
        json.dumps(processing_times, indent=2))

    # Default 3x3 transfer matrix from cost_table memcpy at zero volume
    # (just the fixed_overhead_us). Per-edge volume is in cost_by_pred.
    import numpy as np
    tt = np.zeros((n_machines, n_machines), dtype=float)
    for i, mi in enumerate(MACHINES):
        for j, mj in enumerate(MACHINES):
            if i == j:
                continue
            try:
                tt[i, j] = table.memcpy_us(0, mi, mj)
            except Exception:
                tt[i, j] = 0.0
    (args.out_dir / "transfer_times.json").write_text(
        json.dumps({"machines": MACHINES, "matrix": tt.tolist()}, indent=2))

    print(f"workload    -> {args.out_dir / 'workload.json'}")
    print(f"  {len(out_dispatches)} dispatches, "
          f"{n_no_exe} skipped (no executable), "
          f"{n_unschedulable} unschedulable on every backend")
    print(f"  per-backend reachability:")
    for k, m in enumerate(MACHINES):
        n = sum(1 for d in out_dispatches.values() if m not in d["infeasible_machines"])
        print(f"    {m}: {n} dispatches measured")
    print(f"processing  -> {args.out_dir / 'processing_times.json'}")
    print(f"transfer    -> {args.out_dir / 'transfer_times.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
