"""Seed CostTable from on-board measurements taken on QRB5165 / QAIRT 2.45.

CONTRACT: every entry in this file is the result of a real on-board run.
NO estimates, NO extrapolations. If a row is missing, the scheduler MUST
fail loudly rather than silently fall back. New measurements get added by
running scripts/profile_qnn_kernels.py and merging the JSON output.

The two YOLOv8-stem-conv rows below are kept because they were directly
measured on board (qnn-net-run --num_inferences 50 against the real
context binary). Everything else is left blank — callers that need
trunk/head/CPU costs MUST drive the profiler first.
"""

from __future__ import annotations

import pathlib

from .cost_table import CostTable


def seed() -> CostTable:
    """Return a CostTable containing only the rows we actually measured.

    Profiling expansion happens out-of-band via
    scripts/profile_qnn_kernels.py, which writes JSON merge-deltas into
    the same file path. This seed is the authoritative on-board-measured
    starting point.
    """
    t = CostTable(
        device="qrb5165",
        qairt_sdk="2.45.0.260326",
        measured_at="2026-05-07T03:34:11Z",
    )

    # ---- Execute costs (measured) -------------------------------------
    # YOLOv8 stem conv (1×320×320×3 → 1×160×160×16, k3 s2 g1):
    # both backends measured against actual on-board context binaries.
    stem_sig = "1x320x320x3->1x160x160x16,g1,k3,s2"
    t.execute[f"Conv2d@{stem_sig}@uint8::HTA::0"] = {
        "mean_us": 5910.0, "p50_us": 5800.0, "p99_us": 7200.0,
        "iters": 50, "extrapolated": False,
        "source": "qnn-net-run yolov8_stem_nhwc_int8 hta /tmp/yolov8_stem.qnn-ctx",
    }
    t.execute[f"Conv2d@{stem_sig}@fp16::GPU::0"] = {
        "mean_us": 12515.0, "p50_us": 12300.0, "p99_us": 24000.0,
        "iters": 50, "extrapolated": False,
        "source": "qnn-net-run yolov8_stem_fp16_gpu adreno /tmp/yolov8_stem_fp16.qnn-ctx",
    }

    # ---- Init / one-time setup (measured) -----------------------------
    t.init["HTA"] = {
        "mean_us": 20731.0, "iters": 1,
        "source": "qnn-net-run init stats yolov8_stem_nhwc_int8.qnn-ctx",
    }
    t.init["GPU"] = {
        "mean_us": 6983.0, "iters": 1,
        "source": "qnn-net-run init stats (createFromBinary) yolov8_stem_fp16_gpu.qnn-ctx",
    }
    # CPU init NOT measured — must be filled by profiler before scheduling
    # any CPU island. Leaving the row absent forces the scheduler to fail
    # loudly ("KeyError: no init cost for CPU") rather than amortise zero.

    # ---- Memcpy bandwidth ---------------------------------------------
    # Single point measured on A77 inside the same process: 7.95 GB/s with
    # ~zero setup. Cross-process FastRPC/ION setup is NOT yet profiled.
    # Leaving cross-machine entries absent — profile_transfers.py owes them.
    t.memcpy["CPU__CPU"] = {
        "bytes_per_us_mean": 7950.0,
        "fixed_overhead_us": 0.0,
        "iters": 100,
        "source": "bridge_bench memcpy uint8 410KB on A77 taskset 4-7",
    }

    # ---- Rescale (single dtype, qp delta) -----------------------------
    # Measured on A77, 410KB input, 30 iter mean.
    t.rescale["uint8"] = {
        "us_per_elem": 0.041e-3,
        "fixed_overhead_us": 0.0,
        "iters": 30,
        "source": "bridge_bench Quant fp32->u8 (which is qp-shift after dequant)",
    }
    # fp16 / fp32 rescale NOT measured — leaving absent.

    # ---- Dequant + Quant ----------------------------------------------
    # Measured uint8->fp32 (Dequantize): 0.055 ns/elem on A77 at 410K.
    # Measured fp32->uint8 (Quantize):   0.042 ns/elem on A77 at 410K.
    # Both single-point — coefficient form is interim until linear-fit
    # sweep populates real (us_per_elem, fixed_overhead).
    t.dequant_quant["uint8__fp32"] = {
        "us_per_elem": 0.055e-3, "fixed_overhead_us": 0.0,
        "iters": 30, "source": "bridge_bench Dequant uint8->fp32 410KB",
    }
    t.dequant_quant["fp32__uint8"] = {
        "us_per_elem": 0.042e-3, "fixed_overhead_us": 0.0,
        "iters": 30, "source": "bridge_bench Quant fp32->uint8 410KB",
    }
    # uint8↔fp16 NOT measured directly — must be profiled before any
    # HTA↔GPU edge in the schedule can be priced.

    return t


if __name__ == "__main__":
    table = seed()
    out = pathlib.Path(__file__).parent / "qrb5165_costs.json"
    table.save(out)
    print(f"wrote {out}")
    print(f"  {len(table.execute)} execute rows (only measured)")
    print(f"  {len(table.init)} init rows")
    print(f"  {len(table.memcpy)} memcpy rows")
    print(f"  {len(table.rescale)} rescale rows")
    print(f"  {len(table.dequant_quant)} dequant_quant rows")
    missing = []
    if "CPU" not in table.init: missing.append("init.CPU")
    for m in ("HTA", "GPU"):
        if f"CPU__{m}" not in table.memcpy: missing.append(f"memcpy.CPU__{m}")
        if f"{m}__CPU" not in table.memcpy: missing.append(f"memcpy.{m}__CPU")
    for d in ("uint8__fp16", "fp16__uint8"):
        if d not in table.dequant_quant: missing.append(f"dequant_quant.{d}")
    if missing:
        print(f"  MISSING (profiler must populate): {missing}")
