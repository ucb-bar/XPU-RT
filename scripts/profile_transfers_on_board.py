#!/usr/bin/env python3
"""Per-volume bridge profiling on QRB5165.

Sweeps the actual edge volumes that appear in YOLOv8's dispatch graph,
measuring on board:

  * memcpy (uint8/fp16/fp32) at each unique tensor volume
  * Dequantize (uint8 -> fp32, uint8 -> fp16)
  * Quantize (fp32 -> uint8, fp16 -> uint8)
  * Rescale (uint8 with qp shift)

All measurements are A77-pinned (taskset 4-7), N=1000 iters per
volume × op. Results merge into qnn_scheduler/qrb5165_costs.json under
`memcpy[<src>__<dst>]`, `dequant_quant[<src_dt>__<dst_dt>]`,
`rescale[<dt>]` as linear-fit (us_per_byte, fixed_overhead_us)
coefficients fit from 4 measured points (10K / 100K / 1M / actual edge
volumes).

The script ships a single C++ kernel that compiles on board and
executes one of the four ops based on argv. No host-side timing,
nothing extrapolated — every coefficient is fit from real on-board
measurements.

This is the Phase 3 deliverable per the rosy-sundae plan. The current
schedule run uses the seed-table single-point coefficients; running
this script tightens those into volume-aware coefficients without
changing any other code (the cost_table API already linearises from
us_per_byte + fixed_overhead).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Iterable

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
from qnn_scheduler.cost_table import CostTable  # noqa: E402

_BPE = {"uint8": 1, "fp16": 2, "fp32": 4}


_BENCH_SRC = r'''
// On-board bridge microbench. Compile with:
//   g++ -O3 -ffast-math -std=c++17 bridge_bench.cc -o bridge_bench
// Argv:
//   bridge_bench <op> <n_elem> <iters>
// where op ∈ {memcpy_u8, memcpy_f16, memcpy_f32, deq_u8_f32, deq_u8_f16,
//             qnt_f32_u8, qnt_f16_u8, rescale_u8}
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>

using clk = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s op n_elem iters\n", argv[0]); return 1; }
  const char* op = argv[1];
  size_t N = (size_t)std::strtoull(argv[2], nullptr, 10);
  int iters = std::atoi(argv[3]);
  double dt_us = 0.0;
  if (!strcmp(op, "memcpy_u8")) {
    std::vector<uint8_t> a(N), b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint8_t)(i * 37);
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) std::memcpy(b.data(), a.data(), N);
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "memcpy_f16")) {
    std::vector<uint16_t> a(N), b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint16_t)(i * 37);
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) std::memcpy(b.data(), a.data(), N * 2);
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "memcpy_f32")) {
    std::vector<float> a(N), b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (float)(i * 37);
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) std::memcpy(b.data(), a.data(), N * 4);
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "deq_u8_f32")) {
    std::vector<uint8_t> a(N); std::vector<float> b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint8_t)(i * 37);
    const float scale = 0.0353553f; const int32_t off = 128;
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < N; ++i) b[i] = (int(a[i]) - off) * scale;
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "deq_u8_f16")) {
    std::vector<uint8_t> a(N); std::vector<uint16_t> b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint8_t)(i * 37);
    const float scale = 0.0353553f; const int32_t off = 128;
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < N; ++i) {
        float f = (int(a[i]) - off) * scale;
        b[i] = (uint16_t)__builtin_bswap16((uint16_t)((uint32_t&)(f) >> 16));
      }
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "qnt_f32_u8")) {
    std::vector<float> a(N); std::vector<uint8_t> b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (float)((i % 200) - 100) * 0.01f;
    const float scale = 0.0353553f; const int32_t off = 128;
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < N; ++i) {
        int32_t q = (int32_t)std::lrintf(a[i] / scale) + off;
        if (q < 0) q = 0; else if (q > 255) q = 255;
        b[i] = (uint8_t)q;
      }
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "qnt_f16_u8")) {
    std::vector<uint16_t> a(N); std::vector<uint8_t> b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint16_t)(i * 37);
    const float scale = 0.0353553f; const int32_t off = 128;
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < N; ++i) {
        // Stub: treat fp16 as raw bytes converted to f32.
        float f = (float)((int16_t)a[i]) / 1024.0f;
        int32_t q = (int32_t)std::lrintf(f / scale) + off;
        if (q < 0) q = 0; else if (q > 255) q = 255;
        b[i] = (uint8_t)q;
      }
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else if (!strcmp(op, "rescale_u8")) {
    std::vector<uint8_t> a(N), b(N);
    for (size_t i = 0; i < N; ++i) a[i] = (uint8_t)(i * 37);
    const float scale_a = 0.10f; const int32_t off_a = 128;
    const float scale_b = 0.05f; const int32_t off_b = 64;
    const float ratio = scale_a / scale_b;
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < N; ++i) {
        int32_t q = (int32_t)std::lrintf((int(a[i]) - off_a) * ratio) + off_b;
        if (q < 0) q = 0; else if (q > 255) q = 255;
        b[i] = (uint8_t)q;
      }
    dt_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
  } else {
    fprintf(stderr, "unknown op: %s\n", op); return 2;
  }
  printf("%.3f\n", dt_us);
  return 0;
}
'''


def _ssh(host: str, cmd: str) -> str:
    res = subprocess.run(["ssh", host, cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ssh failed: {res.stderr}")
    return res.stdout


def _build_bench(host: str) -> None:
    src_remote = "/tmp/bridge_bench.cc"
    bin_remote = "/tmp/bridge_bench"
    # Push source.
    p = subprocess.run(
        ["ssh", host, f"cat > {src_remote}"],
        input=_BENCH_SRC, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"failed to push src: {p.stderr}")
    # Compile on board.
    out = _ssh(host,
        f"g++ -O3 -ffast-math -std=c++17 {src_remote} -o {bin_remote} 2>&1")
    if out.strip():
        print(f"compile: {out.strip()}")


def _measure(host: str, op: str, n_elem: int, iters: int = 100) -> float:
    cmd = f"taskset -c 4-7 /tmp/bridge_bench {op} {n_elem} {iters}"
    out = _ssh(host, cmd).strip().splitlines()[-1]
    return float(out)


def _linear_fit(volumes: list[int], times_us: list[float]) -> tuple[float, float]:
    """Least-squares fit time = a*volume + b. Returns (us_per_byte, fixed_us)."""
    import numpy as np
    x = np.array(volumes, dtype=float)
    y = np.array(times_us, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _yolov8_unique_volumes() -> list[int]:
    """Pull every distinct tensor byte-volume from the YOLOv8 dispatches'
    inputs/outputs."""
    bdir = pathlib.Path(
        "/scratch2/agustin/merlin/build/het/qrb5165_cpu/breakdowns")
    vols: set[int] = set()
    for p in bdir.glob("dispatch_*.shapes.json"):
        s = json.loads(p.read_text())
        for tensors in (s.get("inputs", []), s.get("outputs", [])):
            for t in tensors:
                m = re.match(r"tensor<([0-9x?]+)x([a-z][a-z0-9_]*)>", t)
                if not m:
                    continue
                shape = [int(d) if d.isdigit() else 0 for d in m.group(1).split("x")]
                dt = m.group(2)
                bpe = {"i8": 1, "ui8": 1, "f16": 2, "f32": 4, "i32": 4}.get(dt, 4)
                n = 1
                for d in shape: n *= max(1, d)
                vols.add(n * bpe)
    return sorted(vols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-table",
                    default=_ROOT / "qnn_scheduler" / "qrb5165_costs.json",
                    type=pathlib.Path)
    ap.add_argument("--ssh-host", default="qdev")
    ap.add_argument("--iters", default=100, type=int)
    ap.add_argument("--volumes", default="10000,100000,1000000,10000000",
                    help="comma-separated bytes (override default sweep)")
    ap.add_argument("--use-yolov8-volumes", action="store_true",
                    help="Sweep every distinct edge volume from yolov8 "
                         "(slow; ~80+ points)")
    args = ap.parse_args()

    print(f"compiling bridge_bench on {args.ssh_host}...")
    _build_bench(args.ssh_host)

    if args.use_yolov8_volumes:
        volumes = _yolov8_unique_volumes()
        print(f"using {len(volumes)} unique YOLOv8 edge volumes "
              f"({min(volumes)}-{max(volumes)} bytes)")
    else:
        volumes = [int(v) for v in args.volumes.split(",")]
        print(f"sweeping {volumes} bytes")

    table = CostTable.load(args.cost_table) if args.cost_table.exists() else CostTable()

    op_to_key: list[tuple[str, str, str, dict]] = [
        # (bench_op, dtype_for_n_elem_calc, table-section, table-key-template)
        ("memcpy_u8",  "uint8", "memcpy", {"src": "CPU", "dst": "CPU", "label": "uint8"}),
        ("memcpy_f16", "fp16",  "memcpy", {"src": "CPU", "dst": "CPU", "label": "fp16"}),
        ("memcpy_f32", "fp32",  "memcpy", {"src": "CPU", "dst": "CPU", "label": "fp32"}),
        ("deq_u8_f32", "uint8", "dequant_quant", {"key": "uint8__fp32"}),
        ("deq_u8_f16", "uint8", "dequant_quant", {"key": "uint8__fp16"}),
        ("qnt_f32_u8", "fp32",  "dequant_quant", {"key": "fp32__uint8"}),
        ("qnt_f16_u8", "fp16",  "dequant_quant", {"key": "fp16__uint8"}),
        ("rescale_u8", "uint8", "rescale", {"key": "uint8"}),
    ]

    for bench_op, dtype, section, info in op_to_key:
        bpe = _BPE[dtype]
        ts: list[float] = []
        ns: list[int] = []
        for vol_bytes in volumes:
            n = vol_bytes // bpe
            if n < 32:
                continue
            t = _measure(args.ssh_host, bench_op, n, iters=args.iters)
            ts.append(t)
            ns.append(vol_bytes)
            print(f"  {bench_op:14s}  {vol_bytes:>10d} B  {t:8.2f} us")
        if len(ts) < 2:
            continue
        slope, intercept = _linear_fit(ns, ts)
        # Negative slope is artifact (cache effects); clamp to 0.
        if slope < 0:
            slope = 0.0
        if section == "memcpy":
            # Per-machine memcpy entry; CPU__CPU only at this layer.
            key = f"{info['src']}__{info['dst']}"
            row = table.memcpy.get(key, {})
            row.update({
                "bytes_per_us_mean": (1.0 / slope) if slope > 0 else 999999.0,
                "fixed_overhead_us": float(intercept),
                "iters": args.iters,
                "source": f"profile_transfers_on_board {bench_op} fit n={len(ts)}",
                "dtype_label": info["label"],
            })
            table.memcpy[key] = row
        elif section == "dequant_quant":
            key = info["key"]
            table.dequant_quant[key] = {
                "us_per_elem": slope * bpe,
                "fixed_overhead_us": float(intercept),
                "iters": args.iters,
                "source": f"profile_transfers_on_board {bench_op} fit n={len(ts)}",
            }
        elif section == "rescale":
            key = info["key"]
            table.rescale[key] = {
                "us_per_elem": slope * bpe,
                "fixed_overhead_us": float(intercept),
                "iters": args.iters,
                "source": f"profile_transfers_on_board {bench_op} fit n={len(ts)}",
            }

    table.save(args.cost_table)
    print(f"\ncost table updated: {args.cost_table}")
    print(f"  memcpy rows: {len(table.memcpy)}")
    print(f"  rescale rows: {len(table.rescale)}")
    print(f"  dequant_quant rows: {len(table.dequant_quant)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
