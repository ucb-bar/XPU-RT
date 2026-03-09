#!/usr/bin/env python3
"""
Batch-run IREE module benchmarks for a directory of *.vmfb + corresponding *.mlir.

For each <prefix>_benchmark.vmfb:
  - find matching <prefix>_benchmark.mlir
  - extract the first module entry from:  util.func public @<NAME>(
  - run:
      ./tools/iree-benchmark-module --module=<vmfb> --device=local-task
        --function=<NAME> --input=1xi32=1 --task_topology_cpu_ids=... --benchmark_repetitions=...
  - save full stdout/stderr to <out_dir>/logs/<basename>.log
  - parse the "real_time_mean" row, extract mean time + unit
  - write aggregated CSV to <out_dir>/results.csv

Usage example:
  ./run_vmfb_benchmarks.py \
    --input_dir ~/sched_vmfb/dronet \
    --out_dir   ~/sched_vmfb/out_dronet \
    --bench_tool ./tools/iree-benchmark-module \
    --device local-task \
    --input_spec 1xi32=1 \
    --task_topology_cpu_ids 0,1,2,3 \
    --benchmark_repetitions 10
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List


UTIL_FUNC_PUBLIC_RE = re.compile(r"^\s*util\.func\s+public\s+@([^( \t]+)\s*\(")
# Example line:
# BM_<name>/process_time/real_time_mean        0.421 ms ...
REAL_TIME_MEAN_RE = re.compile(
    r"^\s*BM_(?P<bench>.+?)/process_time/real_time_mean\s+"
    r"(?P<time>[0-9]*\.?[0-9]+)\s*(?P<unit>ns|us|ms|s)\b"
)

DISPATCH_ID_RE = re.compile(r"\$async_dispatch_(\d+)_")


@dataclass
class BenchResult:
    vmfb_path: Path
    mlir_path: Path
    module_name: str
    dispatch_id: Optional[int]
    mean_time: Optional[float]
    mean_unit: Optional[str]
    mean_time_ns: Optional[float]
    log_path: Path
    returncode: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--bench_tool", type=Path, default=Path("./tools/iree-benchmark-module"))
    p.add_argument("--device", type=str, default="local-task")
    p.add_argument("--input_spec", type=str, default="1xi32=1")
    p.add_argument("--task_topology_cpu_ids", type=str, default="0,1,2,3")
    p.add_argument("--benchmark_repetitions", type=int, default=10)

    p.add_argument("--glob_vmfb", type=str, default="*_benchmark.vmfb")
    p.add_argument("--glob_mlir", type=str, default="*_benchmark.mlir")

    # If you want to pass extra flags verbatim:
    p.add_argument("--extra_flag", action="append", default=[],
                   help="Extra flag to pass to iree-benchmark-module (repeatable). "
                        "Example: --extra_flag=--benchmark_min_time=0.5")
    return p.parse_args()


def find_first_public_util_func(mlir_path: Path) -> str:
    # Stream the file; avoid loading large mlir into memory.
    with mlir_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = UTIL_FUNC_PUBLIC_RE.match(line)
            if m:
                return m.group(1)
    raise RuntimeError(f"No 'util.func public @...' found in {mlir_path}")


def extract_dispatch_id_from_name(name: str) -> Optional[int]:
    m = DISPATCH_ID_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def unit_to_ns_factor(unit: str) -> float:
    if unit == "ns":
        return 1.0
    if unit == "us":
        return 1_000.0
    if unit == "ms":
        return 1_000_000.0
    if unit == "s":
        return 1_000_000_000.0
    raise ValueError(f"Unknown time unit: {unit}")


def parse_real_time_mean(output_text: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    # If the benchmark fails, the line may be missing.
    for line in output_text.splitlines():
        m = REAL_TIME_MEAN_RE.match(line)
        if m:
            t = float(m.group("time"))
            u = m.group("unit")
            t_ns = t * unit_to_ns_factor(u)
            return t, u, t_ns
    return None, None, None


def run_one(
    vmfb_path: Path,
    mlir_path: Path,
    module_name: str,
    args: argparse.Namespace,
    logs_dir: Path,
) -> BenchResult:
    dispatch_id = extract_dispatch_id_from_name(module_name)

    log_path = logs_dir / (vmfb_path.name + ".log")

    cmd: List[str] = [
        str(args.bench_tool),
        f"--module={str(vmfb_path)}",
        f"--device={args.device}",
        f"--function={module_name}",
        f"--input={args.input_spec}",
        f"--task_topology_cpu_ids={args.task_topology_cpu_ids}",
        f"--benchmark_repetitions={args.benchmark_repetitions}",
    ]
    cmd.extend(args.extra_flag)

    header = []
    header.append(datetime.now().isoformat())
    header.append("Running " + " ".join(cmd))
    header_text = "\n".join(header) + "\n"

    print("Running:", " ".join(cmd), flush=True)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    full_out = header_text + proc.stdout

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(full_out, encoding="utf-8")

    mean_time, mean_unit, mean_time_ns = parse_real_time_mean(proc.stdout)

    return BenchResult(
        vmfb_path=vmfb_path,
        mlir_path=mlir_path,
        module_name=module_name,
        dispatch_id=dispatch_id,
        mean_time=mean_time,
        mean_unit=mean_unit,
        mean_time_ns=mean_time_ns,
        log_path=log_path,
        returncode=proc.returncode,
    )


def match_mlir_for_vmfb(vmfb_path: Path, input_dir: Path) -> Path:
    # Expect: <stem>.vmfb  <-> <stem>.mlir
    # where stem includes "_benchmark" already, as in your example.
    mlir_candidate = input_dir / (vmfb_path.stem + ".mlir")
    if mlir_candidate.exists():
        return mlir_candidate
    raise RuntimeError(f"Missing MLIR for {vmfb_path.name}: expected {mlir_candidate}")


def write_csv(results: List[BenchResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dispatch_id",
            "module_name",
            "vmfb_path",
            "mlir_path",
            "mean_time",
            "mean_unit",
            "mean_time_ns",
            "returncode",
            "log_path",
        ])
        for r in results:
            w.writerow([
                "" if r.dispatch_id is None else r.dispatch_id,
                r.module_name,
                str(r.vmfb_path),
                str(r.mlir_path),
                "" if r.mean_time is None else f"{r.mean_time:.6g}",
                "" if r.mean_unit is None else r.mean_unit,
                "" if r.mean_time_ns is None else f"{r.mean_time_ns:.6f}",
                r.returncode,
                str(r.log_path),
            ])


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    bench_tool: Path = args.bench_tool.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: --input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 2
    if not bench_tool.exists():
        print(f"ERROR: --bench_tool not found: {bench_tool}", file=sys.stderr)
        return 2

    logs_dir = out_dir / "logs"
    out_csv = out_dir / "results.csv"

    vmfbs = sorted(input_dir.glob(args.glob_vmfb))
    if not vmfbs:
        print(f"ERROR: no vmfb matched {args.glob_vmfb} in {input_dir}", file=sys.stderr)
        return 2

    results: List[BenchResult] = []
    failures = 0

    for vmfb in vmfbs:
        try:
            mlir = match_mlir_for_vmfb(vmfb, input_dir)
            module_name = find_first_public_util_func(mlir)
            res = run_one(vmfb, mlir, module_name, args, logs_dir)
            results.append(res)

            if res.returncode != 0 or res.mean_time is None:
                failures += 1

            status = "OK"
            if res.returncode != 0:
                status = f"FAIL(rc={res.returncode})"
            elif res.mean_time is None:
                status = "NO_MEAN"
            print(f"[{status}] dispatch={res.dispatch_id} mean={res.mean_time}{res.mean_unit} vmfb={vmfb.name}")

        except Exception as e:
            failures += 1
            # Still record something minimal? For now: print and continue.
            print(f"[FAIL] vmfb={vmfb.name} error={e}", file=sys.stderr)

    write_csv(results, out_csv)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote logs: {logs_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


