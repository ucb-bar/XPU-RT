#!/usr/bin/env python3
"""Per-dispatch profiling on a physical SpaceMiT K1 over SSH.

Why this exists alongside profile_remote.sh
-------------------------------------------
profile_remote.sh sweeps *topologies* (1,2,3,4 cores) and keeps only
`real_time_mean`. Two things the K1 work needs that it does not give:

1. **Core pinning that distinguishes clusters.** On the K1, cores 0-3 (cluster
   0) carry the IME extension and cores 4-7 (cluster 1) do not, and the two
   clusters have separate L2s. "One core" is therefore not one number -- it is
   at least two, and whether they differ is an experimental question rather
   than an assumption. The cluster is encoded in the `hw` label, because a
   measurement belongs to the place it was taken.

2. **A distribution, not a mean.** Scheduling nominal service time off a mean
   throws away exactly what deadline analysis needs. Every repetition row is
   kept, and median / p90 / p99 / min / max / CV are reported. The scheduler
   gets the median; the tail stays available.

Outputs, deliberately two files:

  results.csv    the existing IREE-shaped schema, byte-compatible with what
                 profile_loader.py already parses -- median in `mean_time`, so
                 nothing downstream needs to change.
  profile.jsonl  the richer record (all samples, percentiles, cpu ids,
                 cluster, board clock) that the CSV has no room for.

The CSV schema is not extended. Adding columns there would mean touching every
consumer; a sidecar costs nothing to ignore.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# "BM_<name>/process_time/real_time   79.9 us  85.0 us  8945 ..."
_ROW = re.compile(
    r"^\s*BM_(?P<bench>.+?)/process_time/real_time\s+"
    r"(?P<time>[0-9]*\.?[0-9]+)\s*(?P<unit>ns|us|ms|s)\b"
)
# module_<model>$async_dispatch_<N>_embedded_elf_riscv_64_benchmark.vmfb
_DISPATCH = re.compile(r"\$async_dispatch_(\d+)_")
_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def sh(host: str, cmd: str, timeout: int = 900) -> str:
    r = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh {host}: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def topo_tag(n_cores: int) -> str:
    """XPU-RT keys profiles off combination *size*, not core ids.

    workload_factory.topo_tag_for_combination turns a 1-core combination into
    'topo_0' regardless of which physical core it is. So cluster 1 pinned to
    core 4 is still 'topo_0' here, and the cluster lives in the hw label
    instead. Diverging from that convention would make the profile unfindable.
    """
    return "topo_" + "_".join(str(i) for i in range(n_cores))


def parse_samples(stdout: str) -> dict[str, list[float]]:
    """Every repetition row, in ms, keyed by benchmark name."""
    out: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        name = m.group("bench")
        # skip the aggregate rows benchmark appends (_mean/_median/_stddev/_cv)
        if name.endswith(("_mean", "_median", "_stddev", "_cv")):
            continue
        out.setdefault(name, []).append(
            float(m.group("time")) * _TO_MS[m.group("unit")])
    return out


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def profile_one(host: str, remote_root: str, bench_tool: str, zip_path: Path,
                cpu_ids: str, reps: int, model: str, basename: str,
                hw_label: str, target: str, clock_mhz: float,
                profile_root: Path, dry_run: bool = False) -> Path:
    stage = Path(tempfile.mkdtemp(prefix="k1prof_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(stage)
        rdir = f"{remote_root}/bench/{model}_{hw_label}"
        sh(host, f"rm -rf {rdir} && mkdir -p {rdir}")
        tar = subprocess.run(["tar", "czf", "-", "-C", str(stage), "."],
                             capture_output=True, check=True)
        subprocess.run(["ssh", host, f"tar xzf - -C {rdir}"],
                       input=tar.stdout, check=True, timeout=1800)

        vmfbs = sorted(p.name for p in stage.glob("*.vmfb"))
        records = []
        for name in vmfbs:
            m = _DISPATCH.search(name)
            if not m:
                # _encoding_N modules are data-tiling helpers, not graph
                # dispatches; they have no id in the dispatch graph.
                continue
            did = int(m.group(1))
            # single-quote for the local shell, escape $ for the remote one
            esc = name.replace("$", r"\$")
            cmd = (f"cd {rdir} && {bench_tool} --module=\"{esc}\" "
                   f"--device=local-task --task_topology_cpu_ids={cpu_ids} "
                   f"--benchmark_repetitions={reps}")
            if dry_run:
                print("  would run:", cmd)
                continue
            try:
                out = sh(host, cmd)
            except RuntimeError as e:
                print(f"  WARN dispatch_{did}: {e}", file=sys.stderr)
                continue
            samples = parse_samples(out)
            if not samples:
                print(f"  WARN dispatch_{did}: no timing rows parsed",
                      file=sys.stderr)
                continue
            bench_name, xs = next(iter(samples.items()))
            med = statistics.median(xs)
            rec = {
                "model": model, "basename": basename, "dispatch_id": did,
                "module_name": bench_name, "hw_label": hw_label,
                "target": target, "cpu_ids": cpu_ids,
                "cluster": 0 if int(cpu_ids.split(",")[0]) < 4 else 1,
                "n_cores": len(cpu_ids.split(",")),
                "reps": len(xs), "clock_mhz": clock_mhz,
                "samples_ms": xs,
                "median_ms": med,
                "mean_ms": statistics.fmean(xs),
                "min_ms": min(xs), "max_ms": max(xs),
                "p90_ms": pct(xs, 90), "p99_ms": pct(xs, 99),
                "stdev_ms": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                "cv_pct": (statistics.pstdev(xs) / med * 100.0)
                          if len(xs) > 1 and med else 0.0,
                "cycles_est": int(round(med * 1e-3 * clock_mhz * 1e6)),
            }
            records.append(rec)
            print(f"  dispatch_{did:<3} median={med*1000:9.2f} us  "
                  f"p99={rec['p99_ms']*1000:9.2f} us  cv={rec['cv_pct']:.2f}%")

        if dry_run or not records:
            return Path()

        out_dir = (profile_root / hw_label / target / model / basename
                   / topo_tag(len(cpu_ids.split(","))))
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "results.csv", records)
        _write_jsonl(out_dir / "profile.jsonl", records)
        print(f"  -> {out_dir}/results.csv  ({len(records)} dispatches)")
        return out_dir
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# Exactly the columns scripts/export_profile_db_to_results_csv.py defines.
CSV_COLUMNS = ["dispatch_id", "module_name", "vmfb_path", "mlir_path",
               "mean_time", "mean_unit", "mean_time_ns", "returncode",
               "log_path", "source", "op", "shape", "cycles"]


def _write_csv(path: Path, records: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in sorted(records, key=lambda x: x["dispatch_id"]):
            # `mean_time` carries the MEDIAN on purpose: it is the robust
            # statistic the scheduler should use as nominal service time. The
            # column name is fixed by the existing schema.
            w.writerow([r["dispatch_id"], r["module_name"], "", "",
                        f"{r['median_ms']:.6f}", "ms",
                        f"{r['median_ms']*1e6:.3f}", 0, "",
                        "k1_measured", "", "", r["cycles_est"]])


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in sorted(records, key=lambda x: x["dispatch_id"]):
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("MODELBLASTER_K1_HOST", "k1"))
    ap.add_argument("--remote-root",
                    default=os.environ.get("MODELBLASTER_K1_REMOTE_ROOT", "/root/mb_k1"))
    ap.add_argument("--bench-tool", default="/root/mb_k1/tools/iree-benchmark-module")
    ap.add_argument("--vmfb-root", default="gen/vmfb")
    ap.add_argument("--profile-root", default="gen/profile")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--models", default="mlp,dronet")
    ap.add_argument("--hw", default="RVV,scalar", help="compiled variants to profile")
    ap.add_argument("--cpu-ids", default="0",
                    help="comma-separated physical core ids to pin to")
    ap.add_argument("--hw-label-suffix", default="",
                    help="appended to the hw label, e.g. '_c1' for cluster 1")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--clock-mhz", type=float, default=1600.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    vmfb_root = Path(a.vmfb_root)
    profile_root = Path(a.profile_root)
    rc = 0
    for model in [m for m in a.models.split(",") if m]:
        for hw in [h for h in a.hw.split(",") if h]:
            base = vmfb_root / model / a.target / hw
            zips = sorted(base.glob("*/*_benchmarks.zip"))
            if not zips:
                print(f"SKIP {model}/{hw}: no *_benchmarks.zip under {base}",
                      file=sys.stderr)
                rc = 1
                continue
            for z in zips:
                basename = z.parent.name
                label = f"{hw}{a.hw_label_suffix}"
                print(f"[{model}/{label}] {basename}  cpus={a.cpu_ids} "
                      f"reps={a.reps}")
                try:
                    profile_one(a.host, a.remote_root, a.bench_tool, z,
                                a.cpu_ids, a.reps, model, basename, label,
                                a.target, a.clock_mhz, profile_root, a.dry_run)
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED {model}/{label}: {e}", file=sys.stderr)
                    rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
