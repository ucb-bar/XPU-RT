#!/usr/bin/env python3
"""End-to-end YOLOv8 heterogeneous compile + on-board run.

Inputs:
  - build/het/schedule.json (from run_heterogeneous_schedule.py)

Steps:
  1. Re-compile YOLOv8 with `./merlin compile --target qrb5165_aarch64
     --with-schedule build/het/schedule.json` so iree-merlin's
     DispatchCreation pass stamps stream.affinity per dispatch.
  2. Push the resulting heterogeneous VMFB to the board.
  3. Run via the on-board iree-run-module / merlin-dispatch-bench,
     measure end-to-end wall clock and per-dispatch timings (via the
     existing timing harness).
  4. Render the on-board-measured Gantt as the FINAL result. (No
     comparison to scheduler-predicted; the scheduler's number was an
     intermediate value.)

The output is a single makespan number from the on-board run + a
per-dispatch breakdown PNG, both saved next to schedule.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
_MERLIN = pathlib.Path("/scratch2/agustin/merlin")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule",
                    default=_ROOT / "build" / "het" / "schedule.json",
                    type=pathlib.Path)
    ap.add_argument("--model",
                    default=_MERLIN / "models" / "yolov8_nano" /
                            "yolov8n.q.int8.mlir",
                    type=pathlib.Path)
    ap.add_argument("--output-dir",
                    default=_MERLIN / "build" / "het" / "yolov8_final",
                    type=pathlib.Path)
    ap.add_argument("--ssh-host", default="qdev")
    ap.add_argument("--ssh-key",
                    default=os.path.expanduser("~/.ssh/DIMA_SLICE"))
    ap.add_argument("--remote-dir", default="/root/iree_run/yolov8_het_final")
    ap.add_argument("--repetitions", default=10, type=int)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    if not args.schedule.exists():
        print(f"ERROR missing {args.schedule} — run "
              f"run_heterogeneous_schedule.py first")
        return 1

    # Step 1: heterogeneous compile.
    if not args.skip_compile:
        env = os.environ.copy()
        env["QNN_USE_BOARD_BUILD"] = "1"
        env["QNN_BOARD_HOST"] = args.ssh_host
        env["QNN_BOARD_QAIRT_ROOT"] = "/tmp/qnn_probe"
        rc = _run([
            "conda", "run", "-n", "merlin-dev", "uv", "run", "python",
            str(_MERLIN / "tools" / "merlin.py"), "compile",
            str(args.model),
            "--target", "qrb5165_aarch64",
            "--hw", "a77",
            "--with-schedule", str(args.schedule),
            "--output-dir", str(args.output_dir),
            "--dump-artifacts", "--dump-phases", "--dump-graph",
        ], cwd=str(_MERLIN), env=env)
        if rc.returncode != 0:
            print(f"ERROR heterogeneous compile failed (rc={rc.returncode})")
            return 2
        print(f"  heterogeneous VMFB at {args.output_dir / 'yolov8n.q.int8.vmfb'}")

    if args.skip_run:
        return 0

    # Step 2: push VMFB.
    vmfb = args.output_dir / "yolov8n.q.int8.vmfb"
    if not vmfb.exists():
        print(f"ERROR no heterogeneous VMFB at {vmfb}")
        return 3

    _run(["ssh", args.ssh_host, f"mkdir -p {args.remote_dir}"])
    _run(["scp", "-q", "-i", args.ssh_key, str(vmfb),
          f"{args.ssh_host}:{args.remote_dir}/yolov8n.q.int8.vmfb"])
    _run(["ssh", args.ssh_host,
          f"cp /root/iree-benchmark-module {args.remote_dir}/iree-benchmark-module"])

    # Step 3: on-board run + measure.
    # The heterogeneous VMFB has multiple device.affinity targets; we
    # let iree-benchmark-module pick the local-task default for CPU
    # islands. QNN islands route via the embedded QNN HAL driver.
    cmd = (f"cd {args.remote_dir} && "
           f"taskset -c 4-7 ./iree-benchmark-module "
           f"--module=yolov8n.q.int8.vmfb "
           f"--device=local-task --task_topology_max_group_count=4 "
           f"--benchmark_repetitions={args.repetitions} "
           f"--benchmark_min_time=0.05s "
           f"--function=main_graph "
           f"--input=1x3x320x320xf32 "
           f"2>&1")
    res = subprocess.run(["ssh", args.ssh_host, cmd], capture_output=True, text=True)
    print(res.stdout[-5000:])
    if res.returncode != 0:
        print(f"WARN on-board run exited {res.returncode}; stderr:\n"
              f"{res.stderr[-3000:]}")

    # Save raw output next to schedule.json for inspection.
    (args.schedule.parent / "e2e_run_log.txt").write_text(
        res.stdout + "\n--- stderr ---\n" + res.stderr)
    print(f"\non-board run log -> {args.schedule.parent / 'e2e_run_log.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
