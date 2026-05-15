#!/usr/bin/env python3
"""Drive tools/board_roundtrip.py once per QRB5165 target.

For each target T in {qrb5165_aarch64, qrb5165_qnn_gpu, qrb5165_qnn_hta},
push the per-dispatch VMFBs from `build/het/<T>/breakdowns/` to qdev,
benchmark each one against the appropriate IREE HAL device:

    qrb5165_aarch64  -> --device=local-task --task_topology_cpu_ids=4,5,6,7
    qrb5165_qnn_gpu  -> --device=qnn?backend=gpu
    qrb5165_qnn_hta  -> --device=qnn?backend=hta

Each invocation produces `build/het/<T>/breakdowns/profiled_manifest.json`.
Dispatches whose VMFB is a placeholder (no real ctxbin) WILL fail at run
time — board_roundtrip records that as a missing sample, which the next
phase translates into an explicit `infeasible: true` row in the cost
table.

This is a wrapper, not a re-implementation: the underlying tool is
`tools/board_roundtrip.py` (in the merlin submodule), which already
handles the CPU per-dispatch profile loop. This script just supplies the
right --device per target.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve()
_XPU_RT_ROOT = _HERE.parent.parent
_MERLIN = pathlib.Path("/scratch2/agustin/merlin")


_TARGETS = {
    "qrb5165_aarch64": {
        "build_dir": "build/het/qrb5165_cpu",
        "remote_dir": "/root/iree_run/yolov8_het_cpu",
        "device": None,  # uses --task-topology-cpu-ids
        "extra": ["--task-topology-cpu-ids", "4,5,6,7"],
    },
    "qrb5165_qnn_gpu": {
        "build_dir": "build/het/qrb5165_gpu",
        "remote_dir": "/root/iree_run/yolov8_het_gpu",
        "device": "qnn?backend=gpu",
        "extra": [],
    },
    "qrb5165_qnn_hta": {
        "build_dir": "build/het/qrb5165_hta",
        "remote_dir": "/root/iree_run/yolov8_het_hta",
        "device": "qnn?backend=hta",
        "extra": [],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", action="append",
                    choices=list(_TARGETS),
                    help="Repeatable. If omitted, all targets are profiled.")
    ap.add_argument("--ssh-host", default="qdev")
    ap.add_argument("--ssh-key",
                    default=os.path.expanduser("~/.ssh/DIMA_SLICE"))
    ap.add_argument("--repetitions", default=5, type=int)
    ap.add_argument("--push", action="store_true",
                    help="Push per-dispatch VMFBs + on-board iree-benchmark-module "
                         "before benchmarking (skip if already pushed).")
    args = ap.parse_args()

    targets = args.target or list(_TARGETS)
    for tname in targets:
        tcfg = _TARGETS[tname]
        bdir = _MERLIN / tcfg["build_dir"]
        if not (bdir / "breakdowns" / "manifest.json").exists():
            print(f"[skip] {tname}: missing {bdir}/breakdowns/manifest.json — "
                  f"run tools/breakdown_vmfb.py first")
            continue

        rdir = tcfg["remote_dir"]

        if args.push:
            # Manual push: scp the per-dispatch VMFBs + shapes + manifest.
            # Reuse on-board /root/iree-benchmark-module instead of cross-build.
            print(f"[push] {tname} -> {args.ssh_host}:{rdir}")
            subprocess.run(["ssh", args.ssh_host,
                            f"mkdir -p {rdir}/breakdowns"], check=True)
            for ext in ("*.vmfb", "*.shapes.json", "manifest.json"):
                files = list((bdir / "breakdowns").glob(ext))
                if files:
                    subprocess.run(
                        ["scp", "-q", "-i", args.ssh_key,
                         *[str(f) for f in files],
                         f"{args.ssh_host}:{rdir}/breakdowns/"], check=True)
            subprocess.run(
                ["ssh", args.ssh_host,
                 f"cp /root/iree-benchmark-module {rdir}/iree-benchmark-module"],
                check=True)

        cmd = [
            "conda", "run", "-n", "merlin-dev", "uv", "run", "python",
            str(_MERLIN / "tools" / "board_roundtrip.py"),
            "--output-dir", str(bdir),
            "--ssh-host", args.ssh_host,
            "--ssh-key", args.ssh_key,
            "--runtime-bin", "/tmp/skipped",  # skip-push set; not needed
            "--remote-dir", rdir,
            "--repetitions", str(args.repetitions),
            "--profile-key", tname,
            "--skip-push",
        ]
        if tcfg["device"]:
            cmd += ["--device", tcfg["device"]]
        cmd += tcfg["extra"]
        print(f"[bench] {tname}: {' '.join(shlex.quote(c) for c in cmd[-12:])}")
        rc = subprocess.run(cmd, cwd=str(_MERLIN))
        if rc.returncode != 0:
            print(f"[warn] {tname} board_roundtrip exited {rc.returncode} "
                  f"(some dispatches may have failed; that's expected for "
                  f"placeholder VMFBs — they'll be marked infeasible)")

    print("\nDONE. Per-target manifests:")
    for tname in targets:
        bdir = _MERLIN / _TARGETS[tname]["build_dir"]
        pm = bdir / "breakdowns" / "profiled_manifest.json"
        if pm.exists():
            print(f"  {tname}: {pm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
