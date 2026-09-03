#!/usr/bin/env python3
"""Convert an fq FPGA uartlog into an IREE-shape results.csv the XPU-RT
scheduler can ingest.

The modelblaster harness prints the profile block straight into the uartlog:
    === MODELBLASTER_PROFILE_BEGIN [<model>] ===
    dispatch_id,name,op,shape,cycles
    ...
    === MODELBLASTER_PROFILE_END [<model>] ===
so an FPGA run submitted through the AWS `fq` queue carries everything
profile_writer needs. (modelblaster.validation.firesim_runner does this
inline, but its queue path targets the local garden queue, not AWS fq.)

Usage:
  uartlog_to_profile.py --uartlog U --model vint --quant int8 \
      --backend gemmini_q31 --cpu firesim_rocket_saturn \
      --out-root <zcs>/gen/profile/<tag> [--clock-mhz 60]
"""
import argparse, os, subprocess, sys, re

# modelblaster lives inside the zephyr-chipyard-sw submodule. Resolve its
# checkout path from .gitmodules (the submodule NAME is the stable identifier;
# the path is whatever .gitmodules says) rather than hardcoding an absolute
# path -- this file used to carry one, which made it unusable from any clone
# but the author's.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _zcs_root() -> str:
    try:
        p = subprocess.run(
            ["git", "-C", _ROOT, "config", "-f", ".gitmodules",
             "--get", "submodule.zephyr-chipyard-sw.path"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        p = ""
    return os.path.join(_ROOT, p or "zephyr-chipyard-sw")


sys.path.insert(0, _zcs_root())
from modelblaster.validation.runner_common import parse_profile
from modelblaster.pipeline.profile_writer import write_profile, ProfileMeta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uartlog", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--cpu", default="firesim_rocket_saturn")
    ap.add_argument("--source", default="firesim")
    ap.add_argument("--cores", default="0")
    ap.add_argument("--clock-mhz", type=float, default=60.0)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", default=None, help="model tag inside the uartlog")
    a = ap.parse_args()

    text = open(a.uartlog, errors="ignore").read()
    recs = parse_profile(text, a.tag)
    if not recs:
        # single-model harness omits the [model] tag
        recs = parse_profile(text)
    if not recs:
        sys.exit(f"no MODELBLASTER_PROFILE block in {a.uartlog}")
    tot = sum(int(r["cycles"]) for r in recs)
    print(f"  parsed {len(recs)} dispatches, {tot:,} cycles "
          f"({tot/a.clock_mhz/1e6:.2f} s at {a.clock_mhz:.0f} MHz)")
    meta = ProfileMeta(model=a.model, quant=a.quant, backend=a.backend,
                       cores=tuple(int(c) for c in a.cores.split(",")),
                       source=a.source, cpu=a.cpu, clock_mhz=a.clock_mhz)
    path = write_profile(recs, meta, a.out_root)
    print(f"  wrote {path}")

if __name__ == "__main__":
    main()
