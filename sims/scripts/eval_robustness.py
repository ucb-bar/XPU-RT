#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robustness / generalization eval (task ROB / #63) — measures the overfit gap.

Runs a body-rate controller (a trained policy, or the analytic fixed baseline)
through :mod:`eval_bodyrate_tracking` under a sweep of OUT-OF-DISTRIBUTION plants
(off-nominal thrust-to-weight / moment-scale / motor-lag) and tabulates how much
tracking degrades. Point it at the non-DR and the DR policy to show whether
domain randomization closes the gap:

    <env_isaaclab python> sims/scripts/eval_robustness.py \
        --label nondr --checkpoint <nondr_run>/model_2999.pt
    <env_isaaclab python> sims/scripts/eval_robustness.py \
        --label dr    --checkpoint <dr_run>/model_2999.pt
    <env_isaaclab python> sims/scripts/eval_robustness.py --label fixed --controller fixed

This driver itself launches no Isaac; it shells out one eval per (condition) and
reads each run's ``summary.json``. Each condition is ~1 Isaac boot (~1-2 min).
"""

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "eval_bodyrate_tracking.py")
_PY = "/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python"
_SCRATCH = "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad"

# (name, {plant overrides}) — nominal first, then OOD plants outside the DR training band.
CONDITIONS = [
    ("nominal",        {}),
    ("low_authority",  {"--ood_t2w": "1.5"}),    # weaker thrust than trained
    ("high_authority", {"--ood_t2w": "2.4"}),    # stronger thrust than trained
    ("weak_moment",    {"--ood_mscale": "0.006"}),  # sluggish rotational authority
    ("laggy_motor",    {"--ood_tau": "0.05"}),   # 50 ms lag, beyond DR's 30 ms
]


def run_condition(label, controller, checkpoint, name, overrides):
    out = os.path.join(_SCRATCH, f"rob_{label}_{name}")
    cmd = [_PY, _EVAL, "--headless", "--controller", controller, "--out", out]
    if controller == "checkpoint":
        cmd += ["--checkpoint", checkpoint]
    for k, v in overrides.items():
        cmd += [k, v]
    print(f"[rob] {label}/{name}: {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, cwd="/scratch/agustin/projects/DIMA/train_out",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with open(os.path.join(out, "summary.json")) as f:
            s = json.load(f)
        return {"rms": s.get("combined_rms_rad_s"),
                "ss_wx": s.get("step_wx", {}).get("ss_err_frac"),
                "ss_wy": s.get("step_wy", {}).get("ss_err_frac"),
                "ss_wz": s.get("step_wz", {}).get("ss_err_frac")}
    except Exception as e:
        return {"rms": None, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="Short name for this policy (nondr/dr/fixed).")
    p.add_argument("--controller", choices=["checkpoint", "fixed"], default="checkpoint")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    results = {}
    for name, ov in CONDITIONS:
        results[name] = run_condition(args.label, args.controller, args.checkpoint, name, ov)

    nominal_rms = results["nominal"].get("rms")
    print(f"\n=== ROBUSTNESS: {args.label} ({args.controller}) ===")
    print(f"{'condition':16s} {'RMS(rad/s)':>11s} {'vs nominal':>11s}  ss[wx,wy,wz]")
    for name, _ in CONDITIONS:
        r = results[name]
        rms = r.get("rms")
        if rms is None:
            print(f"{name:16s} {'FAIL':>11s}  ({r.get('error','?')[:40]})")
            continue
        degr = f"{rms/nominal_rms:.2f}x" if nominal_rms else "-"
        ss = [r.get("ss_wx"), r.get("ss_wy"), r.get("ss_wz")]
        ss_s = ",".join("nan" if v is None else f"{v:.2f}" for v in ss)
        print(f"{name:16s} {rms:>11.4f} {degr:>11s}  [{ss_s}]")

    out_json = os.path.join(_SCRATCH, f"robustness_{args.label}.json")
    with open(out_json, "w") as f:
        json.dump({"label": args.label, "controller": args.controller,
                   "checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"\n[rob] wrote {out_json}")


if __name__ == "__main__":
    sys.exit(main())
