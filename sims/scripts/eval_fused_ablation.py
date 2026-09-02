"""Sensor-aggregation ablation — ACCURACY side (task #62, complements the compute side).

Flies the SAME trained FusedSensorNet through eval_forest_nav_fused under a sweep
of sensor SUBSETS (zero-skipping the masked modalities) and tabulates closed-loop
flight quality (success rate, mean lateral offset) vs subset. Pair it with
``vitfly/models/ablate_fused_compute.py`` (MACs/latency vs subset) to get the full
quality-vs-cost picture that feeds the Phase-2 Pareto and answers "which sensors
actually matter."

    <env_isaaclab py> sims/scripts/eval_fused_ablation.py \
        --weights <fused_bc>/best.pt --trail straight --episodes 6

This driver launches no Isaac itself — it shells one eval per subset and reads
each run's JSON. Each subset = ~1 Isaac boot.
"""

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "eval_forest_nav_fused.py")
_PY = "/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python"
_SCRATCH = "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad"

# (name, modalities-to-zero-skip). Mirrors ablate_fused_compute.SUBSETS where possible.
SUBSETS = [
    ("full",              ""),
    ("no_camera",         "front_grey"),
    ("no_tof",            "tof_cross"),
    ("no_downstack",      "down_tof,optical_flow,baro"),
    ("state_only",        "front_grey,tof_cross"),
    ("camera+tof_only",   "down_tof,optical_flow,baro,quat,body_rates"),
    ("no_flow",           "optical_flow"),
    ("no_baro",           "baro"),
]


def run(weights, trail, episodes, name, mask_off):
    out = os.path.join(_SCRATCH, f"ablacc_{trail}_{name}.json")
    cmd = [_PY, _EVAL, "--headless", "--weights", weights, "--trail", trail,
           "--episodes", str(episodes), "--out", out]
    if mask_off:
        cmd += ["--mask_off", mask_off]
    print(f"[abl] {name}: mask_off='{mask_off or 'none'}'", flush=True)
    subprocess.run(cmd, cwd="/scratch/agustin/projects/DIMA/train_out",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        a = json.load(open(out))["agg"]
        return {"success": a["success_rate"], "offset": a["mean_offset"],
                "progress": a["mean_progress"], "jerk": a["mean_jerk"]}
    except Exception as e:
        return {"success": None, "error": str(e)[:50]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--trail", choices=["straight", "curved", "slalom", "gate"], default="straight")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--with_humans", action="store_true")
    args = p.parse_args()

    rows = {name: run(args.weights, args.trail, args.episodes, name, off) for name, off in SUBSETS}

    print(f"\n=== SENSOR-AGGREGATION ABLATION (accuracy) — trail={args.trail} ===")
    print(f"{'subset':18s} {'success':>8s} {'offset(m)':>10s} {'progress':>9s} {'jerk':>8s}")
    for name, _ in SUBSETS:
        r = rows[name]
        if r.get("success") is None:
            print(f"{name:18s} {'FAIL':>8s}  ({r.get('error','?')})"); continue
        print(f"{name:18s} {r['success']:>8.2f} {r['offset']:>10.3f} {r['progress']:>9.2f} {r['jerk']:>8.4f}")

    out = os.path.join(_SCRATCH, f"ablation_accuracy_{args.trail}.json")
    json.dump({"trail": args.trail, "weights": args.weights, "rows": rows}, open(out, "w"), indent=2)
    print(f"\n[abl] wrote {out}")
    print("[abl] pair with vitfly/models/ablate_fused_compute.py (MACs/latency) for the full Pareto.")


if __name__ == "__main__":
    sys.exit(main())
