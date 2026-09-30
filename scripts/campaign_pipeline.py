#!/usr/bin/env python3
"""The showdown with BOTH measured latencies injected per arm: the control cadence the runtime
delivers (`--sched_latency_ms`, hold on the last command) and the camera->goal delay of its
chain (`--percep_latency_ms`, a transport delay on the goal, refreshed every `--percep_hold_ms`).

This is the free-running form of RoSE's co-simulation: the vehicle advances at its physics rate
and each command is held until the next one arrives; the on-board latencies decide how old the
command and the goal are. Nothing here is typed in -- every arm's numbers come from
`measured_timing.derive()` (the traced ROS runs and the XPU-RT board traces) at launch, and the
per-row CSV records the exact values used.

Arms (a name, the traced-run key its numbers come from, and the camera rate):
    xpu        XPU-RT long25 trace       control gap from the trace, chain = per-frame median
    ship@25    ROS as shipped, 25 Hz     control gap = timer gap, chain = camera->goal median
    spin@45    ROS, YOLO on 4 harts, one executor, 45 Hz
    cspin@45   ROS, same layout, control chained to the goal topic (control gap = goal gap)
    cp3@45     ROS, three processes, control chained

    scripts/campaign_pipeline.py [--speeds 1.4 1.2 ...] [--gains fixed cal] [--dry]
"""
from __future__ import annotations
import argparse, csv, math, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import measured_timing as MT   # noqa: E402

# the simulator tree and the Isaac interpreter: $XPURT_SIM_TREE / $ISAAC_PY, the two names
# scripts/env.sh resolves from scripts/env.local.sh, else this checkout and this interpreter
CAN = os.environ.get("CAN") or os.environ.get("XPURT_SIM_TREE") or REPO
PY = os.environ.get("ISAAC_PY", sys.executable)
W = f"{CAN}/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT = os.path.join(REPO, "results/codesign_feedback/campaign_pipeline")
CSV = os.path.join(OUT, "pipeline.csv")
ARMS = [("xpu", None, 25), ("ship", "ship", 25), ("spin", "spin", 45), ("cspin", "cspin", 45), ("cp3", "cp3", 45)]


def arm_numbers(d):
    out = {}
    xl = d.get("xpurt_chain_long", {}).get("long25") or d.get("xpurt_chain_long", {}).get("long45")
    g = d.get("xpurt_ctrl_gap", {})
    if xl and g:
        out["xpu"] = {"gap": g["mean"], "chain": xl["chain_median_ms"], "hold": 40.0, "src": "xpurt_long"}
    for name, key, hz in ARMS[1:]:
        a = d.get("ros_arms", {}).get(f"{key}@{hz}")
        if a and a.get("gap_mean_pooled") and a.get("e2e_goal_med_pooled"):
            out[name] = {"gap": a["gap_mean_pooled"], "chain": a["e2e_goal_med_pooled"],
                         "hold": max(1000.0 / hz, a["e2e_goal_med_pooled"]), "src": f"ros_traced/{hz}_{key}"}
    return out


def have(cruise, lat, moment, plat):
    if not os.path.exists(CSV):
        return 0
    n = 0
    for r in csv.DictReader(open(CSV)):
        if (abs(float(r["cruise_speed"]) - cruise) < 1e-6 and abs(float(r["sched_latency_ms"]) - lat) < 1e-3
                and abs(float(r["moment_scale"]) - moment) < 1e-6 and abs(float(r["percep_latency_ms"]) - plat) < 1e-3):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", nargs="+", type=float, default=[1.4, 1.2, 1.6, 1.0, 1.8, 2.0])
    ap.add_argument("--gains", nargs="+", default=["fixed", "cal"])
    ap.add_argument("--eps", type=int, default=12); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "tmp"), exist_ok=True); os.makedirs(os.path.join(OUT, "dumps"), exist_ok=True)
    nums = arm_numbers(MT.derive())
    for k, v in nums.items():
        hz = 1000.0 / (10.0 * math.ceil(v["gap"] / 10.0))
        print(f"{k:<7} control gap {v['gap']:6.2f} ms -> {hz:5.1f} Hz   chain {v['chain']:7.2f} ms  hold {v['hold']:5.1f}  ({v['src']})")
    if a.dry:
        return 0
    env = dict(os.environ, TMPDIR=os.path.join(OUT, "tmp"))
    for policy in a.gains:
        for cru in a.speeds:
            for name, v in nums.items():
                eff = 1000.0 / (10.0 * math.ceil(v["gap"] / 10.0))
                moment = 0.0055 if policy == "fixed" else round(0.5 / eff, 5)
                n = have(cru, v["gap"], moment, v["chain"])
                if n >= a.eps:
                    print(f"skip {name} {policy} cruise={cru} ({n} rows)"); continue
                print(f"=== {name} {policy} cruise={cru} gap={v['gap']:.2f} chain={v['chain']:.2f} hold={v['hold']:.1f} moment={moment}", flush=True)
                cmd = [PY, "sims/scripts/sweep_rate_demo.py", "--headless", "--controller", "rl", "--weights", W,
                       "--sim_dt", "0.01", "--decimation", "1", "--obstacle_level", "8", "--prop_density", "0.30",
                       "--sched_latency_ms", f"{v['gap']:.2f}", "--percep_latency_ms", f"{v['chain']:.2f}",
                       "--percep_hold_ms", f"{v['hold']:.1f}", "--moment_scale", str(moment),
                       "--cruise_speed", str(cru), "--walk_speed", "0.0", "--episodes", str(a.eps), "--seed", "1000",
                       "--max_steps", "1800", "--dump_figure_data", os.path.join(OUT, "dumps", f"{name}_{policy}_c{cru}"),
                       "--sweep-csv", CSV]
                p = subprocess.run(cmd, cwd=CAN, env=env, capture_output=True, text=True, timeout=3000)
                for line in p.stdout.splitlines():
                    if "[SWEEP]" in line or "Traceback" in line or "Error" in line:
                        print(line[:160], flush=True)
    print("PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
