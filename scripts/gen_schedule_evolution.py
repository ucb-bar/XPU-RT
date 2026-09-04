#!/usr/bin/env python3
"""Fresh AOT-opt + runtime-feedback sequence for the schedule-evolution mega plot.
Story (user's order, REAL — accept/reject measured, nothing faked):
  og (fused, RVV, singletons) -> +sharding -> +unfuse (ModelBlaster graph rewrite) ->
  runtime feedback (board-calibration re-solve) -> +other (unfuse+shard combined).
Each config is re-solved fresh with greedy (this YOLO workload is slack-rich → greedy near-optimal AND
tractable, where CP-SAT is intractable on the fused+shard model). Emits scheduled_*.json + _metrics.json,
prints the panel list for compose_schedule_evolution.py.
"""
import json, os, subprocess, sys

REPO = os.environ.get("XPURT_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP = os.environ.get("XPURT_EVO_OUTDIR", f"{REPO}/results/codesign_feedback")
os.makedirs(SP, exist_ok=True)
BASE = f"{REPO}/data/toplevel/networks_k1_mb_3model_4hz_yolo_ctrl.json"
FUSED = "gen_mb/vmfb/yolov8_nano/spacemit_x60/rvv_x60/yolov8_nano.ctrl.int8/yolov8_nano.ctrl.int8_dispatch_graph.json"
UNFUSED = "gen_mb/vmfb/yolov8_nano/spacemit_x60/rvv_x60/yolov8_nano.unfused.int8/yolov8_nano.unfused.int8_dispatch_graph.json"
ENV = dict(os.environ, XPURT_CPSAT_WORKERS="0")
OUTDIR = f"{REPO}/data/toplevel"


def make_spec(name, unfuse, shard):
    d = json.load(open(BASE))
    d["networks"]["yolov8_nano"]["dispatch_deps_path"] = UNFUSED if unfuse else FUSED
    d["scheduler"]["machine_combination_mode"] = "shard" if shard else "singletons"
    d["hardware"]["profile"]["topo_tag_override"] = (not shard)
    p = f"{OUTDIR}/_evo_{name}.json"
    json.dump(d, open(p, "w"), indent=1)
    return p


def solve(spec, calibrate, tl=600):
    # Greedy: the YOLO 3-model workload is slack-rich (4 Hz, 250 ms deadline), so greedy is
    # near-optimal AND tractable — CP-SAT is intractable on the fused+shard combinatorial model.
    stem = os.path.splitext(os.path.basename(spec))[0]
    cmd = [f"{REPO}/.venv/bin/python", "scripts/run_xpurt_schedule.py", "--networks-json", spec,
           "--solver", "greedy", "--profiled", "--max-periodic-iters", "1"]
    if calibrate:
        cmd += ["--board-calibration"]
    try:
        r = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=tl)
    except subprocess.TimeoutExpired:
        print(f"  {stem}: TIMEOUT after {tl}s — skipping", flush=True)
        class _R: returncode = 124
        r = _R()
    sched = f"{REPO}/schedules/scheduled_{stem}_greedy_profiled.json"
    m = json.load(open(sched.replace(".json", "_metrics.json"))) if os.path.exists(sched.replace(".json", "_metrics.json")) else {}
    print(f"  solved {stem}: miss={m.get('deadline_miss_count','?')} mk={m.get('makespan_ms',0):.1f} "
          f"exit={r.returncode} exists={os.path.exists(sched)}", flush=True)
    return sched


ROUNDS = [
    ("og", False, False, False, "none", "og · fused YOLO, RVV, 1-hart"),
    ("shard", False, True, False, "shard", "+ sharding (multi-hart widths)"),
    ("unfuse", True, False, False, "none", "+ unfuse (ModelBlaster graph rewrite: fused→unfused)"),
    ("unfuse_cal", True, False, True, "none", "runtime feedback (re-solve on board-calibrated costs)"),
    ("unfuse_shard", True, True, False, "shard", "+ other: unfuse × sharding combined"),
]

if __name__ == "__main__":
    panels = []
    for name, unf, shd, cal, hi, title in ROUNDS:
        spec = make_spec(name, unf, shd)
        sched = solve(spec, cal)
        panels.append(f"{title}|{hi}|{sched}")
    print("\n=== PANELS (feed to compose_schedule_evolution.py) ===")
    for p in panels:
        print(f'  --panel "{p}"')
    json.dump(panels, open(f"{SP}/evo_panels.json", "w"), indent=1)
    print(f"\nspec (for deadlines): {make_spec('og', False, False)}")
