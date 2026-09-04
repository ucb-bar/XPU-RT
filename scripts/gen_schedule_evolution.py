#!/usr/bin/env python3
"""Generate the schedule-evolution mega-plot sequence — the honest AOT → runtime-feedback → fix story.

Workload: the contended sensor-fusion stack (mlp_control + fused_full + yolov8_nano_64x96 + ffn/attn),
where the beats are all REAL (nothing faked; each panel is a fresh solve or a measured re-cost):

  1. og           — RVV, singletons (1 net per hart): baseline, misses control deadlines
  2. + shard + IME — AOT levers (multi-hart widths + matrix-engine routing): meets every deadline ON THE GANTT
  3. runtime feedback — the panel-2 schedule RE-COST on the measured board (+31% per-op inflation,
                        scripts/recost_schedule_on_board.py): deadlines the Gantt promised are now MISSED
  4. re-schedule on board-calibrated costs — CP-SAT re-solves knowing the true costs: deadlines RECOVERED

So the miss trajectory is 4 → 0 → 4 → 0: the AOT opts fix it on paper, the board breaks it, the feedback
re-schedule fixes it for real. Emits the four panel schedules + panels.json under results/codesign_feedback/
sensor_evo/, ready for scripts/compose_schedule_evolution.py.

NOTE: CP-SAT is non-deterministic (workers=0 → many workers; time-limited incumbents vary run to run), and
different 0-miss predicted schedules re-cost to different board-miss counts. The panel schedules committed
under results/codesign_feedback/sensor_evo/ are therefore the CANONICAL figure inputs; a fresh run of this
script may land on a different (still honest) sequence and should be eyeballed before replacing them.

Env: XPURT_REPO (repo root), XPURT_PY (child interpreter), XPURT_CPSAT_WORKERS=0 (set automatically),
XPURT_EVO_TIME_LIMIT (per-solve CP-SAT seconds; the board-cal fix gets 2×).
"""
import json, os, subprocess, sys, shutil

REPO = os.environ.get("XPURT_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_venv_py = f"{REPO}/.venv/bin/python"
PY = os.environ.get("XPURT_PY") or (_venv_py if os.path.exists(_venv_py) else sys.executable)
ENV = dict(os.environ, XPURT_CPSAT_WORKERS="0")
BASE = f"{REPO}/data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json"  # shard+IME base
OUTDIR = os.environ.get("XPURT_EVO_OUTDIR", f"{REPO}/results/codesign_feedback/sensor_evo")
CAL = f"{REPO}/results/codesign_feedback/k1_board_calibration.json"
TL = os.environ.get("XPURT_EVO_TIME_LIMIT", "120")
os.makedirs(OUTDIR, exist_ok=True)


def make_spec(name, mode, ime):
    d = json.load(open(BASE))
    d["scheduler"]["machine_combination_mode"] = mode
    d["scheduler"]["enable_impls"] = ime
    d["hardware"]["profile"]["topo_tag_override"] = False   # keep the base's profile-topo resolution
    p = f"{REPO}/data/toplevel/_evo_sen_{name}.json"
    json.dump(d, open(p, "w"), indent=1)
    return p


def solve(spec, board_cal=False):
    stem = os.path.splitext(os.path.basename(spec))[0]
    tl = int(TL) * (2 if board_cal else 1)      # the board-cal fix is the hard solve — give it more time
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec, "--solver", "milp",
           "--scheduler", "cpsat", "--profiled", "--max-periodic-iters", "1", "--time-limit", str(tl)]
    if board_cal:
        cmd += ["--board-calibration"]
    subprocess.run(cmd, cwd=REPO, env=ENV, timeout=tl * 2 + 60, capture_output=True, text=True)
    return f"{REPO}/schedules/scheduled_{stem}_cpsat_profiled.json"


def recost(schedule, spec, out):
    subprocess.run([PY, "scripts/recost_schedule_on_board.py", "--schedule", schedule, "--spec", spec,
                    "--calibration", CAL, "--out", out], cwd=REPO, env=ENV, check=True)


def stash(src, dst):
    shutil.copy(src, f"{OUTDIR}/{dst}.json")
    shutil.copy(src.replace(".json", "_metrics.json"), f"{OUTDIR}/{dst}_metrics.json")
    return f"{OUTDIR}/{dst}.json"


def mk_miss(p):
    m = json.load(open(p.replace(".json", "_metrics.json")))
    return m.get("makespan_ms", 0), m.get("deadline_miss_count", "?")


if __name__ == "__main__":
    og_spec = make_spec("og", "singletons", False)
    aot_spec = BASE                                     # shard + IME already enabled

    p1 = stash(solve(og_spec), "p1_og")
    p2 = stash(solve(aot_spec), "p2_aot")               # predicted AOT-optimal (meets the Gantt)
    p3 = f"{OUTDIR}/p3_feedback.json"; recost(p2, aot_spec, p3)   # panel-2 schedule on the board
    p4 = stash(solve(aot_spec, board_cal=True), "p4_fix")        # re-solve on board-calibrated costs

    rel = lambda p: os.path.relpath(p, REPO)        # repo-relative → portable panels.json (render from repo root)
    panels = [
        f"og · RVV, 1 net per hart (singletons)|none|{rel(p1)}",
        f"+ shard + IME  (AOT: multi-hart widths + matrix-engine routing)|shard|{rel(p2)}",
        f"runtime feedback — measured on the K1 board (+31%): control deadlines now missed|none|{rel(p3)}",
        f"re-schedule on board-calibrated costs: all deadlines recovered|none|{rel(p4)}",
    ]
    json.dump(panels, open(f"{OUTDIR}/panels.json", "w"), indent=1)
    for tag, p in [("1 og", p1), ("2 aot", p2), ("3 feedback", p3), ("4 fix", p4)]:
        mk, ms = mk_miss(p)
        print(f"  panel {tag:12s}: {mk:6.2f} ms  {ms} miss")
    print(f"\nwrote {OUTDIR}/panels.json  ->  render with:")
    print(f"  {PY} scripts/compose_schedule_evolution.py --spec {BASE} \\\n"
          f"      --panels-json {OUTDIR}/panels.json")
