#!/usr/bin/env python3
"""Generate the schedule-evolution mega-plot sequence — the honest AOT → runtime-feedback → fix story.

Workload: the contended sensor-fusion stack (mlp_control + fused_full + yolov8_nano_64x96 + ffn/attn),
where the beats are all REAL (nothing faked; each panel is a fresh solve or a calibrated re-cost):

  1. og           — RVV singleton dispatches: baseline, misses instance deadlines
  2. + shard + IME — AOT levers (multi-hart widths + matrix-engine routing): meets every deadline ON THE GANTT
  3. runtime feedback — the panel-2 schedule RE-COST with heterogeneous K1 calibration
                        (measured per-dispatch/per-op entries and aggregate fallback;
                        see scripts/recost_schedule_on_board.py): deadlines the Gantt promised are now MISSED
  4. re-schedule on board-calibrated costs — CP-SAT re-solves knowing the true costs: deadlines RECOVERED

The AOT levers fix the modeled baseline, calibrated replay can expose misses, and the feedback
re-schedule recovers them. Emits the four panel schedules + panels.json under results/codesign_feedback/
sensor_evo/, ready for scripts/compose_schedule_evolution.py. The renderer is the source of truth for
instance-level miss counts; the re-cost metrics sidecar counts late dispatches instead.

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


def gen_hardcoded():
    """The ORIGINAL hand-scripted 4-step sequence (default). Kept verbatim so the committed
    canonical figure inputs under results/codesign_feedback/sensor_evo/ remain reproducible."""
    og_spec = make_spec("og", "singletons", False)
    aot_spec = BASE                                     # shard + IME already enabled

    p1 = stash(solve(og_spec), "p1_og")
    p2 = stash(solve(aot_spec), "p2_aot")               # predicted AOT-optimal (meets the Gantt)
    p3 = f"{OUTDIR}/p3_feedback.json"; recost(p2, aot_spec, p3)   # panel-2 schedule on the board
    p4 = stash(solve(aot_spec, board_cal=True), "p4_fix")        # re-solve on board-calibrated costs

    rel = lambda p: os.path.relpath(p, REPO)        # repo-relative → portable panels.json (render from repo root)
    panels = [
        f"Baseline · RVV singletons|none|{rel(p1)}|AOT baseline",
        f"AOT optimize — Shard + IME|shard+ime|{rel(p2)}|both scheduling levers enabled",
        f"K1-calibrated replay|none|{rel(p3)}|same assignment, calibrated durations",
        f"K1-calibrated re-solve|none|{rel(p4)}|new assignment, calibrated durations",
    ]
    json.dump(panels, open(f"{OUTDIR}/panels.json", "w"), indent=1)
    for tag, p in [("1 og", p1), ("2 aot", p2), ("3 feedback", p3), ("4 fix", p4)]:
        mk, ms = mk_miss(p)
        print(f"  panel {tag:12s}: {mk:6.2f} ms  {ms} miss")
    print(f"\nwrote {OUTDIR}/panels.json  ->  render with:")
    print(f"  {PY} scripts/compose_schedule_evolution.py --spec {BASE} \\\n"
          f"      --panels-json {OUTDIR}/panels.json")


# ---- FULLY-AUTOMATIC MODE ---------------------------------------------------------------
# The four panels become a BY-PRODUCT of scripts/run_codesign_loop.py's real automatic loop
# (predicted lever search + board-feedback arm). Nothing about the sequence is hand-chosen:
# panel 1 = the loop's baseline schedule, panel 2 = the AOT schedule the loop converged on
# (whatever levers its MEASURED search accepted), panel 3 = the board re-cost of that schedule,
# panel 4 = the board-calibrated re-solve the loop automatically triggered. Titles/highlights
# are derived from the loop_report, so the figure literally reports the loop's decisions.
AUTO_OUTDIR = os.environ.get("XPURT_EVO_AUTO_OUTDIR", f"{REPO}/results/codesign_feedback/sensor_evo_auto")
LOOP_OUT = "results/codesign_loop"          # run_codesign_loop.py default --out-dir


def gen_from_loop(solver="cpsat", board_solver="cpsat", objective="lateness",
                  workload=None, stem=None, replay=False):
    """Run the loop on `workload` and build the four panels from its trajectory.

    WHY THIS TAKES A WORKLOAD. A hardcoded spec would let the figure that
    demonstrates the loop demonstrate it only on a single taskset -- and
    "the loop is automatic" is a claim about the loop, not about that spec. Each
    workload gets its own panel dir and its own output stem so rungs cannot overwrite
    one another.
    """
    base = workload or BASE
    base = base if os.path.isabs(base) else os.path.join(REPO, base)
    wl_stem = os.path.splitext(os.path.basename(base))[0]
    auto_outdir = (AUTO_OUTDIR if workload is None
                   else f"{REPO}/results/codesign_feedback/evo_{stem or wl_stem}")
    os.makedirs(auto_outdir, exist_ok=True)
    # bounded CP-SAT workers (a heavy Isaac job may be co-resident); NEVER 0 here.
    env = dict(os.environ)
    # DETERMINISM VS SPEED, made explicit. A published figure should be an artifact
    # someone can regenerate. It is not, at >1 CP-SAT worker under a time limit: two
    # runs of w3_ffn_dronet gave AOT stages of 1 and 5 misses, and on w4 even the
    # BASELINE moved (18 vs 26 misses for the identical spec), because the baseline is
    # itself a CP-SAT solve. --replay pins one worker and seed 42 so the figure is
    # reproducible; the default keeps the faster multi-worker search.
    env["XPURT_CPSAT_WORKERS"] = ("1" if replay
                                  else os.environ.get("XPURT_CPSAT_WORKERS", "6"))
    tl = os.environ.get("XPURT_EVO_TIME_LIMIT", "45")
    # the board-calibrated re-solve is the hard solve (it must actually FIND the recovered
    # 0-miss assignment, not just a feasible one) — give CP-SAT a generous budget so the
    # recovery beat is robust to CP-SAT's multi-worker non-determinism (≈170 s to optimum here).
    btl = os.environ.get("XPURT_EVO_BOARD_TIME_LIMIT", str(max(int(tl) * 2, 200)))
    cmd = [PY, "scripts/run_codesign_loop.py", "--workload", base,
           "--solver", solver, "--time-limit", tl, "--objective", objective,
           "--board-calibration", CAL, "--board-solver", board_solver,
           "--board-time-limit", btl]
    if replay:
        cmd.append("--replay")
    print("running the automatic loop (predicted search + board arm):\n  " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)

    report = json.load(open(f"{REPO}/{LOOP_OUT}/{wl_stem}/loop_report.json"))
    bf = report.get("board_feedback", {})
    if not bf.get("enabled"):
        raise SystemExit(f"board-feedback arm did not run: {bf.get('reason', 'unknown')}")
    stages = {s["stage"]: s for s in bf["stages"]}
    applied = report.get("levers_applied", [])
    hl = "+".join([l for l in ("shard", "ime") if l in applied]) or "none"
    # `capitalize()` alone renders the IME engine as "Ime" in a paper figure.
    _NICE = {"ime": "IME", "shard": "Shard", "unfuse": "Unfuse", "fuse": "Fuse",
             "split": "Split"}
    lev_txt = (" + ".join(_NICE.get(l, l.capitalize()) for l in applied)
               if applied else "no levers")

    def _abs(rel):
        return rel if os.path.isabs(rel) else f"{REPO}/{rel}"

    # copy the loop's canonical panel schedules into the auto figure's own dir (self-contained)
    def _copy(stage_key, dst):
        src = _abs(stages[stage_key]["sched"])
        shutil.copy(src, f"{auto_outdir}/{dst}.json")
        ms = src.replace(".json", "_metrics.json")
        if os.path.exists(ms):
            shutil.copy(ms, f"{auto_outdir}/{dst}_metrics.json")
        return f"{auto_outdir}/{dst}.json"

    rel = lambda p: os.path.relpath(p, REPO)
    p1 = _copy("baseline", "a1_baseline")
    p2 = _copy("aot-optimized", "a2_aot")
    p3 = _copy("board-recost", "a3_board_recost")
    panels = [
        f"Baseline · RVV singletons|none|{rel(p1)}|loop baseline (levers stripped)",
        # "Auto-loop", not "AOT": in --from-loop mode these levers were CHOSEN by the
        # loop's measured search, which is the whole claim of the figure. The shipped
        # schedule_evolution_auto.png says "Auto-loop optimize — Shard" and no script
        # on disk emitted that string -- panels.json had been hand-edited, so the one
        # figure that demonstrates automation was the one figure with a hand-written
        # panel title. It is emitted here now.
        f"Auto-loop optimize — {lev_txt}|{hl}|{rel(p2)}|"
        f"levers from the loop's measured search",
        f"K1-calibrated replay|none|{rel(p3)}|same assignment, board durations",
    ]
    tags = [("1 baseline", p1), ("2 aot", p2), ("3 board-recost", p3)]
    if "board-resolve" in stages:
        p4 = _copy("board-resolve", "a4_board_resolve")
        panels.append(f"K1-calibrated re-solve|none|{rel(p4)}|re-optimized on board costs")
        tags.append(("4 board-resolve", p4))

    json.dump(panels, open(f"{auto_outdir}/panels.json", "w"), indent=1)
    print("\nautomatic loop trajectory (instance-miss, the figure's source-of-truth):")
    for s in bf["stages"]:
        print(f"  {s['stage']:16s} [{s['cost']:5s}]  {s['instance_misses']} miss")
    print(f"\nwrote {auto_outdir}/panels.json  ->  rendering schedule_evolution_auto ...")
    out = (f"{REPO}/results/codesign_feedback/schedule_evolution_auto"
           if workload is None else
           f"{REPO}/results/codesign_feedback/evolution_{stem or wl_stem}")
    subprocess.run([PY, "scripts/compose_schedule_evolution.py", "--spec", base,
                    "--panels-json", f"{auto_outdir}/panels.json", "--out", out],
                   cwd=REPO, env=env, check=True)
    print(f"\nrendered {out}.png / {out}.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", default=None,
                    help="spec to run the loop on (default: the sensor shard+IME base). "
                         "Each workload writes its own panels dir and figure stem.")
    ap.add_argument("--stem", default=None,
                    help="output stem override, e.g. w5; default is the spec basename")
    ap.add_argument("--replay", action="store_true",
                    help="generate the figure DETERMINISTICALLY: one CP-SAT worker, "
                         "seed 42, so the same inputs redraw the same figure. Slower, "
                         "and the honest setting for a published artifact -- at 4-6 "
                         "workers the arc moves between runs.")
    ap.add_argument("--from-loop", action="store_true",
                    help="FULLY-AUTOMATIC mode: run scripts/run_codesign_loop.py (with the board-feedback "
                         "arm) and build the four panels from the loop's ACTUAL trajectory. Renders to "
                         "results/codesign_feedback/schedule_evolution_auto (the committed "
                         "schedule_evolution_mega is left untouched).")
    ap.add_argument("--search-solver", choices=["greedy", "cpsat"], default="cpsat",
                    help="--from-loop: scheduler for the loop's predicted lever search.")
    ap.add_argument("--board-solver", choices=["greedy", "cpsat"], default="cpsat",
                    help="--from-loop: scheduler for the board-calibrated re-solve.")
    ap.add_argument("--objective", choices=["makespan", "lateness", "misses"], default="lateness",
                    help="--from-loop: the loop's acceptance objective. Default 'lateness' is the "
                         "deadline-correct one (credits levers that pull instances in ahead of "
                         "deadline even off the makespan critical path).")
    a = ap.parse_args()
    if a.from_loop:
        gen_from_loop(solver=a.search_solver, board_solver=a.board_solver,
                      objective=a.objective, workload=a.workload, stem=a.stem,
                      replay=a.replay)
    else:
        gen_hardcoded()
