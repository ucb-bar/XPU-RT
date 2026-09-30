#!/usr/bin/env python3
"""One board round: an eligible rewrite -> measured cost -> re-solve -> verdict.

WHAT THIS IS FOR. `run_modelblaster_arm.py` stops at *eligible*: host-verified and
graph-gate-passed. It deliberately does not schedule anything, because a rewritten graph
has no profile and scheduling it on costs derived from its parent measures the derivation
rather than the rewrite. This script is the missing step -- rebuild, reprofile ON THE
BOARD, and only then re-solve and adjudicate:

    rewritten IR -> emit_dispatch_graph -> run_model_k1.sh (board) -> stage profile
                 -> run_xpurt_schedule -> compare_candidates (nine terms)

Both arms are profiled in the SAME session with the SAME curated kernels and filed under
their own basenames (`<net>.<tag>`), so nothing clobbers the tree the baseline was solved
from and the only difference between the two schedules is the rewrite. That basename
discipline is not cosmetic: `gen/mb/profile` is a symlink and the profiler writes in
place, so a shared basename means the second run overwrites what the first was solved
from, and the comparison silently becomes a schedule against itself.

Requires: the cross toolchain (`eval "$(scripts/setup_spacemit_toolchain.sh)"`) and a
reachable board (ssh alias `k1`, or $MODELBLASTER_K1_HOST).

Example -- the ffn_block split the scheduler's own advice asked for:

  scripts/run_board_round.py \\
      --workload data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \\
      --net ffn_block \\
      --baseline-ir  build/k1_xpurt/ffn_block/int8/graph.json \\
      --candidate-ir results/modelblaster_arm/sensor_s4_ffn/ffn_block.split.graph.json \\
      --seed-build-dir build/k1_xpurt/ffn_block/int8 \\
      --out-dir results/board_loop/ffn_split
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)
MB = os.environ.get("MB_ROOT") or os.path.join(REPO, "ModelBlaster")


def sh(cmd, env=None, cwd=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, cwd=cwd or REPO, env=e, capture_output=True, text=True)


def arm_model(net, tag):
    """The per-arm model identity: `<net>_<tag>`, quant left at int8.

    WHY THE NAME AND NOT THE QUANT. The runbook says to give a copied IR a distinct
    quant, and that cannot be done: generate_kernels accepts only fp32/fp16/int8 and
    refuses an IR whose quant disagrees with its --quant. The model NAME is the free
    field, and it separates everything that matters -- build dir, dispatch graph
    basename, profile basename -- because each is derived from `ir["name"]`.
    """
    return f"{net}_{tag}"


def profile_dir(net, tag, target, hw, topo="topo_0"):
    """Where the loader looks for a measured profile, keyed by `<model>.int8`."""
    m = arm_model(net, tag)
    return os.path.join(REPO, "gen", "profile_mb", hw, target, m,
                        f"{m}.int8", f"{m}_{target}_{hw}_{m}.int8", topo)


def stage_ir(ir_path, net, tag, work_dir, log):
    """A copy of the IR renamed to this arm's model, so no derived path is shared.

    The runbook's step 4 warning, mechanised: `emit_dispatch_graph` derives its output
    path from `ir["name"]`/`ir["quant"]`, so a candidate that keeps the baseline's
    identity overwrites the baseline's dispatch graph -- and the profiler, writing in
    place through the `gen/mb/profile` symlink, overwrites the costs it was solved from.
    """
    g = json.load(open(ir_path))
    m = arm_model(net, tag)
    g["name"] = m
    g["quant"] = "int8"
    out = os.path.join(work_dir, f"{m}.int8.graph.json")
    json.dump(g, open(out, "w"), indent=1)
    log(f"  staged IR {os.path.relpath(ir_path, REPO) if ir_path.startswith(REPO) else ir_path}"
        f" as name={m} ({len(g.get('ops') or []) } ops)")
    return out


def board_profile(ir, net, tag, seed_dir, target, hw, cores, log):
    """Rebuild for the target and profile on the physical board.

    `run_model_k1.sh` step 1 re-extracts the graph from the torch model unless MB_IR is
    set; MB_IR is what makes a REWRITE profilable at all, since no torch model
    corresponds to it. It needs weights/goldens beside it, which is what --seed-build-dir
    supplies.
    """
    work = os.path.join("/tmp", f"mbk1_{net}_{tag}")
    shutil.rmtree(work, ignore_errors=True)
    # The tag IS the quant. run_model_k1.sh refuses an MB_IR whose `quant` disagrees
    # with its --quant argument (a good guard: it is how a rewrite gets profiled and
    # filed under the baseline's name), and every derived path -- build dir, dispatch
    # graph, profile basename -- keys off it. So seed the build dir under the tag.
    dst = os.path.join(work, arm_model(net, tag), "int8")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(seed_dir):
        s = os.path.join(seed_dir, f)
        if os.path.isfile(s):
            shutil.copy2(s, dst)
    staged = stage_ir(ir, net, tag, dst, log)
    out_root = os.path.join(REPO, "results", "board_runs", f"{net}_{tag}")
    shutil.rmtree(out_root, ignore_errors=True)
    env = {"MB_IR": staged, "PROFILE_OUT_ROOT": out_root, "OUT_ROOT": work,
           "PATH": os.path.dirname(PY) + os.pathsep + os.environ.get("PATH", "")}
    if cores:
        env["MB_CORES"] = cores
    m = arm_model(net, tag)
    log(f"  board: run_model_k1.sh {m} int8 {hw} 0  (MB_IR={os.path.basename(staged)})")
    r = sh([os.path.join(MB, "scripts/run_model_k1.sh"), m, "int8", hw, "0"], env=env)
    if r.returncode != 0:
        log(f"  BOARD RUN FAILED:\n{(r.stderr or r.stdout)[-1500:]}")
        return None
    if "max_abs_err=0" not in r.stdout:
        # The board's own golden compare. Without it the timings describe a kernel that
        # computes something else, which is worse than no timings.
        log("  BOARD VERIFY DID NOT REPORT max_abs_err=0 — refusing to use these costs")
        return None
    n_ref = r.stdout.count("falling back to reference_impl")
    if n_ref:
        log(f"  WARNING: {n_ref} curated kernel(s) fell back to the scalar reference; "
            f"these costs measure the reference, not the curated kernel")
    src = os.path.join(out_root, hw, target, m, f"{m}.int8",
                       f"{m}_{target}_{hw}_{m}.int8", "topo_0", "results.csv")
    if not os.path.exists(src):
        log(f"  board run produced no results.csv at {src}")
        return None
    dest_dir = profile_dir(net, tag, target, hw)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_dir, "results.csv"))
    rows = list(open(src))[1:]
    log(f"  measured {len(rows)} dispatch(es) -> "
        f"{os.path.relpath(dest_dir, REPO)}/results.csv")
    return {"tag": tag, "profile": os.path.relpath(dest_dir, REPO),
            "n_dispatches": len(rows), "staged_ir": staged,
            "reference_fallbacks": n_ref}


def emit_graph(ir, target, hw, log):
    r = sh([PY, os.path.join(MB, "pipeline/emit_dispatch_graph.py"), "--ir", ir,
            "--out-root", "gen/mb/vmfb", "--target", target, "--hw", hw])
    if r.returncode != 0:
        log(f"  emit_dispatch_graph failed: {(r.stderr or r.stdout)[-400:]}")
        return None
    log("  " + (r.stdout or "").strip().splitlines()[-1])
    return True


def write_spec(workload, net, tag, target, hw, out_path):
    """The workload with one net repointed at the tagged graph. Everything else -- every
    other net, every period, every window -- is byte-identical, so the two arms differ
    only by the rewrite."""
    spec = copy.deepcopy(json.load(open(workload)))
    m = arm_model(net, tag)
    # The network KEY stays `net`, so its period, window and criticality are untouched;
    # only which graph (and therefore which measured profile) it points at changes.
    spec["networks"][net]["dispatch_deps_path"] = (
        f"gen/mb/vmfb/{m}/{target}/{hw}/{m}.int8/{m}.int8_dispatch_graph.json")
    json.dump(spec, open(out_path, "w"), indent=1)
    return out_path


def solve(spec_path, solver, time_limit, board_cal, log):
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
           "--profiled", "--max-periodic-iters", "1"]
    sfx = "greedy_profiled"
    if solver == "cpsat":
        cmd += ["--solver", "milp", "--scheduler", "cpsat"]
        sfx = "cpsat_profiled"
    else:
        cmd += ["--solver", "greedy"]
    if time_limit:
        cmd += ["--time-limit", str(time_limit)]
    if board_cal:
        cmd += ["--board-calibration", board_cal]
    r = sh(cmd)
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}.json")
    if not os.path.exists(sched):
        log(f"  SOLVE FAILED: {(r.stderr or r.stdout)[-600:]}")
        return None
    m = json.load(open(sched.replace(".json", "_metrics.json")))
    log(f"  solved {stem}: makespan {float(m['makespan_ms']):.2f} ms, "
        f"op-miss {m['op_deadline_miss_count']}")
    return sched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--net", required=True, help="the net the rewrite applies to")
    ap.add_argument("--baseline-ir", required=True)
    ap.add_argument("--candidate-ir", required=True)
    ap.add_argument("--seed-build-dir", required=True,
                    help="a ModelBlaster build dir for this net holding weights.npz and "
                         "io.npz; the board run needs them beside the IR")
    ap.add_argument("--baseline-tag", default="base2")
    ap.add_argument("--candidate-tag", default="cand2")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--hw", default="rvv_x60")
    ap.add_argument("--cores", default=None, help="MB_CORES for a multi-hart profile")
    ap.add_argument("--solver", choices=["greedy", "cpsat"], default="greedy")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--board-calibration", default=None,
                    help="re-solve both arms against measured board multipliers too")
    ap.add_argument("--critical-models", default=None)
    ap.add_argument("--heavy-model", default=None)
    ap.add_argument("--out-dir", default="results/board_loop/round")
    a = ap.parse_args()

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(os.path.join(out_dir, "specs"), exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"== board round: {a.net} ==")
    arms = {}
    for label, ir, tag in (("baseline", a.baseline_ir, a.baseline_tag),
                           ("candidate", a.candidate_ir, a.candidate_tag)):
        log(f"[{label}] tag={tag}")
        ir_abs = ir if os.path.isabs(ir) else os.path.join(REPO, ir)
        if not os.path.exists(ir_abs):
            log(f"  no IR at {ir_abs}")
            return 2
        prof = board_profile(ir_abs, a.net, tag, a.seed_build_dir, a.target, a.hw,
                             a.cores, log)
        if prof is None:
            return 1
        if not emit_graph(prof["staged_ir"], a.target, a.hw, log):
            return 1
        spec = write_spec(a.workload, a.net, tag, a.target, a.hw,
                          os.path.join(out_dir, "specs", f"{a.net}_{tag}.json"))
        sched = solve(spec, a.solver, a.time_limit, a.board_calibration, log)
        if sched is None:
            return 1
        arms[label] = dict(prof, spec=os.path.relpath(spec, REPO),
                           schedule=os.path.relpath(sched, REPO))

    # The nine-term verdict, in process rather than by eye.
    crit = a.critical_models
    if crit is None:
        nets = json.load(open(a.workload))["networks"]
        crit = ",".join(n for n, v in nets.items()
                        if float(v.get("window_duration") or 0) > 0)
    heavy = a.heavy_model
    if heavy is None:
        nets = json.load(open(a.workload))["networks"]
        heavy = max(nets, key=lambda n: float(nets[n].get("period") or 0))
    vjson = os.path.join(out_dir, "verdict.json")
    r = sh([PY, "scripts/compare_candidates.py",
            "--baseline-schedule", arms["baseline"]["schedule"],
            "--candidate-schedule", arms["candidate"]["schedule"],
            "--windows-from", arms["candidate"]["spec"],
            "--critical-models", crit, "--heavy-model", heavy,
            "--baseline-label", f"board-{a.baseline_tag}",
            "--candidate-label", f"board-{a.candidate_tag}",
            "--json", vjson])
    log((r.stdout or "").strip())
    if r.returncode == 2:
        log("REFUSED — see above; the two arms are not comparable")
    report = {
        "schema": "board_round/v1",
        "net": a.net, "workload": a.workload, "solver": a.solver,
        "critical_models": crit.split(","), "heavy_model": heavy,
        "arms": arms,
        "verdict": json.load(open(vjson)) if os.path.exists(vjson) else None,
        "accepted": (r.returncode == 0),
        "note": ("Both arms measured on the physical board in one session with the same "
                 "curated kernels, filed under separate basenames. `accepted` is "
                 "candidate_objective.accept(): nine lexicographic terms, hard deadline "
                 "misses first, makespan seventh."),
    }
    json.dump(report, open(os.path.join(out_dir, "board_round.json"), "w"), indent=1)
    open(os.path.join(out_dir, "board_round.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nreport: {os.path.relpath(os.path.join(out_dir, 'board_round.json'), REPO)}")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
