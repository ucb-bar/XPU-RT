#!/usr/bin/env python3
"""Sweep scheduling policies over one workload and rank them by the project's
own real-time objective.

What this is
------------
For each solver in the sweep this runs `scripts/run_xpurt_schedule.py` on the
same networks JSON, renders the resulting schedule into trace rows
(`xpu-rt/schedule_trace.py`), scores those rows with `xpu-rt/trace_metrics.py`,
and ranks the candidates with the lexicographic comparison in
`xpu-rt/candidate_objective.py`. It emits a CSV summary, a Gantt per cell, a
side-by-side composite, and a per-term comparison plot.

Why it does not compute its own metrics
---------------------------------------
The restored version of this script ranked by makespan. Makespan is the
*seventh* term of this project's acceptance order and, on a periodic workload,
ranking by it can invert the decision: a policy that finishes the batch sooner
by deferring a 100 Hz control loop is worse, not better. The order is fixed in
`candidate_objective`:

    hard deadline misses -> max lateness -> frequency compliance ->
    p99 response -> heavy-model max latency -> heavy throughput ->
    makespan -> energy/utilization -> standalone kernel cycles (LAST)

Every term carries a measured tolerance; a difference inside it is a tie that
falls through. This script only assembles the inputs and prints the verdict.

Comparability, which is the thing that quietly breaks
-----------------------------------------------------
Two solvers can only be compared if they scheduled the same work. Two ways that
fails here, both handled explicitly rather than hoped about:

* the greedy family runs an iterative periodic-instance refinement loop and will
  grow `num_instances` mid-run, so it can end up scoring more instances than the
  MILP path did. `--max-periodic-iters 1` (the default here, not upstream's 4)
  holds the op set fixed, and `--horizon-ms` pins the instance counts in the
  effective config so every solver starts from the same one;
* `prune_periodic` drops periodic instances that fall entirely after the
  *non-periodic* makespan, which is solver-dependent. On an all-periodic
  workload it is a no-op; when it is not, the per-model instance counts are
  compared across cells and a mismatch is reported loudly and recorded in the
  manifest, because ranking runs of different lengths is ranking different
  experiments.

Profile source
--------------
`--gen-root` (plus `--profile-target` / `--topo-tag` / `--profile-hw`) overrides
where the per-op cost profiles come from, by writing an effective config rather
than editing anything. Re-running the identical sweep against corrected kernel
profiles is a flag change, not a code change:

    scripts/profile_schedulers.py --networks-json data/toplevel/networks_k1_mlp_dronet.json \\
        --gen-root gen_fixed --tag fixed

Unavailable solvers
-------------------
MOSEK needs a license and CP-SAT needs `ortools`. Both are probed before the
sweep runs and reported as `unavailable` with the reason; a missing solver is
never silently dropped from the table, because a blank row and an absent row
read very differently to whoever reads the CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import candidate_objective as objective  # noqa: E402
import schedule_trace  # noqa: E402
import trace_metrics
from schedule_scoring import heavy_stats, score  # noqa: F401  # noqa: E402


# --------------------------------------------------------------- solver table

GREEDY_FAMILY = ("greedy", "greedy_periodic", "decomposed")

#: Extra imports a registry scheduler needs before it can run at all. Probed up
#: front so an absent package is reported as "unavailable", not as a traceback
#: 40 seconds into the cell.
SOLVER_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "cpsat": ("ortools.sat.python.cp_model",),
    "cpsat_memory": ("ortools.sat.python.cp_model",),
    "mosek": ("cvxpy", "mosek"),
    "milp_gurobi": ("cvxpy", "gurobipy"),
    "milp_highs": ("cvxpy",),
    "milp_scip": ("cvxpy",),
    "milp_cbc": ("cvxpy",),
}

#: Substrings that mean "this solver is not usable here", as opposed to "this
#: solver ran and failed". Kept separate from the probe because a license can
#: expire between the probe and the solve.
UNAVAILABLE_MARKERS = (
    "no module named",
    "modulenotfounderror",
    "license",
    "MSK_RES_ERR_LICENSE".lower(),
    "solver mosek is not installed",
    "you need to install",
)


def solver_argv(name: str) -> List[str]:
    if name in GREEDY_FAMILY:
        return ["--solver", name]
    return ["--solver", "milp", "--scheduler", name]


def output_tag(name: str) -> str:
    """The infix `run_xpurt_schedule.py` puts in its output filenames.

    Mirrors that script's naming rules exactly; `mosek` deliberately has no
    infix there, for back-compat with consumers of the canonical outputs.
    """
    if name in GREEDY_FAMILY:
        return f"_{name}"
    return "" if name == "mosek" else f"_{name}"


def probe(name: str) -> Optional[str]:
    """None if the solver can run, else a human reason it cannot."""
    for mod in SOLVER_REQUIREMENTS.get(name, ()):
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            return f"{mod} not importable ({type(exc).__name__}: {exc})"
    if name == "mosek":
        try:
            import mosek  # noqa: WPS433
            env = mosek.Env()
            env.checkoutlicense(mosek.feature.pton)
        except Exception as exc:  # noqa: BLE001
            return f"MOSEK license unavailable ({type(exc).__name__}: {exc})"
    return None


def classify_failure(text: str) -> str:
    low = (text or "").lower()
    return "unavailable" if any(m in low for m in UNAVAILABLE_MARKERS) else "error"


# ----------------------------------------------------------- effective config

def build_effective_config(path: str, out_dir: str, *, gen_root: Optional[str],
                           profile_target: Optional[str],
                           topo_tag: Optional[str],
                           profile_hw: Optional[str],
                           horizon_ms: Optional[float],
                           tag: str) -> Tuple[str, dict, dict]:
    """Materialise the config the whole sweep will actually use.

    Returns (path, config, overrides_applied). Written rather than mutated so
    the sweep's timing basis is a file on disk that can be diffed against the
    original -- the `gen_root` bug this repo already hit was exactly a run
    labelled with one timing basis while reading another.
    """
    with open(path) as f:
        cfg = json.load(f)
    applied: dict = {}

    hw = cfg.setdefault("hardware", {})
    prof = hw.setdefault("profile", {})
    if gen_root:
        applied["gen_root"] = [prof.get("gen_root"), gen_root]
        prof["gen_root"] = gen_root
    if profile_target:
        applied["profile_target"] = [prof.get("target"), profile_target]
        prof["target"] = profile_target
    if topo_tag:
        applied["topo_tag"] = [prof.get("topo_tag"), topo_tag]
        prof["topo_tag"] = topo_tag
    if profile_hw:
        pairs = dict(kv.split("=", 1) for kv in profile_hw.split(",") if "=" in kv)
        applied["profile_hw"] = [dict(hw.get("profile_hw") or {}), pairs]
        hw.setdefault("profile_hw", {}).update(pairs)

    nets = cfg.get("networks") or {}
    pinned = {}
    for nid, info in nets.items():
        T = info.get("period")
        if T is None:
            continue
        if horizon_ms is not None:
            info["num_instances"] = max(1, int(math.ceil(float(horizon_ms) / float(T))))
        pinned[nid] = info.get("num_instances")
    if horizon_ms is not None:
        applied["horizon_ms"] = horizon_ms
    applied["pinned_num_instances"] = pinned

    stem = os.path.splitext(os.path.basename(path))[0] + (f"_{tag}" if tag else "")
    cfg_dir = os.path.join(out_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    eff = os.path.join(cfg_dir, stem + ".json")
    with open(eff, "w") as f:
        json.dump(cfg, f, indent=2)
    return eff, cfg, applied


# ------------------------------------------------------------------- one cell

def run_cell(networks_json: str, name: str, *, timeout: int, profiled: bool,
             max_periodic_iters: int, time_limit: Optional[float],
             log_dir: str, extra: List[str]) -> dict:
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
           "--networks-json", networks_json]
    cmd += solver_argv(name)
    if name in GREEDY_FAMILY:
        cmd += ["--max-periodic-iters", str(max_periodic_iters)]
    if time_limit is not None:
        cmd += ["--time-limit", str(time_limit)]
    cmd += ["--profiled" if profiled else "--no-profiled"]
    cmd += extra

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with open(log_path, "w") as f:
            f.write((exc.stdout or "") + "\n" + (exc.stderr or ""))
        return {"solver": name, "status": "timeout",
                "detail": f"exceeded {timeout}s", "wall_s": float(timeout),
                "log": log_path, "cmd": " ".join(cmd)}
    wall = time.time() - t0
    with open(log_path, "w") as f:
        f.write(proc.stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        blob = (proc.stderr or "") + (proc.stdout or "")
        tail = blob.strip().splitlines()
        tail = tail[-1] if tail else f"exit {proc.returncode}"
        return {"solver": name, "status": classify_failure(blob),
                "detail": tail[:240], "wall_s": round(wall, 1),
                "log": log_path, "cmd": " ".join(cmd)}

    stem = os.path.splitext(os.path.basename(networks_json))[0]
    ptag = "_profiled" if profiled else ""
    sched = os.path.join(REPO, "schedules",
                         f"scheduled_{stem}{output_tag(name)}{ptag}.json")
    if not os.path.exists(sched):
        return {"solver": name, "status": "error",
                "detail": f"no schedule at {sched}", "wall_s": round(wall, 1),
                "log": log_path, "cmd": " ".join(cmd)}
    solver_s = None
    for line in (proc.stdout or "").splitlines():
        if "solver_s=" in line:
            try:
                solver_s = float(line.rsplit("solver_s=", 1)[1].split()[0])
            except Exception:  # noqa: BLE001
                pass
    return {"solver": name, "status": "ok", "detail": "", "wall_s": round(wall, 1),
            "solver_s": solver_s, "schedule": sched, "log": log_path,
            "cmd": " ".join(cmd)}


# -------------------------------------------------------------------- scoring

def advise(schedule_path: str) -> dict:
    """The deadline-aware advisor's read on one cell.

    The version of this script recovered from 413aba1 existed to run the
    advisor on every scheduler and print what it recommended, and that is worth
    keeping: the ranking says which policy wins, the advisor says what is
    *stopping* the losers. It is advisory only and never feeds the ranking --
    which is why it is three extra columns rather than a term.
    """
    report_path = schedule_path.replace(".json", "_report.json")
    if not os.path.exists(report_path):
        return {}
    try:
        import advisor as advisor_mod
        with open(report_path) as f:
            diag = advisor_mod.advise_schedule(json.load(f))
        top = next((r for r in diag.recommendations if r.kind != "none"), None)
        return {
            "bottleneck_backend": diag.bottleneck_backend or "",
            "granularity_verdict": diag.granularity_verdict,
            "top_recommendation": f"{top.kind}:{top.target}" if top else "",
        }
    except Exception as exc:  # noqa: BLE001 - advice must never break a cell
        return {"top_recommendation": f"advisor failed: {exc}"}


def write_csv(path: str, cells: List[dict], models: List[str]) -> str:
    base = ["rank", "solver", "status", "detail", "solver_s", "wall_s",
            "n_dispatches", "periodic_instances",
            "hard_deadline_misses", "max_lateness_ms",
            "frequency_shortfall_frac", "p99_response_ms",
            "heavy_model", "heavy_max_latency_ms", "heavy_throughput_hz",
            "makespan_ms", "utilization_pct", "standalone_service_us",
            "queue_share_pct", "verdict_vs_best",
            "bottleneck_backend", "granularity_verdict", "top_recommendation",
            "schedule_json", "gantt_png"]
    per = []
    for m in models:
        per += [f"{m}.instances", f"{m}.misses", f"{m}.miss_rate_pct",
                f"{m}.worst_lateness_ms", f"{m}.p99_ms", f"{m}.hz",
                f"{m}.required_hz"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base + per, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow(c)
    return path


def plot_terms(cells: List[dict], out_png: str, title: str) -> Optional[str]:
    ok = [c for c in cells if c["status"] == "ok"]
    if not ok:
        return None
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import plot_k1_evolution as pk  # noqa: F401  (imports the paper rcParams)

    panels = [
        ("1. hard deadline misses", "hard_deadline_misses", ""),
        ("2. max lateness (ms)", "max_lateness_ms", ""),
        ("4. p99 response (ms)", "p99_response_ms", ""),
        ("7. makespan (ms)", "makespan_ms", ""),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(pk.DOUBLE_COL, 46 * pk.MM))
    names = [c["solver"] for c in ok]
    best = names[0] if ok and ok[0].get("rank") == 1 else None
    for ax, (label, key, _unit) in zip(axes, panels):
        vals = [float(c.get(key) or 0.0) for c in ok]
        cols = [pk.C_DRONET if n == best else pk.C_MUTED for n in names]
        ax.bar(range(len(vals)), vals, color=cols, width=0.62)
        top = max(vals) if max(vals) > 0 else 1.0
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.03, f"{v:.3g}", ha="center", fontsize=4.4)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=4.4)
        ax.set_title(label, loc="left", pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=6, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks-json",
                    default="data/toplevel/networks_k1_mlp_dronet.json")
    ap.add_argument("--solvers",
                    default="greedy,greedy_periodic,heft,edf,cpsat,mosek")
    ap.add_argument("--out-dir", default="results/scheduler_sweep")
    # Not "" by default. `run_xpurt_schedule.py` derives its output paths from
    # the config's basename and writes them into the shared `schedules/` and
    # `plots/` namespaces, so a sweep run on the unmodified stem overwrites the
    # canonical artifacts for that workload -- including
    # scheduled_networks_k1_mlp_dronet_greedy_profiled.json, which is the B1
    # rung of the published figure in scripts/plot_k1_evolution.py, and which
    # is not version controlled. A sweep is an experiment, not a re-publication
    # of the canonical schedule; it gets its own stem.
    ap.add_argument("--tag", default="sweep",
                    help="suffix for the effective config stem, so a sweep "
                         "never overwrites the canonical schedule for this "
                         "workload and sweeps against different profile trees "
                         "do not overwrite each other (default: 'sweep')")
    # Profile source. The whole point of these being flags: when the fused-conv
    # RVV kernels land, this sweep is re-run with a different --gen-root and
    # nothing in this file changes.
    ap.add_argument("--gen-root", default=None,
                    help="profile tree to read per-op costs from (overrides "
                         "hardware.profile.gen_root)")
    ap.add_argument("--profile-target", default=None)
    ap.add_argument("--topo-tag", default=None)
    ap.add_argument("--profile-hw", default=None,
                    help="comma-separated kind=backend overrides, "
                         "e.g. cpu_p=RVV_fused,cpu_e=RVV_c1")
    ap.add_argument("--horizon-ms", type=float, default=None,
                    help="pin every periodic net to ceil(horizon/period) "
                         "instances so all solvers schedule the same work")
    ap.add_argument("--critical-models", default=None,
                    help="comma-separated models whose deadlines are hard "
                         "(default: every periodic model)")
    ap.add_argument("--heavy-model", default=None,
                    help="the heavy/background model (default: the one with "
                         "the most service time)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--time-limit", type=float, default=None,
                    help="MILP-path solver time limit in seconds")
    ap.add_argument("--max-periodic-iters", type=int, default=1,
                    help="greedy-family refinement passes; 1 keeps the op set "
                         "identical to the MILP path (default 1, not 4)")
    ap.add_argument("--no-profiled", action="store_true")
    ap.add_argument("--window-ms", type=float, default=None,
                    help="Gantt x-range (default: the longest makespan)")
    ap.add_argument("--extra", default="",
                    help="extra args forwarded verbatim to run_xpurt_schedule.py")
    args = ap.parse_args()

    nets_path = args.networks_json
    if not os.path.isabs(nets_path):
        nets_path = os.path.join(REPO, nets_path)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    eff_path, eff_cfg, applied = build_effective_config(
        nets_path, out_dir, gen_root=args.gen_root,
        profile_target=args.profile_target, topo_tag=args.topo_tag,
        profile_hw=args.profile_hw, horizon_ms=args.horizon_ms, tag=args.tag)
    print(f"effective config: {eff_path}")
    for k, v in applied.items():
        print(f"  {k}: {v}")

    nets = eff_cfg.get("networks") or {}
    periodic = {k for k, v in nets.items() if v.get("period") is not None}
    windows_ms = {k: float(v.get("window_duration") or v.get("period"))
                  for k, v in nets.items() if v.get("period") is not None}
    critical = tuple(s.strip() for s in args.critical_models.split(",") if s.strip()) \
        if args.critical_models else ()

    names = [s.strip() for s in args.solvers.split(",") if s.strip()]
    extra = args.extra.split() if args.extra else []

    # Probe first, so the table can say "unavailable" instead of "error".
    availability = {n: probe(n) for n in names}
    for n, why in availability.items():
        if why:
            print(f"  {n}: UNAVAILABLE -- {why}")

    cells: List[dict] = []
    results: Dict[str, dict] = {}
    for n in names:
        why = availability[n]
        if why:
            cells.append({"solver": n, "status": "unavailable", "detail": why})
            continue
        print(f"\n=== {n} ===", flush=True)
        res = run_cell(eff_path, n, timeout=args.timeout,
                       profiled=not args.no_profiled,
                       max_periodic_iters=args.max_periodic_iters,
                       time_limit=args.time_limit,
                       log_dir=os.path.join(out_dir, "logs"), extra=extra)
        print(f"  {res['status']} in {res['wall_s']}s"
              + (f" -- {res['detail']}" if res.get("detail") else ""))
        if res["status"] != "ok":
            cells.append({"solver": n, "status": res["status"],
                          "detail": res["detail"], "wall_s": res["wall_s"]})
            continue
        results[n] = res

    # Pick the heavy model once, from the first successful schedule, so every
    # cell is scored against the same choice.
    heavy = args.heavy_model
    if heavy is None and results:
        first = json.load(open(next(iter(results.values()))["schedule"]))
        by_model: Dict[str, float] = defaultdict(float)
        for d in (first.get("dispatches") or {}).values():
            by_model[trace_metrics.model_of(d.get("job_name", ""))] += \
                float(d.get("duration", 0.0))
        heavy = max(by_model, key=by_model.get) if by_model else None
    print(f"\nheavy model: {heavy}   critical: {critical or 'all periodic'}")

    outcomes: List[objective.CandidateOutcome] = []
    scored: Dict[str, dict] = {}
    gantt_panels: List[dict] = []
    models_seen: List[str] = []
    inst_counts: Dict[str, Dict[str, int]] = {}
    trace_dir = os.path.join(out_dir, "traces")
    for n, res in results.items():
        sched = json.load(open(res["schedule"]))
        summary, outcome, rows = score(n, sched, windows_ms, critical, heavy)
        trace_path = schedule_trace.write_trace_csv(
            rows, os.path.join(trace_dir, f"{n}.csv"))
        outcomes.append(outcome)
        for m in summary.get("per_model", {}):
            if m not in models_seen:
                models_seen.append(m)
        inst_counts[n] = {m: d["instances"]
                          for m, d in summary.get("per_model", {}).items()}
        util = summary.get("per_cluster_utilization_pct") or {}
        row = {
            "solver": n, "status": "ok", "detail": "",
            "solver_s": res.get("solver_s"), "wall_s": res["wall_s"],
            "n_dispatches": summary.get("n_dispatches"),
            "periodic_instances": summary.get("periodic_instances"),
            "hard_deadline_misses": outcome.total_misses(),
            "max_lateness_ms": round(outcome.worst_lateness(), 3),
            "frequency_shortfall_frac": round(outcome.worst_frequency_shortfall(), 4),
            "p99_response_ms": round(outcome.worst_p99(), 3),
            "heavy_model": heavy,
            "heavy_max_latency_ms": round(outcome.heavy_max_latency_ms, 3),
            "heavy_throughput_hz": round(outcome.heavy_throughput_hz, 3),
            "makespan_ms": round(outcome.makespan_ms, 3),
            "utilization_pct": (round(outcome.utilization_pct, 2)
                                if outcome.utilization_pct is not None else ""),
            "standalone_service_us": outcome.standalone_cycles,
            "queue_share_pct": summary.get("queue_share_pct"),
            "schedule_json": os.path.relpath(res["schedule"], REPO),
        }
        for m, d in summary.get("per_model", {}).items():
            row.update({
                f"{m}.instances": d["instances"],
                f"{m}.misses": d["instance_deadline_misses"],
                f"{m}.miss_rate_pct": d["instance_deadline_miss_rate_pct"],
                f"{m}.worst_lateness_ms": d["worst_lateness_ms"],
                f"{m}.p99_ms": d["response_p99_ms"],
                f"{m}.hz": d["achieved_frequency_hz"],
                f"{m}.required_hz": d["required_frequency_hz"],
            })
        row.update(advise(res["schedule"]))
        scored[n] = row
        gantt_panels.append({"solver": n, "rows": rows,
                             "sched": sched["dispatches"],
                             "periods": schedule_trace.periods_ms(sched),
                             "machines": schedule_trace.machines(sched),
                             "trace": trace_path})
        print(trace_metrics.format_summary(f"{n:16s}", summary))

    # Epoch comparability: same op set, or the ranking is between experiments.
    comparability = "ok"
    if len(inst_counts) > 1:
        ref_name, ref = next(iter(inst_counts.items()))
        for n, c in inst_counts.items():
            if c != ref:
                comparability = (f"instance counts differ: {ref_name}={ref} "
                                 f"vs {n}={c}")
                break
    if comparability != "ok":
        print(f"\n!! COMPARABILITY WARNING: {comparability}\n"
              f"   the ranking below compares runs of different lengths; pin "
              f"them with --horizon-ms before trusting it")

    ranked = objective.rank(outcomes)
    order = {o.label: i + 1 for i, o in enumerate(ranked)}
    best = ranked[0] if ranked else None
    for o in outcomes:
        scored[o.label]["rank"] = order[o.label]
        if best is not None and o.label != best.label:
            order_sign, why = objective.compare(o, best)
            # A rank-2 cell that ties the winner on every term is not "worse".
            # `rank` is stable, so the tie is broken by input order, and
            # printing "loses to" about a tie would invent a result the
            # objective explicitly declined to give.
            scored[o.label]["verdict_vs_best"] = (
                f"tie -- {why}" if order_sign == 0 else why)
        else:
            scored[o.label]["verdict_vs_best"] = "best"

    # ------------------------------------------------------------- artifacts
    import plot_k1_evolution as pk
    window = args.window_ms
    if window is None:
        window = max((float(scored[n]["makespan_ms"]) for n in scored), default=140.0)
    all_models = {trace_metrics.model_of(r["job_name"])
                  for p in gantt_panels for r in p["rows"]}
    colours = pk.model_colours(all_models)
    periods_all: Dict[str, float] = {}
    for p in gantt_panels:
        periods_all.update(p["periods"])
    deadline_model = heavy if heavy in periods_all else (
        sorted(periods_all)[0] if periods_all else None)
    cores = gantt_panels[0]["machines"] if gantt_panels else None

    plots_dir = os.path.join(out_dir, "gantt")
    os.makedirs(plots_dir, exist_ok=True)
    ordered = sorted(gantt_panels, key=lambda p: order.get(p["solver"], 99))
    for p in ordered:
        r = order.get(p["solver"], "?")
        png, _ = pk.render_gantt_panels(
            [{"title": f"{p['solver']}  (rank {r})", "rows": p["rows"],
              "sched": p["sched"]}],
            os.path.join(plots_dir, f"gantt_{p['solver']}"),
            periods=periods_all, cores=cores, window_ms=window,
            deadline_model=deadline_model, colours=colours,
            xlabel="Predicted time (ms)", panel_labels=False,
            panel_height_mm=34.0)
        scored[p["solver"]]["gantt_png"] = os.path.relpath(png, REPO)
    composite = None
    if ordered:
        composite, _ = pk.render_gantt_panels(
            [{"title": f"{p['solver']}  (rank {order.get(p['solver'], '?')})",
              "rows": p["rows"], "sched": p["sched"]} for p in ordered],
            os.path.join(plots_dir, "gantt_composite"),
            periods=periods_all, cores=cores, window_ms=window,
            deadline_model=deadline_model, colours=colours,
            xlabel="Predicted time (ms)")

    cells = [c for c in cells] + [scored[n] for n in scored]
    cells.sort(key=lambda c: (c.get("rank") or 999, c["solver"]))
    csv_path = write_csv(os.path.join(out_dir, "scheduler_sweep.csv"),
                         cells, models_seen)
    terms_png = plot_terms(cells, os.path.join(out_dir, "objective_terms.png"),
                           f"lexicographic objective terms -- "
                           f"{os.path.basename(eff_path)}")

    manifest = {
        "networks_json": os.path.relpath(nets_path, REPO),
        "effective_config": os.path.relpath(eff_path, REPO),
        "overrides": applied,
        "solvers_requested": names,
        "availability": {k: (v or "available") for k, v in availability.items()},
        "critical_models": list(critical) or sorted(periodic),
        "heavy_model": heavy,
        "comparability": comparability,
        "compaction_enabled": bool(os.environ.get("XPURT_COMPACT", "0")
                                   in ("1", "true", "True")),
        "automerge_enabled": bool(os.environ.get("XPURT_AUTOMERGE", "0")
                                  in ("1", "true", "True")),
        "objective": "xpu-rt/candidate_objective.py lexicographic order",
        "metrics": "xpu-rt/trace_metrics.py summarise_trace",
        "numbers_are": "PREDICTED from the solver's own per-op profile, not "
                       "measured on the board",
        "ranking": [o.label for o in ranked],
        "csv": os.path.relpath(csv_path, REPO),
        "composite_gantt": os.path.relpath(composite, REPO) if composite else None,
        "objective_terms_png": (os.path.relpath(terms_png, REPO)
                                if terms_png else None),
    }
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 100)
    print(f"{'rank':>4}  {'solver':<16}{'miss':>6}{'late_ms':>10}"
          f"{'freq_sf':>9}{'p99_ms':>9}{'mkspan_ms':>11}{'util%':>7}  status")
    print("-" * 100)
    for c in cells:
        if c["status"] != "ok":
            print(f"{'-':>4}  {c['solver']:<16}{'':>52}  "
                  f"{c['status']}: {c.get('detail', '')[:60]}")
            continue
        print(f"{c['rank']:>4}  {c['solver']:<16}"
              f"{c['hard_deadline_misses']:>6}{c['max_lateness_ms']:>10.2f}"
              f"{c['frequency_shortfall_frac']:>9.3f}{c['p99_response_ms']:>9.2f}"
              f"{c['makespan_ms']:>11.2f}"
              f"{(c['utilization_pct'] if c['utilization_pct'] != '' else 0):>7}"
              f"  ok")
    print("-" * 100)
    for c in cells:
        if c["status"] == "ok" and c.get("verdict_vs_best") != "best":
            v = c["verdict_vs_best"]
            rel = "ties" if v.startswith("tie -- ") else "loses to"
            print(f"  {c['solver']:<16} {rel} {best.label}: "
                  f"{v.removeprefix('tie -- ')}")
    print(f"\nwinner under the lexicographic objective: "
          f"{best.label if best else 'none'}")
    print(f"csv:      {csv_path}")
    print(f"manifest: {man_path}")
    if composite:
        print(f"gantt:    {composite}")
    if terms_png:
        print(f"terms:    {terms_png}")
    print("NOTE: every number here is PREDICTED from the per-op cost profiles "
          "in the effective config, not measured on the board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
