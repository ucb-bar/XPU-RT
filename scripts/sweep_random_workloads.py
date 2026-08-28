#!/usr/bin/env python3
"""
Sweep gen_random_workload.py across the core-count hardware configs
(single_core, dual_core, quad_core) and schedule every config that comes
out of it.

Hardware is the only axis that varies by default: every other generator
knob sits at gen_random_workload.py's own default, so the out-of-the-box
sweep is exactly three configs.  The other knobs are still exposed as
flags (each takes a comma-separated list) if you want to widen the grid.

Two phases:

  1. GENERATE -- the cartesian product of the generator's knobs (hardware,
     seed, --min-scale/--max-scale, --period-headroom, --max-gap,
     --num-instances/--cap-instances) is written to
     data/toplevel/generated-data/<sweep>/.  gen_random_workload.py's own
     validate() gates each point: a config that fails is recorded as
     `gen-invalid` and never scheduled.
  2. SCHEDULE -- every surviving config is run through
     scripts/run_xpurt_schedule.py once per --solver.  The plot and
     scheduled JSON the runner drops in plots/ and schedules/ are moved
     into the sweep directory, and the makespans are collected.

While phase 2 runs, every scheduler run is tracked (queued / running / its
final status): each finished run prints as soon as it lands, a status block
listing what is still in flight is printed every --status-interval seconds,
and the summary ends with a table of all runs.

Scheduler runs are bounded by --max-time (per run, wall clock).  Shortening
it also shortens the MILP solve (--time-limit is capped to fit) but not the
greedy loop, which is bounded by --max-periodic-iters instead -- lower that
too if greedy runs are being killed at the deadline rather than finishing.

Everything lands in runs/sweeps/<sweep>/:
    results.csv     one row per (config x solver) run
    results.json    the same rows plus the resolved sweep axes
    schedules/      scheduled_*.json moved out of the repo's schedules/
    plots/          *.png moved out of the repo's plots/
    logs/           full stdout+stderr of every generator and solver run

The runner needs numpy (and, for --solver milp, cvxpy+mosek), which the
generator does not.  If the interpreter running this script can't import
numpy, a sibling conda env that can is used for the scheduling phase; use
--scheduler-python to pin one explicitly.

Usage:
    # what would run, without running it
    python scripts/sweep_random_workloads.py --dry-run

    # the default grid (single_core, dual_core, quad_core x greedy solvers)
    python scripts/sweep_random_workloads.py --name nightly --jobs 4

    # just one of them
    python scripts/sweep_random_workloads.py --hardware dual_core

    # widen an axis back out on top of the core-count sweep
    python scripts/sweep_random_workloads.py --seeds 0,1,2,3,4,5,6,7 \\
        --scale 1.2:5.0 --period-headroom 1.25 --max-gap 1.0 \\
        --solver greedy,greedy_periodic

    # re-run only what's missing (existing rows are kept)
    python scripts/sweep_random_workloads.py --name nightly --resume

    # keep it short: 2 min per run, lighter greedy refinement, status every 15s
    python scripts/sweep_random_workloads.py --max-time 120 \\
        --max-periodic-iters 2 --status-interval 15
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN_SCRIPT = os.path.join("scripts", "gen_random_workload.py")
RUN_SCRIPT = os.path.join("scripts", "run_xpurt_schedule.py")
HARDWARE_BANK = os.path.join("data", "banks", "hardware_bank.json")
CONFIG_ROOT = os.path.join("data", "toplevel", "generated-data")
SWEEP_ROOT = os.path.join("runs", "sweeps")

# The swept hardware axis: the three core-count configs from the hardware
# bank, smallest first.  The bank holds other entries (qrb5165_hetero, the
# firesim/spike boards); they are reachable with --hardware but are not part
# of the default sweep.
CORE_HARDWARE = ["single_core", "dual_core", "quad_core"]

# run_xpurt_schedule.py's output naming (see its "Output naming" comment):
# plots/<stem><solver_tag><_profiled>.png and
# schedules/scheduled_<stem><solver_tag><_profiled>.json.  MILP deliberately
# carries no infix.
SOLVER_TAG = {
    "milp": "",
    "greedy": "_greedy",
    "greedy_periodic": "_greedy_periodic",
    "decomposed": "_decomposed",
}

RE_GREEDY_MAKESPAN = re.compile(
    r"Final greedy makespan:\s*([0-9.]+)\s*ms\s*\(after\s*(\d+)\s*iteration")
RE_NONPERIODIC_MAKESPAN = re.compile(
    r"Makespan \(non-periodic\):\s*([0-9.]+)\s*ms")

FIELDNAMES = [
    "status", "config", "solver",
    "hardware", "seed", "min_scale", "max_scale", "period_headroom",
    "max_gap", "num_instances", "cap_instances",
    "n_networks", "n_periodic", "n_sporadic", "n_edges", "period_ms",
    "num_operations", "makespan_ms", "makespan_nonperiodic_ms",
    "greedy_iters", "gen_seconds", "sched_seconds",
    "schedule_json", "plot_png", "log", "detail",
]


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

def _num(v: float) -> str:
    """Filename-safe float: 1.2 -> '1p2', 5.0 -> '5'."""
    return format(v, "g").replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class Point:
    """One generator invocation."""
    hardware: str
    seed: int
    min_scale: float
    max_scale: float
    period_headroom: float
    max_gap: float
    num_instances: str
    cap_instances: Optional[int]

    @property
    def stem(self) -> str:
        cap = f"_cap{self.cap_instances}" if self.cap_instances else ""
        return (f"networks_random_{self.hardware}_seed{self.seed}"
                f"_sc{_num(self.min_scale)}-{_num(self.max_scale)}"
                f"_ph{_num(self.period_headroom)}"
                f"_gap{_num(self.max_gap)}"
                f"_ni{self.num_instances}{cap}")


def build_grid(args) -> List[Point]:
    points = [
        Point(hardware=hw, seed=seed, min_scale=lo, max_scale=hi,
              period_headroom=ph, max_gap=gap,
              num_instances=ni, cap_instances=args.cap_instances)
        for hw, seed, (lo, hi), ph, gap, ni in itertools.product(
            args.hardware, args.seeds, args.scale,
            args.period_headroom, args.max_gap, args.num_instances)
    ]
    if args.limit:
        points = points[:args.limit]
    return points


# --------------------------------------------------------------------------
# Interpreter for the scheduling phase
# --------------------------------------------------------------------------

def _imports(python: str, module: str) -> bool:
    try:
        return subprocess.run([python, "-c", f"import {module}"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_scheduler_python(explicit: Optional[str]) -> str:
    """Find an interpreter that can run run_xpurt_schedule.py (needs numpy)."""
    if explicit:
        return explicit
    env = os.environ.get("XPURT_SCHEDULER_PYTHON")
    if env:
        return env
    if _imports(sys.executable, "numpy"):
        return sys.executable
    # Sibling conda envs, scheduler-looking names first.
    candidates = sorted(
        glob.glob(os.path.join(sys.prefix, "envs", "*", "bin", "python")),
        key=lambda p: (0 if "sched" in p else 1, p))
    for cand in candidates:
        if _imports(cand, "numpy"):
            return cand
    raise SystemExit(
        "no interpreter with numpy found for the scheduling phase; pass "
        "--scheduler-python /path/to/python (or set XPURT_SCHEDULER_PYTHON)")


# --------------------------------------------------------------------------
# Live status of the scheduler runs
# --------------------------------------------------------------------------

def hms(seconds: float) -> str:
    """0:07, 3:21, 1:04:09."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


class Tracker:
    """State of every phase-2 run, so the sweep can say what it is doing.

    Each run is queued -> running -> done; the pool worker calls start()/
    finish() around the subprocess, the monitor thread reads snapshot()
    every --status-interval seconds, and the summary prints the final table.
    """

    def __init__(self, jobs: List[Tuple["Point", str, str]], max_time: Optional[float]):
        self.max_time = max_time
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._runs: Dict[Tuple[str, str], Dict[str, object]] = {}
        for point, config, solver in jobs:
            self._runs[(point.stem, solver)] = {
                "stem": point.stem, "config": config, "solver": solver,
                "state": "queued", "status": "", "started": None,
                "seconds": None, "makespan": "", "detail": "",
            }

    def start(self, point: "Point", solver: str) -> None:
        with self._lock:
            run = self._runs[(point.stem, solver)]
            run["state"] = "running"
            run["started"] = time.time()

    def finish(self, point: "Point", solver: str, row: Dict[str, object]) -> None:
        with self._lock:
            run = self._runs[(point.stem, solver)]
            run["state"] = "done"
            run["status"] = str(row.get("status", "?"))
            run["makespan"] = row.get("makespan_ms", "")
            run["detail"] = str(row.get("detail", ""))
            run["seconds"] = row.get(
                "sched_seconds",
                round(time.time() - run["started"], 2) if run["started"] else None)

    def snapshot(self) -> List[Dict[str, object]]:
        with self._lock:
            return [dict(run) for run in self._runs.values()]

    def counts(self) -> Tuple[int, int, List[Dict[str, object]], Dict[str, int]]:
        """(queued, running, finished runs, finished status -> count)."""
        queued = running = 0
        done: List[Dict[str, object]] = []
        by_status: Dict[str, int] = {}
        for run in self.snapshot():
            if run["state"] == "queued":
                queued += 1
            elif run["state"] == "running":
                running += 1
            else:
                done.append(run)
                key = str(run["status"])
                by_status[key] = by_status.get(key, 0) + 1
        return queued, running, done, by_status

    def print_status(self) -> None:
        """The periodic in-flight block."""
        queued, running, done, by_status = self.counts()
        breakdown = ", ".join(f"{k} {v}" for k, v in sorted(by_status.items()))
        print(f"  .. {hms(time.time() - self.started_at)} in: {running} running, "
              f"{queued} queued, {len(done)} done"
              + (f" ({breakdown})" if breakdown else ""))
        now = time.time()
        live = sorted((r for r in self.snapshot() if r["state"] == "running"),
                      key=lambda r: r["started"] or now)
        budget = f"/{hms(self.max_time)}" if self.max_time else ""
        for run in live:
            elapsed = hms(now - float(run["started"] or now))
            print(f"       running {elapsed}{budget}  {str(run['solver']):<16} "
                  f"{run['stem']}")

    def print_table(self) -> None:
        """The end-of-sweep per-run table."""
        runs = sorted(self.snapshot(), key=lambda r: (r["stem"], r["solver"]))
        if not runs:
            return
        print("\n  scheduler runs")
        print(f"    {'status':<12} {'solver':<16} {'elapsed':>8} "
              f"{'makespan(ms)':>13}  config")
        for run in runs:
            status = str(run["status"] or run["state"])
            secs = run["seconds"]
            elapsed = hms(float(secs)) if isinstance(secs, (int, float)) else "-"
            mk = run["makespan"]
            makespan = f"{float(mk):.2f}" if isinstance(mk, (int, float)) else "-"
            print(f"    {status:<12} {str(run['solver']):<16} {elapsed:>8} "
                  f"{makespan:>13}  {run['stem']}")
            if run["detail"]:
                print(f"      -- {str(run['detail'])[:150]}")


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

def config_stats(path: str) -> Dict[str, object]:
    with open(path) as f:
        cfg = json.load(f)
    nets = cfg["networks"]
    periodic = [v for v in nets.values() if "period" in v]
    return {
        "n_networks": len(nets),
        "n_periodic": len(periodic),
        "n_sporadic": len(nets) - len(periodic),
        "n_edges": len(cfg.get("edges", [])),
        "period_ms": periodic[0]["period"] if periodic else "",
    }


def generate(point: Point, args, log_dir: str) -> Tuple[str, Optional[str], str, float]:
    """Run the generator for one grid point.

    Returns (status, config path or None, detail, seconds).
    """
    out = os.path.join(CONFIG_ROOT, args.name, point.stem + ".json")
    abs_out = os.path.join(REPO_ROOT, out)
    if args.resume and os.path.exists(abs_out):
        return "reused", out, "", 0.0

    cmd = [sys.executable, GEN_SCRIPT, str(point.seed),
           "--repo-root", REPO_ROOT,
           "--hardware", point.hardware,
           "--min-scale", str(point.min_scale),
           "--max-scale", str(point.max_scale),
           "--period-headroom", str(point.period_headroom),
           "--max-gap", str(point.max_gap),
           "--num-instances", point.num_instances,
           "-o", out]
    if point.cap_instances:
        cmd += ["--cap-instances", str(point.cap_instances)]

    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - started

    log = os.path.join(log_dir, f"gen_{point.stem}.log")
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + proc.stdout + proc.stderr)

    if proc.returncode == 0:
        return "generated", out, "", elapsed
    # exit 2 is validate()'s "generated config failed validation"; anything
    # else is an unusable bank entry, a bad flag combination, or a crash.
    status = "gen-invalid" if proc.returncode == 2 else "gen-error"
    detail = " | ".join(
        line.strip() for line in (proc.stderr or proc.stdout).splitlines()
        if line.strip())[-400:]
    return status, None, detail, elapsed


def schedule(point: Point, config: str, solver: str, args,
             sweep_dir: str, sched_python: str) -> Dict[str, object]:
    stem = point.stem
    tag = SOLVER_TAG[solver]
    log = os.path.join(sweep_dir, "logs", f"sched_{stem}{tag or '_milp'}.log")

    row: Dict[str, object] = {
        "config": config, "solver": solver, "log": os.path.relpath(log, REPO_ROOT),
    }

    dest_json = os.path.join(sweep_dir, "schedules",
                             f"scheduled_{stem}{tag}.json")
    if args.resume and os.path.exists(dest_json):
        row.update(status="reused", schedule_json=os.path.relpath(dest_json, REPO_ROOT))
        row.update(read_schedule_metadata(dest_json))
        return row

    cmd = [sched_python, RUN_SCRIPT,
           "--networks-json", config,
           "--solver", solver,
           "--time-limit", str(args.time_limit),
           "--max-periodic-iters", str(args.max_periodic_iters)]

    deadline = args.max_time if args.max_time and args.max_time > 0 else None
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=deadline)
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, timed_out = -1, True
    row["sched_seconds"] = round(time.time() - started, 2)

    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + out + err)

    if timed_out:
        # The partial stdout is in the log; its last line says how far the
        # run got, which is what you need to decide whether to raise
        # --max-time or shrink the workload.
        where = last_progress(out)
        row.update(status="timeout",
                   detail=f"killed at --max-time {deadline}s"
                          + (f"; last output: {where}" if where else ""))
        return row
    if rc != 0:
        tail = [l.strip() for l in (err or out).splitlines() if l.strip()]
        row.update(status="sched-error", detail=" | ".join(tail[-3:])[-400:])
        return row

    m = RE_GREEDY_MAKESPAN.search(out)
    if m:
        row["makespan_ms"] = float(m.group(1))
        row["greedy_iters"] = int(m.group(2))
    m = RE_NONPERIODIC_MAKESPAN.search(out)
    if m:
        row["makespan_nonperiodic_ms"] = float(m.group(1))

    # Move the runner's artifacts out of the repo-level plots/ and schedules/
    # and into the sweep directory.
    for kind, directory, prefix, ext, dest in (
        ("schedule_json", "schedules", "scheduled_", ".json", dest_json),
        ("plot_png", "plots", "", ".png",
         os.path.join(sweep_dir, "plots", f"{stem}{tag}.png")),
    ):
        src = find_artifact(directory, prefix, stem, solver, ext)
        if src:
            shutil.move(src, dest)
            row[kind] = os.path.relpath(dest, REPO_ROOT)

    if row.get("schedule_json"):
        row.update(read_schedule_metadata(
            os.path.join(REPO_ROOT, str(row["schedule_json"]))))
    row["status"] = "ok"
    return row


def last_progress(text: str, limit: int = 120) -> str:
    """Last short, non-blank stdout line -- the runner also prints huge
    single-line dumps (the full job-name list), which say nothing useful."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and len(line) <= limit:
            return line
    return ""


def find_artifact(directory: str, prefix: str, stem: str, solver: str,
                  ext: str) -> Optional[str]:
    """Locate the plot / scheduled JSON run_xpurt_schedule.py just wrote.

    The exact names are known (`<prefix><stem><solver_tag>[_profiled]<ext>`),
    so they are tried first.  The glob fallback covers a differing profiled
    suffix, and has to reject the other solvers' outputs by hand: `greedy`'s
    tag is a prefix of `greedy_periodic`'s, and MILP's tag is empty, so a
    plain `<stem><tag>*` glob would let parallel runs steal each other's
    files.
    """
    tag = SOLVER_TAG[solver]
    base = os.path.join(REPO_ROOT, directory)
    for name in (f"{prefix}{stem}{tag}_profiled{ext}", f"{prefix}{stem}{tag}{ext}"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    other_tags = [t for s, t in SOLVER_TAG.items() if s != solver and t]
    hits = sorted(glob.glob(os.path.join(base, f"{prefix}{stem}{tag}*{ext}")),
                  key=os.path.getmtime, reverse=True)
    for hit in hits:
        rest = os.path.basename(hit)[len(prefix) + len(stem) + len(tag):]
        if any(rest.startswith(other[len(tag):]) for other in other_tags
               if other.startswith(tag) and other != tag):
            continue
        return hit
    return None


def read_schedule_metadata(path: str) -> Dict[str, object]:
    try:
        with open(path) as f:
            meta = json.load(f).get("metadata", {})
    except (OSError, ValueError):
        return {}
    out: Dict[str, object] = {"num_operations": meta.get("num_operations", "")}
    # metadata["makespan"] covers every op (periodic included); the runner's
    # stdout number is non-periodic only, so keep both.
    if meta.get("makespan") is not None:
        out["makespan_ms"] = round(float(meta["makespan"]), 2)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_scale(text: str) -> Tuple[float, float]:
    try:
        lo, hi = text.split(":")
        lo, hi = float(lo), float(hi)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--scale wants MIN:MAX (e.g. 1.2:5.0), got {text!r}")
    if not 0 < lo <= hi:
        raise argparse.ArgumentTypeError(
            f"--scale MIN must be > 0 and <= MAX, got {text!r}")
    return lo, hi


def csv_list(cast):
    def parse(text: str):
        return [cast(part) for part in text.split(",") if part.strip()]
    return parse


def check_hardware(names: List[str], ap: argparse.ArgumentParser) -> None:
    """Reject bank entries that don't exist or are marked unavailable."""
    with open(os.path.join(REPO_ROOT, HARDWARE_BANK)) as f:
        configs = json.load(f)["configs"]
    for name in names:
        entry = configs.get(name)
        if entry is None:
            ap.error(f"unknown hardware {name!r}; the bank has "
                     f"{sorted(configs)}")
        if not entry.get("available", True):
            reason = entry.get("unavailable_reason", "marked unavailable")
            ap.error(f"hardware {name!r} is not usable: {reason}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--name", default=None,
                    help="sweep name; outputs go to runs/sweeps/<name>/ and "
                         "configs to data/toplevel/generated-data/<name>/ "
                         "(default: sweep_<timestamp>)")

    # Generator axes.
    ap.add_argument("--hardware", type=csv_list(str), default=CORE_HARDWARE,
                    help="comma-separated hardware bank entries "
                         f"(default: {','.join(CORE_HARDWARE)})")
    # The remaining generator axes are pinned to gen_random_workload.py's own
    # defaults so that hardware is the only thing the default sweep varies.
    ap.add_argument("--seeds", type=csv_list(int), default=[0],
                    help="comma-separated RNG seeds (default: 0)")
    ap.add_argument("--scale", type=csv_list(parse_scale),
                    default=[(1.2, 5.0)],
                    help="comma-separated MIN:MAX window/period scale bands "
                         "(default: 1.2:5.0)")
    ap.add_argument("--period-headroom", type=csv_list(float), default=[1.25],
                    help="comma-separated --period-headroom values (default: 1.25)")
    ap.add_argument("--max-gap", type=csv_list(float), default=[1.0],
                    help="comma-separated --max-gap values (default: 1.0)")
    ap.add_argument("--num-instances", type=csv_list(str), default=["auto"],
                    help="comma-separated --num-instances values (default: auto)")
    ap.add_argument("--cap-instances", type=int, default=None,
                    help="passed through to the generator on every point")

    # Scheduler axis.
    ap.add_argument("--solver", type=csv_list(str),
                    default=["greedy", "greedy_periodic", "decomposed"],
                    help="comma-separated solvers (default: the three greedy "
                         "variants; 'milp' needs cvxpy+mosek)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="(milp) solver time limit in seconds (default: 20, "
                         "or 80%% of --max-time when that is the smaller)")
    ap.add_argument("--max-periodic-iters", type=int, default=4,
                    help="(greedy) periodic refinement iterations (default: 4)")

    # Execution.
    ap.add_argument("--jobs", type=int, default=4,
                    help="scheduler runs in parallel (default: 4)")
    ap.add_argument("--max-time", "--timeout", dest="max_time", type=float,
                    default=900,
                    help="wall-clock budget for one scheduler run, in seconds "
                         "(default: 900; 0 = no deadline).  A run still going "
                         "at the deadline is killed and recorded as 'timeout', "
                         "so to shorten runs that actually finish, lower "
                         "--max-periodic-iters (greedy) as well")
    ap.add_argument("--status-interval", type=float, default=30,
                    help="seconds between in-flight status blocks during "
                         "scheduling (default: 30; 0 = only print runs as "
                         "they finish)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only take the first N grid points")
    ap.add_argument("--resume", action="store_true",
                    help="keep configs and schedules that already exist")
    ap.add_argument("--skip-schedule", action="store_true",
                    help="generate the configs and stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and exit")
    ap.add_argument("--scheduler-python", default=None,
                    help="interpreter for run_xpurt_schedule.py "
                         "(default: this one if it has numpy, else a sibling "
                         "conda env that does)")
    args = ap.parse_args(argv)

    for solver in args.solver:
        if solver not in SOLVER_TAG:
            ap.error(f"unknown solver {solver!r}; pick from {sorted(SOLVER_TAG)}")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if args.time_limit is None:
        # A MILP solve that runs past the deadline is killed with nothing to
        # show for it, so keep the solver's own limit inside the budget.
        args.time_limit = 20.0
        if args.max_time and args.max_time > 0:
            args.time_limit = min(args.time_limit, 0.8 * args.max_time)

    check_hardware(args.hardware, ap)
    if not args.name:
        args.name = "sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    points = build_grid(args)
    n_runs = len(points) * len(args.solver)

    print(f"sweep {args.name}")
    print(f"  hardware        : {', '.join(args.hardware)}")
    print(f"  seeds           : {', '.join(str(s) for s in args.seeds)}")
    print(f"  scale bands     : {', '.join(f'{lo}:{hi}' for lo, hi in args.scale)}")
    print(f"  period-headroom : {', '.join(str(v) for v in args.period_headroom)}")
    print(f"  max-gap         : {', '.join(str(v) for v in args.max_gap)}")
    print(f"  num-instances   : {', '.join(args.num_instances)}")
    print(f"  solvers         : {', '.join(args.solver)}")
    print(f"  max-time        : "
          + (f"{args.max_time:g}s per scheduler run" if args.max_time > 0
             else "no deadline")
          + f" (milp --time-limit {args.time_limit:g}s, "
            f"greedy --max-periodic-iters {args.max_periodic_iters})")
    print(f"  {len(points)} configs x {len(args.solver)} solvers = {n_runs} scheduler runs")

    if args.dry_run:
        for p in points:
            print(f"    {p.stem}")
        return 0

    sweep_dir = os.path.join(REPO_ROOT, SWEEP_ROOT, args.name)
    for sub in ("schedules", "plots", "logs"):
        os.makedirs(os.path.join(sweep_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, CONFIG_ROOT, args.name), exist_ok=True)

    sched_python = (None if args.skip_schedule
                    else resolve_scheduler_python(args.scheduler_python))
    if sched_python:
        print(f"  scheduler python: {sched_python}")

    rows: List[Dict[str, object]] = []
    point_base: Dict[Point, Dict[str, object]] = {}
    results_csv = os.path.join(sweep_dir, "results.csv")
    csv_file = open(results_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    lock = threading.Lock()

    def record(row: Dict[str, object]) -> None:
        with lock:
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()

    # ---- phase 1: generate -------------------------------------------------
    print(f"\n[1/2] generating {len(points)} configs")
    runnable: List[Tuple[Point, str]] = []
    for i, point in enumerate(points, 1):
        status, config, detail, secs = generate(point, args,
                                                os.path.join(sweep_dir, "logs"))
        base = dict(asdict(point), gen_seconds=round(secs, 2))
        base["cap_instances"] = point.cap_instances or ""
        if config is None:
            record(dict(base, status=status, config="", solver="", detail=detail))
            print(f"  [{i}/{len(points)}] {status:<12} {point.stem}")
            print(f"      {detail[:160]}")
            continue
        base.update(config_stats(os.path.join(REPO_ROOT, config)))
        runnable.append((point, config))
        print(f"  [{i}/{len(points)}] {status:<12} {point.stem} "
              f"({base['n_networks']} nets, {base['n_periodic']} periodic, "
              f"{base['n_edges']} edges)")
        point_base[point] = base
        if args.skip_schedule:
            # No solver rows are coming, so the config itself is the result.
            record(dict(base, status=status, config=config, solver=""))

    if args.skip_schedule:
        csv_file.close()
        finish(rows, args, sweep_dir, results_csv)
        return 0

    # ---- phase 2: schedule -------------------------------------------------
    jobs = [(p, cfg, solver) for p, cfg in runnable for solver in args.solver]
    print(f"\n[2/2] scheduling {len(jobs)} runs ({args.jobs} at a time)")
    tracker = Tracker(jobs, args.max_time)
    done = 0

    def work(job):
        point, config, solver = job
        tracker.start(point, solver)
        try:
            row = schedule(point, config, solver, args, sweep_dir, sched_python)
        except Exception as e:  # keep one bad run from killing the sweep
            row = {"config": config, "solver": solver, "status": "sweep-error",
                   "detail": f"{type(e).__name__}: {e}"}
        tracker.finish(point, solver, row)
        return point, row

    # Periodic "what is still running" block, so a long sweep is never silent.
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.wait(args.status_interval):
            tracker.print_status()

    monitor_thread = None
    if args.status_interval > 0 and jobs:
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # as_completed, not map: a finished run should print when it
            # finishes, not when every run submitted before it has too.
            futures = [pool.submit(work, job) for job in jobs]
            for future in as_completed(futures):
                point, row = future.result()
                done += 1
                merged = dict(point_base[point])
                merged.update(row)
                record(merged)
                mk = merged.get("makespan_ms", "")
                secs = merged.get("sched_seconds", "")
                print(f"  [{done}/{len(jobs)}] {str(merged['status']):<12} "
                      f"{merged['solver']:<16} {point.stem}"
                      + (f"  {hms(float(secs))}" if secs != "" else "")
                      + (f"  makespan={mk} ms" if mk != "" else "")
                      + (f"  -- {str(merged.get('detail', ''))[:120]}"
                         if merged.get("detail") else ""))
    finally:
        stop_monitor.set()
        if monitor_thread:
            monitor_thread.join(timeout=1)

    csv_file.close()
    finish(rows, args, sweep_dir, results_csv, tracker)
    return 0


def finish(rows: List[Dict[str, object]], args, sweep_dir: str,
           results_csv: str, tracker: Optional[Tracker] = None) -> None:
    results_json = os.path.join(sweep_dir, "results.json")
    with open(results_json, "w") as f:
        json.dump({
            "sweep": args.name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "axes": {
                "hardware": args.hardware,
                "seeds": args.seeds,
                "scale": [list(s) for s in args.scale],
                "period_headroom": args.period_headroom,
                "max_gap": args.max_gap,
                "num_instances": args.num_instances,
                "cap_instances": args.cap_instances,
                "solver": args.solver,
            },
            "rows": rows,
        }, f, indent=2)

    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "?"))
        counts[status] = counts.get(status, 0) + 1

    print("\nsummary")
    for status in sorted(counts):
        print(f"  {status:<12} {counts[status]}")

    if tracker is not None:
        tracker.print_table()

    scheduled = [r for r in rows
                 if r.get("status") in ("ok", "reused")
                 and isinstance(r.get("makespan_ms"), (int, float))]
    if scheduled:
        by_solver: Dict[str, List[float]] = {}
        for r in scheduled:
            by_solver.setdefault(str(r["solver"]), []).append(float(r["makespan_ms"]))
        print("\n  makespan (ms) by solver")
        for solver in sorted(by_solver):
            vals = sorted(by_solver[solver])
            mid = vals[len(vals) // 2]
            print(f"    {solver:<16} n={len(vals):<4} min={vals[0]:>10.2f} "
                  f"median={mid:>10.2f} max={vals[-1]:>10.2f}")

        worst = sorted(scheduled, key=lambda r: -float(r["makespan_ms"]))[:5]
        print("\n  longest makespans")
        for r in worst:
            print(f"    {float(r['makespan_ms']):>10.2f} ms  {r['solver']:<16} "
                  f"{os.path.basename(str(r['config']))}")

    print(f"\nresults : {os.path.relpath(results_csv, REPO_ROOT)}")
    print(f"          {os.path.relpath(results_json, REPO_ROOT)}")
    print(f"artifacts: {os.path.relpath(sweep_dir, REPO_ROOT)}/{{schedules,plots,logs}}")


if __name__ == "__main__":
    raise SystemExit(main())
