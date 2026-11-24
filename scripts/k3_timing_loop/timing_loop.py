#!/usr/bin/env python3
"""The K3 timing loop as ONE command: plan -> build -> board runs -> checks -> measure -> compare -> fix the
timings -> plan again, until plan and board agree or the round limit is reached.

It wraps the step scripts of this folder; the planner and ModelBlaster are used as they are.

  per round k
    tree     k = 0: the raw profile tree.  k > 0: fit.py over EVERY finished round, apply.py into a
             corrected copy of the raw tree, tree_audit.py (must PASS)
    plan     a copy of the workload config named after the round, solve.sh with the round's tree,
             plan_summary.py (what the planner predicts per model; what moved against the round before)
    board    job.sh: board free check and our lock, build with the stamps walker, N runs, md5-checked copy
    analyze  golden_c.sh (C-rounding goldens belong to this build) and analyze.py; every run must pass
             every check, otherwise the loop STOPS (it never fits on a bad round)
    verdict  stop_rule(): every model's board / planned frame time within +-delta AND no model that the
             plan puts on time has a late frame -> stop; else next round (up to --max-rounds)

Gap rule "layer" (default): each (model, layer, option) row that ran gets its own measured run time and its
own measured start gap; rows that never ran get a fall-back.
  Never forget a measurement: a row's run-time correction is the MEDIAN of its per-round values over
  every round it ran in (two rounds: their mean; one odd round cannot pull it far; a row the planner
  left keeps what was measured). Its gap is the one from the most recent round it ran in, not only the
  latest round: with "latest only" a layer's twin row lost its gap as soon as the layer moved off it,
  looked cheaper again, and the planner moved the layer back.
Gap rule "table" (--gap-rule table): the wake-ups are charged by the PLANNER, not stored on rows. The tree
holds the run-time corrections plus one flat same-thread start cost on every row; the loop measures a
four-number table (sender core kind -> receiver core kind, mean gap of hand-offs to another worker minus
the flat cost, pooled over the models and over every finished round; handoff_table.py) and the sender's
delay (the worker that woke another thread starts its own next layer late), writes them into the round's
workload config as scheduler.handoff_ms and scheduler.sender_ms, keeps the table in a file, checks it
against every round and re-measures a value that drifted. A hand-off pair without a sample stops the loop.

State: <out>/loop_state.json lists every round's tree, plan, raw folder, summary and verdict, and is
rewritten after each finished step. Running the same command again continues after the last finished
step: a finished board round is never repeated, and a round whose runs are on the board but not yet
copied back is only fetched. --start-from adopts rounds that were run before.

Paths: relative paths are relative to the folder the command is run from; that folder is also what the
paths written into the state and the result files are relative to. Everything the loop writes goes under
--out unless --results-root, --raw-root, --tree-root or --solve-dir say otherwise.

usage:
  timing_loop.py --config <workload.json>     any workload; starts from the profiled times in gen/k3_raw_profiles,
                                              gap rule table, results in ./k3tl_<workload>
  timing_loop.py --config <workload.json> [--out <loop folder>] [--raw-tree <raw profile tree>] \\
      [--runs 20] [--max-rounds 5] [--delta 0.03] [--gap-rule table|layer] [--prefix r] [--models a,b,c] \\
      [--modelblaster <checkout with the stamps walker>] [--host k3w] [--start-from <raw folder>[,<tree>] ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))          # the XPU-RT checkout this file is in (the planner used)
BASE = os.getcwd()                                           # relative paths, in and out, are relative to this folder
DEFAULT_RAW_TREE = os.path.join(REPO, "gen/k3_raw_profiles")  # per-layer times profiled on the K3, every option
SCHEMA = "k3work.timing_loop/v1"                             # state file format (name kept so earlier loop folders continue)
TREE_ENTRIES = "t023_tree_entries.csv"                       # written into a corrected tree by apply.py
STEPS = ("tree", "plan", "board", "analyze", "verdict")


# ------------------------------------------------------------------ the stop rule (pure)
def stop_rule(models: Dict[str, dict], delta: float, abs_tol_ms: float = 0.0) -> dict:
    """models: {name: {planned_ms, board_ms, late_frames, planned_late_frames}} for one round.

    A model agrees when |board / planned - 1| <= delta (or, if abs_tol_ms > 0, when board and plan differ by
    no more than abs_tol_ms: a frame of a few microseconds cannot be judged in percent).
    A model is clean when the plan puts none of its frames late and the board had no late frame either;
    a model the plan itself puts late is not held against the loop here (the plan predicts the misses).
    Converged = every model agrees and is clean."""
    per, reasons = {}, []
    for name in sorted(models):
        m = models[name]
        planned, board = float(m["planned_ms"]), float(m["board_ms"])
        ratio = board / planned if planned > 0 else float("inf")
        agrees = abs(ratio - 1.0) <= delta + 1e-9 or (abs_tol_ms > 0 and abs(board - planned) <= abs_tol_ms + 1e-12)
        plan_on_time = int(m.get("planned_late_frames", 0)) == 0
        clean = not plan_on_time or int(m.get("late_frames", 0)) == 0
        per[name] = dict(board_over_planned=ratio, agrees=bool(agrees), plan_on_time=plan_on_time, clean=bool(clean))
        if not agrees:
            reasons.append(f"{name}: board / planned = {ratio:.3f}, outside +-{100 * delta:.0f}%")
        if not clean:
            reasons.append(f"{name}: the plan puts it on time but the board had {int(m['late_frames'])} late frame(s)")
    return dict(converged=not reasons, reasons=reasons, per_model=per)


# ------------------------------------------------------------------ the kept hand-off table and its drift check (pure)
PAIR_NAMES = {("cpu_p", "cpu_p"): "X100 -> X100", ("cpu_p", "cpu_e"): "X100 -> A100", ("cpu_e", "cpu_p"): "A100 -> X100", ("cpu_e", "cpu_e"): "A100 -> A100"}


def table_drift(table_ms: Dict[str, Dict[str, float]], flat_ms: float, measured: Dict[str, dict], drift_us: float = 2.0,
                min_samples: int = 200) -> Dict[str, dict]:
    """Compare one round's measured hand-off gaps with the table the plan was made with.

    table_ms: {from kind: {to kind: ms}} as given to the planner; flat_ms: the same-thread start cost in the tree.
    measured: {"X100 -> A100": {n, mean_gap_us, median_gap_us}, ...} from that round (all four pairs must be present
    as keys; a pair that did not occur has n = 0).
    Per pair -> {table_us, measured_mean_us, measured_median_us, n, off_by_us, verdict} with verdict
      "kept"             enough samples, mean within drift_us of the table value (+ the flat cost)
      "updated"          enough samples and off by more than drift_us: re-measure this pair
      "too few samples"  fewer than min_samples: nothing to judge, the table value stays
    A pair the board used (n > 0) for which the table has no value is an error, never a guess."""
    out = {}
    for (src, dst), name in PAIR_NAMES.items():
        if name not in measured:
            raise ValueError(f"the round's measurement has no entry for the hand-off pair {name}")
        m = measured[name]
        val = (table_ms.get(src) or {}).get(dst)
        if val is None:
            if int(m["n"]) > 0:
                raise ValueError(f"the table has no value for {name}, but the board made {m['n']} such hand-offs")
            out[name] = dict(table_us=None, measured_mean_us=None, measured_median_us=None, n=0, off_by_us=None, verdict="too few samples")
            continue
        charged = (val + flat_ms) * 1e3
        if int(m["n"]) < min_samples:
            out[name] = dict(table_us=val * 1e3, measured_mean_us=(m["mean_gap_us"] if m["n"] else None),
                             measured_median_us=(m["median_gap_us"] if m["n"] else None), n=int(m["n"]), off_by_us=None, verdict="too few samples")
            continue
        off = float(m["mean_gap_us"]) - charged
        out[name] = dict(table_us=val * 1e3, measured_mean_us=float(m["mean_gap_us"]), measured_median_us=float(m["median_gap_us"]), n=int(m["n"]),
                         off_by_us=off, verdict="updated" if abs(off) > drift_us + 1e-9 else "kept")
    return out


SENDER_NAMES = {"cpu_p": "X100", "cpu_e": "A100"}


def sender_drift(sender_ms: Dict[str, float], flat_ms: float, measured: Dict[str, dict], drift_us: float = 2.0,
                 min_samples: int = 200) -> Dict[str, dict]:
    """The same check for the sender's delay (the worker that woke another thread starts its own next layer late).
    sender_ms: {core kind: ms} as given to the planner; measured: {"X100": {n, mean_gap_us, median_gap_us}, "A100": ...}
    for one round (both kinds must be present as keys). Verdicts as in table_drift."""
    out = {}
    for kind, name in SENDER_NAMES.items():
        if name not in measured:
            raise ValueError(f"the round's measurement has no entry for the sender kind {name}")
        m, val = measured[name], (sender_ms or {}).get(kind)
        n = int(m["n"])
        if val is None:
            if n >= min_samples:
                raise ValueError(f"the table has no sender's delay for {name}, but the board showed {n} such starts")
            out[name] = dict(table_us=None, measured_mean_us=None, measured_median_us=None, n=n, off_by_us=None, verdict="too few samples")
        elif n < min_samples:
            out[name] = dict(table_us=val * 1e3, measured_mean_us=(m["mean_gap_us"] if n else None), measured_median_us=(m["median_gap_us"] if n else None),
                             n=n, off_by_us=None, verdict="too few samples")
        else:
            off = float(m["mean_gap_us"]) - (val + flat_ms) * 1e3
            out[name] = dict(table_us=val * 1e3, measured_mean_us=float(m["mean_gap_us"]), measured_median_us=float(m["median_gap_us"]), n=n,
                             off_by_us=off, verdict="updated" if abs(off) > drift_us + 1e-9 else "kept")
    return out


# ------------------------------------------------------------------ layout and state
class Layout:
    """Where each round's files live. Round k has the tag <prefix><k>."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = os.path.abspath(a.out)
        self.results_root = os.path.abspath(a.results_root or os.path.join(self.out, "results"))
        self.raw_root = os.path.abspath(a.raw_root or os.path.join(self.out, "raw"))
        self.tree_root = os.path.abspath(a.tree_root or os.path.join(self.out, "trees"))
        self.solve_dir = os.path.abspath(a.solve_dir or os.path.join(self.out, "solve"))
        self.stem = a.workload_name or re.sub(r"^networks_", "", os.path.splitext(os.path.basename(a.config))[0])
        self.name = re.sub(r"[^A-Za-z0-9_]", "_", a.name or os.path.basename(self.out))     # prefix of build and board folder names
        self.modelblaster = os.path.abspath(a.modelblaster or os.path.join(REPO, "zephyr-chipyard-sw/modelblaster"))

    def tag(self, k: int) -> str:
        return f"{self.a.prefix}{k}"

    def raw(self, k: int) -> str:
        return os.path.join(self.raw_root, f"{self.tag(k)}_stamps")

    def job(self, k: int) -> str:
        """Name of round k's board job: its build folder in ModelBlaster and its folder on the board."""
        return f"{self.name}_{self.tag(k)}"

    def build(self, k: int) -> str:
        return os.path.join(self.modelblaster, "build", f"k3_timing_loop_{self.job(k)}")       # job.sh's OUT_ROOT

    def tree(self, k: int) -> str:
        return os.path.abspath(self.a.raw_tree) if k == 0 else os.path.join(self.tree_root, f"{self.a.tree_prefix}{self.tag(k)}")

    def corr_dir(self, k: int) -> str:
        return os.path.join(self.results_root, f"corr_{self.tag(k)}")

    def results(self, k: int) -> str:
        return os.path.join(self.results_root, self.tag(k))

    def config_name(self, k: int) -> str:
        return f"networks_{self.stem}_{self.tag(k)}"

    def config(self, k: int) -> str:
        return os.path.join(self.solve_dir, self.config_name(k) + ".json")

    def schedule(self, k: int) -> str:
        return os.path.join(self.solve_dir, "schedules", f"scheduled_{self.config_name(k)}_decomposed_profiled.json")

    def solve_log(self, k: int) -> str:
        return os.path.join(self.solve_dir, "logs", f"solve_{self.config_name(k)}_decomposed.log")


def rel(path: str) -> str:
    return os.path.relpath(path, BASE) if os.path.isabs(path) else path


def save_state(path: str, state: dict) -> None:
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


class LoopStop(Exception):
    """A check failed: the loop stops here (fail closed)."""


# ------------------------------------------------------------------ the real steps
class Steps:
    def __init__(self, lay: Layout):
        self.lay, self.a = lay, lay.a
        self.env = dict(os.environ, K3TL_MODELBLASTER=lay.modelblaster)
        if self.a.models:
            self.env["K3TL_NETS"] = self.a.models
            self.env["K3TL_MODELS"] = self.a.models.replace(",", " ")

    def sh(self, cmd: List[str], log: str, env: Optional[dict] = None) -> int:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            return subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT, env=env or self.env).returncode

    def py(self, script: str, args: List[str], log: str) -> int:
        return self.sh([sys.executable, os.path.join(HERE, script)] + args, log)

    # -- tree
    def make_tree(self, k: int, rounds: List[dict]) -> dict:
        lay = self.lay
        table_rule = self.a.gap_rule == "table"
        cdir, tree = lay.corr_dir(k), lay.tree(k)
        os.makedirs(cdir, exist_ok=True)
        log, corr = os.path.join(lay.out, "logs", f"{lay.tag(k)}_tree.log"), os.path.join(cdir, "corrections.json")
        args = ["--raw-tree", rel(self.a.raw_tree), "--gap-from", "seen", "--out", rel(corr)]
        for r in rounds:
            args += ["--round", f"{r['tag']}={rel(r['raw'])}:{r['runs']}"]
        if self.py("fit.py", args, log) != 0:
            raise LoopStop(f"round {k}: the fit failed, see {rel(log)}")
        extra, info = [], {}
        if table_rule:
            t = self.table_for_round(k, rounds, log)
            extra = ["--flat-gap-ms", repr(t["flat_start_cost_ms"])]
            info = dict(handoff_table=rel(os.path.join(cdir, "handoff_table.json")), handoff_ms=t["handoff_ms"],
                        sender_ms=(None if self.a.no_sender else t.get("sender_ms")),
                        flat_start_cost_ms=t["flat_start_cost_ms"], table_source=t["source"], table_updates=t.get("updated_pairs", []),
                        table_samples={k2: v["n"] for k2, v in t.get("pairs", {}).items()})
        if os.path.isdir(tree):                      # left over from an interrupted step: ours, rebuilt from scratch
            shutil.rmtree(tree)
        if self.py("apply.py", ["--raw-tree", rel(self.a.raw_tree), "--corrections", rel(corr), "--out", rel(tree)] + extra, log) != 0:
            raise LoopStop(f"round {k}: building the corrected tree failed, see {rel(log)}")
        audit = ["--profile-audit", self.a.profile_audit] if self.a.profile_audit else []
        if self.py("tree_audit.py", ["--raw-tree", rel(self.a.raw_tree), "--tree", rel(tree), "--corrections", rel(corr)] + audit, log) != 0:
            raise LoopStop(f"round {k}: the tree audit did not pass, see {rel(log)}")
        doc = json.load(open(corr))
        return dict(tree=rel(tree), corrections=rel(corr), rows_measured=len(doc["ops"]), fitted_from=[r["tag"] for r in rounds],
                    gap_from=("planner table" if table_rule else doc.get("gap_from")), audit="PASS", **info)

    # -- the kept hand-off table (gap rule "table")
    def measure_table(self, rounds: List[dict], path: str, log: str, allow_missing: bool = False) -> dict:
        targs = ["--out", rel(path)] + (["--allow-missing"] if allow_missing else [])
        for r in rounds:
            targs += ["--round", f"{r['tag']}={rel(r['raw'])}:{r['runs']}"]
        if self.py("handoff_table.py", targs, log) != 0:
            raise LoopStop(f"the hand-off table could not be measured from {[r['tag'] for r in rounds]} (a pair without a sample?), see {rel(log)}")
        return json.load(open(path))

    def check_round_against_table(self, k: int, raw: str, log: str) -> Optional[dict]:
        """Drift check of round k: its own measured gaps against the table its plan was made with (None: no table)."""
        used = os.path.join(self.lay.corr_dir(k), "handoff_table.json")
        if not os.path.exists(used):
            return None
        t = json.load(open(used))
        m = self.measure_table([dict(tag=self.lay.tag(k), raw=raw, runs=self.a.runs)], os.path.join(self.lay.results(k), "handoff_vs_plan.json"), log, True)
        try:
            out = dict(pairs=table_drift(t["handoff_ms"], t["flat_start_cost_ms"], m["pairs"], self.a.table_drift_us, self.a.table_min_samples),
                       per_model=m["per_round"][self.lay.tag(k)]["models"])
            if t.get("sender_ms") and not self.a.no_sender:
                out["sender"] = sender_drift(t["sender_ms"], t["flat_start_cost_ms"], m["sender"], self.a.table_drift_us, self.a.table_min_samples)
            return out
        except ValueError as e:
            raise LoopStop(f"round {k}: {e}")

    def table_for_round(self, k: int, rounds: List[dict], log: str) -> dict:
        """The table round k is planned with: the kept one (loop folder), else --handoff-table, else measured from the
        rounds so far. Pairs that the last round's drift check marked "updated" take the new pooled value (unless --table-frozen)."""
        lay = self.lay
        kept = os.path.join(lay.out, "board_table.json")
        cdir = lay.corr_dir(k)
        before = os.path.join(lay.corr_dir(k - 1), "handoff_table.json")      # a loop started before the table was a kept file
        if os.path.exists(kept) or os.path.exists(before):
            t = json.load(open(kept if os.path.exists(kept) else before))
            t["source"], t["updated_pairs"] = "kept from the round before", []
            prev = lay.tag(k - 1)
            check_path = os.path.join(lay.results(k - 1), "table_check.json")
            if not os.path.exists(check_path):           # a round analysed before this check existed
                c = self.check_round_against_table(k - 1, rounds[-1]["raw"], log)
                if c:
                    with open(check_path, "w") as f:
                        json.dump(c, f, indent=1)
            chk = json.load(open(check_path)) if os.path.exists(check_path) else {}
            drifted = [n for n, v in chk.get("pairs", {}).items() if v["verdict"] == "updated"]
            s_drift = [n for n, v in chk.get("sender", {}).items() if v["verdict"] == "updated"]
            need_sender = not self.a.no_sender and not t.get("sender_ms")
            if (drifted or s_drift or need_sender) and not (self.a.table_frozen and not need_sender):
                pooled = self.measure_table(rounds, os.path.join(cdir, "handoff_pooled.json"), log, True)
                if need_sender:                                      # the table had no sender's delay yet: measured from every round so far
                    t["sender_ms"], t["sender"] = pooled["sender_ms"], pooled["sender"]
                    t["sender_by_threads_woken"] = pooled["sender_by_threads_woken"]
                    t["updated_pairs"].append("sender's delay (first measured)")
                for name in ([] if self.a.table_frozen else s_drift):
                    key = pooled["sender"][name]["planner_key"]
                    if pooled["sender"][name]["delay_ms"] is not None:
                        t["sender_ms"][key] = round(pooled["sender"][name]["mean_gap_us"] / 1e3 - t["flat_start_cost_ms"], 6)
                        t["updated_pairs"].append(f"sender's delay {name}")
                for name in ([] if self.a.table_frozen else drifted):
                    src, dst = pooled["pairs"][name]["planner_key"]
                    if pooled["pairs"][name]["table_ms"] is not None:
                        t["handoff_ms"][src][dst] = round(pooled["pairs"][name]["mean_gap_us"] / 1e3 - t["flat_start_cost_ms"], 6)
                        t["updated_pairs"].append(name)
                t["source"] = f"kept; re-measured after round {prev}: {', '.join(t['updated_pairs'])}" if t["updated_pairs"] else t["source"]
                t["pairs"] = pooled["pairs"]
                t["rounds"] = pooled["rounds"]
        elif self.a.handoff_table:
            t = json.load(open(self.a.handoff_table))
            if "flat_start_cost_ms" not in t and "floor_ms" in t:
                t["flat_start_cost_ms"] = t["floor_ms"]
            t["source"], t["updated_pairs"] = f"given: {rel(os.path.abspath(self.a.handoff_table))}", []
        else:
            t = self.measure_table(rounds, os.path.join(cdir, "handoff_measured.json"), log)
            t["source"], t["updated_pairs"] = "measured from rounds " + ", ".join(r["tag"] for r in rounds), []
        t["valid_for"] = self.a.table_note
        t["used_for_round"] = lay.tag(k)
        t.pop("per_round", None)
        for path in [os.path.join(cdir, "handoff_table.json"), kept] + ([self.a.save_table] if self.a.save_table else []):
            with open(path, "w") as f:
                json.dump(t, f, indent=1)
                f.write("\n")
        return t

    # -- plan
    def write_config(self, k: int) -> str:
        base = json.load(open(self.a.config))
        base_dir = os.path.dirname(os.path.abspath(self.a.config))
        for name, v in base["networks"].items():
            p = v["dispatch_deps_path"]
            if not os.path.isabs(p):
                hit = [c for c in (os.path.normpath(os.path.join(r, p)) for r in (base_dir, REPO, BASE)) if os.path.exists(c)]
                if not hit:
                    raise LoopStop(f"{name}: cannot resolve dispatch_deps_path {p}")
                v["dispatch_deps_path"] = hit[0]
        tpath = os.path.join(self.lay.corr_dir(k), "handoff_table.json")
        if self.a.gap_rule == "table" and k > 0:
            if not os.path.exists(tpath):
                raise LoopStop(f"round {k}: no hand-off table at {rel(tpath)}")
            t = json.load(open(tpath))
            base.setdefault("scheduler", {})["handoff_ms"] = t["handoff_ms"]
            snd = {kind: v for kind, v in (t.get("sender_ms") or {}).items() if v is not None}
            if snd and not self.a.no_sender:       # the worker that wakes another thread starts its own next layer this much later
                base["scheduler"]["sender_ms"] = snd
        base["_comment"] = f"timing loop round {self.lay.tag(k)}: copy of {rel(os.path.abspath(self.a.config))} with absolute dispatch_deps_path; nothing else changed"
        dst = self.lay.config(k)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            json.dump(base, f, indent=1)
        return dst

    def solve(self, k: int, tree: str) -> None:
        log = os.path.join(self.lay.out, "logs", f"{self.lay.tag(k)}_solve.log")
        # The planner is this checkout. By default the schedule picture is not drawn: drawing takes about 40 of the
        # 47 minutes of a 3-model solve and the plan file is byte-identical without it (run_schedule_noplot.py;
        # --plot draws it).
        cmd = ["bash", os.path.join(HERE, "solve.sh"), self.lay.config(k), tree, self.lay.solve_dir]
        env = dict(self.env, K3TL_PLOT="1" if self.a.plot else "0")
        if self.sh(cmd, log, env) != 0 or not os.path.exists(self.lay.schedule(k)):
            raise LoopStop(f"round {k}: the solve failed, see {rel(log)} and {rel(self.lay.solve_log(k))}")

    def plan(self, k: int, tree: str, prev_schedule: Optional[str]) -> dict:
        lay = self.lay
        cfg = self.write_config(k)
        if not os.path.exists(lay.schedule(k)):
            self.solve(k, tree)
        return self.describe_plan(k, lay.schedule(k), cfg, tree, prev_schedule, lay.solve_log(k))

    def describe_plan(self, k: int, schedule: str, cfg: str, tree: str, prev_schedule: Optional[str], solve_log: Optional[str]) -> dict:
        lay = self.lay
        odir = lay.corr_dir(k) if k > 0 else lay.results(k)
        tag = f"plan_{lay.tag(k)}" + (f"_vs_{lay.tag(k - 1)}" if prev_schedule else "")
        args = ["--new", rel(schedule), "--config", rel(cfg), "--out", rel(odir), "--tag", tag]
        if prev_schedule:
            args += ["--old", rel(prev_schedule)]
        if k > 0 and os.path.exists(os.path.join(tree, TREE_ENTRIES)):
            args += ["--tree", rel(tree)]
        if solve_log and os.path.exists(solve_log):
            args += ["--solve-log", rel(solve_log)]
        if self.py("plan_summary.py", args, os.path.join(lay.out, "logs", f"{lay.tag(k)}_plan.log")) != 0:
            raise LoopStop(f"round {k}: reading the plan failed")
        d = json.load(open(os.path.join(odir, tag + ".json")))
        out = dict(schedule=rel(schedule), config=rel(cfg), solver_says=d.get("solver_says"), plan_summary=rel(os.path.join(odir, tag + ".json")), models={})
        for net, b in d["models"].items():
            out["models"][net] = dict(planned_frame_ms=b["planned_frame_ms"]["median"], planned_late_frames=b["planned_late_frames"],
                                      layers=b.get("layers"), layers_moved=b.get("layers_whose_main_option_changed"),
                                      dispatches_moved_share=b.get("dispatches_whose_option_changed_share"))
        out["rows_used"], out["rows_used_still_guessed"] = self.guessed_rows(schedule, tree) if k > 0 else (None, None)
        return out

    @staticmethod
    def guessed_rows(schedule: str, tree: str):
        """(rows the plan uses, how many of them have a fall-back run time): distinct (model, layer, option)."""
        path = os.path.join(tree, TREE_ENTRIES)
        if not os.path.exists(path):
            return None, None
        src = {(r["network"], int(r["layer"]), r["core_kind"], r["kernel_type"], int(r["cores"])): r["source"].split()[0]
               for r in csv.DictReader(open(path))}
        kind = {"CPU_P": "x100", "CPU_E": "a100"}
        used = set()
        for key, v in json.load(open(schedule))["dispatches"].items():
            if float(v["duration"]) <= 0:
                continue
            m = re.match(r"^(.*?)(\d+)_dispatch_(\d+)$", key)
            t = v["hardware_target"].split("+")
            used.add((m.group(1), int(m.group(3)), kind[t[0].split("#")[0]], v.get("impl", ""), len(t)))
        return len(used), sum(1 for u in used if src.get(u, "measured") != "measured")

    # -- board
    def traces_complete(self, raw: str) -> bool:
        return all(os.path.exists(os.path.join(raw, f"trace_r{i}.csv")) and os.path.getsize(os.path.join(raw, f"trace_r{i}.csv")) > 0
                   for i in range(1, self.a.runs + 1))

    def board(self, k: int, schedule: str, cfg: str) -> dict:
        lay = self.lay
        raw = lay.raw(k)
        log = os.path.join(lay.out, "logs", f"{lay.tag(k)}_board.log")
        attempt, waited_until = 0, time.time() + 3600 * self.a.board_wait_h
        while attempt < self.a.board_attempts:
            attempt += 1
            if self.traces_complete(raw):
                break
            env = dict(self.env, K3TL_HOST=self.a.host, STALL_S=str(self.a.stall_s))
            if self.a.owner:
                env["K3TL_OWNER"] = self.a.owner
            old_cfg = os.path.join(raw, "config.txt")
            if os.path.exists(old_cfg):                 # an earlier attempt: never repeat board runs that already happened
                text = open(old_cfg).read()
                shutil.copyfile(old_cfg, os.path.join(raw, f"config_attempt_{time.strftime('%H%M%S')}.txt"))
                if "binary md5 (local build)" in text:
                    env["K3TL_REUSE_BUILD"] = "1"
                if re.search(r"^batch: (LAUNCHED|STATUS: DONE)", text, re.M):
                    env["K3TL_REUSE_BUILD"] = env["K3TL_FETCH_ONLY"] = "1"
            rc = self.sh(["bash", os.path.join(HERE, "job.sh"), lay.job(k), rel(raw), rel(schedule), rel(cfg), str(self.a.runs), "1"], log, env)
            if rc == 0 and self.traces_complete(raw):
                break
            if rc == 5:                                  # board taken (another loop, a colleague): wait and ask again, never force
                if time.time() > waited_until:
                    raise LoopStop(f"round {k}: the board was not free for {self.a.board_wait_h} h (see {rel(log)}); run the same command again later")
                attempt -= 1
                time.sleep(60)
        if not self.traces_complete(raw):
            raise LoopStop(f"round {k}: the board job did not bring back {self.a.runs} runs after {self.a.board_attempts} attempts, see {rel(log)}; "
                           f"run the same command again to continue")
        cfg_txt = open(os.path.join(raw, "config.txt")).read()
        if "MISMATCH" in cfg_txt:
            raise LoopStop(f"round {k}: md5 mismatch in the copied runs or the binary, see {rel(raw)}/config.txt")
        return dict(raw=rel(raw), runs=self.a.runs)

    # -- analyze
    def analyze(self, k: int, raw: str, tree: str) -> dict:
        lay = self.lay
        log = os.path.join(lay.out, "logs", f"{lay.tag(k)}_analyze.log")
        build = lay.build(k)                          # absent for an adopted round: its C-rounding goldens must be there already
        golden = os.path.join(os.path.dirname(os.path.abspath(raw)), "_golden_check", "c_float")       # where loop_lib looks
        if os.path.isdir(build) and self.sh(["bash", os.path.join(HERE, "golden_c.sh"), golden, build], log) != 0:
            raise LoopStop(f"round {k}: the C-rounding goldens do not belong to this build, see {rel(log)}")
        label = "before" if k == 0 else f"after {k}"
        rc = self.py("analyze.py", ["--raw", rel(raw), "--reps", str(self.a.runs), "--out", rel(lay.results(k)), "--label", label,
                                         "--profile-root", rel(tree)], log)
        spath = os.path.join(lay.results(k), "summary.json")
        if not os.path.exists(spath):
            raise LoopStop(f"round {k}: the analysis failed, see {rel(log)}")
        out = self.read_summary(k, spath, rc)
        if self.a.gap_rule == "table":               # what the plan charged against what the board paid, and what the table leaves out
            c = self.check_round_against_table(k, raw, log)
            if c:
                with open(os.path.join(lay.results(k), "table_check.json"), "w") as f:
                    json.dump(c, f, indent=1)
                out["table_check"] = c["pairs"]
                if "sender" in c:
                    out["sender_check"] = c["sender"]
                for net, m in c["per_model"].items():
                    out["models"][net].update(handoffs_per_frame_plan=m["handoffs_per_frame_plan"],
                                              sender_delay_us_per_frame=m["sender_delay"]["us_per_frame"],
                                              sender_delay_per_frame=m["sender_delay"]["per_frame"],
                                              model_switch_us_per_frame=m["model_switch"]["us_per_frame"],
                                              model_switch_per_frame=m["model_switch"]["per_frame"])
        return out

    def read_summary(self, k: int, spath: str, rc: int = 0) -> dict:
        s = json.load(open(spath))
        if rc != 0 or s["runs_ok"] != s["runs"]:
            raise LoopStop(f"round {k}: {s['runs'] - s['runs_ok']} of {s['runs']} runs failed a check (runs {s['flagged_runs']}); "
                           f"nothing is fitted on this round. See {rel(os.path.dirname(spath))}/runs.csv")
        tc = s.get("plan_vs_profile_tree")
        if tc and tc["max_abs_diff_us"] > 0.01:
            raise LoopStop(f"round {k}: the plan's durations are not the tree's (max difference {tc['max_abs_diff_us']:.3f} us)")
        out = dict(summary=rel(spath), runs=s["runs"], runs_ok=s["runs_ok"], runs_with_any_late_frame=s["runs_with_any_late_frame"],
                   worst_lateness_ms=s["worst_lateness_ms"], models={})
        for net, b in s["models"].items():
            out["models"][net] = dict(planned_ms=b["planned_frame_ms"], board_ms=b["board_frame_p50_ms"], board_p75_ms=b["board_frame_p75_ms"],
                                      board_max_ms=b["board_frame_max_ms"], board_over_planned=b["board_over_planned"],
                                      late_frames=b["late_frames_total"], runs_with_a_late_frame=b["runs_with_a_late_frame"],
                                      planned_late_frames=b["planned_late_frames_per_run"], handoffs_per_frame=b["handoffs_per_frame"],
                                      discrepancy_ms=b["discrepancy_p50_ms"])
        return out


# ------------------------------------------------------------------ the loop
def new_round(k: int, tag: str) -> dict:
    return dict(index=k, tag=tag, steps={})


def run_loop(a: argparse.Namespace, steps: Optional[Steps] = None, say=print) -> dict:
    lay = Layout(a)
    steps = steps or Steps(lay)
    os.makedirs(lay.out, exist_ok=True)
    spath = os.path.join(lay.out, "loop_state.json")
    if os.path.exists(spath):
        state = json.load(open(spath))
        if state.get("schema") != SCHEMA:
            raise SystemExit(f"{spath}: unknown schema")
        say(f"continuing {rel(spath)}: {len(state['rounds'])} round(s) on record, status {state['status']}")
    else:
        state = dict(schema=SCHEMA, created=time.strftime("%Y-%m-%d %H:%M:%S"), rounds=[], status="running", message="",
                     args={k: (rel(v) if isinstance(v, str) and os.path.exists(v) else v) for k, v in vars(a).items()})
        for i, spec in enumerate(a.start_from or []):          # rounds that were run by hand: adopted, not re-run
            raw, _, tree = spec.partition(",")
            tree = tree or lay.tree(i)
            r = new_round(i, lay.tag(i))
            r["adopted"] = True
            if not steps.traces_complete(os.path.join(BASE, raw) if not os.path.isabs(raw) else raw):
                raise SystemExit(f"--start-from {raw}: not a finished round ({a.runs} runs expected)")
            r["steps"]["tree"] = dict(done=True, tree=rel(tree), adopted=True)
            r["steps"]["plan"] = dict(done=False, adopted_schedule=os.path.join(rel(raw), "schedule.json"),
                                      adopted_config=os.path.join(rel(raw), "config.json"))
            r["steps"]["board"] = dict(done=True, raw=rel(raw), runs=a.runs, adopted=True)
            state["rounds"].append(r)
        save_state(spath, state)
    state["status"], state["message"] = "running", ""

    def absp(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(BASE, p)

    try:
        k = 0
        while True:
            if k == len(state["rounds"]):
                state["rounds"].append(new_round(k, lay.tag(k)))
            r = state["rounds"][k]
            st = r["steps"]
            prev = state["rounds"][k - 1] if k > 0 else None
            # tree
            if not st.get("tree", {}).get("done"):
                say(f"[{r['tag']}] tree: " + ("the raw tree" if k == 0 else f"fit over {k} round(s), corrected copy, audit"))
                if k == 0:
                    st["tree"] = dict(done=True, tree=rel(lay.tree(0)))
                else:
                    done = [dict(tag=x["tag"], raw=x["steps"]["board"]["raw"], runs=x["steps"]["board"]["runs"]) for x in state["rounds"][:k]]
                    st["tree"] = dict(done=True, **steps.make_tree(k, done))
                save_state(spath, state)
            tree = absp(st["tree"]["tree"])
            # plan
            if not st.get("plan", {}).get("done"):
                prev_sched = absp(prev["steps"]["plan"]["schedule"]) if prev else None
                if "adopted_schedule" in st.get("plan", {}):
                    say(f"[{r['tag']}] plan: adopted {st['plan']['adopted_schedule']}")
                    info = steps.describe_plan(k, absp(st["plan"]["adopted_schedule"]), absp(st["plan"]["adopted_config"]), tree, prev_sched, None)
                    info["adopted"] = True
                else:
                    say(f"[{r['tag']}] plan: solving with {rel(tree)} (this takes a while)")
                    info = steps.plan(k, tree, prev_sched)
                st["plan"] = dict(done=True, **info)
                save_state(spath, state)
            # board
            if not st.get("board", {}).get("done"):
                say(f"[{r['tag']}] board: build, {a.runs} runs")
                st["board"] = dict(done=True, **steps.board(k, absp(st["plan"]["schedule"]), absp(st["plan"]["config"])))
                save_state(spath, state)
            # analyze (also for adopted rounds: the loop judges every round with the same code)
            if not st.get("analyze", {}).get("done"):
                say(f"[{r['tag']}] analyze: checks and tables")
                st["analyze"] = dict(done=True, **steps.analyze(k, absp(st["board"]["raw"]), tree))
                save_state(spath, state)
            # verdict
            if not st.get("verdict", {}).get("done"):
                v = stop_rule(st["analyze"]["models"], a.delta, a.abs_tol_ms)
                last = k + 1 >= a.max_rounds
                v["decision"] = "stop: plan and board agree" if v["converged"] else ("stop: round limit reached" if last else "next round")
                st["verdict"] = dict(done=True, **v)
                save_state(spath, state)
                say(f"[{r['tag']}] verdict: {v['decision']}" + ("" if v["converged"] else " (" + "; ".join(v["reasons"]) + ")"))
            if st["verdict"]["decision"] == "stop: round limit reached" and k + 1 < a.max_rounds:
                st["verdict"]["decision"] = "next round"          # the same command, called again with a higher --max-rounds
                save_state(spath, state)
            write_report(lay, state)
            if st["verdict"]["decision"] != "next round":
                state["status"], state["message"] = "stopped", st["verdict"]["decision"]
                break
            k += 1
    except LoopStop as e:
        state["status"], state["message"] = "failed", str(e)
        say(f"STOPPED: {e}")
    finally:
        save_state(spath, state)
        write_report(lay, state)
    return state


def write_report(lay: Layout, state: dict) -> str:
    nets = sorted({n for r in state["rounds"] for n in r["steps"].get("analyze", {}).get("models", {})})
    lines = [f"# Timing loop: {lay.stem}", "",
             f"Status: **{state['status']}**" + (f" ({state['message']})" if state.get("message") else "") + f". Updated {time.strftime('%Y-%m-%d %H:%M')}.",
             f"Stop rule: every model's board / planned frame time within +-{100 * lay.a.delta:.0f}% and no late frame for a model the plan puts on time; "
             f"at most {lay.a.max_rounds} rounds; {lay.a.runs} board runs per round. Board numbers are measured; planned numbers come from the planner.", ""]
    head = ["round", "timings from"] + [f"{n}: planned / board ms (board / planned); late frames" for n in nets] + \
           ["runs with a late frame", "layers moved", "rows measured so far", "rows the plan uses that are still guessed", "verdict"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in state["rounds"]:
        st = r["steps"]
        an, pl, tr = st.get("analyze", {}), st.get("plan", {}), st.get("tree", {})
        cells = [r["tag"] + (" (adopted)" if r.get("adopted") else ""),
                 "raw profiles" if r["index"] == 0 else "rounds " + ", ".join(tr.get("fitted_from", [])) if tr.get("fitted_from") else rel(tr.get("tree", "?"))]
        for n in nets:
            m = an.get("models", {}).get(n)
            p = pl.get("models", {}).get(n, {})
            if m:
                cells.append(f"{m['planned_ms']:.3f} / {m['board_ms']:.3f} ({m['board_over_planned']:.3f}); {m['late_frames']}"
                             + (f" (plan: {m['planned_late_frames']} per run)" if m["planned_late_frames"] else ""))
            elif p:
                cells.append(f"{p['planned_frame_ms']:.3f} / not run yet")
            else:
                cells.append("")
        moved = [f"{n.split('_')[0]} {p.get('layers_moved')}/{p.get('layers')}" for n, p in pl.get("models", {}).items() if p.get("layers_moved") is not None]
        cells += [f"{an['runs_with_any_late_frame']} of {an['runs']}" if an else "", ", ".join(moved) or "", str(tr.get("rows_measured", "")) if r["index"] else "",
                  (f"{pl['rows_used_still_guessed']} of {pl['rows_used']}" if pl.get("rows_used") else ""), st.get("verdict", {}).get("decision", "")]
        lines.append("| " + " | ".join(cells) + " |")
    if lay.a.gap_rule == "table":
        lines += ["", "Gap rule: the planner's hand-off table. Tree = run-time corrections + one flat start cost; table = mean gap of hand-offs to "
                  "another worker by core kinds, minus the flat cost, from every finished round.", "",
                  "| round | flat start cost | X100 -> X100 | X100 -> A100 | A100 -> X100 | A100 -> A100 | table from | hand-offs per frame, plan / board | not in the table: sender delay, model switch (us per frame) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in state["rounds"]:
            tr, an = r["steps"].get("tree", {}), r["steps"].get("analyze", {})
            if "handoff_ms" not in tr:
                continue
            h = tr["handoff_ms"]
            ho = "; ".join(f"{n.split('_')[0]} {m.get('handoffs_per_frame_plan', float('nan')):.1f} / {m['handoffs_per_frame']:.1f}" for n, m in an.get("models", {}).items())
            res = "; ".join(f"{n.split('_')[0]} {m.get('sender_delay_us_per_frame', float('nan')):.0f}, {m.get('model_switch_us_per_frame', float('nan')):.1f}"
                            for n, m in an.get("models", {}).items())
            lines.append(f"| {r['tag']} | {tr['flat_start_cost_ms'] * 1e3:.2f} us | {h['cpu_p']['cpu_p'] * 1e3:.2f} | {h['cpu_p']['cpu_e'] * 1e3:.2f} | "
                         f"{h['cpu_e']['cpu_p'] * 1e3:.2f} | {h['cpu_e']['cpu_e'] * 1e3:.2f} us | {tr.get('table_source', '')} | {ho} | {res} |")
        lines += ["", f"Table check per round (a pair with {lay.a.table_min_samples} or more samples whose mean gap is more than {lay.a.table_drift_us:g} us off the value "
                  "charged is re-measured for the next round" + (" - switched off, report only" if lay.a.table_frozen else "") + "):", "",
                  "| round | pair | charged (table + flat) | measured median / mean | samples | verdict |", "|---|---|---|---|---|---|"]
        for r in state["rounds"]:
            tr, an = r["steps"].get("tree", {}), r["steps"].get("analyze", {})
            both = list((an.get("table_check") or {}).items()) + [(f"sender's delay, {n} sends", v) for n, v in (an.get("sender_check") or {}).items()]
            for name, v in both:
                charged = "none" if v["table_us"] is None else f"{v['table_us'] + tr.get('flat_start_cost_ms', 0) * 1e3:.2f} us"
                meas = "-" if v["measured_mean_us"] is None else f"{v['measured_median_us']:.2f} / {v['measured_mean_us']:.2f} us"
                lines.append(f"| {r['tag']} | {name} | {charged} | {meas} | {v['n']} | {v['verdict']} |")
        snd = [(r["tag"], r["steps"]["tree"].get("sender_ms")) for r in state["rounds"] if r["steps"].get("tree", {}).get("sender_ms")]
        if snd:
            lines += ["", "Sender's delay given to the planner (the worker that wakes another thread starts its own next layer this much later): "
                      + "; ".join(f"{tag}: X100 {v.get('cpu_p', 0) * 1e3:.2f} us, A100 {v.get('cpu_e', 0) * 1e3:.2f} us" for tag, v in snd) + "."]
    lines += ["", "Files per round: results in `" + rel(lay.results_root) + "/<round>/`, corrections and plan checks in `" + rel(lay.results_root) +
              "/corr_<round>/`, raw runs in `" + rel(lay.raw_root) + "/<round>_stamps/`, state in `" + rel(os.path.join(lay.out, 'loop_state.json')) + "`."]
    path = os.path.join(lay.out, "loop_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="loop folder: loop_state.json, loop_report.md, logs/, and by default results/, raw/, trees/, solve/ "
                    "(default: k3tl_<workload> in the current folder)")
    ap.add_argument("--config", required=True, help="workload config (networks json): any workload whose models have profiles in the raw tree")
    ap.add_argument("--raw-tree", default=DEFAULT_RAW_TREE, help=f"raw profile tree, never modified (default: {os.path.relpath(DEFAULT_RAW_TREE, REPO)})")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.03)
    ap.add_argument("--abs-tol-ms", type=float, default=0.0, help="also accept a model whose board and planned frame differ by at most this (off by default)")
    ap.add_argument("--gap-rule", choices=("layer", "table"), default="table")
    ap.add_argument("--prefix", default="r", help="round tags: <prefix>0, <prefix>1, ...")
    ap.add_argument("--models", default="", help="comma list of the workload's models (default: the config's networks)")
    ap.add_argument("--modelblaster", default="", help="ModelBlaster checkout with the stamps walker (default: <this repo>/zephyr-chipyard-sw/modelblaster)")
    ap.add_argument("--name", default="", help="prefix of the build folders in ModelBlaster and of our folders on the board (default: the --out folder's name)")
    ap.add_argument("--start-from", nargs="*", default=[], metavar="RAW[,TREE]", help="finished rounds to adopt, in order (round 0 first)")
    ap.add_argument("--workload-name", default="", help="name used in config/schedule file names (default: from --config)")
    ap.add_argument("--results-root", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--tree-root", default="")
    ap.add_argument("--tree-prefix", default="corr_")
    ap.add_argument("--solve-dir", default="")
    ap.add_argument("--profile-audit", default="", help="audit script of the profiling campaign that measured the raw tree; if given, the tree "
                                                        "audit re-runs it on every corrected folder (tree_audit.py)")
    ap.add_argument("--host", default="k3w", help="ssh alias of the board on this machine")
    ap.add_argument("--stall-s", type=int, default=45)
    ap.add_argument("--board-attempts", type=int, default=4)
    ap.add_argument("--board-wait-h", type=float, default=6.0, help="how long to keep asking for a board that is taken")
    ap.add_argument("--owner", default="", help="name on our board lock; give each loop that runs at the same time its own")
    ap.add_argument("--plot", action="store_true", help="let the solver draw its schedule picture (slow for big workloads; the plan is the same)")
    ap.add_argument("--handoff-table", default="", help="gap rule table: a table to use as given (json: flat_start_cost_ms, handoff_ms); "
                                                        "without it the loop measures one from its rounds and keeps it in <out>/board_table.json")
    ap.add_argument("--table-drift-us", type=float, default=2.0)
    ap.add_argument("--table-min-samples", type=int, default=200)
    ap.add_argument("--table-frozen", action="store_true", help="report drift, never update the table")
    ap.add_argument("--no-sender", action="store_true", help="gap rule table: do not give the planner the sender's delay (scheduler.sender_ms)")
    ap.add_argument("--save-table", default="", help="also write the kept table to this file (the board's table, with what it is valid for)")
    ap.add_argument("--table-note", default="K3, PREEMPT_RT kernel, SCHED_FIFO 80", help="written into the kept table: what it is valid for")
    return ap


def fill_defaults(a: argparse.Namespace) -> argparse.Namespace:
    """The options whose default depends on the config: the model list and the loop folder."""
    if not a.models:
        a.models = ",".join(json.load(open(a.config))["networks"])
    if not a.out:
        a.out = "k3tl_" + (a.workload_name or re.sub(r"^networks_", "", os.path.splitext(os.path.basename(a.config))[0]))
    return a


def main() -> int:
    a = fill_defaults(parser().parse_args())
    state = run_loop(a)
    print(f"loop {state['status']}: {state['message']}; report {rel(os.path.join(os.path.abspath(a.out), 'loop_report.md'))}")
    return 0 if state["status"] == "stopped" else 1


if __name__ == "__main__":
    sys.exit(main())
