#!/usr/bin/env python3
"""
Generate a random top-level workload JSON (data/toplevel/networks_*.json format)
from a seed.

Everything the generator emits is backed by data that exists on disk:

  * `hardware` comes verbatim from data/banks/hardware_bank.json.
  * `networks` are drawn from data/banks/model_bank.json, restricted to the
    models that have profile data for the chosen hardware's `profile.target`.
  * Before a model is used, the generator resolves its profile results.csv on
    *every* backend in the hardware config -- mirroring profile_loader's search
    -- so `use_profiled: true` never hits the strict-mode FileNotFoundError.
  * Periods and windows are sized from the model's *measured* runtime on that
    hardware (see `reference_duration_ms`), scaled by a random factor in
    [--min-scale, --max-scale] (default 1.2x - 5x).
  * No `p_core_speedup` is emitted: it only scales the synthetic-timing
    fallback, so applying it on top of profiled CSVs would fabricate timings.

Workload shape:

  * Periodic models form ONE dependency chain sharing a single period T, with
    staggered `start_time` offsets so their windows run back-to-back inside each
    frame.  A shared period is required for correctness: workload_factory
    expands an edge between two periodic networks into instance-i -> instance-i
    edges, so a downstream task with a shorter period would be asked to finish
    before its producer starts.
  * Sporadic models (the yolos) get non-overlapping [min_start_t, max_end_t]
    windows laid into the gaps along the timeline, with no edges touching the
    periodic chain.  An edge from a periodic network to a non-periodic one fans
    out to *every* instance, which would serialise the whole workload.

Usage:
    python scripts/gen_random_workload.py 1234
    python scripts/gen_random_workload.py 1234 --hardware quad_core
    python scripts/gen_random_workload.py 1234 --hardware qnn --stdout
    python scripts/gen_random_workload.py --list

Run with --list to see the banks the current checkout can generate from.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REPO_ROOT_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HARDWARE_BANK = os.path.join("data", "banks", "hardware_bank.json")
MODEL_BANK = os.path.join("data", "banks", "model_bank.json")

# The profile sweeps write 1e9 us (= 1e6 ms) for a dispatch that cannot run on
# that backend at all.  Those rows mark "unsupported", they are not timings, and
# summing them produces nonsense durations.
SENTINEL_MS = 1e6


# --------------------------------------------------------------------------
# Profile lookup -- mirrors xpu-rt/profile_loader.py
# --------------------------------------------------------------------------

def find_profile_csv(repo_root: str, *, hw: str, target: str, model: str,
                     basename: str, topo_tag: str) -> Optional[str]:
    """Mirror of profile_loader.find_profile_csv (both on-disk layouts)."""
    root = os.path.join(repo_root, "gen", "profile")
    for pat in (
        os.path.join(root, hw, target, model, basename, "*", topo_tag, "results.csv"),
        os.path.join(root, hw, target, model, basename, topo_tag, "results.csv"),
    ):
        hits = glob.glob(pat)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def load_dispatch_times_ms(csv_path: str) -> Dict[int, float]:
    """Read a profile results.csv into {dispatch_id: ms}, sentinels included."""
    times: Dict[int, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw_id = (row.get("dispatch_id") or "").strip()
            if not raw_id:
                continue
            try:
                did = int(raw_id)
                t = float(row.get("mean_time", 0.0) or 0.0)
            except ValueError:
                continue
            unit = (row.get("mean_unit") or "ms").strip().lower()
            if unit == "us":
                t /= 1000.0
            elif unit == "s":
                t *= 1000.0
            times[did] = t
    return times


def dispatch_ids(repo_root: str, rel_path: str) -> List[int]:
    """Dispatch ids declared by a *_dispatch_graph.json, in graph order."""
    with open(os.path.join(repo_root, rel_path)) as f:
        data = json.load(f)
    out = []
    for info in (data.get("dispatches") or {}).values():
        did = info.get("id")
        if isinstance(did, int):
            out.append(did)
    return out


# --------------------------------------------------------------------------
# Model timing
# --------------------------------------------------------------------------

@dataclass
class ModelTiming:
    """Measured cost of one network on one hardware config."""
    name: str
    spec: dict
    n_dispatches: int
    per_backend_serial_ms: Dict[str, float]   # inf when a dispatch is unsupported
    best_per_dispatch_ms: float
    worst_per_dispatch_ms: float
    csv_paths: Dict[str, str]
    sentinels: Dict[str, int] = field(default_factory=dict)
    missing_rows: Dict[str, int] = field(default_factory=dict)

    @property
    def reference_ms(self) -> float:
        """
        The duration periods/windows are sized against: the model's runtime on
        the fastest backend that can run *all* of it.

        This is the honest reading of "the model's runtime on the hardware" --
        a network's dispatches form a dependency chain, so its latency is at
        least a serial pass over them.  `best_per_dispatch_ms` (each dispatch on
        whichever backend is fastest for it) is a lower bound the scheduler can
        only approach, and sizing periods against it produces deadlines nothing
        can meet.  When no single backend can run the whole network -- every
        candidate has an unsupported dispatch -- we fall back to that bound,
        since a split across backends is then the only way to run it at all.
        """
        finite = [v for v in self.per_backend_serial_ms.values() if math.isfinite(v)]
        return min(finite) if finite else self.best_per_dispatch_ms

    @property
    def reference_basis(self) -> str:
        finite = {k: v for k, v in self.per_backend_serial_ms.items() if math.isfinite(v)}
        if finite:
            return f"serial on {min(finite, key=finite.get)}"
        return "best-per-dispatch across backends (no backend runs it whole)"


def measure(repo_root: str, name: str, spec: dict, hw_pool: List[str],
            target: str, topo_tag: str) -> Tuple[Optional[ModelTiming], List[str]]:
    """
    Measure `spec` on every backend in `hw_pool`.

    Returns (timing, problems).  `timing` is None when the model is unusable on
    this hardware config, and `problems` says why.
    """
    problems: List[str] = []
    rel = spec["dispatch_deps_path"]
    if not os.path.exists(os.path.join(repo_root, rel)):
        return None, [f"{name}: dispatch graph missing: {rel}"]

    ids = dispatch_ids(repo_root, rel)
    if not ids:
        return None, [f"{name}: dispatch graph declares no dispatches: {rel}"]

    model = spec["profile_model"]
    basename = spec["profile_basename"]

    on_disk = os.path.basename(os.path.dirname(rel))
    if on_disk != basename:
        problems.append(
            f"{name}: profile_basename {basename!r} does not match the "
            f"dispatch path's directory {on_disk!r} -- profile_loader derives "
            f"the basename from the path, so the lookup would use {on_disk!r}"
        )

    per_backend: Dict[str, float] = {}
    csv_paths: Dict[str, str] = {}
    sentinels: Dict[str, int] = {}
    missing_rows: Dict[str, int] = {}
    times: Dict[str, Dict[int, float]] = {}

    for hw in hw_pool:
        path = find_profile_csv(repo_root, hw=hw, target=target, model=model,
                                basename=basename, topo_tag=topo_tag)
        if path is None:
            problems.append(
                f"{name} @ {hw}/{topo_tag}: no results.csv under "
                f"gen/profile/{hw}/{target}/{model}/{basename}/.../{topo_tag}/ "
                f"-- use_profiled would raise in strict mode"
            )
            continue
        csv_paths[hw] = os.path.relpath(path, repo_root)
        t = load_dispatch_times_ms(path)
        times[hw] = t
        # Dispatches with no CSV row are the zero-cost IR ops the codegen drops
        # (view, chunk2_c1, ...).  Strict mode costs them at 0, so we do too.
        missing_rows[hw] = sum(1 for i in ids if i not in t)
        sentinels[hw] = sum(1 for i in ids if t.get(i, 0.0) >= SENTINEL_MS)
        total = 0.0
        for i in ids:
            v = t.get(i, 0.0)
            if v >= SENTINEL_MS:
                total = math.inf
                break
            total += v
        per_backend[hw] = total

    if problems:
        return None, problems

    best = 0.0
    worst = 0.0
    for i in ids:
        vals = [times[hw][i] for hw in hw_pool
                if i in times[hw] and times[hw][i] < SENTINEL_MS]
        best += min(vals) if vals else 0.0
        worst += max(vals) if vals else 0.0

    return ModelTiming(
        name=name, spec=spec, n_dispatches=len(ids),
        per_backend_serial_ms=per_backend,
        best_per_dispatch_ms=best, worst_per_dispatch_ms=worst,
        csv_paths=csv_paths, sentinels=sentinels, missing_rows=missing_rows,
    ), []


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def draw_count(rng: random.Random, spec: dict) -> int:
    bounds = spec.get("count") or {}
    lo = int(bounds.get("min", 1))
    hi = int(bounds.get("max", lo))
    return rng.randint(min(lo, hi), max(lo, hi))


def task_keys(name: str, count: int) -> List[str]:
    """
    Names for `count` draws of one model.

    A single draw keeps the bare model name; multiple draws get letter suffixes
    (dronet_small_a, dronet_small_b).  The key is also used as the network's
    `identifier`, and workload_factory derives each periodic instance's display
    name as identifier + instance index -- so two draws of the same model with a
    shared identifier would both expand to "dronet_small0", "dronet_small1", ...
    and be indistinguishable in the legend and in the emitted schedule JSON.
    Digit suffixes would collide the same way ("dronet_small1" + instance 0 vs
    "dronet_small0" + instance 1), hence letters.
    """
    if count == 1:
        return [name]
    return [f"{name}_{chr(ord('a') + i)}" for i in range(count)]


def generate(args, hw_bank: dict, model_bank: dict) -> dict:
    rng = random.Random(args.seed)
    repo_root = args.repo_root

    configs = hw_bank["configs"]
    hw_name = args.hardware or hw_bank.get("default")
    if hw_name == "random":
        usable = sorted(k for k, v in configs.items() if v.get("available", True))
        if not usable:
            raise SystemExit("no available hardware configs in the bank")
        hw_name = rng.choice(usable)
    if hw_name not in configs:
        raise SystemExit(
            f"--hardware {hw_name!r} is not in the bank; choose from: "
            + ", ".join(sorted(configs))
        )
    hw_cfg = configs[hw_name]

    if not hw_cfg.get("available", True):
        reason = hw_cfg.get("unavailable_reason", ["no reason recorded"])
        raise SystemExit(
            f"hardware config {hw_name!r} is marked unavailable in the bank:\n  "
            + "\n  ".join(reason if isinstance(reason, list) else [reason])
        )

    hardware = {
        "machines": dict(hw_cfg["machines"]),
        "profile_hw": dict(hw_cfg["profile_hw"]),
        "profile": dict(hw_cfg["profile"]),
    }
    # Deduplicated, order-preserving: quad_core maps two slots onto each backend.
    hw_pool = list(dict.fromkeys(hardware["profile_hw"].values()))
    target = hardware["profile"]["target"]
    topo_tag = hardware["profile"].get("topo_tag", "topo_0")
    if not hardware["profile"].get("topo_tag_override", False):
        # With singleton machine kinds every combination is one core, so
        # topo_tag_for_combination() derives topo_0 regardless of the tag above.
        if all(int(n) == 1 for n in hardware["machines"].values()):
            topo_tag = "topo_0"

    platform = (model_bank["platforms"].get(target) or {})
    models = platform.get("models") or {}
    if not models:
        raise SystemExit(
            f"model bank has no models for target {target!r} "
            f"(hardware config {hw_name!r})"
        )

    # --- measure every candidate on this hardware -------------------------
    timings: Dict[str, ModelTiming] = {}
    skipped: Dict[str, List[str]] = {}
    for name, spec in models.items():
        t, problems = measure(repo_root, name, spec, hw_pool, target, topo_tag)
        if t is None:
            skipped[name] = problems
        else:
            timings[name] = t

    for name, problems in skipped.items():
        print(f"warning: dropping {name!r} from the draw:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)

    periodic = [n for n in models if n in timings
                and models[n].get("role") == "periodic"]
    sporadic = [n for n in models if n in timings
                and models[n].get("role") == "sporadic"]
    if not periodic:
        raise SystemExit(
            f"no usable periodic models for {target!r} -- nothing to build a "
            "periodic chain from"
        )

    scale = lambda: rng.uniform(args.min_scale, args.max_scale)

    # --- periodic chain ---------------------------------------------------
    # One shared period T with staggered start_time offsets.  See the module
    # docstring for why the period has to be shared.
    chain: List[dict] = []
    offset = 0
    for name in periodic:
        spec = models[name]
        t = timings[name]
        for key in task_keys(name, draw_count(rng, spec)):
            factor = scale()
            window = max(1, int(math.ceil(t.reference_ms * factor)))
            chain.append({
                "key": key,
                "model": name,
                "window": window,
                "start_time": offset,
                "factor": round(factor, 3),
                "duration_ms": round(t.reference_ms, 3),
            })
            offset += window

    frame = offset  # total occupied time inside one period
    period = max(1, int(math.ceil(frame * rng.uniform(1.0, args.period_headroom))))

    # --- sporadic tasks in the gaps --------------------------------------
    sporadic_tasks: List[dict] = []
    cursor = rng.uniform(0.0, float(period))
    for name in sporadic:
        spec = models[name]
        t = timings[name]
        for key in task_keys(name, draw_count(rng, spec)):
            factor = scale()
            window = math.ceil(t.reference_ms * factor)
            start = round(cursor, 3)
            end = round(start + window, 3)
            sporadic_tasks.append({
                "key": key,
                "model": name,
                "min_start_t": start,
                "max_end_t": end,
                "factor": round(factor, 3),
                "duration_ms": round(t.reference_ms, 3),
            })
            cursor = end + rng.uniform(0.0, args.max_gap * window)

    horizon = max([s["max_end_t"] for s in sporadic_tasks] or [float(period)])

    # --- assemble ---------------------------------------------------------
    networks: Dict[str, dict] = {}
    next_id = 0
    for item in chain:
        spec = models[item["model"]]
        entry = {
            "id": next_id,
            "identifier": item["key"],
            "dispatch_deps_path": spec["dispatch_deps_path"],
            "period": period,
            "window_duration": item["window"],
        }
        if item["start_time"]:
            entry["start_time"] = item["start_time"]
        if args.num_instances != "auto":
            entry["num_instances"] = int(args.num_instances)
        elif args.cap_instances:
            entry["num_instances"] = max(1, min(
                args.cap_instances, int(math.ceil(horizon / period))))
        networks[item["key"]] = entry
        next_id += 1

    for item in sporadic_tasks:
        spec = models[item["model"]]
        networks[item["key"]] = {
            "id": next_id,
            "identifier": item["key"],
            "dispatch_deps_path": spec["dispatch_deps_path"],
            "min_start_t": item["min_start_t"],
            "max_end_t": item["max_end_t"],
        }
        next_id += 1

    edges = [{"from": chain[i]["key"], "to": chain[i + 1]["key"]}
             for i in range(len(chain) - 1)]

    notes = [
        f"Randomly generated by scripts/gen_random_workload.py (seed={args.seed}).",
        f"Hardware: {hw_name} -- {hw_cfg.get('description', '')}",
        f"Periodic chain (shared period {period} ms, sequential edges, "
        f"staggered start_time so windows run back-to-back inside each frame): "
        + " -> ".join(c["key"] for c in chain) + ".",
        f"Sporadic tasks placed in the gaps with non-overlapping windows, no "
        f"edges to the periodic chain: "
        + (", ".join(s["key"] for s in sporadic_tasks) or "(none)") + ".",
        "Windows are the model's measured runtime on this hardware scaled by a "
        f"random factor in [{args.min_scale}, {args.max_scale}]:",
    ]
    for item in chain + sporadic_tasks:
        t = timings[item["model"]]
        notes.append(
            f"  {item['key']}: {item['duration_ms']} ms ({t.reference_basis}, "
            f"{t.n_dispatches} dispatches) x {item['factor']}"
        )
    notes.append(
        "No p_core_speedup: it scales only the synthetic-timing fallback, so "
        "applying it on top of profiled CSVs would fabricate timings."
    )
    for name in sorted({i["model"] for i in chain + sporadic_tasks}):
        note = models[name].get("variant_note")
        if note:
            notes.append(f"  {name}: {note}")
    if skipped:
        notes.append("Dropped from the draw (no usable profile data): "
                     + ", ".join(sorted(skipped)) + ".")

    return {
        "_comment": notes,
        "hardware": hardware,
        "scheduler": {
            "random_seed": args.seed,
            "solver_verbosity": args.solver_verbosity,
            "time_limit": args.time_limit,
            "use_profiled": True,
            "prune_periodic": True,
            "restrict_makespan_to_nonperiodic": True,
        },
        "networks": networks,
        "edges": edges,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(config: dict, repo_root: str) -> List[str]:
    """Re-check the finished document.  Returns a list of problems."""
    problems: List[str] = []
    networks = config["networks"]
    edges = config["edges"]

    for name, info in networks.items():
        rel = info.get("dispatch_deps_path", "")
        if not rel:
            problems.append(f"{name}: empty dispatch_deps_path")
        elif not os.path.exists(os.path.join(repo_root, rel)):
            problems.append(f"{name}: dispatch_deps_path does not exist: {rel}")

    ids = [info["id"] for info in networks.values()]
    if len(set(ids)) != len(ids):
        problems.append("duplicate network ids")

    names = set(networks)
    for e in edges:
        for side in ("from", "to"):
            if e[side] not in names:
                problems.append(f"edge {side} unknown network {e[side]!r}")
        if e["from"] == e["to"]:
            problems.append(f"self-loop on {e['from']!r}")

    # Acyclicity (Kahn).
    indeg = {n: 0 for n in networks}
    succ: Dict[str, List[str]] = {n: [] for n in networks}
    for e in edges:
        if e["from"] in succ and e["to"] in indeg:
            succ[e["from"]].append(e["to"])
            indeg[e["to"]] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if seen != len(networks):
        problems.append("edge set is cyclic")

    is_periodic = {n: "period" in i for n, i in networks.items()}

    # A periodic->non-periodic edge fans out to every instance; a
    # non-periodic->periodic edge does the same in reverse.  Both serialise the
    # workload against the whole periodic expansion.
    for e in edges:
        if e["from"] in networks and e["to"] in networks:
            if is_periodic[e["from"]] != is_periodic[e["to"]]:
                problems.append(
                    f"edge {e['from']} -> {e['to']} crosses periodic/non-periodic; "
                    "workload_factory fans it out to every instance"
                )

    # Periodic chain: one shared period, and each instance-i -> instance-i edge
    # must be satisfiable by the start_time offsets.
    periods = {i["period"] for n, i in networks.items() if is_periodic[n]}
    if len(periods) > 1:
        problems.append(
            f"periodic networks have differing periods {sorted(periods)}; edges "
            "between them expand instance-i -> instance-i, which is infeasible "
            "unless the periods match"
        )
    for e in edges:
        a, b = networks.get(e["from"], {}), networks.get(e["to"], {})
        if "period" not in a or "period" not in b:
            continue
        a_end = a.get("start_time", 0) + a["window_duration"]
        b_start = b.get("start_time", 0)
        if b_start < a_end:
            problems.append(
                f"{e['to']} starts at {b_start} inside {e['from']}'s window "
                f"(ends {a_end}) -- instance-i edge is infeasible"
            )
    for n, i in networks.items():
        if not is_periodic[n]:
            continue
        span = i.get("start_time", 0) + i["window_duration"]
        if span > i["period"]:
            problems.append(
                f"{n}: start_time+window_duration ({span}) exceeds period "
                f"({i['period']}) -- consecutive instances overlap"
            )

    # Sporadic windows must not overlap each other.
    windows = sorted(
        ((i["min_start_t"], i["max_end_t"], n) for n, i in networks.items()
         if not is_periodic[n] and "min_start_t" in i),
    )
    for (s0, e0, n0), (s1, _, n1) in zip(windows, windows[1:]):
        if s1 < e0:
            problems.append(
                f"window overlap: {n1} starts at {s1} before {n0} ends at {e0}")

    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_banks(repo_root: str) -> Tuple[dict, dict]:
    with open(os.path.join(repo_root, HARDWARE_BANK)) as f:
        hw_bank = json.load(f)
    with open(os.path.join(repo_root, MODEL_BANK)) as f:
        model_bank = json.load(f)
    return hw_bank, model_bank


def do_list(repo_root: str, hw_bank: dict, model_bank: dict) -> int:
    for hw_name, hw_cfg in hw_bank["configs"].items():
        target = hw_cfg["profile"]["target"]
        mark = "" if hw_cfg.get("available", True) else "   [UNAVAILABLE]"
        slots = ", ".join(f"{k}={v}" for k, v in hw_cfg["profile_hw"].items())
        print(f"\n{hw_name}{mark}\n  target {target}   {slots}")
        if not hw_cfg.get("available", True):
            for line in hw_cfg.get("unavailable_reason", []):
                print(f"    {line}")
            continue
        hw_pool = list(dict.fromkeys(hw_cfg["profile_hw"].values()))
        topo = hw_cfg["profile"].get("topo_tag", "topo_0")
        if not hw_cfg["profile"].get("topo_tag_override", False):
            if all(int(n) == 1 for n in hw_cfg["machines"].values()):
                topo = "topo_0"
        models = (model_bank["platforms"].get(target) or {}).get("models") or {}
        if not models:
            print(f"    (no models in the bank for {target})")
        for name, spec in models.items():
            t, problems = measure(repo_root, name, spec, hw_pool, target, topo)
            if t is None:
                print(f"    {name:<28} UNUSABLE: {problems[0]}")
                continue
            per = "  ".join(
                f"{hw}={'unsupported' if math.isinf(v) else format(v, '.2f')}"
                for hw, v in t.per_backend_serial_ms.items())
            print(f"    {name:<28} {spec['role']:<9} "
                  f"{t.n_dispatches:>4} dispatches   ref={t.reference_ms:8.2f} ms"
                  f"   [{per}]")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("seed", type=int, nargs="?",
                    help="RNG seed; also written to scheduler.random_seed")
    ap.add_argument("--repo-root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--list", action="store_true",
                    help="print the banks and each model's measured cost, then exit")
    ap.add_argument("--hardware", help="hardware bank entry, or 'random' "
                                       "(default: the bank's `default`)")
    ap.add_argument("-o", "--out",
                    help="output path (default: data/toplevel/networks_random_<hw>_seed<SEED>.json)")
    ap.add_argument("--stdout", action="store_true", help="write to stdout instead")

    ap.add_argument("--min-scale", type=float, default=1.2,
                    help="lower bound on window/period as a multiple of measured runtime")
    ap.add_argument("--max-scale", type=float, default=5.0,
                    help="upper bound on window/period as a multiple of measured runtime")
    ap.add_argument("--period-headroom", type=float, default=1.25,
                    help="period is the chain's occupied time times U(1.0, this)")
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="idle gap after a sporadic task, as a fraction of its window")

    ap.add_argument("--num-instances", default="auto",
                    help="explicit num_instances for every periodic network, or "
                         "'auto' to let workload_factory's horizon heuristic decide")
    ap.add_argument("--cap-instances", type=int, default=None,
                    help="with --num-instances auto, emit an explicit cap computed "
                         "from the laid-out horizon, at most this many")

    ap.add_argument("--solver-verbosity", type=int, default=2)
    ap.add_argument("--time-limit", type=int, default=20)
    args = ap.parse_args(argv)

    args.repo_root = os.path.abspath(args.repo_root)
    hw_bank, model_bank = load_banks(args.repo_root)

    if args.list:
        return do_list(args.repo_root, hw_bank, model_bank)
    if args.seed is None:
        ap.error("seed is required (or pass --list)")
    if not 0 < args.min_scale <= args.max_scale:
        ap.error("--min-scale must be > 0 and <= --max-scale")
    if args.num_instances != "auto":
        try:
            if int(args.num_instances) <= 0:
                raise ValueError
        except ValueError:
            ap.error("--num-instances must be a positive integer or 'auto'")

    config = generate(args, hw_bank, model_bank)

    problems = validate(config, args.repo_root)
    if problems:
        print("generated config failed validation:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    text = json.dumps(config, indent=2) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0

    hw_name = config["hardware"]["profile"]["target"]
    for key, cfg in hw_bank["configs"].items():
        if cfg["profile"]["target"] == hw_name and cfg["machines"] == config["hardware"]["machines"]:
            hw_name = key
            break
    out = args.out or os.path.join(
        args.repo_root, "data", "toplevel",
        f"networks_random_{hw_name}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)

    n_periodic = sum(1 for v in config["networks"].values() if "period" in v)
    period = next((v["period"] for v in config["networks"].values() if "period" in v), 0)
    print(f"wrote {out}")
    print(f"  hardware : {hw_name}  {config['hardware']['profile_hw']}")
    print(f"  networks : {len(config['networks'])} "
          f"({n_periodic} periodic @ period={period} ms, "
          f"{len(config['networks']) - n_periodic} sporadic)")
    print(f"  edges    : {len(config['edges'])} (acyclic, chain over the periodic set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
