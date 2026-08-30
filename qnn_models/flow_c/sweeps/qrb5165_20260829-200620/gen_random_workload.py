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
  * Periods come from the bank's declared `period_ms` per model -- the rate
    the task runs at.  Windows come from the bank's `window_ms` where a model
    declares one (drawn the same way, no scaling), and otherwise from the
    model's *measured* runtime on that hardware scaled by a random factor in
    [--min-scale, --max-scale] (default 1.2x - 5x).  Either way the draw is
    squeezed to fit inside the period.
  * No `p_core_speedup` is emitted: it only scales the synthetic-timing
    fallback, so applying it on top of profiled CSVs would fabricate timings.

Workload shape:

  * Each periodic model runs at its OWN period, drawn from the model bank's
    `period_ms` -- the rate the task actually runs at (a control MLP at
    100+ Hz, a perception net at 5-20 Hz), not something derived from how
    long the network happens to take.  Periods are then snapped to integer
    multiples of the fastest one, so the hyperperiod is the slowest period
    rather than an lcm of unrelated numbers.
  * `window_duration` is the instance's DEADLINE inside its period, drawn
    from the model's `window_ms` band (or, without one, sized as the measured
    runtime times a random factor in [--min-scale, --max-scale]) and squeezed
    if need be so a model's copies fit one frame with --period-headroom to
    spare.  A window that does not fit means consecutive instances of the
    same network overlap.  Declaring the band is what keeps a fast network's
    deadline meaningful: mlp_control runs in 0.393 ms, so any multiple of
    that is a sub-millisecond deadline no other task can be scheduled
    around.
  * The taskset is then checked against the hardware: each periodic task is
    packed onto the backend that stays least loaded, and any config asking
    for more than --max-utilization of a core has the worst offender's
    period doubled until it fits.  Heterogeneity is the point -- dronet is
    28 ms on RVV and 454 ms on scalar, so two cores is not two dronets.
  * How many copies of each model are drawn comes from the bank's `count`
    range, gated by an optional `probability` -- so a model can be common in
    the mix or occasional in it without changing what it looks like when it
    does show up.  Copies of one model share its period, run back-to-back
    inside the frame, and are chained by edges.  A model that cannot
    meaningfully run alongside itself declares `max_copies` and the draw is
    clamped there: mlp_control caps at 1, because two chained control MLPs
    per frame is two control loops on one plant, not a busier drone.  Models at *different* rates
    get no edges between them: workload_factory pairs periodic instances by
    index, which only says something when both sides tick together.
  * Sporadic models (the yolos) get non-overlapping [min_start_t, max_end_t]
    windows laid into the gaps along the timeline, with no edges touching the
    periodic taskset.  An edge from a periodic network to a non-periodic one
    fans out to *every* instance, which would serialise the whole workload.
  * The document carries an explicit `horizon_ms` and a `num_instances` per
    periodic network (ceil(horizon / period), inside --max-ops).  Left to
    itself the toolchain sizes periodic instances from the *non-periodic*
    makespan, which is zero for a workload of nothing but periodic tasks --
    that is what collapsed every all-periodic schedule to a single instance
    of each network.

Usage:
    python scripts/gen_random_workload.py 1234
    python scripts/gen_random_workload.py 1234 --hardware spike_quad_core
    python scripts/gen_random_workload.py 1234 --horizon-periods 6 --stdout
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

# PORT (qrb5165 sweep): the banks live beside this copy of the generator,
# not under <repo>/data/banks -- XPU-RT has no data/banks tree, and the
# qrb5165_flowc entries are specific to this sweep. Overridable with
# --banks-dir; everything else in this file is the RoSE original.
BANKS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banks")
HARDWARE_BANK = "hardware_bank.json"
MODEL_BANK = "model_bank.json"

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
    unrunnable: List[int] = []
    for i in ids:
        vals = [times[hw][i] for hw in hw_pool
                if i in times[hw] and times[hw][i] < SENTINEL_MS]
        if not vals:
            # Empty for two very different reasons:
            #   - no backend has a row for this dispatch at all -- it is one
            #     of the zero-cost IR ops the codegen drops (view, chunk2_c1),
            #     which profile_loader's strict mode also costs at 0; or
            #   - every backend that does have a row marks it unsupported --
            #     nothing in this hardware config can run the dispatch.
            # Both used to cost 0, which sized windows against work no
            # backend can do: smolvlm_vision_v3_bundles measured 417 ms here
            # while the scheduler, charging profile_loader's unsupported
            # penalty for the same 46 dispatches, put it at 46,000,417 ms.
            if any(i in times[hw] for hw in hw_pool):
                unrunnable.append(i)
            continue
        best += min(vals)
        worst += max(vals)

    if unrunnable:
        shown = ", ".join(str(i) for i in unrunnable[:8])
        more = f" (+{len(unrunnable) - 8} more)" if len(unrunnable) > 8 else ""
        return None, [
            f"{name}: {len(unrunnable)} of {len(ids)} dispatches are "
            f"unsupported on every backend in this hardware config "
            f"({'/'.join(hw_pool)}) -- dispatch_ids {shown}{more}. "
            f"No schedule can run this network here."
        ]

    return ModelTiming(
        name=name, spec=spec, n_dispatches=len(ids),
        per_backend_serial_ms=per_backend,
        best_per_dispatch_ms=best, worst_per_dispatch_ms=worst,
        csv_paths=csv_paths, sentinels=sentinels, missing_rows=missing_rows,
    ), []


# --------------------------------------------------------------------------
# Draws
# --------------------------------------------------------------------------

def draw_count(rng: random.Random, spec: dict) -> int:
    """
    How many copies of one model this generation gets.

    An optional `probability` gates the model first: below it the count draw
    is skipped and the model sits this one out.  That is a different shape
    from `count.min: 0`, which can only thin a model to 1-in-(max+1) and
    still gives it a draw every time -- the gate is for a model that should
    appear rarely but at full strength when it does.  No bank entry uses it
    now: yolov8_nano is `count: {0, 5}` instead, so how *loaded* a workload
    is comes out of the same draw as whether it has detections at all.  The
    probability draw is taken unconditionally so the RNG stream stays
    aligned across models.
    """
    p = spec.get("probability")
    drawn = rng.random() < float(p) if p is not None else True
    bounds = spec.get("count") or {}
    lo = int(bounds.get("min", 1))
    hi = int(bounds.get("max", lo))
    count = rng.randint(min(lo, hi), max(lo, hi))
    return count if drawn else 0


def copy_cap(spec: dict) -> Optional[int]:
    """
    The bank's ceiling on how many copies of one model may be live at once.

    `count` says how many copies a generation *draws*; `max_copies` says how
    many the workload is allowed to hold.  They are separate because the
    copies of a model are concurrent by construction -- they share a period
    and are chained back-to-back inside the frame -- so two copies of a
    control net is two control loops running against the same plant, which
    is not a taskset the drone stack ever has.  mlp_control declares 1 for
    exactly that reason.  Absent, the model is uncapped and `count` alone
    decides.
    """
    cap = spec.get("max_copies")
    return None if cap is None else max(0, int(cap))


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


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------

# Fallback period band for a periodic model the bank gives no `period_ms`:
# a multiple of the model's own measured runtime.  Feasible, but arbitrary
# -- declare `period_ms` instead, it is the number that carries meaning.
DEFAULT_PERIOD_SPAN = (2.0, 8.0)


def draw_period_ms(rng: random.Random, spec: dict, reference_ms: float) -> float:
    """The model's period in ms, before harmonic snapping."""
    bounds = spec.get("period_ms") or {}
    lo, hi = bounds.get("min"), bounds.get("max")
    if lo is None or hi is None:
        lo = reference_ms * DEFAULT_PERIOD_SPAN[0]
        hi = reference_ms * DEFAULT_PERIOD_SPAN[1]
    lo, hi = float(lo), float(hi)
    return rng.uniform(min(lo, hi), max(lo, hi))


def draw_window_factor(rng: random.Random, spec: dict, reference_ms: float,
                       min_scale: float, max_scale: float) -> float:
    """The instance's deadline, as a multiple of the model's measured runtime.

    A model that declares `window_ms` has its deadline drawn straight from
    that band, the same way its period is -- the scale factors do not apply
    to it.  Scaling is the wrong instrument for a task whose runtime is tiny
    against its rate: 5x of mlp_control's 0.393 ms is a 2 ms deadline, which
    is faithful to the measurement and useless as a taskset, because nothing
    else can be scheduled against a deadline that tight (and, on a 1 s
    timeline, it draws as a sub-pixel sliver).  A deadline is a property of
    the loop the task sits in, so it is declared, like the period.

    Returned as a factor rather than milliseconds because `fit_windows` works
    in factors: it squeezes a deadline toward 1.0 (the measured runtime, the
    tightest deadline that can still be met) when a model's copies do not fit
    one frame.  That floor is why the draw is clamped up to 1.0 here -- a
    band whose low end is under the model's own runtime asks for a deadline
    the network cannot meet on any backend.
    """
    bounds = spec.get("window_ms") or {}
    lo, hi = bounds.get("min"), bounds.get("max")
    if lo is None or hi is None:
        return rng.uniform(min_scale, max_scale)
    lo, hi = float(lo), float(hi)
    ms = rng.uniform(min(lo, hi), max(lo, hi))
    if reference_ms <= 0:
        return 1.0
    return max(1.0, ms / reference_ms)


@dataclass
class RateGroup:
    """One periodic model's copies, all running at that model's own period.

    A group is the unit that shares a period, so it is also the unit that
    can carry dependency edges: workload_factory expands an edge between two
    periodic networks into instance-i -> instance-i edges, which only says
    something when both sides tick at the same rate.  Copies within a group
    are chained; nothing crosses between groups.
    """
    model: str
    period: int
    drawn: List[float]        # deadline slack as drawn, before any fitting
    factors: List[float]      # ...and after fitting it into `period`
    keys: List[str] = field(default_factory=list)
    windows: List[int] = field(default_factory=list)
    starts: List[int] = field(default_factory=list)
    stretched: bool = False

    def refit(self, ref_ms: float, period: int, headroom: float) -> None:
        """Re-derive the windows for a new period, from the drawn slack.

        Refitting the already-squeezed factors would keep a deadline that
        was tightened to fit a period the task no longer has -- a task
        whose period doubled for utilization would carry the tight
        deadline of the rate it was moved off.
        """
        self.period, self.factors, dropped = fit_windows(
            ref_ms, self.drawn[:len(self.keys)], period, headroom)
        if dropped:
            self.keys = self.keys[:len(self.factors)]
            self.drawn = self.drawn[:len(self.factors)]


def fit_windows(ref_ms: float, factors: List[float], period: int,
                headroom: float) -> Tuple[int, List[float], int]:
    """
    Size one model's windows so all its copies fit inside one period.

    Returns (period, factors, dropped) -- `dropped` copies were cut from the
    end of `factors`, and the period only moves as a last resort.

    `window_duration` is a deadline, not a duration: the instance has to
    finish inside [start_time + i*T, start_time + i*T + window].  Consecutive
    instances therefore overlap unless a model's copies fit in one frame,
    which is what the old generator had backwards -- it summed the windows
    and called the sum a period, so the period grew with the deadline slack
    instead of the slack being bounded by the rate.

    Three moves, in the order that costs the least meaning:
      1. squeeze the deadlines toward the measured runtime (a tight deadline
         is still a real workload, and never goes below 1.0x);
      2. drop a copy (two of a model at this rate is more than the frame
         holds, but the rate itself is the declared, meaningful number);
      3. only for a single copy that still does not fit, double the period,
         which is the honest statement that the model cannot run this often.
    """
    factors = list(factors)
    dropped = 0

    def windows(fs: List[float]) -> int:
        return sum(max(1, int(math.ceil(ref_ms * f))) for f in fs)

    for _ in range(32):
        budget = period / max(1.0, headroom)
        if windows(factors) <= budget:
            return period, factors, dropped
        wanted = sum(ref_ms * f for f in factors)
        # Aim below the budget by one ms per window: each one is rounded up
        # to whole ms, and without that slack the squeeze lands just over
        # the line and the period gets stretched for a rounding error.
        target = budget - len(factors)
        if wanted > 0 and target > 0:
            squeezed = [max(1.0, f * target / wanted) for f in factors]
            if windows(squeezed) <= budget:
                return period, squeezed, dropped
        floored = [max(1.0, f) for f in factors]
        if windows(floored) <= budget:
            return period, floored, dropped
        if len(factors) > 1:
            factors = floored[:-1]
            dropped += 1
            continue
        factors = floored
        period *= 2
    return period, factors, dropped


def harmonize(groups: List["RateGroup"], timings: Dict[str, ModelTiming],
              headroom: float) -> None:
    """
    Snap the periods back into a divisibility ladder, in place.

    Walking fastest-first, each period is raised to the next multiple of
    the one below it.  Periods only ever grow, so windows that fitted
    still fit; they are refit anyway, from the drawn slack, so a task
    whose period grew gets the deadline its new rate allows.
    """
    prev = 0
    for g in sorted(groups, key=lambda g: g.period):
        if prev and g.period % prev:
            g.refit(timings[g.model].reference_ms,
                    prev * int(math.ceil(g.period / prev)), headroom)
        prev = g.period


def backend_capacity(hardware: dict) -> Dict[str, int]:
    """{backend: how many machine slots map to it}, e.g. {RVV: 2, scalar: 2}."""
    capacity: Dict[str, int] = {}
    for kind, hw in hardware["profile_hw"].items():
        capacity[hw] = capacity.get(hw, 0) + int(hardware["machines"].get(kind, 0))
    return {hw: n for hw, n in capacity.items() if n > 0}


def periodic_utilization(groups: List["RateGroup"], timings: Dict[str, ModelTiming],
                         capacity: Dict[str, int]) -> Tuple[float, Dict[str, float]]:
    """
    Peak backend utilization of the periodic set alone.

    Each periodic task is placed on the backend that ends up least loaded
    relative to that backend's core count (longest-first, which is the usual
    partitioned-EDF heuristic), charged at *that backend's* measured runtime
    over its period.  The peak is max over backends of load / cores.

    Above 1.0 the periodic set asks for more of some backend than the
    hardware has, so instances miss their windows in every schedule no
    matter which solver runs -- which is what a period drawn without regard
    to the hardware produces (two dronet copies at 50 ms is 1.14 RVV cores,
    and spike_single_core has one).  Heterogeneity matters here: dronet is
    28 ms on RVV and 454 ms on scalar, so "there are two cores" is not the
    same as "there is capacity for two dronets".
    """
    load = {hw: 0.0 for hw in capacity}
    if not capacity:
        return 0.0, load

    items: List[Tuple[float, ModelTiming, int]] = []
    for g in groups:
        t = timings[g.model]
        for _ in g.keys:
            items.append((t.reference_ms / g.period, t, g.period))
    items.sort(key=lambda it: -it[0])

    for _, t, period in items:
        best_hw: Optional[str] = None
        best_after = math.inf
        for hw, cores in capacity.items():
            ms = t.per_backend_serial_ms.get(hw, math.inf)
            if not math.isfinite(ms):
                continue
            after = (load[hw] + ms / period) / cores
            if after < best_after:
                best_hw, best_after = hw, after
        if best_hw is None:
            # No backend runs this network whole; it has to be split across
            # them, and reference_ms is already the best-per-dispatch bound.
            # Charge it to the least loaded backend so it still counts.
            best_hw = min(capacity, key=lambda hw: load[hw] / capacity[hw])
            load[best_hw] += t.reference_ms / period
        else:
            load[best_hw] += t.per_backend_serial_ms[best_hw] / period

    peak = max(load[hw] / capacity[hw] for hw in capacity)
    return peak, load


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

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

    # --- optional model-mix control ---------------------------------------
    # Without these the draw uses every usable model in the bank, so adding a
    # model (fused_full, vint) silently changes every existing sweep point.
    # --include-models pins the draw to an explicit set; --exclude-models
    # removes one. Together they let a sweep run matched arms -- e.g. dronet
    # vs fused_full in the same periodic slot -- instead of hoping the RNG
    # draws a comparable mix.
    if getattr(args, "include_models", None):
        want = [m.strip() for m in args.include_models.split(",") if m.strip()]
        missing = [m for m in want if m not in models]
        if missing:
            raise SystemExit(
                f"--include-models: {missing} not in the bank for {target!r} "
                f"(have: {sorted(models)})")
        models = {k: v for k, v in models.items() if k in want}
    if getattr(args, "exclude_models", None):
        drop = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
        models = {k: v for k, v in models.items() if k not in drop}
    if not models:
        raise SystemExit("model filters removed every model from the draw")

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
            "periodic taskset from"
        )

    scale = lambda: rng.uniform(args.min_scale, args.max_scale)
    notes_rate: List[str] = []
    notes_cap: List[str] = []

    # --- periodic taskset: one period per model ---------------------------
    # Every draw is taken before anything is sized, so the RNG stream does
    # not depend on how the fitting below turns out.
    draws: List[Tuple[str, int, float, List[float]]] = []
    for name in periodic:
        spec = models[name]
        count = draw_count(rng, spec)
        raw_period = draw_period_ms(rng, spec, timings[name].reference_ms)
        factors = [draw_window_factor(rng, spec, timings[name].reference_ms,
                                      args.min_scale, args.max_scale)
                   for _ in range(count)]
        # `max_copies` caps the draw *after* it is taken, not by narrowing the
        # bank's `count` band, so the RNG stream is untouched: every other
        # model still sees the same periods, window factors and sporadic
        # layout it saw before the cap existed, and only the capped model's
        # extra copies go away.  Narrowing the band instead would change how
        # many values randint consumes and redraw the whole workload.
        cap = copy_cap(spec)
        if cap is not None and count > cap:
            notes_cap.append(
                f"  {name}: {count} copies -> {cap} -- copies of a model are "
                f"concurrent (shared period, back-to-back in the frame), and "
                f"the bank allows {cap} of this one at a time")
            count = cap
            factors = factors[:cap]
        if count:
            draws.append((name, count, raw_period, factors))

    if not draws:
        raise SystemExit(
            "every periodic model was drawn zero times -- nothing to build a "
            "periodic taskset from.  Raise count.min or probability for at "
            f"least one of: {', '.join(periodic)}"
        )

    # Harmonic snap.  Sorted fastest-first, every period becomes an integer
    # multiple of the one below it -- so each divides the next, and the
    # hyperperiod is the slowest period rather than an lcm of unrelated
    # numbers.  (Multiples of a shared base is not enough: 15 ms and 20 ms
    # are both multiples of 5 and still only repeat together every 60 ms.)
    # The hyperperiod is what the horizon and every instance count are
    # measured in, so letting it run to an lcm is letting the workload size
    # itself by accident.
    draws.sort(key=lambda d: d[2])
    base = max(1, int(round(draws[0][2])))
    groups: List[RateGroup] = []
    prev = base
    for name, count, raw_period, factors in draws:
        period = prev * max(1, int(round(raw_period / prev)))
        prev = period
        ref = timings[name].reference_ms
        fitted, fitted_factors, dropped = fit_windows(
            ref, factors, period, args.period_headroom)
        g = RateGroup(model=name, period=fitted, drawn=factors,
                      factors=fitted_factors,
                      keys=task_keys(name, count - dropped),
                      stretched=fitted != period)
        if dropped:
            notes_rate.append(
                f"  {name}: {count} copies -> {count - dropped} -- "
                f"{ref:.3f} ms each does not fit {count} times into a "
                f"{period} ms frame with {args.period_headroom}x headroom")
        if g.stretched:
            notes_rate.append(
                f"  {name}: period stretched {period} -> {fitted} ms -- one "
                f"copy at {ref:.3f} ms does not fit a {period} ms frame with "
                f"{args.period_headroom}x headroom")
        prev = g.period          # a stretch inside fit_windows counts too
        groups.append(g)

    # Utilization relief.  A period drawn from the bank knows the task's
    # rate but not the hardware it landed on, so the taskset can come out
    # over capacity (dronet twice at 50 ms is 1.14 RVV cores; spike's
    # single-core config has one).  Halve the worst offender's rate until
    # it fits -- doubling a period keeps the harmonic structure, so the
    # hyperperiod does not blow up on the way.
    capacity = backend_capacity(hardware)
    for _ in range(12):
        peak, load = periodic_utilization(groups, timings, capacity)
        if peak <= args.max_utilization:
            break
        victim = max(groups, key=lambda g: len(g.keys) * timings[g.model].reference_ms / g.period)
        was = victim.period
        victim.refit(timings[victim.model].reference_ms, victim.period * 2,
                     args.period_headroom)
        notes_rate.append(
            f"  {victim.model}: period {was} -> {victim.period} ms -- the "
            f"taskset needed {peak:.2f} of a core against a "
            f"{args.max_utilization} budget")
    # Doubling a period keeps it a multiple of everything below it but can
    # push it past something above it (10 and 30 ms -> 20 and 30), so
    # re-snap the ladder before anything is measured against it.
    harmonize(groups, timings, args.period_headroom)
    peak, load = periodic_utilization(groups, timings, capacity)

    # Windows and phases.  Copies of a model run back-to-back inside the
    # frame, so a chain edge between them is satisfiable instance by
    # instance; fit_windows already guaranteed the group fits.
    for g in groups:
        ref = timings[g.model].reference_ms
        g.windows = [max(1, int(math.ceil(ref * f))) for f in g.factors]
        offset = 0
        g.starts = []
        for w in g.windows:
            g.starts.append(offset)
            offset += w

    # Harmonized, so this is the lcm; computed as one anyway, because a
    # hyperperiod that is quietly wrong silently mis-sizes every count.
    hyperperiod = math.lcm(*(g.period for g in groups))
    horizon = float(hyperperiod * args.horizon_periods)

    # --- sporadic tasks in the gaps --------------------------------------
    sporadic_tasks: List[dict] = []
    cursor = rng.uniform(0.0, float(hyperperiod))
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

    # The periodic tasks have to keep ticking for as long as there is any
    # work on the machine, so the horizon covers the sporadic span too.
    span = max([s["max_end_t"] for s in sporadic_tasks] or [0.0])
    if not args.unbounded_nonperiodic:
        horizon = max(horizon, span)
    else:
        # Unbounded non-periodic tasks have no release window, so there is no
        # sporadic *span* to cover -- but they still take time, and the
        # makespan is whenever the last of them finishes. If the horizon were
        # left at the hyperperiod, the periodic groups would stop ticking
        # early and the schedule tail would again contain only non-periodic
        # work (exactly the "control loop dies partway" failure the windowed
        # layout produces). So size the horizon to cover an ESTIMATE of when
        # the non-periodic work completes, and let the --max-ops loop below
        # bound it. n_cores divides because non-periodic jobs run in parallel
        # across cores; reference_ms is the single-backend serial estimate, so
        # this is deliberately conservative (over-covering is safe -- it only
        # means the periodic groups tick a little longer than strictly needed).
        span = 0.0
        if args.horizon_covers_nonperiodic:
            n_cores = max(1, sum(int(v) for v in hw_cfg["machines"].values()))
            est = sum(timings[s["model"]].reference_ms for s in sporadic_tasks)
            horizon = max(horizon, est / n_cores)

    # ...but bounded by an operation budget, because horizon/period is an
    # operation count and the schedulers are superlinear in it.  yolov8_nano
    # alone stretches the span past 3 s, which at an 8 ms control period is
    # 375 mlp_control instances and ~2600 operations.
    sporadic_ops = sum(timings[s["model"]].n_dispatches for s in sporadic_tasks)

    # --- instance counts --------------------------------------------------
    # Emitted explicitly rather than left to workload_factory's heuristic,
    # which sizes periodic instances from the *non-periodic* makespan and so
    # returns exactly one instance for a workload that has no sporadic task
    # in it at all.
    def instances_at(span_ms: float, period: int) -> int:
        if args.num_instances != "auto":
            return int(args.num_instances)
        n = max(1, int(math.ceil(span_ms / period)))
        if args.cap_instances:
            n = min(n, args.cap_instances)
        return n

    def ops_at(span_ms: float) -> int:
        return sporadic_ops + sum(
            timings[g.model].n_dispatches * len(g.keys) * instances_at(span_ms, g.period)
            for g in groups)

    # Shrink against the count the workload actually ends up with, not
    # against horizon/period: every network rounds its own count up, so the
    # closed-form bound lands over budget by up to one instance each.
    wanted_horizon = horizon
    while horizon > hyperperiod and ops_at(horizon) > args.max_ops:
        over = ops_at(horizon) / float(args.max_ops)
        horizon = max(float(hyperperiod), horizon / max(over, 1.02))
    capped = horizon < wanted_horizon

    # The shrink only has periodic instances to give back, so a draw whose
    # sporadic tasks alone exceed the budget lands over it however far the
    # horizon falls -- five yolov8_nano is ~1060 operations before a single
    # periodic instance.  Say so rather than emitting a quietly oversized
    # workload: the schedulers are superlinear in the operation count.
    if capped and horizon < span:
        print(f"warning: the periodic tasks cover {horizon:.0f} ms of a "
              f"{span:.0f} ms sporadic span -- {len(sporadic_tasks)} sporadic "
              f"task(s) are {sporadic_ops} operations on their own against a "
              f"--max-ops {args.max_ops} budget, and the horizon shrank to "
              f"stay inside it. The control loop stops ticking a long way "
              f"before the last detection finishes. Raise --max-ops (a "
              f"horizon covering the whole span needs roughly "
              f"{sporadic_ops + sum(timings[g.model].n_dispatches * len(g.keys) * int(math.ceil(span / g.period)) for g in groups)}"
              f"), or lower the sporadic count band in the model bank.",
              file=sys.stderr)

    instances_for = lambda period: instances_at(horizon, period)

    # An explicit --num-instances ignores the horizon, so there the horizon
    # has to catch up to it: `horizon_ms` is what the periodic trim cuts
    # against downstream, and a workload asking for 100 instances of a
    # 16 ms task inside a 336 ms horizon would have three quarters of them
    # thrown away after they were scheduled.
    #
    # Only for an explicit count.  Doing it in the `auto` case would be
    # circular -- the count is ceil(horizon/period), so count*period is
    # always >= horizon, and raising the horizon to it raises the count
    # again, which is how a workload capped at --max-ops 1200 came back
    # out at 1280 operations.  The `auto` count already covers the horizon
    # by construction: the last instance opens at (n-1)*period < horizon.
    if args.num_instances != "auto":
        horizon = max(horizon, max(instances_for(g.period) * g.period
                                   for g in groups))

    # --- assemble ---------------------------------------------------------
    networks: Dict[str, dict] = {}
    next_id = 0
    for g in groups:
        spec = models[g.model]
        for key, window, start in zip(g.keys, g.windows, g.starts):
            entry = {
                "id": next_id,
                "identifier": key,
                "dispatch_deps_path": spec["dispatch_deps_path"],
                "period": g.period,
                "window_duration": window,
                "num_instances": instances_for(g.period),
            }
            if start:
                entry["start_time"] = start
            networks[key] = entry
            next_id += 1

    for item in sporadic_tasks:
        spec = models[item["model"]]
        net_entry = {
            "id": next_id,
            "identifier": item["key"],
            "dispatch_deps_path": spec["dispatch_deps_path"],
        }
        if not args.unbounded_nonperiodic:
            net_entry["min_start_t"] = item["min_start_t"]
            net_entry["max_end_t"] = item["max_end_t"]
        networks[item["key"]] = net_entry
        next_id += 1

    # Chains run inside a rate group only.  Across groups the periods
    # differ, and workload_factory pairs periodic instances by index, so a
    # cross-rate edge would tie the 8 ms task's instance i to the 96 ms
    # task's instance i -- a dependency between events 88 ms apart that
    # claims to be the same tick.
    edges = [{"from": g.keys[i], "to": g.keys[i + 1]}
             for g in groups for i in range(len(g.keys) - 1)]

    total_ops = ops_at(horizon)

    notes = [
        f"Randomly generated by scripts/gen_random_workload.py (seed={args.seed}).",
        f"Hardware: {hw_name} -- {hw_cfg.get('description', '')}",
        "Periodic taskset -- each model runs at its OWN period, drawn from the "
        "model bank's period_ms and snapped to a multiple of the fastest "
        f"task's ({base} ms) so the hyperperiod is just the slowest period "
        f"({hyperperiod} ms):",
    ]
    for g in groups:
        t = timings[g.model]
        util = len(g.keys) * t.reference_ms / g.period
        notes.append(
            f"  {g.model}: period {g.period} ms ({1000.0 / g.period:.1f} Hz), "
            f"{len(g.keys)} copy(ies), {t.reference_ms:.3f} ms each "
            f"({t.reference_basis}, {t.n_dispatches} dispatches), "
            f"{util * 100:.1f}% of a core, "
            f"{instances_for(g.period)} instances over the horizon")
        wb = models[g.model].get("window_ms")
        basis = (f"drawn from window_ms {wb['min']}-{wb['max']} ms, then fitted "
                 f"to the period" if wb
                 else f"measured runtime x U({args.min_scale}, {args.max_scale})")
        for key, window, start, factor in zip(g.keys, g.windows, g.starts, g.factors):
            notes.append(
                f"    {key}: window {window} ms ({basis}; deadline = "
                f"{factor:.2f}x measured runtime), start_time {start} ms")
    notes.append(
        f"Peak backend utilization of the periodic set: {peak:.2f} "
        f"(budget {args.max_utilization}) -- "
        + ", ".join(f"{hw} {load[hw]:.2f}/{capacity[hw]}" for hw in sorted(load))
        + ".")
    if notes_cap:
        notes.append("Copies capped by the model bank's max_copies:")
        notes.extend(notes_cap)
    if notes_rate:
        notes.append("Rates adjusted to fit this hardware:")
        notes.extend(notes_rate)
    horizon_note = f"Horizon {horizon:.0f} ms = {args.horizon_periods} x hyperperiod"
    if span > hyperperiod * args.horizon_periods:
        horizon_note += f", widened to cover the sporadic span ({span:.0f} ms)"
    if capped:
        horizon_note += (f", then shortened from {wanted_horizon:.0f} ms to stay "
                         f"inside --max-ops {args.max_ops}")
    notes.append(horizon_note + "; num_instances is ceil(horizon / period) per "
                 f"network, {total_ops} operations in total.")
    notes.append(
        "Sporadic tasks placed in the gaps with non-overlapping windows, no "
        "edges to the periodic taskset (an edge across the boundary fans out "
        "to every periodic instance): "
        + (", ".join(f"{s['key']} [{s['min_start_t']:.0f}, {s['max_end_t']:.0f}] ms"
                     for s in sporadic_tasks) or "(none)") + ".")
    notes.append(
        "Edges chain the copies of one model, which share its period; models "
        "at different rates get none, because workload_factory pairs periodic "
        "instances by index and that only means something at equal periods.")
    notes.append(
        "No p_core_speedup: it scales only the synthetic-timing fallback, so "
        "applying it on top of profiled CSVs would fabricate timings.")
    for name in sorted({g.model for g in groups} | {s["model"] for s in sporadic_tasks}):
        note = models[name].get("variant_note")
        if note:
            notes.append(f"  {name}: {note}")
    if skipped:
        notes.append("Dropped from the draw (no usable profile data): "
                     + ", ".join(sorted(skipped)) + ".")

    return {
        "_comment": notes,
        "hardware": hardware,
        # How long the workload is meant to run.  run_xpurt_schedule and
        # workload_factory both read it: it is the floor on how far periodic
        # instances are grown and on where the periodic trim cuts, so a
        # taskset with no sporadic work in it still ticks for the full span
        # instead of collapsing to one instance of each network.
        "horizon_ms": round(horizon, 3),
        "hyperperiod_ms": hyperperiod,
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

    # Edges between periodic networks expand instance-i -> instance-i, so
    # the two sides have to tick together: same period, same instance
    # count, and the consumer's window must open after the producer's
    # closes within the frame.
    for e in edges:
        a, b = networks.get(e["from"], {}), networks.get(e["to"], {})
        if "period" not in a or "period" not in b:
            continue
        if a["period"] != b["period"]:
            problems.append(
                f"edge {e['from']} -> {e['to']} joins periods "
                f"{a['period']} and {b['period']}; instance-i -> instance-i "
                "only means something at equal periods"
            )
            continue
        if a.get("num_instances") != b.get("num_instances"):
            problems.append(
                f"edge {e['from']} -> {e['to']} joins {a.get('num_instances')} "
                f"and {b.get('num_instances')} instances; the pairing would "
                "drop the tail"
            )
        a_end = a.get("start_time", 0) + a["window_duration"]
        b_start = b.get("start_time", 0)
        if b_start < a_end:
            problems.append(
                f"{e['to']} starts at {b_start} inside {e['from']}'s window "
                f"(ends {a_end}) -- instance-i edge is infeasible"
            )

    # Each periodic network's window has to close before its next instance
    # opens, and every network has to say how many instances it runs.
    for n, i in networks.items():
        if not is_periodic[n]:
            continue
        span = i.get("start_time", 0) + i["window_duration"]
        if span > i["period"]:
            problems.append(
                f"{n}: start_time+window_duration ({span}) exceeds period "
                f"({i['period']}) -- consecutive instances overlap"
            )
        count = i.get("num_instances")
        if not isinstance(count, int) or count < 1:
            problems.append(f"{n}: num_instances must be a positive int, got {count!r}")

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

def load_banks(banks_dir: str) -> Tuple[dict, dict]:
    with open(os.path.join(banks_dir, HARDWARE_BANK)) as f:
        hw_bank = json.load(f)
    with open(os.path.join(banks_dir, MODEL_BANK)) as f:
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
    ap.add_argument("--banks-dir", default=BANKS_DIR_DEFAULT,
                    help="directory holding hardware_bank.json + model_bank.json")
    ap.add_argument("--list", action="store_true",
                    help="print the banks and each model's measured cost, then exit")
    ap.add_argument("--hardware", help="hardware bank entry, or 'random' "
                                       "(default: the bank's `default`)")
    ap.add_argument("-o", "--out",
                    help="output path (default: data/toplevel/generated-data/"
                         "networks_random_<hw>_seed<SEED>.json)")
    ap.add_argument("--stdout", action="store_true", help="write to stdout instead")

    ap.add_argument("--min-scale", type=float, default=1.2,
                    help="lower bound on window/period as a multiple of measured runtime")
    ap.add_argument("--max-scale", type=float, default=5.0,
                    help="upper bound on window/period as a multiple of measured runtime")
    ap.add_argument("--period-headroom", type=float, default=1.25,
                    help="a model's copies must fit in period/this, leaving the "
                         "rest of the frame idle (default: 1.25)")
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="idle gap after a sporadic task, as a fraction of its window")
    ap.add_argument("--max-utilization", type=float, default=0.75,
                    help="periodic load budget per backend core; a taskset over "
                         "it has the offending model's period doubled until it "
                         "fits (default: 0.75)")

    ap.add_argument("--horizon-periods", type=float, default=3.0,
                    help="how many hyperperiods the workload runs for; sets "
                         "num_instances = ceil(horizon/period) (default: 3)")
    ap.add_argument("--max-ops", type=int, default=1200,
                    help="operation budget for the whole workload; the horizon "
                         "is shortened to stay inside it (default: 1200)")
    ap.add_argument("--num-instances", default="auto",
                    help="explicit num_instances for every periodic network, or "
                         "'auto' to derive it from the horizon")
    ap.add_argument("--include-models", default=None, metavar="A,B,C",
                    help="restrict the draw to these bank models (comma list). "
                         "Use to run matched arms, e.g. --include-models "
                         "mlp_control,fused_full,yolov8_nano against the same "
                         "list with dronet in place of fused_full.")
    ap.add_argument("--exclude-models", default=None, metavar="A,B",
                    help="drop these bank models from the draw (comma list).")
    ap.add_argument("--unbounded-nonperiodic", action="store_true",
                    help="Emit non-periodic ('sporadic') tasks WITHOUT "
                         "min_start_t/max_end_t. They then carry no release "
                         "window or deadline and the scheduler packs them as "
                         "early as dependencies allow, while periodic tasks "
                         "stay period-bound. Also stops the horizon being "
                         "widened to cover the sporadic span, so the horizon "
                         "is driven purely by the periodic hyperperiod -- "
                         "which is what removes the idle-gap degeneracy that "
                         "the cursor-based window layout produces.")
    ap.add_argument("--no-horizon-covers-nonperiodic",
                    dest="horizon_covers_nonperiodic",
                    action="store_false", default=True,
                    help="(with --unbounded-nonperiodic) do NOT extend the "
                         "horizon to cover the estimated completion of the "
                         "unbounded non-periodic work. Default is to extend "
                         "it, so periodic groups keep ticking for the whole "
                         "schedule; pass this to let the horizon stay at the "
                         "periodic hyperperiod instead.")
    ap.add_argument("--cap-instances", type=int, default=None,
                    help="with --num-instances auto, cap the derived count at "
                         "this many instances per network")

    ap.add_argument("--solver-verbosity", type=int, default=2)
    ap.add_argument("--time-limit", type=int, default=20)
    args = ap.parse_args(argv)

    args.repo_root = os.path.abspath(args.repo_root)
    hw_bank, model_bank = load_banks(args.banks_dir)

    if args.list:
        return do_list(args.repo_root, hw_bank, model_bank)
    if args.seed is None:
        ap.error("seed is required (or pass --list)")
    if not 0 < args.min_scale <= args.max_scale:
        ap.error("--min-scale must be > 0 and <= --max-scale")
    if args.period_headroom < 1.0:
        ap.error("--period-headroom must be >= 1.0")
    if not 0 < args.max_utilization <= 1.0:
        ap.error("--max-utilization must be in (0, 1]")
    if args.horizon_periods <= 0:
        ap.error("--horizon-periods must be > 0")
    if args.max_ops < 1:
        ap.error("--max-ops must be >= 1")
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

    # The bank entry we generated from, taken from the notes rather than
    # matched back by (target, machines): spike_het and spike_single_core
    # are the same target with the same machine counts, so the reverse
    # lookup named every spike_single_core workload "spike_het".
    hw_name = config["_comment"][1].split("Hardware: ", 1)[1].split(" --", 1)[0]
    out = args.out or os.path.join(
        args.repo_root, "data", "toplevel", "generated-data",
        f"networks_random_{hw_name}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)

    nets = config["networks"]
    per = {n: v for n, v in nets.items() if "period" in v}
    print(f"wrote {out}")
    print(f"  hardware : {hw_name}  {config['hardware']['profile_hw']}")
    print(f"  horizon  : {config['horizon_ms']:.0f} ms "
          f"(hyperperiod {config['hyperperiod_ms']} ms)")
    print(f"  networks : {len(nets)} ({len(per)} periodic, "
          f"{len(nets) - len(per)} sporadic)")
    for n, v in per.items():
        print(f"    {n:<22} period {v['period']:>6} ms  window "
              f"{v['window_duration']:>6} ms  x{v['num_instances']} instances")
    for n, v in nets.items():
        if n not in per:
            if "min_start_t" in v:
                print(f"    {n:<22} sporadic  [{v['min_start_t']:.0f}, "
                      f"{v['max_end_t']:.0f}] ms")
            else:
                print(f"    {n:<22} non-periodic (unbounded: no release "
                      f"window, scheduled as early as deps allow)")
    print(f"  edges    : {len(config['edges'])} (acyclic, chains within a rate group)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
