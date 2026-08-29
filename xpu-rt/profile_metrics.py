"""
Profile-derived worst-case timing metrics for horizon heuristics.

Used by workload_factory to replace a fixed 2.0 multiplier when
hardware.profile + profiled results.csv data are available.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import workload_spec
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _first_nonempty(*vals: Optional[str], default: str = "") -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def parse_profile_hardware(networks_data: dict) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Returns (gen_root, target, topo_tag, hw_p, hw_e) or None if hardware block missing.
    """
    hw = networks_data.get("hardware")
    if not isinstance(hw, dict):
        return None
    prof = hw.get("profile") or {}
    phw = hw.get("profile_hw") or {}
    gen_root = _first_nonempty(
        prof.get("gen_root"),
        hw.get("gen_root"),
        default="gen",
    ).strip().strip("/\\")
    target = _first_nonempty(
        prof.get("target"),
        hw.get("profile_target"),
        default="spacemit_x60",
    )
    topo = _first_nonempty(
        prof.get("topo_tag"),
        hw.get("topo_tag"),
        default="topo_0_1_2_3",
    )
    hw_p = _first_nonempty(
        phw.get("cpu_p"),
        hw.get("cpu_p_profile_hw"),
        default="RVV",
    )
    hw_e = _first_nonempty(
        phw.get("cpu_e"),
        hw.get("cpu_e_profile_hw"),
        default="scalar",
    )
    return gen_root, target, topo, hw_p, hw_e


def _find_profile_csv(
    repo_base: str,
    *,
    gen_root: str,
    model: str,
    target: str,
    hw: str,
    basename: str,
    topo_tag: str,
) -> Optional[str]:
    profile_root = os.path.join(repo_base, gen_root, "profile")
    pat1 = os.path.join(
        profile_root, hw, target, model, basename, "*", topo_tag, "results.csv"
    )
    matches = glob.glob(pat1)
    if not matches:
        pat2 = os.path.join(
            profile_root, hw, target, model, basename, topo_tag, "results.csv"
        )
        matches = glob.glob(pat2)
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))


def _pick_csv_pair_for_network(
    repo_base: str,
    gen_root: str,
    target: str,
    topo: str,
    hw_p: str,
    hw_e: str,
    net_key: str,
    net_info: dict,
    dispatch_rel: str,
) -> Tuple[Optional[str], Optional[str]]:
    basename = workload_spec.basename_from_dispatch_deps_path(dispatch_rel)
    for model in workload_spec.model_candidates(net_key, net_info, basename):
        csv_p = _find_profile_csv(
            repo_base,
            gen_root=gen_root,
            model=model,
            target=target,
            hw=hw_p,
            basename=basename,
            topo_tag=topo,
        )
        csv_e = _find_profile_csv(
            repo_base,
            gen_root=gen_root,
            model=model,
            target=target,
            hw=hw_e,
            basename=basename,
            topo_tag=topo,
        )
        if csv_p or csv_e:
            return csv_p, csv_e
    return None, None


def load_times_ms_by_dispatch_id(csv_path: Optional[str]) -> Dict[int, float]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    out: Dict[int, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get("dispatch_id")
            if raw_id is None or str(raw_id).strip() == "":
                continue
            try:
                did = int(raw_id)
            except ValueError:
                continue
            try:
                mean = float(row.get("mean_time", 0.0))
            except ValueError:
                continue
            unit = (row.get("mean_unit") or "ms").strip()
            if unit == "us":
                mean_ms = mean / 1000.0
            elif unit == "s":
                mean_ms = mean * 1000.0
            else:
                mean_ms = mean
            out[did] = mean_ms
    return out


def worst_case_layer_sum_ms_for_network(
    repo_base_path: str,
    net_key: str,
    net_info: dict,
    gen_root: str,
    target: str,
    topo: str,
    hw_p: str,
    hw_e: str,
    *,
    per_dispatch: str = "max",
) -> Optional[float]:
    """
    Sum over dispatch-graph nodes of {max,min}(time_cpu_p, time_cpu_e) in ms.

    `per_dispatch` controls the per-node aggregation:
      - "max": worst HW per op (legacy default, used by external callers
        that want a robust upper bound across HW assignments).
      - "min": best HW per op — i.e. the schedule's best-case lower bound.
        Used by `profile_based_horizon_ms` because the horizon already
        adds the periodic-interference factor 1/(1-F_p) on top; per-op
        max would double-count by assuming the scheduler picks the worst
        HW for every op AND that periodic still steals (1-F_p) of CPU.
    """
    if per_dispatch not in ("max", "min"):
        raise ValueError(f"per_dispatch must be 'max' or 'min', got {per_dispatch!r}")
    # Lazy import avoids circular import at module load.
    from workload_factory import resolve_dispatch_deps_path

    rel = (net_info.get("dispatch_deps_path") or "").strip()
    full = resolve_dispatch_deps_path(repo_base_path, rel)
    if not full or not os.path.exists(full):
        return None

    csv_p, csv_e = _pick_csv_pair_for_network(
        repo_base_path,
        gen_root,
        target,
        topo,
        hw_p,
        hw_e,
        net_key,
        net_info,
        rel,
    )
    times_p = load_times_ms_by_dispatch_id(csv_p)
    times_e = load_times_ms_by_dispatch_id(csv_e)

    with open(full) as f:
        graph = json.load(f)
    dispatches: Dict[str, Any] = graph.get("dispatches") or {}

    if not dispatches:
        return None

    total = 0.0
    n_used = 0
    for node in dispatches.values():
        if not isinstance(node, dict):
            continue
        did = node.get("id")
        if not isinstance(did, int):
            try:
                did = int(did)
            except (TypeError, ValueError):
                continue
        t_p = times_p.get(did)
        t_e = times_e.get(did)
        if t_p is None and t_e is None:
            continue
        n_used += 1
        if t_p is None:
            total += float(t_e)
        elif t_e is None:
            total += float(t_p)
        else:
            agg = max if per_dispatch == "max" else min
            total += agg(float(t_p), float(t_e))
    if n_used == 0:
        return None
    return total


def max_periodic_window_fraction(
    networks: Dict[str, dict],
    repo_base_path: str,
    gen_root: str,
    target: str,
    topo: str,
    hw_p: str,
    hw_e: str,
    *,
    window_epsilon: float = 1e-6,
) -> Optional[float]:
    """
    Largest S_p / D among periodic workloads, where D is the period T for
    JSON automatic-periodic entries (when present) and the slice window W
    otherwise. The period denominator is the queuing-theory utilization
    fraction (worst-case CPU consumed by the periodic per unit time);
    using window_duration overestimates utilization when W < T.
      - JSON automatic periodic: D = period (falls back to window_duration
        when period is absent)
      - Windowed slice groups (same identifier + dispatch_deps_path):
        D = max_end_t - min_start_t (matches the legacy semantics —
        these slices have no separate period field)
    """
    frac_max = 0.0

    for net_key, net_info in networks.items():
        if not isinstance(net_info, dict):
            continue
        if not workload_spec.is_automatic_periodic(net_info):
            continue
        # Prefer period T; fall back to window_duration if T is absent.
        try:
            period = float(net_info.get("period")) if net_info.get("period") is not None else None
        except (TypeError, ValueError):
            period = None
        try:
            wdur = float(net_info.get("window_duration")) if net_info.get("window_duration") is not None else None
        except (TypeError, ValueError):
            wdur = None
        denom = period if (period is not None and period > 0) else wdur
        if denom is None or denom <= 0:
            continue
        s_p = worst_case_layer_sum_ms_for_network(
            repo_base_path,
            net_key,
            net_info,
            gen_root,
            target,
            topo,
            hw_p,
            hw_e,
        )
        if s_p is None or s_p <= 0:
            continue
        frac_max = max(frac_max, s_p / denom)

    windowed_groups: Dict[Tuple[str, str], List[Tuple[str, dict]]] = defaultdict(list)
    for net_key, net_info in networks.items():
        if not isinstance(net_info, dict):
            continue
        if not workload_spec.is_windowed_slice(net_info):
            continue
        ident = str(net_info.get("identifier") or "")
        dpath = str(net_info.get("dispatch_deps_path") or "")
        if not ident or not dpath:
            continue
        windowed_groups[(ident, dpath)].append((net_key, net_info))

    for (_ident, _dpath), members in windowed_groups.items():
        wins = [
            float(m[1]["max_end_t"]) - float(m[1]["min_start_t"])
            for m in members
            if m[1].get("max_end_t") is not None and m[1].get("min_start_t") is not None
        ]
        if not wins:
            continue
        w0 = wins[0]
        if any(abs(w - w0) > window_epsilon for w in wins[1:]):
            continue
        if w0 <= 0:
            continue
        rep_key, rep = members[0]
        s_p = worst_case_layer_sum_ms_for_network(
            repo_base_path,
            rep_key,
            rep,
            gen_root,
            target,
            topo,
            hw_p,
            hw_e,
        )
        if s_p is None or s_p <= 0:
            continue
        frac_max = max(frac_max, s_p / w0)

    if frac_max <= 0.0:
        return None
    return frac_max


def nonperiodic_worst_case_layer_sum_ms(
    networks: Dict[str, dict],
    repo_base_path: str,
    gen_root: str,
    target: str,
    topo: str,
    hw_p: str,
    hw_e: str,
    *,
    per_dispatch: str = "max",
) -> Optional[float]:
    """
    Sum of per-network worst-case layer sums for networks that are neither
    automatic periodic nor windowed time-slice instances. `per_dispatch`
    is forwarded to worst_case_layer_sum_ms_for_network and controls
    whether each dispatch is summed at its worst-HW (default; matches
    scripts/worst_case_nonperiodic_duration.py) or best-HW choice.
    """
    total = 0.0
    any_ok = False
    for net_key, net_info in networks.items():
        if not isinstance(net_info, dict):
            continue
        if workload_spec.is_automatic_periodic(net_info):
            continue
        if workload_spec.is_windowed_slice(net_info):
            continue
        s = worst_case_layer_sum_ms_for_network(
            repo_base_path,
            net_key,
            net_info,
            gen_root,
            target,
            topo,
            hw_p,
            hw_e,
            per_dispatch=per_dispatch,
        )
        if s is None:
            continue
        any_ok = True
        total += s
    if not any_ok:
        return None
    return total


def profile_based_horizon_ms(networks_data: dict, repo_base_path: str) -> Optional[float]:
    """
    Horizon (ms) = S_np / (1 - F_p) where:
      S_np = sum of worst-case per-graph layer sums for non-periodic, non-windowed networks
      F_p  = max over periodic groups of (S_p / T)  — utilization fraction over the
             period (NOT window): periodic tasks consume f of CPU per unit time on
             average, so nonperiodic gets (1 - f) of CPU and needs S_np / (1 - f)
             wall time to complete.

    Returns None if hardware/profile data or CSVs are missing, metrics cannot be
    computed, or F_p >= 1 (periodic is unschedulable on its own).
    """
    parsed = parse_profile_hardware(networks_data)
    if parsed is None:
        return None
    gen_root, target, topo, hw_p, hw_e = parsed
    networks = networks_data.get("networks") or {}
    if not isinstance(networks, dict):
        return None

    # Best-HW-per-op for nonperiodic. Combined with the 1/(1-F_p)
    # interference factor below, this approximates the schedule the
    # scheduler will actually find: each op runs on its best HW, but
    # periodic slices steal F_p of CPU on average. Per-op max here would
    # double-count (worst HW + interference) and inflate the horizon
    # 5-10x for heterogeneous configs (e.g. dronet RVV=39ms scalar=297ms).
    s_np = nonperiodic_worst_case_layer_sum_ms(
        networks, repo_base_path, gen_root, target, topo, hw_p, hw_e,
        per_dispatch="min",
    )
    f_p = max_periodic_window_fraction(
        networks, repo_base_path, gen_root, target, topo, hw_p, hw_e
    )
    if s_np is None or s_np <= 0.0 or f_p is None or f_p <= 0.0:
        return None
    if f_p >= 1.0:
        # Periodic alone already saturates its period. Caller must
        # decide what to do — we can't bound a horizon meaningfully.
        return None
    return s_np / (1.0 - f_p)
