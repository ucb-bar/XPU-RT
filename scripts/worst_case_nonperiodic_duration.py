#!/usr/bin/env python3
"""
Sum per-layer worst-case mean durations for non-periodic networks in a networks JSON.

For each dispatch node in each included network's dispatch graph, we take
    worst_ms(dispatch_id) = max(time_ms on CPU_P profile HW, time_ms on CPU_E profile HW)
where times come from paired results.csv under <gen_root>/profile/...

Default inclusion rule (see --include-windowed-instances):
  - Skip JSON entries that declare automatic periodic expansion: both `period` and
    `window_duration` are set.
  - Skip explicit time-window slice instances: both `min_start_t` and `max_end_t`
    are set (e.g. dronet0..dronet4), unless --include-windowed-instances.

If the same logical dispatch_id appears in multiple graph nodes (e.g. fused
variants dispatch_13_1, dispatch_13_2 with the same id), this script counts the
worst-case time once per node (sequential "every op runs on the slower core").

Usage:
  python scripts/worst_case_nonperiodic_duration.py \\
    --networks-json data/toplevel/networks_periodic_profile.json

  python scripts/worst_case_nonperiodic_duration.py \\
    --networks-json data/toplevel/networks_periodic_profile.json \\
    --only mobilenet
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys_xpu = os.path.join(_repo_root, "xpu-rt")
for _p in (_sys_xpu, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xpu_rt.scheduler.workload_factory import resolve_dispatch_deps_path  # noqa: E402


def _first_nonempty(*vals: Optional[str], default: str = "") -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def _load_hardware(
    networks_data: dict,
    repo_base: str,
    networks_json_path: str,
) -> Tuple[str, str, str, str, str, str]:
    """Returns gen_root, target, topo_tag, hw_p, hw_e."""
    hw = networks_data.get("hardware") or {}
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


def _basename_from_dispatch_deps_path(path: str) -> str:
    if not path:
        return ""
    return os.path.basename(os.path.dirname(path))


def _model_candidates(net_key: str, net_info: dict, basename: str) -> List[str]:
    out: List[str] = []
    bm = os.path.basename(basename).split(".")[0] if basename else ""
    for c in (net_key, net_info.get("identifier"), bm):
        if isinstance(c, str) and c and c not in out:
            out.append(c)
    return out or [net_key]


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
) -> Tuple[Optional[str], Optional[str], str]:
    basename = _basename_from_dispatch_deps_path(dispatch_rel)
    for model in _model_candidates(net_key, net_info, basename):
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
            return csv_p, csv_e, model
    return None, None, _model_candidates(net_key, net_info, basename)[0]


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


def _is_automatic_periodic(net_info: dict) -> bool:
    return net_info.get("period") is not None and net_info.get(
        "window_duration"
    ) is not None


def _is_windowed_slice(net_info: dict) -> bool:
    return (
        net_info.get("min_start_t") is not None
        and net_info.get("max_end_t") is not None
    )


def _should_include_network(
    net_key: str,
    net_info: dict,
    *,
    only_keys: Optional[List[str]],
    include_windowed: bool,
) -> bool:
    if only_keys and net_key not in only_keys:
        return False
    if _is_automatic_periodic(net_info):
        return False
    if _is_windowed_slice(net_info) and not include_windowed:
        return False
    return True


def worst_case_sum_for_network(
    repo_base: str,
    net_key: str,
    net_info: dict,
    gen_root: str,
    target: str,
    topo: str,
    hw_p: str,
    hw_e: str,
) -> Tuple[float, List[dict], dict]:
    """
    Returns (total_worst_case_ms, detail_rows, meta).
    """
    rel = (net_info.get("dispatch_deps_path") or "").strip()
    full = resolve_dispatch_deps_path(repo_base, rel)
    if not full or not os.path.exists(full):
        raise FileNotFoundError(f"dispatch graph not found for '{net_key}': {rel!r}")

    csv_p, csv_e, model_used = _pick_csv_pair_for_network(
        repo_base,
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

    total = 0.0
    details: List[dict] = []
    missing: List[Tuple[str, int]] = []

    for node_key, node in dispatches.items():
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
            missing.append((node_key, did))
            continue
        if t_p is None:
            worst = float(t_e)
        elif t_e is None:
            worst = float(t_p)
        else:
            worst = max(float(t_p), float(t_e))
        total += worst
        details.append(
            {
                "network": net_key,
                "graph_node": node_key,
                "dispatch_id": did,
                "ms_cpu_p": t_p,
                "ms_cpu_e": t_e,
                "worst_ms": worst,
            }
        )

    return total, details, {
        "network": net_key,
        "model_profile": model_used,
        "csv_p": csv_p,
        "csv_e": csv_e,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum max(RVV,scalar) profiled times per dispatch node for non-periodic networks."
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default="data/toplevel/networks_periodic_profile.json",
        help="Networks JSON (same shape as run_xpurt_schedule).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=_repo_root,
        help="XPU-RT repo root.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="If set, only these network keys (e.g. mobilenet).",
    )
    parser.add_argument(
        "--include-windowed-instances",
        action="store_true",
        help="Include networks with min_start_t+max_end_t (default: exclude).",
    )
    parser.add_argument(
        "--dump-details",
        action="store_true",
        help="Print per-node table to stdout.",
    )
    args = parser.parse_args()

    repo_base = os.path.abspath(args.repo_root)
    nj = args.networks_json
    if not os.path.isabs(nj):
        nj = os.path.join(repo_base, nj)
    with open(nj) as f:
        networks_data = json.load(f)

    gen_root, target, topo, hw_p, hw_e = _load_hardware(
        networks_data, repo_base, nj
    )
    networks: Dict[str, Any] = networks_data.get("networks") or {}

    grand_total = 0.0
    per_net: List[dict] = []

    for net_key, net_info in networks.items():
        if not isinstance(net_info, dict):
            continue
        if not _should_include_network(
            net_key,
            net_info,
            only_keys=list(args.only) if args.only else None,
            include_windowed=args.include_windowed_instances,
        ):
            continue
        total, details, meta = worst_case_sum_for_network(
            repo_base,
            net_key,
            net_info,
            gen_root,
            target,
            topo,
            hw_p,
            hw_e,
        )
        grand_total += total
        meta["worst_case_total_ms"] = total
        meta["num_nodes"] = len(details)
        per_net.append(meta)
        if meta["missing"]:
            print(
                f"warning: {net_key}: {len(meta['missing'])} graph nodes had no "
                f"profile row on either core (first: {meta['missing'][:3]})",
                file=sys.stderr,
            )
        if args.dump_details:
            for row in details:
                print(row)

    print(
        json.dumps(
            {
                "networks_json": nj,
                "gen_root": gen_root,
                "profile_hw_cpu_p": hw_p,
                "profile_hw_cpu_e": hw_e,
                "included_networks": [m["network"] for m in per_net],
                "per_network_ms": {m["network"]: m["worst_case_total_ms"] for m in per_net},
                "grand_total_worst_case_ms": grand_total,
                "meta": per_net,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
