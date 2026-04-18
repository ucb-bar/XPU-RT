#!/usr/bin/env python3
"""
Worst-case duration of all layers of a *periodic* (windowed) task vs its period window.

For each group of JSON networks that share the same periodic workload (same
`identifier` + `dispatch_deps_path`) and use explicit time windows
(`min_start_t`, `max_end_t`), we compute:

  S = sum over dispatch-graph nodes of max(time_ms on CPU_P HW, time_ms on CPU_E HW)
  W = window length = max_end_t - min_start_t  (must be identical across slices; default 55 ms in your setup)

  fraction = S / W   (worst-case fraction of one period window consumed by one full graph run)

This matches "every layer runs on the non-suitable (slower) core" per layer, summed.

Generalization:
  - Any number of slice instances (dronet0..dronetN) with the same W and same graph.
  - Optional `--window-ms` overrides W if JSON slices disagree or you want 1/frequency directly.
  - Optional `--include-automatic-periodic`: also handle a single JSON entry with both
    `period` and `window_duration` (no unrolled keys); then W = window_duration, N = 1 for fraction.

Usage:
  python scripts/worst_case_periodic_window_fraction.py \\
    --networks-json data/toplevel/networks_periodic_profile.json

  python scripts/worst_case_periodic_window_fraction.py \\
    --networks-json data/toplevel/networks_periodic_profile.json \\
    --only-identifier dronet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
_sys_xpu = os.path.join(_repo_root, "xpu-rt")
for _p in (_sys_xpu, _repo_root, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import worst_case_nonperiodic_duration as wcnp  # noqa: E402


def _is_windowed_slice(net_info: dict) -> bool:
    return (
        net_info.get("min_start_t") is not None
        and net_info.get("max_end_t") is not None
    )


def _is_automatic_periodic(net_info: dict) -> bool:
    return net_info.get("period") is not None and net_info.get(
        "window_duration"
    ) is not None


def _window_ms_from_slice(net_info: dict) -> float:
    return float(net_info["max_end_t"]) - float(net_info["min_start_t"])


def _group_key_windowed(net_info: dict) -> Tuple[str, str]:
    ident = str(net_info.get("identifier") or "")
    path = str(net_info.get("dispatch_deps_path") or "")
    return (ident, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worst-case layer sum / period window for windowed periodic networks."
    )
    parser.add_argument(
        "--networks-json",
        type=str,
        default="data/toplevel/networks_periodic_profile.json",
    )
    parser.add_argument("--repo-root", type=str, default=_repo_root)
    parser.add_argument(
        "--window-ms",
        type=float,
        default=None,
        help="Override window W (ms). If omitted, taken from each slice (max_end_t - min_start_t).",
    )
    parser.add_argument(
        "--only-identifier",
        type=str,
        default=None,
        help="Only groups whose identifier matches (e.g. dronet).",
    )
    parser.add_argument(
        "--include-automatic-periodic",
        action="store_true",
        help="Also include entries with period+window_duration (no min/max slice).",
    )
    parser.add_argument(
        "--window-epsilon",
        type=float,
        default=1e-6,
        help="Max abs diff allowed between slice windows when checking consistency.",
    )
    parser.add_argument(
        "--dump-details",
        action="store_true",
        help="Include per-node rows from worst_case_sum_for_network in JSON output.",
    )
    args = parser.parse_args()

    repo_base = os.path.abspath(args.repo_root)
    nj = args.networks_json
    if not os.path.isabs(nj):
        nj = os.path.join(repo_base, nj)
    with open(nj) as f:
        data = json.load(f)

    gen_root, target, topo, hw_p, hw_e = wcnp._load_hardware(data, repo_base, nj)
    networks: Dict[str, Any] = data.get("networks") or {}

    windowed_groups: Dict[Tuple[str, str], List[Tuple[str, dict]]] = defaultdict(list)
    automatic_candidates: List[Tuple[str, dict]] = []

    for net_key, net_info in networks.items():
        if not isinstance(net_info, dict):
            continue
        if _is_windowed_slice(net_info):
            windowed_groups[_group_key_windowed(net_info)].append((net_key, net_info))
        elif _is_automatic_periodic(net_info):
            automatic_candidates.append((net_key, net_info))

    results: List[dict] = []

    for (ident, dpath), members in sorted(windowed_groups.items()):
        if args.only_identifier and ident != args.only_identifier:
            continue
        if not ident or not dpath:
            continue
        wins = [_window_ms_from_slice(m[1]) for m in members]
        if args.window_ms is not None:
            W = float(args.window_ms)
            win_note = "cli_override"
        else:
            W = wins[0]
            for w in wins[1:]:
                if abs(w - W) > args.window_epsilon:
                    raise ValueError(
                        f"Group identifier={ident!r}: inconsistent window ms "
                        f"{wins!r}; use --window-ms or fix JSON."
                    )
            win_note = "from_json_slices"

        rep_key, rep_info = members[0]
        S, details, meta = wcnp.worst_case_sum_for_network(
            repo_base,
            rep_key,
            rep_info,
            gen_root,
            target,
            topo,
            hw_p,
            hw_e,
        )
        N = len(members)
        if W <= 0:
            raise ValueError(f"Non-positive window W={W} for identifier={ident!r}")

        frac = S / W
        out = {
            "group": "windowed_slices",
            "identifier": ident,
            "dispatch_deps_path": dpath,
            "slice_network_keys": [m[0] for m in members],
            "num_slice_instances": N,
            "window_ms": W,
            "window_source": win_note,
            "worst_case_layer_sum_ms": S,
            "fraction_worst_case_sum_over_window": frac,
            "total_worst_case_ms_all_slices_serial": N * S,
            "total_window_budget_ms": N * W,
            "fraction_total_sum_over_total_windows": (N * S) / (N * W),
            "profile": {
                "gen_root": gen_root,
                "cpu_p_hw": hw_p,
                "cpu_e_hw": hw_e,
                "csv_p": meta.get("csv_p"),
                "csv_e": meta.get("csv_e"),
                "model_profile": meta.get("model_profile"),
            },
            "missing_profile_nodes": meta.get("missing"),
            "num_graph_nodes": len(details),
        }
        if args.dump_details:
            out["per_node"] = details
        results.append(out)

    if args.include_automatic_periodic:
        for net_key, net_info in automatic_candidates:
            if args.only_identifier and str(net_info.get("identifier")) != args.only_identifier:
                continue
            W = float(net_info["window_duration"])
            if args.window_ms is not None:
                W = float(args.window_ms)
            S, details, meta = wcnp.worst_case_sum_for_network(
                repo_base,
                net_key,
                net_info,
                gen_root,
                target,
                topo,
                hw_p,
                hw_e,
            )
            frac = S / W if W > 0 else float("nan")
            out = {
                "group": "automatic_periodic_json",
                "network_key": net_key,
                "identifier": net_info.get("identifier"),
                "dispatch_deps_path": net_info.get("dispatch_deps_path"),
                "period_ms": float(net_info.get("period", 0) or 0),
                "window_ms": W,
                "worst_case_layer_sum_ms": S,
                "fraction_worst_case_sum_over_window": frac,
                "profile": {
                    "gen_root": gen_root,
                    "cpu_p_hw": hw_p,
                    "cpu_e_hw": hw_e,
                    "csv_p": meta.get("csv_p"),
                    "csv_e": meta.get("csv_e"),
                    "model_profile": meta.get("model_profile"),
                },
                "missing_profile_nodes": meta.get("missing"),
                "num_graph_nodes": len(details),
            }
            if args.dump_details:
                out["per_node"] = details
            results.append(out)

    print(
        json.dumps(
            {
                "networks_json": nj,
                "groups": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
