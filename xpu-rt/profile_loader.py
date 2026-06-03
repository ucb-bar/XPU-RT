"""
Profile data loading utilities for the XPU-RT scheduler.

Handles discovering, parsing, and assembling profiled dispatch runtimes
from CSV files produced by runtime/scripts/profile_remote.sh.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from workload_factory import (
    resolve_dispatch_deps_path,
    topo_tag_for_combination,
    machine_type_prefix,
)


def load_profiled_times(csv_path: str) -> dict[int, dict]:
    """
    Load profiled runtimes from a CSV file.

    Expected columns:
      - dispatch_id
      - module_name (optional)
      - mean_time
      - mean_unit (assumed 'ms' if missing)

    Returns:
      dict mapping dispatch_id (int) -> {"time_ms": float, "module_name": str}
    """
    profiled: dict[int, dict] = {}
    if not os.path.exists(csv_path):
        return profiled
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dispatch_id_str = row.get("dispatch_id")
            if not dispatch_id_str:
                continue
            try:
                dispatch_id = int(dispatch_id_str)
            except ValueError:
                continue

            module_name = row.get("module_name", "")
            try:
                mean_time = float(row.get("mean_time", 0.0))
            except ValueError:
                continue
            unit = row.get("mean_unit", "ms")
            if unit == "us":
                mean_time_ms = mean_time / 1000.0
            elif unit == "s":
                mean_time_ms = mean_time * 1000.0
            else:
                mean_time_ms = mean_time
            profiled[dispatch_id] = {
                "time_ms": mean_time_ms,
                "module_name": module_name,
            }
    return profiled


def find_profile_csv(
    repo_base_path: str,
    *,
    model: str,
    target: str,
    hw: str,
    basename: str,
    topo_tag: str = "topo_0_1_2_3",
) -> str | None:
    """
    Find a profiling results.csv produced by runtime/scripts/profile_remote.sh.

    Expected layout:
      gen/profile/<hw>/<target>/<model>/<basename>/<input_tag>/<topo_tag>/results.csv

    We pick the most recently modified match.
    """
    profile_root = os.path.join(repo_base_path, "gen", "profile")

    # New layout (with input_tag subdir).
    pat1 = os.path.join(profile_root, hw, target, model, basename, "*", topo_tag, "results.csv")
    matches = glob.glob(pat1)

    # Back-compat layout (no input_tag subdir).
    if not matches:
        pat2 = os.path.join(profile_root, hw, target, model, basename, topo_tag, "results.csv")
        matches = glob.glob(pat2)

    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))


def _basename_from_dispatch_deps_path(path: str) -> str:
    """Extract the parent directory name from a dispatch deps path."""
    return os.path.basename(os.path.dirname(path)) if path else ""


def _model_candidates(net_id: str, net_info: dict, dispatch_deps_path: str) -> list[str]:
    """Return candidate model names to try when searching for profiling CSVs."""
    basename = _basename_from_dispatch_deps_path(dispatch_deps_path) or f"{net_id}.q.int8"
    basename_model = os.path.basename(basename).split(".")[0]
    candidates = []
    for c in (net_id, net_info.get("identifier"), basename_model):
        if isinstance(c, str) and c and c not in candidates:
            candidates.append(c)
    return candidates or [net_id]


def _resolve_topo_for(
    hw: str, combo: list[str], topo_tag_override
) -> str:
    """Pick the topo tag to use for a given (hw, combo).

    `topo_tag_override` may be:
      - None: use topo_tag_for_combination(combo)
      - str:  apply the same override to every (hw, combo)
      - dict[str, str]: per-hw override; falls back to combo size if hw
        not in the dict.
    """
    if topo_tag_override is None:
        return topo_tag_for_combination(combo)
    if isinstance(topo_tag_override, str):
        return topo_tag_override
    if isinstance(topo_tag_override, dict):
        if hw in topo_tag_override:
            return topo_tag_override[hw]
        return topo_tag_for_combination(combo)
    raise TypeError(
        f"topo_tag_override must be None, str, or dict[str, str]; "
        f"got {type(topo_tag_override).__name__}"
    )


def _load_all_topo_profiles(
    net_id: str,
    net_info: dict,
    dispatch_deps_path: str,
    repo_base_path: str,
    profile_target: str,
    combo_hw: list[str],
    machine_combinations: list[list[str]],
    topo_tag_override=None,
) -> dict[tuple[str, str], dict[int, dict]]:
    """
    Load profiles for all (hw, topo) combinations for a network.

    Returns {(hw, topo): {dispatch_id: {"time_ms", "module_name"}}}.

    `topo_tag_override`: see _resolve_topo_for. None = derive from
    combo size; str = single override applied to every (hw, combo);
    dict[hw, topo] = per-hw override that lets a "cluster" cpu_e (one
    machine, four physical harts under the hood) read its multi-core
    profile data while a singleton cpu_p still reads topo_0.
    """
    basename = _basename_from_dispatch_deps_path(dispatch_deps_path) or f"{net_id}.q.int8"
    candidates = _model_candidates(net_id, net_info, dispatch_deps_path)
    profiles: dict[tuple[str, str], dict[int, dict]] = {}

    hw_types = set(combo_hw)
    # Build the (hw, topo) cross-product to look up. With per-hw overrides
    # the same hw might still pick different topos across combos (it
    # shouldn't in practice, since each kind has one cluster), but we
    # handle that by collecting topos across all (hw, combo) pairs.
    pairs: set[tuple[str, str]] = set()
    for hw in hw_types:
        for combo in machine_combinations:
            # only meaningful (hw, combo) pairs — combo's hw matters
            # because we look up profile under that hw. Iterate hw_types
            # to be safe; the resolver picks the topo.
            pairs.add((hw, _resolve_topo_for(hw, combo, topo_tag_override)))

    for hw, topo in pairs:
        for model_candidate in candidates:
            csv_path = find_profile_csv(
                repo_base_path, model=model_candidate,
                target=profile_target, hw=hw, basename=basename, topo_tag=topo,
            )
            if csv_path:
                prof = load_profiled_times(csv_path)
                if prof:
                    profiles[(hw, topo)] = prof
                    if model_candidate != net_id:
                        print(f"  (info) profile fallback: {net_id}/{hw}/{topo} -> model={model_candidate}")
                break
    return profiles


def load_profiled_processing_times(
    networks: dict,
    repo_base_path: str,
    machine_combinations: list[list[str]],
    combo_hw: list[str],
    profile_target: str,
    cpu_p_profile_hw: str,
    cpu_e_profile_hw: str,
    rng: np.random.Generator,
    p_core_speedup: float,
    topo_tag_override=None,
    strict: bool = True,
) -> tuple[dict[str, list[float]], dict[int, dict], dict[int, dict], dict[str, dict[str, dict[int, dict]]]]:
    """
    Load profiled processing times for all networks and dispatches.

    For each dispatch in each network, builds a list of processing times
    (one per machine combination) from profiled CSVs.

    `strict` (default True): when a (network, hw, topo) profile CSV is
    missing — or has no entry for a specific dispatch_id — raise loudly.
    The previous behaviour silently substituted ``rng.uniform(2.0, 10.0)``
    per missing dispatch, which let schedules be generated against
    fictional timings (every yolov8_nano-on-RVV op was a random number
    on the FireSim run, because that profile sweep had never been
    captured). The synthetic-fallback path is preserved behind
    `strict=False` for cases where partial coverage is intentional.

    Returns:
      (processing_times, combined_profiled_p, combined_profiled_e,
       profiled_by_network) where:
      - processing_times: {prefixed_dispatch_name: [time_per_combination]}
      - combined_profiled_p: {dispatch_id: {"time_ms", "module_name"}} for P-cores
        — keyed by dispatch_id alone, so multi-network workloads collide:
        the second network's update() overwrites the first's entries for
        any shared dispatch_id. Use `profiled_by_network` for any
        consumer that needs to identify which network a dispatch came
        from (e.g. emitting `module_name` into the schedule JSON).
      - combined_profiled_e: same shape as combined_profiled_p, for E-cores.
      - profiled_by_network: {net_id: {"p": {...}, "e": {...}}} —
        network-keyed view of the same data, no collisions. `net_id` is
        the base network identifier from `networks` (e.g. "dronet",
        "yolov8_nano"), not a periodic instance like "dronet0".
    """
    processing_times: dict[str, list[float]] = {}
    combined_profiled_p: dict[int, dict] = {}
    combined_profiled_e: dict[int, dict] = {}
    profiled_by_network: dict[str, dict[str, dict[int, dict]]] = {}
    # Aggregate missing-data findings before raising so the user sees
    # *every* gap at once, not just the first one — saves an iter cycle.
    missing: list[str] = []

    for net_id, net_info in networks.items():
        dispatch_deps_path = net_info.get("dispatch_deps_path", "")
        full_dispatch_path = resolve_dispatch_deps_path(repo_base_path, dispatch_deps_path)
        if not os.path.exists(full_dispatch_path):
            if strict:
                missing.append(
                    f"  - {net_id}: dispatch_deps_path not found at "
                    f"{full_dispatch_path!r}"
                )
            continue

        all_profiles = _load_all_topo_profiles(
            net_id, net_info, dispatch_deps_path,
            repo_base_path, profile_target, combo_hw, machine_combinations,
            topo_tag_override=topo_tag_override,
        )
        # Verify every (hw, topo) requested by the schedule has a profile.
        hw_types = set(combo_hw)
        requested_pairs: set[tuple[str, str]] = set()
        for hw in hw_types:
            for combo in machine_combinations:
                requested_pairs.add((hw, _resolve_topo_for(hw, combo, topo_tag_override)))
        for (hw, topo) in requested_pairs:
            if (hw, topo) not in all_profiles:
                if strict:
                    missing.append(
                        f"  - {net_id} @ {hw}/{topo}: no profile CSV under "
                        f"gen/profile/{hw}/{profile_target}/<model>/<basename>/.../{topo}/results.csv"
                    )
        if not all_profiles:
            if not strict:
                continue
            # else: error after the loop with the full list.

        with open(full_dispatch_path, "r") as f:
            dispatch_data = json.load(f)
        dispatches = dispatch_data.get("dispatches", {})

        net_prefix = f"{net_id}_"
        for dispatch_name, dispatch_info in dispatches.items():
            dispatch_id = dispatch_info.get("id", None)
            prefixed_name = f"{net_prefix}{dispatch_name}"

            # Networks may declare a preferred profile_hw (e.g. dronet on
            # RVV, mlp_control on scalar) to keep each network resident on
            # one cluster. The runtime's heterogeneous binary has separate
            # per-backend intermediate buffers, so cross-kind dispatching
            # within a network reads/writes the wrong buffer copy and
            # produces incorrect outputs. Pinning at the scheduler level
            # avoids that by penalising non-preferred-hw combos so the
            # solver never picks them.
            preferred_hw = net_info.get("preferred_hw")
            # Scaled penalty: 1000x the dispatch's own preferred-hw cost,
            # capped at 100 ms. Big enough to dominate the optimizer (no
            # solver picks a wrong-kind combo when a valid one exists),
            # small enough not to blow up the LP's numeric range or the
            # horizon-estimate sums.
            PIN_PENALTY_MULT = 1000.0
            PIN_PENALTY_CAP_MS = 100.0

            combo_times = []
            preferred_t = None  # cache the preferred-hw time to scale the penalty
            for ci, combo in enumerate(machine_combinations):
                hw = combo_hw[ci]
                topo = _resolve_topo_for(hw, combo, topo_tag_override)
                prof = all_profiles.get((hw, topo))

                t_ms = None
                if prof and isinstance(dispatch_id, int) and dispatch_id in prof:
                    t_ms = prof[dispatch_id]["time_ms"]

                # ── Tile-dispatch fallback ────────────────────────────────
                # When apply_split_hint rewrites the IR to split an op into
                # N tiles, the new dispatches are named `<orig>.tile_<i>` and
                # have fresh dispatch_ids that the profile DB doesn't know
                # about. Without this fallback they default to 0 cycles,
                # which makes the scheduler think the work has vanished
                # (bookkeeping fiction).
                #
                # For ASYMMETRIC splits (tile_oc_fraction in split_from),
                # the tile cost = parent cost * fraction. This lets the
                # scheduler model proportional splits where one tile takes
                # most of the work (going to the fast accelerator) and the
                # other takes a small fraction (going to the slow one) so
                # both finish at the same wall-clock — the genuine
                # parallelism win.
                #
                # The fallback: look at split_from metadata on the dispatch
                # to find the parent op_id and n_splits, then use parent
                # cycles / n_splits as the tile's per-combo time. This is
                # the "linear scaling along split axis" assumption — exact
                # for matrix-mul-like work and a good first approximation
                # for conv2d. The decision-loop's measure_candidate.sh path
                # can later override this with the actual measured per-tile
                # cycles (round 5 of artifacts/decision_loop/ confirmed
                # measured tile_0 ≈ tile_1 ≈ orig/2 for linear_s8 N-splits).
                if t_ms is None and prof and ".tile_" in dispatch_name:
                    split_from = dispatch_info.get("split_from") or {}
                    parent_id = split_from.get("op_id") or split_from.get("orig")
                    n_splits = split_from.get("n_splits", 1)
                    # Asymmetric split: each tile carries its own fraction
                    # (sum across all tiles == 1.0). Symmetric splits don't
                    # set this and fall back to 1/n_splits.
                    tile_fraction = split_from.get("tile_oc_fraction") or \
                                    split_from.get("tile_fraction") or \
                                    (1.0 / float(n_splits) if n_splits >= 1 else 1.0)
                    if isinstance(parent_id, int) and parent_id in prof and n_splits >= 1:
                        t_ms = float(prof[parent_id]["time_ms"]) * float(tile_fraction)
                    elif isinstance(parent_id, str):
                        # parent_id encoded as the original dispatch name
                        # ("dispatch_177"). The split rewrite REMOVED the
                        # parent from the graph, so we can't look it up
                        # via dispatches[parent_id]. Parse the integer
                        # suffix directly — the dispatch_graph.json convention
                        # is `dispatch_<int>` for all unsplit ops.
                        cand_id: int | None = None
                        if parent_id.startswith("dispatch_"):
                            tail = parent_id[len("dispatch_"):]
                            if tail.isdigit():
                                cand_id = int(tail)
                        if cand_id is None:
                            # Last-chance fallback: scan the graph for any
                            # dispatch whose name matches (handles
                            # non-standard naming schemes).
                            for cand_name, cand_info in dispatches.items():
                                if cand_name == parent_id:
                                    maybe_id = cand_info.get("id")
                                    if isinstance(maybe_id, int):
                                        cand_id = maybe_id
                                    break
                        if isinstance(cand_id, int) and cand_id in prof:
                            t_ms = float(prof[cand_id]["time_ms"]) * float(tile_fraction)

                if t_ms is not None:
                    base_t = float(t_ms)
                else:
                    if strict:
                        # Per-dispatch misses are typically zero-cost
                        # IR ops (view, chunk2_c1) — the codegen filters
                        # those out of the kernel dispatch table, so the
                        # profile CSV never has an entry for them.
                        # Treat as cost=0; the runtime walker also skips
                        # them (dispatch_id < 0 short-circuit, see
                        # generate_xpurt_main.py). Whole-profile misses
                        # (no CSV at all for a (network, hw, topo))
                        # are still fatal — that's the synthetic-random
                        # failure mode this strict mode is here to
                        # catch.
                        base_t = 0.0
                    else:
                        p_ms_synth = float(rng.uniform(2.0, 10.0))
                        core_type = machine_type_prefix(combo[0])
                        if core_type == "CPU_P":
                            base_t = p_ms_synth
                        else:
                            base_t = p_ms_synth * p_core_speedup
                if preferred_hw is not None and hw == preferred_hw and preferred_t is None:
                    preferred_t = base_t
                combo_times.append(base_t)

            if preferred_hw is not None:
                penalty = min(PIN_PENALTY_CAP_MS,
                              PIN_PENALTY_MULT * (preferred_t or 1.0))
                for ci, combo in enumerate(machine_combinations):
                    if combo_hw[ci] != preferred_hw:
                        combo_times[ci] += penalty

            processing_times[prefixed_name] = combo_times

        net_bucket = profiled_by_network.setdefault(net_id, {"p": {}, "e": {}})
        for (hw, topo), prof in all_profiles.items():
            if hw == cpu_p_profile_hw:
                combined_profiled_p.update(prof)
                net_bucket["p"].update(prof)
            else:
                combined_profiled_e.update(prof)
                net_bucket["e"].update(prof)

    if strict and missing:
        raise FileNotFoundError(
            "profile_loader: required profile data is missing. "
            "Schedules generated against synthetic random times produce "
            "fictional predicted timelines (this used to be silent — see "
            "the rng.uniform(2.0, 10.0) fallback). Either:\n"
            "  1. Run the missing profile sweeps (runtime/scripts/profile_remote.sh\n"
            "     or profile_dispatches.py — make sure the harness flags match\n"
            "     the consumer of the schedule, e.g. xpurt_demo's BACKENDS /\n"
            "     prj.conf overlay), OR\n"
            "  2. Pass strict=False to load_profiled_processing_times to opt\n"
            "     into the synthetic fallback explicitly.\n"
            f"Missing entries ({len(missing)}):\n" + "\n".join(missing)
        )

    return processing_times, combined_profiled_p, combined_profiled_e, profiled_by_network
