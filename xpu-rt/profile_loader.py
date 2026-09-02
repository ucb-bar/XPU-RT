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


# The profile sweeps write 1e9 us (= 1e6 ms) for a dispatch that cannot run on
# that backend at all.  Those rows mark "unsupported", they are not timings,
# and a sentinel that reaches the solver as a duration becomes fiction: one
# unsupported dispatch adds 1e6 ms to the makespan, and run_xpurt_schedule.py
# then sizes every periodic net against that (ceil(makespan / period)), which
# is how a 3-network workload turned into 450k operations.  Sentinels never
# become durations here -- see _penalise_unsupported below.
SENTINEL_MS = 1e6

# The "never pick this combination" idiom: 1000x the dispatch's own best real
# cost, capped at 100 ms.  Big enough to dominate the optimizer (no solver
# picks a penalised combo when a valid one exists), small enough not to blow
# up the LP's numeric range or the horizon-estimate sums.  Used both for
# unsupported-backend combos and for preferred_hw pinning.
COMBO_PENALTY_MULT = 1000.0
COMBO_PENALTY_CAP_MS = 100.0


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


#: Fraction of a conv/linear kernel's cost that does NOT shrink when the op is
#: tiled along OC/N.  MEASURED on AWS F2 (f2_dual_small_norose_tacit_q31_60mhz,
#: dronet conv_modules.0, N1 IC3 112x112 -> OC32 56x56 K3 S2), by running each
#: tile alone with the other hart idle and fitting t(OC) = a + b*OC:
#:
#:     gemmini_q31 (gemmini_tiled_conv)      417 + 24.7*OC us   -> a/t(32) = 0.35
#:     rvv         (rvv_vsmul_vnclip)       1049 + 14.0*OC us   -> a/t(32) = 0.70
#:
#: Both fits hold to <=4% over OC in {2,4,8,16,24,28,32}.  The fixed term is
#: real work, not overhead: the gemmini kernel transposes the WHOLE input
#: NCHW->NHWC before it looks at OC at all, and the rvv kernel's
#: (oh, ow, ic, kh, kw) trip count is OC-independent because OC is its VECTOR
#: LANE dimension.  On the other nine dronet convs the fixed fraction measured
#: ~1.0 on gemmini: conv_modules.5 costs 69 us at OC=64 and 69 us at OC=62.
#:
#: 1.0 is the DEFAULT on purpose.  The error is asymmetric: over-costing a tile
#: only means the scheduler declines to split (which was the right answer for
#: every dronet layer), while under-costing it makes the scheduler choose a
#: split that loses -- with parent/N it chose an even 16/16 split of
#: conv_modules.0 that measured 2809 us against 2747 us unsplit.  Set
#: XPURT_TILE_FIXED_FRACTION only when the per-(backend, op) fixed fraction has
#: actually been measured, or profile the SPLIT graph and let the tiles carry
#: their own rows.
_DEFAULT_TILE_FIXED_FRACTION = 1.0


def _tile_fixed_fraction() -> float:
    raw = os.environ.get("XPURT_TILE_FIXED_FRACTION")
    if raw is None:
        return _DEFAULT_TILE_FIXED_FRACTION
    try:
        f = float(raw)
    except ValueError:
        raise SystemExit(f"XPURT_TILE_FIXED_FRACTION={raw!r} is not a number")
    if not 0.0 <= f <= 1.0:
        raise SystemExit(
            f"XPURT_TILE_FIXED_FRACTION={f} out of range; it is a FRACTION of "
            f"the parent cost that does not shrink with the tile, so it must "
            f"be in [0, 1].")
    return f


def _load_tile_widths(dispatch_deps_path: str) -> dict:
    """`{new_dispatch_id: tile_width}` for a split graph, else `{}`.

    The width is the tile's own extent along the split axis (`tile_oc` for a
    conv OC split, `tile_n` for a linear N split or a pointwise E split,
    `tile_oh` for a conv row split, `tile_c` for a pool channel split),
    recorded by
    `apply_split_hint`.  It is NOT recoverable as parent/N: `tile_sizes`
    partitions an axis into tiles that need not be equal, and an uneven
    partition is the interesting case precisely because the two backends do
    not have the same cost curve.
    """
    try:
        with open(dispatch_deps_path) as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for op in g.get("ops", []):
        sf = op.get("split_from") or {}
        did = op.get("dispatch_id")
        if not sf or did is None:
            continue
        w = next((sf[k] for k in ("tile_oc", "tile_n", "tile_oh", "tile_c")
                  if isinstance(sf.get(k), int)), None)
        if isinstance(w, int) and w > 0:
            out[int(did)] = w
    return out


def _load_tile_axes(dispatch_deps_path: str) -> dict:
    """`{new_dispatch_id: (axis, window_rows, padded_IH)}` for a split graph.

    The axis decides which cost model applies, and getting that wrong is not a
    small error: the slab quantum below is a property of the OUTPUT-CHANNEL
    blocking, so applying it to a tile whose width is a count of ROWS asks
    "how many 32-channel slabs is 7 rows", which is meaningless.  It also
    happens to answer 1, which reads as "this tile is free" -- an OH split of a
    14-row conv would be costed at 1.0x the parent for both tiles and look like
    a free 2x.  Hence an explicit axis rather than a `hw` lookup alone.

    `window_rows` is the padded input band the tile reads -- its output rows
    plus the kernel's halo -- which is what an OH tile's cost is proportional
    to; `padded_IH` is the parent's, so the two together give both the tile's
    share and the halo's duplication factor.
    """
    try:
        with open(dispatch_deps_path) as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for op in g.get("ops", []):
        sf = op.get("split_from") or {}
        did = op.get("dispatch_id")
        if not sf or did is None:
            continue
        axis = sf.get("axis") or ("OC" if "tile_oc" in sf else
                                  "N" if "tile_n" in sf else None)
        if axis is None:
            continue
        wr = sf.get("window_rows")
        pih = None
        if sf.get("parent_IH") is not None and sf.get("parent_PH") is not None:
            pih = int(sf["parent_IH"]) + 2 * int(sf["parent_PH"])
        # Halo is KH > 1, read off the tile's own shape. Do NOT infer it from
        # window_rows > tile_oh: a 1x1 conv at stride 2 spans 2*tile_oh-1 input
        # rows and still has no halo, because it reads one row per output row
        # and simply skips the rest. Getting that backwards would tax the one
        # case (KH == 1) where the gather is pure waste at the cheaper rate.
        kh = int((op.get("shape") or {}).get("KH", 0) or 0)
        out[int(did)] = (axis, wr, pih, kh > 1)
    return out


def _load_id_remap(dispatch_deps_path: str) -> dict:
    """`{old_dispatch_id: [new_dispatch_id, ...]}` for an IR-split model.

    `pipeline/apply_split_hint.py` turns one dispatch into N tiles and
    RENUMBERS every dispatch after it, emitting the mapping as `id_remap`.
    Profiles are keyed by dispatch_id and were measured on the UNSPLIT
    graph, so indexing them with split ids silently hands every dispatch
    after the split point its neighbour's cost. Observed on a 2-way split
    of dronet conv2d_s8[0] (FPGA job 374): conv_modules.0.tile_1 was
    scheduled with maxpool's 0.218024 ms, maxpool1 with batchnorm's
    0.031082 ms, and so on down the graph. The model still computed the
    right answer -- data deps come from the model graph, not the schedule
    -- but every placement and duration after the split was wrong, so the
    two tiles serialised instead of running concurrently.

    Returns {} when the graph carries no remap, i.e. every unsplit model,
    so this is a no-op on the existing flow.
    """
    try:
        with open(dispatch_deps_path) as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    raw = g.get("id_remap") or {}
    out = {}
    for old, new in raw.items():
        try:
            out[int(old)] = [int(n) for n in new]
        except (TypeError, ValueError):
            continue
    return out


# Blocking quantum per backend: the width below which a tile costs the same as
# a full one, so splitting under it buys nothing and pays for the extra tile.
#
#   rvv  32 -- TILE_OC is clamped to vsetvlmax_e32m4() = VLEN(256)*LMUL(4)/SEW(32)
#             in kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c:303-309.
#   gemmini 16 -- the systolic array DIM (cores/gemmini/include/gemmini_params.h:8).
#
# MEASURED (FPGA, quad hetero, 1000 MHz target): on rvv the slab model
#     ratio = sum_i ceil(w_i/Q) / ceil(W/Q)
# predicted all 35 sweep cells to a MEDIAN ABSOLUTE ERROR OF 0.008 -- an OC=64
# conv split 32/32 measured 1.00, split 16/48 measured 1.49 against 1.50
# predicted. On gemmini the SAME form fails (median error 0.55, max 1.33)
# because its cost is dominated by a per-tile input NCHW->NHWC transpose that no
# slab count captures, so gemmini keeps the conservative affine fallback below.
#: What an OH (output-row) split actually costs, as a multiple of the halo-only
#: prediction `parent * sum(window_rows)/padded_IH`.  An OH tile is not free of
#: the parent's work the way the halo model alone suggests: the codegen gathers
#: the tile's padded input band and scatters its output rows back, because in
#: NCHW neither end of a row band is contiguous.  Measured on AWS F2 (dronet,
#: 20 cells, experiments/shard_dim/results/ohvsoc_cells.csv) the tax is bimodal:
#:
#:   KH > 1  rvv     0.47-1.22  median 0.80   the tile's working set shrinks and
#:                                            the cache win outpays the copies
#:   KH > 1  gemmini 1.48-2.27  median 2.16   gemmini is ~10x faster per conv, so
#:                                            the same bytes have nothing to hide
#:                                            behind
#:   KH == 1 either  1.76-2.32  median ~2.0   no halo, so the gather copies the
#:                                            whole input to buy nothing
#:
#: The spread inside a cell is wide, so this is a model for CHOOSING an axis,
#: not for predicting a latency: re-profile the split graph when the number
#: itself matters.  Set XPURT_OH_COPY_TAX to override with a measured value.
_OH_COPY_TAX = {
    ("rvv", True): 0.80, ("rvv", False): 1.77,
    # gemmini re-fit after the zero-copy kernel landed (fq 470/471): its OH
    # wrapper no longer gathers or scatters, so the tax fell from 2.16/2.10 to
    # these. That flipped the axis choice on 6 of 10 gemmini convs.
    ("gemmini", True): 1.22, ("gemmini", False): 1.2,
}
_OH_COPY_TAX_DEFAULT = 2.0


def _oh_copy_tax(hw: str | None, has_halo: bool) -> float:
    env = os.environ.get("XPURT_OH_COPY_TAX")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    fam = "rvv" if hw and "rvv" in hw else ("gemmini" if hw and "gemmini" in hw else None)
    return _OH_COPY_TAX.get((fam, has_halo), _OH_COPY_TAX_DEFAULT)


_BACKEND_SPLIT_QUANTUM = {
    "V256D128_rvv": 32,
    "rvv": 32,
    "rvv_f16": 32,
}


def _quantum_for(hw: str | None) -> int | None:
    """The blocking quantum for `hw`, or None when no validated model exists.

    Returning None is meaningful: it selects the conservative affine model
    rather than guessing a quantum. A wrong quantum would UNDER-cost a split
    and make the scheduler choose it, which is the failure direction that
    actually costs wall-clock.
    """
    if not hw:
        return None
    return _BACKEND_SPLIT_QUANTUM.get(hw)


def _apply_id_remap(prof: dict, remap: dict, widths: dict | None = None,
                    fixed_fraction: float | None = None,
                    hw: str | None = None, axes: dict | None = None) -> dict:
    """Re-key a profile measured on the unsplit graph onto split ids.

    Splitting a dispatch renumbers every id after it, so a profile keyed by
    dispatch_id has to be translated before it can be joined against a split
    graph.  That part is bookkeeping.  The interesting part is what a TILE
    should be costed at, and the answer this used to give -- parent/N -- is
    wrong in two separate ways:

      1. **It assumes the tiles are equal.**  `apply_split_hint` accepts an
         explicit `tile_sizes` partition, and an uneven one is the useful case:
         the two backends do not share a cost curve, so the split that balances
         them is not the even one.  Width now comes from the tile's own
         `split_from.tile_oc` (see `_load_tile_widths`).

      2. **It assumes cost is PROPORTIONAL to width** -- i.e. that a tile with
         half the channels does half the work.  It does not.  Measured on AWS
         F2 for dronet conv_modules.0 (see `_DEFAULT_TILE_FIXED_FRACTION` for
         the fits), an OC=4-of-32 tile costs 515 us on gemmini_q31 where
         parent/8 predicts 148 (-71%), and 1107 us on rvv where parent/8
         predicts 187 (-83%).  Every conv kernel here has a large
         OC-INDEPENDENT term, so a tile costs

             t(w) = T_parent * (f + (1 - f) * w / W_parent)

         with `f` the fraction that does not shrink.  That form fits the
         measurements to <=4%; proportional-to-width is off by up to 91%.

    Under-costing tiles is what made the scheduler pick a 16/16 split of
    conv_modules.0 that MEASURED 2809 us against 2747 us unsplit.  So `f`
    defaults to 1.0 -- a tile is costed at its parent's full cost -- and the
    scheduler only splits when someone has supplied a measured fixed fraction
    (XPURT_TILE_FIXED_FRACTION) or profiled the split graph so the tiles carry
    their own rows.  This is still an ESTIMATE either way; re-profile the split
    graph when tile costs matter.
    """
    if not remap:
        return prof
    widths = widths or {}
    f = _tile_fixed_fraction() if fixed_fraction is None else fixed_fraction
    out = {}
    for old, entry in prof.items():
        tiles = remap.get(old)
        if not tiles:
            continue
        n = len(tiles)
        parent_ms = entry.get("time_ms", 0.0)
        # Parent extent along the split axis = the tiles' widths summed. Falls
        # back to "every tile is 1/n" only when the graph carries no widths at
        # all (a graph produced before apply_split_hint recorded them).
        tile_w = [widths.get(t) for t in tiles]
        total_w = sum(w for w in tile_w if w) if all(tile_w) else 0
        for k, new_id in enumerate(tiles):
            e = dict(entry)
            if n > 1:
                axis, wr, pih, halo = (axes or {}).get(
                    new_id, (None, None, None, False))
                q = _quantum_for(hw)
                if axis == "OH" and wr and pih:
                    # Row split. Deliberately NOT the slab model: the quantum is
                    # a property of output-CHANNEL blocking, and OH's whole
                    # advantage is that rows carry no such quantum -- which is
                    # why an OC=32 conv that cannot be split on rvv at all can
                    # still be split 4 ways along OH. Cost is proportional to
                    # the padded input band the tile reads (halo included, so a
                    # narrow tile pays for its overlap), times the measured tax
                    # for the gather/scatter the NCHW layout forces.
                    e["time_ms"] = parent_ms * (wr / pih) * _oh_copy_tax(
                        hw, has_halo=halo)
                elif axis in ("E", "C") and total_w and all(tile_w):
                    # Pointwise element range, or a pool's channel range.
                    # LINEAR in the tile's width, with no slab quantum: neither
                    # kind blocks its inner loop on an output-channel tile the
                    # way a conv does, so `ceil(w/q)/ceil(W/q)` would be
                    # answering a question about hardware that is not in the
                    # loop. It is not a harmless approximation either -- for
                    # DroNet's maxpool2d_s8, C=32 is exactly the rvv quantum,
                    # so the slab model prices both halves of a 2-way split at
                    # a full slab and reports a 2x split as worth 1.0x.
                    e["time_ms"] = parent_ms * (tile_w[k] / total_w)
                elif q and total_w and all(tile_w):
                    # Slab model -- validated on rvv to 0.008 median error.
                    # A tile costs ceil(w/q) slabs; the parent costs
                    # ceil(W/q). Sub-quantum tiles therefore cost a FULL
                    # slab, which is exactly why an OC=32 conv cannot be
                    # 2-way split on rvv without doubling the work.
                    par_slabs = -(-total_w // q)
                    e["time_ms"] = parent_ms * (
                        (-(-tile_w[k] // q)) / par_slabs) if par_slabs else parent_ms
                else:
                    frac = (tile_w[k] / total_w) if total_w else (1.0 / n)
                    e["time_ms"] = parent_ms * (f + (1.0 - f) * frac)
                e["module_name"] = f"{entry.get('module_name', '')}.tile_{k}"
            out[new_id] = e
    return out


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
                prof = _apply_id_remap(
                    prof, _load_id_remap(dispatch_deps_path),
                    _load_tile_widths(dispatch_deps_path), hw=hw,
                    axes=_load_tile_axes(dispatch_deps_path))
                if prof:
                    profiles[(hw, topo)] = prof
                    if model_candidate != net_id:
                        print(f"  (info) profile fallback: {net_id}/{hw}/{topo} -> model={model_candidate}")
                break
    return profiles


def _synthetic_time(rng: np.random.Generator, combo: list[str],
                    p_core_speedup: float) -> float:
    """A made-up per-dispatch cost, for `strict=False` callers only."""
    p_ms_synth = float(rng.uniform(2.0, 10.0))
    if machine_type_prefix(combo[0]) == "CPU_P":
        return p_ms_synth
    return p_ms_synth * p_core_speedup


def _penalise_unsupported(combo_times: list) -> None:
    """
    Replace the `None` slots (unsupported sentinels) in `combo_times` with a
    cost the solver will never choose, in place.

    An operation has to be assigned *somewhere*, so an unsupported
    combination needs a number. It must be clearly worse than every real
    option and still bounded: the raw 1e6 ms sentinel is bounded too, but it
    is large enough to dominate any real makespan, which is what produced
    12-hour schedules out of sub-second networks. Scaling off the dispatch's
    own best real cost keeps the penalty in the same numeric range as the
    rest of the model.
    """
    real = [v for v in combo_times if v is not None]
    if not real:
        return
    best_real = min(real)
    penalty = best_real + min(COMBO_PENALTY_CAP_MS,
                              COMBO_PENALTY_MULT * (best_real or 1.0))
    for ci, v in enumerate(combo_times):
        if v is None:
            combo_times[ci] = penalty


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
    # Dispatches no backend in this hardware config can run at all.
    unrunnable: list[str] = []

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

            # `None` marks a combination whose profile row is an unsupported
            # sentinel. Those slots are filled in below, once the dispatch's
            # best *real* cost is known — a sentinel is not a timing, so it
            # must never be summed into a duration.
            combo_times: list[float | None] = []
            for ci, combo in enumerate(machine_combinations):
                hw = combo_hw[ci]
                topo = _resolve_topo_for(hw, combo, topo_tag_override)
                prof = all_profiles.get((hw, topo))

                t_ms = None
                if prof and isinstance(dispatch_id, int) and dispatch_id in prof:
                    t_ms = prof[dispatch_id]["time_ms"]

                if t_ms is not None and float(t_ms) >= SENTINEL_MS:
                    combo_times.append(None)
                    continue

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
                        base_t = _synthetic_time(rng, combo, p_core_speedup)
                combo_times.append(base_t)

            if all(v is None for v in combo_times):
                # No combination in this hardware config can run this
                # dispatch. Any number we invent is fiction — the old code
                # passed the 1e6 ms sentinel straight through and the
                # schedule inherited it — so record it and fail below.
                unrunnable.append(
                    f"  - {net_id}/{dispatch_name} (dispatch_id={dispatch_id}): "
                    f"unsupported on every profiled backend "
                    f"({', '.join(sorted(set(combo_hw)))})"
                )
                if strict:
                    continue
                combo_times = [
                    _synthetic_time(rng, combo, p_core_speedup)
                    for combo in machine_combinations
                ]
            else:
                _penalise_unsupported(combo_times)

            if preferred_hw is not None:
                preferred_t = next(
                    (combo_times[ci] for ci in range(len(machine_combinations))
                     if combo_hw[ci] == preferred_hw),
                    None,
                )
                penalty = min(COMBO_PENALTY_CAP_MS,
                              COMBO_PENALTY_MULT * (preferred_t or 1.0))
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

    if strict and unrunnable:
        raise ValueError(
            "profile_loader: some dispatches are unsupported on every backend "
            "in this hardware config, so no schedule can run them. The profile "
            "CSVs mark them with the 1e9 us sentinel; costing them as if they "
            "were timings puts ~1e6 ms per dispatch into the makespan and "
            "produces a fictional schedule. Either:\n"
            "  1. Add a backend that supports them to the workload's\n"
            "     hardware.profile_hw (e.g. CPU alongside HTA/DSP), OR\n"
            "  2. Re-bundle or re-compile the network so every dispatch has a\n"
            "     backend that can run it, OR\n"
            "  3. Drop the network from this hardware config.\n"
            f"Unrunnable dispatches ({len(unrunnable)}):\n" + "\n".join(unrunnable)
        )
    if unrunnable and not strict:
        # Non-strict callers opted into synthetic timings, but say so —
        # these dispatches have no measured cost on any backend here.
        print(f"  (warning) {len(unrunnable)} dispatch(es) unsupported on every "
              f"profiled backend; costed with synthetic times (strict=False)")

    return processing_times, combined_profiled_p, combined_profiled_e, profiled_by_network
