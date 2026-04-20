"""Token-efficient Recipe IR views for LLM consumption.

A 65-op recipe serialised verbatim easily blows past a reasonable
prompt budget. :func:`recipe_to_llm_view` produces a compact JSON-ish
summary: per-family op counts, the first N "banner" ops verbatim,
a terse middle section, and every op addressable by a stable
``op_id`` hash so the LLM can reference them without re-sending the
full body.

:func:`diff_views` compares two views and returns only what changed —
added, removed, and mutated op-ids — which is what the LLM sees after
each successful transform.

Both views are deterministic (stable op-id hashing, sorted family
keys, no timestamps) so prompts are cacheable.
"""

from __future__ import annotations

import hashlib
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.ir.recipe.serialize import _op_to_dict

# Rough mapping from op-name prefix to a coarse "family" used for the
# per-phase counts. Intentionally small; anything uncategorised lands
# in ``other``.
_FAMILY_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("recipe.propose_", "propose"),
    ("recipe.require_", "verify"),
    ("recipe.select_", "candidate"),
    ("recipe.fuse", "candidate"),
    ("recipe.tile", "candidate"),
    ("recipe.vectorize", "candidate"),
    ("recipe.place_on_device", "candidate"),
    ("recipe.request_triton_kernel", "candidate"),
    ("recipe.request_exo_kernel", "candidate"),
    ("recipe.materialize_ukernel", "candidate"),
    ("recipe.lower_to_accel", "candidate"),
    ("recipe.layout_normalize", "candidate"),
    ("recipe.insert_copy_boundary", "candidate"),
    ("recipe.reassociate", "candidate"),
    ("recipe.blackbox", "candidate"),
    ("recipe.segment_boundary", "scope"),
    ("recipe.region", "scope"),
    ("recipe.segment", "scope"),
    ("recipe.anchor", "scope"),
    ("recipe.guard", "scope"),
    ("recipe.bind_payload", "scope"),
    ("recipe.alternatives", "choice"),
    ("recipe.defer_choice", "choice"),
    ("recipe.promote_candidate", "choice"),
    ("recipe.rank", "choice"),
    ("recipe.require_eqsat", "choice"),
    ("recipe.require_solver", "choice"),
    ("recipe.search_budget", "choice"),
    ("recipe.from_", "provenance"),
    ("recipe.lineage", "provenance"),
    ("recipe.promote", "provenance"),
    ("recipe.reject", "provenance"),
    ("recipe.feedback", "provenance"),
    ("recipe.backend_", "fact"),
    ("recipe.kernel_contract", "fact"),
    ("recipe.transfer_cost", "fact"),
    ("recipe.local_mem_fit", "fact"),
    ("recipe.fusible_with", "fact"),
    ("recipe.calibration", "fact"),
    ("recipe.export_issue", "fact"),
    ("recipe.graph_break", "fact"),
    ("recipe.unsupported_operator", "fact"),
    ("recipe.guard_failure", "fact"),
    ("recipe.quantization_intent", "fact"),
    ("recipe.tile_divisible", "fact"),
    ("recipe.contiguous_layout", "fact"),
)


def _family_of(op_name: str) -> str:
    for prefix, family in _FAMILY_BY_PREFIX:
        if op_name.startswith(prefix):
            return family
    return "other"


def _op_id(op_dict: dict[str, Any], index: int) -> str:
    """Stable short hash identifying an op within a view.

    Derived from the op's ``_op`` name, its properties, and its
    positional index so callers can deterministically point at one
    op even when the module has many copies of the same shape.
    """
    payload = str(index) + "|" + str(sorted(op_dict.items()))
    return "op_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


def _compact_entry(op_dict: dict[str, Any], op_id: str) -> dict[str, Any]:
    """Shrink an op dict for the middle-section while keeping the
    addressable identifiers an agent needs to pass on the next turn.

    Always surfaces ``sym_name`` and ``payload_region_id`` (the recipe-
    level + payload-level region names respectively) so an agent looking
    at ``view_recipe.middle`` can read region symbols directly without
    paging up to ``banner`` or asking ``get_dossier``.
    """
    entry: dict[str, Any] = {"op_id": op_id, "_op": op_dict["_op"]}
    # Recipe-level + payload-level region names — the two forms an
    # agent's propose_invent_slot.grouped_regions can take.
    for key in ("sym_name", "payload_region_id", "region_ref"):
        if key in op_dict and op_dict[key]:
            entry[key] = op_dict[key]
    # One additional hint key for non-region ops.
    for hint_key in ("region", "op_target", "kernel_id", "target", "name"):
        if hint_key in op_dict and hint_key not in entry:
            entry[hint_key] = op_dict[hint_key]
            break
    return entry


def recipe_to_llm_view(
    module: ModuleOp,
    *,
    max_ops: int = 80,
    focus: str | None = None,
    banner_size: int | None = None,
) -> dict[str, Any]:
    """Produce a compact, deterministic view of a Recipe IR module.

    Layout of the returned dict::

        {
          "hash": "sha256:...",
          "counts": {"propose": 7, "candidate": 12, ...},
          "total_ops": 65,
          "banner": [ {op_id, _op, ...props}, ...first K verbatim ],
          "middle": [ {op_id, _op, region?} , ...compacted rows ],
          "focused": { op_id: {full op dict} }   # present when `focus` set
        }

    Args:
        module: Recipe IR ``ModuleOp``.
        max_ops: Hard cap on the total number of rows included. When
            the module has more ops than this, the middle section is
            truncated with a ``"_truncated": N`` marker.
        focus: Optional ``op_id`` to inline verbatim alongside its
            neighbours. Used when the LLM asks to zoom into one op.
        banner_size: How many leading ops to include verbatim. Default
            is ``min(8, max_ops // 4)``.

    The view is strictly data-only — no wall-clock times, no object
    identities — so it hashes identically across runs.
    """
    if banner_size is None:
        banner_size = min(8, max(1, max_ops // 4))

    # Walk module body ops in order.
    op_dicts: list[dict[str, Any]] = []
    for op in module.body.block.ops:
        op_dicts.append(_op_to_dict(op))

    op_ids = [_op_id(d, i) for i, d in enumerate(op_dicts)]

    # Per-family counts.
    counts: dict[str, int] = {}
    for d in op_dicts:
        fam = _family_of(d["_op"])
        counts[fam] = counts.get(fam, 0) + 1

    total_ops = len(op_dicts)
    banner: list[dict[str, Any]] = []
    middle: list[dict[str, Any]] = []
    focused: dict[str, dict[str, Any]] = {}

    banner_slice = op_dicts[:banner_size]
    rest = op_dicts[banner_size:]
    banner_ids = op_ids[:banner_size]
    rest_ids = op_ids[banner_size:]

    for d, oid in zip(banner_slice, banner_ids):
        banner.append({"op_id": oid, **d})

    # Middle section: compact rows, truncated to fit max_ops.
    remaining_budget = max(0, max_ops - len(banner))
    if len(rest) > remaining_budget:
        kept = rest[:remaining_budget]
        kept_ids = rest_ids[:remaining_budget]
        truncated = len(rest) - remaining_budget
    else:
        kept = rest
        kept_ids = rest_ids
        truncated = 0

    for d, oid in zip(kept, kept_ids):
        middle.append(_compact_entry(d, oid))

    if truncated:
        middle.append({"_truncated": truncated})

    # Focus handling: inline the named op verbatim plus its two
    # neighbours (if any).
    if focus is not None:
        for i, oid in enumerate(op_ids):
            if oid == focus:
                lo = max(0, i - 1)
                hi = min(len(op_dicts), i + 2)
                for j in range(lo, hi):
                    focused[op_ids[j]] = op_dicts[j]
                break

    # Module-level hash: stable over op order + content.
    mod_hash = (
        "sha256:"
        + hashlib.sha256(("|".join(f"{d['_op']}:{sorted(d.items())}" for d in op_dicts)).encode()).hexdigest()[:16]
    )

    view: dict[str, Any] = {
        "hash": mod_hash,
        "counts": dict(sorted(counts.items())),
        "total_ops": total_ops,
        "banner": banner,
        "middle": middle,
    }
    if focused:
        view["focused"] = focused
    return view


def diff_views(view_a: dict[str, Any], view_b: dict[str, Any]) -> dict[str, Any]:
    """Return what changed between two views.

    Args:
        view_a: The earlier view (from :func:`recipe_to_llm_view`).
        view_b: The later view.

    Returns a dict with three keys::

        {
          "hash_before": ..., "hash_after": ...,
          "added":    [{op_id, _op, ...}],
          "removed":  [{op_id, _op, ...}],
          "unchanged_count": int,
        }

    Ops are identified by their ``op_id``. This is not a semantic diff
    — two ops that are structurally identical but positionally
    different will show up as removed+added.
    """
    ids_a = {entry["op_id"]: entry for entry in _all_entries(view_a)}
    ids_b = {entry["op_id"]: entry for entry in _all_entries(view_b)}

    added_ids = sorted(ids_b.keys() - ids_a.keys())
    removed_ids = sorted(ids_a.keys() - ids_b.keys())

    return {
        "hash_before": view_a.get("hash"),
        "hash_after": view_b.get("hash"),
        "added": [ids_b[i] for i in added_ids],
        "removed": [ids_a[i] for i in removed_ids],
        "unchanged_count": len(ids_a.keys() & ids_b.keys()),
    }


def _all_entries(view: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten banner + middle into a single list of op-id-bearing rows."""
    entries: list[dict[str, Any]] = []
    for section in ("banner", "middle"):
        for entry in view.get(section, []):
            if "op_id" in entry:
                entries.append(entry)
    return entries


def estimate_tokens(view: dict[str, Any]) -> int:
    """Rough token estimate for budget checks in tests.

    Uses a simple 4-chars-per-token heuristic over a JSON serialisation.
    Cheap and good enough for asserting that a view stays under N tokens.
    """
    import json

    serialised = json.dumps(view, default=str, sort_keys=True)
    return (len(serialised) + 3) // 4


__all__ = [
    "diff_views",
    "estimate_tokens",
    "recipe_to_llm_view",
]
