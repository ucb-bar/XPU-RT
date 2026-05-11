"""Action Space (Milestone 04) — IR-backed decision sites + candidates.

Turns Region Dossier V2 into an enumerated, IR-grounded action space
that the LLM Strategist/Tactician will later choose from. This stage
does **not** select candidates and does **not** apply Recipe edits.
It only enumerates legal/illegal moves deterministically.

Outputs:

- ``02_graph_analysis/action_space.mlir`` — canonical, IR-flavored;
  every emitted op is the source of truth.
- ``02_graph_analysis/decision_sites.json`` — projection: where can the
  agent act?
- ``02_graph_analysis/candidate_actions.json`` — projection: what exact
  actions can the agent pick?
- ``02_graph_analysis/llm_action_space.json`` — projection: the compact
  LLM-facing view; illegal candidates hidden by default.
- ``02_graph_analysis/action_space_validation.json`` — cross-check report.

All four JSON files include ``source.action_space_ir_sha256`` so a
downstream auditor can pin the exact IR state they were derived from.

Candidate families (per the M-04 spec):

1. **extension_closure** for ``source_classification == opaque_fallback``
2. **set_tile_params** for matmul-like regions, drawn from the dossier's
   ``working_set_curve`` (no invented tile sizes).
3. **set_accumulator_fp16 / quantize_fp8 / enable_fast_math** for
   structured compute-heavy regions; legality decided by the
   ``numerical_sensitivity`` audit (M-03.5 made this trustworthy).
4. **fuse_producer_consumer** for transient single-consumer edges in
   ``tensor_use_def_graph`` where neither end is opaque.
5. **assign_device** for every region (single-device baseline), so the
   schema is ready for multi-device targets without a forced edit.

Read-only against compiler core. No FXImporter / capture / pipeline
edits.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.graph_compilation.region_dossier import (
    TargetProfile,
    load_target_profile,
)

# --------------------------------------------------------------------------- #
# ID helpers
# --------------------------------------------------------------------------- #

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe(s: str) -> str:
    out = _SAFE_ID_RE.sub("_", s).strip("_")
    return out or "x"


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _site_id(kind: str, region_id: str, *, suffix: str = "") -> str:
    base = f"site_{kind}_{_safe(region_id)}__{_short_hash(region_id)}"
    return base + (f"_{suffix}" if suffix else "")


def _candidate_id(
    kind: str, region_id: str, label: str, *, region_extra: str = "",
) -> str:
    """Stable id for a candidate.

    M-37.9 (Fix 1): when ``region_extra`` is supplied, it is mixed
    into the hash so two regions named ``matmul_0`` in different
    models with different shapes produce distinct candidate_ids. Pass
    empty (the default) for backward compatibility with proposers
    that don't yet have a content-aware extra (their ids stay
    region_id-named).
    """
    hash_input = f"{region_id}/{label}"
    if region_extra:
        hash_input += f"@{region_extra}"
    return (
        f"cand_{kind}_{_safe(region_id)}_{_safe(label)}__{_short_hash(hash_input)}"
    )


def _region_content_hash(dossier: dict[str, Any]) -> str:
    """Shape-distinctive hash over a region's actual tensor shape.

    Two regions in different models that share a region_id but have
    different shapes (e.g. tiny_mlp matmul_0 with M=4,K=64,N=128 vs
    holdout_mlp_odd_shapes matmul_0 with M=7,K=63,N=129) hash
    differently because their ``region_shape`` projections differ.

    Falls back to a working_set_curve + legality + source hash for
    legacy dossiers (region_dossier_v1) that don't carry the
    ``region_shape`` field. Empty / missing dossier yields empty
    string.
    """
    if not dossier:
        return ""
    region_shape = dossier.get("region_shape")
    if region_shape:
        payload = json.dumps(
            region_shape, sort_keys=True, separators=(",", ":"),
        )
    else:
        payload = json.dumps(
            {
                "source": dossier.get("source", {}),
                "working_set_curve": dossier.get("working_set_curve", []),
                "legality_constraints": dossier.get("legality_constraints", []),
                "kind": dossier.get("kind", ""),
            },
            sort_keys=True, separators=(",", ":"),
        )
    return _short_hash(payload)


# --------------------------------------------------------------------------- #
# Result dataclass + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionSpaceResult:
    action_space_mlir_path: Path
    action_space_ir_sha256: str
    decision_sites_path: Path
    candidate_actions_path: Path
    llm_action_space_path: Path
    action_space_validation_path: Path
    site_count: int
    candidate_count_total: int
    candidate_count_legal: int


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


_OPAQUE_PREFIX = "opaque_"


def _is_opaque(kind: str) -> bool:
    return kind.startswith(_OPAQUE_PREFIX)


def _is_matmul_like(kind: str) -> bool:
    return kind in {"matmul", "conv"}


def _is_structured(kind: str) -> bool:
    return not _is_opaque(kind) and kind not in {"tensor_empty", "unknown"}


@dataclass
class _Cand:
    candidate_id: str
    site_id: str
    kind: str
    region_id: str
    label: str
    recipe_delta: list[dict[str, Any]]
    legality: dict[str, Any]
    cost_preview: dict[str, Any]
    evidence: dict[str, str]


@dataclass
class _Site:
    site_id: str
    kind: str
    region_id: str
    priority: int
    reason: str
    candidate_ids: list[str]
    extra: dict[str, Any]


# ---- Family 1: extension closure ---------------------------------------- #


def _gen_extension_closure(
    region: dict[str, Any],
    dossier: dict[str, Any],
    gap: dict[str, Any] | None,
    region_dossier_ref: str,
) -> tuple[_Site | None, list[_Cand]]:
    if dossier["source"]["source_classification"] != "opaque_fallback":
        return None, []
    rid = region["region_id"]
    site_id = _site_id("close_gap", rid)
    candidates: list[_Cand] = []
    has_signatures = bool(dossier["source"]["payload_ops"])
    has_evidence = (
        gap is not None
        and "reference_semantics" in gap.get("required_evidence", [])
        and bool(gap.get("extension_id"))
    )

    # Candidate 1: create_payload_lowering_extension
    cid = _candidate_id("ext", rid, "create_payload_lowering_extension")
    payload = {
        "op": "CreatePayloadLoweringExtension",
        "region": rid,
        "gap_id": (gap or {}).get("gap_id"),
        "extension_id": (gap or {}).get("extension_id"),
    }
    legal = bool(has_evidence)
    candidates.append(
        _Cand(
            candidate_id=cid,
            site_id=site_id,
            kind="create_payload_lowering_extension",
            region_id=rid,
            label="create_payload_lowering_extension",
            recipe_delta=[payload],
            legality=(
                {"ok": True}
                if legal
                else {
                    "ok": False,
                    "reason": (
                        "no matching gap with reference_semantics + extension_id"
                        if not gap
                        else "gap missing reference_semantics or extension_id"
                    ),
                }
            ),
            cost_preview={"static_relative_cost": 0.40, "numerics_ok": True},
            evidence={
                "region_dossier": region_dossier_ref,
                "gap_id": (gap or {}).get("gap_id", ""),
            },
        )
    )

    # Candidate 2: create_kernel_contract
    cid2 = _candidate_id("ext", rid, "create_kernel_contract")
    candidates.append(
        _Cand(
            candidate_id=cid2,
            site_id=site_id,
            kind="create_kernel_contract",
            region_id=rid,
            label="create_kernel_contract",
            recipe_delta=[
                {"op": "CreateKernelContract", "region": rid}
            ],
            legality=(
                {"ok": True}
                if has_signatures
                else {
                    "ok": False,
                    "reason": "no shape/dtype signatures available on opaque op",
                }
            ),
            cost_preview={"static_relative_cost": 0.50},
            evidence={"region_dossier": region_dossier_ref},
        )
    )

    # Candidate 3: keep_as_fallback (always legal, low priority)
    cid3 = _candidate_id("ext", rid, "keep_as_fallback")
    candidates.append(
        _Cand(
            candidate_id=cid3,
            site_id=site_id,
            kind="keep_as_fallback",
            region_id=rid,
            label="keep_as_fallback",
            recipe_delta=[{"op": "KeepAsFallback", "region": rid}],
            legality={"ok": True},
            cost_preview={"static_relative_cost": 1.00, "numerics_ok": True},
            evidence={"region_dossier": region_dossier_ref},
        )
    )

    site = _Site(
        site_id=site_id,
        kind="extension_closure",
        region_id=rid,
        priority=1 if gap else 3,
        reason=(
            "opaque fallback with matching gap requiring extension or fallback decision"
            if gap
            else "opaque fallback (no matching gap record yet)"
        ),
        candidate_ids=[c.candidate_id for c in candidates],
        extra={"gap_id": (gap or {}).get("gap_id", "")},
    )
    return site, candidates


# ---- Family 2: tiling --------------------------------------------------- #


def _gen_tiling(
    region: dict[str, Any],
    dossier: dict[str, Any],
    region_dossier_ref: str,
) -> tuple[_Site | None, list[_Cand]]:
    if not _is_matmul_like(region["kind"]):
        return None, []
    curve = dossier["working_set_curve"]
    if not curve:
        return None, []
    rid = region["region_id"]
    site_id = _site_id("tile", rid)

    # Baseline live_bytes for relative-cost normalization: max live_bytes seen.
    max_live = max(t["live_bytes"] for t in curve) or 1

    # M-37.9 Fix 3b: derive the region's actual M/N/K (when matmul) so the
    # cost preview can penalize tiles that would force boundary handling.
    # Greedy sorts by static_relative_cost; without this penalty it
    # always picks the smallest tile (cheapest cache footprint) even
    # when that tile doesn't divide cleanly and the differential check
    # will then fail. The penalty steers greedy toward clean-divide
    # tiles when one exists at the same legality level.
    region_dims: dict[str, int] = {}
    region_shape = (dossier or {}).get("region_shape") or {}
    if region_shape.get("kind") == "matmul":
        inp = region_shape.get("input_shapes") or []
        if (
            len(inp) >= 2
            and len(inp[0]) == 2
            and len(inp[1]) == 2
            and inp[0][1] == inp[1][0]
        ):
            region_dims = {
                "M": int(inp[0][0]),
                "K": int(inp[0][1]),
                "N": int(inp[1][1]),
            }

    candidates: list[_Cand] = []
    for tile_entry in curve:
        tile = tile_entry["tile"]
        live = int(tile_entry["live_bytes"])
        fits_l2 = bool(tile_entry["fits_l2"])
        fits_scratchpad = bool(tile_entry["fits_scratchpad"])
        # Smaller tiles are cheaper (relative). Square root scaling roughly
        # tracks how much pressure the tile places on caches.
        rel_cost = round((live / max_live) ** 0.5, 4)

        # M-37.9 Fix 3b: boundary penalty. When the region's actual
        # dimensions are known and the tile does NOT divide them
        # cleanly, multiply the cost by 1.5 so a clean-divide tile
        # (when one exists) sorts ahead. The penalty value is tuned
        # so a clean-divide tile twice the cache footprint (cost
        # ~sqrt(2) ≈ 1.41) still wins. ``boundary_required`` is
        # surfaced in cost_preview so the validator + agent can see it.
        boundary_required = False
        if region_dims and {"M", "N", "K"} <= set(tile):
            tM = int(tile.get("M", 0)) or 0
            tN = int(tile.get("N", 0)) or 0
            tK = int(tile.get("K", 0)) or 0
            if tM > 0 and tN > 0 and tK > 0:
                boundary_required = not (
                    region_dims["M"] % tM == 0
                    and region_dims["N"] % tN == 0
                    and region_dims["K"] % tK == 0
                )
        if boundary_required:
            rel_cost = round(rel_cost * 1.5, 4)

        # Tile-size label: M_N_K for matmul, raw dict otherwise.
        if {"M", "N", "K"} <= set(tile):
            label = f"tile_M{tile['M']}_N{tile['N']}_K{tile['K']}"
        else:
            label = "tile_" + "_".join(f"{k}{v}" for k, v in sorted(tile.items()))

        cid = _candidate_id(
            "tile", rid, label,
            region_extra=_region_content_hash(dossier),
        )
        legality: dict[str, Any] = (
            {"ok": True}
            if fits_l2
            else {"ok": False, "reason": "tile working set does not fit L2"}
        )

        candidates.append(
            _Cand(
                candidate_id=cid,
                site_id=site_id,
                kind="set_tile_params",
                region_id=rid,
                label=label,
                recipe_delta=[
                    {
                        "op": "SetTileParams",
                        "region": rid,
                        "tile": dict(tile),
                    }
                ],
                legality=legality,
                cost_preview={
                    "static_relative_cost": rel_cost,
                    "live_bytes": live,
                    "fits_scratchpad": fits_scratchpad,
                    "fits_l2": fits_l2,
                    # M-37.9 Fix 3b: surface boundary requirement so
                    # the agent / validator can read it without
                    # re-deriving from shape + tile.
                    "boundary_required": boundary_required,
                    "region_dims": dict(region_dims) if region_dims else None,
                },
                evidence={
                    "region_dossier": region_dossier_ref,
                    "payload_ref": (
                        dossier["source"]["payload_ops"][0]["payload_ref"]
                        if dossier["source"]["payload_ops"]
                        else ""
                    ),
                },
            )
        )

    bottleneck = next(iter(dossier["cost"]["bottleneck_resource"].values()))
    priority = 1 if bottleneck == "compute" else 2
    site = _Site(
        site_id=site_id,
        kind="tiling",
        region_id=rid,
        priority=priority,
        reason=(
            f"{bottleneck}-bound matmul-like region with {len(candidates)} legal/illegal tile choices"
        ),
        candidate_ids=[c.candidate_id for c in candidates],
        extra={"bottleneck_resource": bottleneck},
    )
    return site, candidates


# ---- Family 6: dispatch mode (M-50) ------------------------------------- #


# Legal dispatch modes per archetype/granularity. Today's contracts are
# all NORMAL granularity → SYNC and ASYNC are legal. PERSISTENT requires
# MEGA granularity (megakernel work, not yet wired). INLINE requires
# MICRO granularity (ukernel work, not yet wired). Both PERSISTENT and
# INLINE are emitted as ILLEGAL candidates with typed reasons so the
# action space surfaces the option without admitting it.
_DISPATCH_MODES_FOR_NORMAL = ("sync", "async")
_DISPATCH_MODES_FOR_MEGA = ("persistent",)
_DISPATCH_MODES_FOR_MICRO = ("inline",)
_ALL_DISPATCH_MODES = ("sync", "async", "persistent", "inline")


def _gen_dispatch_modes(
    region: dict[str, Any],
    dossier: dict[str, Any],
    region_dossier_ref: str,
) -> tuple[_Site | None, list[_Cand]]:
    """Emit one site per kernel-bearing region with a candidate per
    dispatch mode (M-50). Legal modes track the contract's granularity
    (NORMAL / MEGA / MICRO); illegal-by-granularity modes still appear
    as candidates with typed legality.reason so the agent surface
    sees the bounded option set.

    Today every contract that lands here is NORMAL granularity (M-40
    materialises only NORMAL contracts). PERSISTENT and INLINE
    candidates are emitted ILLEGAL until M-MEGA / M-MICRO land.
    """
    if not _is_matmul_like(region["kind"]):
        return None, []
    rid = region["region_id"]
    site_id = _site_id("dispatch", rid)

    candidates: list[_Cand] = []
    legal_modes = set(_DISPATCH_MODES_FOR_NORMAL)
    for mode in _ALL_DISPATCH_MODES:
        legal: dict[str, Any]
        if mode in legal_modes:
            legal = {"ok": True}
        elif mode == "persistent":
            legal = {
                "ok": False,
                "reason": (
                    "PERSISTENT requires MEGA granularity; M-40 "
                    "materialises only NORMAL contracts today"
                ),
            }
        elif mode == "inline":
            legal = {
                "ok": False,
                "reason": (
                    "INLINE requires MICRO granularity; ukernel "
                    "dispatch path not yet wired"
                ),
            }
        else:  # pragma: no cover — defensive
            legal = {"ok": False, "reason": f"unknown dispatch mode {mode!r}"}

        label = f"dispatch_{mode}"
        cid = _candidate_id(
            "dispatch", rid, label,
            region_extra=_region_content_hash(dossier),
        )
        candidates.append(
            _Cand(
                candidate_id=cid,
                site_id=site_id,
                kind="set_dispatch_mode",
                region_id=rid,
                label=label,
                recipe_delta=[
                    {
                        "op": "SetDispatchMode",
                        "region": rid,
                        "mode": mode,
                    }
                ],
                legality=legal,
                cost_preview={
                    # All modes have the same static-cost-relative-to-each-other
                    # at this layer; the M-49 differential is what actually
                    # measures runtime cost.
                    "static_relative_cost": 1.0 if mode == "sync" else (
                        0.95 if mode == "async" else 0.0
                    ),
                    "dispatch_model": mode,
                },
                evidence={
                    "region_dossier": region_dossier_ref,
                    "payload_ref": (
                        dossier["source"]["payload_ops"][0]["payload_ref"]
                        if dossier["source"]["payload_ops"]
                        else ""
                    ),
                },
            )
        )

    site = _Site(
        site_id=site_id,
        kind="dispatch",
        region_id=rid,
        # Dispatch is lower-priority than tiling — the tile choice
        # changes the kernel; the dispatch mode changes when/how it
        # runs. Greedy will pick tile first.
        priority=3,
        reason=(
            f"dispatch-mode site for {rid} with "
            f"{sum(1 for c in candidates if c.legality['ok'])}/{len(candidates)} "
            f"legal modes (NORMAL granularity)"
        ),
        candidate_ids=[c.candidate_id for c in candidates],
        extra={"granularity": "normal"},
    )
    return site, candidates


# ---- Family 3: numerics ------------------------------------------------- #


def _gen_numerics(
    region: dict[str, Any],
    dossier: dict[str, Any],
    region_dossier_ref: str,
    profile: TargetProfile,
) -> tuple[list[_Site], list[_Cand]]:
    """Emit one site per numeric mode (set_accumulator_fp16, quantize_fp8,
    enable_fast_math). Skip entirely for non-structured regions or
    regions with no compute (kind in {tensor_empty, transpose})."""
    rid = region["region_id"]
    kind = region["kind"]
    if not _is_structured(kind):
        return [], []
    if kind in {"transpose"}:  # pure data movement — no numerics decisions
        return [], []

    sens = dossier["numerical_sensitivity"]
    sites: list[_Site] = []
    cands: list[_Cand] = []

    def _make(
        site_kind: str,
        op: str,
        cand_label: str,
        recipe_args: dict[str, Any],
        sens_key: str,
        dtype_required: str | None = None,
    ) -> None:
        status = sens[sens_key]["status"]
        eps = sens[sens_key]["eps_out"]
        site_id_ = _site_id(site_kind, rid)
        cid = _candidate_id("num", rid, cand_label)
        legal = status == "safe"
        if dtype_required and dtype_required not in profile.supported_dtypes:
            legal = False
            reason = f"target {profile.target_id} does not support {dtype_required}"
        elif not legal:
            reason = f"{sens_key} status = {status} (eps_out = {eps})"
        else:
            reason = ""
        delta = {"op": op, "region": rid, **recipe_args}
        # ``cand.kind`` reuses ``site_kind`` (already snake_case + stable),
        # while ``recipe_delta[*].op`` keeps the CamelCase op name for the
        # eventual Recipe IR commit milestone.
        cand = _Cand(
            candidate_id=cid,
            site_id=site_id_,
            kind=site_kind,
            region_id=rid,
            label=cand_label,
            recipe_delta=[delta],
            legality={"ok": legal} if legal else {"ok": False, "reason": reason},
            cost_preview={
                "static_relative_cost": _numerics_cost_estimate(op),
                "numerics_ok": legal,
                "eps_out": eps,
                "status": status,
            },
            evidence={"region_dossier": region_dossier_ref},
        )
        cands.append(cand)
        sites.append(
            _Site(
                site_id=site_id_,
                kind=site_kind,
                region_id=rid,
                priority=3,
                reason=f"{sens_key} status = {status}; potential numerical-mode change",
                candidate_ids=[cid],
                extra={"sensitivity": sens[sens_key]},
            )
        )

    _make(
        "set_accumulator_fp16",
        "SetAccumulator",
        "set_accumulator_fp16",
        {"dtype": "fp16"},
        "fp16_accum",
        dtype_required="fp16",
    )
    _make(
        "quantize_fp8",
        "QuantizeFP8",
        "quantize_fp8_e4m3",
        {"format": "e4m3"},
        "fp8_e4m3",
    )
    _make(
        "enable_fast_math",
        "EnableFastMath",
        "enable_fast_math",
        {},
        "fast_math",
    )
    return sites, cands


_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(s: str) -> str:
    """Convert a CamelCase identifier to snake_case (FP8 → fp8 idempotent)."""
    return _CAMEL_BOUNDARY_RE.sub(r"\1_\2", s).lower()


def op_lowercase(op_camel: str) -> str:  # backward-compat alias
    return _camel_to_snake(op_camel)


def _numerics_cost_estimate(op: str) -> float:
    if op == "QuantizeFP8":
        return 0.55
    if op == "SetAccumulator":
        return 0.65
    if op == "EnableFastMath":
        return 0.85
    return 1.00


# ---- Family 7: contract_feedback proposals (gap #4) ---------------------- #


def _gen_feedback_proposals(
    *,
    run_dir: Path,
    regions_by_id: dict[str, dict[str, Any]],
) -> tuple[list[_Site], list[_Cand]]:
    """Read ``04_kernel_codegen/contract_feedback_proposals.json``
    (M-59 aggregate, written by the auction's M-59 hook) and emit one
    candidate per allowlisted proposal at the matching region.

    The candidate's ``recipe_delta`` is the proposal's typed Recipe-IR
    op + args (e.g. ``SetLayout`` / ``WidenDtype`` / etc.) so the
    selection round surfaces it as a real action. When the proposal
    file is absent (first iteration) Family 7 is a no-op.
    """
    sites: list[_Site] = []
    candidates: list[_Cand] = []
    aggregate_path = run_dir / "04_kernel_codegen" / "contract_feedback_proposals.json"
    if not aggregate_path.exists():
        return sites, candidates
    try:
        body = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return sites, candidates

    entries = body.get("entries") or []
    # Group proposals by region — the aggregate keys by task_id, but
    # the M-57 auction's task_id encodes the region via the request
    # body. We re-read the request to recover the region.
    requests_dir = run_dir / "04_kernel_codegen" / "requests"
    region_by_task: dict[str, str] = {}
    if requests_dir.exists():
        for rp in sorted(requests_dir.glob("*.request.json")):
            try:
                rb = json.loads(rp.read_text(encoding="utf-8"))
                region_by_task[str(rb.get("task_id", ""))] = str(
                    rb.get("region_id", "") or ""
                )
            except (OSError, json.JSONDecodeError):
                continue

    for entry in entries:
        task_id = str(entry.get("task_id", ""))
        region_id = region_by_task.get(task_id, "")
        if not region_id or region_id not in regions_by_id:
            continue
        proposals = entry.get("proposals") or []
        if not proposals:
            continue
        site_id = _site_id("feedback_proposal", region_id)
        cand_ids: list[str] = []
        for i, p in enumerate(proposals):
            op = str(p.get("op") or "")
            args = dict(p.get("args") or {})
            if not op:
                continue
            cand_id = _candidate_id(
                kind="feedback_proposal",
                region_id=region_id,
                label=f"{op}_{i}",
            )
            cand_ids.append(cand_id)
            label = f"feedback_{op}_{p.get('source_kind', '')}"
            recipe_delta = [{
                "op": op,
                "args": args,
                "source_kind": p.get("source_kind", ""),
                "source_provider": p.get("source_provider", ""),
                "applies_when": p.get("applies_when", ""),
            }]
            candidates.append(_Cand(
                candidate_id=cand_id,
                site_id=site_id,
                kind="feedback_proposal",
                region_id=region_id,
                label=label,
                recipe_delta=recipe_delta,
                legality={"allowed": True, "reason": "from_typed_allowlist"},
                cost_preview={
                    "expected_perf_gain": float(p.get("measured_gain", 0.0) or 0.0),
                },
                evidence={
                    "source_proposals_file": str(
                        aggregate_path.relative_to(run_dir)
                    ),
                    "source_task_id": task_id,
                },
            ))
        if cand_ids:
            sites.append(_Site(
                site_id=site_id,
                kind="feedback_proposal",
                region_id=region_id,
                priority=20,
                reason=(
                    "M-59 contract_feedback proposals from prior auction; "
                    "applying the recommendation re-enters the action space"
                ),
                candidate_ids=cand_ids,
                extra={"source_proposals_file": str(
                    aggregate_path.relative_to(run_dir)
                )},
            ))
    return sites, candidates


# ---- Family 4: fusion ---------------------------------------------------- #


def _gen_fusion(
    use_def: dict[str, Any],
    region_map_regions: list[dict[str, Any]],
    dossier_by_id: dict[str, dict[str, Any]],
    profile: TargetProfile,
    region_dossier_ref_by_id: dict[str, str],
) -> tuple[list[_Site], list[_Cand]]:
    """Walk transient single-consumer edges in the use-def graph and emit
    fuse_producer_consumer candidates. Strict criteria: producer output
    is transient, consumer_count==1, reuse_horizon ≤ 1 (immediate
    consumer), neither endpoint is opaque."""
    region_kind = {r["region_id"]: r["kind"] for r in region_map_regions}
    sites: list[_Site] = []
    cands: list[_Cand] = []
    for tensor in use_def.get("tensors", []):
        if tensor.get("consumer_count") != 1:
            continue
        if tensor.get("producer_lifetime_class") != "transient":
            continue
        horizon = tensor.get("reuse_horizon", -1)
        if not (0 <= horizon <= 1):
            continue
        producer = tensor.get("producer_region", "")
        consumers = tensor.get("consumer_regions", [])
        if not consumers:
            continue
        consumer = consumers[0]
        if producer in ("", "input", "output") or consumer in ("", "input", "output"):
            continue
        if producer == consumer:
            continue
        pkind = region_kind.get(producer, "")
        ckind = region_kind.get(consumer, "")
        if _is_opaque(pkind) or _is_opaque(ckind):
            continue
        if pkind in {"tensor_empty", "unknown"} or ckind in {"tensor_empty", "unknown"}:
            continue

        # Estimate fused live bytes: sum of tensor bytes flowing across
        # the boundary (the transient itself — once fused, it never leaves
        # registers/scratchpad). Plus a small overhead for both regions'
        # other inputs/outputs that still need to be live.
        prod_dossier = dossier_by_id.get(producer)
        cons_dossier = dossier_by_id.get(consumer)
        if prod_dossier is None or cons_dossier is None:
            continue
        prod_bytes = prod_dossier["cost"]["bytes"]
        cons_bytes = cons_dossier["cost"]["bytes"]
        fused_live = max(prod_bytes, cons_bytes)
        fits_l2 = fused_live <= profile.l2_bytes
        fits_scratchpad = fused_live <= profile.scratchpad_bytes

        site_id = _site_id("fuse", producer, suffix=_short_hash(consumer))
        label = f"fuse_{_safe(producer)}_into_{_safe(consumer)}"
        cid = _candidate_id("fuse", producer, label)
        legal = fits_l2
        legality = (
            {"ok": True}
            if legal
            else {"ok": False, "reason": "fused working-set estimate does not fit L2"}
        )
        cand = _Cand(
            candidate_id=cid,
            site_id=site_id,
            kind="fuse_producer_consumer",
            region_id=producer,
            label=label,
            recipe_delta=[
                {
                    "op": "FuseProducerConsumer",
                    "producer": producer,
                    "consumer": consumer,
                    "via_tensor": tensor["tensor_id"],
                }
            ],
            legality=legality,
            cost_preview={
                "static_relative_cost": 0.78 if legal else 1.10,
                "fused_live_bytes": fused_live,
                "fits_scratchpad": fits_scratchpad,
                "fits_l2": fits_l2,
            },
            evidence={
                "producer_dossier": region_dossier_ref_by_id.get(producer, ""),
                "consumer_dossier": region_dossier_ref_by_id.get(consumer, ""),
                "tensor_id": tensor["tensor_id"],
            },
        )
        cands.append(cand)
        sites.append(
            _Site(
                site_id=site_id,
                kind="fusion",
                region_id=producer,
                priority=2,
                reason=(
                    f"transient single-consumer output flows into {consumer}; "
                    f"fits_l2={fits_l2}"
                ),
                candidate_ids=[cid],
                extra={"consumer_region_id": consumer, "tensor_id": tensor["tensor_id"]},
            )
        )
    return sites, cands


# ---- Family 5: placement ------------------------------------------------- #


def _gen_placement(
    region: dict[str, Any],
    dossier: dict[str, Any],
    region_dossier_ref: str,
    profile: TargetProfile,
) -> tuple[_Site, list[_Cand]]:
    rid = region["region_id"]
    site_id = _site_id("place", rid)
    candidates: list[_Cand] = []
    envelope = dossier["placement_envelope"]["devices"]
    devices = sorted({env["device"] for env in envelope})
    if not devices:
        devices = [profile.target_id]
    for dev in devices:
        env = next((e for e in envelope if e["device"] == dev), None)
        memory_fit = bool(env["memory_fit"]) if env else False
        latency = env["estimated_latency_us"] if env else 0.0
        label = f"assign_{_safe(dev)}"
        cid = _candidate_id("place", rid, label)
        legal = memory_fit
        legality = (
            {"ok": True}
            if legal
            else {"ok": False, "reason": f"region bytes do not fit on {dev}"}
        )
        candidates.append(
            _Cand(
                candidate_id=cid,
                site_id=site_id,
                kind="assign_device",
                region_id=rid,
                label=label,
                recipe_delta=[
                    {"op": "AssignDevice", "region": rid, "device": dev}
                ],
                legality=legality,
                cost_preview={
                    "static_relative_cost": 1.00,
                    "estimated_latency_us": latency,
                    "memory_fit": memory_fit,
                },
                evidence={"region_dossier": region_dossier_ref},
            )
        )
    site = _Site(
        site_id=site_id,
        kind="placement",
        region_id=rid,
        priority=4,
        reason=f"placement_envelope spans {len(devices)} device(s)",
        candidate_ids=[c.candidate_id for c in candidates],
        extra={"devices": devices},
    )
    return site, candidates


# --------------------------------------------------------------------------- #
# action_space.mlir text emit
# --------------------------------------------------------------------------- #


def _mlir_attr(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v} : i64"
    if isinstance(v, float):
        return f"{v} : f64"
    if v is None:
        return '"null"'
    if isinstance(v, list):
        return "[" + ", ".join(_mlir_attr(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_mlir_attr(val)}" for k, val in sorted(v.items())) + " }"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _emit_attrs(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_mlir_attr(d[k])}" for k in sorted(d))


def _emit_action_space_mlir(
    *,
    model_id: str,
    target_id: str,
    sites: list[_Site],
    candidates: list[_Cand],
) -> str:
    lines: list[str] = []
    head = {
        "model_id": model_id,
        "target_id": target_id,
        "site_count": len(sites),
        "candidate_count": len(candidates),
    }
    lines.append(f"compgen.action_space @{_safe(model_id)} attributes {{ {_emit_attrs(head)} }} {{")
    for s in sites:
        attrs = {
            "kind": s.kind,
            "region": s.region_id,
            "priority": s.priority,
            "reason": s.reason,
            "candidates": s.candidate_ids,
        }
        for k, v in s.extra.items():
            attrs[k] = v
        lines.append(f"  compgen.decision_site @{_safe(s.site_id)} attributes {{ {_emit_attrs(attrs)} }}")
    for c in candidates:
        cattrs = {
            "site": c.site_id,
            "kind": c.kind,
            "region": c.region_id,
            "label": c.label,
            "legality_ok": c.legality["ok"],
            "legality_reason": c.legality.get("reason", ""),
            "static_relative_cost": float(c.cost_preview.get("static_relative_cost", 1.0)),
        }
        recipe_block_lines = ["  compgen.candidate @" + _safe(c.candidate_id)
                              + " attributes { " + _emit_attrs(cattrs) + " } {"]
        for op_dict in c.recipe_delta:
            op_name = op_dict.get("op", "")
            body = {k: v for k, v in op_dict.items() if k != "op"}
            recipe_block_lines.append(
                f"    recipe.{op_lowercase(op_name)} attributes {{ {_emit_attrs(body)} }}"
            )
        recipe_block_lines.append("  }")
        lines.extend(recipe_block_lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point: build_action_space
# --------------------------------------------------------------------------- #


def build_action_space(
    run_dir: Path, target_yaml_path: Path
) -> ActionSpaceResult:
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "02_graph_analysis"
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"02_graph_analysis/ missing under {run_dir}; "
            "build_region_dossiers must run first"
        )
    profile = load_target_profile(Path(target_yaml_path))

    region_map = _read_json(out_dir / "region_map.json")
    use_def = _read_json(out_dir / "tensor_use_def_graph.json")
    graph_dossier = _read_json(out_dir / "graph_dossier_v2.json")
    # Per-region dossiers
    dossier_by_id: dict[str, dict[str, Any]] = {}
    region_dossier_ref_by_id: dict[str, str] = {}
    for rid, ref in graph_dossier["region_dossiers"].items():
        dossier_by_id[rid] = _read_json(run_dir / ref)
        region_dossier_ref_by_id[rid] = ref

    # Optional gap_action_queue.json (post-discovery enrichment)
    from compgen.graph_compilation.artifacts import stage_dir

    gd_dir = stage_dir(run_dir, "gap_discovery")
    gap_lookup: dict[str, dict[str, Any]] = {}
    if Path(gd_dir).is_dir():
        queue_path = Path(gd_dir) / "gap_action_queue.json"
        if queue_path.exists():
            try:
                queue = _read_json(queue_path)
                for g in queue.get("gaps", []):
                    region_id = g.get("region_id", "")
                    if region_id and region_id not in gap_lookup:
                        gap_lookup[region_id] = g
            except (json.JSONDecodeError, OSError):
                pass

    sites: list[_Site] = []
    candidates: list[_Cand] = []
    for region in region_map.get("regions", []):
        rid = region["region_id"]
        dossier = dossier_by_id.get(rid)
        ref = region_dossier_ref_by_id.get(rid, "")
        if dossier is None:
            continue

        # Family 1: extension closure (opaque only)
        s1, c1 = _gen_extension_closure(region, dossier, gap_lookup.get(rid), ref)
        if s1 is not None:
            sites.append(s1)
            candidates.extend(c1)

        # Opaque regions get NO tiling, fusion, or numerics candidates.
        if _is_opaque(region["kind"]):
            # Family 5 (placement) still emitted so every region has a
            # baseline placement decision — this is the contract the
            # multi-device milestone will extend.
            sp, cp = _gen_placement(region, dossier, ref, profile)
            sites.append(sp)
            candidates.extend(cp)
            continue

        # Family 2: tiling
        s2, c2 = _gen_tiling(region, dossier, ref)
        if s2 is not None:
            sites.append(s2)
            candidates.extend(c2)

        # Family 6 (M-50): dispatch mode — emit a site per kernel-bearing
        # region with one candidate per legal mode (sync/async legal for
        # NORMAL granularity; persistent/inline emitted as illegal until
        # MEGA/MICRO granularity contracts ship). Same placement
        # condition as tiling — only matmul-like regions get dispatch
        # candidates today.
        s6, c6 = _gen_dispatch_modes(region, dossier, ref)
        if s6 is not None:
            sites.append(s6)
            candidates.extend(c6)

        # Family 3: numerics
        s3, c3 = _gen_numerics(region, dossier, ref, profile)
        sites.extend(s3)
        candidates.extend(c3)

        # Family 5: placement (always emit baseline)
        sp, cp = _gen_placement(region, dossier, ref, profile)
        sites.append(sp)
        candidates.extend(cp)

    # Family 4: fusion (cross-region; needs full lookup)
    s4, c4 = _gen_fusion(
        use_def, region_map.get("regions", []), dossier_by_id, profile,
        region_dossier_ref_by_id,
    )
    sites.extend(s4)
    candidates.extend(c4)

    # Family 7: contract_feedback proposals (M-59 + gap #4 closure).
    # Reads any contract_feedback_proposals.json left by a prior
    # iteration's auction and emits one candidate per allowlisted
    # proposal at the matching region. The candidate's recipe_delta
    # records the suggested change so the next selection round
    # surfaces it as an actionable Recipe-IR op.
    s7, c7 = _gen_feedback_proposals(
        run_dir=run_dir,
        regions_by_id={r["region_id"]: r for r in region_map.get("regions", []) or []},
    )
    sites.extend(s7)
    candidates.extend(c7)

    # ------------------------------------------------------------------ #
    # 1. action_space.mlir (canonical)
    # ------------------------------------------------------------------ #
    model_id = graph_dossier.get("model_id", "model")
    mlir_text = _emit_action_space_mlir(
        model_id=model_id,
        target_id=profile.target_id,
        sites=sites,
        candidates=candidates,
    )
    mlir_path = out_dir / "action_space.mlir"
    mlir_path.write_text(mlir_text, encoding="utf-8")
    ir_sha = "sha256:" + hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()

    source_block = {
        "action_space_ir": mlir_path.relative_to(run_dir).as_posix(),
        "action_space_ir_sha256": ir_sha,
    }

    # ------------------------------------------------------------------ #
    # 2. decision_sites.json
    # ------------------------------------------------------------------ #
    decision_sites = {
        "schema_version": "decision_sites_v1",
        "model_id": model_id,
        "target_id": profile.target_id,
        "source": source_block,
        "sites": [
            {
                "site_id": s.site_id,
                "kind": s.kind,
                "region_id": s.region_id,
                "priority": s.priority,
                "reason": s.reason,
                "candidate_ids": list(s.candidate_ids),
                **{k: v for k, v in s.extra.items() if k != "sensitivity"},
            }
            for s in sites
        ],
    }
    decision_sites_path = out_dir / "decision_sites.json"
    decision_sites_path.write_text(
        json.dumps(decision_sites, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 3. candidate_actions.json
    # ------------------------------------------------------------------ #
    candidate_actions = {
        "schema_version": "candidate_actions_v1",
        "model_id": model_id,
        "target_id": profile.target_id,
        "source": source_block,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "site_id": c.site_id,
                "kind": c.kind,
                "region_id": c.region_id,
                "label": c.label,
                "recipe_delta": list(c.recipe_delta),
                "legality": dict(c.legality),
                "cost_preview": dict(c.cost_preview),
                "evidence": dict(c.evidence),
            }
            for c in candidates
        ],
    }
    candidate_actions_path = out_dir / "candidate_actions.json"
    candidate_actions_path.write_text(
        json.dumps(candidate_actions, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 4. llm_action_space.json (compact, illegal hidden)
    # ------------------------------------------------------------------ #
    legal_cands = [c for c in candidates if c.legality.get("ok")]
    illegal_cands = [c for c in candidates if not c.legality.get("ok")]
    legal_by_site: dict[str, list[_Cand]] = {}
    for c in legal_cands:
        legal_by_site.setdefault(c.site_id, []).append(c)
    site_by_id = {s.site_id: s for s in sites}
    ranked: list[dict[str, Any]] = []
    # Sort sites: priority asc, then by total flops/latency from dossier (desc).
    def _site_sort_key(sid: str) -> tuple[int, float, str]:
        s = site_by_id[sid]
        d = dossier_by_id.get(s.region_id, {})
        latency = -float(
            (d.get("cost", {}).get("estimated_latency_us") or {}).get(profile.target_id, 0.0)
        )
        return (s.priority, latency, sid)

    for sid in sorted(legal_by_site, key=_site_sort_key):
        s = site_by_id[sid]
        ranked.append(
            {
                "site_id": sid,
                "region_id": s.region_id,
                "kind": s.kind,
                "priority": s.priority,
                "why": s.reason,
                "legal_candidates": [
                    {
                        "candidate_id": c.candidate_id,
                        "label": c.label,
                        "cost_preview": dict(c.cost_preview),
                    }
                    for c in legal_by_site[sid]
                ],
            }
        )
    llm_action_space = {
        "schema_version": "llm_action_space_v1",
        "model_id": model_id,
        "target_id": profile.target_id,
        "source": source_block,
        "summary": {
            "candidate_count_total": len(candidates),
            "candidate_count_legal": len(legal_cands),
            "hidden_illegal_candidates": len(illegal_cands),
            "site_count": len(sites),
            "ranked_site_count": len(ranked),
        },
        "ranked_sites": ranked,
    }
    llm_action_space_path = out_dir / "llm_action_space.json"
    llm_action_space_path.write_text(
        json.dumps(llm_action_space, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 5. action_space_validation.json
    # ------------------------------------------------------------------ #
    region_ids = {r["region_id"] for r in region_map.get("regions", [])}
    cand_ids = {c.candidate_id for c in candidates}
    site_ids = {s.site_id for s in sites}
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    _add(
        "every_site_region_exists",
        all(s.region_id in region_ids for s in sites),
        "",
    )
    bad_cand_site = [c.candidate_id for c in candidates if c.site_id not in site_ids]
    _add(
        "every_candidate_site_exists",
        not bad_cand_site,
        f"orphans={bad_cand_site[:5]}",
    )
    bad_cand_region = [c.candidate_id for c in candidates if c.region_id not in region_ids]
    _add(
        "every_candidate_region_exists",
        not bad_cand_region,
        f"orphans={bad_cand_region[:5]}",
    )
    bad_recipe = [c.candidate_id for c in candidates if not c.recipe_delta]
    _add(
        "every_candidate_has_recipe_delta",
        not bad_recipe,
        f"empty={bad_recipe[:5]}",
    )
    bad_illegal = [c.candidate_id for c in candidates
                   if not c.legality.get("ok") and not c.legality.get("reason")]
    _add(
        "illegal_candidates_have_reason",
        not bad_illegal,
        f"missing={bad_illegal[:5]}",
    )
    # llm_action_space hides every illegal candidate
    visible_ids = {
        c["candidate_id"]
        for site in ranked
        for c in site["legal_candidates"]
    }
    _add(
        "llm_visible_only_legal",
        not (visible_ids & {c.candidate_id for c in illegal_cands}),
        "",
    )
    # Tiling candidates only on matmul-like regions
    region_kind = {r["region_id"]: r["kind"] for r in region_map.get("regions", [])}
    bad_tile = [
        c.candidate_id for c in candidates
        if c.kind == "set_tile_params" and not _is_matmul_like(region_kind.get(c.region_id, ""))
    ]
    _add(
        "tiling_candidates_only_on_matmul_like",
        not bad_tile,
        f"violations={bad_tile[:5]}",
    )
    # Tiling candidates' tile dict must come from the dossier's working_set_curve
    bad_invented = []
    for c in candidates:
        if c.kind != "set_tile_params":
            continue
        d = dossier_by_id.get(c.region_id, {})
        curve_tiles = [t["tile"] for t in d.get("working_set_curve", [])]
        ours = c.recipe_delta[0].get("tile") if c.recipe_delta else None
        if ours not in curve_tiles:
            bad_invented.append(c.candidate_id)
    _add(
        "tile_sizes_only_from_working_set_curve",
        not bad_invented,
        f"invented={bad_invented[:5]}",
    )
    # Opaque regions get no tiling/fusion/numerics candidates
    opaque_rids = {
        r["region_id"] for r in region_map.get("regions", [])
        if _is_opaque(r["kind"])
    }
    bad_opaque_decisions = [
        c.candidate_id for c in candidates
        if c.region_id in opaque_rids
        and c.kind in {"set_tile_params", "fuse_producer_consumer",
                       "quantize_fp8", "set_accumulator", "enable_fast_math",
                       "setaccumulator", "enablefastmath", "quantizefp8"}
    ]
    _add(
        "opaque_regions_no_tiling_or_fusion_or_numerics",
        not bad_opaque_decisions,
        f"violations={bad_opaque_decisions[:5]}",
    )
    # FP8 candidates obey numerical_sensitivity
    bad_fp8 = []
    for c in candidates:
        if c.kind != "quantize_fp8":
            continue
        d = dossier_by_id.get(c.region_id, {})
        st = d.get("numerical_sensitivity", {}).get("fp8_e4m3", {}).get("status")
        if c.legality.get("ok") and st != "safe":
            bad_fp8.append(c.candidate_id)
    _add(
        "fp8_candidates_obey_numerical_sensitivity",
        not bad_fp8,
        f"violations={bad_fp8[:5]}",
    )
    # action_space_ir_sha256 is consistent across the three projection JSONs
    sha_links = {
        "decision_sites": decision_sites["source"]["action_space_ir_sha256"],
        "candidate_actions": candidate_actions["source"]["action_space_ir_sha256"],
        "llm_action_space": llm_action_space["source"]["action_space_ir_sha256"],
    }
    _add(
        "json_projections_share_ir_sha256",
        len(set(sha_links.values())) == 1,
        f"shas={sha_links}",
    )

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    validation = {
        "schema_version": "action_space_validation_v1",
        "overall": overall,
        "totals": {
            "site_count": len(sites),
            "candidate_count_total": len(candidates),
            "candidate_count_legal": len(legal_cands),
            "candidate_count_illegal": len(illegal_cands),
            "tiling_candidates": sum(1 for c in candidates if c.kind == "set_tile_params"),
            "fusion_candidates": sum(1 for c in candidates if c.kind == "fuse_producer_consumer"),
            "extension_closure_candidates": sum(
                1 for c in candidates
                if c.kind in {"create_payload_lowering_extension",
                              "create_kernel_contract", "keep_as_fallback"}
            ),
            "fp8_candidates": sum(1 for c in candidates if c.kind == "quantize_fp8"),
        },
        "source": source_block,
        "checks": checks,
    }
    validation_path = out_dir / "action_space_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    return ActionSpaceResult(
        action_space_mlir_path=mlir_path,
        action_space_ir_sha256=ir_sha,
        decision_sites_path=decision_sites_path,
        candidate_actions_path=candidate_actions_path,
        llm_action_space_path=llm_action_space_path,
        action_space_validation_path=validation_path,
        site_count=len(sites),
        candidate_count_total=len(candidates),
        candidate_count_legal=len(legal_cands),
    )
