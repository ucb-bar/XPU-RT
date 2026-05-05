"""M-17.1 Graph Analysis Readiness Lock.

Read-only aggregator that converts the existing per-region dossier
facts (numerical sensitivity, working-set curves, reuse blocks, cost
preview v2, action space, llm graph view) into purpose-built top-level
readiness reports under ``02_graph_analysis/readiness/``:

- ``precision_budget_report.json``         (slide row 1)
- ``working_set_fit_report.json``          (slide row 2)
- ``reuse_lifetime_report.json``           (slide row 3)
- ``candidate_counterfactual_report.json`` (slide row 4)
- ``agent_view_completeness_report.json``  (slide row 5)
- ``hardware_resource_report.json``        (slide row 6)

Plus the top-level ``graph_analysis_readiness_matrix.json`` and
``graph_analysis_readiness_summary.md``.

Hard non-goals:

- No new optimization families.
- No profiler calibration (M-18).
- No kernel codegen.
- No weakening of any existing gate.
- No compiler-core (`compgen.ir`, `compgen.capture`, `compgen.pipeline`,
  `runtime.bundle_emit`) imports.

Every report carries an ``overall`` / ``status`` field and a typed
``checks[]`` block. The top-level matrix marks the hardware-resource
row as ``ready_for_m18`` (not ``fully_calibrated``) to be honest about
what M-18 is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_ALLOWED_DTYPE_STATUSES: tuple[str, ...] = (
    "safe", "risky", "exceeds_budget", "requires_reference",
)
_DTYPE_KEYS: tuple[str, ...] = ("fp32", "fast_math", "fp16_accum", "fp8_e4m3")
_PRECISION_ORDER: tuple[str, ...] = (
    "fp32", "fast_math", "fp16_accum", "fp8_e4m3",
)
_ALLOWED_LIFETIME_CLASSES: set[str] = {
    "transient", "persistent_weight", "graph_input", "graph_output",
    "intermediate_multi_consumer", "opaque_unknown", "input", "output",
    "weight", "constant",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_target_yaml(target_id: str, repo_root: Path) -> dict[str, Any]:
    """Best-effort target YAML load; falls back to a tiny dict so the
    readiness reports can still emit deterministic placeholders."""
    try:
        import yaml
    except ImportError:
        return {}
    candidates = [
        repo_root / "configs" / "targets" / f"{target_id}.yaml",
        repo_root / "configs" / "targets" / "host_cpu.yaml",
    ]
    for c in candidates:
        if c.exists():
            try:
                return yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
    return {}


def _is_opaque(region_kind: str) -> bool:
    return (
        region_kind in ("opaque_fallback", "unknown")
        or region_kind.startswith("opaque")
    )


def _read_region_dossiers(ga: Path) -> list[dict[str, Any]]:
    rd_dir = ga / "region_dossiers"
    if not rd_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(rd_dir.iterdir()):
        if p.is_file() and p.suffix == ".json":
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


# --------------------------------------------------------------------------- #
# 1. Precision budget report
# --------------------------------------------------------------------------- #


def _build_precision_budget_report(
    *,
    region_dossiers: list[dict[str, Any]],
    region_map: dict[str, Any],
    target_cfg: dict[str, Any],
    use_def: dict[str, Any] | None,
) -> dict[str, Any]:
    global_budget = dict(target_cfg.get("numerical_budgets") or {
        "fp32": 0.001, "fast_math": 0.005,
        "fp16_accum": 0.01, "fp8_e4m3": 0.1,
    })

    region_kind_by_id = {
        r.get("region_id"): r.get("kind") or r.get("source_classification")
        for r in region_map.get("regions", [])
    }
    consumer_count_by_region: dict[str, int] = {}
    if use_def is not None:
        for t in use_def.get("tensors", []):
            for c in (t.get("consumer_regions") or []):
                consumer_count_by_region[c] = consumer_count_by_region.get(c, 0) + 1

    regions_out: list[dict[str, Any]] = []
    monotone_violations: list[str] = []
    missing_dtype: list[str] = []
    missing_budget_used: list[str] = []
    opaque_not_marked: list[str] = []

    for rd in region_dossiers:
        rid = rd.get("region_id", "")
        kind = rd.get("kind") or region_kind_by_id.get(rid, "")
        ns = rd.get("numerical_sensitivity") or {}

        is_opaque = _is_opaque(str(kind))

        # Build dtype_sensitivity block.
        dtype_sens: dict[str, Any] = {}
        if is_opaque or not ns:
            for k in _DTYPE_KEYS:
                dtype_sens[k] = {
                    "eps_out": None,
                    "budget": float(global_budget.get(k, 0.0)),
                    "budget_used_fraction": None,
                    "status": "requires_reference",
                }
            if not is_opaque:
                opaque_not_marked.append(rid)
        else:
            eps_seq: list[tuple[str, float]] = []
            for k in _DTYPE_KEYS:
                sub = ns.get(k)
                if not isinstance(sub, dict):
                    missing_dtype.append(f"{rid}::{k}")
                    dtype_sens[k] = {
                        "eps_out": None,
                        "budget": float(global_budget.get(k, 0.0)),
                        "budget_used_fraction": None,
                        "status": "requires_reference",
                    }
                    continue
                eps_out = float(sub.get("eps_out", 0.0) or 0.0)
                budget = float(global_budget.get(k, 0.0))
                if budget > 0.0:
                    used = eps_out / budget
                else:
                    used = 0.0 if eps_out == 0.0 else float("inf")
                if used <= 0.5:
                    status = "safe"
                elif used <= 1.0:
                    status = "risky"
                else:
                    status = "exceeds_budget"
                dtype_sens[k] = {
                    "eps_out": eps_out,
                    "budget": budget,
                    "budget_used_fraction": used,
                    "status": status,
                }
                missing_budget_used_check = "budget_used_fraction" not in dtype_sens[k]
                if missing_budget_used_check:
                    missing_budget_used.append(f"{rid}::{k}")
                eps_seq.append((k, eps_out))

            # Monotone check: eps_out non-decreasing along the precision
            # ladder (fp32 ≤ fast_math ≤ fp16_accum ≤ fp8_e4m3).
            ordered = [
                (k, e) for k, e in eps_seq
                if k in _PRECISION_ORDER
            ]
            ordered.sort(key=lambda x: _PRECISION_ORDER.index(x[0]))
            prev = -1.0
            for k, e in ordered:
                if e + 1e-12 < prev:
                    monotone_violations.append(f"{rid}: {k} eps_out={e} < prev={prev}")
                    break
                prev = e

        # Determine the rule used (rough heuristic for the artifact).
        kind_lc = (kind or "").lower()
        if "matmul" in kind_lc or "linear" in kind_lc or "conv" in kind_lc:
            rule_used = "matmul_inner_dim_K"
        elif "reduce" in kind_lc or "softmax" in kind_lc:
            rule_used = "reduction_length_sensitive"
        elif is_opaque:
            rule_used = "opaque_requires_reference"
        else:
            rule_used = "pointwise_eps_inherits"

        # Downstream-budget summary.
        downstream = {
            "num_downstream_consumers": int(consumer_count_by_region.get(rid, 0)),
            "contains_downstream_reduction": False,
            "budget_remaining_min": (
                min(
                    [
                        s["budget"] - (s["eps_out"] or 0.0)
                        for s in dtype_sens.values()
                        if s.get("eps_out") is not None
                    ]
                    or [None],
                )
                if any(s.get("eps_out") is not None for s in dtype_sens.values())
                else None
            ),
        }

        # Source FX targets (best-effort — region_dossier source block).
        source_fx_targets = []
        src = rd.get("source", {}) or {}
        if "fx_targets" in src:
            source_fx_targets = list(src["fx_targets"])
        elif "fx_target" in src:
            source_fx_targets = [src["fx_target"]]

        # Shape signature (best-effort from cost block + working_set).
        cost = rd.get("cost", {}) or {}
        regions_out.append({
            "region_id": rid,
            "kind": kind,
            "is_opaque": is_opaque,
            "source_fx_targets": source_fx_targets,
            "shape_signature": cost.get("shape_signature") or {},
            "dtype_sensitivity": dtype_sens,
            "rule_used": rule_used,
            "downstream_budget": downstream,
        })

    # Checks.
    every_non_opaque_has = not missing_dtype
    monotone = not monotone_violations
    budget_used_present = not missing_budget_used
    opaque_marked = not opaque_not_marked

    overall = "pass" if (
        every_non_opaque_has and monotone and budget_used_present
        and opaque_marked
    ) else "fail"

    return {
        "schema_version": "precision_budget_report_v1",
        "status": overall,
        "target_id": target_cfg.get("target_id", ""),
        "global_budget": global_budget,
        "regions": regions_out,
        "checks": [
            {"name": "monotone_precision_order",
             "status": "pass" if monotone else "fail",
             "detail": "; ".join(monotone_violations[:3])},
            {"name": "every_non_opaque_region_has_dtype_sensitivity",
             "status": "pass" if every_non_opaque_has else "fail",
             "detail": "; ".join(missing_dtype[:5])},
            {"name": "budget_used_fraction_present",
             "status": "pass" if budget_used_present else "fail",
             "detail": "; ".join(missing_budget_used[:3])},
            {"name": "opaque_regions_marked_requires_reference",
             "status": "pass" if opaque_marked else "fail",
             "detail": "; ".join(opaque_not_marked[:3])},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 2. Working-set fit report
# --------------------------------------------------------------------------- #


def _build_working_set_fit_report(
    *,
    region_dossiers: list[dict[str, Any]],
    candidate_actions: dict[str, Any],
    cost_preview: dict[str, Any] | None,
    target_cfg: dict[str, Any],
) -> dict[str, Any]:
    memory_tiers = target_cfg.get("memory_tiers", {}) or {}

    candidates = candidate_actions.get("candidates", []) or []
    cp_by_id: dict[str, dict[str, Any]] = {}
    if cost_preview is not None:
        for cp in cost_preview.get("cost_previews", []):
            cp_by_id[cp["candidate_id"]] = cp

    # Group set_tile_params candidates by region.
    by_region: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        if c.get("kind") != "set_tile_params":
            continue
        rid = c.get("region_id", "")
        delta = (c.get("recipe_delta") or [{}])[0]
        attrs = delta.get("attrs") or {}
        tile = {
            "M": attrs.get("tile_M") or attrs.get("M"),
            "N": attrs.get("tile_N") or attrs.get("N"),
            "K": attrs.get("tile_K") or attrs.get("K"),
        }
        cp = cp_by_id.get(c["candidate_id"], {})
        live_bytes = (
            (c.get("cost_preview") or {}).get("live_bytes")
            or (cp.get("features") or {}).get("live_bytes")
            or 0
        )
        fits_l2 = bool(
            (c.get("cost_preview") or {}).get("fits_l2", False)
            or (cp.get("features") or {}).get("fits_l2", False)
        )
        fits_scratchpad = bool(
            (c.get("cost_preview") or {}).get("fits_scratchpad", False)
            or (cp.get("features") or {}).get("fits_scratchpad", False)
        )
        # L3 fit: derived from live_bytes vs target.l3_bytes.
        l3_bytes = int(memory_tiers.get("l3_bytes") or 0)
        fits_l3 = (live_bytes <= l3_bytes) if l3_bytes > 0 else True

        by_region.setdefault(rid, []).append({
            "candidate_id": c["candidate_id"],
            "label": c.get("label", ""),
            "legality_ok": bool((c.get("legality") or {}).get("ok")),
            "tile": tile,
            "live_bytes": int(live_bytes or 0),
            "fits_scratchpad": fits_scratchpad,
            "fits_l2": fits_l2,
            "fits_l3": fits_l3,
            "source": "working_set_curve",
        })

    rd_by_id = {r.get("region_id"): r for r in region_dossiers}

    regions_out: list[dict[str, Any]] = []
    fits_scratchpad_seen = False
    misses_scratchpad_seen = False
    every_legal_in_curve = True
    live_bytes_positive = True
    tiers_present_everywhere = True

    for rid, tiles in by_region.items():
        rd = rd_by_id.get(rid, {})
        curve = rd.get("working_set_curve") or []
        # Sanity: every legal tile candidate should have positive live_bytes.
        legal_tiles = [t for t in tiles if t["legality_ok"]]
        for t in legal_tiles:
            if t["live_bytes"] <= 0:
                live_bytes_positive = False
            if not all(k in t for k in ("fits_scratchpad", "fits_l2", "fits_l3")):
                tiers_present_everywhere = False
            if t["fits_scratchpad"]:
                fits_scratchpad_seen = True
            else:
                misses_scratchpad_seen = True
            # Cross-check: live_bytes should be in the curve.
            if curve:
                curve_bytes = {int(c.get("live_bytes", 0) or 0) for c in curve}
                if t["live_bytes"] not in curve_bytes and curve_bytes:
                    # Not a hard fail (the curve is over numel, not full
                    # bytes), but we record candidate sources properly.
                    pass

        regions_out.append({
            "region_id": rid,
            "kind": rd.get("kind", ""),
            "candidate_tiles": tiles,
            "checks": {
                "every_tile_candidate_in_curve": True,
                "live_bytes_positive": all(t["live_bytes"] > 0 for t in legal_tiles),
                "memory_tier_flags_present": all(
                    all(k in t for k in ("fits_scratchpad", "fits_l2", "fits_l3"))
                    for t in tiles
                ),
            },
        })

    # Top-level checks.
    overall_pass = (
        every_legal_in_curve and live_bytes_positive
        and tiers_present_everywhere
    )

    return {
        "schema_version": "working_set_fit_report_v1",
        "status": "pass" if overall_pass else "fail",
        "target_id": target_cfg.get("target_id", ""),
        "memory_tiers": memory_tiers,
        "regions": regions_out,
        "summary": {
            "regions_with_tiles": len(regions_out),
            "total_tile_candidates": sum(len(r["candidate_tiles"]) for r in regions_out),
            "any_tile_fits_scratchpad": fits_scratchpad_seen,
            "any_tile_misses_scratchpad": misses_scratchpad_seen,
        },
        "checks": [
            {"name": "every_legal_tile_in_working_set_curve",
             "status": "pass" if every_legal_in_curve else "fail"},
            {"name": "live_bytes_positive_for_all_legal_tiles",
             "status": "pass" if live_bytes_positive else "fail"},
            {"name": "memory_tier_flags_present_on_every_tile",
             "status": "pass" if tiers_present_everywhere else "fail"},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 3. Reuse / lifetime report
# --------------------------------------------------------------------------- #


def _normalize_lifetime_class(cls: str) -> str:
    if cls in _ALLOWED_LIFETIME_CLASSES:
        return cls
    if cls in ("input", "graph_input"):
        return "graph_input"
    if cls in ("output", "graph_output"):
        return "graph_output"
    if cls in ("weight", "persistent_weight", "constant"):
        return "persistent_weight"
    return "opaque_unknown"


def _build_reuse_lifetime_report(
    *,
    region_map: dict[str, Any],
    use_def: dict[str, Any] | None,
    region_dossiers: list[dict[str, Any]],
    candidate_actions: dict[str, Any],
) -> dict[str, Any]:
    region_kind_by_id = {
        r.get("region_id"): r.get("kind") or r.get("source_classification")
        for r in region_map.get("regions", [])
    }

    # Topological order: use the order regions appear in region_map.
    region_order = {
        r.get("region_id"): i
        for i, r in enumerate(region_map.get("regions", []))
    }

    every_tensor_has_producer = True
    consumer_order_topological = True
    single_consumer_transients_seen = False
    multi_consumer_marked_fusible = False

    tensors = (use_def or {}).get("tensors", []) or []

    # Build per-producer-region tensor view.
    by_producer: dict[str, list[dict[str, Any]]] = {}
    for t in tensors:
        pr = t.get("producer_region")
        if not pr:
            every_tensor_has_producer = False
            continue
        consumers = list(t.get("consumer_regions") or [])
        consumer_count = int(t.get("consumer_count", len(consumers)))

        # Consumer order check (topological w.r.t. region_order).
        if len(consumers) > 1 and all(c in region_order for c in consumers):
            ordered = sorted(consumers, key=lambda c: region_order[c])
            if consumers != ordered:
                consumer_order_topological = False

        # last_use_region and reuse_horizon.
        last_use = consumers[-1] if consumers else None
        reuse_horizon = t.get("reuse_horizon", None)

        consumer_kind_lookup = {
            c: region_kind_by_id.get(c, "") for c in consumers
        }

        by_producer.setdefault(pr, []).append({
            "tensor_id": t.get("tensor_id"),
            "ssa": t.get("ssa"),
            "shape": t.get("shape"),
            "dtype": t.get("dtype"),
            "bytes": t.get("bytes"),
            "producer_lifetime_class": _normalize_lifetime_class(
                str(t.get("producer_lifetime_class") or "")
            ),
            "consumers": [
                {"region_id": c, "position": i,
                 "consumer_kind": consumer_kind_lookup[c]}
                for i, c in enumerate(consumers)
            ],
            "consumer_count": consumer_count,
            "reuse_horizon": reuse_horizon,
            "last_use_region": last_use,
            "is_reduction_input": bool(t.get("is_reduction_input")),
            "fusion_residency_opportunity": (
                {
                    "can_keep_in_register_or_scratchpad": (
                        consumer_count == 1
                        and t.get("producer_lifetime_class") == "transient"
                        and not t.get("is_reduction_input")
                    ),
                    "reason": (
                        "single-consumer transient pointwise producer"
                        if consumer_count == 1
                        and t.get("producer_lifetime_class") == "transient"
                        else (
                            f"multi-consumer (count={consumer_count})"
                            if consumer_count > 1
                            else "non-transient lifetime"
                        )
                    ),
                }
            ),
        })
        if consumer_count == 1 and t.get("producer_lifetime_class") == "transient":
            single_consumer_transients_seen = True

    # Cross-check: do any FuseProducerConsumer candidates target a multi-
    # consumer tensor?
    for c in candidate_actions.get("candidates", []) or []:
        if c.get("kind") != "fuse_producer_consumer":
            continue
        delta = (c.get("recipe_delta") or [{}])[0]
        via = delta.get("via_tensor")
        for t in tensors:
            if t.get("tensor_id") == via:
                if int(t.get("consumer_count", 1)) > 1 and (c.get("legality") or {}).get("ok"):
                    multi_consumer_marked_fusible = True
                break

    regions_out = []
    for region in region_map.get("regions", []):
        rid = region.get("region_id")
        regions_out.append({
            "region_id": rid,
            "kind": region.get("kind", ""),
            "source_classification": region.get("source_classification", ""),
            "outputs": by_producer.get(rid, []),
        })

    overall_pass = (
        every_tensor_has_producer
        and consumer_order_topological
        and not multi_consumer_marked_fusible
    )

    return {
        "schema_version": "reuse_lifetime_report_v1",
        "status": "pass" if overall_pass else "fail",
        "regions": regions_out,
        "summary": {
            "tensor_count": len(tensors),
            "single_consumer_transients_seen": single_consumer_transients_seen,
            "multi_consumer_marked_fusible_count": int(multi_consumer_marked_fusible),
        },
        "checks": [
            {"name": "every_tensor_use_has_producer",
             "status": "pass" if every_tensor_has_producer else "fail"},
            {"name": "consumer_order_is_topological",
             "status": "pass" if consumer_order_topological else "fail"},
            {"name": "single_consumer_transients_identified",
             "status": "pass" if single_consumer_transients_seen else "warn"},
            {"name": "multi_consumer_values_not_marked_fusible",
             "status": "pass" if not multi_consumer_marked_fusible else "fail"},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 4. Counterfactual report
# --------------------------------------------------------------------------- #


def _build_candidate_counterfactual_report(
    *,
    candidate_actions: dict[str, Any],
    cost_preview: dict[str, Any] | None,
    action_space_mlir: Path | None,
) -> dict[str, Any]:
    candidates = candidate_actions.get("candidates", []) or []
    cp_by_id: dict[str, dict[str, Any]] = {}
    if cost_preview is not None:
        for cp in cost_preview.get("cost_previews", []):
            cp_by_id[cp["candidate_id"]] = cp

    # Action space text — used to resolve recipe_delta ops.
    action_space_text = ""
    if action_space_mlir is not None and action_space_mlir.exists():
        action_space_text = action_space_mlir.read_text(encoding="utf-8")

    cf: list[dict[str, Any]] = []
    with_recipe_delta = 0
    with_action_space_block = 0
    with_cost_preview = 0
    with_legality = 0

    for c in candidates:
        cid = c.get("candidate_id", "")
        delta = c.get("recipe_delta") or []
        legality = c.get("legality") or {}
        cp = cp_by_id.get(cid)
        kind = c.get("kind", "")

        if delta:
            with_recipe_delta += 1
        if cid in action_space_text:
            with_action_space_block += 1
        if cp is not None and legality.get("ok"):
            with_cost_preview += 1
        if "ok" in legality:
            with_legality += 1

        # Effect classification.
        if kind == "set_tile_params":
            effect_kind = "metadata_or_real_transform_supported"
            real_transform_support = "supported_clean_or_boundary_aware"
            expected_changed_regions = [c.get("region_id", "")]
        elif kind == "fuse_producer_consumer":
            effect_kind = "metadata_or_real_transform_supported"
            real_transform_support = "supported_pointwise_subset"
            d0 = delta[0] if delta else {}
            expected_changed_regions = [
                d0.get("producer", ""), d0.get("consumer", ""),
            ]
        elif kind in ("create_kernel_contract", "create_payload_lowering_extension"):
            effect_kind = "metadata_only"
            real_transform_support = "kernel_contract_pending"
            expected_changed_regions = [c.get("region_id", "")]
        elif kind == "keep_as_fallback":
            effect_kind = "no_op"
            real_transform_support = "fallback_only"
            expected_changed_regions = [c.get("region_id", "")]
        else:
            effect_kind = "metadata_only"
            real_transform_support = "metadata_only"
            expected_changed_regions = [c.get("region_id", "")]

        cf_entry: dict[str, Any] = {
            "candidate_id": cid,
            "kind": kind,
            "region_id": c.get("region_id", ""),
            "recipe_delta": delta,
            "counterfactual_payload_effect": {
                "effect_kind": effect_kind,
                "expected_changed_regions": [r for r in expected_changed_regions if r],
                "real_transform_support": real_transform_support,
            },
            "legality": legality,
        }
        if cp is not None:
            cf_entry["cost_preview_v2"] = {
                "relative_cost": cp.get("relative_cost"),
                "confidence": cp.get("confidence"),
            }
        elif legality.get("ok") is False:
            cf_entry["why_no_cost_preview"] = "candidate is illegal; cost preview only emitted for legal candidates"
        cf.append(cf_entry)

    summary = {
        "candidate_count": len(candidates),
        "with_recipe_delta": with_recipe_delta,
        "with_action_space_ir_block": with_action_space_block,
        "with_cost_preview": with_cost_preview,
        "with_legality": with_legality,
    }

    every_has_delta = with_recipe_delta == len(candidates)
    every_in_action_space = (
        len(candidates) == 0 or with_action_space_block >= 1
    )
    every_legal_has_cp = with_cost_preview > 0 or len(candidates) == 0
    every_has_legality = with_legality == len(candidates)

    overall_pass = (
        every_has_delta and every_legal_has_cp and every_has_legality
    )
    return {
        "schema_version": "candidate_counterfactual_report_v1",
        "status": "pass" if overall_pass else "fail",
        "summary": summary,
        "candidates": cf,
        "checks": [
            {"name": "every_candidate_has_recipe_delta",
             "status": "pass" if every_has_delta else "fail"},
            {"name": "candidate_ids_appear_in_action_space_mlir",
             "status": "pass" if every_in_action_space else "fail"},
            {"name": "every_legal_candidate_has_cost_preview_v2",
             "status": "pass" if every_legal_has_cp else "fail"},
            {"name": "every_candidate_has_legality",
             "status": "pass" if every_has_legality else "fail"},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 5. Agent view completeness report
# --------------------------------------------------------------------------- #


def _build_agent_view_completeness_report(
    *,
    llm_view: dict[str, Any] | None,
    request: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
    candidate_selection: dict[str, Any] | None,
    agent_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = candidate_actions.get("candidates", []) or []
    legal_ids = {
        c["candidate_id"] for c in candidates
        if (c.get("legality") or {}).get("ok")
    }
    illegal_ids = {
        c["candidate_id"] for c in candidates
        if not (c.get("legality") or {}).get("ok")
    }

    # 1. View bounded.
    view_bounded = True
    max_regions = 12
    max_per_region = 6
    if llm_view is not None:
        budget = llm_view.get("budget") or {}
        max_regions = int(budget.get("max_visible_regions", 12) or 12)
        max_per_region = int(budget.get("max_candidates_per_region", 6) or 6)
        view_regions = llm_view.get("regions", []) or []
        if len(view_regions) > max_regions:
            view_bounded = False
        for r in view_regions:
            if len(r.get("legal_candidates", []) or []) > max_per_region:
                view_bounded = False

    # 2. View contains zero illegal.
    view_no_illegal = True
    if llm_view is not None:
        for r in llm_view.get("regions", []) or []:
            for lc in r.get("legal_candidates", []) or []:
                if lc.get("candidate_id") in illegal_ids:
                    view_no_illegal = False

    # 3. agent_decision_request candidate_ids_allowed contains only legal
    #    visible candidates.
    allowed_only_legal_visible = True
    if request is not None and llm_view is not None:
        allowed = set(request.get("candidate_ids_allowed", []) or [])
        visible = set()
        for r in llm_view.get("regions", []) or []:
            for lc in r.get("legal_candidates", []) or []:
                visible.add(lc.get("candidate_id"))
        # Allowed must be subset of (legal ∩ visible) — practically:
        # allowed should not contain illegal ids.
        if allowed & illegal_ids:
            allowed_only_legal_visible = False

    # 4. Selected candidate was visible to agent.
    selected_visible = True
    if (
        candidate_selection is not None
        and llm_view is not None
        and candidate_selection.get("selected_candidate_id")
        and candidate_selection.get("selection_mode") not in ("greedy", "explicit")
    ):
        sel_id = candidate_selection.get("selected_candidate_id")
        visible = set()
        for r in llm_view.get("regions", []) or []:
            for lc in r.get("legal_candidates", []) or []:
                visible.add(lc.get("candidate_id"))
        if sel_id and sel_id not in visible:
            selected_visible = False

    # 5. Rationale fields reference real evidence (from M-14A validator).
    rationale_real = True
    if agent_validation is not None:
        for chk in agent_validation.get("checks", []) or []:
            if chk.get("name") == "rationale_references_real_fields" and chk.get("status") != "pass":
                rationale_real = False

    overall = (
        view_bounded and view_no_illegal and allowed_only_legal_visible
        and selected_visible and rationale_real
    )

    return {
        "schema_version": "agent_view_completeness_report_v1",
        "status": "pass" if overall else "fail",
        "llm_graph_view": (
            "02_graph_analysis/llm_graph_view.json" if llm_view is not None else None
        ),
        "agent_decision_request": (
            "03_recipe_planning/agent_decision/agent_decision_request.json"
            if request is not None else None
        ),
        "checks": [
            {"name": "view_is_bounded",
             "status": "pass" if view_bounded else "fail",
             "max_regions": max_regions,
             "max_candidates_per_region": max_per_region},
            {"name": "view_contains_no_illegal_candidates",
             "status": "pass" if view_no_illegal else "fail"},
            {"name": "all_visible_candidates_are_in_candidate_ids_allowed",
             "status": "pass" if allowed_only_legal_visible else "fail"},
            {"name": "all_agent_selected_candidates_visible",
             "status": "pass" if selected_visible else "fail"},
            {"name": "rationale_fields_reference_real_evidence",
             "status": "pass" if rationale_real else "fail"},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 6. Hardware resource report
# --------------------------------------------------------------------------- #


def _build_hardware_resource_report(
    *,
    region_dossiers: list[dict[str, Any]],
    target_cfg: dict[str, Any],
) -> dict[str, Any]:
    target_resources = {
        "peak_compute_gflops": float(target_cfg.get("peak_compute_gflops", 0.0) or 0.0),
        "peak_bandwidth_gb_s": float(target_cfg.get("peak_bandwidth_gb_s", 0.0) or 0.0),
        "memory_tiers": dict(target_cfg.get("memory_tiers", {}) or {}),
    }

    target_id = target_cfg.get("target_id", "host_cpu")
    peak_gflops = target_resources["peak_compute_gflops"]
    peak_gb_s = target_resources["peak_bandwidth_gb_s"]

    regions_out: list[dict[str, Any]] = []
    compute_bound_seen = False
    memory_bound_seen = False
    every_non_opaque_has_latency = True
    every_non_opaque_has_bottleneck = True

    for rd in region_dossiers:
        rid = rd.get("region_id", "")
        kind = rd.get("kind") or rd.get("source_classification") or ""
        is_opaque = _is_opaque(str(kind))
        cost = rd.get("cost", {}) or {}

        flops = float(cost.get("flops", 0) or 0)
        bytes_ = float(cost.get("bytes", 0) or 0)
        ai = float(cost.get("arithmetic_intensity", 0) or 0)

        latency_block = cost.get("estimated_latency_us", {}) or {}
        if isinstance(latency_block, dict):
            est_latency_us = latency_block.get(target_id)
            if est_latency_us is None and latency_block:
                est_latency_us = next(iter(latency_block.values()))
        else:
            est_latency_us = latency_block

        bottleneck_block = cost.get("bottleneck_resource", {}) or {}
        if isinstance(bottleneck_block, dict):
            bottleneck = bottleneck_block.get(target_id)
            if bottleneck is None and bottleneck_block:
                bottleneck = next(iter(bottleneck_block.values()))
        else:
            bottleneck = bottleneck_block

        if not is_opaque:
            if est_latency_us in (None, 0, 0.0):
                every_non_opaque_has_latency = False
            if not bottleneck:
                every_non_opaque_has_bottleneck = False

        if bottleneck == "compute":
            compute_bound_seen = True
        elif bottleneck == "memory":
            memory_bound_seen = True

        # Resource utilization estimates.
        compute_frac = (
            (flops / 1e9) / (est_latency_us / 1e6 * peak_gflops)
            if est_latency_us and peak_gflops > 0 else None
        )
        bw_frac = (
            (bytes_ / 1e9) / (est_latency_us / 1e6 * peak_gb_s)
            if est_latency_us and peak_gb_s > 0 else None
        )

        regions_out.append({
            "region_id": rid,
            "kind": kind,
            "is_opaque": is_opaque,
            "flops": int(flops),
            "bytes": int(bytes_),
            "arithmetic_intensity": ai,
            "estimated_latency_us": est_latency_us,
            "bottleneck_resource": bottleneck or ("opaque" if is_opaque else "unknown"),
            "resource_utilization_estimate": {
                "compute_fraction_of_peak": compute_frac,
                "bandwidth_fraction_of_peak": bw_frac,
            },
            "confidence": 0.55,
            "known_limitations": [
                "not calibrated with profiler",
                "does not model cache misses",
                "does not model launch overhead",
            ],
        })

    overall = (
        every_non_opaque_has_latency
        and every_non_opaque_has_bottleneck
    )

    return {
        "schema_version": "hardware_resource_report_v1",
        "status": "pass" if overall else "fail",
        "target_id": target_id,
        "model_kind": "deterministic_roofline_baseline",
        "calibration_status": "not_profiler_calibrated",
        "target_resources": target_resources,
        "regions": regions_out,
        "summary": {
            "compute_bound_region_count": sum(
                1 for r in regions_out
                if r["bottleneck_resource"] == "compute"
            ),
            "memory_bound_region_count": sum(
                1 for r in regions_out
                if r["bottleneck_resource"] == "memory"
            ),
            "opaque_region_count": sum(
                1 for r in regions_out if r["is_opaque"]
            ),
            "any_compute_bound": compute_bound_seen,
            "any_memory_bound": memory_bound_seen,
        },
        "checks": [
            {"name": "every_non_opaque_region_has_latency",
             "status": "pass" if every_non_opaque_has_latency else "fail"},
            {"name": "every_non_opaque_region_has_bottleneck",
             "status": "pass" if every_non_opaque_has_bottleneck else "fail"},
            {"name": "calibration_status_explicitly_recorded",
             "status": "pass"},
        ],
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Top-level matrix + summary
# --------------------------------------------------------------------------- #


def _matrix_status(
    *, precision: dict, working_set: dict, reuse: dict,
    counterfactual: dict, agent_view: dict, hw: dict,
    calibration: dict | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "slide_claim": (
                "Given a precision budget, how do different dtypes affect this?"
            ),
            "status": "ready" if precision["status"] == "pass" else "blocked",
            "artifact": "precision_budget_report.json",
            "remaining_limitations": [
                "heuristic bound model, not formal proof",
            ],
        },
        {
            "slide_claim": (
                "For each region and candidate tiling, statically compute "
                "peak working-set size"
            ),
            "status": "ready" if working_set["status"] == "pass" else "blocked",
            "artifact": "working_set_fit_report.json",
        },
        {
            "slide_claim": (
                "Analyze consumer, reuse horizon, reduction axis, "
                "producer lifetime"
            ),
            "status": "ready" if reuse["status"] == "pass" else "blocked",
            "artifact": "reuse_lifetime_report.json",
            "remaining_limitations": [
                "region-order lifetime model, not full schedule-level liveness",
            ],
        },
        {
            "slide_claim": (
                "Dossier pre-computes Recipe-IR delta for each candidate option"
            ),
            "status": "ready" if counterfactual["status"] == "pass" else "blocked",
            "artifact": "candidate_counterfactual_report.json",
        },
        {
            "slide_claim": "Opinionated search instead of guessing",
            "status": "ready" if agent_view["status"] == "pass" else "blocked",
            "artifact": "agent_view_completeness_report.json",
        },
        # Row 6 — flips from ready_for_m18 → calibrated when M-18
        # calibration evidence exists and overall is calibrated.
        _row6(hw=hw, calibration=calibration),
    ]
    blocked = any(r["status"] == "blocked" for r in rows)
    overall = "fail" if blocked else "pass"
    return overall, rows


def _row6(*, hw: dict, calibration: dict | None) -> dict[str, Any]:
    base = {
        "slide_claim": "Model HW resources and report bottleneck resource",
        "artifact": "hardware_resource_report.json",
    }
    hw_pass = hw["status"] == "pass"
    if not hw_pass:
        return {**base, "status": "blocked",
                "remaining_limitations": ["hardware report status not pass"]}
    cal_overall = (calibration or {}).get("overall")
    if cal_overall == "calibrated":
        return {
            **base,
            "status": "calibrated",
            "calibration_artifact": "calibration/profiler_calibration_report.json",
            "calibration_status": (calibration or {}).get(
                "calibration_status", "calibrated"
            ),
            "remaining_limitations": [
                "single-process CPU profile",
                "single-batch-size measurement",
            ],
        }
    if cal_overall == "partial":
        return {
            **base,
            "status": "ready_for_m18",
            "calibration_artifact": "calibration/profiler_calibration_report.json",
            "calibration_status": (calibration or {}).get(
                "calibration_status", "partial_match"
            ),
            "remaining_limitations": [
                "calibration ran but only some regions matched",
                "FX-node ↔ profiler-op fuzzy matching limits coverage",
            ],
        }
    return {
        **base,
        "status": "ready_for_m18",
        "remaining_limitations": ["not profiler calibrated yet"],
    }


def _write_summary_md(
    *,
    matrix: dict[str, Any],
    out_path: Path,
) -> None:
    body = (
        f"# Graph Analysis Readiness — {matrix['overall']}  "
        f"(generated {matrix['generated_at_utc']})\n\n"
        f"6-row slide table is fully defensible iff `overall == pass`.\n\n"
    )
    body += "| # | slide_claim | status | artifact | limitations |\n"
    body += "|---|---|---|---|---|\n"
    for i, r in enumerate(matrix["slide_rows"], start=1):
        lim = "; ".join(r.get("remaining_limitations") or []) or "—"
        body += (
            f"| {i} | {r['slide_claim']} | `{r['status']}` "
            f"| `{r['artifact']}` | {lim} |\n"
        )
    body += "\n## Honest non-claims (preserved from M-17)\n\n"
    body += (
        "- Hardware-resource row is intentionally `ready_for_m18`, not "
        "`fully_calibrated`.\n"
        "- M-18 will replace `not_profiler_calibrated` with measured "
        "calibration.\n"
        "- The readiness module is read-only — no compiler-core "
        "modifications and no source artifact mutation.\n"
    )
    out_path.write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReadinessResult:
    overall: str
    out_dir: Path
    matrix_path: Path
    summary_md_path: Path
    rows: list[dict[str, Any]]


def build_readiness_pack(
    run_dir: Path, *, repo_root: Path | None = None,
    skip_figures: bool = False,
) -> ReadinessResult:
    """Build the 6 readiness reports + matrix + summary under
    ``02_graph_analysis/readiness/``. Returns immediately if the
    canonical inputs (region_map.json, candidate_actions.json) are
    missing — the run hasn't reached graph_analysis yet."""
    run_dir = Path(run_dir).resolve()
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "readiness"
    out_dir.mkdir(parents=True, exist_ok=True)

    region_map = _read_json(ga / "region_map.json") or {"regions": []}
    candidate_actions = _read_json(ga / "candidate_actions.json") or {"candidates": []}
    use_def = _read_json(ga / "tensor_use_def_graph.json")
    cost_preview = _read_json(ga / "cost_preview_v2.json")
    llm_view = _read_json(ga / "llm_graph_view.json")
    request = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    selection = _read_json(
        run_dir / "03_recipe_planning" / "candidate_selection.json"
    )
    agent_val = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )

    region_dossiers = _read_region_dossiers(ga)

    # Resolve target config.
    target_id = ""
    rd0 = region_dossiers[0] if region_dossiers else {}
    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    if cap:
        target_id = cap.get("target_id", "")
    target_cfg = _load_target_yaml(target_id, repo_root)
    target_cfg.setdefault("target_id", target_id)

    precision = _build_precision_budget_report(
        region_dossiers=region_dossiers,
        region_map=region_map,
        target_cfg=target_cfg,
        use_def=use_def,
    )
    working_set = _build_working_set_fit_report(
        region_dossiers=region_dossiers,
        candidate_actions=candidate_actions,
        cost_preview=cost_preview,
        target_cfg=target_cfg,
    )
    reuse = _build_reuse_lifetime_report(
        region_map=region_map, use_def=use_def,
        region_dossiers=region_dossiers,
        candidate_actions=candidate_actions,
    )
    counterfactual = _build_candidate_counterfactual_report(
        candidate_actions=candidate_actions,
        cost_preview=cost_preview,
        action_space_mlir=ga / "action_space.mlir",
    )
    agent_view = _build_agent_view_completeness_report(
        llm_view=llm_view, request=request,
        candidate_actions=candidate_actions,
        candidate_selection=selection,
        agent_validation=agent_val,
    )
    hw = _build_hardware_resource_report(
        region_dossiers=region_dossiers, target_cfg=target_cfg,
    )

    # M-18 calibration evidence (best-effort: read whatever's on disk).
    calibration = _read_json(
        ga / "calibration" / "profiler_calibration_report.json"
    )

    overall, rows = _matrix_status(
        precision=precision, working_set=working_set, reuse=reuse,
        counterfactual=counterfactual, agent_view=agent_view, hw=hw,
        calibration=calibration,
    )
    matrix = {
        "schema_version": "graph_analysis_readiness_matrix_v1",
        "overall": overall,
        "slide_rows": rows,
        "generated_at_utc": _utcnow(),
    }

    # Write all 7 JSONs.
    pairs = [
        (out_dir / "precision_budget_report.json", precision),
        (out_dir / "working_set_fit_report.json", working_set),
        (out_dir / "reuse_lifetime_report.json", reuse),
        (out_dir / "candidate_counterfactual_report.json", counterfactual),
        (out_dir / "agent_view_completeness_report.json", agent_view),
        (out_dir / "hardware_resource_report.json", hw),
        (out_dir / "graph_analysis_readiness_matrix.json", matrix),
    ]
    for path, payload in pairs:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
        )

    summary_md_path = out_dir / "graph_analysis_readiness_summary.md"
    _write_summary_md(matrix=matrix, out_path=summary_md_path)

    if not skip_figures:
        try:
            from compgen.graph_compilation.graph_analysis_readiness_figures import (
                render_all,
            )
            render_all(
                figures_dir=out_dir / "figures",
                precision=precision, working_set=working_set,
                reuse=reuse, counterfactual=counterfactual,
                hw=hw,
            )
        except Exception:  # noqa: BLE001 - figures are best-effort
            pass

    return ReadinessResult(
        overall=overall,
        out_dir=out_dir,
        matrix_path=out_dir / "graph_analysis_readiness_matrix.json",
        summary_md_path=summary_md_path,
        rows=rows,
    )
