"""Graph Dossier V3 — Unified Agent View (Milestone 10B).

Read-only aggregator that joins the existing graph-analysis and
recipe-planning artifacts into a single agent-facing dossier:

- ``02_graph_analysis/graph_dossier_v3.json`` — full per-region join
  (facts + decision sites + legal/illegal candidates + selected
  candidate + obligation status).
- ``02_graph_analysis/graph_dossier_v3.mlir`` — compact text IR
  projection (one ``dossier.region`` op per region, hand-rolled string
  emission consistent with ``action_space.mlir``).
- ``02_graph_analysis/graph_dossier_v3_validation.json`` — V3R001-V3R011
  pass/fail with concrete details on every failure.
- ``02_graph_analysis/llm_graph_view.json`` — bounded prompt-friendly
  view containing **only legal candidates**, ranked for prompt budget.

This module performs **no new analysis**. Numerical-sensitivity,
working-set, cost and reuse fields are projected verbatim or as simple
sums from existing v2 region dossiers. Candidate legality, costs, and
recipe deltas come straight from ``candidate_actions.json``.

Hard invariants:

- ``01_payload_lowering/`` and all upstream stages are read-only.
- Existing artifacts under ``02_graph_analysis/`` and
  ``03_recipe_planning/`` are read-only — v3 emits new files alongside
  them, never mutates them.
- The v3 outputs live under ``02_graph_analysis/`` but are emitted
  *after* recipe_planning has run (so the planning-side fields can
  populate). That means ``graph_analysis.output_hash`` recorded by the
  manifest pre-dates v3; v3 is byte-pinned by its own internal
  ``source.<input>_sha256`` fields instead.
- Idempotent: running M-10B twice on the same run dir produces
  byte-identical artifacts (no timestamps in the body — every
  ``generated_at_utc`` field is recorded in a ``meta`` block excluded
  from the idempotence test, and JSON is always emitted with
  ``sort_keys=True``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# IR-projection helpers (mirrors action_space.py: text-only, no xDSL deps)
# --------------------------------------------------------------------------- #


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_]+")


def _safe(s: str) -> str:
    out = _SAFE_ID_RE.sub("_", s).strip("_")
    return out or "x"


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
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _emit_attrs(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_mlir_attr(d[k])}" for k in sorted(d))


# --------------------------------------------------------------------------- #
# Result dataclass + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphDossierV3Result:
    overall: str  # "pass" | "fail"
    out_dir: Path
    json_path: Path
    mlir_path: Path
    validation_path: Path
    llm_view_path: Path
    failures: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


_LLM_VIEW_BUDGET = {
    "max_visible_regions": 12,
    "max_candidates_per_region": 6,
}


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _project_facts(region_dossier: dict[str, Any] | None) -> dict[str, Any]:
    """Project a per-region v2 dossier into the v3 ``facts`` shape.

    Numerical sensitivity is copied verbatim. Working-set summary is
    derived from ``reuse.{inputs,outputs}.bytes`` (sum of bytes, count
    of transients) — no new analysis.
    """
    if region_dossier is None:
        return {
            "dossier_ref": None,
            "cost": None,
            "numerical_sensitivity_summary": None,
            "working_set_summary": None,
            "legality_constraints": [],
        }
    cost = region_dossier.get("cost", {}) or {}
    nums = region_dossier.get("numerical_sensitivity", {}) or {}
    reuse = region_dossier.get("reuse", {}) or {}
    inputs = list(reuse.get("inputs", []) or [])
    outputs = list(reuse.get("outputs", []) or [])
    transient_input_count = sum(
        1 for t in inputs if t.get("lifetime_class") == "transient"
    )
    transient_output_count = sum(
        1 for t in outputs if t.get("lifetime_class") == "transient"
    )
    return {
        "dossier_ref": None,  # caller fills with relative path
        "cost": {
            "flops": cost.get("flops"),
            "bytes": cost.get("bytes"),
            "arithmetic_intensity": cost.get("arithmetic_intensity"),
            "estimated_latency_us": cost.get("estimated_latency_us"),
            "bottleneck_resource": cost.get("bottleneck_resource"),
        },
        "numerical_sensitivity_summary": {
            mode: dict(nums[mode]) for mode in nums
        } if nums else None,
        "working_set_summary": {
            "input_count": len(inputs),
            "output_count": len(outputs),
            "input_bytes_total": sum(int(t.get("bytes", 0) or 0) for t in inputs),
            "output_bytes_total": sum(int(t.get("bytes", 0) or 0) for t in outputs),
            "transient_input_count": transient_input_count,
            "transient_output_count": transient_output_count,
        },
        "legality_constraints": list(region_dossier.get("legality_constraints", []) or []),
    }


def _build_region_row(
    *,
    region: dict[str, Any],
    region_dossier: dict[str, Any] | None,
    region_dossier_ref: str | None,
    sites_for_region: list[dict[str, Any]],
    candidates_for_region: list[dict[str, Any]],
    selection: dict[str, Any] | None,
    obligation_for_selected: dict[str, Any] | None,
    obligation_status_record: dict[str, Any] | None,
    obligation_status_source_stage: str | None,
) -> dict[str, Any]:
    facts = _project_facts(region_dossier)
    facts["dossier_ref"] = region_dossier_ref

    # Site projections.
    sites_proj = [
        {
            "site_id": s.get("site_id", ""),
            "kind": s.get("kind", ""),
            "priority": s.get("priority"),
            "reason": s.get("reason", ""),
            "gap_id": s.get("gap_id", ""),
            "candidate_count": len(s.get("candidate_ids", []) or []),
            **({"devices": s["devices"]} if "devices" in s else {}),
            **({"sensitivity": s["sensitivity"]} if "sensitivity" in s else {}),
        }
        for s in sites_for_region
    ]

    legal: list[dict[str, Any]] = []
    illegal: list[dict[str, Any]] = []
    for c in candidates_for_region:
        leg = c.get("legality", {}) or {}
        proj = {
            "candidate_id": c.get("candidate_id", ""),
            "site_id": c.get("site_id", ""),
            "kind": c.get("kind", ""),
            "label": c.get("label", ""),
            "cost_preview": dict(c.get("cost_preview", {}) or {}),
            "recipe_delta": list(c.get("recipe_delta", []) or []),
        }
        if leg.get("ok") is True:
            proj["legality_reason"] = ""
            legal.append(proj)
        else:
            proj["rejection_reason"] = leg.get("reason", "")
            illegal.append(proj)

    selected_block: dict[str, Any] | None = None
    if selection is not None and selection.get("region_id") == region.get("region_id"):
        cand_id = selection.get("selected_candidate_id", "")
        recipe_op_id = ""
        obligation_id = ""
        if obligation_for_selected is not None:
            recipe_op_id = obligation_for_selected.get("recipe_op_id", "")
            obligation_id = obligation_for_selected.get("id", "")
        # Project the obligation status record into a normalized shape
        # (V3R007: M-09 wins over M-08; v2 status carries `discharged`).
        obligation_status: dict[str, Any] | None = None
        if obligation_status_record is not None:
            declared = obligation_status_record.get("declared_refinement", "")
            remaining = list(obligation_status_record.get("remaining", []) or [])
            status = obligation_status_record.get("status", "")
            entry: dict[str, Any] = {
                "source_stage": obligation_status_source_stage,
                "declared_refinement": declared,
                "remaining": remaining,
                "status": status,
            }
            if "discharged" in obligation_status_record:
                entry["discharged"] = list(
                    obligation_status_record.get("discharged", []) or []
                )
            obligation_status = entry
        selected_block = {
            "candidate_id": cand_id,
            "site_id": selection.get("site_id", ""),
            "kind": selection.get("candidate_kind", ""),
            "label": selection.get("label", ""),
            "rationale_primary": (selection.get("rationale", {}) or {}).get(
                "primary_reason", ""
            ),
            "selected_at_utc": selection.get("selected_at_utc", ""),
            "recipe_op_id": recipe_op_id,
            "obligation_id": obligation_id,
            "obligation_status": obligation_status,
        }

    return {
        "region_id": region.get("region_id", ""),
        "kind": region.get("kind", ""),
        "module_id": region.get("module_id", ""),
        "source_classification": region.get("source_classification", ""),
        "fx_nodes": list(region.get("fx_nodes", []) or []),
        "estimated": dict(region.get("estimated", {}) or {}),
        "facts": facts,
        "decision_sites": sites_proj,
        "legal_candidates": legal,
        "illegal_candidates": illegal,
        "selected": selected_block,
    }


def _emit_v3_mlir(
    *, model_id: str, target_id: str, summary: dict[str, Any],
    regions: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    head = {
        "schema_version": "graph_dossier_v3",
        "model_id": model_id,
        "target_id": target_id,
        "total_regions": int(summary.get("total_regions", 0)),
        "total_legal_candidates": int(summary.get("total_legal_candidates", 0)),
        "total_illegal_candidates": int(summary.get("total_illegal_candidates", 0)),
        "selected_candidate_count": int(summary.get("selected_candidate_count", 0)),
        "obligation_count": int(summary.get("obligation_count", 0)),
        "obligation_status_source_stage":
            summary.get("obligation_status_source_stage") or "",
    }
    lines.append(
        f"compgen.dossier @{_safe(model_id)} "
        f"attributes {{ {_emit_attrs(head)} }} {{"
    )
    for r in regions:
        sel = r.get("selected") or {}
        ostat = sel.get("obligation_status") or {}
        attrs = {
            "region_id": r["region_id"],
            "kind": r.get("kind", ""),
            "module_id": r.get("module_id", ""),
            "source_classification": r.get("source_classification", ""),
            "legal_candidate_count": len(r.get("legal_candidates", [])),
            "illegal_candidate_count": len(r.get("illegal_candidates", [])),
            "selected_candidate": sel.get("candidate_id", ""),
            "recipe_op": sel.get("recipe_op_id", ""),
            "obligation": sel.get("obligation_id", ""),
            "obligation_status": ostat.get("status", "") if ostat else "",
            "obligation_status_source": ostat.get("source_stage", "") if ostat else "",
        }
        lines.append(
            f"  dossier.region @{_safe(r['region_id'])} "
            f"attributes {{ {_emit_attrs(attrs)} }}"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _build_llm_graph_view(
    *, model_id: str, target_id: str, regions: list[dict[str, Any]],
    critical_path: list[str],
) -> dict[str, Any]:
    crit = list(critical_path or [])

    def _priority(r: dict[str, Any]) -> tuple[int, int, int, str]:
        sel = 0 if r.get("selected") is not None else 1
        on_crit = 0 if r["region_id"] in crit else 1
        legal_count_neg = -len(r.get("legal_candidates", []))
        return (sel, on_crit, legal_count_neg, r["region_id"])

    ordered = sorted(regions, key=_priority)
    cap_r = _LLM_VIEW_BUDGET["max_visible_regions"]
    cap_c = _LLM_VIEW_BUDGET["max_candidates_per_region"]
    visible = ordered[:cap_r]
    truncated = len(regions) > cap_r

    out_regions: list[dict[str, Any]] = []
    actual_candidates_total = 0
    for r in visible:
        nss = (r.get("facts", {}) or {}).get("numerical_sensitivity_summary") or {}
        cost = (r.get("facts", {}) or {}).get("cost") or {}
        facts_brief = {
            "flops": cost.get("flops"),
            "bytes": cost.get("bytes"),
            "bottleneck_resource": cost.get("bottleneck_resource"),
            "fp16_accum_status": (nss.get("fp16_accum") or {}).get("status"),
            "fp8_e4m3_status": (nss.get("fp8_e4m3") or {}).get("status"),
            "fast_math_status": (nss.get("fast_math") or {}).get("status"),
        }
        sites_brief = [
            {
                "site_id": s["site_id"],
                "kind": s["kind"],
                "priority": s["priority"],
                "reason": s["reason"],
            }
            for s in r.get("decision_sites", [])[:cap_c]
        ]
        legal = r.get("legal_candidates", [])[:cap_c]
        legal_brief = [
            {
                "candidate_id": c["candidate_id"],
                "site_id": c["site_id"],
                "kind": c["kind"],
                "label": c["label"],
                "static_relative_cost": (c.get("cost_preview") or {}).get(
                    "static_relative_cost"
                ),
            }
            for c in legal
        ]
        actual_candidates_total += len(legal_brief)
        sel = r.get("selected")
        sel_brief = None
        if sel is not None:
            ostat = sel.get("obligation_status") or {}
            sel_brief = {
                "candidate_id": sel["candidate_id"],
                "kind": sel["kind"],
                "obligation_status": ostat.get("status", "") if ostat else "",
            }
        out_regions.append(
            {
                "region_id": r["region_id"],
                "kind": r.get("kind", ""),
                "source_classification": r.get("source_classification", ""),
                "facts_brief": facts_brief,
                "decision_sites_brief": sites_brief,
                "legal_candidates": legal_brief,
                "selected": sel_brief,
            }
        )

    return {
        "schema_version": "llm_graph_view_v1",
        "model_id": model_id,
        "target_id": target_id,
        "budget": {
            "max_visible_regions": cap_r,
            "max_candidates_per_region": cap_c,
            "actual_regions": len(out_regions),
            "actual_candidates_total": actual_candidates_total,
            "truncated": truncated,
        },
        "regions": out_regions,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_v3(
    *, dossier: dict[str, Any], llm_view: dict[str, Any],
    region_map: dict[str, Any], decision_sites: dict[str, Any],
    candidate_actions: dict[str, Any],
    candidate_selection: dict[str, Any] | None,
    semantic_obligations: dict[str, Any] | None,
) -> dict[str, Any]:
    region_ids_in_map = {r["region_id"] for r in region_map.get("regions", [])}
    sites_index: dict[str, dict[str, Any]] = {
        s["site_id"]: s for s in decision_sites.get("sites", [])
    }
    candidates_index: dict[str, dict[str, Any]] = {
        c["candidate_id"]: c for c in candidate_actions.get("candidates", [])
    }
    obligation_ids: set[str] = set()
    obligation_by_id: dict[str, dict[str, Any]] = {}
    if semantic_obligations is not None:
        for ob in semantic_obligations.get("obligations", []):
            obligation_ids.add(ob.get("id", ""))
            obligation_by_id[ob.get("id", "")] = ob

    checks: list[dict[str, Any]] = []
    counts = {
        "regions_checked": 0,
        "sites_checked": 0,
        "candidates_checked": 0,
        "obligations_checked": 0,
    }

    def _add(rule_id: str, name: str, fail_details: list[str]) -> None:
        checks.append(
            {
                "id": rule_id,
                "name": name,
                "status": "pass" if not fail_details else "fail",
                "fail_count": len(fail_details),
                "details": fail_details,
            }
        )

    # V3R001
    fails: list[str] = []
    for r in dossier.get("regions", []):
        counts["regions_checked"] += 1
        if r["region_id"] not in region_ids_in_map:
            fails.append(f"region {r['region_id']!r} not in region_map")
    _add("V3R001", "regions_in_region_map", fails)

    # V3R002
    fails = []
    for r in dossier.get("regions", []):
        for s in r.get("decision_sites", []):
            counts["sites_checked"] += 1
            sid = s["site_id"]
            if sid not in sites_index:
                fails.append(
                    f"region {r['region_id']!r} references site {sid!r} not in decision_sites"
                )
                continue
            if sites_index[sid].get("region_id") != r["region_id"]:
                fails.append(
                    f"region {r['region_id']!r} references site {sid!r} that "
                    f"belongs to region {sites_index[sid].get('region_id')!r}"
                )
    _add("V3R002", "decision_sites_resolve", fails)

    # V3R003 + V3R004
    fails_legal: list[str] = []
    fails_illegal: list[str] = []
    for r in dossier.get("regions", []):
        for c in r.get("legal_candidates", []):
            counts["candidates_checked"] += 1
            cand = candidates_index.get(c["candidate_id"])
            if cand is None:
                fails_legal.append(
                    f"legal candidate {c['candidate_id']!r} not in candidate_actions"
                )
            elif (cand.get("legality") or {}).get("ok") is not True:
                fails_legal.append(
                    f"legal candidate {c['candidate_id']!r} has legality.ok != true"
                )
        for c in r.get("illegal_candidates", []):
            counts["candidates_checked"] += 1
            cand = candidates_index.get(c["candidate_id"])
            if cand is None:
                fails_illegal.append(
                    f"illegal candidate {c['candidate_id']!r} not in candidate_actions"
                )
            elif (cand.get("legality") or {}).get("ok") is True:
                fails_illegal.append(
                    f"illegal candidate {c['candidate_id']!r} has legality.ok == true"
                )
    _add("V3R003", "legal_candidates_legal_in_actions", fails_legal)
    _add("V3R004", "illegal_candidates_illegal_in_actions", fails_illegal)

    # V3R005
    fails = []
    for r in dossier.get("regions", []):
        sel = r.get("selected")
        if sel is None:
            continue
        legal_ids = {c["candidate_id"] for c in r.get("legal_candidates", [])}
        if sel["candidate_id"] not in legal_ids:
            fails.append(
                f"region {r['region_id']!r} selected candidate "
                f"{sel['candidate_id']!r} not in its legal_candidates"
            )
        if (
            candidate_selection is not None
            and sel["candidate_id"]
            != candidate_selection.get("selected_candidate_id")
        ):
            fails.append(
                f"region {r['region_id']!r} selected.candidate_id "
                f"{sel['candidate_id']!r} disagrees with "
                f"candidate_selection.selected_candidate_id "
                f"{candidate_selection.get('selected_candidate_id')!r}"
            )
    _add("V3R005", "selected_candidate_consistent", fails)

    # V3R006
    fails = []
    for r in dossier.get("regions", []):
        sel = r.get("selected")
        if sel is None:
            continue
        oid = sel.get("obligation_id", "")
        if not oid:
            continue
        counts["obligations_checked"] += 1
        ob = obligation_by_id.get(oid)
        if ob is None:
            fails.append(
                f"region {r['region_id']!r} obligation {oid!r} not in semantic_obligations"
            )
            continue
        if ob.get("source_candidate") != sel["candidate_id"]:
            fails.append(
                f"region {r['region_id']!r} obligation {oid!r} source_candidate "
                f"{ob.get('source_candidate')!r} != selected.candidate_id "
                f"{sel['candidate_id']!r}"
            )
    _add("V3R006", "obligation_links_to_selected", fails)

    # V3R007
    expected_source = dossier.get("summary", {}).get("obligation_status_source_stage")
    declared_sources_seen = set()
    for r in dossier.get("regions", []):
        sel = r.get("selected")
        if sel is None:
            continue
        ostat = sel.get("obligation_status")
        if ostat:
            declared_sources_seen.add(ostat.get("source_stage"))
    fails = []
    if declared_sources_seen and expected_source not in declared_sources_seen:
        fails.append(
            f"summary.obligation_status_source_stage={expected_source!r} but "
            f"per-selected statuses use {sorted(declared_sources_seen)!r}"
        )
    valid_sources = {None, "differential_verification", "post_lowering"}
    bad = [s for s in declared_sources_seen if s not in valid_sources]
    if bad:
        fails.append(f"unknown obligation_status source_stage values: {sorted(bad)!r}")
    _add("V3R007", "obligation_status_precedence", fails)

    # V3R008
    fails = []
    illegal_ids_per_region = {
        r["region_id"]: {c["candidate_id"] for c in r.get("illegal_candidates", [])}
        for r in dossier.get("regions", [])
    }
    for r in llm_view.get("regions", []):
        for c in r.get("legal_candidates", []):
            if c["candidate_id"] in illegal_ids_per_region.get(r["region_id"], set()):
                fails.append(
                    f"llm_graph_view region {r['region_id']!r} contains illegal "
                    f"candidate {c['candidate_id']!r}"
                )
    _add("V3R008", "llm_view_no_illegal_candidates", fails)

    # V3R009
    fails = []
    cap_r = llm_view["budget"]["max_visible_regions"]
    cap_c = llm_view["budget"]["max_candidates_per_region"]
    if len(llm_view["regions"]) > cap_r:
        fails.append(
            f"llm_graph_view.regions ({len(llm_view['regions'])}) > "
            f"max_visible_regions ({cap_r})"
        )
    for r in llm_view["regions"]:
        if len(r["legal_candidates"]) > cap_c:
            fails.append(
                f"region {r['region_id']!r} has {len(r['legal_candidates'])} "
                f"legal candidates > max_candidates_per_region ({cap_c})"
            )
    expected_truncated = (
        len(dossier.get("regions", [])) > cap_r
    )
    if bool(llm_view["budget"].get("truncated")) != expected_truncated:
        fails.append(
            f"truncated={llm_view['budget'].get('truncated')!r} disagrees "
            f"with regions_total={len(dossier.get('regions', []))} vs cap={cap_r}"
        )
    _add("V3R009", "llm_view_within_budget", fails)

    # V3R010
    fails = []
    expected_obl = (
        len(semantic_obligations.get("obligations", []))
        if semantic_obligations is not None else 0
    )
    actual = int(dossier.get("summary", {}).get("obligation_count", 0))
    if actual != expected_obl:
        fails.append(
            f"summary.obligation_count={actual} != "
            f"len(semantic_obligations)={expected_obl}"
        )
    _add("V3R010", "obligation_count_matches", fails)

    # V3R011: SHA-pinning sanity (every source.*_sha256 is non-null when its
    # corresponding source.<input> path is non-null, and is null otherwise).
    fails = []
    src = dossier.get("source", {}) or {}
    for key, val in sorted(src.items()):
        if key.endswith("_sha256"):
            continue
        path = val
        sha_key = f"{key}_sha256"
        sha = src.get(sha_key)
        if path and not sha:
            fails.append(f"source.{key} is set but {sha_key} is null/missing")
        if not path and sha:
            fails.append(f"source.{key} is null but {sha_key} is set")
    _add("V3R011", "source_sha256_pinning", fails)

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    return {
        "schema_version": "graph_dossier_v3_validation_v1",
        "overall": overall,
        "checks": checks,
        "counts": counts,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_graph_dossier_v3(run_dir: Path) -> GraphDossierV3Result:
    """Aggregate the existing graph-analysis + planning artifacts into the
    M-10B unified agent view. Read-only against every input."""
    run_dir = Path(run_dir).resolve()
    ga_dir = run_dir / "02_graph_analysis"
    rp_dir = run_dir / "03_recipe_planning"
    if not ga_dir.is_dir():
        raise FileNotFoundError(f"02_graph_analysis/ missing under {run_dir}")

    # Required inputs.
    region_map_path = ga_dir / "region_map.json"
    decision_sites_path = ga_dir / "decision_sites.json"
    candidate_actions_path = ga_dir / "candidate_actions.json"
    llm_action_space_path = ga_dir / "llm_action_space.json"
    graph_dossier_v2_path = ga_dir / "graph_dossier_v2.json"
    action_space_mlir_path = ga_dir / "action_space.mlir"
    for p in (
        region_map_path, decision_sites_path, candidate_actions_path,
        graph_dossier_v2_path,
    ):
        if not p.exists():
            raise FileNotFoundError(f"M-10B input missing: {p}")

    # Optional inputs (nullable when run stopped early).
    candidate_selection_path = rp_dir / "candidate_selection.json"
    recipe_gate_path = rp_dir / "recipe_gate_verdict.json"
    semantic_obligations_path = rp_dir / "semantic_obligations.json"
    pl_status_path = rp_dir / "post_lowering" / "semantic_obligations_status.json"
    dv_status_path = (
        rp_dir / "differential_verification" / "semantic_obligations_status.json"
    )

    region_map = _read_json(region_map_path)
    decision_sites = _read_json(decision_sites_path)
    candidate_actions = _read_json(candidate_actions_path)
    graph_dossier_v2 = _read_json(graph_dossier_v2_path)
    candidate_selection = _read_json_or_none(candidate_selection_path)
    semantic_obligations = _read_json_or_none(semantic_obligations_path)
    pl_status = _read_json_or_none(pl_status_path)
    dv_status = _read_json_or_none(dv_status_path)

    model_id = graph_dossier_v2.get("model_id", "")
    target_id = graph_dossier_v2.get("target_id", "")

    # Index inputs.
    sites_by_region: dict[str, list[dict[str, Any]]] = {}
    for s in decision_sites.get("sites", []):
        sites_by_region.setdefault(s["region_id"], []).append(s)

    candidates_by_region: dict[str, list[dict[str, Any]]] = {}
    for c in candidate_actions.get("candidates", []):
        candidates_by_region.setdefault(c["region_id"], []).append(c)

    region_dossiers_map = graph_dossier_v2.get("region_dossiers", {}) or {}

    # Obligation status precedence: M-09 > M-08 > none.
    obligation_status_source_stage: str | None
    chosen_status_records: dict[str, dict[str, Any]] = {}
    if dv_status is not None:
        obligation_status_source_stage = "differential_verification"
        for r in dv_status.get("statuses", []):
            chosen_status_records[r.get("obligation", "")] = r
    elif pl_status is not None:
        obligation_status_source_stage = "post_lowering"
        for r in pl_status.get("statuses", []):
            chosen_status_records[r.get("obligation", "")] = r
    else:
        obligation_status_source_stage = None

    obligations_by_source_candidate: dict[str, dict[str, Any]] = {}
    if semantic_obligations is not None:
        for ob in semantic_obligations.get("obligations", []):
            sc = ob.get("source_candidate", "")
            if sc:
                obligations_by_source_candidate[sc] = ob

    # Build per-region rows in region_map order.
    regions_v3: list[dict[str, Any]] = []
    selected_count = 0
    for region in region_map.get("regions", []):
        rid = region["region_id"]
        region_dossier_ref = region_dossiers_map.get(rid)
        region_dossier = (
            _read_json(run_dir / region_dossier_ref)
            if region_dossier_ref and (run_dir / region_dossier_ref).exists()
            else None
        )
        sel: dict[str, Any] | None = None
        obligation_for_selected = None
        obligation_status_record = None
        if (
            candidate_selection is not None
            and candidate_selection.get("region_id") == rid
        ):
            sel = candidate_selection
            cand_id = sel.get("selected_candidate_id", "")
            obligation_for_selected = obligations_by_source_candidate.get(cand_id)
            if obligation_for_selected is not None:
                obligation_status_record = chosen_status_records.get(
                    obligation_for_selected.get("id", "")
                )
        row = _build_region_row(
            region=region,
            region_dossier=region_dossier,
            region_dossier_ref=region_dossier_ref,
            sites_for_region=sites_by_region.get(rid, []),
            candidates_for_region=candidates_by_region.get(rid, []),
            selection=sel,
            obligation_for_selected=obligation_for_selected,
            obligation_status_record=obligation_status_record,
            obligation_status_source_stage=obligation_status_source_stage,
        )
        if row.get("selected") is not None:
            selected_count += 1
        regions_v3.append(row)

    total_legal = sum(len(r.get("legal_candidates", [])) for r in regions_v3)
    total_illegal = sum(len(r.get("illegal_candidates", [])) for r in regions_v3)
    summary = {
        **(graph_dossier_v2.get("summary", {}) or {}),
        "total_regions": len(regions_v3),
        "total_decision_sites": sum(
            len(r.get("decision_sites", [])) for r in regions_v3
        ),
        "total_legal_candidates": total_legal,
        "total_illegal_candidates": total_illegal,
        "selected_candidate_count": selected_count,
        "obligation_count": (
            len(semantic_obligations.get("obligations", []))
            if semantic_obligations is not None else 0
        ),
        "obligation_status_source_stage": obligation_status_source_stage,
    }

    source = {
        "graph_dossier_v2": str(graph_dossier_v2_path.relative_to(run_dir)),
        "graph_dossier_v2_sha256": _sha256_file(graph_dossier_v2_path),
        "region_map": str(region_map_path.relative_to(run_dir)),
        "region_map_sha256": _sha256_file(region_map_path),
        "decision_sites": str(decision_sites_path.relative_to(run_dir)),
        "decision_sites_sha256": _sha256_file(decision_sites_path),
        "candidate_actions": str(candidate_actions_path.relative_to(run_dir)),
        "candidate_actions_sha256": _sha256_file(candidate_actions_path),
        "llm_action_space": (
            str(llm_action_space_path.relative_to(run_dir))
            if llm_action_space_path.exists() else None
        ),
        "llm_action_space_sha256": _sha256_or_none(llm_action_space_path),
        "action_space_mlir": (
            str(action_space_mlir_path.relative_to(run_dir))
            if action_space_mlir_path.exists() else None
        ),
        "action_space_mlir_sha256": _sha256_or_none(action_space_mlir_path),
        "candidate_selection": (
            str(candidate_selection_path.relative_to(run_dir))
            if candidate_selection_path.exists() else None
        ),
        "candidate_selection_sha256": _sha256_or_none(candidate_selection_path),
        "recipe_gate_verdict": (
            str(recipe_gate_path.relative_to(run_dir))
            if recipe_gate_path.exists() else None
        ),
        "recipe_gate_verdict_sha256": _sha256_or_none(recipe_gate_path),
        "semantic_obligations": (
            str(semantic_obligations_path.relative_to(run_dir))
            if semantic_obligations_path.exists() else None
        ),
        "semantic_obligations_sha256": _sha256_or_none(semantic_obligations_path),
        "post_lowering_status": (
            str(pl_status_path.relative_to(run_dir))
            if pl_status_path.exists() else None
        ),
        "post_lowering_status_sha256": _sha256_or_none(pl_status_path),
        "differential_verification_status": (
            str(dv_status_path.relative_to(run_dir))
            if dv_status_path.exists() else None
        ),
        "differential_verification_status_sha256": _sha256_or_none(dv_status_path),
    }

    dossier = {
        "schema_version": "graph_dossier_v3",
        "model_id": model_id,
        "target_id": target_id,
        "source": source,
        "summary": summary,
        "critical_path": list(graph_dossier_v2.get("critical_path", []) or []),
        "regions": regions_v3,
    }

    llm_view = _build_llm_graph_view(
        model_id=model_id, target_id=target_id, regions=regions_v3,
        critical_path=list(graph_dossier_v2.get("critical_path", []) or []),
    )

    validation = _validate_v3(
        dossier=dossier, llm_view=llm_view,
        region_map=region_map, decision_sites=decision_sites,
        candidate_actions=candidate_actions,
        candidate_selection=candidate_selection,
        semantic_obligations=semantic_obligations,
    )

    # Add a meta block excluded from the idempotence test (timestamps).
    meta_dossier = {**dossier, "meta": {"generated_at_utc": _utcnow()}}
    meta_validation = {**validation, "meta": {"generated_at_utc": _utcnow()}}
    meta_llm_view = {**llm_view, "meta": {"generated_at_utc": _utcnow()}}

    json_path = ga_dir / "graph_dossier_v3.json"
    json_path.write_text(
        json.dumps(meta_dossier, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mlir_path = ga_dir / "graph_dossier_v3.mlir"
    mlir_path.write_text(
        _emit_v3_mlir(
            model_id=model_id, target_id=target_id,
            summary=summary, regions=regions_v3,
        ),
        encoding="utf-8",
    )
    validation_path = ga_dir / "graph_dossier_v3_validation.json"
    validation_path.write_text(
        json.dumps(meta_validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    llm_view_path = ga_dir / "llm_graph_view.json"
    llm_view_path.write_text(
        json.dumps(meta_llm_view, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    failures = tuple(
        f"{c['id']} {c['name']}: {d}"
        for c in validation["checks"] if c["status"] != "pass"
        for d in c["details"]
    )

    return GraphDossierV3Result(
        overall=validation["overall"],
        out_dir=ga_dir,
        json_path=json_path,
        mlir_path=mlir_path,
        validation_path=validation_path,
        llm_view_path=llm_view_path,
        failures=failures,
    )
