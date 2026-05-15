"""Per-model inspection harness.

Runs the full Section 20 stack on a single model and produces an
organized inspection bundle a human can browse to assess quality.

Per-model artifacts captured (under ``<inspection_dir>/<model_id>/``):

- ``run/`` — full pipeline output (every stage's artifacts)
- ``INSPECTION.md`` — annotated index pointing at every key file
- ``decision_summary.json`` — distilled view of greedy's pick + rationale
- ``validation_summary.json`` — every validator row + pass/fail/detail
- ``warm_cache_summary.json`` — promoted_candidates count + hit/miss
- ``pass_card_visibility.json`` — how many of the 60 cards reached the agent
``analysis_summary_index.json`` — which summaries are available

The cross-model aggregator (:func:`aggregate_inspection_packs`) emits
``OVERVIEW.md`` showing per-model status side by side.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# Per-model inspection
# --------------------------------------------------------------------------- #


@dataclass
class InspectionPack:
    """Distilled view of one model run."""

    model_id: str
    target_id: str
    run_dir: Path
    decision_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    warm_cache_summary: dict[str, Any] = field(default_factory=dict)
    pass_card_visibility: dict[str, Any] = field(default_factory=dict)
    analysis_summary_index: dict[str, Any] = field(default_factory=dict)
    load_bearing_gate_attribution: dict[str, Any] = field(default_factory=dict)
    typed_outcome: str = ""
    errors: list[str] = field(default_factory=list)
    decision_seconds: float = 0.0
    generated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_id": self.target_id,
            "run_dir": str(self.run_dir),
            "decision_summary": self.decision_summary,
            "validation_summary": self.validation_summary,
            "warm_cache_summary": self.warm_cache_summary,
            "pass_card_visibility": self.pass_card_visibility,
            "analysis_summary_index": self.analysis_summary_index,
            "load_bearing_gate_attribution": self.load_bearing_gate_attribution,
            "typed_outcome": self.typed_outcome,
            "errors": self.errors,
            "decision_seconds": self.decision_seconds,
            "generated_at_utc": self.generated_at_utc,
        }


# --------------------------------------------------------------------------- #
# G6: load-bearing-gate attribution
# --------------------------------------------------------------------------- #


def _attribute_load_bearing_gate(
    *,
    cost_preview: dict[str, Any],
    refinement_status: str,
    target_default_tiles: tuple[tuple[int, int, int], ...] = (
        (16, 16, 16), (32, 32, 32), (64, 64, 32), (128, 128, 32), (256, 256, 64),
    ),
    selected_label: str,
) -> dict[str, Any]:
    """Identify which gate carried the load for this model's outcome.

    The post-candidate-selection + verification pipeline has
    three structurally-distinct admission paths a model can travel:

    1. **bit_equality** — clean_divide AND single_k_iter holds. The
       legacy strict path. Discharged by exact equality at every case.
    2. **clean_divide_tolerance_eps** — clean_divide AND K_iters > 1.
       Recipe gate's single_k_iter rule downgrades the
       declaration; 's combined-tolerance criterion accepts the
       per-case deviation.
    3. **boundary_tolerance_eps** — tile doesn't divide cleanly.
       Recipe gate declares tolerance_eps; 's combined-tolerance
       accepts.

    Cross-cutting: shape-fit clean-divide tile emission
    only matters if the picked tile differs from the target-profile
    default set. We surface that as a separate boolean.

    Returns a dict with explicit attribution so per-model OVERVIEW
    rows can answer "which gate carried the load" without re-reading
    the run dir."""
    boundary_required = bool(cost_preview.get("boundary_required") or False)
    region_dims = cost_preview.get("region_dims") or {}
    # Parse the picked tile from the candidate label.
    tile_M = tile_N = tile_K = None
    import re as _re
    m = _re.search(r"tile_M(\d+)_N(\d+)_K(\d+)", selected_label or "")
    if m:
        tile_M, tile_N, tile_K = int(m.group(1)), int(m.group(2)), int(m.group(3))
    is_target_default = (
        tile_M is not None
        and (tile_M, tile_N, tile_K) in target_default_tiles
    )
    is_shape_fit_emission = (
        tile_M is not None
        and not is_target_default
    )
    M_dim = region_dims.get("M", 0) or 0
    N_dim = region_dims.get("N", 0) or 0
    K_dim = region_dims.get("K", 0) or 0
    # ``clean_divide`` may not appear in cost_preview directly (it lives
    # in recipe_gate's extra dict), so derive it structurally from the
    # picked tile and region dims.
    clean_divide_val = cost_preview.get("clean_divide")
    if clean_divide_val is None and tile_M is not None and M_dim > 0:
        clean_divide_val = (
            M_dim % tile_M == 0
            and N_dim % tile_N == 0
            and K_dim % tile_K == 0
        )
    clean_divide = clean_divide_val
    single_k_iter = (
        tile_K is not None and K_dim > 0 and tile_K >= K_dim
    )

    # Classify the verification path.
    if not boundary_required and clean_divide is True and single_k_iter:
        path = "bit_equality"
        load_bearing_gate = "exact_equality_per_case"
    elif not boundary_required and clean_divide is True:
        path = "clean_divide_tolerance_eps"
        load_bearing_gate = (
            "recipe_gate.single_k_iter (M-37.12) "
            "+ M-12 Higham semantic bound (M-37.13)"
        )
    elif boundary_required:
        path = "boundary_tolerance_eps"
        load_bearing_gate = (
            "recipe_gate.boundary_required → tolerance_eps "
            "+ M-12 Higham semantic bound (M-37.13)"
        )
    else:
        path = "unclassified"
        load_bearing_gate = "unknown — inspect run dir manually"

    return {
        "verification_path": path,
        "load_bearing_gate": load_bearing_gate,
        "shape_fit_tile_picked": is_shape_fit_emission,
        "selected_tile": (
            {"M": tile_M, "N": tile_N, "K": tile_K} if tile_M is not None else None
        ),
        "single_k_iter": single_k_iter,
        "boundary_required": boundary_required,
        "clean_divide": clean_divide,
        "refinement_status": refinement_status,
    }


def inspect_model_run(
    *,
    model_yaml: Path,
    target_yaml: Path,
    out_dir: Path,
    stop_after: str = "agent-decision-request",
) -> InspectionPack:
    """Run the model end-to-end and produce a structured inspection pack.

    The run lands at ``out_dir / "run"`` (cleaned if it exists).
    The pack is returned to the caller and also serialized as
    ``out_dir / "inspection_pack.json"`` plus a markdown
    ``out_dir / "INSPECTION.md"`` browsable summary.
    """
    from compgen.graph_compilation.run import run_graph_compilation

    model_yaml = Path(model_yaml).resolve()
    target_yaml = Path(target_yaml).resolve()
    out_dir = Path(out_dir).resolve()
    run_dir = out_dir / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = model_yaml.stem
    target_id = target_yaml.stem
    pack = InspectionPack(model_id=model_id, target_id=target_id, run_dir=run_dir)

    started = time.perf_counter()
    raised: Exception | None = None
    try:
        run_graph_compilation(
            model_yaml,
            target_yaml,
            run_dir,
            stop_after=stop_after,
            selection_mode="greedy",
        )
    except Exception as exc:  # noqa: BLE001 - we classify
        raised = exc
        pack.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        pack.decision_seconds = time.perf_counter() - started

    # --- Classify outcome ---
    if raised is not None:
        msg = str(raised)
        type_name = type(raised).__name__
        if (
            "M-15B" in msg
            or "downstream" in msg.lower()
            or "Unsupported" in type_name
        ):
            pack.typed_outcome = "typed_blocked"
        else:
            pack.typed_outcome = "error"
    else:
        if (run_dir / "verification_report.json").exists():
            pack.typed_outcome = "verified"
        else:
            pack.typed_outcome = "stopped_early"

    # --- Decision summary ---
    sel_path = run_dir / "03_recipe_planning" / "candidate_selection.json"
    sel = _read_json(sel_path)
    pack.decision_summary = {
        "selected_candidate_id": sel.get("selected_candidate_id", ""),
        "candidate_kind": sel.get("candidate_kind", ""),
        "label": sel.get("label", ""),
        "region_id": sel.get("region_id", ""),
        "rationale_primary": (sel.get("rationale", {}) or {}).get(
            "primary_reason", ""
        ),
        "static_relative_cost": (sel.get("cost_preview", {}) or {}).get(
            "static_relative_cost", None
        ),
        "warm_cache_hit": (
            "warm-cache" in (sel.get("rationale", {}) or {}).get(
                "primary_reason", ""
            ).lower()
        ),
    }

    # --- Validation summary ---
    val_path = (
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    val = _read_json(val_path)
    if val:
        checks = val.get("checks") or []
        pack.validation_summary = {
            "overall": val.get("overall", "unknown"),
            "check_count": len(checks),
            "pass_count": sum(1 for c in checks if c.get("status") == "pass"),
            "fail_count": sum(1 for c in checks if c.get("status") == "fail"),
            "failures": [
                {"name": c["name"], "detail": c.get("detail", "")}
                for c in checks if c.get("status") != "pass"
            ],
        }

    # --- Warm-cache summary ---
    req_path = (
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    req = _read_json(req_path)
    promoted = req.get("promoted_candidates") or []
    selected_id = pack.decision_summary.get("selected_candidate_id", "")
    matched_in_promoted = False
    matching_recipe_ids: list[str] = []
    for pc in promoted:
        if pc.get("candidate_id") == selected_id or pc.get("recipe_id") == selected_id:
            matched_in_promoted = True
            recipe_id = pc.get("recipe_id", "")
            if recipe_id and recipe_id not in matching_recipe_ids:
                matching_recipe_ids.append(recipe_id)
    pack.warm_cache_summary = {
        "promoted_candidates_count": len(promoted),
        "promoted_hit": matched_in_promoted,
        "matching_recipe_ids": matching_recipe_ids[:5],
        "promotion_retrieval_disabled_by_env": req.get(
            "promotion_retrieval_disabled_by_env", False
        ),
    }

    # --- Pass-card visibility ---
    cards_in_request = req.get("pass_cards") or []
    families = {}
    for card in cards_in_request:
        fam = card.get("family", "")
        families[fam] = families.get(fam, 0) + 1
    pack.pass_card_visibility = {
        "card_count": len(cards_in_request),
        "passes_allowed_count": len(req.get("passes_allowed") or []),
        "families": families,
        "first_pass_id": cards_in_request[0].get("pass_id", "") if cards_in_request else "",
    }

    # --- Analysis-summary index ---
    summaries = req.get("analysis_summaries") or []
    pack.analysis_summary_index = {
        "summary_count": len(summaries),
        "available_count": sum(1 for s in summaries if s.get("available")),
        "by_level": _count_by_key(summaries, "level"),
    }

    # --- Load-bearing gate attribution (G6) ---
    cost_preview = sel.get("cost_preview") or {}
    diff_path = (
        run_dir / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    diff_report = _read_json(diff_path)
    refinement_status = (
        (diff_report.get("error") or {}).get("refinement_status", "")
        if diff_report else ""
    )
    selected_label = sel.get("label", "")
    pack.load_bearing_gate_attribution = _attribute_load_bearing_gate(
        cost_preview=cost_preview,
        refinement_status=refinement_status,
        selected_label=selected_label,
    )

    # --- Persist ---
    (out_dir / "inspection_pack.json").write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "INSPECTION.md").write_text(
        _render_inspection_markdown(pack), encoding="utf-8",
    )
    return pack


def _count_by_key(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        v = item.get(key, "")
        out[v] = out.get(v, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_inspection_markdown(pack: InspectionPack) -> str:
    d = pack.decision_summary
    v = pack.validation_summary
    w = pack.warm_cache_summary
    c = pack.pass_card_visibility
    a = pack.analysis_summary_index
    rd = pack.run_dir

    lines: list[str] = []
    lines.append(f"# Inspection — `{pack.model_id}` on `{pack.target_id}`")
    lines.append("")
    lines.append(f"_Generated: `{pack.generated_at_utc}` · "
                 f"decision_seconds: `{pack.decision_seconds:.2f}` · "
                 f"outcome: **`{pack.typed_outcome}`**_")
    lines.append("")

    # Header status row
    status_emoji = {
        "verified": "✅", "typed_blocked": "⊝",
        "stopped_early": "🟡", "error": "❌",
    }.get(pack.typed_outcome, "❓")
    lines.append(f"## Status: {status_emoji} {pack.typed_outcome}")
    lines.append("")
    if pack.errors:
        lines.append("### Errors")
        for err in pack.errors:
            lines.append(f"- `{err[:300]}`")
        lines.append("")

    # Decision
    lines.append("## Greedy decision (M-37.5: warm-aware)")
    lines.append("")
    lines.append(
        f"- **Selected candidate**: `{d.get('selected_candidate_id') or '(none)'}`"
    )
    lines.append(f"- **Candidate kind / pass id**: `{d.get('candidate_kind') or '(none)'}`")
    lines.append(f"- **Region id**: `{d.get('region_id') or '(none)'}`")
    lines.append(f"- **Static relative cost**: `{d.get('static_relative_cost')}`")
    lines.append(
        f"- **Warm-cache hit**: "
        f"{'✅ yes' if d.get('warm_cache_hit') else '⊝ no'}"
    )
    lines.append("- **Rationale**:")
    lines.append("")
    lines.append(f"  > {d.get('rationale_primary', '(none)')}")
    lines.append("")

    # G6: load-bearing-gate attribution.
    attr = pack.load_bearing_gate_attribution or {}
    if attr:
        lines.append("## Load-bearing gate attribution (M-37.13 G6)")
        lines.append("")
        lines.append(
            "Which structural rule's change would flip this model's "
            "outcome — i.e. the gate that carried the load for the "
            "verified status."
        )
        lines.append("")
        lines.append(
            f"- **Verification path**: `{attr.get('verification_path')}`"
        )
        lines.append(
            f"- **Load-bearing gate**: {attr.get('load_bearing_gate')}"
        )
        lines.append(
            f"- **Shape-fit tile picked (M-37.11)**: "
            f"{'✅ yes' if attr.get('shape_fit_tile_picked') else '⊝ no'}"
        )
        sel_tile = attr.get("selected_tile") or {}
        if sel_tile:
            lines.append(
                f"- **Selected tile**: M={sel_tile.get('M')} "
                f"N={sel_tile.get('N')} K={sel_tile.get('K')}"
            )
        lines.append(
            f"- **single_k_iter**: "
            f"{'✅ yes' if attr.get('single_k_iter') else '⊝ no'}"
        )
        lines.append(
            f"- **boundary_required**: "
            f"{'⚠ yes' if attr.get('boundary_required') else '✅ no'}"
        )
        lines.append(
            f"- **clean_divide**: `{attr.get('clean_divide')}`"
        )
        lines.append(
            f"- **M-12 refinement status**: "
            f"`{attr.get('refinement_status') or '(none)'}`"
        )
        lines.append("")

    # Warm cache
    lines.append("## Warm-cache effectiveness (M-37.2)")
    lines.append("")
    lines.append(
        f"- **Promoted candidates surfaced**: `{w.get('promoted_candidates_count', 0)}`"
    )
    lines.append(
        f"- **Promoted hit**: "
        f"{'✅ yes' if w.get('promoted_hit') else '⊝ no'}"
    )
    if w.get("matching_recipe_ids"):
        lines.append("- **Matching recipe ids**:")
        for rid in w["matching_recipe_ids"]:
            lines.append(f"  - `{rid}`")
    lines.append(
        f"- **Retrieval disabled by env**: "
        f"`{w.get('promotion_retrieval_disabled_by_env', False)}`"
    )
    lines.append("")

    # Validation
    lines.append("## Agent decision validation (M-31..M-34 invariants)")
    lines.append("")
    if v:
        lines.append(
            f"- **Overall**: `{v.get('overall', 'unknown')}` "
            f"(`{v.get('pass_count', 0)}` pass / `{v.get('fail_count', 0)}` fail)"
        )
        if v.get("failures"):
            lines.append("- **Failures**:")
            for fail in v["failures"][:10]:
                detail = fail.get("detail", "")[:200]
                lines.append(f"  - `{fail.get('name')}` — {detail}")
        else:
            lines.append("- All validator rows passed.")
    else:
        lines.append("- _No validation report on disk (greedy mode does not invoke the validator)._")
    lines.append("")

    # Pass-card visibility
    lines.append("## Pass-card visibility (M-31 + M-33.6)")
    lines.append("")
    lines.append(f"- **Cards in request**: `{c.get('card_count', 0)}`")
    lines.append(f"- **Passes allowed**: `{c.get('passes_allowed_count', 0)}`")
    if c.get("families"):
        lines.append("- **Families**:")
        for fam, count in sorted(c["families"].items()):
            lines.append(f"  - `{fam}`: `{count}` card(s)")
    lines.append("")

    # Analysis summaries
    lines.append("## Analysis summaries (M-32)")
    lines.append("")
    lines.append(f"- **Total summaries known**: `{a.get('summary_count', 0)}`")
    lines.append(f"- **Available on disk**: `{a.get('available_count', 0)}`")
    if a.get("by_level"):
        lines.append("- **By level**:")
        for lvl, cnt in sorted(a["by_level"].items()):
            lines.append(f"  - `{lvl}`: `{cnt}` summaries")
    lines.append("")

    # Where to look
    lines.append("## Where to look on disk")
    lines.append("")
    lines.append(f"- **Run dir**: [`{rd.name}/`](./{rd.name}/)")
    lines.append("- **Capture stage**:")
    lines.append(f"  - [`{rd.name}/00_graph_capture/capture_report.json`]"
                 f"(./{rd.name}/00_graph_capture/capture_report.json)")
    lines.append(f"  - [`{rd.name}/00_graph_capture/dynamo_summary.json`]"
                 f"(./{rd.name}/00_graph_capture/dynamo_summary.json)")
    lines.append("- **Payload lowering**:")
    lines.append(f"  - [`{rd.name}/01_payload_lowering/lowering_summary.json`]"
                 f"(./{rd.name}/01_payload_lowering/lowering_summary.json)")
    lines.append(f"  - [`{rd.name}/01_payload_lowering/dialect_coverage.json`]"
                 f"(./{rd.name}/01_payload_lowering/dialect_coverage.json)")
    lines.append("- **Graph analysis**:")
    lines.append(f"  - [`{rd.name}/02_graph_analysis/graph_dossier_v3.json`]"
                 f"(./{rd.name}/02_graph_analysis/graph_dossier_v3.json)")
    lines.append(f"  - [`{rd.name}/02_graph_analysis/candidate_actions.json`]"
                 f"(./{rd.name}/02_graph_analysis/candidate_actions.json)")
    lines.append(f"  - [`{rd.name}/02_graph_analysis/cost_preview_v2.json`]"
                 f"(./{rd.name}/02_graph_analysis/cost_preview_v2.json)")
    lines.append("- **Recipe planning**:")
    lines.append(f"  - [`{rd.name}/03_recipe_planning/candidate_selection.json`]"
                 f"(./{rd.name}/03_recipe_planning/candidate_selection.json)")
    lines.append(f"  - [`{rd.name}/03_recipe_planning/recipe_summary.json`]"
                 f"(./{rd.name}/03_recipe_planning/recipe_summary.json)")
    lines.append(f"  - [`{rd.name}/03_recipe_planning/agent_decision/"
                 f"agent_decision_request.json`]"
                 f"(./{rd.name}/03_recipe_planning/agent_decision/"
                 f"agent_decision_request.json) — full inline 60 pass cards")
    lines.append("- **Trust audit**:")
    lines.append(f"  - [`{rd.name}/import_provenance.json`]"
                 f"(./{rd.name}/import_provenance.json)")
    lines.append(f"  - [`{rd.name}/agent_decision_trace_0000.json`]"
                 f"(./{rd.name}/agent_decision_trace_0000.json)")
    lines.append(f"  - [`{rd.name}/run_manifest.json`]"
                 f"(./{rd.name}/run_manifest.json)")
    lines.append(f"  - [`{rd.name}/stage_ledger.jsonl`]"
                 f"(./{rd.name}/stage_ledger.jsonl)")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Cross-model overview
# --------------------------------------------------------------------------- #


def aggregate_inspection_packs(
    packs: list[InspectionPack],
    *,
    out_path: Path,
) -> Path:
    """Render OVERVIEW.md comparing N model runs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# CompGen model inspection — cross-model overview")
    lines.append("")
    lines.append(f"_Generated: `{_utc_now()}`_")
    lines.append("")
    lines.append(
        f"**{len(packs)} models** — Section 20 stack on each, "
        f"with full per-model artifacts captured under `<model_id>/`."
    )
    lines.append("")
    lines.append("## Status table")
    lines.append("")
    lines.append("| Model | Outcome | Selected candidate | Pass | Warm-hit | Promoted | Cards | Summaries | Decision (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pack in packs:
        d = pack.decision_summary
        w = pack.warm_cache_summary
        c = pack.pass_card_visibility
        a = pack.analysis_summary_index
        outcome_emoji = {
            "verified": "✅",
            "typed_blocked": "⊝",
            "stopped_early": "🟡",
            "error": "❌",
        }.get(pack.typed_outcome, "❓")
        warm = "✅" if d.get("warm_cache_hit") else "⊝"
        lines.append(
            f"| [`{pack.model_id}`](./{pack.model_id}/INSPECTION.md) | "
            f"{outcome_emoji} `{pack.typed_outcome}` | "
            f"`{(d.get('selected_candidate_id') or '(none)')[:42]}` | "
            f"`{d.get('candidate_kind') or '(none)'}` | "
            f"{warm} | "
            f"{w.get('promoted_candidates_count', 0)} | "
            f"{c.get('card_count', 0)} | "
            f"{a.get('available_count', 0)}/{a.get('summary_count', 0)} | "
            f"{pack.decision_seconds:.2f} |"
        )
    lines.append("")

    # Outcomes histogram
    lines.append("## Outcomes")
    lines.append("")
    by_outcome: dict[str, list[str]] = {}
    for pack in packs:
        by_outcome.setdefault(pack.typed_outcome, []).append(pack.model_id)
    for outcome, models in sorted(by_outcome.items()):
        lines.append(f"- **{outcome}** (`{len(models)}`): "
                     + ", ".join(f"`{m}`" for m in models))
    lines.append("")

    # G6: load-bearing-gate attribution per model.
    have_attribution = any(
        pack.load_bearing_gate_attribution for pack in packs
    )
    if have_attribution:
        lines.append("## Load-bearing gate attribution (M-37.13 G6)")
        lines.append("")
        lines.append(
            "For each model, this column names the gate that admitted "
            "the verified outcome — the structural rule whose change "
            "would flip the outcome. Honest accounting of which gate "
            "carried the load makes a passing run replayable as "
            "evidence rather than a coincidence."
        )
        lines.append("")
        lines.append(
            "| Model | Verification path | Load-bearing gate | "
            "Shape-fit tile? | single_k_iter | refinement_status |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for pack in packs:
            attr = pack.load_bearing_gate_attribution
            lines.append(
                f"| `{pack.model_id}` | "
                f"`{attr.get('verification_path', '?')}` | "
                f"{attr.get('load_bearing_gate', '?')} | "
                f"{'✅' if attr.get('shape_fit_tile_picked') else '⊝'} | "
                f"{'✅' if attr.get('single_k_iter') else '⊝'} | "
                f"`{attr.get('refinement_status') or '(none)'}` |"
            )
        lines.append("")
        # Distribution summary.
        path_counts: dict[str, int] = {}
        for pack in packs:
            p = pack.load_bearing_gate_attribution.get(
                "verification_path", "?"
            )
            path_counts[p] = path_counts.get(p, 0) + 1
        lines.append("Distribution:")
        lines.append("")
        for p, n in sorted(path_counts.items()):
            lines.append(f"- `{p}`: {n} model(s)")
        lines.append("")

    # Family coverage
    lines.append("## Pass-card family coverage")
    lines.append("")
    all_families: dict[str, int] = {}
    for pack in packs:
        for fam, cnt in (pack.pass_card_visibility.get("families") or {}).items():
            all_families[fam] = all_families.get(fam, 0) + cnt
    if all_families:
        lines.append("Cards exposed across all model runs (each card seen N times = N model runs):")
        lines.append("")
        for fam, cnt in sorted(all_families.items()):
            lines.append(f"- `{fam}`: `{cnt}` card-exposures across runs")
        lines.append("")

    # Honest residuals across runs
    failures_seen: list[tuple[str, str]] = []
    for pack in packs:
        for fail in (pack.validation_summary.get("failures") or []):
            failures_seen.append((pack.model_id, fail.get("name", "")))
    if failures_seen:
        lines.append("## Validation failures (anything to investigate)")
        lines.append("")
        for model, name in failures_seen:
            lines.append(f"- `{model}` — `{name}`")
        lines.append("")
    else:
        lines.append("## Validation failures")
        lines.append("")
        lines.append(
            "_No validation failures recorded on these greedy runs. "
            "(Greedy mode exercises the structural / phase / refinement "
            "invariants but does not exercise the M-34 pass_plan rows "
            "since greedy emits no pass_plan.)_"
        )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
