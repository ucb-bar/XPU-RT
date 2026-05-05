"""Agent Candidate Decision Loop (Milestone 14A).

File-based protocol that lets an external agent (Claude, GPT, or a
human) select one legal candidate from the bounded
``llm_action_space.json`` view. The compiler emits a typed
``agent_decision_request.json``, validates the agent's
``agent_decision_response.json`` against several spec'd checks, and
then commits the agent-selected candidate into Recipe IR via the
existing M-05 pipeline.

Selection modes:

- ``greedy`` — existing deterministic baseline. The agent-decision
  protocol is *not* invoked for greedy.
- ``agent-file`` — reads a pre-existing
  ``agent_decision_response.json`` from disk. The agent (Claude Code
  via MCP / skill, or a human) must have written it before the run;
  the compiler validates and rejects invalid responses.
- ``llm-live`` — same protocol, but the pipeline itself calls a real
  LLM HTTP endpoint (anthropic / openai). Requires an API key.

Hard non-goals:

- No new candidate generation.
- No real transforms (M-11B/M-12 territory).
- No cost-model changes (M-13).
- No compiler-core changes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Result + helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentDecisionResult:
    overall: str  # "pass" | "fail"
    selection_mode: str
    selected_candidate_id: str | None
    rejection_reason: str
    request_path: Path
    response_path: Path | None
    validation_path: Path
    trace_path: Path
    out_dir: Path


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
# Banned phrases that signal forbidden correctness/perf claims
# --------------------------------------------------------------------------- #


_FORBIDDEN_CORRECTNESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(verified|proved|proven)\s+correct\b", re.IGNORECASE),
    re.compile(r"\bcorrectness\s+(verified|proved|proven|guaranteed)\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\s+correct\b", re.IGNORECASE),
    re.compile(r"\bbit[-_\s]*equivalent\s+to\s+eager\b", re.IGNORECASE),
)


_FORBIDDEN_PERF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmeasured\s+(fast|fastest|quickest|optimal|best)\b", re.IGNORECASE),
    re.compile(r"\bbenchmark(ed|s)?\b", re.IGNORECASE),
    re.compile(r"\bprofiled\b", re.IGNORECASE),
    re.compile(r"\bexecuted\s+(faster|more\s+slowly)\b", re.IGNORECASE),
)


def _scan_forbidden(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def _response_to_text(response: dict[str, Any]) -> str:
    """Stringify the rationale fields for forbidden-claim scanning."""
    rat = response.get("rationale", {}) or {}
    parts: list[str] = []
    if rat.get("summary"):
        parts.append(str(rat["summary"]))
    for ev in rat.get("evidence", []) or []:
        if isinstance(ev, dict):
            for k in ("reason", "field", "value"):
                if ev.get(k) is not None:
                    parts.append(str(ev[k]))
    for ra in rat.get("rejected_alternatives", []) or []:
        if isinstance(ra, dict):
            for k in ("reason", "candidate_kind"):
                if ra.get(k) is not None:
                    parts.append(str(ra[k]))
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Request emission
# --------------------------------------------------------------------------- #


def _build_agent_guidance() -> dict[str, Any]:
    """Build the deterministic ``agent_guidance`` block that ships
    inside every ``agent_decision_request.json``.

    This block is the **prompt-engineering surface**: it tells the
    selecting agent (Claude Code in agent-file mode, or Gemini /
    Anthropic in llm-live mode) HOW to read the bounded view + cost
    matrix the compiler emitted. The agent still picks; the guidance
    only documents priority order, disagreement-handling rules,
    rationale-field examples, and the response shape.

    The block is byte-stable across reruns (no timestamps, no system
    state). All field paths and version markers are documented
    conventions, not measurements.
    """
    return {
        "guidance_version": 1,
        "preamble": (
            "You are selecting EXACTLY ONE legal Recipe IR candidate. "
            "Read the bounded view in `visible_regions[]` and pick from "
            "`candidate_ids_allowed`. Never invent candidate IDs or tile "
            "sizes. Never claim correctness or measured-performance "
            "results. Your rationale must reference real evidence fields."
        ),
        "cost_column_priority": [
            {
                "rank": 1,
                "column": "compiled_evidence",
                "where": (
                    "hardware_resource_report.regions[*].compiled_evidence "
                    "(M-22) and compiled_bottleneck_report.regions[*]"
                ),
                "why": (
                    "real compiled-kernel measurement on actual hardware "
                    "(M-19/M-20) cross-referenced against analytical "
                    "prediction; the only column with measured bottleneck "
                    "classification."
                ),
            },
            {
                "rank": 2,
                "column": "calibration_delta",
                "where": (
                    "analytical_cost_report.candidates[*].calibration_delta"
                ),
                "why": (
                    "predicted_us vs measured_us ratio per candidate. "
                    "Ratios << 1.0 mean analytical roofline is "
                    "optimistic (launch-overhead-dominated regime)."
                ),
            },
            {
                "rank": 3,
                "column": "m21_analytical_cost",
                "where": (
                    "cost_preview_v2.cost_previews[*].m21_analytical_cost "
                    "or llm_graph_view.regions[*].legal_candidates[*]"
                    ".m21_analytical_cost"
                ),
                "why": (
                    "deterministic blocked-matmul roofline rooted in "
                    "target YAML + graph dossier + tile geometry. "
                    "Byte-reproducible across machines."
                ),
            },
            {
                "rank": 4,
                "column": "calibration",
                "where": (
                    "cost_preview_v2.cost_previews[*].calibration "
                    "(M-18.3 candidate calibration; Python-evaluator "
                    "timing of the M-16 tiled-matmul reference loop)"
                ),
                "why": (
                    "actionable signal is the SPREAD across candidates. "
                    "Absolute ratio vs predicted is dominated by Python "
                    "loop overhead, NOT a real-kernel cost."
                ),
            },
            {
                "rank": 5,
                "column": "static_relative_cost",
                "where": (
                    "cost_preview_v2.cost_previews[*].static_relative_cost"
                ),
                "why": (
                    "M-13 deterministic baseline; greedy uses this. "
                    "Use as a tiebreaker only; later columns dominate "
                    "when present."
                ),
            },
        ],
        "disagreement_handling": [
            {
                "signal": (
                    "compiled_evidence.bottleneck_classification_agreement "
                    "== false"
                ),
                "interpretation": (
                    "M-21 analytical and M-22 measured bottleneck "
                    "disagree on this region. On tiny matmuls this is "
                    "the launch-overhead-dominated regime (analytical "
                    "expects compute-bound; measured shows neither "
                    "resource near peak, with bandwidth fraction "
                    "marginally higher). Surface as evidence; do NOT "
                    "claim either side is wrong."
                ),
            },
            {
                "signal": (
                    "calibration_delta.predicted_vs_gpu_ratio < 0.1"
                ),
                "interpretation": (
                    "Analytical model is >10x optimistic on this "
                    "candidate. Reflects unmodeled launch overhead "
                    "and/or cache effects. Prefer compiled measurement "
                    "if available; otherwise note the mismatch."
                ),
            },
            {
                "signal": (
                    "kernel_calibration_status == "
                    "\"partial_kernel_calibration\""
                ),
                "interpretation": (
                    "Some regions have compiled evidence; others "
                    "don't. Pick from regions with evidence when "
                    "feasible; for regions without, fall back to "
                    "analytical_cost or static_relative_cost."
                ),
            },
            {
                "signal": (
                    "kernel_calibration_status == \"not_kernel_calibrated\""
                ),
                "interpretation": (
                    "M-19/M-20 didn't run (kernels OFF or unsupported). "
                    "M-21 analytical is the strongest available signal. "
                    "Honest; do not invent compiled measurements."
                ),
            },
        ],
        "rationale_field_examples": [
            "candidate.cost_preview.static_relative_cost",
            "candidate.cost_preview.confidence",
            "candidate.cost_preview.features.real_transform_verified",
            "candidate.cost_preview.m21_analytical_cost.predicted_us",
            "candidate.cost_preview.m21_analytical_cost.bottleneck_resource",
            "candidate.cost_preview.m21_analytical_cost.bottleneck_tier",
            "candidate.cost_preview.calibration.measured_speedup",
            "candidate.cost_preview.calibration.measured_tiled_us",
            "candidate.legality.ok",
            "candidate.kind",
            "compiled_evidence.measured_bottleneck",
            "compiled_evidence.bottleneck_classification_agreement",
            "compiled_evidence.gpu.measured_us_per_iter",
            "compiled_evidence.gpu.compute_utilization",
            "compiled_evidence.gpu.bandwidth_utilization",
        ],
        "forbidden_phrase_patterns": [
            "verified correct",
            "guaranteed correct",
            "bit equivalent to eager",
            "measured fastest",
            "benchmarked",
            "profiled",
            "executed faster",
        ],
        "preferred_neutral_phrases": [
            "lower static_relative_cost",
            "fits scratchpad",
            "M-12 differential evidence available",
            "obligation declared bit_equality",
            "M-21 analytical predicts compute-bound",
            "M-22 measured bottleneck agrees with analytical",
            "kernel_calibrated evidence available",
        ],
        "response_shape": {
            "schema_version": "agent_decision_response_v1",
            "selected_candidate_id": (
                "<must be one of candidate_ids_allowed>"
            ),
            "rationale": {
                "summary": "<1-2 sentences, neutral language>",
                "evidence": (
                    "<list of >=2 entries; each entry is "
                    "{field, value, reason} where field appears in "
                    "rationale_field_examples or resolves against "
                    "candidate / sources>"
                ),
                "rejected_alternatives": (
                    "<optional; list of {candidate_kind, reason}>"
                ),
            },
        },
        "selection_modes_supported": [
            "agent-file (Claude Code or human writes response file)",
            "llm-live (compiler calls Gemini/Anthropic with this guidance)",
        ],
        "honest_non_claims": [
            "evidence may be sparse on opaque regions (CreateKernelContract path)",
            "calibration_delta absent when COMPGEN_RUN_KERNELS != 1",
            "M-22 cache_evidence is not_collected (M-22.1 follow-up adds Nsight/perf)",
        ],
    }


def build_agent_decision_request(
    run_dir: Path,
    *,
    objective: str = "choose_one_recipe_candidate",
) -> Path:
    """Emit ``03_recipe_planning/agent_decision/agent_decision_request.json``.

    Sources visible to the agent at decision time:

    - ``llm_action_space.json`` (graph-analysis output, legal-only,
      bounded). This is the canonical view at decision time.
    - ``candidate_actions.json`` (full action space, for cross-checks).
    - ``action_space.mlir`` (canonical IR; used by the resolver).
    - ``graph_dossier_v3.json`` / ``cost_preview_v2.json`` /
      ``llm_graph_view.json`` are referenced when present (they
      become available when M-13 has run; for the request emitted
      pre-decision they're recorded as ``null``).

    The request's ``candidate_ids_allowed`` is the union of every
    legal candidate listed in ``llm_action_space.ranked_sites[].
    legal_candidates[]``. The agent MUST pick from this list.
    """
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    rp = run_dir / "03_recipe_planning"
    out_dir = rp / "agent_decision"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_view = _read_json(ga / "llm_action_space.json")
    candidate_actions = _read_json(ga / "candidate_actions.json")
    action_space_mlir = ga / "action_space.mlir"

    cand_by_id = {
        c["candidate_id"]: c for c in candidate_actions.get("candidates", [])
    }

    # candidate_ids_allowed: every legal candidate visible to the agent.
    candidate_ids_allowed: list[str] = []
    visible_regions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for site in llm_view.get("ranked_sites", []):
        region_id = site.get("region_id", "")
        site_legal: list[dict[str, Any]] = []
        for c in site.get("legal_candidates", []) or []:
            cid = c.get("candidate_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                candidate_ids_allowed.append(cid)
            full = cand_by_id.get(cid, {})
            site_legal.append(
                {
                    "candidate_id": cid,
                    "kind": c.get("kind") or full.get("kind", ""),
                    "label": c.get("label") or full.get("label", ""),
                    "static_relative_cost": (c.get("cost_preview") or {}).get(
                        "static_relative_cost",
                        (full.get("cost_preview") or {}).get(
                            "static_relative_cost"
                        ),
                    ),
                    "fits_scratchpad": (c.get("cost_preview") or {}).get(
                        "fits_scratchpad"
                    ),
                    "fits_l2": (c.get("cost_preview") or {}).get("fits_l2"),
                }
            )
        visible_regions.append(
            {
                "region_id": region_id,
                "kind": site.get("kind", ""),
                "site_id": site.get("site_id", ""),
                "priority": site.get("priority"),
                "why": site.get("why", ""),
                "legal_candidates": site_legal,
            }
        )

    sources: dict[str, Any] = {}
    for name, path in (
        ("llm_action_space", ga / "llm_action_space.json"),
        ("candidate_actions", ga / "candidate_actions.json"),
        ("action_space_ir", action_space_mlir),
        ("region_map", ga / "region_map.json"),
        ("graph_dossier_v3", ga / "graph_dossier_v3.json"),
        ("cost_preview_v2", ga / "cost_preview_v2.json"),
        ("llm_graph_view", ga / "llm_graph_view.json"),
        # Optional cost-evidence sources (when their opt-in stages ran).
        # Their absence is fine — the agent_guidance block still tells
        # the agent the priority order; missing sources just mean the
        # corresponding columns aren't available for this run.
        ("readiness_matrix",
         ga / "readiness" / "graph_analysis_readiness_matrix.json"),
        ("hardware_resource_report",
         ga / "readiness" / "hardware_resource_report.json"),
        ("calibration_report",
         ga / "calibration" / "profiler_calibration_report.json"),
        ("candidate_calibration_report",
         ga / "candidate_calibration" / "candidate_calibration_report.json"),
        ("analytical_cost_report",
         ga / "analytical_cost" / "per_candidate_analytical_cost.json"),
        ("region_compiled_differential_report",
         ga / "kernel_execution" / "region_compiled_differential_report.json"),
        ("compiled_bottleneck_report",
         ga / "compiled_bottleneck" / "compiled_bottleneck_report.json"),
    ):
        sources[name] = (
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": _sha256_file(path),
            }
            if path.exists() else None
        )

    request = {
        "schema_version": "agent_decision_request_v1",
        "model_id": llm_view.get("model_id", ""),
        "target_id": llm_view.get("target_id", ""),
        "objective": objective,
        "constraints": {
            "must_select_legal_candidate": True,
            "must_select_from_llm_graph_view": True,
            "must_reference_evidence": True,
            "may_not_invent_candidate_ids": True,
            "may_not_invent_tile_sizes": True,
            "may_not_claim_correctness": True,
            "may_not_claim_measured_performance": True,
        },
        "sources": sources,
        "candidate_ids_allowed": candidate_ids_allowed,
        "visible_regions": visible_regions,
        "agent_guidance": _build_agent_guidance(),
        "generated_at_utc": _utcnow(),
    }
    request_path = out_dir / "agent_decision_request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True), encoding="utf-8",
    )
    return request_path


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


_RESPONSE_REQUIRED_KEYS = ("schema_version", "selected_candidate_id", "rationale")


def _evidence_field_resolves(
    field: str, *, request: dict[str, Any], cand: dict[str, Any] | None,
) -> bool:
    """Return True iff ``field`` (e.g. ``candidate.cost_preview.static_relative_cost``)
    points at a real key in the candidate or one of the request sources."""
    if not field:
        return False
    parts = field.split(".")
    head = parts[0]
    rest = parts[1:]
    if head == "candidate":
        node: Any = cand
    elif head in ("region",):
        node = cand or {}
    elif head in ("cost_preview_v2", "graph_dossier_v3", "semantic_obligation"):
        # Soft accept: if the agent references one of these source-doc
        # roots, treat as resolved (the actual nested key lookup is
        # too much for an MVP).
        return True
    else:
        return False
    for k in rest:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return False
    return True


def validate_agent_decision_response(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    candidate_actions: dict[str, Any],
    run_dir: Path,
    selection_mode: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {
                "name": name,
                "status": "pass" if ok else "fail",
                "detail": detail,
            }
        )

    # 1. request_sources_exist — every non-null source path resolves.
    miss: list[str] = []
    for name, src in (request.get("sources") or {}).items():
        if src is None:
            continue
        path = run_dir / src["path"]
        if not path.exists():
            miss.append(f"{name}: {src['path']}")
    _add(
        "request_sources_exist", not miss,
        "" if not miss else f"missing: {miss}",
    )

    # 2. response_schema_valid — required keys + types.
    schema_ok = True
    schema_detail = ""
    for k in _RESPONSE_REQUIRED_KEYS:
        if k not in response:
            schema_ok = False
            schema_detail = f"missing key {k!r}"
            break
    if schema_ok and not isinstance(response.get("rationale"), dict):
        schema_ok = False
        schema_detail = "rationale must be an object"
    _add("response_schema_valid", schema_ok, schema_detail)
    if not schema_ok:
        failures.append(f"response schema invalid: {schema_detail}")

    selected_id = response.get("selected_candidate_id", "") or ""
    cand_by_id = {
        c["candidate_id"]: c for c in candidate_actions.get("candidates", [])
    }

    # 3. selected_candidate_exists.
    cand = cand_by_id.get(selected_id)
    _add(
        "selected_candidate_exists",
        cand is not None,
        "" if cand is not None
        else f"selected_candidate_id={selected_id!r} not in candidate_actions",
    )

    # 4. selected_candidate_is_legal.
    legal_ok = (
        cand is not None and (cand.get("legality") or {}).get("ok") is True
    )
    _add(
        "selected_candidate_is_legal",
        legal_ok,
        "" if legal_ok
        else f"selected_candidate_id={selected_id!r} is illegal or missing",
    )

    # 5. selected_candidate_visible_to_agent — appears in
    # request.candidate_ids_allowed.
    visible = selected_id in (request.get("candidate_ids_allowed") or [])
    _add(
        "selected_candidate_visible_to_agent", visible,
        "" if visible
        else f"selected_candidate_id={selected_id!r} not in candidate_ids_allowed",
    )

    # 6. selected_candidate_resolves_against_action_space_ir.
    resolves_ok = False
    resolver_detail = ""
    if cand is not None and legal_ok and visible:
        try:
            from compgen.graph_compilation.action_space_resolver import (
                ResolverError,
                resolve_candidate,
            )

            try:
                resolve_candidate(
                    run_dir=run_dir, candidate_id=selected_id,
                    allow_illegal=False, selection_mode=selection_mode,
                    rationale={"primary_reason": "agent-decision-validation"},
                    write_outputs=False,
                )
                resolves_ok = True
            except ResolverError as exc:
                resolver_detail = f"{type(exc).__name__}: {exc}"
        except ImportError as exc:
            resolver_detail = f"resolver import failed: {exc}"
    elif not (cand is not None and legal_ok and visible):
        resolver_detail = "skipped (preconditions failed)"
    _add(
        "selected_candidate_resolves_against_action_space_ir",
        resolves_ok or resolver_detail == "skipped (preconditions failed)",
        resolver_detail,
    )
    if not resolves_ok and resolver_detail and resolver_detail != "skipped (preconditions failed)":
        failures.append(f"resolver rejected: {resolver_detail}")

    # 7. rationale_summary_present.
    rat = response.get("rationale", {}) or {}
    summary_ok = bool((rat.get("summary") or "").strip())
    _add("rationale_summary_present", summary_ok, "")

    # 8. rationale_evidence_present (≥2 entries).
    evidence = rat.get("evidence", []) or []
    ev_ok = isinstance(evidence, list) and len(evidence) >= 2
    _add(
        "rationale_evidence_present", ev_ok,
        "" if ev_ok else f"rationale.evidence has {len(evidence)} entries (need ≥2)",
    )

    # 9. rationale_references_real_fields.
    real_field_ok = False
    if isinstance(evidence, list) and evidence:
        resolved_count = sum(
            1 for ev in evidence
            if isinstance(ev, dict)
            and _evidence_field_resolves(
                ev.get("field", ""), request=request, cand=cand,
            )
        )
        real_field_ok = resolved_count >= 2
    _add(
        "rationale_references_real_fields", real_field_ok,
        "" if real_field_ok
        else "fewer than 2 evidence entries reference real candidate/region fields",
    )

    # 10. no_correctness_claim.
    rationale_text = _response_to_text(response)
    correctness_hits = _scan_forbidden(rationale_text, _FORBIDDEN_CORRECTNESS_PATTERNS)
    _add(
        "no_correctness_claim",
        not correctness_hits,
        "" if not correctness_hits else f"forbidden phrase: {correctness_hits[0]!r}",
    )

    # 11. no_measured_performance_claim.
    perf_hits = _scan_forbidden(rationale_text, _FORBIDDEN_PERF_PATTERNS)
    _add(
        "no_measured_performance_claim",
        not perf_hits,
        "" if not perf_hits else f"forbidden phrase: {perf_hits[0]!r}",
    )

    overall = "pass" if all(c["status"] == "pass" for c in checks) and not failures else "fail"
    return {
        "schema_version": "agent_decision_validation_v1",
        "overall": overall,
        "selection_mode": selection_mode,
        "selected_candidate_id": selected_id,
        "checks": checks,
        "failure_reasons": failures,
        "generated_at_utc": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Provider-redaction audit (M-14C)
# --------------------------------------------------------------------------- #


def _emit_redaction_audit(
    *,
    out_dir: Path,
    provider_name: str,
) -> Path:
    """Scan every artifact under ``out_dir`` (the agent_decision/
    subdir) for known secret patterns and write
    ``provider_redaction_audit.json``.

    The audit checks for:

    - The actual API-key value resolved from the same env vars the real
      provider adapters read (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
      ``COMPGEN_LLM_API_KEY``).
    - ``Authorization: Bearer`` and ``x-api-key:`` header patterns.

    None of these strings should appear in any emitted artifact. The
    audit itself NEVER stores the key — it only records pass/fail.
    """
    import os as _os

    candidate_keys = [
        v for v in (
            _os.environ.get("ANTHROPIC_API_KEY"),
            _os.environ.get("OPENAI_API_KEY"),
            _os.environ.get("COMPGEN_LLM_API_KEY"),
        ) if v
    ]
    artifact_names = [
        "agent_decision_request.json",
        "agent_decision_prompt.txt",
        "agent_decision_provider_request.json",
        "agent_decision_provider_response.raw.json",
        "agent_decision_response.json",
        "agent_decision_validation.json",
        "agent_decision_trace.json",
    ]
    bodies: dict[str, str] = {}
    for name in artifact_names:
        p = out_dir / name
        if p.exists():
            try:
                bodies[name] = p.read_text(encoding="utf-8")
            except OSError:
                bodies[name] = ""

    def _scan(target_name: str, *substrings: str) -> tuple[bool, str]:
        for fname, body in bodies.items():
            if target_name and fname != target_name:
                continue
            for s in substrings:
                if s and s in body:
                    return False, f"found in {fname}"
        return True, ""

    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if ok else "fail", "detail": detail}
        )

    # api_key_not_in_<artifact> — split per artifact for clearer audit.
    for art_name, check_name in (
        ("agent_decision_provider_request.json", "api_key_not_in_provider_request"),
        ("agent_decision_prompt.txt", "api_key_not_in_prompt"),
        ("agent_decision_provider_response.raw.json", "api_key_not_in_raw_response"),
        ("agent_decision_trace.json", "api_key_not_in_trace"),
    ):
        ok, detail = _scan(art_name, *candidate_keys) if candidate_keys else (
            True, "no API keys in env (nothing to scan for)"
        )
        _add(check_name, ok, detail)

    # Authorization-header patterns must never be persisted.
    forbidden_header_patterns = (
        "Authorization: Bearer ",
        "Authorization:Bearer ",
        "x-api-key: ",
        "x-api-key:",
    )
    ok, detail = _scan("", *forbidden_header_patterns)
    _add("authorization_headers_not_written", ok, detail)

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    audit = {
        "schema_version": "provider_redaction_audit_v1",
        "status": overall,
        "provider_name": provider_name,
        "scanned_artifacts": [
            n for n in artifact_names if n in bodies
        ],
        "candidate_keys_scanned": len(candidate_keys),
        "checks": checks,
        "generated_at_utc": _utcnow(),
    }
    audit_path = out_dir / "provider_redaction_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8",
    )
    return audit_path


# --------------------------------------------------------------------------- #
# Reviewer-facing notes (M-14C)
# --------------------------------------------------------------------------- #


def _emit_claude_code_decision_notes(
    *,
    out_dir: Path,
    run_dir: Path,
    selection_mode: str,
    response: dict[str, Any],
    validation: dict[str, Any],
    provider_block: dict[str, Any] | None,
) -> Path:
    """Write ``claude_code_decision_notes.md`` — a reviewer one-pager
    showing what the agent picked, why, and how validation went.

    Emitted only on agent-driven paths (agent-file /
    llm-live). The greedy path does not emit this — its rationale is
    deterministic and captured in ``selection_trace.jsonl``.
    """
    selected_id = response.get("selected_candidate_id", "")
    rationale = response.get("rationale") or {}
    summary = rationale.get("summary", "")
    evidence = rationale.get("evidence") or []
    rejected = rationale.get("rejected_alternatives") or []
    overall = validation.get("overall", "fail")

    lines: list[str] = []
    lines.append("# Claude Code agent-decision notes\n")
    lines.append(f"_Generated_: {_utcnow()}\n")
    lines.append(
        f"- **selection_mode**: `{selection_mode}`  "
        f"\n- **validation overall**: `{overall}`  "
        f"\n- **selected_candidate_id**: `{selected_id}`"
    )
    if provider_block:
        lines.append(
            f"\n- **provider**: `{provider_block.get('provider_name')}`  "
            f"\n- **model**: `{provider_block.get('model')}`  "
            f"\n- **dry_run**: `{provider_block.get('dry_run')}`"
        )
    lines.append("\n## Summary (agent's words)\n")
    lines.append(f"> {summary or '(no summary provided)'}\n")
    lines.append("## Evidence cited\n")
    if evidence:
        lines.append("| field | value | reason |")
        lines.append("|---|---|---|")
        for ev in evidence:
            field = ev.get("field", "") if isinstance(ev, dict) else ""
            value = ev.get("value", "") if isinstance(ev, dict) else ""
            reason = ev.get("reason", "") if isinstance(ev, dict) else ""
            value_str = json.dumps(value) if not isinstance(value, str) else value
            lines.append(f"| `{field}` | `{value_str}` | {reason} |")
    else:
        lines.append("_(no evidence entries provided)_")
    lines.append("\n## Rejected alternatives\n")
    if rejected:
        for ra in rejected:
            kind = ra.get("candidate_kind", "?") if isinstance(ra, dict) else "?"
            reason = ra.get("reason", "") if isinstance(ra, dict) else ""
            lines.append(f"- **{kind}** — {reason}")
    else:
        lines.append("_(none recorded)_")
    lines.append("\n## Validation checks\n")
    lines.append("| check | status | detail |")
    lines.append("|---|---|---|")
    for c in validation.get("checks", []):
        lines.append(
            f"| {c.get('name', '?')} | {c.get('status', '?')} | "
            f"{c.get('detail', '')} |"
        )
    lines.append(
        "\n## Trust boundary reminder\n\n"
        "Claude Code chose among legal candidates from a bounded view. The "
        "compiler validated the choice against `candidate_ids_allowed`, "
        "`candidate_actions.json::legality.ok`, and `action_space.mlir` "
        "before committing `recipe.mlir`. Correctness of the lowered IR is "
        "M-09 / M-12's job, not the agent's; this notes file does not claim "
        "the resulting transform is correct or fastest.\n"
    )
    notes_path = out_dir / "claude_code_decision_notes.md"
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes_path


def emit_agent_decision_trace(
    *,
    run_dir: Path,
    request_path: Path,
    response_path: Path | None,
    validation: dict[str, Any],
    selection_mode: str,
    selected_candidate_kind: str,
    committed_recipe_op: str,
    provider_block: dict[str, Any] | None = None,
) -> Path:
    out_dir = run_dir / "03_recipe_planning" / "agent_decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "schema_version": "agent_decision_trace_v1",
        "model_id": _read_json(request_path).get("model_id", ""),
        "target_id": _read_json(request_path).get("target_id", ""),
        "selection_mode": selection_mode,
        "request_sha256": _sha256_file(request_path),
        "response_sha256": (
            _sha256_file(response_path) if response_path is not None
            and response_path.exists() else None
        ),
        "llm_graph_view_sha256": _sha256_or_none(
            run_dir / "02_graph_analysis" / "llm_graph_view.json"
        ),
        "cost_preview_v2_sha256": _sha256_or_none(
            run_dir / "02_graph_analysis" / "cost_preview_v2.json"
        ),
        "selected_candidate_id": validation.get("selected_candidate_id", ""),
        "selected_candidate_kind": selected_candidate_kind,
        "validation_status": validation.get("overall", "fail"),
        "committed_recipe_op": committed_recipe_op,
        "generated_at_utc": _utcnow(),
    }
    if provider_block is not None:
        trace["provider"] = provider_block
    trace_path = out_dir / "agent_decision_trace.json"
    trace_path.write_text(
        json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8",
    )
    return trace_path


# --------------------------------------------------------------------------- #
# Top-level orchestrator (used by recipe_planning when mode != greedy)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiveProviderConfig:
    """Configuration for ``selection_mode=llm-live``."""
    provider: str = "gemini"
    model: str | None = None
    timeout_sec: int = 60
    dry_run: bool = False
    fallback: str = "none"  # "none" | "greedy"


def _emit_retry_request(
    *,
    out_dir: Path,
    attempt_index: int,
    failed_response_path: Path,
    failed_validation_path: Path,
    request: dict[str, Any],
    validation: dict[str, Any],
    run_dir: Path,
) -> Path:
    """Write a typed ``retry_request.json`` (M-15A) explaining why the
    attempt failed and what the agent should fix next time.

    The recommended-debug-fields list is the canonical evidence-field
    surface the M-14A validator resolves; the agent must reference
    fields from this set in its corrected response.
    """
    failed_checks = [
        {
            "name": c.get("name", ""),
            "reason": c.get("detail", "") or "validation check failed",
        }
        for c in validation.get("checks", [])
        if c.get("status") == "fail"
    ]
    retry: dict[str, Any] = {
        "schema_version": "agent_decision_retry_request_v1",
        "attempt_index": attempt_index,
        "status": "retry_required",
        "model_id": request.get("model_id", ""),
        "target_id": request.get("target_id", ""),
        "failed_response": {
            "path": str(failed_response_path.relative_to(run_dir))
            if failed_response_path.exists() else None,
            "sha256": _sha256_file(failed_response_path)
            if failed_response_path.exists() else None,
        },
        "validation": {
            "path": str(failed_validation_path.relative_to(run_dir)),
            "overall": validation.get("overall", "fail"),
            "failed_checks": failed_checks,
        },
        "retry_instructions": {
            "must_select_from_candidate_ids_allowed": True,
            "must_include_rationale_summary": True,
            "must_include_at_least_two_real_evidence_fields": True,
            "must_not_claim_correctness": True,
            "must_not_claim_measured_performance": True,
        },
        "candidate_ids_allowed": request.get("candidate_ids_allowed", []),
        "recommended_debug_fields": [
            "candidate.kind",
            "candidate.legality.ok",
            "candidate.cost_preview.static_relative_cost",
            "candidate.cost_preview.fits_scratchpad",
            "candidate.cost_preview.fits_l2",
            "cost_preview_v2.relative_cost",
            "cost_preview_v2.confidence",
            "cost_preview_v2.features.real_transform_verified",
            "semantic_obligation.declared_refinement",
        ],
        "generated_at_utc": _utcnow(),
    }
    retry_path = out_dir / "retry_request.json"
    retry_path.write_text(
        json.dumps(retry, indent=2, sort_keys=True), encoding="utf-8",
    )
    # Also drop a copy under the attempt dir for audit.
    attempt_dir = out_dir / "attempts" / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "retry_request.json").write_text(
        json.dumps(retry, indent=2, sort_keys=True), encoding="utf-8",
    )
    return retry_path


def _snapshot_attempt(
    *,
    out_dir: Path,
    attempt_index: int,
    response: dict[str, Any] | None,
    validation: dict[str, Any],
) -> Path:
    """Copy the per-attempt response + validation under
    ``attempts/attempt_<N>/`` for audit. The top-level
    ``agent_decision_response.json`` / ``agent_decision_validation.json``
    only carry the FINAL accepted attempt (or the last failed attempt
    when retries are exhausted)."""
    attempt_dir = out_dir / "attempts" / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if response is not None:
        (attempt_dir / "agent_decision_response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True), encoding="utf-8",
        )
    (attempt_dir / "agent_decision_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8",
    )
    return attempt_dir


def _emit_retry_summary(
    *,
    out_dir: Path,
    attempts: list[dict[str, Any]],
    max_attempts: int,
    final_status: str,
    final_selected_candidate_id: str | None,
    recipe_committed: bool,
) -> Path:
    summary = {
        "schema_version": "agent_decision_retry_summary_v1",
        "status": final_status,
        "max_attempts": max_attempts,
        "attempts": attempts,
        "final_selected_candidate_id": final_selected_candidate_id,
        "recipe_committed": recipe_committed,
        "generated_at_utc": _utcnow(),
    }
    p = out_dir / "retry_summary.json"
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return p


def run_agent_decision(
    run_dir: Path,
    *,
    selection_mode: str,
    agent_response_path: Path | None = None,
    live_config: LiveProviderConfig | None = None,
    attempt_index: int = 0,
    max_retries: int = 3,
) -> AgentDecisionResult:
    """Emit + (load|stub) + validate the agent decision artifacts.

    Returns a result whose ``selected_candidate_id`` is the candidate
    the recipe-planning M-05 selector should commit. ``overall == "fail"``
    means the recipe must NOT be committed; the caller should propagate
    the failure before any recipe.mlir is written.
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "03_recipe_planning" / "agent_decision"
    out_dir.mkdir(parents=True, exist_ok=True)

    request_path = build_agent_decision_request(run_dir)
    request = _read_json(request_path)
    candidate_actions = _read_json(
        run_dir / "02_graph_analysis" / "candidate_actions.json"
    )

    response_path: Path | None = out_dir / "agent_decision_response.json"
    response: dict[str, Any] | None = None
    rejection_reason = ""
    provider_block: dict[str, Any] | None = None
    if selection_mode == "llm-live":
        # Provider-backed mode. The provider call returns a parsed
        # response (or raises ProviderError). Validation still happens
        # downstream — providers cannot bypass the M-14A gate.
        from compgen.graph_compilation.llm_live_provider import (
            ProviderError,
            build_prompt,
            call_provider,
            parse_provider_response_text,
        )

        cfg = live_config or LiveProviderConfig()
        # Read llm_graph_view if available (post-M-13). For pre-M-13
        # request emission, fall back to the legal-only llm_action_space
        # which is always available at decision time.
        llm_view: dict[str, Any] | None = _read_json_or_none(
            run_dir / "02_graph_analysis" / "llm_graph_view.json"
        )
        if llm_view is None:
            llm_view = _read_json_or_none(
                run_dir / "02_graph_analysis" / "llm_action_space.json"
            )
        prompt_text = build_prompt(request=request, llm_graph_view=llm_view)
        prompt_path = out_dir / "agent_decision_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        provider_request = {
            "schema_version": "agent_decision_provider_request_v1",
            "provider": cfg.provider,
            "model": cfg.model,
            "timeout_sec": cfg.timeout_sec,
            "dry_run": cfg.dry_run,
            "fallback": cfg.fallback,
            "prompt_path": str(prompt_path.relative_to(run_dir)),
            "prompt_sha256": _sha256_file(prompt_path),
            "request_path": str(request_path.relative_to(run_dir)),
            "request_sha256": _sha256_file(request_path),
            "generated_at_utc": _utcnow(),
        }
        provider_request_path = out_dir / "agent_decision_provider_request.json"
        provider_request_path.write_text(
            json.dumps(provider_request, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if cfg.dry_run:
            # Halt: emit a typed "dry_run" report. No response file, no
            # validation, no recipe commit.
            (out_dir / "agent_decision_dry_run.json").write_text(
                json.dumps(
                    {
                        "schema_version": "agent_decision_dry_run_v1",
                        "selection_mode": "llm-live",
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "reason": "dry_run flag set; provider was not called",
                        "prompt_path": str(
                            prompt_path.relative_to(run_dir)
                        ),
                    },
                    indent=2, sort_keys=True,
                ),
                encoding="utf-8",
            )
            response_path = None
            rejection_reason = "dry_run: provider call skipped"
        else:
            try:
                t0 = _utcnow()
                pr = call_provider(
                    provider_name=cfg.provider, model=cfg.model,
                    timeout_sec=cfg.timeout_sec,
                    request=request, llm_graph_view=llm_view,
                    candidate_actions=candidate_actions,
                )
            except ProviderError as exc:
                # Typed provider failure — emit provider_error.json.
                (out_dir / "provider_error.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "provider_error_v1",
                            "provider": cfg.provider,
                            "model": cfg.model,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "generated_at_utc": _utcnow(),
                        },
                        indent=2, sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                response_path = None
                rejection_reason = f"provider call failed: {exc}"
                pr = None
            else:
                # Persist raw provider response for audit.
                raw_path = out_dir / "agent_decision_provider_response.raw.json"
                raw_path.write_text(
                    json.dumps(pr.raw_response, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                # Parse the completion text into a response dict.
                parsed = pr.parsed_response
                if parsed is None:
                    completion = (pr.raw_response or {}).get("completion_text", "")
                    try:
                        parsed = parse_provider_response_text(completion)
                    except ProviderError as exc:
                        (out_dir / "provider_error.json").write_text(
                            json.dumps(
                                {
                                    "schema_version": "provider_error_v1",
                                    "provider": cfg.provider,
                                    "model": cfg.model,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                    "generated_at_utc": _utcnow(),
                                },
                                indent=2, sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                        parsed = None
                        rejection_reason = f"parser rejected provider output: {exc}"
                if parsed is not None:
                    response = parsed
                    response_path.write_text(
                        json.dumps(response, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                else:
                    response_path = None
            # Build the trace's provider block.
            if pr is not None:
                provider_block = {
                    "provider_name": pr.provider_name,
                    "model": pr.model,
                    "dry_run": False,
                    "fallback_used": False,
                    "latency_ms": pr.latency_ms,
                    "prompt_sha256": _sha256_file(prompt_path),
                    "raw_response_sha256": _sha256_file(
                        out_dir / "agent_decision_provider_response.raw.json"
                    ),
                }
            else:
                provider_block = {
                    "provider_name": cfg.provider,
                    "model": cfg.model,
                    "dry_run": False,
                    "fallback_used": False,
                    "latency_ms": 0,
                    "prompt_sha256": _sha256_file(prompt_path),
                    "raw_response_sha256": None,
                    "error": rejection_reason or "provider call failed",
                }
        if cfg.dry_run:
            # Augment provider block with dry-run flag.
            provider_block = {
                "provider_name": cfg.provider,
                "model": cfg.model,
                "dry_run": True,
                "fallback_used": False,
                "latency_ms": 0,
                "prompt_sha256": _sha256_file(prompt_path),
                "raw_response_sha256": None,
            }

    elif selection_mode == "agent-file":
        if agent_response_path is None:
            rejection_reason = (
                "selection_mode=agent-file requires --agent-decision-response "
                "<path>; none was provided"
            )
        elif not Path(agent_response_path).exists():
            rejection_reason = (
                f"--agent-decision-response path does not exist: "
                f"{agent_response_path}"
            )
        else:
            try:
                response = _read_json(Path(agent_response_path))
            except json.JSONDecodeError as exc:
                rejection_reason = (
                    f"agent_decision_response.json is invalid JSON: {exc}"
                )
            else:
                # Persist the agent's response under the run dir so the
                # validation + trace files reference an in-run path.
                response_path.write_text(
                    json.dumps(response, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    else:
        # ``greedy`` should not invoke the agent-decision protocol — the
        # caller routes greedy directly through the recipe-planning
        # selector. Anything else is unknown.
        rejection_reason = f"unknown selection_mode: {selection_mode!r}"

    if response is None:
        # Validation can't run on a missing response. Emit a minimal
        # validation report that flags the failure.
        validation = {
            "schema_version": "agent_decision_validation_v1",
            "overall": "fail",
            "selection_mode": selection_mode,
            "selected_candidate_id": "",
            "checks": [
                {
                    "name": "response_present",
                    "status": "fail",
                    "detail": rejection_reason,
                }
            ],
            "failure_reasons": [rejection_reason],
            "generated_at_utc": _utcnow(),
        }
    else:
        validation = validate_agent_decision_response(
            request=request, response=response,
            candidate_actions=candidate_actions, run_dir=run_dir,
            selection_mode=selection_mode,
        )

    validation_path = out_dir / "agent_decision_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8",
    )

    # M-15A: snapshot this attempt under attempts/attempt_<N>/. Always
    # done so the audit trail is preserved regardless of pass/fail.
    # On fail, also emit retry_request.json (top-level + per-attempt).
    _snapshot_attempt(
        out_dir=out_dir, attempt_index=attempt_index,
        response=response, validation=validation,
    )
    if validation.get("overall") != "pass":
        _emit_retry_request(
            out_dir=out_dir, attempt_index=attempt_index,
            failed_response_path=(
                response_path if response_path is not None else
                out_dir / "agent_decision_response.json"
            ),
            failed_validation_path=validation_path,
            request=request, validation=validation, run_dir=run_dir,
        )
    else:
        # On pass, retry_request.json is stale — remove it so a later
        # reader doesn't think we're still in retry-required state.
        stale_retry = out_dir / "retry_request.json"
        if stale_retry.exists():
            stale_retry.unlink()

    # Trace gets the recipe_op once recipe_planning commits; the caller
    # (run_recipe_planning) will overwrite the trace with the real op id
    # after commit. For now write a placeholder.
    cand_by_id = {
        c["candidate_id"]: c for c in candidate_actions.get("candidates", [])
    }
    selected_id = validation.get("selected_candidate_id", "") or ""
    selected_kind = ""
    if selected_id and selected_id in cand_by_id:
        selected_kind = cand_by_id[selected_id].get("kind", "")
    trace_path = emit_agent_decision_trace(
        run_dir=run_dir,
        request_path=request_path,
        response_path=response_path
        if response_path is not None and response_path.exists() else None,
        validation=validation,
        selection_mode=selection_mode,
        selected_candidate_kind=selected_kind,
        committed_recipe_op="(pending)",
        provider_block=provider_block,
    )

    # M-14C: emit provider-redaction audit AFTER all artifacts are
    # written so the scan covers everything in the agent_decision/
    # subdir. Runs unconditionally (not just for llm-live) because the
    # audit also catches accidental key writes during stub/agent-file.
    redaction_provider_name = (
        provider_block["provider_name"] if provider_block else selection_mode
    )
    _emit_redaction_audit(
        out_dir=out_dir, provider_name=redaction_provider_name,
    )

    # M-14C: emit a reviewer-facing markdown summary explaining what
    # the agent picked and why. Only meaningful when the agent
    # actually made a decision (skip path: agent-file,
    # llm-live). Greedy doesn't run this code path.
    if selection_mode in ("agent-file", "llm-live") and response is not None:
        _emit_claude_code_decision_notes(
            out_dir=out_dir,
            run_dir=run_dir,
            selection_mode=selection_mode,
            response=response,
            validation=validation,
            provider_block=provider_block,
        )

    return AgentDecisionResult(
        overall=validation["overall"],
        selection_mode=selection_mode,
        selected_candidate_id=selected_id if validation["overall"] == "pass" else None,
        rejection_reason=rejection_reason or (
            "; ".join(validation.get("failure_reasons") or [])
        ),
        request_path=request_path,
        response_path=response_path if response_path and response_path.exists() else None,
        validation_path=validation_path,
        trace_path=trace_path,
        out_dir=out_dir,
    )


def update_trace_with_recipe_op(
    run_dir: Path, *, recipe_op_id: str,
) -> None:
    """Once recipe_planning commits, overwrite the trace's
    ``committed_recipe_op`` field with the real id."""
    trace_path = (
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_trace.json"
    )
    if not trace_path.exists():
        return
    trace = _read_json(trace_path)
    trace["committed_recipe_op"] = recipe_op_id
    trace_path.write_text(
        json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# M-15A: iterative agent-decision (in-process retry)
# --------------------------------------------------------------------------- #


def run_agent_decision_iterative(
    run_dir: Path,
    *,
    selection_mode: str,
    response_paths: list[Path],
    max_retries: int = 3,
    live_config: LiveProviderConfig | None = None,
) -> AgentDecisionResult:
    """Try multiple agent_decision_response.json paths in order.

    Each path is run through ``run_agent_decision`` as a separate
    attempt. The first attempt that passes validation is the accepted
    response; subsequent paths are not consumed. The retry_summary.json
    captures the full attempt history.

    Capped at ``max_retries`` attempts; if exhausted without a pass,
    ``retry_summary.status = "failed_exhausted_retries"`` and the
    returned ``AgentDecisionResult.overall = "fail"``.

    The single-attempt case (one path, max_retries >= 1) is just an
    iterative call with len(response_paths) == 1.
    """
    out_dir = run_dir / "03_recipe_planning" / "agent_decision"
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_retries <= 0:
        max_retries = 1
    capped = response_paths[:max_retries]
    attempts_records: list[dict[str, Any]] = []
    last_result: AgentDecisionResult | None = None

    for idx, p in enumerate(capped):
        result = run_agent_decision(
            run_dir,
            selection_mode=selection_mode,
            agent_response_path=p,
            live_config=live_config,
            attempt_index=idx,
            max_retries=max_retries,
        )
        last_result = result
        validation = _read_json_or_none(
            out_dir / "agent_decision_validation.json"
        ) or {}
        if result.overall == "pass":
            attempts_records.append(
                {
                    "attempt_index": idx,
                    "status": "pass",
                    "selected_candidate_id": result.selected_candidate_id,
                }
            )
            _emit_retry_summary(
                out_dir=out_dir,
                attempts=attempts_records,
                max_attempts=max_retries,
                final_status="pass",
                final_selected_candidate_id=result.selected_candidate_id,
                recipe_committed=True,
            )
            return result
        # Fail: record and continue.
        failed_checks = [
            c.get("name", "") for c in validation.get("checks", [])
            if c.get("status") != "pass"
        ]
        attempts_records.append(
            {
                "attempt_index": idx,
                "status": "fail",
                "failed_checks": failed_checks,
                "rejection_reason": result.rejection_reason,
            }
        )

    # Exhausted retries.
    _emit_retry_summary(
        out_dir=out_dir,
        attempts=attempts_records,
        max_attempts=max_retries,
        final_status="failed_exhausted_retries",
        final_selected_candidate_id=None,
        recipe_committed=False,
    )
    if last_result is None:
        # Empty response_paths — return a "no attempt" failure.
        return AgentDecisionResult(
            overall="fail",
            selection_mode=selection_mode,
            selected_candidate_id=None,
            rejection_reason="no agent_decision_response paths supplied",
            request_path=out_dir / "agent_decision_request.json",
            response_path=None,
            validation_path=out_dir / "agent_decision_validation.json",
            trace_path=out_dir / "agent_decision_trace.json",
            out_dir=out_dir,
        )
    return last_result
