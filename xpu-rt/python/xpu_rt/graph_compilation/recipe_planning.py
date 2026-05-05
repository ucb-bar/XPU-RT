"""Recipe Planning stage (Milestone 05).

Selects a candidate from the M-04 action space and commits the
embedded Recipe delta to ``03_recipe_planning/recipe.mlir``.

This is the first milestone where the agent role actually *acts*. The
LLM (or its deterministic stand-ins) selects only candidate IDs; the
compiler resolves each ID against ``02_graph_analysis/action_space.mlir``
via the M-04.5 resolver and appends the verified Recipe op to
``recipe.mlir``.

Selection modes:

- ``greedy`` — pick the highest-priority site, then the legal candidate
  with the lowest ``static_relative_cost``, tie-broken by candidate_id
  lexicographic order. Deterministic baseline for ablations.
- ``agent-file`` — agent-driven selection. The pipeline emits an
  ``agent_decision_request.json``, an external agent (Claude Code via
  MCP/skill) writes ``agent_decision_response.json``, and the planner
  validates + commits it.
- ``llm-live`` — same protocol as ``agent-file`` but the pipeline
  itself calls a real LLM HTTP endpoint (anthropic / openai). Requires
  an API key; secondary path to ``agent-file``.

This stage does **not**:

- apply transforms,
- mutate Payload IR,
- run verifier gates beyond legality recorded by M-04,
- benchmark, profile, or call any kernel codegen,
- modify compiler core.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.graph_compilation.action_space_resolver import (
    ResolvedCandidate,
    ResolverError,
    resolve_candidate,
)
from compgen.graph_compilation.artifacts import ArtifactRef, StageRecord
from compgen.graph_compilation.hashing import sha256_file, sha256_tree

SUPPORTED_SELECTION_MODES: tuple[str, ...] = (
    "greedy", "agent-file", "llm-live",
)


@dataclass(frozen=True)
class SelectionTraceEntry:
    timestamp_utc: str
    selection_mode: str
    candidate_id: str
    site_id: str
    region_id: str
    kind: str
    legal: bool
    static_relative_cost: float
    decision: str  # "selected" | "skipped_illegal" | "considered" | "rejected_no_legal_in_site"
    reason: str


# --------------------------------------------------------------------------- #
# Selection policies
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _select_greedy(
    decision_sites: dict[str, Any],
    candidate_actions: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[SelectionTraceEntry], str]:
    """Greedy single-pick policy:

    1. Sort sites by ``priority`` ascending (lower number = higher priority).
    2. Within each site, pick the legal candidate with the lowest
       ``static_relative_cost``; ties broken by candidate_id lex order.
    3. Return the first such selection.

    Returns ``(selected_candidate_json, trace, primary_reason)``.
    """
    cand_by_id = {c["candidate_id"]: c for c in candidate_actions["candidates"]}
    sites_sorted = sorted(
        decision_sites["sites"],
        key=lambda s: (int(s["priority"]), s["site_id"]),
    )
    trace: list[SelectionTraceEntry] = []
    selected: dict[str, Any] | None = None

    for site in sites_sorted:
        legal_in_site: list[dict[str, Any]] = []
        for cid in site["candidate_ids"]:
            c = cand_by_id.get(cid)
            if c is None:
                continue
            if c["legality"]["ok"]:
                legal_in_site.append(c)
                trace.append(
                    SelectionTraceEntry(
                        timestamp_utc=_utcnow(),
                        selection_mode="greedy",
                        candidate_id=cid,
                        site_id=site["site_id"],
                        region_id=c["region_id"],
                        kind=c["kind"],
                        legal=True,
                        static_relative_cost=float(
                            c["cost_preview"].get("static_relative_cost", 1.0)
                        ),
                        decision="considered",
                        reason="legal candidate in highest-priority site",
                    )
                )
            else:
                trace.append(
                    SelectionTraceEntry(
                        timestamp_utc=_utcnow(),
                        selection_mode="greedy",
                        candidate_id=cid,
                        site_id=site["site_id"],
                        region_id=c["region_id"],
                        kind=c["kind"],
                        legal=False,
                        static_relative_cost=float(
                            c["cost_preview"].get("static_relative_cost", 1.0)
                        ),
                        decision="skipped_illegal",
                        reason=c["legality"].get("reason", "illegal"),
                    )
                )

        if not legal_in_site:
            trace.append(
                SelectionTraceEntry(
                    timestamp_utc=_utcnow(),
                    selection_mode="greedy",
                    candidate_id="",
                    site_id=site["site_id"],
                    region_id=site["region_id"],
                    kind=site["kind"],
                    legal=False,
                    static_relative_cost=0.0,
                    decision="rejected_no_legal_in_site",
                    reason="every candidate in this site is illegal",
                )
            )
            continue

        legal_in_site.sort(
            key=lambda c: (
                float(c["cost_preview"].get("static_relative_cost", 1.0)),
                c["candidate_id"],
            )
        )
        chosen = legal_in_site[0]
        trace.append(
            SelectionTraceEntry(
                timestamp_utc=_utcnow(),
                selection_mode="greedy",
                candidate_id=chosen["candidate_id"],
                site_id=site["site_id"],
                region_id=chosen["region_id"],
                kind=chosen["kind"],
                legal=True,
                static_relative_cost=float(
                    chosen["cost_preview"].get("static_relative_cost", 1.0)
                ),
                decision="selected",
                reason=(
                    f"highest-priority site (priority={site['priority']}); "
                    f"lowest static_relative_cost legal candidate"
                ),
            )
        )
        selected = chosen
        primary = (
            f"greedy: highest-priority site {site['site_id']!r} "
            f"(priority={site['priority']}); lowest-cost legal candidate "
            f"({chosen['cost_preview'].get('static_relative_cost', 1.0)})"
        )
        return selected, trace, primary

    return None, trace, "greedy: no legal candidate across any site"


_SELECTORS = {
    "greedy": _select_greedy,
}


# --------------------------------------------------------------------------- #
# recipe.mlir emit
# --------------------------------------------------------------------------- #


def _safe(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def _camel_to_snake(s: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch.isupper() and i > 0 and (s[i - 1].islower() or s[i - 1].isdigit()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


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
        return (
            "{ "
            + ", ".join(f"{k} = {_mlir_attr(val)}" for k, val in sorted(v.items()))
            + " }"
        )
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _emit_recipe_mlir(
    *,
    model_id: str,
    target_id: str,
    action_space_ir_sha256: str,
    selection_mode: str,
    selected: ResolvedCandidate | None,
    rationale_primary: str,
) -> str:
    head_attrs = {
        "model_id": model_id,
        "target_id": target_id,
        "selection_mode": selection_mode,
        "source_action_space_sha256": action_space_ir_sha256,
        "recipe_op_count": 1 if selected else 0,
    }
    lines = [
        f"recipe.module @{_safe(model_id)}_{_safe(target_id)} attributes "
        f"{{ {', '.join(f'{k} = {_mlir_attr(v)}' for k, v in sorted(head_attrs.items()))} }} {{"
    ]
    if selected is not None:
        for idx, op_dict in enumerate(selected.recipe_delta):
            op_name = op_dict.get("op", "Unknown")
            op_snake = _camel_to_snake(op_name)
            body = {k: v for k, v in op_dict.items() if k != "op"}
            body.setdefault("source_candidate", selected.candidate_id)
            body.setdefault("rationale", rationale_primary)
            attr_text = ", ".join(
                f"{k} = {_mlir_attr(val)}" for k, val in sorted(body.items())
            )
            lines.append(
                f"  recipe.{op_snake} @recipe_{idx:04d} attributes {{ {attr_text} }}"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecipePlanningResult:
    out_dir: Path
    recipe_mlir_path: Path
    candidate_selection_path: Path
    selection_trace_path: Path
    recipe_validation_path: Path
    recipe_summary_path: Path
    selected_candidate_id: str | None
    selection_mode: str
    overall: str  # "pass" | "fail" | "no_candidate"


def run_recipe_planning(
    run_dir: Path,
    *,
    selection_mode: str = "greedy",
    rationale_primary: str | None = None,
    agent_decision_response_path: Path | None = None,
    agent_decision_response_paths: list[Path] | None = None,
    agent_max_retries: int = 3,
    live_provider_config: Any | None = None,
) -> RecipePlanningResult:
    """Run the M-05 recipe-planning stage on an existing run directory.

    When ``selection_mode != "greedy"``, the M-14A agent-decision
    protocol runs first: it emits ``agent_decision_request.json``,
    obtains a response (user-provided for ``agent-file``, real LLM
    HTTP call for ``llm-live``), validates it, and only then proceeds
    to commit the agent-selected candidate. A failed validation
    hard-aborts before ``recipe.mlir`` is written.
    """
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"selection_mode={selection_mode!r} not in {SUPPORTED_SELECTION_MODES}"
        )
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs.
    decision_sites = json.loads(
        (ga / "decision_sites.json").read_text(encoding="utf-8")
    )
    candidate_actions = json.loads(
        (ga / "candidate_actions.json").read_text(encoding="utf-8")
    )
    graph_dossier = json.loads(
        (ga / "graph_dossier_v2.json").read_text(encoding="utf-8")
    )

    model_id = graph_dossier.get("model_id", "model")
    target_id = decision_sites.get("target_id", "host_cpu")
    action_space_ir_sha256 = decision_sites["source"]["action_space_ir_sha256"]

    # ------------------------------------------------------------------ #
    # M-14A agent-decision protocol (agent-file + llm-live modes).
    # ------------------------------------------------------------------ #
    agent_selected_id: str | None = None
    if selection_mode in ("agent-file", "llm-live"):
        from compgen.graph_compilation.agent_decision import (
            run_agent_decision,
            run_agent_decision_iterative,
        )

        # M-15A: when a list of response paths is provided (or just
        # a single one), use the iterative wrapper so the retry trail
        # accumulates under attempts/attempt_<N>/ and retry_summary.json
        # gets the full history. The single-path call is a special
        # case (one attempt). For llm-live (which doesn't consume
        # external response files), the single-shot call still applies
        # and the iterative wrapper treats it as 1 attempt.
        if (
            selection_mode == "agent-file"
            and (agent_decision_response_paths or agent_decision_response_path)
        ):
            paths = list(agent_decision_response_paths or [])
            if agent_decision_response_path is not None:
                paths = [agent_decision_response_path] + paths
            agent_result = run_agent_decision_iterative(
                run_dir,
                selection_mode=selection_mode,
                response_paths=paths,
                max_retries=agent_max_retries,
                live_config=live_provider_config,
            )
        else:
            agent_result = run_agent_decision(
                run_dir,
                selection_mode=selection_mode,
                agent_response_path=agent_decision_response_path,
                live_config=live_provider_config,
            )
        if agent_result.overall != "pass":
            # Fail BEFORE recipe.mlir is written. Retry artifacts (when
            # selection_mode=agent-file) are already on disk under
            # attempts/attempt_<N>/ and retry_request.json — Claude Code
            # reads them and re-invokes with a corrected response.
            raise RuntimeError(
                f"M-14A agent decision failed for selection_mode="
                f"{selection_mode!r}: {agent_result.rejection_reason}"
            )
        agent_selected_id = agent_result.selected_candidate_id

    if agent_selected_id is not None:
        # Override the legacy selector's pick with the agent-selected
        # candidate. We still build a synthetic trace entry so the
        # candidate_selection.json fields stay consistent.
        cand_by_id = {
            c["candidate_id"]: c for c in candidate_actions["candidates"]
        }
        sel_cand = cand_by_id.get(agent_selected_id)
        if sel_cand is None:
            raise RuntimeError(
                f"agent-selected candidate {agent_selected_id!r} not in "
                f"candidate_actions.json (validation should have caught this)"
            )
        # Locate the site that hosts this candidate.
        site = next(
            (
                s for s in decision_sites["sites"]
                if agent_selected_id in s.get("candidate_ids", [])
            ),
            None,
        )
        site_id = site["site_id"] if site is not None else ""
        site_priority = int(site["priority"]) if site is not None else 0
        trace = [
            SelectionTraceEntry(
                timestamp_utc=_utcnow(),
                selection_mode=selection_mode,
                candidate_id=agent_selected_id,
                site_id=site_id,
                region_id=sel_cand["region_id"],
                kind=sel_cand["kind"],
                legal=True,
                static_relative_cost=float(
                    (sel_cand.get("cost_preview") or {}).get(
                        "static_relative_cost", 1.0
                    )
                ),
                decision="selected",
                reason=(
                    f"{selection_mode}: agent-selected candidate via "
                    f"agent_decision_response.json"
                ),
            )
        ]
        primary = (
            f"{selection_mode}: agent-selected candidate "
            f"{agent_selected_id!r} (site_id={site_id!r}, "
            f"priority={site_priority})"
        )
        selected_json = sel_cand
    else:
        selector = _SELECTORS[selection_mode]
        selected_json, trace, primary = selector(decision_sites, candidate_actions)

    # Resolve through the canonical IR — this is the firewall against
    # JSON-only fakery. The resolver re-checks every cross-projection
    # sha256 and the recipe_delta consistency.
    resolved: ResolvedCandidate | None = None
    resolver_error: str | None = None
    if selected_json is not None:
        try:
            resolved, _ = resolve_candidate(
                run_dir=run_dir,
                candidate_id=selected_json["candidate_id"],
                allow_illegal=False,  # legal-only by construction
                selection_mode=selection_mode,
                rationale={
                    "primary_reason": rationale_primary or primary,
                    "evidence": _build_evidence(selected_json),
                },
                write_outputs=False,  # we own the writes below
            )
        except ResolverError as exc:
            resolver_error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ #
    # 1. recipe.mlir
    # ------------------------------------------------------------------ #
    recipe_mlir_path = out_dir / "recipe.mlir"
    recipe_mlir_path.write_text(
        _emit_recipe_mlir(
            model_id=model_id,
            target_id=target_id,
            action_space_ir_sha256=action_space_ir_sha256,
            selection_mode=selection_mode,
            selected=resolved,
            rationale_primary=rationale_primary or primary,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 2. candidate_selection.json
    # ------------------------------------------------------------------ #
    candidate_selection_path = out_dir / "candidate_selection.json"
    if resolved is not None:
        sel = {
            "schema_version": "candidate_selection_v1",
            "model_id": model_id,
            "target_id": target_id,
            "selected_candidate_id": resolved.candidate_id,
            "site_id": resolved.site_id,
            "region_id": resolved.region_id,
            "candidate_kind": resolved.kind,
            "label": resolved.label,
            "selection_mode": selection_mode,
            "selected_at_utc": _utcnow(),
            "source": dict(resolved.source),
            "legality": {
                "ok": resolved.legality_ok,
                "reason": resolved.legality_reason,
            },
            "rationale": {
                "primary_reason": rationale_primary or primary,
                "evidence": _build_evidence(selected_json) if selected_json else [],
            },
            "recipe_delta": list(resolved.recipe_delta),
            "cost_preview": dict(resolved.cost_preview),
            "evidence": dict(resolved.evidence),
        }
    else:
        sel = {
            "schema_version": "candidate_selection_v1",
            "model_id": model_id,
            "target_id": target_id,
            "selected_candidate_id": None,
            "selection_mode": selection_mode,
            "selected_at_utc": _utcnow(),
            "source": {
                "action_space_ir": str((ga / "action_space.mlir").relative_to(run_dir)),
                "action_space_ir_sha256": action_space_ir_sha256,
            },
            "rationale": {
                "primary_reason": (
                    "no candidate selected"
                    if not resolver_error
                    else f"resolver error: {resolver_error}"
                ),
                "evidence": [],
            },
            "recipe_delta": [],
            "resolver_error": resolver_error,
        }
    candidate_selection_path.write_text(
        json.dumps(sel, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 3. selection_trace.jsonl
    # ------------------------------------------------------------------ #
    selection_trace_path = out_dir / "selection_trace.jsonl"
    with selection_trace_path.open("w", encoding="utf-8") as f:
        for t in trace:
            f.write(
                json.dumps(
                    {
                        "schema_version": "selection_trace_event_v1",
                        "timestamp_utc": t.timestamp_utc,
                        "selection_mode": t.selection_mode,
                        "candidate_id": t.candidate_id,
                        "site_id": t.site_id,
                        "region_id": t.region_id,
                        "kind": t.kind,
                        "legal": t.legal,
                        "static_relative_cost": t.static_relative_cost,
                        "decision": t.decision,
                        "reason": t.reason,
                    }
                )
                + "\n"
            )

    # ------------------------------------------------------------------ #
    # 4. recipe_summary.json
    # ------------------------------------------------------------------ #
    recipe_summary_path = out_dir / "recipe_summary.json"
    summary = {
        "schema_version": "recipe_summary_v1",
        "model_id": model_id,
        "target_id": target_id,
        "selection_mode": selection_mode,
        "selected_candidate_id": resolved.candidate_id if resolved else None,
        "site_id": resolved.site_id if resolved else None,
        "region_id": resolved.region_id if resolved else None,
        "candidate_kind": resolved.kind if resolved else None,
        "recipe_op_count": 1 if resolved else 0,
        "trace_event_count": len(trace),
        "considered_candidates": sum(
            1 for t in trace if t.decision in ("considered", "selected")
        ),
        "skipped_illegal_candidates": sum(
            1 for t in trace if t.decision == "skipped_illegal"
        ),
        "source": {
            "action_space_ir": str((ga / "action_space.mlir").relative_to(run_dir)),
            "action_space_ir_sha256": action_space_ir_sha256,
        },
    }
    recipe_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # M-14A: backfill the agent-decision trace with the committed
    # recipe_op id (always recipe_0000 for the single-candidate MVP).
    if (
        selection_mode in ("agent-file", "llm-live")
        and resolved is not None
    ):
        from compgen.graph_compilation.agent_decision import (
            update_trace_with_recipe_op,
        )

        update_trace_with_recipe_op(run_dir, recipe_op_id="recipe_0000")

    # ------------------------------------------------------------------ #
    # 5. recipe_validation.json
    # ------------------------------------------------------------------ #
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    _add("recipe_mlir_exists", recipe_mlir_path.exists(), "")
    _add(
        "recipe_mlir_non_empty",
        recipe_mlir_path.exists() and recipe_mlir_path.stat().st_size > 0,
        "",
    )
    if resolved is not None:
        _add("selected_candidate_resolves", True, "")
        _add("selected_candidate_legal", resolved.legality_ok, resolved.legality_reason)
        text = recipe_mlir_path.read_text(encoding="utf-8")
        _add(
            "recipe_mlir_references_selected_candidate",
            resolved.candidate_id in text,
            "",
        )
        # recipe_delta in recipe.mlir matches candidate_actions projection: we
        # use the same _camel_to_snake mapping as the resolver, which is the
        # same mapping used by the action_space emitter.
        ok_delta = all(
            f"recipe.{_camel_to_snake(op.get('op', ''))} " in text
            for op in resolved.recipe_delta
        )
        _add("recipe_mlir_recipe_delta_matches_projection", ok_delta, "")
    else:
        # When no candidate was selected, recipe_validation is a soft fail
        # (overall=no_candidate) rather than hard-fail — empty action spaces
        # are valid (e.g. a model with only opaque regions whose extension
        # closures are themselves illegal).
        _add(
            "selected_candidate_resolves",
            resolver_error is None,
            resolver_error or "no candidate selected",
        )
    overall = (
        "pass"
        if resolved is not None and all(c["status"] == "pass" for c in checks)
        else (
            "fail"
            if resolver_error is not None
            else "no_candidate"
        )
    )
    validation = {
        "schema_version": "recipe_validation_v1",
        "overall": overall,
        "selection_mode": selection_mode,
        "checks": checks,
        "source": {
            "action_space_ir": str((ga / "action_space.mlir").relative_to(run_dir)),
            "action_space_ir_sha256": action_space_ir_sha256,
        },
    }
    recipe_validation_path = out_dir / "recipe_validation.json"
    recipe_validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    return RecipePlanningResult(
        out_dir=out_dir,
        recipe_mlir_path=recipe_mlir_path,
        candidate_selection_path=candidate_selection_path,
        selection_trace_path=selection_trace_path,
        recipe_validation_path=recipe_validation_path,
        recipe_summary_path=recipe_summary_path,
        selected_candidate_id=resolved.candidate_id if resolved else None,
        selection_mode=selection_mode,
        overall=overall,
    )


def _build_evidence(c: dict[str, Any]) -> list[str]:
    cp = c.get("cost_preview", {})
    items = []
    if "static_relative_cost" in cp:
        items.append(f"static_relative_cost={cp['static_relative_cost']}")
    if cp.get("fits_scratchpad") is not None:
        items.append(f"fits_scratchpad={cp['fits_scratchpad']}")
    if cp.get("fits_l2") is not None:
        items.append(f"fits_l2={cp['fits_l2']}")
    if "live_bytes" in cp:
        items.append(f"live_bytes={cp['live_bytes']}")
    if "estimated_latency_us" in cp:
        items.append(f"estimated_latency_us={cp['estimated_latency_us']}")
    return items


def stage_record(run_dir: Path, *, selection_mode: str) -> StageRecord:
    """Convert an existing 03_recipe_planning/ output into a StageRecord
    for the run manifest's ``stages`` list. Reads only on-disk state."""
    started = _utcnow()
    out_dir = run_dir / "03_recipe_planning"
    ga = run_dir / "02_graph_analysis"

    inputs: list[ArtifactRef] = []
    for path in sorted(ga.rglob("*")):
        if not path.is_file():
            continue
        inputs.append(
            ArtifactRef(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="file",
            )
        )
    outputs: list[ArtifactRef] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        outputs.append(
            ArtifactRef(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="file",
            )
        )
    return StageRecord(
        stage_id="recipe_planning",
        status="pass",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        report_path="03_recipe_planning/recipe_validation.json",
        input_hash=sha256_tree(ga),
        output_hash=sha256_tree(out_dir),
        llm_calls=0,  # bumped by llm-live HTTP path; agent-file/greedy stay 0
        started_at_utc=started,
        finished_at_utc=_utcnow(),
    )
