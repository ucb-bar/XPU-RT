"""MCP diagnose tools: expose the existing unsupported-op recovery pipeline.

The capture pipeline already runs detect → introspect → classify on
every ``capture_frontend_artifact`` call. These tools surface those
results to the LLM in a compact, decision-ready shape so the LLM can
decide which recovery strategy to apply.

No new recovery logic lives here — this is a thin view over
:mod:`compgen.capture.unsupported`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from compgen.capture.unsupported import (
    UnsupportedOpResolution,
    recover_unsupported_operators,
)
from compgen.capture.unsupported.introspect import runtime_versions
from compgen.ir.payload.decompositions import DECOMPOSITION_TABLE
from compgen.mcp.session import SessionManager


# Strategy -> one-line LLM hint. These mirror the bucket/strategy
# pairs produced by classify.py so the LLM can pick a concrete tool.
_STRATEGY_HINTS: dict[str, str] = {
    "known_payload_decomposition": (
        "Already lowers through the registered Payload decomposition. "
        "No action required."
    ),
    "synthesized_external_call": (
        "Eligible for an external-call translation. Call "
        "synthesize_translation to wire it up."
    ),
    "explicit_blackbox": (
        "Requires an opaque-boundary fallback. Call register_blackbox "
        "to mark the target, or propose a custom decomp via "
        "synthesize_decomp if one exists."
    ),
}


def _resolution_to_dict(
    resolution: UnsupportedOpResolution,
) -> dict[str, Any]:
    """Flatten a resolution dataclass for the LLM.

    Deliberately omits the reference_callable (not serialisable) and
    keeps shape + strategy details the LLM needs to make a decision.
    """
    issue = resolution.issue
    dossier = resolution.dossier
    cls = resolution.classification
    example_inputs = [asdict(e) for e in issue.example_inputs]
    example_output = (
        asdict(issue.example_output) if issue.example_output else None
    )
    return {
        "target": issue.target,
        "count": issue.count,
        "stage": issue.stage,
        "reason": issue.reason,
        "node_names": list(issue.node_names),
        "source_locations": list(issue.source_locations),
        "example_inputs": example_inputs,
        "example_output": example_output,
        "dossier": {
            "namespace": dossier.namespace,
            "operator": dossier.operator,
            "overload": dossier.overload,
            "schema": dossier.schema,
            "is_aten": dossier.is_aten,
            "is_custom": dossier.is_custom,
            "is_torchao_like": dossier.is_torchao_like,
            "is_view": dossier.is_view,
            "has_meta_kernel": dossier.has_meta_kernel,
            "has_any_kernel": dossier.has_any_kernel,
            "payload_decomposition_registered": (
                dossier.payload_decomposition_registered
            ),
        },
        "classification": {
            "bucket": cls.bucket,
            "strategy": cls.strategy,
            "confidence": cls.confidence,
            "reason": cls.reason,
            "llm_hint": _STRATEGY_HINTS.get(cls.strategy, ""),
        },
        "verification": {
            "schema_ok": resolution.verification.schema_ok,
            "eager_reference_ok": resolution.verification.eager_reference_ok,
            "meta_reference_ok": resolution.verification.meta_reference_ok,
            "messages": list(resolution.verification.messages),
        },
        "promotion": {
            "cache_key": resolution.promotion.cache_key,
            "policy": resolution.promotion.policy,
        },
        "recommended_strategy": _recommend(resolution),
        "recommended_tool": _recommend_tool(resolution),
    }


def _recommend(resolution: UnsupportedOpResolution) -> str:
    """Collapse the classification into one action word for the LLM."""
    strategy = resolution.classification.strategy
    if strategy == "known_payload_decomposition":
        return "none"
    if strategy == "synthesized_external_call":
        return "translation"
    # Both quantization_wrapper + opaque_custom_op + blackbox_boundary
    # map to explicit_blackbox, but when the dossier claims an ATen
    # op exists we might attempt a decomp first.
    if resolution.dossier.is_aten:
        return "decomp"
    return "blackbox"


def _recommend_tool(resolution: UnsupportedOpResolution) -> str:
    """Which recovery tool the LLM should invoke first."""
    reco = _recommend(resolution)
    return {
        "none": "",
        "translation": "synthesize_translation",
        "decomp": "synthesize_decomp",
        "blackbox": "register_blackbox",
    }.get(reco, "")


def diagnose_model_compatibility(
    sm: SessionManager,
    *,
    session_id: str,
    include_recovered: bool = False,
) -> dict[str, Any]:
    """Return the LLM-facing unsupported-op report for the session.

    The report draws from the capture artifact's existing recovery
    run (nothing is re-executed here). Shape::

        {
          ok: True,
          recoverable: bool,
          num_issues: int,
          issues: [ _resolution_to_dict, ... ],
          recommended_actions: ["synthesize_decomp foo", ...],
        }
    """
    session = sm.get(session_id)
    compiled = session.require_compiled()
    resolutions = list(compiled.capture_artifact.unsupported_resolutions)

    # Drop already-recovered rows unless asked (the LLM rarely needs them).
    if not include_recovered:
        resolutions = [
            r for r in resolutions
            if r.classification.strategy != "known_payload_decomposition"
        ]

    issues_out = [_resolution_to_dict(r) for r in resolutions]
    # Recoverable = every remaining issue maps to a strategy we know
    # how to apply. The only failure case is a missing schema or
    # missing eager reference (verification.schema_ok == False).
    recoverable = all(i["verification"]["schema_ok"] for i in issues_out)

    recommended_actions: list[str] = []
    for i in issues_out:
        tool = i["recommended_tool"]
        if tool:
            recommended_actions.append(f"{tool}: {i['target']}")

    return {
        "ok": True,
        "session_id": session_id,
        "recoverable": recoverable,
        "num_issues": len(issues_out),
        "issues": issues_out,
        "recommended_actions": recommended_actions,
        "runtime_versions": dict(compiled.capture_artifact.runtime_versions),
    }


def diagnose_exported_program(
    exported_program: Any,
    *,
    supported_targets: set[str] | None = None,
) -> dict[str, Any]:
    """Standalone helper (no session) that a user can call directly.

    Useful for ``compile_with_llm(recover_unsupported=True)`` which
    runs diagnosis before opening an MCP session.
    """
    supported = supported_targets or set(DECOMPOSITION_TABLE.keys())
    versions = runtime_versions()
    resolutions = recover_unsupported_operators(
        exported_program,
        supported_targets=supported,
        runtime_versions=versions,
    )
    rows = [_resolution_to_dict(r) for r in resolutions]
    return {
        "ok": True,
        "num_issues": len(rows),
        "issues": rows,
        "recoverable": all(r["verification"]["schema_ok"] for r in rows),
    }


DIAGNOSE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "diagnose_model_compatibility",
        "description": (
            "Summarise which operators in the session's model lack a "
            "registered Payload lowering, classify each, and recommend "
            "a recovery tool."
        ),
        "phase": "inspect",
        "handler": diagnose_model_compatibility,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "include_recovered": {"type": "boolean", "default": False},
            },
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "DIAGNOSE_TOOLS",
    "diagnose_exported_program",
    "diagnose_model_compatibility",
]
