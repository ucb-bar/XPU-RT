"""MCP recovery tools: synthesize_decomp, synthesize_translation,
register_blackbox, resolve_unsupported_op.

Each tool is a thin wrapper over the existing
:mod:`compgen.capture.unsupported` synthesisers. Splitting them into
four discrete tools (rather than one aggregate) is deliberate:

* The LLM's tool-selection attribution is clearer — you can
  tell from the transcript which strategy was picked.
* Each strategy becomes individually gradable in P3 (the
  ``~/.compgen/extensions/_state.json`` tracks per-tool invocations
  so the cross-session graduation loop can promote the exact wrapper
  the LLM picked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.capture.unsupported import UnsupportedOpResolution
from compgen.capture.unsupported.classify import UnsupportedClassification
from compgen.capture.unsupported.synthesize_decomp import (
    SynthesizedDecomposition,
    synthesize_export_decomposition,
)
from compgen.capture.unsupported.synthesize_translation import (
    SynthesizedPayloadTranslation,
    synthesize_payload_translation,
)
from compgen.capture.unsupported.verify import verify_unsupported_resolution
from compgen.mcp.session import McpSession, SessionManager


# ---------------------------------------------------------------------------
# Per-session recovery state (lives on the session's metadata dict)
# ---------------------------------------------------------------------------


@dataclass
class RecoveryBookkeeping:
    """Tracks which ops have been resolved and how.

    Stored at ``session.metadata['recovery']``; persists for the
    lifetime of the session. Feeds back into diagnose calls so the
    LLM sees updated state on each iteration.
    """

    decomps: dict[str, str] = field(default_factory=dict)          # target -> description
    translations: dict[str, str] = field(default_factory=dict)     # target -> callee_name
    blackboxes: set[str] = field(default_factory=set)              # target set
    failed: dict[str, str] = field(default_factory=dict)           # target -> error


def _recovery_state(session: McpSession) -> RecoveryBookkeeping:
    state = session.metadata.get("recovery")
    if not isinstance(state, RecoveryBookkeeping):
        state = RecoveryBookkeeping()
        session.metadata["recovery"] = state
    return state


def _find_resolution(
    session: McpSession, op_target: str,
) -> UnsupportedOpResolution | None:
    compiled = session.require_compiled()
    for res in compiled.capture_artifact.unsupported_resolutions:
        if res.target == op_target:
            return res
    return None


# ---------------------------------------------------------------------------
# Individual strategy tools
# ---------------------------------------------------------------------------


def synthesize_decomp(
    sm: SessionManager,
    *,
    session_id: str,
    op_target: str,
) -> dict[str, Any]:
    """Attempt an ATen allow-list decomposition for ``op_target``.

    Only the ATen allow-list in
    :mod:`compgen.capture.unsupported.synthesize_decomp` can supply
    a real decomp today. For off-list ops we return ``ok=False`` with
    ``reason='not_on_allow_list'`` so the LLM can fall back to
    ``register_blackbox`` or ``synthesize_translation``.
    """
    session = sm.get(session_id)
    resolution = _find_resolution(session, op_target)
    if resolution is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"Op {op_target!r} not in session's unsupported set.",
        }

    decomp: SynthesizedDecomposition | None = synthesize_export_decomposition(
        op_target, resolution.dossier,
    )
    if decomp is None:
        state = _recovery_state(session)
        state.failed[op_target] = "not_on_allow_list"
        return {
            "ok": False, "session_id": session_id,
            "op_target": op_target,
            "reason": "not_on_allow_list",
            "remediation_hint": (
                "Allow-list decomps are limited to common ATen ops. "
                "Try register_blackbox or synthesize_translation for "
                "off-list targets."
            ),
        }

    state = _recovery_state(session)
    state.decomps[op_target] = decomp.description
    state.failed.pop(op_target, None)
    return {
        "ok": True, "session_id": session_id,
        "op_target": op_target,
        "description": decomp.description,
        "strategy": "export_decomposition",
    }


def synthesize_translation(
    sm: SessionManager,
    *,
    session_id: str,
    op_target: str,
) -> dict[str, Any]:
    """Wire the existing synthesised-translation path for ``op_target``.

    The classifier decides whether the op is eligible. We respect
    its decision (reusing the per-op classification from capture)
    rather than re-running it with different assumptions.
    """
    session = sm.get(session_id)
    resolution = _find_resolution(session, op_target)
    if resolution is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"Op {op_target!r} not in session's unsupported set.",
        }

    # If the classifier didn't mark it eligible, force the decision
    # only when the dossier supports it (simple tensor schema).
    cls = resolution.classification
    if cls.strategy != "synthesized_external_call":
        # Try a best-effort re-classification by synthesising directly.
        forced_cls = UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="low",
            reason="LLM override; structural eligibility unchecked",
        )
        translation = synthesize_payload_translation(
            resolution.issue, resolution.dossier, forced_cls,
        )
    else:
        translation = resolution.translation or synthesize_payload_translation(
            resolution.issue, resolution.dossier, cls,
        )

    if translation is None:
        state = _recovery_state(session)
        state.failed[op_target] = "translation_not_eligible"
        return {
            "ok": False, "session_id": session_id,
            "op_target": op_target,
            "reason": "translation_not_eligible",
            "remediation_hint": (
                "The op's schema is too complex for an external-call "
                "translation. Try register_blackbox."
            ),
        }

    state = _recovery_state(session)
    state.translations[op_target] = translation.callee_name
    state.failed.pop(op_target, None)
    return {
        "ok": True, "session_id": session_id,
        "op_target": op_target,
        "callee_name": translation.callee_name,
        "kind": translation.kind,
        "strategy": "payload_translation",
    }


def register_blackbox(
    sm: SessionManager,
    *,
    session_id: str,
    op_target: str,
    runtime_version_override: str | None = None,
) -> dict[str, Any]:
    """Mark ``op_target`` as an explicit opaque-boundary fallback.

    The promotion record embeds the installed runtime versions so the
    blackbox is only reused under compatible runtimes (reuses the
    existing ``build_promotion_record`` cache-key logic).
    """
    session = sm.get(session_id)
    resolution = _find_resolution(session, op_target)
    if resolution is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"Op {op_target!r} not in session's unsupported set.",
        }
    state = _recovery_state(session)
    state.blackboxes.add(op_target)
    state.failed.pop(op_target, None)

    return {
        "ok": True, "session_id": session_id,
        "op_target": op_target,
        "strategy": "explicit_blackbox",
        "promotion_record": {
            "cache_key": resolution.promotion.cache_key,
            "policy": resolution.promotion.policy,
            "runtime_versions": dict(resolution.promotion.runtime_versions),
        },
    }


def resolve_unsupported_op(
    sm: SessionManager,
    *,
    session_id: str,
    op_target: str,
    strategy: str = "auto",
) -> dict[str, Any]:
    """Aggregate: pick a strategy for ``op_target`` and apply it.

    ``strategy`` can be ``"auto" | "decomp" | "translation" | "blackbox"``.
    ``auto`` routes to the recommendation from
    :func:`diagnose_model_compatibility`.
    """
    session = sm.get(session_id)
    resolution = _find_resolution(session, op_target)
    if resolution is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"Op {op_target!r} not in session's unsupported set.",
        }

    if strategy == "auto":
        if resolution.classification.strategy == "synthesized_external_call":
            strategy = "translation"
        elif resolution.dossier.is_aten:
            strategy = "decomp"
        else:
            strategy = "blackbox"

    dispatch = {
        "decomp": synthesize_decomp,
        "translation": synthesize_translation,
        "blackbox": register_blackbox,
    }
    fn = dispatch.get(strategy)
    if fn is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"Unknown strategy {strategy!r}; expected auto|decomp|translation|blackbox",
        }
    result = fn(sm, session_id=session_id, op_target=op_target)
    result["attempted_strategy"] = strategy
    return result


def recovery_status(
    sm: SessionManager, *, session_id: str,
) -> dict[str, Any]:
    """Report the session's accumulated recovery state."""
    session = sm.get(session_id)
    state = _recovery_state(session)
    return {
        "ok": True, "session_id": session_id,
        "decomps": dict(state.decomps),
        "translations": dict(state.translations),
        "blackboxes": sorted(state.blackboxes),
        "failed": dict(state.failed),
    }


RECOVERY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "synthesize_decomp",
        "description": "Install an ATen allow-list decomposition for the given op target.",
        "phase": "transform",
        "handler": synthesize_decomp,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "op_target": {"type": "string"},
            },
            "required": ["session_id", "op_target"],
        },
    },
    {
        "name": "synthesize_translation",
        "description": "Wire a Payload-level external-call translation for the op target.",
        "phase": "transform",
        "handler": synthesize_translation,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "op_target": {"type": "string"},
            },
            "required": ["session_id", "op_target"],
        },
    },
    {
        "name": "register_blackbox",
        "description": "Mark an op target as an explicit opaque-boundary fallback.",
        "phase": "transform",
        "handler": register_blackbox,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "op_target": {"type": "string"},
                "runtime_version_override": {"type": "string"},
            },
            "required": ["session_id", "op_target"],
        },
    },
    {
        "name": "resolve_unsupported_op",
        "description": (
            "Aggregator: pick (auto|decomp|translation|blackbox) and apply "
            "recovery for one unsupported op."
        ),
        "phase": "transform",
        "handler": resolve_unsupported_op,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "op_target": {"type": "string"},
                "strategy": {
                    "type": "string",
                    "enum": ["auto", "decomp", "translation", "blackbox"],
                    "default": "auto",
                },
            },
            "required": ["session_id", "op_target"],
        },
    },
    {
        "name": "recovery_status",
        "description": "Report the session's accumulated recovery decisions.",
        "phase": "inspect",
        "handler": recovery_status,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "RECOVERY_TOOLS",
    "RecoveryBookkeeping",
    "recovery_status",
    "register_blackbox",
    "resolve_unsupported_op",
    "synthesize_decomp",
    "synthesize_translation",
]
