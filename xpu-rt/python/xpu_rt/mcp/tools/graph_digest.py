"""MCP tools exposing the LLM-facing graph digest + chunk views.

Two tools:

* ``analyze_graph`` — returns a shape-free :class:`GraphDigest` of the
  session's loaded model: pattern histogram, dim/dtype/quant spectra,
  FLOP/byte distributions, memory footprint, critical path, fusion
  opportunity count, region index. Default output is the compact
  ``to_prompt_summary`` form; set ``full=True`` to receive the full
  structured dict.

* ``focus_chunk`` — returns a :class:`ChunkView` for a selector
  ``{region_id | pattern_type | cluster_id | node_names}`` with both
  oracle-enumerated :class:`DecisionKnobs` and an open-ended
  :class:`DoFDescription` so the LLM can pick a safe candidate or
  propose a novel one.

Both tools resolve the module + analysis from the session's
:class:`McpSession`:

* If a compile has already run (``session.compiled`` is set), we
  re-run :class:`NetworkAnalyzer` on the captured exported program to
  get a full :class:`NetworkAnalysis` (the compiled model only stores
  the dossier).
* Otherwise we return ``ok: False`` with a remediation hint.
"""

from __future__ import annotations

from typing import Any

from compgen.agent.analyzer import NetworkAnalyzer
from compgen.analysis.graph_digest import (
    build_chunk_view,
    build_digest,
)
from compgen.mcp.session import SessionManager


def _load_session_analysis(sm: SessionManager, session_id: str) -> tuple[Any, Any, Any] | None:
    """Return ``(analysis, module, target_profile)`` for the session, or None."""
    session = sm.get(session_id)
    if session.compiled is None:
        return None
    compiled = session.compiled
    target = compiled.device.profile
    analyzer = NetworkAnalyzer()
    analysis = analyzer.analyze(
        compiled.capture_artifact.exported_program,
        target,
        model_name=type(compiled.model).__name__,
    )
    return analysis, compiled.payload_module, target


def analyze_graph(
    sm: SessionManager,
    *,
    session_id: str,
    full: bool = False,
) -> dict[str, Any]:
    """Return a shape-free :class:`GraphDigest` for the current session.

    Args:
        session_id: MCP session identifier.
        full: When True, return the full structured dict. Defaults to
            False which returns only ``to_prompt_summary()`` (≤2 KB).
    """
    loaded = _load_session_analysis(sm, session_id)
    if loaded is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": ("No compiled model in this session. Call load_model + compile_model first."),
        }
    analysis, module, target = loaded
    digest = build_digest(analysis, module=module, target_name=target.name)
    payload: dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "summary": digest.to_prompt_summary(),
    }
    if full:
        payload["digest"] = digest.to_dict()
    return payload


def focus_chunk(
    sm: SessionManager,
    *,
    session_id: str,
    selector: dict[str, Any] | None = None,
    include_concrete_shapes: bool = False,
) -> dict[str, Any]:
    """Return a :class:`ChunkView` for the given selector."""
    loaded = _load_session_analysis(sm, session_id)
    if loaded is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": "No compiled model in this session.",
        }
    analysis, module, target = loaded
    view = build_chunk_view(
        analysis,
        target,
        selector or {},
        module=module,
        include_concrete_shapes=include_concrete_shapes,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "chunk": view.to_dict(),
    }


GRAPH_DIGEST_TOOLS: list[dict[str, Any]] = [
    {
        "name": "analyze_graph",
        "description": (
            "Return a shape-free graph digest for the current session's "
            "compiled model: pattern histogram, dim/dtype/quant spectra, "
            "FLOP/byte distributions, critical path, fusion opportunities, "
            "and region index. Default output is the compact prompt "
            "summary; pass ``full=true`` for the full structured dict."
        ),
        "phase": "inspect",
        "handler": analyze_graph,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "full": {"type": "boolean"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "focus_chunk",
        "description": (
            "Return a focused-chunk view of one region (selected by "
            "region_id / pattern_type / cluster_id / node_names). The "
            "view carries both oracle-enumerated DecisionKnobs (safe, "
            "bounded candidates) and an open-ended DoFDescription so "
            "the LLM can propose novel optimisations verified later."
        ),
        "phase": "inspect",
        "handler": focus_chunk,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "selector": {"type": "object"},
                "include_concrete_shapes": {"type": "boolean"},
            },
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "GRAPH_DIGEST_TOOLS",
    "analyze_graph",
    "focus_chunk",
]
