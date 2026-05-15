"""H2 — Phase taxonomy: closed-enum lifecycle + transition matrix.

Section 11 Dream 1: an MCP session lives inside one phase at a time;
``enter_phase`` is the only legal way to move forward; backward
transitions are refused unless explicitly marked ``unsafe``.

The taxonomy is a closed enum — adding a phase requires editing this
module. Tools declare their phase set via the existing ``phase`` field
on each tool dict (already populated by the ToolCard bridge).
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Closed-enum phases
# ---------------------------------------------------------------------------

PHASE_SESSION_INIT: Final[str] = "session_init"
PHASE_GRAPH_ANALYSIS: Final[str] = "graph_analysis"
PHASE_RECIPE_PLANNING: Final[str] = "recipe_planning"
PHASE_RECIPE_AUTHORING: Final[str] = "recipe_authoring"
PHASE_KERNEL_CODEGEN: Final[str] = "kernel_codegen"
PHASE_VERIFICATION: Final[str] = "verification"
PHASE_BUNDLE_EMIT: Final[str] = "bundle_emit"

PHASES: Final[tuple[str, ...]] = (
    PHASE_SESSION_INIT,
    PHASE_GRAPH_ANALYSIS,
    PHASE_RECIPE_PLANNING,
    PHASE_RECIPE_AUTHORING,
    PHASE_KERNEL_CODEGEN,
    PHASE_VERIFICATION,
    PHASE_BUNDLE_EMIT,
)

# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

# Forward-only transitions. Backward / cross-phase moves require the
# unsafe override; see ``is_legal_transition``.
LEGAL_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    PHASE_SESSION_INIT: frozenset({PHASE_GRAPH_ANALYSIS}),
    PHASE_GRAPH_ANALYSIS: frozenset({PHASE_RECIPE_PLANNING}),
    PHASE_RECIPE_PLANNING: frozenset({PHASE_RECIPE_AUTHORING}),
    PHASE_RECIPE_AUTHORING: frozenset({PHASE_KERNEL_CODEGEN}),
    PHASE_KERNEL_CODEGEN: frozenset({PHASE_VERIFICATION}),
    PHASE_VERIFICATION: frozenset({PHASE_BUNDLE_EMIT}),
    PHASE_BUNDLE_EMIT: frozenset(),  # terminal
}


class PhaseTransitionError(ValueError):
    """A typed phase-transition violation."""


def is_known_phase(phase: str) -> bool:
    """True iff ``phase`` is one of the closed-enum phase names."""

    return phase in PHASES


def is_legal_transition(*, from_phase: str | None, to_phase: str, unsafe: bool = False) -> bool:
    """Decide whether ``from_phase -> to_phase`` is allowed.

    * Entering the first phase from ``None`` is always legal.
    * Forward-only transitions must follow :data:`LEGAL_TRANSITIONS`.
    * Backward / cross-phase transitions require ``unsafe=True``.
    """

    if not is_known_phase(to_phase):
        return False
    if from_phase is None:
        return True
    if not is_known_phase(from_phase):
        return False
    if to_phase in LEGAL_TRANSITIONS.get(from_phase, frozenset()):
        return True
    return bool(unsafe)


# ---------------------------------------------------------------------------
# Per-tool phase allowlist
# ---------------------------------------------------------------------------

# Phase -> sequence of tool-name prefixes / globs that are allowed to
# run in that phase. The discovery endpoint filters by this map; the
# dispatch gating uses ``tool_phase`` (the existing ``phase`` field on
# tool dicts) and asserts ``tool_phase in <session.current_phase
# allowlist>``.
PHASE_TOOL_ALLOWLIST: Final[dict[str, tuple[str, ...]]] = {
    PHASE_SESSION_INIT: (
        "open_target",
        "list_*",
        "describe_*",
        "list_phase_tools",
        "list_packaged_examples",
        "compgen_echo",
        "compgen_list_targets",
        "compgen_describe_target",
        "enter_phase",
        "get_context_brief",
        "session_summary",
        "checkpoint",
    ),
    PHASE_GRAPH_ANALYSIS: (
        "analyze_graph",
        "diagnose_model_compatibility",
        "load_model",
        "get_dossier",
        "focus_chunk",
        "list_*",
        "lookup_*",
        "enter_phase",
        "checkpoint",
        "compgen_emit_agent_decision_request",
    ),
    PHASE_RECIPE_PLANNING: (
        "propose_decision",
        "list_pending_*",
        "lookup_*",
        "view_recipe",
        "diff_recipe",
        "explain_verification",
        "query_knowledge",
        "suggest_proposals",
        "enter_phase",
        "compgen_emit_agent_decision_request",
        "compgen_commit_agent_decision_response",
    ),
    PHASE_RECIPE_AUTHORING: (
        "step_proposal",
        "batch_propose",
        "propose_invent_slot",
        "apply_recipe",
        "apply_decision",
        "override_decision",
        "view_recipe",
        "diff_recipe",
        "enter_phase",
        "synthesize_decomp",
        "synthesize_translation",
        "resolve_unsupported_op",
    ),
    PHASE_KERNEL_CODEGEN: (
        "request_kernel_codegen",
        "register_kernel_result",
        "lookup_cached_kernel",
        "request_kernel_bench",
        "register_bench_result",
        "lookup_bench_result",
        "request_refinement",
        "register_refinement_attempt",
        "register_blackbox",
        "request_autotune_trial",
        "register_autotune_pick",
        "request_dispatch_decision",
        "register_dispatch_decision",
        "enter_phase",
    ),
    PHASE_VERIFICATION: (
        "verify_proposal",
        "verify_vendor_package",
        "explain_verification",
        "etc_conformance_run",
        "etc_conformance_summarize",
        "etc_megakernel_inspect",
        "recovery_status",
        "enter_phase",
        "compgen_inspect_pipeline_run",
    ),
    PHASE_BUNDLE_EMIT: (
        "bundle_export",
        "compile",
        "compile_embedded",
        "compgen_compile_torch_model",
        "compgen_compile_torch_model_with_vendor",
        "compgen_run_compiled_bundle",
        "promote_in_session_authored_tools",
        "record_lesson",
        "enter_phase",
    ),
}


def is_tool_allowed_in_phase(tool_name: str, phase: str | None) -> bool:
    """Check whether ``tool_name`` is allowed in ``phase``.

    Returns True when:

    * ``phase`` is None (no enforcement — backwards-compat default);
    * the allowlist for ``phase`` contains ``tool_name`` exactly;
    * or one of the allowlist entries is a glob pattern matched by
      :func:`fnmatch.fnmatchcase`.
    """

    if phase is None:
        return True
    allow = PHASE_TOOL_ALLOWLIST.get(phase)
    if allow is None:
        return False
    import fnmatch

    for entry in allow:
        if entry == tool_name or fnmatch.fnmatchcase(tool_name, entry):
            return True
    return False


__all__ = [
    "LEGAL_TRANSITIONS",
    "PHASES",
    "PHASE_BUNDLE_EMIT",
    "PHASE_GRAPH_ANALYSIS",
    "PHASE_KERNEL_CODEGEN",
    "PHASE_RECIPE_AUTHORING",
    "PHASE_RECIPE_PLANNING",
    "PHASE_SESSION_INIT",
    "PHASE_TOOL_ALLOWLIST",
    "PHASE_VERIFICATION",
    "PhaseTransitionError",
    "is_known_phase",
    "is_legal_transition",
    "is_tool_allowed_in_phase",
]
