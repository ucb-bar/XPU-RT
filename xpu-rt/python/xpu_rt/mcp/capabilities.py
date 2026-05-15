"""H3 — capability tokens + per-tool gating (Section 11 Dream 3).

A capability token is a small, named permission an operator grants
when opening the session. High-risk tools declare a
``capabilities_required`` set + a ``caller_must_be`` role; dispatch
refuses calls that don't match.

Like H2 phase gating, H3 capability gating is opt-in via
``COMPGEN_STRICT_CAPABILITIES=1`` (default off). When opt-in is
active, the dispatch envelope returns
``ToolDelta(status="blocked", blocked_reason="capability_missing")``
on refusal.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Closed-enum tokens
# ---------------------------------------------------------------------------

CAPABILITY_OPERATOR_OVERRIDE: Final[str] = "operator_override"
CAPABILITY_PROVIDER_ROLE: Final[str] = "provider_role"
CAPABILITY_AGENT_ROLE: Final[str] = "agent_role"
CAPABILITY_HUMAN_OPERATOR: Final[str] = "human_operator"
CAPABILITY_PROMOTE_TO_LIBRARY: Final[str] = "promote_to_library"
CAPABILITY_REGISTER_KERNEL_RESULT: Final[str] = "register_kernel_result"
CAPABILITY_BUNDLE_EXPORT: Final[str] = "bundle_export"

CAPABILITY_TOKENS: Final[tuple[str, ...]] = (
    CAPABILITY_OPERATOR_OVERRIDE,
    CAPABILITY_PROVIDER_ROLE,
    CAPABILITY_AGENT_ROLE,
    CAPABILITY_HUMAN_OPERATOR,
    CAPABILITY_PROMOTE_TO_LIBRARY,
    CAPABILITY_REGISTER_KERNEL_RESULT,
    CAPABILITY_BUNDLE_EXPORT,
)

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

ROLE_AGENT: Final[str] = "agent"
ROLE_OPERATOR: Final[str] = "operator"
ROLE_KERNEL_PROVIDER: Final[str] = "kernel_provider"

ROLES: Final[tuple[str, ...]] = (
    ROLE_AGENT,
    ROLE_OPERATOR,
    ROLE_KERNEL_PROVIDER,
)

# ---------------------------------------------------------------------------
# Per-tool requirements
# ---------------------------------------------------------------------------

# Tool-name -> (required tokens, required caller role | None).
# Tools not in this map require no capabilities (read-only default).
TOOL_REQUIREMENTS: Final[dict[str, tuple[frozenset[str], str | None]]] = {
    "override_decision": (frozenset({CAPABILITY_OPERATOR_OVERRIDE}), None),
    "register_kernel_result": (
        frozenset({CAPABILITY_PROVIDER_ROLE, CAPABILITY_REGISTER_KERNEL_RESULT}),
        ROLE_KERNEL_PROVIDER,
    ),
    "bundle_export": (frozenset({CAPABILITY_BUNDLE_EXPORT}), None),
    "apply_recipe": (frozenset({CAPABILITY_AGENT_ROLE}), None),
    "promote_in_session_authored_tools": (
        frozenset({CAPABILITY_PROMOTE_TO_LIBRARY}),
        None,
    ),
}


class CapabilityError(ValueError):
    """A typed capability-mismatch."""


def is_known_token(token: str) -> bool:
    """True iff ``token`` is in :data:`CAPABILITY_TOKENS`."""

    return token in CAPABILITY_TOKENS


def is_known_role(role: str) -> bool:
    """True iff ``role`` is in :data:`ROLES`."""

    return role in ROLES


def missing_capabilities(
    *,
    tool_name: str,
    session_caps: frozenset[str],
    caller_role: str,
) -> tuple[frozenset[str], str | None]:
    """Return (missing_tokens, role_mismatch) for ``tool_name``.

    Both elements are empty/None when the call passes; otherwise the
    return value identifies the specific gate that failed so callers
    can build a typed ``ToolDelta(blocked_reason=capability_missing)``.
    """

    req = TOOL_REQUIREMENTS.get(tool_name)
    if req is None:
        return frozenset(), None
    required_tokens, required_role = req
    missing = required_tokens - session_caps
    role_mismatch = None
    if required_role is not None and required_role != caller_role:
        role_mismatch = required_role
    return frozenset(missing), role_mismatch


__all__ = [
    "CAPABILITY_AGENT_ROLE",
    "CAPABILITY_BUNDLE_EXPORT",
    "CAPABILITY_HUMAN_OPERATOR",
    "CAPABILITY_OPERATOR_OVERRIDE",
    "CAPABILITY_PROMOTE_TO_LIBRARY",
    "CAPABILITY_PROVIDER_ROLE",
    "CAPABILITY_REGISTER_KERNEL_RESULT",
    "CAPABILITY_TOKENS",
    "CapabilityError",
    "ROLES",
    "ROLE_AGENT",
    "ROLE_KERNEL_PROVIDER",
    "ROLE_OPERATOR",
    "TOOL_REQUIREMENTS",
    "is_known_role",
    "is_known_token",
    "missing_capabilities",
]
