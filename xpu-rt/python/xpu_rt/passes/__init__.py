"""Pass-card registry.

Compiler passes are exposed to Claude Code as typed metadata, not as
free-form tool calls. Every pass the agent can request must have a
:class:`PassCard` declaring:

- ``pass_id``                    — stable identifier the agent references
- ``level``                      — IR level (``payload``, ``recipe``, ``tile``, ...)
- ``family``                     — ``fusion``, ``tiling``, ``layout``, ``codegen``, ...
- ``reads`` / ``writes``         — input + output artifact roles
- ``preconditions``              — what must hold for the pass to run
- ``invalidates``                — analyses the pass invalidates (multi-level
                                    invalidation discipline; makes this
                                    enforceable)
- ``preserves_refinement``       — ``bit_equality`` / ``tolerance_eps`` / ``unknown``
- ``verification``               — required verification rungs (``structural``,
                                    ``differential``, ``formal``)
- ``cost``                       — ``cheap`` / ``medium`` / ``expensive``
- ``failure_modes``              — typed reasons the pass can refuse to run
- ``mcp_tool``                   — when applicable, the MCP tool a fresh
                                    Claude session would invoke
- ``example_invocation``         — minimal request payload

The registry loads every YAML under ``docs/generated/pass_cards/`` and
asserts the agent's ``passes_allowed`` field references only resolved
IDs. Section 20 builds on this foundation.
"""

from __future__ import annotations

from compgen.passes.cards import (
    PASS_FAMILIES,
    PASS_LEVELS,
    PASS_PHASES,
    PASS_SOURCES,
    REFINEMENT_KINDS,
    PassCard,
    PassCardError,
    PassCardRegistry,
    default_phase_for_family,
    default_registry_root,
    iter_cards,
    load_card,
    phase_index,
    resolve_card_path,
    validate_card,
)
from compgen.passes.scheduler import (
    PassPlanReport,
    PassPlanStep,
    assert_pass_plan_valid,
    inspect_pass_plan,
)

__all__ = [
    "PASS_FAMILIES",
    "PASS_LEVELS",
    "PASS_PHASES",
    "PASS_SOURCES",
    "REFINEMENT_KINDS",
    "PassCard",
    "PassCardError",
    "PassCardRegistry",
    "PassPlanReport",
    "PassPlanStep",
    "assert_pass_plan_valid",
    "default_phase_for_family",
    "default_registry_root",
    "inspect_pass_plan",
    "iter_cards",
    "load_card",
    "phase_index",
    "resolve_card_path",
    "validate_card",
]
