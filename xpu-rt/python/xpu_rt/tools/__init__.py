"""Tool-promotion pipeline.

Unifies Claude-Code-authored tools under a single ToolCard schema +
typed registry + runner so that a *fresh* Claude session can discover,
call, validate, and trust every promoted tool.

The eight-level maturity ladder (T0 → T7) is enforced by
:mod:`compgen.audit.tool_promotion`; the runner here is the
substrate every maturity level builds on.

Public surface:

* :class:`ToolCard` — frozen dataclass declaring a tool's name,
  maturity, entrypoints (python / cli / mcp), input/output JSON
  schemas, allowed write roots, forbidden actions, promotion
  requirements, tests, skill path, and fresh-agent task id.
* :class:`ToolResult` — typed return from :class:`ToolRunner`.
* :func:`iter_tool_cards` — discover ToolCard YAML files under
  ``python/compgen/tools/cards/``.
* :class:`ToolRunner` — validates input, dispatches to the Python
  entrypoint, validates output, writes ``result.json`` and
  ``trace.jsonl``.

Mirrors the existing ``ProviderCard`` pattern at
:mod:`compgen.providers.provider_types` so future
extensions can lift identical YAML loading conventions.
"""

from __future__ import annotations

from compgen.tools.errors import (
    ToolCardError,
    ToolEntrypointError,
    ToolInputSchemaError,
    ToolMaturityError,
    ToolOutputSchemaError,
    ToolRunError,
)
from compgen.tools.tool_card import (
    FORBIDDEN_ACTIONS,
    MATURITY_LEVELS,
    PROMOTION_REQUIREMENT_KEYS,
    TOOL_PHASES,
    TOOL_STATUSES,
    ToolCard,
    ToolPromotionRequirements,
    ToolTests,
    ToolWrites,
)
from compgen.tools.tool_registry import (
    iter_tool_cards,
    load_tool_card,
    tool_cards_root,
)
from compgen.tools.tool_runner import (
    ToolResult,
    ToolRunner,
    resolve_python_entrypoint,
)
from compgen.tools.skill_lint import (
    REQUIRED_SECTIONS,
    SkillLintReport,
    SkillViolation,
    lint_skill,
)

__all__ = [
    "FORBIDDEN_ACTIONS",
    "MATURITY_LEVELS",
    "PROMOTION_REQUIREMENT_KEYS",
    "REQUIRED_SECTIONS",
    "TOOL_PHASES",
    "TOOL_STATUSES",
    "SkillLintReport",
    "SkillViolation",
    "ToolCard",
    "ToolCardError",
    "ToolEntrypointError",
    "ToolInputSchemaError",
    "ToolMaturityError",
    "ToolOutputSchemaError",
    "ToolPromotionRequirements",
    "ToolResult",
    "ToolRunError",
    "ToolRunner",
    "ToolTests",
    "ToolWrites",
    "iter_tool_cards",
    "lint_skill",
    "load_tool_card",
    "resolve_python_entrypoint",
    "tool_cards_root",
]
