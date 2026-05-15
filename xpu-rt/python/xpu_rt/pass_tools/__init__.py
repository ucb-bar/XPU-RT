"""Pass-tool substrate.

A ``PassToolCard`` makes a Recipe-IR-rewriting pass into a typed,
agent-callable tool. The existing :mod:`compgen.passes.cards`
``PassCard`` registry covers the compiler-internal pass scheduler;
this module is its agent-facing twin: every pass tool emits a
``pass_tool_result_v1`` with a ``recipe_delta`` — never a direct
Payload-IR mutation.
"""

from __future__ import annotations

from compgen.pass_tools.pass_tool_types import (
    PassToolCard,
    PassToolCardError,
)

__all__ = [
    "PassToolCard",
    "PassToolCardError",
]
