"""Self-extension: LLM-authored tools that auto-graduate after N trials.

P4 of the LLM-driven compilation surface plan. The LLM proposes a
Python source blob for a new Tool. The source runs inside
:mod:`~compgen.agent.self_extension.sandbox` (restricted imports +
wall-clock cap), its output is scored against a differential scenario,
and the trial is appended to a JSONL log. After N passing trials
across >=2 distinct (workload, target) pairs, the tool graduates into
the live :class:`~compgen.llm.registry.Registry`.

Public API (stable):

- :class:`AuthoredTool` — typed descriptor for an authored tool.
- :func:`run_trial` — execute one authored tool against a scenario.
- :func:`promote_authored_tools` — scan the trial log and materialise
  passing tools into the registry (idempotent).

Nothing here bypasses the verification ladder; it composes on top of
it. The trial scorer is always a callable the *caller* supplies,
never something the LLM chooses — the sandbox only executes code, it
does not decide what "good" means.
"""

from __future__ import annotations

from compgen.agent.self_extension._index import (
    clear_authored_index,
    register_authored_tool,
    snapshot_authored_index,
)
from compgen.agent.self_extension.authored_tool import (
    AuthoredTool,
    AuthoredToolTrial,
    AuthoredToolSource,
)
from compgen.agent.self_extension.graduate import (
    AuthoredGraduationReport,
    promote_authored_tools,
)
from compgen.agent.self_extension.sandbox import (
    SandboxResult,
    SandboxViolation,
    sandbox_invoke,
)
from compgen.agent.self_extension.trials import (
    TrialScenario,
    record_trial,
    run_trial,
)

__all__ = [
    "AuthoredGraduationReport",
    "AuthoredTool",
    "AuthoredToolSource",
    "AuthoredToolTrial",
    "SandboxResult",
    "SandboxViolation",
    "TrialScenario",
    "clear_authored_index",
    "promote_authored_tools",
    "record_trial",
    "register_authored_tool",
    "run_trial",
    "sandbox_invoke",
]
