"""LLM-authored tool descriptor + trial record schema.

An :class:`AuthoredTool` carries the information needed to drive
a trial through :mod:`~compgen.agent.self_extension.sandbox` and to
materialise a live :class:`~compgen.llm.registry.Tool` once the
graduation thresholds are met. Every record in the trial JSONL log
serialises an :class:`AuthoredToolTrial`.

The author (LLM or human) provides:

* ``source`` — the Python source blob.
* ``entry_name`` — the function to invoke (usually equals the tool
  name for clarity).
* ``args`` / ``result`` — typed schema matching :class:`~compgen.llm.registry.Tool`.

Everything else (scoring, gate integration, promotion) lives in sibling
modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# AuthoredTool descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthoredToolSource:
    """The Python source + metadata an LLM authored."""

    source: str
    entry_name: str = "run"
    notes: str = ""

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.source.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AuthoredTool:
    """Typed descriptor for an LLM-authored tool candidate.

    Mirrors :class:`compgen.llm.registry.Tool` on the fields that
    matter for graduation. The ``source`` carries the authored code;
    promotion turns it into a registered ``Tool`` whose ``impl`` calls
    back into the sandboxed entry point.
    """

    name: str
    phase: int
    source: AuthoredToolSource
    description: str = ""
    args_schema: tuple[dict[str, Any], ...] = ()
    result_schema: dict[str, Any] = field(default_factory=dict)
    autocomp_cost_impact: str = "indirect"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "source_digest": self.source.digest,
            "description": self.description,
            "args_schema": list(self.args_schema),
            "result_schema": dict(self.result_schema),
            "autocomp_cost_impact": self.autocomp_cost_impact,
        }


# ---------------------------------------------------------------------------
# Trial record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthoredToolTrial:
    """One pass/fail record written to the trial JSONL log.

    Mirrors the shape of :class:`~compgen.llm.recorder.ToolCallRecord`
    so the same observability primitives apply, but the schema is
    distinct — graduation for authored tools has its own criteria.
    """

    tool_name: str
    source_digest: str
    workload: str
    target: str
    passed: bool
    elapsed_s: float
    session_id: str = ""
    scenario: str = ""
    violation_count: int = 0
    error: str | None = None
    score: float | None = None
    timestamp_iso: str = ""

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def authored_tool_key(tool: AuthoredTool) -> str:
    """Stable key joining tool name + source digest — used by the trial
    log aggregator to group trials for the SAME authored revision."""
    return f"{tool.name}@{tool.source.digest}"


__all__ = [
    "AuthoredTool",
    "AuthoredToolSource",
    "AuthoredToolTrial",
    "authored_tool_key",
]
