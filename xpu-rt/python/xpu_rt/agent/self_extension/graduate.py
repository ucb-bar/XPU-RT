"""N-pass auto-graduation for LLM-authored tools.

Scans the trial JSONL log, counts passing trials per
``(tool_name, source_digest)``, and promotes each authored tool that
clears:

* ``min_passes`` (default 5) — total passing trials.
* ``min_workloads`` (default 2) — distinct workload labels.
* ``min_targets`` (default 2) — distinct target labels.

The promotion step compiles the authored source in
:func:`~compgen.agent.self_extension.sandbox.sandbox_invoke` at
registry-call time (so every invocation is still sandboxed) and
registers a real :class:`~compgen.llm.registry.Tool` whose ``impl``
delegates into the sandbox.

Idempotent: a state file alongside the trial log records which
``(tool_name, source_digest)`` pairs have already graduated; reruns
are no-ops. When the authored source changes (different digest), the
new revision is tracked separately and must reclear the thresholds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from compgen.agent.self_extension.authored_tool import (
    AuthoredTool,
    AuthoredToolSource,
)
from compgen.agent.self_extension.sandbox import sandbox_invoke
from compgen.agent.self_extension.trials import default_trial_log_path

if TYPE_CHECKING:   # pragma: no cover
    from compgen.llm.registry import Registry

log = structlog.get_logger()


GRADUATION_STATE_FILENAME = "authored_graduations.json"


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class AuthoredGraduationReport:
    trials_scanned: int = 0
    candidates_found: int = 0
    candidates_already_applied: int = 0
    new_tools_registered: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _Aggregate:
    tool_name: str = ""
    source_digest: str = ""
    source: str = ""
    entry_name: str = "run"
    passed_trials: int = 0
    total_trials: int = 0
    workloads: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)
    last_error: str | None = None


def _aggregate_trials(
    log_path: Path,
) -> tuple[dict[str, _Aggregate], int]:
    """Walk JSONL; group by (tool_name, source_digest)."""
    agg: dict[str, _Aggregate] = defaultdict(_Aggregate)
    total_lines = 0
    if not log_path.exists():
        return dict(agg), 0
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        total_lines += 1
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = entry.get("tool_name")
        digest = entry.get("source_digest")
        if not name or not digest:
            continue
        key = f"{name}@{digest}"
        a = agg[key]
        a.tool_name = name
        a.source_digest = digest
        a.total_trials += 1
        if entry.get("passed"):
            a.passed_trials += 1
        if entry.get("workload"):
            a.workloads.add(entry["workload"])
        if entry.get("target"):
            a.targets.add(entry["target"])
        if not entry.get("passed") and entry.get("error"):
            a.last_error = str(entry["error"])
    return dict(agg), total_lines


# ---------------------------------------------------------------------------
# Graduation state
# ---------------------------------------------------------------------------


def _state_path_for(log_path: Path) -> Path:
    return log_path.parent / GRADUATION_STATE_FILENAME


def _load_state(log_path: Path) -> dict[str, Any]:
    p = _state_path_for(log_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:   # noqa: BLE001
        return {}


def _save_state(log_path: Path, state: dict[str, Any]) -> None:
    p = _state_path_for(log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, default=str))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tool construction
# ---------------------------------------------------------------------------


def _materialise_tool(
    agg: _Aggregate, author: AuthoredTool,
):
    """Build a :class:`Tool` whose impl sandboxes the authored source."""
    from compgen.llm.registry import Tool, ToolArg, ToolResult

    source_snapshot = author.source.source
    entry_name = author.source.entry_name

    def _impl(**kwargs: Any) -> dict[str, Any]:
        # Every invocation re-sandboxes. This is the point: graduated
        # authored tools stay bounded by the same policy that green-lit
        # their trials.
        r = sandbox_invoke(
            source_snapshot, entry_name, kwargs=kwargs,
        )
        return {
            "status": "ok" if r.ok else "sandbox_failed",
            "authored": True,
            "tool_name": author.name,
            "source_digest": author.source.digest,
            "elapsed_s": r.elapsed_s,
            "value": r.value,
            "violations": [
                {"kind": v.kind, "detail": v.detail, "location": v.location}
                for v in r.violations
            ],
            "error": r.error,
        }

    args = tuple(
        ToolArg(
            name=a.get("name", "arg"),
            dtype=a.get("dtype", "any"),
            description=a.get("description", ""),
            required=a.get("required", False),
        )
        for a in author.args_schema
    )
    return Tool(
        name=f"{author.name}__authored",
        phase=author.phase,
        kind="tool",
        wraps_pass=f"authored_tool:{author.source.digest}",
        autocomp_cost_impact=author.autocomp_cost_impact,   # type: ignore[arg-type]
        args=args,
        result=ToolResult(
            dtype=author.result_schema.get("dtype", "dict"),
            description=author.result_schema.get(
                "description",
                "Return value wrapped by the self-extension sandbox.",
            ),
        ),
        description=author.description or (
            f"LLM-authored tool graduated after {agg.passed_trials} "
            f"passes across {len(agg.workloads)} workloads and "
            f"{len(agg.targets)} targets."
        ),
        impl=_impl,
        notes=f"source_digest={author.source.digest}",
        stub=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def promote_authored_tools(
    registry: "Registry",
    *,
    authored_index: dict[str, AuthoredTool] | None = None,
    log_path: Path | None = None,
    min_passes: int = 5,
    min_workloads: int = 2,
    min_targets: int = 2,
) -> AuthoredGraduationReport:
    """Scan the trial log, materialise authored tools that cleared thresholds.

    Args:
        registry: The live registry to mutate.
        authored_index: Optional mapping ``"name@digest" -> AuthoredTool``
            supplying the full source for tools that appear in the log.
            When a candidate is found in the log but not in the index,
            it's skipped with an error — we never synthesise source we
            don't have a cryptographic record of.
        log_path: JSONL trial log. Defaults to
            :func:`default_trial_log_path`.
        min_passes / min_workloads / min_targets: Graduation thresholds.

    Returns:
        :class:`AuthoredGraduationReport`.
    """
    report = AuthoredGraduationReport()
    path = log_path or default_trial_log_path()

    aggregates, total_lines = _aggregate_trials(path)
    report.trials_scanned = total_lines
    if not aggregates:
        return report

    state = _load_state(path)
    applied: dict[str, Any] = state.setdefault("applied", {})

    for key, a in aggregates.items():
        if a.passed_trials < min_passes:
            continue
        if len(a.workloads) < min_workloads:
            continue
        if len(a.targets) < min_targets:
            continue
        report.candidates_found += 1

        if key in applied:
            report.candidates_already_applied += 1
            continue

        authored = (authored_index or {}).get(key)
        if authored is None:
            report.errors.append(
                f"{key}: aggregate cleared thresholds but no AuthoredTool "
                f"supplied in authored_index; skipping."
            )
            continue

        try:
            tool = _materialise_tool(a, authored)
            registry.register_tool(tool)
            applied[key] = {
                "tool_name": tool.name,
                "source_digest": a.source_digest,
                "passed_trials": a.passed_trials,
                "workloads": sorted(a.workloads),
                "targets": sorted(a.targets),
            }
            report.new_tools_registered.append({
                "tool_name": tool.name,
                "source_digest": a.source_digest,
                "passed_trials": a.passed_trials,
                "workloads": sorted(a.workloads),
                "targets": sorted(a.targets),
            })
            log.info(
                "self_extension.graduated",
                tool=tool.name, passed_trials=a.passed_trials,
            )
        except Exception as exc:   # noqa: BLE001
            report.errors.append(f"register_tool({key}): {type(exc).__name__}: {exc}")

    if report.new_tools_registered:
        _save_state(path, state)

    return report


__all__ = [
    "AuthoredGraduationReport",
    "GRADUATION_STATE_FILENAME",
    "promote_authored_tools",
]
