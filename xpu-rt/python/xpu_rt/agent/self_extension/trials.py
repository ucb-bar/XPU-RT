"""Execute LLM-authored tools against a scenario + record trials.

A :class:`TrialScenario` bundles (1) the kwargs to hand the sandboxed
entry point, (2) a scoring callable that decides pass/fail, and (3)
labels identifying the (workload, target) pair this trial counts for.

The trial runner is deliberately thin — it's the glue between the
sandbox and the JSONL log. Callers own the scoring logic; we only
record its boolean verdict (plus optional score).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.self_extension.authored_tool import (
    AuthoredTool,
    AuthoredToolTrial,
)
from compgen.agent.self_extension.sandbox import SandboxResult, sandbox_invoke

log = structlog.get_logger()


DEFAULT_TRIAL_LOG_SUBDIR = "authored_trials.jsonl"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class TrialScenario:
    """One trial specification.

    Attributes:
        workload: A free-form workload label (e.g. ``"distilbert"``).
        target: A free-form target label (e.g. ``"cuda_a100"``).
        kwargs: Arguments forwarded to the authored tool's entry point.
        scorer: Callable ``(sandbox_result) -> (passed: bool, score: float|None)``.
            Receives the raw :class:`SandboxResult` (including the
            authored tool's return value). Must NEVER raise — if the
            scorer might fail, catch internally and return ``(False, None)``.
        name: Human-readable scenario id for the transcript.
        timeout_s: Wall-clock cap for the sandbox invocation.
    """

    workload: str
    target: str
    scorer: Callable[[SandboxResult], tuple[bool, float | None]]
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    timeout_s: float = 5.0


# ---------------------------------------------------------------------------
# Log location resolution
# ---------------------------------------------------------------------------


def default_trial_log_path() -> Path:
    """Resolve the trial log path.

    Follows the same convention as the other session-scoped state:
    honour ``COMPGEN_SESSION_DIR`` first, then fall back to
    ``~/.compgen/transcripts/authored_trials.jsonl``.
    """
    env = os.environ.get("COMPGEN_SESSION_DIR")
    if env:
        return Path(env).expanduser() / DEFAULT_TRIAL_LOG_SUBDIR
    return Path("~/.compgen/transcripts").expanduser() / DEFAULT_TRIAL_LOG_SUBDIR


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------


def record_trial(
    trial: AuthoredToolTrial,
    *,
    log_path: Path | None = None,
) -> Path:
    """Append one trial record to the JSONL log; return the log path."""
    path = log_path or default_trial_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(trial.to_json_line() + "\n")
    return path


def run_trial(
    tool: AuthoredTool,
    scenario: TrialScenario,
    *,
    session_id: str = "",
    log_path: Path | None = None,
) -> AuthoredToolTrial:
    """Invoke ``tool`` sandboxed against ``scenario`` and record the result.

    Never raises — every failure mode (sandbox violation, scorer
    indecision, authored-tool exception) lands as ``passed=False`` in
    the trial record so the graduation path remains deterministic.
    """
    started = time.perf_counter()
    sandbox = sandbox_invoke(
        tool.source.source,
        tool.source.entry_name,
        kwargs=scenario.kwargs,
        timeout_s=scenario.timeout_s,
    )
    try:
        passed, score = scenario.scorer(sandbox)
    except Exception as exc:  # noqa: BLE001
        # Scorer bug → fail closed. Record the reason.
        passed, score = False, None
        sandbox.error = (sandbox.error or "") + f" | scorer_raised: {exc}"

    elapsed = time.perf_counter() - started
    trial = AuthoredToolTrial(
        tool_name=tool.name,
        source_digest=tool.source.digest,
        workload=scenario.workload,
        target=scenario.target,
        passed=bool(passed and sandbox.ok),
        elapsed_s=round(elapsed, 6),
        session_id=session_id,
        scenario=scenario.name or f"{scenario.workload}:{scenario.target}",
        violation_count=len(sandbox.violations),
        error=sandbox.error,
        score=score,
        timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    record_trial(trial, log_path=log_path)
    log.info(
        "self_extension.trial",
        tool=tool.name,
        passed=trial.passed,
        workload=scenario.workload,
        target=scenario.target,
    )
    return trial


__all__ = [
    "TrialScenario",
    "default_trial_log_path",
    "record_trial",
    "run_trial",
]
