"""Solver request/response artifact writers.

. Every planner writes ``<artifact_dir>/<problem_id>_request.json``
and ``<artifact_dir>/<problem_id>_response.json``; gates audit them.
"""

from __future__ import annotations

import json
from pathlib import Path

from compgen.solve.solver_types import (
    SolverRequest,
    SolverResponse,
    SolverStatus,
)

__all__ = [
    "write_solver_request",
    "write_solver_response",
    "summarize_response_md",
]


def _dump_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False))


def write_solver_request(request: SolverRequest, path: str | Path) -> Path:
    p = Path(path)
    _dump_json(p, request.to_dict())
    return p


def write_solver_response(response: SolverResponse, path: str | Path) -> Path:
    p = Path(path)
    _dump_json(p, response.to_dict())
    return p


def summarize_response_md(response: SolverResponse) -> str:
    """One-screen Markdown summary; suitable for the evidence pack."""

    lines = [
        f"# Solver response: {response.problem_id}",
        "",
        f"- **problem_kind**: `{response.problem_kind.value}`",
        f"- **selected_backend**: `{response.selected_backend.value}`",
        f"- **backend_availability**: `{response.backend_availability.value}`",
        f"- **status**: `{response.status.value}`",
        f"- **formulation_hash**: `{response.formulation_hash}`",
        f"- **time_ms**: {response.time_ms:.3f}",
    ]
    if response.objective_value is not None:
        lines.append(f"- **objective_value**: {response.objective_value}")
    if response.gap_from_optimum is not None:
        lines.append(f"- **gap_from_optimum**: {response.gap_from_optimum}")
    if response.solution_path:
        lines.append(f"- **solution_path**: `{response.solution_path}`")
    if response.certificate_path:
        lines.append(f"- **certificate_path**: `{response.certificate_path}`")
    if response.status in {SolverStatus.INFEASIBLE, SolverStatus.BLOCKED, SolverStatus.ERROR, SolverStatus.UNSUPPORTED, SolverStatus.TIMEOUT, SolverStatus.NOT_PROVED}:
        if response.infeasibility_reason:
            lines.append(f"- **infeasibility_reason**: {response.infeasibility_reason}")
    if response.counterexample is not None:
        lines.append("")
        lines.append("## Counterexample")
        lines.append("```json")
        lines.append(json.dumps(response.counterexample, sort_keys=True, indent=2))
        lines.append("```")
    if response.caveats:
        lines.append("")
        lines.append("## Caveats")
        for caveat in response.caveats:
            lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"
