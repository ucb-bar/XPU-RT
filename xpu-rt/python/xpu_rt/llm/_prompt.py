"""Shared prompt rendering and response extraction helpers for LLM adapters."""

from __future__ import annotations

import json
from typing import Any

from compgen.llm.base import GenerationRequest


def render_request_prompt(request: GenerationRequest) -> str:
    """Render a provider-agnostic prompt from a request and its context."""
    parts: list[str] = []
    if request.prompt_template:
        parts.append(request.prompt_template.strip())

    ctx = request.context
    _append_section(parts, "IR Summary", ctx.model_ir_summary)
    _append_section(parts, "Target Hardware", ctx.target_profile_summary)
    if ctx.available_transforms:
        _append_section(parts, "Available Transforms", "\n".join(f"- {item}" for item in ctx.available_transforms))
    if ctx.kernel_contracts:
        _append_section(parts, "Kernel Contracts", "\n".join(ctx.kernel_contracts))
    _append_section(parts, "Frontend Diagnostics", ctx.frontend_diagnostics_summary)
    _append_section(parts, "Analysis Dossier", ctx.analysis_dossier_summary)
    _append_section(parts, "Unsupported Operators", ctx.unsupported_operator_summary)
    _append_section(parts, "Extension Packs", ctx.pack_summary)
    _append_section(parts, "Integration Branch", ctx.integration_branch_summary)
    _append_section(parts, "Frontier State", ctx.frontier_summary)
    _append_section(parts, "Legal Actions", ctx.legal_action_summary)
    _append_section(parts, "Hardware Feedback", ctx.hardware_feedback)
    if ctx.prior_attempts:
        _append_section(parts, "Prior Attempts", "\n".join(ctx.prior_attempts))
    _append_section(parts, "Evidence JSON", ctx.evidence_json)

    return "\n\n".join(part for part in parts if part)


def extract_markdown_artifacts(text: str) -> list[str]:
    """Extract fenced-code artifacts from a freeform LLM response."""
    artifacts: list[str] = []
    in_block = False
    block_lines: list[str] = []

    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                artifacts.append("\n".join(block_lines))
                block_lines = []
                in_block = False
            else:
                in_block = True
        elif in_block:
            block_lines.append(line)

    return artifacts or ([text] if text else [])


def parse_json_payload(text: str) -> Any:
    """Parse a JSON object from raw text or JSONL-like output."""
    stripped = text.strip()
    if not stripped:
        raise json.JSONDecodeError("empty payload", text, 0)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    last_error: json.JSONDecodeError | None = None
    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("no JSON payload found", text, 0)


def stringify_json_payload(payload: Any) -> str:
    """Normalize a parsed JSON payload to a compact stable string."""
    return json.dumps(payload, sort_keys=True)


def _append_section(parts: list[str], title: str, body: str) -> None:
    body = body.strip()
    if body:
        parts.append(f"## {title}\n{body}")


__all__ = [
    "extract_markdown_artifacts",
    "parse_json_payload",
    "render_request_prompt",
    "stringify_json_payload",
]
