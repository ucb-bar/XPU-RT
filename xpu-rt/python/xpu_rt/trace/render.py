"""Render ``trace.jsonl`` as a human-readable companion ``trace.log``.

The rendered view keeps NDJSON as the canonical machine log and adds a
grep-friendly plain-text companion:

    [15:46:21.412] [INFO ] <stage:encoding> start                      evt_0000000003
      [15:46:21.413] [INFO ] <pass:fold_transposes_into_dots> start    evt_0000000004
        [15:46:21.414] [DEBUG] <ir_dump:before> hash=sha256:abc123      evt_0000000005
      [15:46:21.425] [INFO ] <pass:fold_transposes_into_dots> end +12.3ms  evt_0000000006

Indent reflects the ``parent_event_id`` chain depth. Rendering is a
pure function of the JSONL file — called once at end-of-compile by
``api.compile_model``. Nothing else in the pipeline reads this file;
``trace.jsonl`` remains the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compgen.trace.events import category_for

_MAX_DETAIL_CHARS = 120
_SNIPPET_WIDTH = 140


def _snippet_lines(text: str, *, max_lines: int) -> list[str]:
    """Return up to ``max_lines`` compact, width-capped lines from ``text``.

    Used to echo prompt/response/rationale previews beneath trace.log
    entries so a reader sees the decision process at a glance.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in str(text).splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if len(stripped) > _SNIPPET_WIDTH:
            stripped = stripped[: _SNIPPET_WIDTH - 1] + "…"
        out.append(stripped)
        if len(out) >= max_lines:
            break
    return out


def _format_detail(event: dict[str, Any]) -> str:
    """Compact key=value tail for the rendered line.

    Picks the handful of payload fields that matter per-kind; anything
    else stays in ``trace.jsonl`` for jq.
    """
    kind = event.get("kind", "")
    p: dict[str, Any] = event.get("payload", {}) or {}
    phase = event.get("phase", "")
    parts: list[str] = []

    if phase == "end":
        elapsed = event.get("elapsed_ms", 0.0)
        parts.append(f"+{elapsed:.1f}ms")
        stats = p.get("stats")
        if stats:
            parts.append(f"stats={stats}")

    if kind == "ir_dump":
        if p.get("phase_tag"):
            parts.append(f"phase={p['phase_tag']}")
        if p.get("ir_hash"):
            parts.append(f"hash={p['ir_hash']}")
    elif kind == "pass_run":
        if p.get("ir_hash_before") and p.get("ir_hash_after"):
            parts.append(f"{p['ir_hash_before']}→{p['ir_hash_after']}")
    elif kind == "decision":
        if p.get("chosen"):
            parts.append(f"chosen={p['chosen']}")
        if p.get("llm_turn_id"):
            parts.append(f"llm_turn={p['llm_turn_id']}")
    elif kind == "mcp_call":
        if p.get("tool"):
            parts.append(f"tool={p['tool']}")
        if p.get("error"):
            parts.append(f"error={p['error']!r}")
        if p.get("duration_ms") is not None:
            parts.append(f"dur={p['duration_ms']}ms")
    elif kind == "oracle_advisory":
        for k in ("confidence", "rationale"):
            if k in p:
                parts.append(f"{k}={p[k]!r}" if k == "rationale" else f"{k}={p[k]}")
    elif kind == "llm_response":
        for k in ("model", "prompt_tokens", "completion_tokens", "latency_ms"):
            if p.get(k):
                parts.append(f"{k}={p[k]}")
        if p.get("has_reasoning"):
            parts.append("reasoning=yes")
        if p.get("num_artifacts"):
            parts.append(f"artifacts={p['num_artifacts']}")
        if p.get("log_file"):
            parts.append(f"log={p['log_file']}")
    elif kind == "llm_prompt":
        if p.get("artifact_type"):
            parts.append(f"artifact={p['artifact_type']}")
    elif kind == "analysis_run" and phase != "end":
        if p.get("target"):
            parts.append(f"target={p['target']}")

    tail = " ".join(parts)
    if len(tail) > _MAX_DETAIL_CHARS:
        tail = tail[: _MAX_DETAIL_CHARS - 1] + "…"
    return tail


def _time_only(ts: str) -> str:
    # "2026-04-22T15:46:21.412Z" -> "15:46:21.412"
    if "T" in ts:
        return ts.split("T", 1)[1].rstrip("Z")
    return ts


def render_trace(trace_path: Path, *, out_path: Path | None = None) -> Path | None:
    """Render a ``trace.jsonl`` file as a human-readable ``trace.log``.

    Args:
        trace_path: The canonical NDJSON trace.
        out_path: Destination file. Defaults to ``trace.log`` next to
            ``trace_path``.

    Returns:
        The written path, or ``None`` if the input does not exist.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return None
    out_path = Path(out_path) if out_path is not None else trace_path.with_name("trace.log")

    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Compute parent-chain depth per event_id.
    parent_by_id: dict[str, str] = {e["event_id"]: e.get("parent_event_id", "") for e in events if "event_id" in e}
    by_id: dict[str, dict[str, Any]] = {e["event_id"]: e for e in events if "event_id" in e}
    depth_cache: dict[str, int] = {}

    def depth(eid: str) -> int:
        if eid in depth_cache:
            return depth_cache[eid]
        parent = parent_by_id.get(eid, "")
        if not parent or parent == eid:
            depth_cache[eid] = 0
            return 0
        # Guard against cycles with a visited set.
        seen = {eid}
        d = 0
        cur = parent
        while cur and cur not in seen:
            seen.add(cur)
            d += 1
            cur = parent_by_id.get(cur, "")
        depth_cache[eid] = d
        return d

    lines: list[str] = []
    for e in events:
        eid = e.get("event_id", "")
        d = depth(eid)
        indent = "  " * d
        ts = _time_only(e.get("ts", ""))
        level = f"{e.get('level', 'INFO'):<5}"
        phase = e.get("phase", "")
        # ``end`` events carry only ``span_id`` in their payload — the
        # descriptive fields (name, stage_name, …) live on the ``start``
        # event. Look up the start to keep the category useful.
        cat_payload = e.get("payload") or {}
        if phase == "end":
            span_id = cat_payload.get("span_id")
            start_evt = by_id.get(span_id) if span_id else None
            if start_evt:
                cat_payload = start_evt.get("payload") or cat_payload
        cat = category_for(e.get("kind", ""), cat_payload)
        phase_tag = f" {phase}" if phase and phase != "point" else ""
        detail = _format_detail(e)
        suffix = f"  {detail}" if detail else ""
        lines.append(f"{indent}[{ts}] [{level}] <{cat}>{phase_tag} {eid}{suffix}")
        # Optional inline preview for prompt/response so a human reader
        # sees the decision process without jumping to a file. The
        # full text always lives in ``trace/turns/NNNN_*.md`` and in
        # the LLMRecorder JSON — we only echo the first few lines.
        preview_payload = e.get("payload") or {}
        kind = e.get("kind", "")
        if kind == "llm_prompt":
            prev = preview_payload.get("prompt_preview") or ""
            for snip in _snippet_lines(prev, max_lines=3):
                lines.append(f"{indent}    │ prompt: {snip}")
        elif kind == "llm_response":
            prev = preview_payload.get("raw_text_preview") or ""
            for snip in _snippet_lines(prev, max_lines=3):
                lines.append(f"{indent}    │ response: {snip}")
            rprev = preview_payload.get("reasoning_preview") or ""
            if rprev:
                for snip in _snippet_lines(rprev, max_lines=2):
                    lines.append(f"{indent}    │ reasoning: {snip}")
        elif kind == "decision":
            rationale = preview_payload.get("rationale") or ""
            if rationale:
                for snip in _snippet_lines(rationale, max_lines=2):
                    lines.append(f"{indent}    │ rationale: {snip}")

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


__all__ = ["render_trace"]
