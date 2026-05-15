"""LLM interaction recorder middleware.

Every LLM call in CompGen passes through the recorder. It wraps any
CompGenLLMProtocol implementor and logs all interactions to disk as
JSON for reproducibility, debugging, and cost tracking.

Each interaction is saved as a JSON file with:
- The full prompt (rendered)
- The raw response text
- Model ID, token counts, latency
- Timestamp
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.llm._prompt import render_request_prompt
from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


@dataclass
class LLMRecorder:
    """Recording middleware for LLM interactions.

    Wraps any CompGenLLMProtocol client and logs all calls to disk.

    Cost: **no extra LLM traffic**. The recorder only captures the
    request we were already sending and the response the provider was
    already returning. ``reasoning`` / ``thought_process`` fields are
    only persisted if the provider surfaced them in
    :attr:`GenerationResponse.metadata` (some APIs expose native
    chain-of-thought as a side-channel at no extra cost).
    """

    wrapped: CompGenLLMProtocol
    log_dir: Path
    enabled: bool = True
    _call_count: int = 0
    # Path of the JSON file written by the most recent ``_record`` call.
    # :class:`TracingLLMRecorder` reads it to stamp the trace event.
    last_log_path: Path | None = None

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Forward to wrapped client and record the interaction."""
        t0 = time.time()
        response = self.wrapped.generate(request)
        self._record(request, response, t0)
        return response

    def generate_structured(self, request: GenerationRequest, schema: dict[str, Any]) -> GenerationResponse:
        """Forward structured generation and record."""
        t0 = time.time()
        response = self.wrapped.generate_structured(request, schema)
        self._record(request, response, t0, schema=schema)
        return response

    def _record(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
        start_time: float,
        schema: dict[str, Any] | None = None,
    ) -> Path | None:
        """Write interaction to disk as JSON. Returns the written path or None."""
        if not self.enabled:
            return None

        self._call_count += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt text for hashing
        prompt_text = render_request_prompt(request)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]

        # Some providers return native reasoning as a side-channel on
        # ``metadata`` — capture it when present (does NOT request it).
        meta = response.metadata or {}
        reasoning = (
            meta.get("reasoning") or meta.get("thought_process") or meta.get("thinking") or meta.get("reasoning_text")
        )

        record = {
            "call_id": self._call_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "model": response.model_id,
            "config": {
                "temperature": request.config.temperature,
                "max_tokens": request.config.max_tokens,
                "top_p": request.config.top_p,
            },
            "prompt_hash": prompt_hash,
            "prompt": prompt_text,
            "prompt_preview": prompt_text[:500],
            "artifact_type": request.artifact_type,
            "context": {
                "target_profile_summary": request.context.target_profile_summary,
                "available_transforms": list(request.context.available_transforms),
                "kernel_contracts": list(request.context.kernel_contracts),
                "frontend_diagnostics_summary": request.context.frontend_diagnostics_summary,
                "analysis_dossier_summary": request.context.analysis_dossier_summary,
                "unsupported_operator_summary": request.context.unsupported_operator_summary,
                "pack_summary": request.context.pack_summary,
                "integration_branch_summary": request.context.integration_branch_summary,
                "frontier_summary": request.context.frontier_summary,
                "legal_action_summary": request.context.legal_action_summary,
            },
            "response": {
                "raw_text_length": len(response.raw_text),
                # Full text — previously only a 200-char preview was
                # persisted, which made the agent's decision process
                # opaque. Preview is kept as a separate field for
                # quick scans.
                "raw_text": response.raw_text,
                "raw_text_preview": response.raw_text[:500],
                "parsed_artifacts": list(response.parsed_artifacts or []),
                "num_artifacts": len(response.parsed_artifacts),
                "reasoning": reasoning,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
            },
            "metadata": meta,
        }

        if schema:
            record["schema"] = schema

        filename = f"{self._call_count:04d}_{prompt_hash}.json"
        out_path = self.log_dir / filename
        out_path.write_text(json.dumps(record, indent=2, default=str))
        self.last_log_path = out_path
        return out_path

    def replay_log(self, log_path: Path) -> GenerationResponse:
        """Load a previously recorded response from a log file."""
        data = json.loads(log_path.read_text())
        return GenerationResponse(
            raw_text=data.get("response", {}).get("raw_text_preview", ""),
            parsed_artifacts=[],
            model_id=data.get("model", ""),
            prompt_tokens=data.get("response", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("response", {}).get("completion_tokens", 0),
            latency_ms=data.get("response", {}).get("latency_ms", 0),
        )

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls recorded."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Approximate total tokens from log files."""
        total = 0
        if self.log_dir.exists():
            for f in self.log_dir.glob("*.json"):
                data = json.loads(f.read_text())
                resp = data.get("response", {})
                total += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
        return total


@dataclass
class ToolCallRecord:
    """One entry in a ToolCallRecorder JSONL log.

    Captures the shape specified in
    ``user_perspective/analysis/llm_control_boundaries.md`` §Recorder
    contract.
    """

    phase: int
    llm_turn_id: str
    kind: str  # tool_call | invent_proposal | observation | verification
    name: str  # tool or invent-slot name
    args: dict[str, Any]
    result: dict[str, Any]
    select_vs_invent: str  # select | invent | na
    recipe_ir_diff: dict[str, Any]  # {before_hash, after_hash, op_delta}
    gate_result: dict[str, Any] | None
    timestamp_iso: str
    elapsed_ms: int

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "llm_turn_id": self.llm_turn_id,
            "kind": self.kind,
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "select_vs_invent": self.select_vs_invent,
            "recipe_ir_diff": self.recipe_ir_diff,
            "gate_result": self.gate_result,
            "timestamp_iso": self.timestamp_iso,
            "elapsed_ms": self.elapsed_ms,
        }
        return json.dumps(payload, sort_keys=True, default=str)


@dataclass
class ToolCallRecorder:
    """JSONL recorder for LLM tool + invent-slot calls.

    Sibling to ``LLMRecorder`` which handles raw LLM API calls. This
    class handles the *higher-level* tool-and-invent-slot invocations
    that sit on top of the LLM's text generations.

    Opens / appends to ``log_path`` (one JSONL file per compilation
    run). Computes a sha256 diff-hash for arbitrary "before" / "after"
    IR-like objects via ``hash_ir()``; when the real Recipe-IR dialect
    produces serializable ops, callers pass those directly.
    """

    log_path: Path
    enabled: bool = True
    _call_count: int = 0

    def hash_ir(self, obj: Any) -> str:
        """Stable SHA-256 hash of any JSON-serializable object.

        Used to populate ``recipe_ir_diff.{before_hash, after_hash}``
        without requiring the full IR to be logged. Callers that want
        full logging should pass the serialized IR as ``op_delta``.
        """
        try:
            serialized = json.dumps(obj, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = repr(obj)
        return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def record(
        self,
        *,
        phase: int,
        name: str,
        kind: str = "tool_call",
        args: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        select_vs_invent: str = "na",
        before: Any = None,
        after: Any = None,
        op_delta: Any = None,
        gate_result: dict[str, Any] | None = None,
        elapsed_ms: int = 0,
        llm_turn_id: str = "",
    ) -> ToolCallRecord:
        """Append one record. Returns the created record for test assertions."""
        record = ToolCallRecord(
            phase=phase,
            llm_turn_id=llm_turn_id,
            kind=kind,
            name=name,
            args=args or {},
            result=result or {},
            select_vs_invent=select_vs_invent,
            recipe_ir_diff={
                "before_hash": self.hash_ir(before) if before is not None else "",
                "after_hash": self.hash_ir(after) if after is not None else "",
                "op_delta": op_delta if op_delta is not None else [],
            },
            gate_result=gate_result,
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_ms=int(elapsed_ms),
        )
        if self.enabled:
            self._call_count += 1
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(record.to_json() + "\n")
        return record

    @property
    def total_calls(self) -> int:
        return self._call_count


__all__ = ["LLMRecorder", "ToolCallRecord", "ToolCallRecorder"]
