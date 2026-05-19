"""Router: per-chunk content classification + structured extraction.

The router decides which of the fixed buckets (``isa | architecture |
intrinsics | examples | constraints | skip``) a chunk belongs in and
extracts typed records — :class:`ISAInstruction`,
:class:`IntrinsicSignature`, :class:`ParameterRange`, constraint
strings, exemplar tags — that downstream code folds into the target's
knowledge card.

Three implementations, all conforming to :class:`Router`:

* :class:`RouterAgentFile` — emits a typed MCP decision request and
  reads back a Claude-Code-authored decision response. This is the
  default for interactive flows (``/xpu-rt-target ingest …``) and costs
  nothing because the agent is already in the loop.
* :class:`RouterLLM` — calls Gemini through
  :class:`xpu_rt.llm.gemini_client.GeminiClient`, gated by
  :func:`xpu_rt.observability.gemini_usage.check_pre_call`. Used for
  headless/batch/CI ingestion.
* :class:`RouterMock` — returns a caller-supplied result table. Pure
  function, used by tests and golden fixtures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from xpu_rt.memory.ingest.chunking import TextChunk
from xpu_rt.memory.target_knowledge import (
    BUCKETS,
    ISAInstruction,
    IntrinsicSignature,
    ParameterRange,
)
from xpu_rt.observability import gemini_usage

logger = logging.getLogger(__name__)


ROUTER_PROMPT_VERSION = "v1"

ROUTER_SYSTEM_PROMPT = """\
You are the content router for XPU-RT's per-target knowledge memory.
Given a chunk of source text from a hardware-target documentation
corpus (README, ISA reference, C header, Scala generator config,
example kernel, etc.), classify it and extract structured records.

Buckets (pick exactly one):
  isa            instruction-level reference (mnemonics, encodings,
                 funct codes, operand layouts, control flow)
  architecture   datapath / microarchitecture / memory hierarchy /
                 dataflow descriptions; high-level explanations
  intrinsics     C-callable macros, runtime headers, library API
                 surfaces
  examples       reference kernels worth showing the agent (matmul,
                 conv, vector kernels, sample drivers)
  constraints    alignment rules, divisibility, scratchpad caps,
                 supported-shape lists, dataflow restrictions
  skip           licensing, build instructions, CI config, README
                 boilerplate, irrelevant noise

Rules:
- Pick the *closest* bucket. Most chunks should NOT be "skip".
- Be conservative on extraction: only emit records that are clearly
  named in the chunk. Don't hallucinate funct codes or signatures.
- Keep summaries short (≤ 200 chars) and factual.
- For 'examples', set exemplar_tags but do not duplicate code into
  summary_md; the pipeline copies the source file separately.
- If the chunk is mostly empty / boilerplate / a license header,
  bucket = "skip" and all record lists empty.

Output must be a single JSON object matching the schema you are
given. Do not wrap in markdown fences. Do not add prose around it.
"""

ROUTER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bucket": {
            "type": "string",
            "enum": list(BUCKETS) + ["skip"],
        },
        "summary_md": {"type": "string"},
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mnemonic": {"type": "string"},
                    "signature": {"type": "string"},
                    "summary": {"type": "string"},
                    "funct_code": {"type": ["integer", "null"]},
                    "latency_cycles": {"type": ["integer", "null"]},
                    "notes": {"type": "string"},
                },
                "required": ["mnemonic"],
            },
        },
        "intrinsics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "c_signature": {"type": "string"},
                    "summary": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "default": {"type": ["string", "number", "null"]},
                    "unit": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
        "exemplar_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["bucket"],
}


@dataclass(frozen=True)
class RouterChunk:
    """One unit handed to a router — chunk + provenance + role hint."""

    chunk: TextChunk
    source_locator: str
    source_kind: str  # one of CONTENT_KINDS from loaders.py
    role_hint: str  # "auto" or a bucket name
    target_id: str
    isa_family: str


@dataclass(frozen=True)
class RouterResult:
    """Structured payload produced by a router for one chunk."""

    bucket: str  # one of BUCKETS or "skip"
    summary_md: str = ""
    instructions: tuple[ISAInstruction, ...] = ()
    intrinsics: tuple[IntrinsicSignature, ...] = ()
    parameters: tuple[ParameterRange, ...] = ()
    constraints: tuple[str, ...] = ()
    exemplar_tags: tuple[str, ...] = ()
    raw_response: str = ""  # for audit; not folded into the card

    @classmethod
    def from_json(cls, payload: str | dict[str, Any]) -> RouterResult:
        body = payload if isinstance(payload, dict) else json.loads(payload)
        bucket = str(body.get("bucket", "skip"))
        if bucket not in BUCKETS and bucket != "skip":
            logger.warning("router returned unknown bucket %r; coercing to 'skip'", bucket)
            bucket = "skip"
        return cls(
            bucket=bucket,
            summary_md=str(body.get("summary_md", "")),
            instructions=tuple(
                ISAInstruction.from_dict(i) for i in body.get("instructions", ()) if i.get("mnemonic")
            ),
            intrinsics=tuple(
                IntrinsicSignature(
                    name=str(i["name"]),
                    c_signature=str(i.get("c_signature", "")),
                    summary=str(i.get("summary", "")),
                    notes=str(i.get("notes", "")),
                )
                for i in body.get("intrinsics", ())
                if i.get("name")
            ),
            parameters=tuple(
                ParameterRange(
                    name=str(p["name"]),
                    description=str(p.get("description", "")),
                    default=p.get("default"),
                    unit=str(p.get("unit", "")),
                    values=tuple(p.get("values", ())),
                )
                for p in body.get("parameters", ())
                if p.get("name")
            ),
            constraints=tuple(str(c) for c in body.get("constraints", ())),
            exemplar_tags=tuple(str(t) for t in body.get("exemplar_tags", ())),
            raw_response=payload if isinstance(payload, str) else json.dumps(payload),
        )


class Router(Protocol):
    """Classify + extract structured records from one chunk."""

    name: str

    def classify(self, item: RouterChunk) -> RouterResult: ...


# ---------------------------------------------------------------------------
# Mock — for tests and golden fixtures
# ---------------------------------------------------------------------------


@dataclass
class RouterMock:
    """Returns a caller-supplied :class:`RouterResult` table.

    The ``table`` callable receives a :class:`RouterChunk` and returns
    a :class:`RouterResult`. Useful for golden tests that don't need a
    real LLM round-trip.
    """

    name: str = "mock"
    table: Callable[[RouterChunk], RouterResult] = field(default=lambda item: RouterResult(bucket="skip"))
    calls: list[RouterChunk] = field(default_factory=list)

    def classify(self, item: RouterChunk) -> RouterResult:
        self.calls.append(item)
        return self.table(item)


# ---------------------------------------------------------------------------
# Live LLM — Gemini, behind the budget gate
# ---------------------------------------------------------------------------


def _build_user_prompt(item: RouterChunk) -> str:
    chunk = item.chunk
    return (
        f"target_id: {item.target_id}\n"
        f"isa_family: {item.isa_family}\n"
        f"source_locator: {item.source_locator}\n"
        f"source_kind: {item.source_kind}\n"
        f"role_hint: {item.role_hint}\n"
        f"chunk: {chunk.chunk_index + 1}/{chunk.chunk_total}\n"
        "---\n"
        f"{chunk.text}\n"
        "---\n\n"
        "Return JSON matching the schema. No prose."
    )


@dataclass
class RouterLLM:
    """Gemini-backed router.

    Constructed by :func:`xpu_rt.memory.ingest.pipeline.IngestPipeline.from_env`
    when the caller selected ``mode="llm-live"``. Every call checks the
    cumulative budget before issuing a request — once the cap is hit,
    :class:`xpu_rt.observability.gemini_usage.GeminiBudgetExceeded` is
    raised and the pipeline halts cleanly.
    """

    model: str = "gemini-2.5-flash"
    name: str = "gemini"

    def classify(self, item: RouterChunk) -> RouterResult:
        from xpu_rt.llm.factory import create_llm_client

        gemini_usage.check_pre_call(source=f"memory.ingest.router/{self.name}")

        client = create_llm_client("gemini", model=self.model)
        from xpu_rt.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext

        prompt = ROUTER_SYSTEM_PROMPT + "\n\n" + _build_user_prompt(item)
        request = GenerationRequest(
            prompt_template=prompt,
            context=PromptContext(
                model_ir_summary="",
                target_profile_summary="",
                available_transforms=[],
                kernel_contracts=[],
                objective=Objective.LATENCY,
            ),
            config=LLMConfig(model=self.model, temperature=0.0, max_tokens=4096),
            artifact_type="ingest_router_v1",
        )
        response = client.generate_structured(request, ROUTER_RESPONSE_SCHEMA)
        raw = response.raw_text or "{}"
        try:
            return RouterResult.from_json(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "RouterLLM: failed to parse Gemini response (%s); coercing to skip", exc
            )
            return RouterResult(bucket="skip", raw_response=raw)


# ---------------------------------------------------------------------------
# Agent-file — Claude Code in the loop, no Gemini spend
# ---------------------------------------------------------------------------


@dataclass
class AgentFileBridge:
    """Plumbing for the agent-file router.

    The agent-file router doesn't make any in-process LLM call — instead
    it asks an *external* Claude Code session to classify each chunk. The
    bridge is the typed shuttle: ``emit_request`` posts a decision
    request to a queue the MCP server exposes, ``wait_response`` blocks
    until the session has committed a response.

    The two callables default to JSONL-on-disk so unit tests can drive
    the bridge without a running MCP server: a request line is appended
    to ``requests.jsonl`` and the test pre-seeds ``responses.jsonl`` with
    the matching reply. The production wiring (task #8) replaces these
    callables with the live MCP tool endpoints.
    """

    emit_request: Callable[[dict[str, Any]], str]
    """Posts a decision request and returns its ``request_id``."""

    wait_response: Callable[[str], dict[str, Any]]
    """Blocks until the response for ``request_id`` is committed."""


@dataclass
class RouterAgentFile:
    """Router that delegates classification to an in-loop Claude Code session.

    Workflow per chunk:
      1. Build a decision-request envelope (system prompt + schema +
         chunk + provenance) and pass to :attr:`bridge.emit_request`.
      2. Block on :attr:`bridge.wait_response` until the session has
         committed a typed response.
      3. Parse the response as JSON, coerce into a :class:`RouterResult`.

    Does **not** call Gemini and therefore does **not** consult
    :func:`gemini_usage.check_pre_call`. The agent-file path is free.
    """

    bridge: AgentFileBridge
    name: str = "agent-file"

    def classify(self, item: RouterChunk) -> RouterResult:
        envelope = {
            "kind": "ingest_router_v1",
            "system": ROUTER_SYSTEM_PROMPT,
            "schema": ROUTER_RESPONSE_SCHEMA,
            "user": _build_user_prompt(item),
            "provenance": {
                "target_id": item.target_id,
                "isa_family": item.isa_family,
                "source_locator": item.source_locator,
                "source_kind": item.source_kind,
                "role_hint": item.role_hint,
                "chunk_index": item.chunk.chunk_index,
                "chunk_total": item.chunk.chunk_total,
            },
        }
        request_id = self.bridge.emit_request(envelope)
        response = self.bridge.wait_response(request_id)
        # Response is a typed dict {"payload": <json-string-or-dict>, ...}.
        payload = response.get("payload", response)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "RouterAgentFile: agent response not valid JSON (%s); coercing to skip", exc
                )
                return RouterResult(bucket="skip", raw_response=payload)
        try:
            return RouterResult.from_json(payload)
        except (KeyError, TypeError) as exc:
            logger.warning(
                "RouterAgentFile: malformed router payload (%s); coercing to skip", exc
            )
            return RouterResult(bucket="skip", raw_response=json.dumps(payload))


__all__ = [
    "AgentFileBridge",
    "ROUTER_PROMPT_VERSION",
    "ROUTER_RESPONSE_SCHEMA",
    "ROUTER_SYSTEM_PROMPT",
    "Router",
    "RouterAgentFile",
    "RouterChunk",
    "RouterLLM",
    "RouterMock",
    "RouterResult",
]
