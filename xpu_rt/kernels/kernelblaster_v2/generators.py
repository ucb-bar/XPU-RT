"""Kernel-generator backends for the KernelBlaster v2 agent loop.

Mirrors the ingestion-router design — generator is a protocol with
three implementations:

* :class:`KernelGeneratorAgentFile` — the in-loop Claude Code session
  produces the kernel via an MCP decision request. No spend.
* :class:`KernelGeneratorLLM` — Gemini-backed, gated by
  :func:`xpu_rt.observability.gemini_usage.check_pre_call`.
* :class:`KernelGeneratorMock` — caller-supplied response table for
  tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBundle
from xpu_rt.kernels.provider import ContractFeedback
from xpu_rt.observability import gemini_usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProposeRequest:
    """One propose call — the prompt the generator was handed."""

    bundle: PromptBundle
    attempt_index: int
    state_hash: str


@dataclass(frozen=True)
class ProposeResponse:
    """One propose call result."""

    kernel_code: str
    language: str = ""
    action: str = ""
    rationale: str = ""
    contract_feedback: tuple[ContractFeedback, ...] = ()
    raw_response: str = ""

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> ProposeResponse:
        feedback_rows = []
        for raw in body.get("contract_feedback", ()):
            feedback_rows.append(
                ContractFeedback(
                    field=str(raw.get("field", "")),
                    current_value=str(raw.get("current_value", "")),
                    suggested_value=str(raw.get("suggested_value", "")),
                    reason=str(raw.get("reason", "")),
                    measured_gain=float(raw.get("measured_gain", 0.0)),
                    kind=str(raw.get("kind", "")),
                    applies_when=str(raw.get("applies_when", "")),
                )
            )
        return cls(
            kernel_code=str(body.get("kernel_code", "")),
            language=str(body.get("language", "")),
            action=str(body.get("action", "")),
            rationale=str(body.get("rationale", "")),
            contract_feedback=tuple(feedback_rows),
            raw_response=json.dumps(body) if not isinstance(body, str) else body,
        )


class KernelGenerator(Protocol):
    """Produce a candidate kernel for one :class:`ProposeRequest`."""

    name: str

    def propose(self, request: ProposeRequest) -> ProposeResponse: ...


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


@dataclass
class KernelGeneratorMock:
    """Caller-supplied table for hermetic tests."""

    name: str = "mock"
    table: Callable[[ProposeRequest], ProposeResponse] = field(
        default=lambda req: ProposeResponse(kernel_code="// mock kernel", action="mock")
    )
    calls: list[ProposeRequest] = field(default_factory=list)

    def propose(self, request: ProposeRequest) -> ProposeResponse:
        self.calls.append(request)
        return self.table(request)


# ---------------------------------------------------------------------------
# Agent-file (Claude Code in the loop)
# ---------------------------------------------------------------------------


@dataclass
class AgentFileBridge:
    """Plumbing for the in-session Claude Code generator.

    Same shape as the ingestion bridge: ``emit_request`` posts a decision
    envelope and returns a ``request_id``; ``wait_response`` blocks until
    the session has committed a response. The two callables default to a
    test-injected pair; the production wiring (task #8) replaces them
    with the live MCP tool endpoints.
    """

    emit_request: Callable[[dict[str, Any]], str]
    wait_response: Callable[[str], dict[str, Any]]


@dataclass
class KernelGeneratorAgentFile:
    """Generator that hands the request to an in-loop Claude Code agent."""

    bridge: AgentFileBridge
    name: str = "agent-file"

    def propose(self, request: ProposeRequest) -> ProposeResponse:
        envelope = {
            "kind": "kernelblaster_v2_propose",
            "system": request.bundle.system,
            "schema": request.bundle.schema,
            "user": request.bundle.user,
            "metadata": {
                **request.bundle.metadata,
                "attempt_index": request.attempt_index,
                "state_hash": request.state_hash,
            },
        }
        request_id = self.bridge.emit_request(envelope)
        response = self.bridge.wait_response(request_id)
        payload: Any = response.get("payload", response)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "KernelGeneratorAgentFile: response not valid JSON (%s); empty proposal", exc
                )
                return ProposeResponse(kernel_code="", action="", raw_response=payload)
        try:
            return ProposeResponse.from_dict(payload)
        except (TypeError, ValueError) as exc:
            logger.warning("KernelGeneratorAgentFile: malformed payload (%s)", exc)
            return ProposeResponse(
                kernel_code="",
                action="",
                raw_response=json.dumps(payload),
            )


# ---------------------------------------------------------------------------
# Gemini live
# ---------------------------------------------------------------------------


@dataclass
class KernelGeneratorLLM:
    """Gemini-backed generator behind the budget gate."""

    model: str = "gemini-2.5-flash"
    name: str = "gemini"

    def propose(self, request: ProposeRequest) -> ProposeResponse:
        from xpu_rt.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
        from xpu_rt.llm.factory import create_llm_client

        gemini_usage.check_pre_call(source=f"kernels.kernelblaster_v2/{self.name}")

        client = create_llm_client("gemini", model=self.model)
        prompt = request.bundle.system + "\n\n" + request.bundle.user
        req = GenerationRequest(
            prompt_template=prompt,
            context=PromptContext(
                model_ir_summary="",
                target_profile_summary="",
                available_transforms=[],
                kernel_contracts=[],
                objective=Objective.LATENCY,
            ),
            config=LLMConfig(model=self.model, temperature=0.0, max_tokens=16384),
            artifact_type="kernelblaster_v2_propose",
        )
        response = client.generate_structured(req, request.bundle.schema)
        raw = response.raw_text or "{}"
        return _parse_propose_response(raw)


# ---------------------------------------------------------------------------
# Robust response parser
# ---------------------------------------------------------------------------


def _parse_propose_response(raw: str) -> "ProposeResponse":
    """Parse Gemini's structured output → :class:`ProposeResponse`.

    Tries three strategies in order:
      1. **strict JSON** (the happy path — the API contract).
      2. **lenient JSON** — strip leading/trailing markdown fences,
         then retry. Catches responses Gemini wraps in ``\`\`\`json``.
      3. **markdown C-fence extraction** — when JSON is hopelessly
         malformed (unterminated strings from un-escaped C source),
         pull a ``\`\`\`c …\`\`\``` block out as ``kernel_code`` and
         leave ``action`` blank. We lose the LLM's stated action /
         rationale but keep the actual code so the cell can compile.

    The fallback's existence is the difference between "0/30 correct,
    20% Gemini-side parse failures" (Track 4 N=1 result) and a
    matrix where every successful LLM call lands a kernel.
    """
    raw = raw.strip()

    # Strategy 1 — strict JSON
    try:
        return ProposeResponse.from_dict(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2 — strip leading ```json / trailing ``` fences
    stripped = _strip_markdown_fence(raw)
    if stripped != raw:
        try:
            return ProposeResponse.from_dict(json.loads(stripped))
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3 — extract a C/Python code fence as the kernel body
    fenced = _extract_code_fence(raw)
    if fenced:
        logger.warning(
            "KernelGeneratorLLM: strict JSON parse failed; "
            "recovered kernel_code from markdown fence (%d chars)",
            len(fenced),
        )
        return ProposeResponse(
            kernel_code=fenced,
            language="c",
            action="recovered_from_markdown_fence",
            raw_response=raw,
        )

    # Strategy 4 — regex-extract the JSON-string-encoded kernel_code
    # value when the JSON itself is truncated / unterminated. This
    # is the dominant Gemini failure mode on long Saturn RVV
    # kernels — the kernel_code value runs past Gemini's token cap,
    # the closing `"` and `}` never arrive, JSON parsing fails,
    # but the prefix of kernel_code is still recoverable.
    recovered_code = _extract_json_kernel_code_prefix(raw)
    if recovered_code:
        logger.warning(
            "KernelGeneratorLLM: strict JSON + fence parses failed; "
            "regex-recovered truncated kernel_code prefix (%d chars)",
            len(recovered_code),
        )
        return ProposeResponse(
            kernel_code=recovered_code,
            language="c",
            action="recovered_from_truncated_json",
            raw_response=raw,
        )

    # Strategy 5 — give up
    logger.warning(
        "KernelGeneratorLLM: failed to parse Gemini response and no "
        "fenced code block / recoverable JSON-string found; emitting "
        "empty response. Raw response head: %r",
        raw[:200],
    )
    return ProposeResponse(kernel_code="", action="", raw_response=raw)


def _extract_json_kernel_code_prefix(raw: str) -> str | None:
    r"""Pull a ``kernel_code`` value out of a malformed JSON envelope.

    Handles the truncated-mid-string case: Gemini emits
    ``{\n  "kernel_code": "...code with \n escapes..."`` and runs
    out of tokens before closing the string. We scan from the
    opening ``"`` after ``"kernel_code":`` and accumulate
    JSON-string-decoded characters until we hit either the closing
    ``"`` (success) or end-of-input (truncation — return what we have).
    """
    import re

    m = re.search(r'"kernel_code"\s*:\s*"', raw)
    if not m:
        return None
    i = m.end()  # index of first character of the string value
    out: list[str] = []
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            decoded = {
                "n": "\n", "t": "\t", "r": "\r",
                '"': '"', "\\": "\\", "/": "/",
                "b": "\b", "f": "\f",
            }.get(nxt)
            if decoded is not None:
                out.append(decoded)
                i += 2
                continue
            if nxt == "u" and i + 5 < n:
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            # Unknown escape — keep the backslash + char verbatim.
            out.append(ch + nxt)
            i += 2
            continue
        if ch == '"':
            # Closing quote — string is well-formed; return collected text.
            recovered = "".join(out)
            return recovered if recovered else None
        out.append(ch)
        i += 1
    # Truncated mid-string — return whatever we accumulated.
    recovered = "".join(out).rstrip()
    return recovered if recovered else None


_FENCE_PATTERNS = (
    "```json",
    "```python",
    "```c",
    "```cpp",
    "```",
)


def _strip_markdown_fence(raw: str) -> str:
    r"""Remove a leading ```\`\`\`{lang}``` and trailing ```\`\`\``` if present."""
    s = raw.strip()
    for fence in _FENCE_PATTERNS:
        if s.startswith(fence):
            # drop the opening fence
            s = s[len(fence):].lstrip("\n")
            # drop the trailing fence if present
            if s.endswith("```"):
                s = s[:-3].rstrip()
            return s
    return raw


def _extract_code_fence(raw: str) -> str | None:
    r"""Return the body of the first ```\`\`\`{c,cpp,python}…\`\`\``` block, if any."""
    import re

    # Prefer C/C++ fences; fall back to language-agnostic ``` blocks.
    for lang_pat in (r"```(?:c|cpp|c\+\+)", r"```\w*"):
        match = re.search(
            lang_pat + r"\s*\n(.*?)```",
            raw,
            flags=re.DOTALL,
        )
        if match:
            body = match.group(1).strip()
            if body:
                return body
    return None
