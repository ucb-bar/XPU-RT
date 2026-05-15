"""Live-LLM provider abstraction (secondary path; opt-in only).

Bounded interface that takes an ``agent_decision_request.json`` plus the
compact ``llm_graph_view.json``, builds a constrained prompt, and calls
a provider to obtain an ``agent_decision_response.json``-shaped dict.
The compiler then runs the unchanged validator before any Recipe
IR commit; this module does NOT bypass that gate.

**This is the secondary agent path.** The recommended default for any
agentic run is ``--selection-mode agent-file`` driven by Claude Code
via MCP/skills (no API key, no token spend, audited via the same
11-check validator). This module exists for unattended / CI /
paper-reproduction use cases where no Claude Code session is running.
See ``feedback_claude_code_is_the_agent.md`` in the user's auto-memory
for the rationale.

Hard non-goals:

No multi-turn retry. (territory.)
- No streaming UI.
- No tool calling.
- No new candidate generation.
- No compiler-core changes.
- No secret persistence: API keys come from environment variables and
  are NEVER written to disk.

Built-in providers:

- ``gemini`` — Google Gemini API adapter via the google-genai SDK.
  Requires ``GEMMINI_API`` (CompGen's repo-local .env spelling) or
  ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``. Every call is auto-recorded
  by ``compgen.observability.gemini_usage`` (token + USD cost) via the
  installed SDK instrumentation.
- ``anthropic`` — Anthropic Messages API adapter (urllib + stdlib only,
  no SDK). Requires ``ANTHROPIC_API_KEY`` or ``COMPGEN_LLM_API_KEY``.
- ``openai`` — OpenAI Chat Completions adapter with JSON-mode output.
  Requires ``OPENAI_API_KEY`` or ``COMPGEN_LLM_API_KEY``.
- ``env`` — reads ``COMPGEN_LLM_PROVIDER`` and dispatches.

Note: there is intentionally no ``mock`` / stub provider. CompGen is
Claude-Code-first: for offline / no-key environments, use
``--selection-mode agent-file`` with Claude Code (or any external
agent) writing a real ``agent_decision_response.json``. The
``llm-live`` HTTP path is the secondary route, and only real provider
adapters are registered.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Errors + result types
# --------------------------------------------------------------------------- #


class ProviderError(RuntimeError):
    """Typed provider-call failure (network error, malformed JSON, etc.)"""


@dataclass(frozen=True)
class ProviderCallResult:
    raw_response: dict[str, Any]
    parsed_response: dict[str, Any] | None
    latency_ms: int
    prompt: str
    provider_name: str
    model: str
    error: str | None = None  # populated when raw response could not be parsed


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


_PROMPT_TEMPLATE = """\
You are selecting one compiler Recipe candidate.

You must return JSON matching agent_decision_response_v1.

# Hard rules

- Select EXACTLY one candidate_id from `candidate_ids_allowed`.
- Do NOT invent candidate IDs, tile sizes, or evidence fields.
- Do NOT select illegal or hidden candidates.
- Use only fields present in the request and visible graph view.
- Do NOT claim the transform is correct.
- Do NOT claim measured performance.
- Return only JSON; no markdown, no prose.

# How to read the cost matrix

Each legal SetTileParams candidate may carry up to four cost columns,
each rooted in a different evidence type. Read the request's
`agent_guidance.cost_column_priority` block; the rank order is:

1. `compiled_evidence` (M-22) — measured bottleneck on real hardware.
   Field: `hardware_resource_report.regions[*].compiled_evidence` or
   `compiled_bottleneck_report.regions[*]`. Strongest evidence.
2. `calibration_delta` (M-21) — predicted_us vs measured_us ratio per
   candidate. Ratios << 1.0 mean analytical roofline is optimistic
   (launch-overhead-dominated regime).
3. `m21_analytical_cost` (M-21) — deterministic blocked-matmul roofline.
   Always present when the run reached graph_analysis. Carries
   `predicted_us`, `bottleneck_resource`, `bottleneck_tier`.
4. `calibration` (M-18.3) — Python-evaluator timing of the M-16 tiled
   loop. Use SPREAD across candidates, not absolute speedup.
5. `static_relative_cost` (M-13) — greedy's deterministic baseline;
   tiebreaker only.

# Disagreement-handling rules

- `bottleneck_classification_agreement == false`: M-21 analytical and
  M-22 measured disagree. Surface the disagreement; do NOT claim either
  side is wrong. Prefer the measured side when picking.
- `predicted_vs_gpu_ratio < 0.1`: analytical is >10x optimistic. Prefer
  compiled measurement when available.
- `kernel_calibration_status == partial_kernel_calibration`: prefer
  calibrated regions' candidates; fall back to analytical for the rest.
- `kernel_calibration_status == not_kernel_calibrated`: M-19/M-20 didn't
  run. M-21 analytical is the strongest available signal.

# Rationale shape

`rationale.summary` must be 1-2 neutral sentences referencing the
strongest evidence column. `rationale.evidence` must be a list of >=2
entries; each entry is `{{field, value, reason}}` where `field` resolves
against the candidate or sources (use `agent_guidance.rationale_field_examples` as a starting set).

Forbidden phrases (the validator rejects these):
{forbidden_phrases}

Preferred neutral phrases:
- "lower static_relative_cost"
- "fits scratchpad" / "tile_working_set_bytes < scratchpad_bytes"
- "M-12 differential evidence available"
- "M-21 analytical predicts compute-bound on bottleneck_tier=scratchpad"
- "M-22 measured bottleneck agrees with analytical"
- "kernel_calibrated evidence available for this region"

# Inputs

Request:
{request_json}

Visible graph view:
{view_json}
"""

# Forbidden phrases the validator rejects. Keep in sync with
# `agent_guidance.forbidden_phrase_patterns` and the hard-coded regex
# in `agent_decision.py::_no_correctness_claim` /
# `_no_measured_performance_claim`.
_FORBIDDEN_PHRASES = (
    "verified correct",
    "guaranteed correct",
    "bit equivalent to eager",
    "measured fastest",
    "benchmarked",
    "profiled",
    "executed faster",
)


def build_prompt(
    *, request: dict[str, Any], llm_graph_view: dict[str, Any] | None,
) -> str:
    """Construct a bounded prompt from the request + the legal-only
    graph view. Both inputs are pretty-printed JSON for human review of
    the emitted ``agent_decision_prompt.txt``."""
    return _PROMPT_TEMPLATE.format(
        forbidden_phrases="\n".join(f'- "{p}"' for p in _FORBIDDEN_PHRASES),
        request_json=json.dumps(request, indent=2, sort_keys=True),
        view_json=json.dumps(
            llm_graph_view if llm_graph_view is not None else {},
            indent=2, sort_keys=True,
        ),
    )


# --------------------------------------------------------------------------- #
# Provider implementations
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Real provider adapters. stdlib-only via urllib.request to avoid
# pulling provider SDKs into the compiler trust boundary.
# --------------------------------------------------------------------------- #


def _http_post_json(
    *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_sec: int,
) -> tuple[int, dict[str, Any], str]:
    """Tiny urllib-only POST helper. Returns (status, parsed_body, raw_text).

    Isolated as a module-level function so tests can monkey-patch it
    without invoking real network I/O.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - https URLs only
        url, data=payload, method="POST",
    )
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            status = int(resp.status)
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise ProviderError(
            f"provider HTTP {exc.code}: {text[:200]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"provider URL error: {exc.reason}") from exc
    except TimeoutError as exc:  # pragma: no cover - depends on remote
        raise ProviderError(f"provider timeout: {exc}") from exc
    try:
        parsed: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"provider returned non-JSON HTTP body: {exc}: {text[:200]}"
        ) from exc
    if status >= 400:
        raise ProviderError(f"provider HTTP {status}: {text[:200]}")
    return status, parsed, text


def _read_api_key(env_name_primary: str) -> str:
    """Resolve API key from environment. Order:
    ``env_name_primary`` then ``COMPGEN_LLM_API_KEY``."""
    key = os.environ.get(env_name_primary, "") or os.environ.get(
        "COMPGEN_LLM_API_KEY", ""
    )
    if not key:
        raise ProviderError(
            f"missing API key: set {env_name_primary} or COMPGEN_LLM_API_KEY"
        )
    return key


def _anthropic_provider(
    *,
    request: dict[str, Any],
    llm_graph_view: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
    model: str,
    timeout_sec: int,
) -> ProviderCallResult:
    """Anthropic Messages API adapter. Sends the bounded prompt; expects
    a JSON-only response. Uses urllib + stdlib only — no SDK."""
    api_key = _read_api_key("ANTHROPIC_API_KEY")
    prompt = build_prompt(request=request, llm_graph_view=llm_graph_view)
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    t0 = time.perf_counter_ns()
    status, parsed, raw_text = _http_post_json(
        url="https://api.anthropic.com/v1/messages",
        headers=headers, body=body, timeout_sec=timeout_sec,
    )
    latency_ms = max(1, (time.perf_counter_ns() - t0) // 1_000_000)
    # Extract assistant text content. Anthropic returns
    # ``{"content": [{"type": "text", "text": "..."}], ...}``.
    text_blocks = [
        b.get("text", "") for b in (parsed.get("content") or [])
        if b.get("type") == "text"
    ]
    completion_text = "\n".join(t for t in text_blocks if t)
    return ProviderCallResult(
        raw_response={
            "provider": "anthropic",
            "model": parsed.get("model", model),
            "completion_text": completion_text,
            "completion_kind": "json",
            "metadata": {
                "id": parsed.get("id"),
                "stop_reason": parsed.get("stop_reason"),
                "usage": parsed.get("usage"),
                "http_status": status,
            },
        },
        parsed_response=None,  # parser runs downstream
        latency_ms=latency_ms,
        prompt=prompt,
        provider_name="anthropic",
        model=parsed.get("model", model),
    )


def _openai_provider(
    *,
    request: dict[str, Any],
    llm_graph_view: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
    model: str,
    timeout_sec: int,
) -> ProviderCallResult:
    """OpenAI Chat Completions API adapter. Requests JSON-mode output
    via ``response_format={"type":"json_object"}``."""
    api_key = _read_api_key("OPENAI_API_KEY")
    prompt = build_prompt(request=request, llm_graph_view=llm_graph_view)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You return only JSON matching agent_decision_response_v1. "
                    "Never include markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    t0 = time.perf_counter_ns()
    status, parsed, raw_text = _http_post_json(
        url="https://api.openai.com/v1/chat/completions",
        headers=headers, body=body, timeout_sec=timeout_sec,
    )
    latency_ms = max(1, (time.perf_counter_ns() - t0) // 1_000_000)
    completion_text = ""
    choices = parsed.get("choices") or []
    if choices:
        completion_text = (choices[0].get("message") or {}).get("content", "")
    return ProviderCallResult(
        raw_response={
            "provider": "openai",
            "model": parsed.get("model", model),
            "completion_text": completion_text,
            "completion_kind": "json",
            "metadata": {
                "id": parsed.get("id"),
                "finish_reason": (choices[0] if choices else {}).get(
                    "finish_reason",
                ),
                "usage": parsed.get("usage"),
                "http_status": status,
            },
        },
        parsed_response=None,
        latency_ms=latency_ms,
        prompt=prompt,
        provider_name="openai",
        model=parsed.get("model", model),
    )


def _gemini_provider(
    *,
    request: dict[str, Any],
    llm_graph_view: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
    model: str,
    timeout_sec: int,
) -> ProviderCallResult:
    """Google Gemini adapter using the google-genai SDK.

    The SDK is patched on first use so every call is recorded by
    ``compgen.observability.gemini_usage``. Requests JSON-mime-type
    output (``application/json``) so the validator's parser sees clean
    JSON without code-fence stripping. Reads the API key via
    ``compgen.llm._env.resolve_api_key`` which checks ``GOOGLE_API_KEY``,
    ``GEMINI_API_KEY``, and the repo's ``GEMMINI_API`` (sic) variable.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise ProviderError(
            "google-genai SDK not installed; run `uv add google-genai` "
            "or use --llm-live-provider {anthropic,openai}."
        ) from exc

    from compgen.llm._env import resolve_api_key
    from compgen.observability.gemini_usage import (
        install_genai_instrumentation,
        tracking_source,
    )

    api_key = resolve_api_key(
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API",
    )
    if not api_key:
        raise ProviderError(
            "missing Gemini API key: set GEMMINI_API in .env, or "
            "GOOGLE_API_KEY / GEMINI_API_KEY in environment"
        )

    install_genai_instrumentation()
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(request=request, llm_graph_view=llm_graph_view)

    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=4096,
        response_mime_type="application/json",
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    t0 = time.perf_counter_ns()
    try:
        with tracking_source(
            "graph_compilation.llm_live_provider:gemini",
            request_path=request.get("source_path", ""),
        ):
            response = client.models.generate_content(
                model=model, contents=prompt, config=config,
            )
    except Exception as exc:  # noqa: BLE001 - SDK exceptions vary
        raise ProviderError(f"Gemini SDK call failed: {exc}") from exc
    latency_ms = max(1, (time.perf_counter_ns() - t0) // 1_000_000)

    completion_text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return ProviderCallResult(
        raw_response={
            "provider": "gemini",
            "model": model,
            "completion_text": completion_text,
            "completion_kind": "json",
            "metadata": {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(
                    usage, "candidates_token_count", 0,
                ),
                "total_tokens": getattr(usage, "total_token_count", 0),
                "finish_reason": "stop",
            },
        },
        parsed_response=None,
        latency_ms=latency_ms,
        prompt=prompt,
        provider_name="gemini",
        model=model,
    )


# Hook for tests / future real providers to override. Maps provider
# name -> callable producing a ``ProviderCallResult`` (or raising
# ``ProviderError``). The CLI wires environment-driven providers
# through ``call_provider`` below.
_PROVIDERS: dict[str, Callable[..., ProviderCallResult]] = {
    "gemini": _gemini_provider,
    "anthropic": _anthropic_provider,
    "openai": _openai_provider,
}


def register_provider(
    name: str, fn: Callable[..., ProviderCallResult],
) -> None:
    """Register a provider callable for use in tests or downstream
    integrations. Real openai/anthropic adapters can be plugged in this
    way without touching the trusted compiler path."""
    _PROVIDERS[name] = fn


def _env_provider(
    *,
    request: dict[str, Any],
    llm_graph_view: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
    model: str,
    timeout_sec: int,
) -> ProviderCallResult:
    """Dispatch based on ``COMPGEN_LLM_PROVIDER``.

    Built-in real providers (``gemini``, ``anthropic``, ``openai``).
    Unsupported names raise ``ProviderError``. For offline / no-key
    workflows, use ``--selection-mode agent-file`` with Claude Code
    rather than this HTTP path.
    """
    provider_name = os.environ.get("COMPGEN_LLM_PROVIDER", "")
    if not provider_name:
        raise ProviderError(
            "COMPGEN_LLM_PROVIDER not set; pass "
            "--llm-live-provider {gemini,anthropic,openai} or set "
            "COMPGEN_LLM_PROVIDER. For offline / Claude-Code-driven "
            "selection use --selection-mode agent-file instead."
        )
    fn = _PROVIDERS.get(provider_name)
    if fn is None:
        raise ProviderError(
            f"unsupported provider {provider_name!r}; built-ins: "
            f"gemini, anthropic, openai. Register additional providers "
            f"via compgen.graph_compilation.llm_live_provider.register_provider()."
        )
    return fn(
        request=request, llm_graph_view=llm_graph_view,
        candidate_actions=candidate_actions,
        model=model, timeout_sec=timeout_sec,
    )


# --------------------------------------------------------------------------- #
# Top-level call
# --------------------------------------------------------------------------- #


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_provider(
    *,
    provider_name: str,
    model: str | None,
    timeout_sec: int,
    request: dict[str, Any],
    llm_graph_view: dict[str, Any] | None,
    candidate_actions: dict[str, Any],
) -> ProviderCallResult:
    """Top-level provider-call entry point.

    ``provider_name="env"`` reads ``COMPGEN_LLM_PROVIDER``; other
    names dispatch directly. Raises ``ProviderError`` on
    unsupported names or hard provider failures.
    """
    fn: Callable[..., ProviderCallResult]
    if provider_name == "env":
        fn = _env_provider
    elif provider_name in _PROVIDERS:
        fn = _PROVIDERS[provider_name]
    else:
        raise ProviderError(
            f"unsupported provider {provider_name!r}; built-ins: "
            f"gemini, anthropic, openai. Use --llm-live-provider "
            f"<name> or register an extension provider."
        )

    # Provider-default model: env > per-provider default > error. The
    # only built-in default is for ``gemini`` (matches GeminiClient's
    # default). For anthropic/openai/env, the caller must supply a
    # model explicitly to avoid silently picking a wrong one.
    if model is None:
        model = os.environ.get("COMPGEN_LLM_MODEL", "")
        if not model:
            if provider_name == "gemini":
                model = "gemini-2.5-flash"
            else:
                raise ProviderError(
                    "no model specified: pass --llm-live-model or set "
                    "COMPGEN_LLM_MODEL"
                )

    result = fn(
        request=request, llm_graph_view=llm_graph_view,
        candidate_actions=candidate_actions,
        model=model, timeout_sec=timeout_sec,
    )
    return result


# --------------------------------------------------------------------------- #
# Response parsing (handles JSON-only or fenced-code-block responses)
# --------------------------------------------------------------------------- #


def parse_provider_response_text(text: str) -> dict[str, Any]:
    """Best-effort parser for provider completions.

    Accepts:

    - A bare JSON object.
    - A JSON object inside ``\\`\\`\\`json ... \\`\\`\\``` fences.

    Raises ``ProviderError`` for everything else (prose, broken JSON,
    multiple objects). The parsed object is then run through the
    validator.
    """
    if not text or not text.strip():
        raise ProviderError("provider response was empty")
    stripped = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1:
            raise ProviderError("malformed code-fenced response")
        stripped = stripped[first_newline + 1:]
        end_fence = stripped.rfind("```")
        if end_fence == -1:
            raise ProviderError("unterminated code-fenced response")
        stripped = stripped[:end_fence].strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider response is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProviderError(
            f"provider response is not a JSON object (got {type(obj).__name__})"
        )
    return obj
