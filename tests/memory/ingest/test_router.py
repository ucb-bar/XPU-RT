"""Tests for the three :class:`Router` implementations.

* :class:`RouterMock` — exercised in test_pipeline.py.
* :class:`RouterAgentFile` — uses a JSONL bridge stub here.
* :class:`RouterLLM` — gated by ``check_pre_call``; we verify the gate
  fires before any Gemini call is made.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from xpu_rt.memory.ingest.chunking import TextChunk
from xpu_rt.memory.ingest.router import (
    AgentFileBridge,
    RouterAgentFile,
    RouterChunk,
    RouterLLM,
    RouterResult,
)
from xpu_rt.observability import gemini_usage as gu


def _chunk(text: str = "hello") -> TextChunk:
    return TextChunk(text=text, start_offset=0, end_offset=len(text), chunk_index=0, chunk_total=1)


def _item(text: str = "hello") -> RouterChunk:
    return RouterChunk(
        chunk=_chunk(text),
        source_locator="/synthetic",
        source_kind="markdown",
        role_hint="auto",
        target_id="demo",
        isa_family="rocc-systolic",
    )


# ---------------------------------------------------------------------------
# RouterAgentFile + JSONL bridge stub
# ---------------------------------------------------------------------------


class _JsonlBridge(AgentFileBridge):
    """In-memory bridge that emits to a list and resolves immediately."""

    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_payload = response_payload

        def emit(envelope: dict[str, Any]) -> str:
            request_id = f"req-{len(self.requests)}"
            self.requests.append({"id": request_id, "envelope": envelope})
            return request_id

        def wait(request_id: str) -> dict[str, Any]:
            return {"request_id": request_id, "payload": self.response_payload}

        super().__init__(emit_request=emit, wait_response=wait)


def test_router_agent_file_round_trip() -> None:
    bridge = _JsonlBridge(
        response_payload={
            "bucket": "isa",
            "summary_md": "mvin loads scratchpad",
            "instructions": [{"mnemonic": "mvin", "signature": "rs1, rs2", "funct_code": 2}],
        }
    )
    router = RouterAgentFile(bridge=bridge)
    result = router.classify(_item())

    assert result.bucket == "isa"
    assert result.instructions[0].mnemonic == "mvin"
    # The agent received a properly-shaped envelope.
    sent = bridge.requests[0]["envelope"]
    assert sent["kind"] == "ingest_router_v1"
    assert "schema" in sent
    assert "user" in sent
    assert sent["provenance"]["target_id"] == "demo"


def test_router_agent_file_handles_string_payload() -> None:
    """Some MCP responses arrive as JSON-stringified payloads."""
    bridge = _JsonlBridge(response_payload={"bucket": "skip"})

    def wait_returning_string(request_id: str) -> dict[str, Any]:
        return {"payload": json.dumps({"bucket": "architecture", "summary_md": "ok"})}

    bridge.wait_response = wait_returning_string
    router = RouterAgentFile(bridge=bridge)
    result = router.classify(_item())
    assert result.bucket == "architecture"


def test_router_agent_file_coerces_malformed_response_to_skip() -> None:
    bridge = _JsonlBridge(response_payload={})
    bridge.wait_response = lambda rid: {"payload": "not valid json"}
    router = RouterAgentFile(bridge=bridge)
    result = router.classify(_item())
    assert result.bucket == "skip"


def test_router_agent_file_does_not_call_check_pre_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent-file router must never consult the Gemini budget gate."""
    called = {"count": 0}

    def fail(**kwargs: Any) -> None:
        called["count"] += 1

    monkeypatch.setattr(gu, "check_pre_call", fail)
    bridge = _JsonlBridge(response_payload={"bucket": "skip"})
    RouterAgentFile(bridge=bridge).classify(_item())
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# RouterLLM + budget gate
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("XPU_RT_REPO_ROOT", str(tmp_path))
    return tmp_path


def test_router_llm_blocks_when_budget_exceeded(
    isolated_usage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    # Burn through $100 so check_pre_call fires.
    gu.record_call("gemini-2.5-flash", 0, 40_000_000)

    # Sentinel that proves we never reached the SDK.
    def must_not_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("RouterLLM reached the LLM client despite breached budget")

    monkeypatch.setattr(
        "xpu_rt.llm.factory.create_llm_client", must_not_call
    )
    router = RouterLLM(model="gemini-2.5-flash")
    with pytest.raises(gu.GeminiBudgetExceeded):
        router.classify(_item())


def test_router_llm_passes_through_when_budget_ok(
    isolated_usage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    # Nothing spent yet → gate is open.

    class FakeResponse:
        raw_text = json.dumps({"bucket": "isa", "summary_md": "ok"})

    class FakeClient:
        def generate_structured(self, request: Any, schema: dict[str, Any]) -> Any:
            return FakeResponse()

    monkeypatch.setattr(
        "xpu_rt.llm.factory.create_llm_client",
        lambda provider, model=None: FakeClient(),
    )
    router = RouterLLM(model="gemini-2.5-flash")
    result = router.classify(_item())
    assert result.bucket == "isa"
    assert result.summary_md == "ok"
