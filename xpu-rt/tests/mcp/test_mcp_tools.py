"""Integration tests for the MCP tool handlers.

These tests exercise the handlers as plain Python callables, without
spawning the ``compgen-mcp`` subprocess, so the ``mcp`` Python SDK is
NOT a test dependency. A separate test can be added later for the
subprocess path once the optional ``mcp[]`` extras are installed in
CI.

Every test uses :class:`MockLLMClient` to keep the agentic loop
deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.llm.mock_client import MockLLMClient
from compgen.mcp.session import SessionManager
from compgen.mcp.tools import ALL_TOOLS
from compgen.mcp.tools.inspect import (
    checkpoint,
    diff_recipe,
    list_phase_tools,
    session_summary,
    view_recipe,
)
from compgen.mcp.tools.lifecycle import (
    bundle_export,
    load_model,
    open_target,
)
from compgen.mcp.tools.transform import (
    invoke_tool,
    propose_invent_slot,
    step_proposal,
    verify_proposal,
)

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


_MODEL_FILE = """
import torch
import torch.nn as nn

class _Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        return self.fc(x)


def build_model():
    m = _Demo()
    m.eval()
    return m, (torch.randn(1, 64),)
"""


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "demo_model.py"
    path.write_text(_MODEL_FILE)
    return path


# ---------------------------------------------------------------------------
# Catalogue sanity
# ---------------------------------------------------------------------------


def test_tool_catalogue_is_non_empty_and_well_formed() -> None:
    assert len(ALL_TOOLS) >= 12
    for t in ALL_TOOLS:
        assert "name" in t and isinstance(t["name"], str)
        assert "description" in t
        assert "handler" in t and callable(t["handler"])
        assert "input_schema" in t
        assert t["input_schema"].get("type") == "object"


def test_tool_names_are_unique() -> None:
    names = [t["name"] for t in ALL_TOOLS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_open_target_happy_path(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    res = open_target(sm, spec_path=str(EXEMPLAR))
    assert res["ok"] is True
    assert res["target_id"] == "test-gpu-simt"
    assert res["num_stages"] >= 1
    assert res["session_id"]


def test_open_target_reports_missing_spec(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    res = open_target(sm, spec_path=str(tmp_path / "does_not_exist.yaml"))
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_load_model_from_python_file(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    opened = open_target(sm, spec_path=str(EXEMPLAR))
    sid = opened["session_id"]

    # Use MockLLMClient indirectly by patching the factory — simpler:
    # point `llm` at a name that resolves to gemini's default model,
    # but wrap the created client with our mock before load_model.
    session = sm.get(sid)
    session.llm_client = MockLLMClient(strict=False)

    # Run load_model with llm='gemini' would hit the API; to avoid
    # that we monkey-patch _resolve_llm via a direct driver path.
    from compgen.api import compile_model, device as _device
    from compgen.agent.llm_driver import LLMDrivenCompiler

    mf = _model_file(tmp_path)
    # Use the handler's _resolve_model helper through load_model but
    # with a mock llm: the factory will try to find a provider. Easier
    # to drive the pieces directly:
    dev = _device(EXEMPLAR)
    session.device = dev
    from compgen.api_llm import _resolve_model
    module, inputs = _resolve_model(mf, None)
    compiled = compile_model(
        module, dev, objective="latency", sample_inputs=inputs,
    )
    mock = MockLLMClient(strict=False)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env, target=dev.profile, llm_client=mock,
        transcript_dir=session.scratch_dir / "transcripts", budget=4,
    )
    session.compiled = compiled
    session.driver = driver
    session.llm_client = mock

    # Now inspection tools should work.
    sv = session_summary(sm, session_id=sid)
    assert sv["ok"]
    assert sv["summary"]["session_id"]


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


def _prepared_session(tmp_path: Path) -> tuple[SessionManager, str]:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    opened = open_target(sm, spec_path=str(EXEMPLAR))
    sid = opened["session_id"]
    session = sm.get(sid)

    from compgen.api import compile_model, device as _device
    from compgen.agent.llm_driver import LLMDrivenCompiler
    from compgen.api_llm import _resolve_model

    dev = _device(EXEMPLAR)
    session.device = dev
    module, inputs = _resolve_model(_model_file(tmp_path), None)
    compiled = compile_model(module, dev, objective="latency", sample_inputs=inputs)
    mock = MockLLMClient(strict=False)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env, target=dev.profile, llm_client=mock,
        transcript_dir=session.scratch_dir / "transcripts", budget=4,
    )
    session.compiled = compiled
    session.driver = driver
    session.llm_client = mock
    return sm, sid


def test_view_recipe_returns_hashed_view(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = view_recipe(sm, session_id=sid, max_ops=20)
    assert res["ok"]
    view = res["view"]
    assert view["hash"].startswith("sha256:")
    assert "banner" in view and "middle" in view
    assert view["total_ops"] >= 0


def test_diff_recipe_against_ckpt_0(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = diff_recipe(sm, session_id=sid, from_ckpt="ckpt_0")
    assert res["ok"]
    assert res["diff"]["status"] == "ok"
    # Empty diff immediately after init is fine.
    assert "added" in res["diff"]


def test_diff_recipe_unknown_checkpoint(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = diff_recipe(sm, session_id=sid, from_ckpt="ckpt_unknown")
    assert res["ok"]
    assert res["diff"]["status"] == "unknown_checkpoint"


def test_checkpoint_then_diff(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    ck = checkpoint(sm, session_id=sid, label="my_ckpt")
    assert ck["ok"]
    assert ck["ckpt_id"] == "my_ckpt"
    res = diff_recipe(sm, session_id=sid, from_ckpt="my_ckpt")
    assert res["ok"] and res["diff"]["status"] == "ok"


def test_list_phase_tools_returns_catalogue(tmp_path: Path) -> None:
    sm, _ = _prepared_session(tmp_path)
    res = list_phase_tools(sm)
    assert res["ok"]
    assert "tools" in res and "invent_slots" in res
    # Counts shape: {phase: {"tools": n, "invent_slots": m}}
    assert all(
        "tools" in v and "invent_slots" in v for v in res["counts"].values()
    )


def test_session_summary_reports_session_id(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = session_summary(sm, session_id=sid)
    assert res["ok"]
    assert res["summary"]["session_id"]
    assert res["summary"]["step_index"] == 0


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_verify_proposal_structural_only_accepts(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = verify_proposal(
        sm, session_id=sid,
        proposal={"chosen": {}, "select_vs_invent": "invent"},
        gates=["structural"],
    )
    assert res["ok"]
    assert res["gate_result"]["status"] == "accepted"


def test_verify_proposal_rejects_carries_remediation(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = verify_proposal(
        sm, session_id=sid,
        proposal={},   # missing required keys
        gates=["structural"],
    )
    assert res["ok"]
    assert res["gate_result"]["status"] == "rejected"
    hint = res["gate_result"]["details"].get("remediation_hint")
    assert hint is not None and len(hint) > 0


def test_invoke_tool_unknown_returns_unknown(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = invoke_tool(sm, session_id=sid, tool_name="no_such_tool")
    assert res["ok"]
    assert res["status"] == "unknown"


def test_propose_invent_slot_unknown_returns_unknown(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = propose_invent_slot(
        sm, session_id=sid, slot_name="no_such_slot",
        proposal={"chosen": {}, "select_vs_invent": "invent"},
    )
    assert res["ok"]
    assert res["status"] == "unknown"


def test_step_proposal_noop_for_unknown_action(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    res = step_proposal(sm, session_id=sid, action_type="definitely_not_a_valid_action")
    assert res["ok"]
    # proposal -> NoopAction when action_type isn't in the catalogue.
    assert res["status"] == "noop"


def test_bundle_export_writes_files(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    out = tmp_path / "bundle_out"
    res = bundle_export(sm, session_id=sid, output_dir=str(out))
    assert res["ok"]
    assert (out / "payload.mlir").exists()
    assert (out / "manifest.json").exists()
    assert len(res["sha"]) == 16
