"""Tests for the P2 diagnose + recovery MCP tools.

Exercised as plain Python handler calls — no subprocess, no ``mcp`` SDK
dependency. The tests use a module whose forward emits
``aten.tanh.default`` so the existing capture pipeline surfaces at
least one unsupported-op issue. Tanh is deliberately off the Payload
decomposition allow-list *and* off the synthesize_decomp allow-list,
giving us one concrete row to drive through each recovery tool.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.diagnose import (
    DIAGNOSE_TOOLS,
    diagnose_exported_program,
    diagnose_model_compatibility,
)
from compgen.mcp.tools.recovery import (
    RECOVERY_TOOLS,
    recovery_status,
    register_blackbox,
    resolve_unsupported_op,
    synthesize_decomp,
    synthesize_translation,
)

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _WithUnsupported(nn.Module):
    """Linear + tanh — ``aten.tanh.default`` falls off our decomp table."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))


def _prepared_session(tmp_path: Path) -> tuple[SessionManager, str]:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()

    dev = _device(EXEMPLAR)
    session.device = dev

    compiled = compile_model(
        _WithUnsupported().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
        recover_unsupported=False,  # let the LLM/MCP path drive recovery
    )
    mock = MockLLMClient(strict=False)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=mock,
        transcript_dir=session.scratch_dir / "transcripts",
        budget=4,
    )
    session.compiled = compiled
    session.driver = driver
    session.llm_client = mock
    return sm, session.session_id


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_diagnose_and_recovery_tools_are_registered() -> None:
    names_d = [t["name"] for t in DIAGNOSE_TOOLS]
    names_r = [t["name"] for t in RECOVERY_TOOLS]
    for t in DIAGNOSE_TOOLS + RECOVERY_TOOLS:
        assert callable(t["handler"])
        assert t["input_schema"]["type"] == "object"
    assert "diagnose_model_compatibility" in names_d
    for expected in (
        "synthesize_decomp",
        "synthesize_translation",
        "register_blackbox",
        "resolve_unsupported_op",
        "recovery_status",
    ):
        assert expected in names_r


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def test_diagnose_model_compatibility_reports_tanh_issue(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    result = diagnose_model_compatibility(sm, session_id=sid)
    assert result["ok"]
    targets = [i["target"] for i in result["issues"]]
    assert "aten.tanh.default" in targets

    tanh_row = next(i for i in result["issues"] if i["target"] == "aten.tanh.default")
    # The dossier + classifier must surface the strategy the LLM can act on.
    assert tanh_row["classification"]["strategy"] in {
        "synthesized_external_call",
        "explicit_blackbox",
    }
    assert tanh_row["recommended_tool"] in {
        "synthesize_translation",
        "synthesize_decomp",
        "register_blackbox",
    }


def test_diagnose_exported_program_standalone_helper(tmp_path: Path) -> None:
    """The no-session standalone helper must produce the same shape."""
    mod = _WithUnsupported().eval()
    ep = torch.export.export(mod, (torch.randn(1, 32),))
    from compgen.capture.unsupported.introspect import runtime_versions

    _ = runtime_versions()
    result = diagnose_exported_program(ep)
    assert result["ok"]
    assert result["num_issues"] >= 1
    assert any(i["target"] == "aten.tanh.default" for i in result["issues"])


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_synthesize_decomp_not_on_allow_list_returns_remediation(tmp_path: Path) -> None:
    """tanh is NOT on the allow-list -> structured failure with remediation hint."""
    sm, sid = _prepared_session(tmp_path)
    result = synthesize_decomp(sm, session_id=sid, op_target="aten.tanh.default")
    assert result["ok"] is False
    assert result["reason"] == "not_on_allow_list"
    assert "remediation_hint" in result


def test_synthesize_translation_succeeds_for_tanh(tmp_path: Path) -> None:
    """tanh has a simple Tensor-in/Tensor-out schema — translation path works."""
    sm, sid = _prepared_session(tmp_path)
    result = synthesize_translation(sm, session_id=sid, op_target="aten.tanh.default")
    assert result["ok"]
    assert result["callee_name"] == "aten_tanh_default"
    assert result["strategy"] == "payload_translation"
    # recovery_status reflects the decision.
    status = recovery_status(sm, session_id=sid)
    assert status["translations"]["aten.tanh.default"] == "aten_tanh_default"


def test_register_blackbox_records_promotion(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    result = register_blackbox(sm, session_id=sid, op_target="aten.tanh.default")
    assert result["ok"]
    assert len(result["promotion_record"]["cache_key"]) == 16
    status = recovery_status(sm, session_id=sid)
    assert "aten.tanh.default" in status["blackboxes"]


def test_resolve_unsupported_op_auto_strategy(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    result = resolve_unsupported_op(
        sm,
        session_id=sid,
        op_target="aten.tanh.default",
        strategy="auto",
    )
    assert result["ok"]
    # For tanh the auto path should pick translation (the classifier said so).
    assert result["attempted_strategy"] in {"translation", "decomp", "blackbox"}


def test_resolve_unsupported_op_blackbox_override(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    result = resolve_unsupported_op(
        sm,
        session_id=sid,
        op_target="aten.tanh.default",
        strategy="blackbox",
    )
    assert result["ok"]
    assert result["attempted_strategy"] == "blackbox"


def test_resolve_unsupported_op_unknown_strategy(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    result = resolve_unsupported_op(
        sm,
        session_id=sid,
        op_target="aten.tanh.default",
        strategy="nonsense",
    )
    assert result["ok"] is False
    assert "Unknown strategy" in result["error"]


def test_recovery_tool_unknown_target(tmp_path: Path) -> None:
    sm, sid = _prepared_session(tmp_path)
    for fn in (synthesize_decomp, synthesize_translation, register_blackbox):
        result = fn(sm, session_id=sid, op_target="aten.there_is_no_such_op.default")
        assert result["ok"] is False
        assert "not in session" in result["error"]
