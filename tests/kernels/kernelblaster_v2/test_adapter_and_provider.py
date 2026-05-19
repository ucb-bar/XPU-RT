"""Adapter + provider tests for KernelBlaster v2."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    KernelGeneratorMock,
    ProposeResponse,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators import EvaluationReport, MockEvaluator
from xpu_rt.kernels.kernelblaster_v2.generators import AgentFileBridge
from xpu_rt.kernels.kernelblaster_v2_adapter import (
    ENV_MODE,
    MODE_LLM_LIVE,
    KernelBlasterV2Adapter,
    KernelBlasterV2Unavailable,
)
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.kernels.providers.kernelblaster_v2 import KernelBlasterV2Provider
from xpu_rt.memory import target_knowledge as tk


@pytest.fixture
def isolated_knowledge_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path))
    return tmp_path


def _seed_minimal_card(target_id: str = "demo_target") -> tk.TargetKnowledgeCard:
    return tk.save(
        tk.TargetKnowledgeCard(
            target_id=target_id,
            target_profile_ref=f"configs/targets/{target_id}.yaml",
            hardware_spec=tk.HardwareSpec(isa_family="rocc-systolic"),
        )
    )


def _contract(target_id: str = "demo_target") -> KernelContract:
    return KernelContract(
        op_family="matmul",
        input_shapes=((64, 64), (64, 64)),
        output_shapes=((64, 64),),
        dtypes=("i8", "i32"),
        layout="row_major",
        target_name=target_id,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_adapter_raises_when_card_missing(isolated_knowledge_dir: Path) -> None:
    adapter = KernelBlasterV2Adapter(
        generator=KernelGeneratorMock(),
        evaluator=MockEvaluator(),
    )
    with pytest.raises(KernelBlasterV2Unavailable, match="no target knowledge card"):
        adapter.search_kernel(_contract())


def test_adapter_raises_when_target_name_missing(isolated_knowledge_dir: Path) -> None:
    _seed_minimal_card()
    adapter = KernelBlasterV2Adapter(
        generator=KernelGeneratorMock(),
        evaluator=MockEvaluator(),
    )
    contract = KernelContract(op_family="matmul", dtypes=("i8",))
    with pytest.raises(KernelBlasterV2Unavailable, match="target_name"):
        adapter.search_kernel(contract)


def test_adapter_runs_loop_in_process(isolated_knowledge_dir: Path) -> None:
    _seed_minimal_card()
    gen = KernelGeneratorMock(
        table=lambda req: ProposeResponse(
            kernel_code="// ok", language="c", action="run-1"
        )
    )
    evl = MockEvaluator(table=lambda c: EvaluationReport(correct=True, score=1.3))
    adapter = KernelBlasterV2Adapter(generator=gen, evaluator=evl)
    result = adapter.search_kernel(_contract())
    assert result.found is True
    assert result.plan == "run-1"
    assert result.metadata["best_action"] == "run-1"


def test_adapter_is_available_signal(isolated_knowledge_dir: Path) -> None:
    adapter = KernelBlasterV2Adapter(
        generator=KernelGeneratorMock(),
        evaluator=MockEvaluator(),
    )
    ok, reason = adapter.is_available(target_id="demo_target")
    assert not ok and "no knowledge card" in reason
    _seed_minimal_card()
    ok, reason = adapter.is_available(target_id="demo_target")
    assert ok and reason == "ok"


def test_adapter_default_generator_requires_explicit_choice(
    isolated_knowledge_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_minimal_card()
    monkeypatch.delenv(ENV_MODE, raising=False)
    adapter = KernelBlasterV2Adapter()  # no generator, no bridge, no mode
    with pytest.raises(KernelBlasterV2Unavailable, match="No generator chosen"):
        adapter.search_kernel(_contract())


def test_adapter_env_selects_llm_live(
    isolated_knowledge_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_minimal_card()
    monkeypatch.setenv(ENV_MODE, MODE_LLM_LIVE)
    adapter = KernelBlasterV2Adapter(evaluator=MockEvaluator())
    gen = adapter._resolve_generator()
    assert gen.name == "gemini"


def test_adapter_with_bridge_picks_agent_file(isolated_knowledge_dir: Path) -> None:
    _seed_minimal_card()
    requests: list[dict] = []

    def emit(envelope: dict) -> str:
        requests.append(envelope)
        return f"req-{len(requests)}"

    def wait(rid: str) -> dict:
        return {
            "payload": {
                "kernel_code": "// agent",
                "language": "c",
                "action": "agent-tile",
            }
        }

    bridge = AgentFileBridge(emit_request=emit, wait_response=wait)
    adapter = KernelBlasterV2Adapter(
        bridge=bridge,
        evaluator=MockEvaluator(table=lambda c: EvaluationReport(correct=True, score=2.0)),
        config=AgentLoopConfig(max_iterations=1),
    )
    result = adapter.search_kernel(_contract())
    assert result.found is True
    assert result.plan == "agent-tile"
    assert requests, "agent-file bridge must have received a request"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_provider_accepts_only_when_card_present(isolated_knowledge_dir: Path) -> None:
    provider = KernelBlasterV2Provider()
    assert not provider.accepts_contract(_contract())
    _seed_minimal_card()
    assert provider.accepts_contract(_contract())


def test_provider_declines_when_adapter_unavailable(
    isolated_knowledge_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider must report a clean failure rather than raise."""
    _seed_minimal_card()
    monkeypatch.delenv(ENV_MODE, raising=False)
    provider = KernelBlasterV2Provider()  # default adapter has no generator choice
    result = provider.search(_contract())
    assert result.found is False
    assert "declined_reason" in result.metadata


def test_provider_routes_to_adapter(isolated_knowledge_dir: Path) -> None:
    _seed_minimal_card()
    adapter = KernelBlasterV2Adapter(
        generator=KernelGeneratorMock(
            table=lambda req: ProposeResponse(kernel_code="// via provider", action="ok", language="c")
        ),
        evaluator=MockEvaluator(table=lambda c: EvaluationReport(correct=True, score=1.5)),
    )
    provider = KernelBlasterV2Provider(adapter=adapter)
    result = provider.search(_contract())
    assert result.found is True
    assert result.kernel_code == "// via provider"
    exports = provider.export_knowledge()
    assert exports
    # export_knowledge drains the buffer.
    assert provider.export_knowledge() == []


def test_provider_priority_above_legacy() -> None:
    """v2 must rank above legacy KB so the registry picks it when both apply."""
    legacy_priority = 90  # kernels.providers.kernelblaster.KernelBlasterProvider.priority
    assert KernelBlasterV2Provider().priority > legacy_priority
