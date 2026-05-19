"""Hermetic tests for the KernelBlaster v2 agent loop.

Drives the loop with :class:`KernelGeneratorMock` + :class:`MockEvaluator`
so no Gemini spend and no compile/run dependencies are involved. The
real evaluator implementations are stubbed; the agent-loop control
flow (propose → evaluate → repair → persist) is exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    AgentLoopResult,
    KernelBlasterV2,
    KernelGeneratorMock,
    ProposeRequest,
    ProposeResponse,
    StateVector,
    derive_state,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators import EvaluationReport, MockEvaluator
from xpu_rt.kernels.kernelblaster_v2.generators import (
    AgentFileBridge,
    KernelGeneratorAgentFile,
    KernelGeneratorLLM,
)
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBuilder
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB, StrategyEntry
from xpu_rt.kernels.provider import ContractFeedback, KernelContract
from xpu_rt.memory import target_knowledge as tk
from xpu_rt.observability import gemini_usage as gu


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_knowledge_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path))
    return tmp_path


def _seed_card(target_id: str = "demo_target") -> tk.TargetKnowledgeCard:
    card = tk.TargetKnowledgeCard(
        target_id=target_id,
        target_profile_ref=f"configs/targets/{target_id}.yaml",
        hardware_spec=tk.HardwareSpec(
            isa_family="rocc-systolic",
            parameters=(
                tk.ParameterRange(name="meshRows", description="mesh rows", default=16),
            ),
            memory_tiers=(
                tk.MemoryTierSpec(name="scratchpad", kind="scratchpad", size_bytes=262144),
            ),
            instructions=(
                tk.ISAInstruction(mnemonic="mvin", signature="rs1, rs2", funct_code=2),
                tk.ISAInstruction(
                    mnemonic="matmul.preload",
                    signature="rs1, rs2",
                    summary="preload weights",
                    funct_code=6,
                ),
            ),
            intrinsics=(
                tk.IntrinsicSignature(
                    name="gemmini_mvin",
                    c_signature="#define gemmini_mvin(d, s)",
                    summary="DMA into scratchpad",
                ),
            ),
            dataflow_modes=("weight_stationary", "output_stationary"),
            constraints=("scratchpad addresses must be DIM-aligned",),
        ),
        exemplars=(
            tk.KernelExemplar(
                name="matmul_ref",
                op_family="matmul",
                path="matmul_ref.c",
                language="c",
            ),
        ),
    )
    return tk.save(card)


def _contract(op: str = "matmul") -> KernelContract:
    return KernelContract(
        region_id="r0",
        op_family=op,
        input_shapes=((128, 256), (256, 64)),
        output_shapes=((128, 64),),
        dtypes=("i8", "i32"),
        layout="row_major",
        target_name="demo_target",
        hardware_key="demo",
        objective="latency",
    )


# ---------------------------------------------------------------------------
# StateVector
# ---------------------------------------------------------------------------


def test_state_vector_canonicalizes_dtype_and_op_family() -> None:
    state = derive_state(_contract("GEMM"))
    assert state.op_family == "matmul"
    # mixed dtypes (i8 input, i32 acc) collapse to "mixed".
    assert state.dtype_class == "mixed"
    assert state.layout_kind == "row_major"
    assert state.archetype == "COMPUTE_TILED"
    # Shape signature bucketizes.
    assert state.shape_signature  # non-empty
    # Hash is stable and 16 hex chars.
    h = state.hash()
    assert len(h) == 16


def test_state_vector_unknown_op_family_preserved_lowercase() -> None:
    contract = KernelContract(op_family="WeirdOp", dtypes=("fp32",), layout="row_major")
    state = derive_state(contract)
    assert state.op_family == "weirdop"
    assert state.archetype == "unknown"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_prompt_builder_includes_target_and_contract(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()
    state = derive_state(contract, target_id=card.target_id)
    builder = PromptBuilder(card=card)
    bundle = builder.build(contract=contract, state=state)
    assert "matmul" in bundle.user
    assert "demo_target" in bundle.user
    assert "## Contract" in bundle.user
    assert "## Target" in bundle.user
    # Schema is the kernel-proposal schema.
    assert "kernel_code" in bundle.schema["properties"]


def test_prompt_builder_surfaces_strategy_rows(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()
    state = derive_state(contract, target_id=card.target_id)
    db = StrategyDB.for_card(card)
    db.record(state=state, action="tile-K=64", accepted=True, speedup=1.4)
    db.record(state=state, action="use-mvin-stride", accepted=True, speedup=1.1)
    builder = PromptBuilder(card=card, strategy_db=db)
    bundle = builder.build(contract=contract, state=state)
    assert "Top strategies" in bundle.user
    assert "tile-K=64" in bundle.user


def test_prompt_builder_filters_isa_by_op_family(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract(op="matmul")
    state = derive_state(contract, target_id=card.target_id)
    bundle = PromptBuilder(card=card).build(contract=contract, state=state)
    # matmul.preload should survive; "mvin" survives via the matmul keyword set.
    assert "matmul.preload" in bundle.user
    assert "gemmini_mvin" in bundle.user


def test_prompt_builder_renders_mega_chain_section(isolated_knowledge_dir: Path) -> None:
    """MEGA-aware branch — when a fused chain is passed in, the prompt
    must expose body[] members + internal_events so the LLM knows to
    emit a single fused kernel rather than the top-level contract's
    single op. Regression guard against the prompt silently dropping
    chain structure."""
    from xpu_rt.ir.payload.contract_graph import ContractEdge, ContractNode, build_contract_graph_from_nodes
    from xpu_rt.ir.payload.contracts import (
        CostEstimate,
        KernelContract as IRKernelContract,
        LayoutKind as IRLayoutKind,
        LayoutRequirement,
    )
    from xpu_rt.kernels.contract_v3 import HardwareEnvelope
    from xpu_rt.kernels.fusion_planner import FusionCluster
    from xpu_rt.kernels.mega_contract_emitter import emit_mega_contract

    env = HardwareEnvelope(
        target_name="demo_target",
        vector_lanes=16,
        scratchpad_bytes=262144,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
    )
    m1_v1 = IRKernelContract(
        op_name="matmul",
        input_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR), LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"}, cost=CostEstimate(flops=1),
        metadata={"input_shapes": [(64, 720), (720, 1440)], "output_shapes": [(64, 1440)], "region_id": "m1", "dispatch_id": "m1"},
    )
    silu_v1 = IRKernelContract(
        op_name="silu",
        input_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"}, cost=CostEstimate(flops=1), fusable=True,
        metadata={"input_shapes": [(64, 1440)], "output_shapes": [(64, 1440)], "region_id": "s1", "dispatch_id": "s1"},
    )
    m2_v1 = IRKernelContract(
        op_name="matmul",
        input_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR), LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(IRLayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"}, cost=CostEstimate(flops=1),
        metadata={"input_shapes": [(64, 1440), (1440, 720)], "output_shapes": [(64, 720)], "region_id": "m2", "dispatch_id": "m2"},
    )
    graph = build_contract_graph_from_nodes(
        [
            ContractNode(op_id="n_m1", contract=m1_v1, op_name="matmul", region_id="m1"),
            ContractNode(op_id="n_s1", contract=silu_v1, op_name="silu", region_id="s1"),
            ContractNode(op_id="n_m2", contract=m2_v1, op_name="matmul", region_id="m2"),
        ],
        [
            ContractEdge("n_m1", "n_s1", 0, (64, 1440), "i8", 1),
            ContractEdge("n_s1", "n_m2", 0, (64, 1440), "i8", 1),
        ],
    )
    mega = emit_mega_contract(
        FusionCluster("cluster_t", ("n_m1", "n_s1", "n_m2"), "test", 2.0),
        graph, env,
    ).contract

    card = _seed_card()
    contract = _contract()
    state = derive_state(contract, target_id=card.target_id)
    bundle = PromptBuilder(card=card).build(contract=contract, state=state, mega_contract=mega)

    assert "## MEGA fused chain" in bundle.user
    assert "body[] length: 3" in bundle.user
    # The chain narrative names each sub-op.
    assert "matmul" in bundle.user
    assert "silu" in bundle.user
    # The "fused kernel" final instruction must appear (not the single-op one).
    assert "fused kernel" in bundle.user
    assert "intermediates held in scratchpad" in bundle.user
    # Metadata signals the MEGA branch to downstream telemetry.
    assert bundle.metadata["is_mega"] is True
    assert bundle.metadata["mega_body_size"] == 3


# ---------------------------------------------------------------------------
# Strategy DB
# ---------------------------------------------------------------------------


def test_strategy_db_record_and_top(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    state = derive_state(_contract(), target_id=card.target_id)
    db = StrategyDB.for_card(card)

    db.record(state=state, action="tile-K=64", accepted=True, speedup=1.4)
    db.record(state=state, action="tile-K=64", accepted=True, speedup=1.6)
    db.record(state=state, action="bad-action", accepted=False, speedup=0.0)

    top = db.top_for_state(state, limit=5)
    actions = [r.action for r in top]
    assert "tile-K=64" in actions
    assert "bad-action" not in actions, "rejected-only rows must be filtered out"
    entry = next(r for r in top if r.action == "tile-K=64")
    assert entry.mean_speedup == pytest.approx(1.5, rel=1e-3)
    assert entry.sample_count == 2
    assert entry.accepted_count == 2


def test_strategy_db_persists_across_instances(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    state = derive_state(_contract(), target_id=card.target_id)
    StrategyDB.for_card(card).record(
        state=state, action="tile-K=32", accepted=True, speedup=1.2
    )
    fresh = StrategyDB.for_card(card)
    top = fresh.top_for_state(state)
    assert top and top[0].action == "tile-K=32"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def test_agent_loop_short_circuits_on_accept(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()

    gen = KernelGeneratorMock(
        table=lambda req: ProposeResponse(
            kernel_code="// kernel ok",
            language="c",
            action="tile-K=64",
            rationale="meet contract",
        )
    )
    evl = MockEvaluator(
        table=lambda c: EvaluationReport(correct=True, score=1.5, cycles=1000)
    )
    loop = KernelBlasterV2(card=card, generator=gen, evaluator=evl)
    result = loop.run(contract)

    assert result.found()
    assert len(result.history) == 1, "should short-circuit after first accept"
    assert result.best is not None
    assert result.best.proposal.action == "tile-K=64"
    assert result.best.report.cycles == 1000

    # Lesson row persisted.
    lessons = list(tk.iter_lessons(card))
    assert any(l.action == "tile-K=64" for l in lessons)
    # Strategy DB updated.
    db = StrategyDB.for_card(card)
    state = derive_state(contract, target_id=card.target_id)
    rows = db.top_for_state(state)
    assert rows and rows[0].action == "tile-K=64"


def test_agent_loop_iterates_until_correct(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()

    proposals = [
        ProposeResponse(kernel_code="// v0 broken", language="c", action="v0"),
        ProposeResponse(kernel_code="// v1 broken", language="c", action="v1"),
        ProposeResponse(kernel_code="// v2 ok", language="c", action="v2-good"),
    ]
    reports = [
        EvaluationReport(correct=False, score=0.0, diff_summary="rows wrong"),
        EvaluationReport(correct=False, score=0.0, diff_summary="cols wrong"),
        EvaluationReport(correct=True, score=2.0, cycles=500),
    ]
    proposal_iter = iter(proposals)
    report_iter = iter(reports)

    gen = KernelGeneratorMock(table=lambda req: next(proposal_iter))
    evl = MockEvaluator(table=lambda c: next(report_iter))
    loop = KernelBlasterV2(
        card=card,
        generator=gen,
        evaluator=evl,
        config=AgentLoopConfig(max_iterations=5),
    )
    result = loop.run(contract)
    assert result.found()
    assert result.best is not None
    assert result.best.proposal.action == "v2-good"
    assert len(result.history) == 3


def test_agent_loop_records_each_attempt_in_strategy_db(
    isolated_knowledge_dir: Path,
) -> None:
    card = _seed_card()
    contract = _contract()
    proposals = [
        ProposeResponse(kernel_code="// a", action="a", language="c"),
        ProposeResponse(kernel_code="// b", action="b", language="c"),
    ]
    reports = [
        EvaluationReport(correct=False, score=0.0),
        EvaluationReport(correct=True, score=1.2),
    ]
    proposal_iter = iter(proposals)
    report_iter = iter(reports)
    gen = KernelGeneratorMock(table=lambda req: next(proposal_iter))
    evl = MockEvaluator(table=lambda c: next(report_iter))
    loop = KernelBlasterV2(
        card=card,
        generator=gen,
        evaluator=evl,
        config=AgentLoopConfig(max_iterations=3),
    )
    loop.run(contract)

    db = StrategyDB.for_card(card)
    state = derive_state(contract, target_id=card.target_id)
    keys = {row.action for row in db.rows.values() if row.state_key == state.key()}
    assert {"a", "b"}.issubset(keys)


def test_agent_loop_handles_generator_exceptions(
    isolated_knowledge_dir: Path,
) -> None:
    card = _seed_card()
    contract = _contract()

    def raise_on_first(req: ProposeRequest) -> ProposeResponse:
        raise RuntimeError("generator outage")

    gen = KernelGeneratorMock(table=raise_on_first)
    evl = MockEvaluator()
    loop = KernelBlasterV2(
        card=card,
        generator=gen,
        evaluator=evl,
        config=AgentLoopConfig(max_iterations=2, strict=False),
    )
    result = loop.run(contract)
    assert result.aborted
    assert "RuntimeError" in result.abort_reason
    assert result.best is None
    # to_provider_result still returns a clean ProviderResult.
    pr = result.to_provider_result()
    assert pr.found is False
    assert pr.metadata["aborted"] is True


def test_agent_loop_to_provider_result(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()
    gen = KernelGeneratorMock(
        table=lambda req: ProposeResponse(
            kernel_code="// good",
            language="c",
            action="tile-K=64",
            contract_feedback=(
                ContractFeedback(
                    field="layout",
                    current_value="row_major",
                    suggested_value="packed_k",
                    reason="systolic prefers packed_k",
                    kind="layout_swap",
                ),
            ),
        )
    )
    evl = MockEvaluator(table=lambda c: EvaluationReport(correct=True, score=1.5, cycles=200))
    result = KernelBlasterV2(card=card, generator=gen, evaluator=evl).run(contract)
    pr = result.to_provider_result()
    assert pr.found is True
    assert pr.kernel_code.startswith("//")
    assert pr.plan == "tile-K=64"
    assert pr.contract_feedback[0].field == "layout"
    assert pr.knowledge_exports
    assert pr.knowledge_exports[0].kind == "kernelblaster_v2_best_kernel"


# ---------------------------------------------------------------------------
# Agent-file generator round-trip
# ---------------------------------------------------------------------------


def test_generator_agent_file_round_trip(isolated_knowledge_dir: Path) -> None:
    card = _seed_card()
    contract = _contract()
    state = derive_state(contract, target_id=card.target_id)
    builder = PromptBuilder(card=card)
    bundle = builder.build(contract=contract, state=state)

    response_payload = {
        "kernel_code": "// agent kernel",
        "language": "c",
        "action": "agent-tile",
        "rationale": "ok",
    }
    requests: list[dict] = []

    def emit(envelope: dict) -> str:
        requests.append(envelope)
        return f"req-{len(requests)}"

    def wait(rid: str) -> dict:
        return {"payload": response_payload}

    bridge = AgentFileBridge(emit_request=emit, wait_response=wait)
    gen = KernelGeneratorAgentFile(bridge=bridge)
    out = gen.propose(ProposeRequest(bundle=bundle, attempt_index=0, state_hash=state.hash()))
    assert out.kernel_code == "// agent kernel"
    assert out.action == "agent-tile"
    assert requests[0]["kind"] == "kernelblaster_v2_propose"
    assert requests[0]["metadata"]["state_hash"] == state.hash()


# ---------------------------------------------------------------------------
# LLM generator + budget gate
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_GEMINI_USAGE_DIR", str(tmp_path / "usage"))
    monkeypatch.setenv("XPU_RT_REPO_ROOT", str(tmp_path))
    return tmp_path


def test_generator_llm_blocks_when_budget_exceeded(
    isolated_usage: Path,
    isolated_knowledge_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    gu.record_call("gemini-2.5-flash", 0, 40_000_000)

    monkeypatch.setattr(
        "xpu_rt.llm.factory.create_llm_client",
        lambda provider, model=None: pytest.fail("LLM client must not be constructed"),
    )

    card = _seed_card()
    contract = _contract()
    state = derive_state(contract, target_id=card.target_id)
    bundle = PromptBuilder(card=card).build(contract=contract, state=state)
    request = ProposeRequest(bundle=bundle, attempt_index=0, state_hash=state.hash())

    with pytest.raises(gu.GeminiBudgetExceeded):
        KernelGeneratorLLM().propose(request)
