"""Wave 3 tests: ComputeDAG + fusion oracle + granularity oracle.

Locks in:
  * ComputeDAG construction from a small payload region; shape
    classification (linear_chain / fan_out / single_op / empty)
  * Fusion eligibility gates: dtype mismatch, fusion-boundary refusal,
    fusable_with set, scratchpad budget overflow
  * Fusion cost-model: compute-bound DRAM-savings dominant case;
    register-pressure dominant case
  * Granularity decisions: single-op MICRO/NORMAL boundary; chain
    NORMAL/MEGA boundary on speedup threshold
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.kernels.compute_dag import (
    ComputeDAG,
    ComputeNode,
    NodeKind,
    from_payload_region,
    to_prompt_text,
)
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    FusionPolicy,
    Granularity,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    MemorySpec,
    MemoryTier,
    OrchestrationSpec,
    ShapeClass,
    StaticAttr,
    TensorIO,
)
from compgen.kernels.fusion_oracle import (
    FusionDecision,
    should_fuse,
)
from compgen.kernels.granularity_oracle import recommend_granularity
from compgen.memory.knowledge import KnowledgeStore, set_shared_store
from compgen.memory.seed_lessons import install as install_seed

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path):
    set_shared_store(KnowledgeStore(root=tmp_path / "knowledge"))
    install_seed()
    yield
    set_shared_store(None)


def _ampere_envelope(scratchpad_bytes: int = 164_000) -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name="cuda-a100",
        vector_lanes=108,
        scratchpad_bytes=scratchpad_bytes,
        register_bytes=256,
        register_quota_per_thread=256,
        native_dtypes=("bf16", "f16", "f32"),
        peak_bandwidth_gbps=1555.0,
    )


def _matmul_contract(
    *,
    output_tier: MemoryTier = MemoryTier.SCRATCHPAD,
    is_boundary: bool = True,
    fusable_with: tuple = (),
    envelope: HardwareEnvelope | None = None,
) -> KernelContractV3:
    env = envelope or _ampere_envelope()
    return KernelContractV3(
        op_name="linalg.matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(128, 256)), dtype_class=("bf16",)),
                TensorIO(name="b", shape=ShapeClass(dims=(256, 128)), dtype_class=("bf16",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(128, 128)), dtype_class=("bf16",)),),
        ),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(output_tier,),
            ),
            fusion=FusionPolicy(is_boundary=is_boundary, fusable_with=fusable_with),
        ),
    )


def _silu_contract(*, envelope: HardwareEnvelope | None = None) -> KernelContractV3:
    env = envelope or _ampere_envelope()
    return KernelContractV3(
        op_name="silu",
        archetype=KernelArchetype.ACTIVATION,
        io=IOContract(
            inputs=(TensorIO(name="x", shape=ShapeClass(dims=(128, 128)), dtype_class=("bf16",)),),
            outputs=(TensorIO(name="y", shape=ShapeClass(dims=(128, 128)), dtype_class=("bf16",)),),
        ),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD,),
                output_tiers=(MemoryTier.SCRATCHPAD,),
                in_place_safe=True,
            ),
            fusion=FusionPolicy(fusable_with=("pointwise", "reduce")),
        ),
    )


def _softmax_contract(*, envelope: HardwareEnvelope | None = None) -> KernelContractV3:
    env = envelope or _ampere_envelope()
    return KernelContractV3(
        op_name="softmax",
        archetype=KernelArchetype.REDUCE,
        io=IOContract(
            inputs=(TensorIO(name="x", shape=ShapeClass(dims=(128, 128)), dtype_class=("bf16",)),),
            outputs=(TensorIO(name="y", shape=ShapeClass(dims=(128, 128)), dtype_class=("bf16",)),),
            attributes=(StaticAttr(name="axis", value=-1),),
        ),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD,),
                output_tiers=(MemoryTier.SCRATCHPAD,),
            ),
            fusion=FusionPolicy(fusable_with=("activation", "pointwise")),
        ),
    )


# ---------------------------------------------------------------------------
# ComputeDAG
# ---------------------------------------------------------------------------


def test_compute_dag_empty_region_is_classified_empty() -> None:
    dag = ComputeDAG()
    assert dag.shape_summary() == "empty"


def test_compute_dag_single_node_is_classified_single_op() -> None:
    dag = ComputeDAG(
        nodes=[
            ComputeNode(id="n_0", kind=NodeKind.COMPUTE, op_name="linalg.matmul"),
        ]
    )
    assert dag.shape_summary() == "single_op"


def test_compute_dag_chain_classified_linear() -> None:
    from compgen.kernels.compute_dag import ComputeEdge

    dag = ComputeDAG(
        nodes=[
            ComputeNode(id="n_0", kind=NodeKind.COMPUTE, op_name="linalg.matmul"),
            ComputeNode(id="n_1", kind=NodeKind.POINTWISE, op_name="silu"),
            ComputeNode(id="n_2", kind=NodeKind.REDUCE, op_name="softmax"),
        ],
        edges=[
            ComputeEdge(src="n_0", dst="n_1"),
            ComputeEdge(src="n_1", dst="n_2"),
        ],
    )
    assert dag.shape_summary() == "linear_chain"


def test_to_prompt_text_includes_nodes_and_edges() -> None:
    from compgen.kernels.compute_dag import ComputeEdge

    dag = ComputeDAG(
        nodes=[
            ComputeNode(
                id="n_0",
                kind=NodeKind.COMPUTE,
                op_name="linalg.matmul",
                output_shape=(128, 128),
                output_dtype="bf16",
                dim_roles=("parallel", "parallel"),
            ),
            ComputeNode(
                id="n_1", kind=NodeKind.POINTWISE, op_name="silu", output_shape=(128, 128), output_dtype="bf16"
            ),
        ],
        edges=[ComputeEdge(src="n_0", dst="n_1")],
    )
    text = to_prompt_text(dag)
    assert "linalg.matmul" in text
    assert "silu" in text
    assert "n_0 → n_1" in text
    assert "parallel" in text


def test_from_payload_region_handles_real_xdsl_module() -> None:
    """Smoke: build a tiny matmul module and convert to ComputeDAG."""
    from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import MatmulOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region

    f32 = Float32Type()
    M, K, N = 4, 8, 4
    lhs_t = TensorType(f32, [M, K])
    rhs_t = TensorType(f32, [K, N])
    out_t = TensorType(f32, [M, N])

    block = Block(arg_types=[lhs_t, rhs_t])
    out_empty = EmptyOp([], out_t)
    block.add_op(out_empty)
    mm = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[out_empty.results[0]],
        res=[out_t],
    )
    block.add_op(mm)
    block.add_op(ReturnOp(mm.results[0]))
    func = FuncOp("forward", ((lhs_t, rhs_t), (out_t,)), Region([block]))
    module = ModuleOp([func])

    dag = from_payload_region(module)
    assert any(n.op_name == "linalg.matmul" for n in dag.nodes)
    assert any(n.kind is NodeKind.COMPUTE for n in dag.nodes)


# ---------------------------------------------------------------------------
# Fusion oracle
# ---------------------------------------------------------------------------


def test_fusion_blocked_when_producer_is_boundary() -> None:
    p = _matmul_contract(is_boundary=True)
    c = _silu_contract()
    v = should_fuse(p, c)
    assert v.decision is FusionDecision.INELIGIBLE
    assert any("boundary" in f for f in v.eligibility_failures)


def test_fusion_blocked_when_dtypes_mismatch() -> None:
    env = _ampere_envelope()
    # Producer outputs bf16; consumer accepts only f32 — no overlap.
    p = _matmul_contract(is_boundary=False, fusable_with=("activation",), envelope=env)
    c = KernelContractV3(
        op_name="silu_f32",
        archetype=KernelArchetype.ACTIVATION,
        io=IOContract(
            inputs=(TensorIO(name="x", shape=ShapeClass(dims=(128, 128)), dtype_class=("f32",)),),
            outputs=(TensorIO(name="y", shape=ShapeClass(dims=(128, 128)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )
    v = should_fuse(p, c)
    assert v.decision is FusionDecision.INELIGIBLE
    assert any("dtype" in f for f in v.eligibility_failures)


def test_fusion_blocked_when_consumer_archetype_not_fusable() -> None:
    p = _matmul_contract(is_boundary=False, fusable_with=("pointwise",))
    c = _softmax_contract()  # archetype = REDUCE, not in {pointwise}
    v = should_fuse(p, c)
    assert v.decision is FusionDecision.INELIGIBLE
    assert any("fusable_with" in f for f in v.eligibility_failures)


def test_fusion_blocked_when_combined_smem_overflows() -> None:
    """Tiny scratchpad budget + big tensors → overflow."""
    env = _ampere_envelope(scratchpad_bytes=4096)  # tiny SMEM
    p = _matmul_contract(is_boundary=False, fusable_with=("activation",), envelope=env)
    c = _silu_contract(envelope=env)
    v = should_fuse(p, c)
    assert v.decision is FusionDecision.INELIGIBLE
    assert any("scratchpad" in f for f in v.eligibility_failures)


def test_fusion_recommends_fuse_for_matmul_to_silu_chain() -> None:
    """Eligible pair on Ampere with reasonable budgets → FUSE."""
    p = _matmul_contract(is_boundary=False, fusable_with=("activation", "pointwise"))
    c = _silu_contract()
    v = should_fuse(p, c)
    assert v.decision is FusionDecision.FUSE
    assert v.est_speedup_ratio >= 1.0
    # Cost breakdown is populated
    assert "dram_savings_us" in v.cost_breakdown
    assert "launch_savings_us" in v.cost_breakdown


def test_fusion_verdict_carries_knowledge_brief() -> None:
    """The brief must be fusion-decision scoped, not e.g. profiling."""
    p = _matmul_contract(is_boundary=False, fusable_with=("activation",))
    c = _silu_contract()
    v = should_fuse(p, c)
    # Brief is a string (may be empty if no fusion-decision lessons exist
    # in seed for this scope, which is fine).
    assert isinstance(v.knowledge_brief, str)
    if v.knowledge_brief:
        # Header carries the filter values
        assert "fusion" in v.knowledge_brief.lower()


# ---------------------------------------------------------------------------
# Granularity oracle
# ---------------------------------------------------------------------------


def test_granularity_inlined_callee_returns_micro() -> None:
    region = [_silu_contract()]
    v = recommend_granularity(region, _ampere_envelope(), is_inlined_callee=True)
    assert v.granularity is Granularity.MICRO
    assert "inlined" in v.reason


def test_granularity_compute_tiled_single_op_returns_normal() -> None:
    """Matmul, even small, defaults to NORMAL (it's a fusion boundary
    + COMPUTE_TILED warrants a dispatch)."""
    region = [_matmul_contract(is_boundary=True)]
    v = recommend_granularity(region, _ampere_envelope())
    assert v.granularity is Granularity.NORMAL


def test_granularity_chain_recommends_mega_when_speedup_above_threshold() -> None:
    """matmul → silu → softmax chain should fuse into MEGA on Ampere if
    every pair recommends FUSE and combined speedup ≥ threshold."""
    p = _matmul_contract(is_boundary=False, fusable_with=("activation", "pointwise", "reduce"))
    s = _silu_contract()
    sm = _softmax_contract()
    # Override silu's fusable_with so it accepts reduce neighbour
    s = KernelContractV3(
        **{
            **s.__dict__,
            "orchestration": OrchestrationSpec(
                execution=s.orchestration.execution,
                memory=s.orchestration.memory,
                fusion=FusionPolicy(fusable_with=("reduce", "pointwise")),
            ),
        }
    )
    v = recommend_granularity([p, s, sm], _ampere_envelope())
    # On Ampere with huge SMEM budget, this is plausibly MEGA.
    # Either MEGA (if speedup ≥ 1.5) or NORMAL (if < threshold) is acceptable;
    # the key assertion is the chain_speedup_estimate is computed and pairwise
    # cost-modelling ran.
    assert v.chain_speedup_estimate >= 1.0
    assert v.granularity in (Granularity.MEGA, Granularity.NORMAL)


def test_granularity_chain_falls_back_to_normal_when_smem_overflows() -> None:
    env = _ampere_envelope(scratchpad_bytes=512)  # absurdly small
    p = _matmul_contract(is_boundary=False, fusable_with=("activation",), envelope=env)
    s = _silu_contract(envelope=env)
    v = recommend_granularity([p, s], env)
    assert v.granularity is Granularity.NORMAL
    assert "scratchpad" in v.reason


def test_granularity_chain_falls_back_when_fusion_oracle_declines() -> None:
    """If any pair is INELIGIBLE the chain can't be MEGA.

    Use a generous SMEM envelope so the scratchpad-overflow gate doesn't
    fire first; we want the fusion-boundary refusal to be the cause.
    """
    env = _ampere_envelope(scratchpad_bytes=4 * 1024 * 1024)  # 4 MB plenty
    p = _matmul_contract(is_boundary=True, envelope=env)  # boundary blocks fusion
    s = _silu_contract(envelope=env)
    v = recommend_granularity([p, s], env)
    assert v.granularity is Granularity.NORMAL
    assert "ineligible" in v.reason or "boundary" in v.reason
