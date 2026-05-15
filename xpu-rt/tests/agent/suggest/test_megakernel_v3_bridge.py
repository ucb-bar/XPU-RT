"""Tests for ``compgen.agent.suggest.megakernel_v3_bridge``.

Locks in:
  * Empty / missing ``fused_region_refs`` raises ValueError
  * Sub-contracts in DEVICE_DRAM are silently promoted to SCRATCHPAD
  * Result MEGA contract has Granularity.MEGA, PERSISTENT dispatch,
    body[] populated, and a linear chain of internal_events
  * The MEGA invariants on the resulting v3 contract pass
    (this is what __post_init__ enforces)
  * Nested-MEGA sub-contracts are filtered out with a note
"""

from __future__ import annotations

import pytest
from compgen.agent.suggest.megakernel_v3_bridge import (
    MegakernelBridgeResult,
    build_mega_contract_from_proposal,
)
from compgen.kernels.contract_v3 import (
    DispatchModel,
    DispatchSpec,
    Granularity,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    MemorySpec,
    MemoryTier,
    OrchestrationSpec,
    ShapeClass,
    TensorIO,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _envelope() -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name="cuda-a100",
        vector_lanes=64,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16", "f32"),
        peak_bandwidth_gbps=672.0,
    )


def _io(name: str) -> IOContract:
    return IOContract(
        inputs=(
            TensorIO(name=f"{name}_a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            TensorIO(name=f"{name}_b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
        ),
        outputs=(TensorIO(name=f"{name}_o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
    )


def _scratchpad_sub(op: str) -> KernelContractV3:
    """A NORMAL sub-contract that already lives in SCRATCHPAD — no
    promotion needed by the bridge."""
    return KernelContractV3(
        op_name=op,
        archetype=KernelArchetype.POINTWISE,
        io=_io(op),
        orchestration=OrchestrationSpec(
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
            ),
        ),
    )


def _dram_sub(op: str) -> KernelContractV3:
    """A NORMAL sub-contract whose tiers are DEVICE_DRAM — bridge must
    promote them to SCRATCHPAD to satisfy MEGA invariants."""
    return KernelContractV3(
        op_name=op,
        archetype=KernelArchetype.POINTWISE,
        io=_io(op),
        orchestration=OrchestrationSpec(
            memory=MemorySpec(
                input_tiers=(MemoryTier.DEVICE_DRAM, MemoryTier.DEVICE_DRAM),
                output_tiers=(MemoryTier.DEVICE_DRAM,),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_fused_region_refs_raises() -> None:
    with pytest.raises(ValueError, match="fused_region_refs"):
        build_mega_contract_from_proposal(
            {},
            contract_for_region=lambda r: _scratchpad_sub(r),
            envelope=_envelope(),
        )


def test_empty_fused_region_refs_raises() -> None:
    with pytest.raises(ValueError, match="fused_region_refs"):
        build_mega_contract_from_proposal(
            {"fused_region_refs": []},
            contract_for_region=lambda r: _scratchpad_sub(r),
            envelope=_envelope(),
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_mega_contract_with_persistent_dispatch() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1", "r2"]},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
    )
    assert isinstance(res, MegakernelBridgeResult)
    c = res.contract
    assert c.granularity is Granularity.MEGA
    assert c.orchestration.dispatch.model is DispatchModel.PERSISTENT
    assert len(c.body) == 3
    assert res.sub_region_ids == ("r0", "r1", "r2")


def test_internal_events_form_linear_chain() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1", "r2", "r3"]},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
    )
    edges = res.contract.internal_events
    # n sub-contracts → n-1 edges in linear chain
    assert len(edges) == 3
    for i, e in enumerate(edges):
        assert e.producer_idx == i
        assert e.consumer_idx == i + 1


def test_outer_op_name_uses_proposal_name_when_provided() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1"], "megakernel_name": "mega.flash_attn"},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
    )
    assert res.contract.op_name == "mega.flash_attn"


def test_outer_op_name_falls_back_to_synthesised_name() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1"]},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
    )
    assert res.contract.op_name.startswith("megakernel.")
    assert "r0" in res.contract.op_name or "_to_" in res.contract.op_name


def test_op_name_override_wins_over_proposal() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0"], "megakernel_name": "ignored"},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
        op_name="explicit.override",
    )
    assert res.contract.op_name == "explicit.override"


# ---------------------------------------------------------------------------
# Tier-promotion safety net
# ---------------------------------------------------------------------------


def test_dram_subkernels_get_promoted_to_scratchpad() -> None:
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1"]},
        contract_for_region=lambda r: _dram_sub(r),
        envelope=_envelope(),
    )
    for sub in res.contract.body:
        for t in (*sub.orchestration.memory.input_tiers, *sub.orchestration.memory.output_tiers):
            assert t in (MemoryTier.SCRATCHPAD, MemoryTier.REGISTER)


def test_scratchpad_subkernels_are_left_alone() -> None:
    """Promotion should only fire when needed; otherwise pass through."""
    sub = _scratchpad_sub("foo")
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["foo"]},
        contract_for_region=lambda r: sub,
        envelope=_envelope(),
    )
    promoted = res.contract.body[0]
    # Tiers identical — bridge didn't rewrite metadata
    assert promoted.orchestration.memory.input_tiers == sub.orchestration.memory.input_tiers


# ---------------------------------------------------------------------------
# Nested-MEGA filter
# ---------------------------------------------------------------------------


def _nested_mega_sub() -> KernelContractV3:
    inner = _scratchpad_sub("inner")
    return KernelContractV3(
        op_name="nested",
        archetype=KernelArchetype.POINTWISE,
        io=_io("nested"),
        granularity=Granularity.MEGA,
        orchestration=OrchestrationSpec(
            dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
            memory=MemorySpec(
                input_tiers=(MemoryTier.DEVICE_DRAM, MemoryTier.DEVICE_DRAM),
                output_tiers=(MemoryTier.DEVICE_DRAM,),
            ),
        ),
        body=(inner,),
    )


def test_nested_mega_subkernels_are_filtered_out_with_note() -> None:
    def factory(rid: str) -> KernelContractV3:
        if rid == "nested":
            return _nested_mega_sub()
        return _scratchpad_sub(rid)

    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "nested", "r1"]},
        contract_for_region=factory,
        envelope=_envelope(),
    )
    # nested was dropped; only 2 sub-kernels remain
    assert len(res.contract.body) == 2
    assert any("nested" in n and "MEGA" in n for n in res.notes)


def test_all_nested_mega_subkernels_raises() -> None:
    """If every region was nested-MEGA, nothing eligible remains."""

    def factory(rid: str) -> KernelContractV3:
        return _nested_mega_sub()

    with pytest.raises(ValueError, match="no eligible sub-contracts"):
        build_mega_contract_from_proposal(
            {"fused_region_refs": ["a", "b"]},
            contract_for_region=factory,
            envelope=_envelope(),
        )


# ---------------------------------------------------------------------------
# Resulting contract passes v3 invariants
# ---------------------------------------------------------------------------


def test_resulting_contract_satisfies_mega_invariants() -> None:
    """If the bridge produced an invalid contract, KernelContractV3's
    __post_init__ would have raised. So merely getting a result is the
    test — but also re-construct it explicitly to be paranoid."""
    res = build_mega_contract_from_proposal(
        {"fused_region_refs": ["r0", "r1"]},
        contract_for_region=lambda r: _scratchpad_sub(r),
        envelope=_envelope(),
    )
    c = res.contract
    # Re-instantiating with the same arguments must also pass.
    KernelContractV3(
        op_name=c.op_name,
        archetype=c.archetype,
        io=c.io,
        granularity=c.granularity,
        orchestration=c.orchestration,
        body=c.body,
        internal_events=c.internal_events,
        metadata=c.metadata,
    )
