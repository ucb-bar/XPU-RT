"""two-stage provider protocol (bid + fulfill).

Coverage:

- ``TestBidPreviewSchema`` — dataclass round-trip (incl. +inf / -inf
  sentinels in JSON).
- ``TestComputeBidFallback`` — providers without a ``bid()`` method
  return a placeholder with ``confidence=0.0``.
- ``TestComputeBidValidation`` — out-of-range fields raise typed
  ``ProviderProtocolViolation``; provider-internal exceptions degrade
  cleanly to a placeholder bid (so one buggy provider doesn't abort
  the auction).
- ``TestClaudeCodeProviderBid`` — cache hit vs miss surface the right
  confidence + time_to_generate.
- ``TestTritonTemplateProviderBid`` — matched template family →
  high-confidence bid; archetype mismatch → placeholder.
- ``TestCollectBids`` — runs over an applicable list, hashes once.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


def _build_v3_matmul_contract():
    from compgen.kernels.contract_v3 import (
        ConcurrencyUnit,
        DispatchModel,
        DispatchSpec,
        EventDecl,
        ExecutionEnvelope,
        FusionPolicy,
        Granularity,
        HardwareEnvelope,
        IOContract,
        KernelArchetype,
        KernelContractV3,
        LayoutKind,
        MemorySpec,
        NumericsSpec,
        ObservabilitySpec,
        OrchestrationSpec,
        PaddingPolicy,
        PerformancePriority,
        ShapeClass,
        StaticAttr,
        SyncSpec,
        TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="cuda_sm75",
        vector_lanes=32,
        scratchpad_bytes=49_152,
        register_bytes=64,
        native_dtypes=("f32",),
        peak_compute_per_dtype={"f32": 8.0},  # 8 TFLOPS
    )
    exe = ExecutionEnvelope(
        hardware=hw,
        memory_budget_bytes=49_152,
        concurrency_unit=ConcurrencyUnit.WARP,
        padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(
        name="A",
        shape=ShapeClass(dims=(128, 128)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    B = TensorIO(
        name="B",
        shape=ShapeClass(dims=(128, 128)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    Y = TensorIO(
        name="Y",
        shape=ShapeClass(dims=(128, 128)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    io = IOContract(
        inputs=(A, B),
        outputs=(Y,),
        attributes=(StaticAttr(name="transpose_b", value=False),),
        numerics=NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=1.0e-3,
            deterministic=True,
        ),
    )
    return KernelContractV3(
        op_name="matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=io,
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=exe,
            sync=SyncSpec(event_decls=(EventDecl(name="done"),)),
            memory=MemorySpec(),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.SYNC),
            observability=ObservabilitySpec(emit_completion_event=True),
        ),
    )


@dataclass
class _LegacyProvider:
    """A KernelProvider without a bid() method — exercises fallback."""

    _name: str = "legacy_no_bid"

    @property
    def name(self) -> str:
        return self._name

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ProviderResult

        return ProviderResult(found=False)

    def export_knowledge(self):  # noqa: D401
        return []


@dataclass
class _MalformedBidProvider:
    """Returns a BidPreview with out-of-range confidence."""

    @property
    def name(self) -> str:
        return "malformed"

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ProviderResult

        return ProviderResult(found=False)

    def export_knowledge(self):  # noqa: D401
        return []

    def bid(self, contract_v3):  # noqa: ANN001
        from compgen.kernels.provider import BidPreview

        return BidPreview(
            provider_name=self.name,
            confidence=2.0,  # OUT OF RANGE
            perf_estimate_us=1.0,
        )


@dataclass
class _RaisesBidProvider:
    """Provider whose bid() raises a non-typed error — must be absorbed."""

    @property
    def name(self) -> str:
        return "raises"

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ProviderResult

        return ProviderResult(found=False)

    def export_knowledge(self):  # noqa: D401
        return []

    def bid(self, contract_v3):  # noqa: ANN001
        raise RuntimeError("internal provider bug")


# --------------------------------------------------------------------------- #
# BidPreview schema
# --------------------------------------------------------------------------- #


class TestBidPreviewSchema:
    def test_round_trip(self) -> None:
        from compgen.kernels.provider import BidPreview

        bid = BidPreview(
            provider_name="x",
            contract_hash="abc",
            perf_estimate_us=42.5,
            confidence=0.7,
            time_to_generate_s_estimate=120.0,
            registers_used=64,
            occupancy=0.5,
            smem_bytes=1024,
            rationale="test",
            cache_hit=True,
        )
        body = json.loads(json.dumps(bid.to_dict()))
        round_tripped = BidPreview.from_dict(body)
        assert round_tripped == bid

    def test_inf_serialization(self) -> None:
        from compgen.kernels.provider import BidPreview

        bid = BidPreview(provider_name="x", perf_estimate_us=float("inf"))
        body = json.loads(json.dumps(bid.to_dict()))
        round_tripped = BidPreview.from_dict(body)
        assert math.isinf(round_tripped.perf_estimate_us)
        assert round_tripped.perf_estimate_us > 0


# --------------------------------------------------------------------------- #
# compute_bid fallback for legacy providers
# --------------------------------------------------------------------------- #


class TestComputeBidFallback:
    def test_no_bid_method_returns_placeholder(self) -> None:
        from compgen.kernels.registry import compute_bid

        bid = compute_bid(_LegacyProvider(), _build_v3_matmul_contract())
        assert bid.confidence == 0.0
        assert bid.rationale == "no_bid_method"
        assert bid.provider_name == "legacy_no_bid"
        assert bid.contract_hash != ""  # canonical hash stamped

    def test_provider_returning_none_is_treated_as_no_bid(self) -> None:
        from compgen.kernels.registry import compute_bid

        @dataclass
        class _NoneBid(_LegacyProvider):
            _name: str = "none_returner"

            def bid(self, contract_v3):  # noqa: ANN001
                return None

        bid = compute_bid(_NoneBid(), _build_v3_matmul_contract())
        assert bid.confidence == 0.0
        assert bid.rationale == "bid_returned_none"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestComputeBidValidation:
    def test_out_of_range_confidence_raises_typed(self) -> None:
        from compgen.kernels.provider import ProviderProtocolViolation
        from compgen.kernels.registry import compute_bid

        with pytest.raises(ProviderProtocolViolation, match="confidence"):
            compute_bid(_MalformedBidProvider(), _build_v3_matmul_contract())

    def test_internal_exception_degrades_to_placeholder(self) -> None:
        from compgen.kernels.registry import compute_bid

        bid = compute_bid(_RaisesBidProvider(), _build_v3_matmul_contract())
        assert bid.confidence == 0.0
        assert bid.rationale.startswith("bid_raised:RuntimeError")

    def test_negative_perf_estimate_rejected(self) -> None:
        from compgen.kernels.provider import (
            BidPreview,
            ProviderProtocolViolation,
        )
        from compgen.kernels.registry import compute_bid

        @dataclass
        class _Neg(_LegacyProvider):
            _name: str = "neg"

            def bid(self, contract_v3):  # noqa: ANN001
                return BidPreview(provider_name=self._name, perf_estimate_us=-1.0)

        with pytest.raises(ProviderProtocolViolation, match="non-negative"):
            compute_bid(_Neg(), _build_v3_matmul_contract())

    def test_contract_hash_mismatch_rejected(self) -> None:
        from compgen.kernels.provider import (
            BidPreview,
            ProviderProtocolViolation,
        )
        from compgen.kernels.registry import compute_bid

        @dataclass
        class _Lying(_LegacyProvider):
            _name: str = "lying"

            def bid(self, contract_v3):  # noqa: ANN001
                return BidPreview(
                    provider_name=self._name,
                    contract_hash="not_the_real_hash",
                    perf_estimate_us=1.0,
                    confidence=0.5,
                )

        with pytest.raises(ProviderProtocolViolation, match="contract_hash mismatch"):
            compute_bid(_Lying(), _build_v3_matmul_contract())


# --------------------------------------------------------------------------- #
# ClaudeCodeKernelProvider bid
# --------------------------------------------------------------------------- #


class TestClaudeCodeProviderBid:
    def test_stub_codegen_emits_high_confidence_cache_hit_proxy(self) -> None:
        from compgen.kernels.providers.claude_code_default import (
            ClaudeCodeKernelProvider,
            StubCodegen,
        )

        p = ClaudeCodeKernelProvider(codegen=StubCodegen())
        bid = p.bid(_build_v3_matmul_contract())
        assert bid.confidence == 0.9
        assert bid.cache_hit is True
        assert bid.time_to_generate_s_estimate == pytest.approx(1.0)

    def test_unknown_codegen_emits_low_confidence_miss(self) -> None:
        from compgen.kernels.providers.claude_code_default import (
            ClaudeCodeKernelProvider,
            CodegenCallable,
        )

        class _Unknown(CodegenCallable):
            def __call__(self, prompt, contract):  # noqa: ANN001, D401
                return "// unused"

        p = ClaudeCodeKernelProvider(codegen=_Unknown())
        bid = p.bid(_build_v3_matmul_contract())
        assert bid.confidence == 0.3
        assert bid.cache_hit is False
        assert bid.time_to_generate_s_estimate >= 60.0


# --------------------------------------------------------------------------- #
# TritonTemplateProvider bid
# --------------------------------------------------------------------------- #


class TestTritonTemplateProviderBid:
    def test_matmul_archetype_high_confidence(self) -> None:
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        p = TritonTemplateProvider()
        bid = p.bid(_build_v3_matmul_contract())
        assert bid.confidence == 0.7
        assert bid.rationale.startswith("template_match_matmul")
        assert bid.perf_estimate_us > 0
        assert bid.time_to_generate_s_estimate < 1.0

    def test_unknown_archetype_zero_confidence(self) -> None:
        from compgen.kernels.contract_v3 import (
            ConcurrencyUnit,
            DispatchModel,
            DispatchSpec,
            EventDecl,
            ExecutionEnvelope,
            FusionPolicy,
            Granularity,
            HardwareEnvelope,
            IOContract,
            KernelArchetype,
            KernelContractV3,
            LayoutKind,
            MemorySpec,
            NumericsSpec,
            ObservabilitySpec,
            OrchestrationSpec,
            PaddingPolicy,
            PerformancePriority,
            ShapeClass,
            SyncSpec,
            TensorIO,
        )
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        # POINTWISE archetype with op_name=copy — POINTWISE itself is
        # a valid archetype but no template family matches "copy".
        hw = HardwareEnvelope(
            target_name="cuda_sm75",
            vector_lanes=32,
            scratchpad_bytes=49_152,
            register_bytes=64,
            native_dtypes=("f32",),
        )
        exe = ExecutionEnvelope(
            hardware=hw,
            memory_budget_bytes=4096,
            concurrency_unit=ConcurrencyUnit.WARP,
            padding=PaddingPolicy.NONE,
            priority=PerformancePriority.LATENCY,
        )
        A = TensorIO(
            name="A",
            shape=ShapeClass(dims=(128,)),
            dtype_class=("f32",),
            layout=LayoutKind.ROW_MAJOR,
            alignment_bytes=64,
        )
        Y = TensorIO(
            name="Y",
            shape=ShapeClass(dims=(128,)),
            dtype_class=("f32",),
            layout=LayoutKind.ROW_MAJOR,
            alignment_bytes=64,
        )
        io = IOContract(
            inputs=(A,),
            outputs=(Y,),
            attributes=(),
            numerics=NumericsSpec(
                accumulator_dtype="f32",
                fast_math=False,
                max_relative_error=0.0,
                deterministic=True,
            ),
        )
        contract = KernelContractV3(
            op_name="copy",
            archetype=KernelArchetype.POINTWISE,
            io=io,
            granularity=Granularity.NORMAL,
            orchestration=OrchestrationSpec(
                execution=exe,
                sync=SyncSpec(event_decls=(EventDecl(name="d"),)),
                memory=MemorySpec(),
                fusion=FusionPolicy(is_boundary=True),
                dispatch=DispatchSpec(model=DispatchModel.SYNC),
                observability=ObservabilitySpec(emit_completion_event=True),
            ),
        )

        p = TritonTemplateProvider()
        bid = p.bid(contract)
        assert bid.confidence == 0.0
        assert "no_template_for_archetype" in bid.rationale


# --------------------------------------------------------------------------- #
# collect_bids
# --------------------------------------------------------------------------- #


class TestCollectBids:
    def test_runs_over_applicable_list(self) -> None:
        from compgen.kernels.providers.claude_code_default import (
            ClaudeCodeKernelProvider,
            StubCodegen,
        )
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider
        from compgen.kernels.registry import collect_bids

        contract = _build_v3_matmul_contract()
        bids = collect_bids(
            [
                _LegacyProvider(),
                ClaudeCodeKernelProvider(codegen=StubCodegen()),
                TritonTemplateProvider(),
            ],
            contract,
        )
        assert len(bids) == 3
        # Stable order with the input list.
        assert bids[0].provider_name == "legacy_no_bid"
        assert bids[1].provider_name == "claude_code_default"
        assert bids[2].provider_name == "triton_templates"
        # All three carry the same canonical contract_hash.
        hashes = {b.contract_hash for b in bids}
        assert len(hashes) == 1, f"all bids should share canonical hash; got {hashes}"
        assert bids[0].contract_hash != ""
