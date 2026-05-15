"""Tests for ``kernels.providers.triton_templates``.

Covers the surface that's exercised *without* GPU hardware (structural
provider behaviour, Phase-2 cost-source metadata). The GPU-gated
compile + run path lives in :mod:`tests.kernels.providers.test_triton_gpu`
if present; the real-timing contract is already covered indirectly
via :mod:`tests.kernels.test_measure` on CUDA.

Phase-11 goal: exercise the 694-LOC file so regressions land with
coverage instead of silently shipping.
"""

from __future__ import annotations

import math

import pytest


def _make_contract(op_family: str = "matmul"):
    """Minimal KernelContract for a matmul-like op."""
    from compgen.kernels.provider import KernelContract

    return KernelContract(
        region_id="r0",
        op_family=op_family,
        input_shapes=((64, 64), (64, 64)),
        output_shapes=((64, 64),),
        dtypes=("f32", "f32"),
        target_name="test-gpu-simt",
    )


class TestTritonTemplateProviderSurface:
    def test_import_public_symbols(self) -> None:
        from compgen.kernels.providers.triton_templates import (
            TritonTemplateProvider,
            triton_available,
        )

        assert callable(triton_available)
        assert TritonTemplateProvider is not None

    def test_provider_has_name(self) -> None:
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        p = TritonTemplateProvider()
        assert p.name == "triton_templates"

    def test_accepts_matmul_family(self) -> None:
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        p = TritonTemplateProvider()
        c = _make_contract("matmul")
        assert p.accepts_contract(c)

    def test_rejects_unknown_family(self) -> None:
        """The provider only claims op families it has templates for —
        everything else gets passed to the next provider in the chain."""
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        p = TritonTemplateProvider()
        c = _make_contract("conv3d_transpose")  # not a provided template
        assert not p.accepts_contract(c)


class TestCostSourceMetadata:
    """Phase-2 introduced a ``cost_source`` field in the provider result
    metadata so callers never confuse "unmeasured" with "0.0 latency".
    These tests lock the convention in."""

    def test_unmeasured_flow_tags_cost_source(self) -> None:
        """On a CPU-only runner, Triton validation skips and the
        provider must tag the result ``cost_source="unmeasured"``
        rather than claiming ``latency_us=0.0``."""
        import torch
        from compgen.kernels.provider import SearchBudget
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        p = TritonTemplateProvider()
        c = _make_contract("matmul")
        result = p.search(c, SearchBudget(max_iterations=1))
        if torch.cuda.is_available():
            # Measured path: either a real latency or explicit unmeasured,
            # never silent 0.0.
            assert result.metadata.get("cost_source") in {"measured_gpu", "unmeasured"}
        else:
            assert result.metadata.get("cost_source") == "unmeasured"
            assert math.isnan(result.latency_us) or result.latency_us == 0.0 or math.isinf(result.latency_us), (
                "latency_us on unmeasured path must be NaN (honest) or 0.0 "
                "(compatibility with old sentinel) — never a fake positive"
            )


@pytest.mark.requires_gpu
class TestTritonRealCompile:
    """Exercised when CUDA + Triton are both available. Confirms the
    provider produces a measured-latency kernel and tags cost_source
    as ``measured_gpu``."""

    def test_matmul_template_measured_on_gpu(self) -> None:
        from compgen.kernels.provider import SearchBudget
        from compgen.kernels.providers.triton_templates import (
            TritonTemplateProvider,
            triton_available,
        )

        if not triton_available():
            pytest.skip("Triton not importable")
        p = TritonTemplateProvider()
        c = _make_contract("matmul")
        result = p.search(c, SearchBudget(max_iterations=1))
        assert result.found
        # Phase-2 contract: measured GPU path yields finite positive
        # latency + measured_gpu cost_source.
        assert result.metadata.get("cost_source") == "measured_gpu"
        assert math.isfinite(result.latency_us)
        assert result.latency_us > 0.0
