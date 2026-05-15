"""KernelProvider 3-method ABC + legacy shim tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.kernels.provider import BidPreview, KernelContract, SearchBudget
from compgen.providers.card_loader import iter_provider_cards
from compgen.providers.adapters import resolve_provider_class
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.legacy_shim import (
    LegacyProviderAdapter,
    wrap_legacy,
)
from compgen.providers.provider_types import (
    PROBE_STATUSES,
    ProviderProbeResult,
)
from compgen.providers.result_v1 import ProviderResultV1


def _cards():
    return {c.provider_id: c for c in iter_provider_cards()}


# ---------------------------------------------------------------------------
# ABC discipline
# ---------------------------------------------------------------------------


def test_kernel_provider_is_abstract():
    """Instantiating the bare ABC must fail."""
    with pytest.raises(TypeError):
        KernelProvider()  # type: ignore[abstract]


def test_kernel_codegen_request_carries_fields():
    req = KernelCodegenRequest(
        task_id="kcodegen_0001",
        contract=None,
        target=None,
        artifact_dir="/tmp/x",
    )
    assert req.task_id == "kcodegen_0001"
    assert req.extras == {}


# ---------------------------------------------------------------------------
# Legacy shim wires the three required methods
# ---------------------------------------------------------------------------


class _Target:
    name = "host_cpu"


def _build_cffi_c_shim() -> LegacyProviderAdapter:
    card = _cards()["cffi_c"]
    cls = resolve_provider_class(card)
    return wrap_legacy(card, cls())


def test_shim_is_a_kernel_provider():
    shim = _build_cffi_c_shim()
    assert isinstance(shim, KernelProvider)
    assert shim.provider_id == "cffi_c"


def test_shim_probe_returns_typed_result():
    shim = _build_cffi_c_shim()
    probe = shim.probe()
    assert isinstance(probe, ProviderProbeResult)
    assert probe.status in PROBE_STATUSES


def test_shim_can_bid_returns_bid_preview():
    shim = _build_cffi_c_shim()
    bid = shim.can_bid(contract=None, target=None)
    assert isinstance(bid, BidPreview)


def test_shim_propose_emits_v1_result(tmp_path: Path):
    shim = _build_cffi_c_shim()
    contract = KernelContract(
        region_id="r0",
        op_family="matmul",
        input_shapes=((64, 64), (64, 64)),
        output_shapes=((64, 64),),
        dtypes=("f32",),
        target_name="host_cpu",
    )
    req = KernelCodegenRequest(
        task_id="kcodegen_test",
        contract=contract,
        target=_Target(),
        artifact_dir=str(tmp_path),
        extras={"budget": SearchBudget(), "contract_hash": "deadbeef"},
    )
    result = shim.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.schema_version == "provider_result_v1"
    assert result.status == "generated"
    assert result.task_id == "kcodegen_test"
    assert result.provider_id == "cffi_c"
    assert result.contract_hash == "deadbeef"
    # Kernel source materialized to disk.
    source_path = Path(result.artifacts["source"])
    assert source_path.is_file()
    assert source_path.read_text().strip()  # non-empty


def test_shim_handles_contract_rejection(tmp_path: Path):
    """cffi-C rejects attention contracts → status=contract_rejected."""

    shim = _build_cffi_c_shim()
    contract = KernelContract(
        region_id="r0",
        op_family="flash_attention",
        input_shapes=((1, 1, 128, 64),) * 3,
        output_shapes=((1, 1, 128, 64),),
        dtypes=("f16",),
        target_name="host_cpu",
    )
    req = KernelCodegenRequest(
        task_id="kcodegen_test",
        contract=contract,
        target=_Target(),
        artifact_dir=str(tmp_path),
        extras={"budget": SearchBudget()},
    )
    result = shim.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "contract_rejected"
    assert result.detail


def test_shim_wraps_every_existing_real_provider():
    """All 5 legacy real-class providers (cffi_c, python_reference, triton,
    autocomp, kernelblaster) wrap into the shim without error."""

    cards = _cards()
    for pid in ("cffi_c", "python_reference", "triton", "autocomp", "kernelblaster"):
        cls = resolve_provider_class(cards[pid])
        shim = wrap_legacy(cards[pid], cls())
        assert isinstance(shim, KernelProvider)
        probe = shim.probe()
        assert isinstance(probe, ProviderProbeResult)
        assert probe.status in PROBE_STATUSES


def test_shim_propose_never_raises_on_known_legacy_providers(tmp_path: Path):
    """Even on contracts the legacy provider doesn't understand, the shim
    returns a typed result rather than raising. Hard rule:
    `propose()` returns; it does not raise."""

    cards = _cards()
    # python_reference is the safest non-network-bound provider; it shares
    # CReferenceProvider semantics.
    cls = resolve_provider_class(cards["python_reference"])
    shim = wrap_legacy(cards["python_reference"], cls())
    contract = KernelContract(
        region_id="r0",
        op_family="conv",  # CReferenceProvider only accepts matmul/pointwise
        input_shapes=((1, 3, 32, 32),),
        output_shapes=((1, 16, 30, 30),),
        dtypes=("f32",),
        target_name="host_cpu",
    )
    req = KernelCodegenRequest(
        task_id="kcodegen_test",
        contract=contract,
        target=_Target(),
        artifact_dir=str(tmp_path),
        extras={"budget": SearchBudget()},
    )
    result = shim.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status in ("contract_rejected", "blocked", "generated", "error")


# ---------------------------------------------------------------------------
# Hard rule 1: ProviderResultV1 status="generated" is NOT a certificate
# ---------------------------------------------------------------------------


def test_generated_status_is_a_claim_not_a_certificate():
    """ProviderResultV1 with status=generated only claims artifacts exist —
    it does NOT certify correctness. The verifier emits the
    KernelCertificate downstream."""

    from compgen.kernels.kernel_certificate import KernelCertificate

    r = ProviderResultV1(
        schema_version="provider_result_v1",
        task_id="x",
        provider_id="cffi_c",
        target_id="host_cpu",
        contract_hash="abc",
        status="generated",
        artifacts={"source": "/tmp/x.c"},
    )
    assert not isinstance(r, KernelCertificate)
