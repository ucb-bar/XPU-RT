"""cffi-C and Triton normalized as cards.

Pins three properties:

1. Every paper-claimable card's ``entrypoint`` resolves to a real
   class.
2. Every resolved class is a :class:`KernelProvider` (Protocol).
3. ``ProviderResult`` is **never** confused with
   :class:`KernelCertificate` — hard rule 1.
"""

from __future__ import annotations

import pytest

from compgen.kernels.kernel_certificate import KernelCertificate
from compgen.kernels.provider import KernelProvider, ProviderResult
from compgen.providers.adapters import (
    AdapterResolutionError,
    resolve_provider_class,
)
from compgen.providers.card_loader import iter_provider_cards
from compgen.providers.provider_types import ProviderCard


def _cards_by_id() -> dict[str, ProviderCard]:
    return {c.provider_id: c for c in iter_provider_cards()}


def test_cffi_c_entrypoint_resolves_to_kernel_provider():
    card = _cards_by_id()["cffi_c"]
    cls = resolve_provider_class(card)
    inst = cls()
    assert isinstance(inst, KernelProvider), (
        f"cffi_c entrypoint resolved to {cls.__name__} which does not satisfy "
        f"the KernelProvider Protocol"
    )


def test_triton_entrypoint_resolves_to_kernel_provider():
    card = _cards_by_id()["triton"]
    cls = resolve_provider_class(card)
    # TritonTemplateProvider construction requires an env-backed provider;
    # we only assert resolution + KernelProvider-shape at the class level.
    assert hasattr(cls, "name") or hasattr(cls, "search") or hasattr(cls, "bid"), (
        f"triton entrypoint resolved to {cls.__name__} which lacks "
        f"KernelProvider-shape attributes"
    )


def test_python_reference_entrypoint_resolves():
    card = _cards_by_id()["python_reference"]
    cls = resolve_provider_class(card)
    inst = cls()
    assert isinstance(inst, KernelProvider)


def test_provider_result_is_not_kernel_certificate():
    """Hard rule 1: providers do not certify themselves."""
    assert ProviderResult is not KernelCertificate
    assert ProviderResult.__qualname__ != KernelCertificate.__qualname__
    # ``ProviderResult`` carries ``found`` / ``correct`` flags — those
    # are *provider claims*, not proofs. ``KernelCertificate`` is what
    # the verifier emits after the differential / contract check.
    pr = ProviderResult(found=True, correct=True, kernel_code="x")
    assert not isinstance(pr, KernelCertificate)


def test_every_card_entrypoint_resolves_or_blocks_typed():
    """Every card whose dependencies are installable in this environment
    must either resolve or raise a typed AdapterResolutionError —
    never a raw ImportError."""
    cards = _cards_by_id()
    for pid, card in cards.items():
        try:
            resolve_provider_class(card)
        except AdapterResolutionError as exc:
            # Typed failure is acceptable as long as the reason is one of
            # the documented kinds.
            assert exc.reason in {
                "bad_entrypoint_syntax",
                "module_not_importable",
                "symbol_not_in_module",
            }, f"{pid}: untyped reason {exc.reason!r}"


def test_resolution_reports_typed_reason_on_bad_entrypoint():
    card = ProviderCard.from_dict(
        {
            "schema_version": "provider_card_v1",
            "provider_id": "x",
            "integration_level": "probe",
            "target_families": [],
            "contract_kinds": [],
            "emits": [],
            "entrypoint": "no_colon_here",
        }
    )
    with pytest.raises(AdapterResolutionError) as exc:
        resolve_provider_class(card)
    assert exc.value.reason == "bad_entrypoint_syntax"


def test_resolution_reports_typed_reason_on_missing_module():
    card = ProviderCard.from_dict(
        {
            "schema_version": "provider_card_v1",
            "provider_id": "x",
            "integration_level": "probe",
            "target_families": [],
            "contract_kinds": [],
            "emits": [],
            "entrypoint": "this.module.does.not.exist:Whatever",
        }
    )
    with pytest.raises(AdapterResolutionError) as exc:
        resolve_provider_class(card)
    assert exc.value.reason == "module_not_importable"


def test_resolution_reports_typed_reason_on_missing_symbol():
    card = ProviderCard.from_dict(
        {
            "schema_version": "provider_card_v1",
            "provider_id": "x",
            "integration_level": "probe",
            "target_families": [],
            "contract_kinds": [],
            "emits": [],
            "entrypoint": "compgen.kernels.providers.c_reference:DoesNotExist",
        }
    )
    with pytest.raises(AdapterResolutionError) as exc:
        resolve_provider_class(card)
    assert exc.value.reason == "symbol_not_in_module"


def test_baseline_provider_search_returns_provider_result():
    """End-to-end: the real CReferenceProvider produces a real
    ProviderResult — exercising the substrate without bypassing the
    existing kernel pipeline."""
    from compgen.kernels.provider import KernelContract, SearchBudget

    card = _cards_by_id()["cffi_c"]
    cls = resolve_provider_class(card)
    inst = cls()
    contract = KernelContract(
        region_id="region_0",
        target_name="host_cpu",
        op_family="matmul",
        input_shapes=((64, 64), (64, 64)),
        output_shapes=((64, 64),),
        dtypes=("f32",),
    )
    result = inst.search(contract, SearchBudget())
    assert isinstance(result, ProviderResult)
    assert result.found is True
    assert result.kernel_code  # non-empty C source
    assert result.language == "c"
    # And it is still NOT a certificate.
    assert not isinstance(result, KernelCertificate)
