"""DialectProviderCard schema tests."""

from __future__ import annotations

import pytest

from compgen.dialects import DialectProviderCard, DialectProviderCardError


def _minimal_body(**overrides):
    body = {
        "schema_version": "dialect_provider_card_v1",
        "dialect_provider_id": "cuda_tile_ir",
        "dialect_name": "cuda_tile",
        "integration_level": "probe",
        "consumes": ["kernel_contract_v3", "compgen_tile_ir"],
        "emits": ["cuda_tile_mlir", "cubin"],
        "entrypoint": "compgen.providers.adapters.cuda_tile_ir:CudaTileDialectProvider",
        "required_env": ["CUDA_TILE_ROOT"],
    }
    body.update(overrides)
    return body


def test_dialect_provider_card_round_trips():
    card = DialectProviderCard.from_dict(_minimal_body())
    assert card.dialect_name == "cuda_tile"
    restored = DialectProviderCard.from_dict(card.to_dict())
    assert restored == card


def test_dialect_provider_card_unknown_integration_level_rejected():
    with pytest.raises(DialectProviderCardError, match="integration_level"):
        DialectProviderCard.from_dict(_minimal_body(integration_level="totally_made_up"))


def test_dialect_provider_card_paper_claimable_at_probe_rejected():
    body = _minimal_body(integration_level="probe", paper_claimable=True)
    with pytest.raises(DialectProviderCardError, match="paper_claimable"):
        DialectProviderCard.from_dict(body)


def test_dialect_provider_card_paper_claimable_at_verify_accepted():
    card = DialectProviderCard.from_dict(
        _minimal_body(integration_level="verify", paper_claimable=True)
    )
    assert card.paper_claimable is True


def test_dialect_provider_card_missing_required_field_rejected():
    body = _minimal_body()
    body.pop("dialect_name")
    with pytest.raises(DialectProviderCardError, match="dialect_name"):
        DialectProviderCard.from_dict(body)
