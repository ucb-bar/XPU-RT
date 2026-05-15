"""ProviderCard / ProviderProbeResult schema tests."""

from __future__ import annotations

import json

import pytest

from compgen.providers.provider_types import (
    BLOCKED_REASONS,
    INTEGRATION_LEVELS,
    PAPER_CLAIMABLE_LEVELS,
    PROBE_STATUSES,
    ProviderCard,
    ProviderCardError,
    ProviderProbeError,
    ProviderProbeResult,
)


def _minimal_card_body(**overrides):
    body = {
        "schema_version": "provider_card_v1",
        "provider_id": "cffi_c",
        "integration_level": "promote",
        "target_families": ["host_cpu"],
        "contract_kinds": ["matmul"],
        "emits": ["c_source"],
        "entrypoint": "compgen.providers.adapters.cffi_c:CffiCProvider",
        "paper_claimable": True,
    }
    body.update(overrides)
    return body


def test_provider_card_round_trips_through_json():
    card = ProviderCard.from_dict(_minimal_card_body())
    restored = ProviderCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored == card


def test_provider_card_missing_required_field_raises():
    body = _minimal_card_body()
    body.pop("integration_level")
    with pytest.raises(ProviderCardError, match="integration_level"):
        ProviderCard.from_dict(body)


@pytest.mark.parametrize("level", ["card_only", "probe", "generate", "verify", "promote"])
def test_provider_card_accepts_every_documented_level(level: str):
    card = ProviderCard.from_dict(
        _minimal_card_body(integration_level=level, paper_claimable=False)
    )
    assert card.integration_level == level


def test_provider_card_rejects_unknown_integration_level():
    with pytest.raises(ProviderCardError, match="integration_level"):
        ProviderCard.from_dict(_minimal_card_body(integration_level="totally_made_up"))


@pytest.mark.parametrize("level", ["card_only", "probe", "generate"])
def test_paper_claimable_rejected_at_pre_verify_levels(level: str):
    with pytest.raises(ProviderCardError, match="paper_claimable"):
        ProviderCard.from_dict(
            _minimal_card_body(integration_level=level, paper_claimable=True)
        )


@pytest.mark.parametrize("level", ["verify", "promote"])
def test_paper_claimable_accepted_at_verify_or_promote(level: str):
    card = ProviderCard.from_dict(
        _minimal_card_body(integration_level=level, paper_claimable=True)
    )
    assert card.paper_claimable is True
    assert card.integration_level in PAPER_CLAIMABLE_LEVELS


def test_integration_levels_constants_match_documented_set():
    assert INTEGRATION_LEVELS == (
        "card_only", "probe", "generate", "verify", "promote",
    )
    assert PAPER_CLAIMABLE_LEVELS == frozenset({"verify", "promote"})


def test_probe_result_available_round_trip():
    r = ProviderProbeResult(
        schema_version="provider_status_v1",
        provider_id="triton",
        status="available",
        version="3.0.0",
        supports=("triton_ttir",),
    )
    body = r.to_dict()
    assert body["blocked_reason"] is None
    assert ProviderProbeResult.from_dict(body) == r


@pytest.mark.parametrize("status", ["blocked", "unsupported", "probe_error", "not_installed"])
def test_probe_result_non_available_requires_typed_reason(status: str):
    with pytest.raises(ProviderProbeError, match="blocked_reason"):
        ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="x",
            status=status,
        )


def test_probe_result_untyped_reason_rejected():
    with pytest.raises(ProviderProbeError, match="blocked_reason"):
        ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="x",
            status="blocked",
            blocked_reason="hand_wave",
        )


def test_probe_result_unknown_status_rejected():
    with pytest.raises(ProviderProbeError, match="status"):
        ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="x",
            status="totally_made_up",
        )


def test_probe_result_available_must_not_carry_reason():
    with pytest.raises(ProviderProbeError, match="available"):
        ProviderProbeResult(
            schema_version="provider_status_v1",
            provider_id="x",
            status="available",
            blocked_reason="env_missing",
        )


def test_probe_statuses_and_blocked_reasons_are_typed_enums():
    assert "available" in PROBE_STATUSES
    assert "blocked" in PROBE_STATUSES
    assert "env_missing" in BLOCKED_REASONS
    assert "license_missing" in BLOCKED_REASONS
    # No "hand_wave" or other free-text statuses.
    assert "hand_wave" not in BLOCKED_REASONS
