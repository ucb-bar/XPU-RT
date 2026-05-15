"""TargetCard schema tests."""

from __future__ import annotations

import pytest

from compgen.targets.target_types import (
    DISPATCH_MODES,
    MEMORY_TIER_KINDS,
    TargetCard,
    TargetCardError,
)


def _minimal_body(**overrides):
    body = {
        "schema_version": "target_card_v1",
        "target_id": "cuda_sm90",
        "family": "cuda",
        "vendor": "nvidia",
        "dispatch_modes": ["sync", "async", "static_plan"],
        "memory_tiers": [
            {"name": "global", "kind": "global", "capacity_bytes": 80_000_000_000},
            {"name": "shared", "kind": "shared", "capacity_bytes": 228 * 1024},
            {"name": "regs", "kind": "registers"},
        ],
    }
    body.update(overrides)
    return body


def test_target_card_round_trips():
    card = TargetCard.from_dict(_minimal_body())
    restored = TargetCard.from_dict(card.to_dict())
    assert restored == card


def test_target_card_empty_dispatch_modes_rejected():
    with pytest.raises(TargetCardError, match="dispatch_modes"):
        TargetCard.from_dict(_minimal_body(dispatch_modes=[]))


def test_target_card_untyped_dispatch_mode_rejected():
    with pytest.raises(TargetCardError, match="dispatch_mode"):
        TargetCard.from_dict(_minimal_body(dispatch_modes=["sync", "wave_hands"]))


def test_target_card_untyped_memory_tier_kind_rejected():
    body = _minimal_body()
    body["memory_tiers"] = [{"name": "dram", "kind": "totally_made_up"}]
    with pytest.raises(TargetCardError, match="kind"):
        TargetCard.from_dict(body)


def test_target_card_missing_required_field_rejected():
    body = _minimal_body()
    body.pop("vendor")
    with pytest.raises(TargetCardError, match="vendor"):
        TargetCard.from_dict(body)


def test_dispatch_modes_and_memory_tier_kinds_are_typed_enums():
    assert "sync" in DISPATCH_MODES
    assert "async" in DISPATCH_MODES
    assert "megakernel" in DISPATCH_MODES
    assert "global" in MEMORY_TIER_KINDS
    assert "explicit" in MEMORY_TIER_KINDS
