"""deterministic provider routing tests."""

from __future__ import annotations

import pytest

from compgen.providers.provider_routing import (
    INTEGRATION_LEVEL_RANK,
    KIND_PREFERENCE,
    route_for,
    route_matrix,
    supported_kinds,
    supported_target_families,
)
from compgen.providers.provider_types import ProviderCard


def _card(provider_id: str, **overrides):
    body = {
        "schema_version": "provider_card_v1",
        "provider_id": provider_id,
        "integration_level": "probe",
        "target_families": ["cuda"],
        "contract_kinds": ["matmul"],
        "emits": ["c_source"],
        "entrypoint": f"x:{provider_id}",
    }
    body.update(overrides)
    return ProviderCard.from_dict(body)


def test_route_is_deterministic():
    """Same inputs → identical output across calls."""

    a = route_for(contract_kind="matmul", target_family="cuda")
    b = route_for(contract_kind="matmul", target_family="cuda")
    assert a == b


def test_route_filters_by_contract_kind():
    cards = (
        _card("p1", contract_kinds=["matmul"]),
        _card("p2", contract_kinds=["attention"]),
        _card("p3", contract_kinds=["matmul", "attention"]),
    )
    assert route_for(contract_kind="matmul", target_family="cuda", cards=cards) == (
        "p1",
        "p3",
    )
    assert route_for(contract_kind="attention", target_family="cuda", cards=cards) == (
        "p2",
        "p3",
    )


def test_route_filters_by_target_family():
    cards = (
        _card("p1", target_families=["cuda"]),
        _card("p2", target_families=["host_cpu"]),
        _card("p3", target_families=["cuda", "host_cpu"]),
    )
    assert route_for(contract_kind="matmul", target_family="host_cpu", cards=cards) == (
        "p2",
        "p3",
    )


def test_route_orders_by_integration_level():
    """promote > verify > generate > probe > card_only."""

    cards = (
        _card("low", integration_level="probe"),
        _card("mid", integration_level="generate"),
        _card("high", integration_level="promote"),
    )
    assert route_for(contract_kind="matmul", target_family="cuda", cards=cards) == (
        "high",
        "mid",
        "low",
    )


def test_route_honors_kind_preference_within_same_level():
    """When two providers tie on integration_level, the per-kind
    preference list breaks the tie."""

    # Both at promote; matmul preference puts cffi_c before triton.
    cards = (
        _card("triton", integration_level="promote"),
        _card("cffi_c", integration_level="promote"),
    )
    assert route_for(contract_kind="matmul", target_family="cuda", cards=cards) == (
        "cffi_c",
        "triton",
    )


def test_route_alphabetical_final_tiebreaker():
    """When integration_level AND kind preference tie, sort by id."""

    # Neither in the matmul preference list; both at probe.
    cards = (
        _card("z_provider", integration_level="probe"),
        _card("a_provider", integration_level="probe"),
    )
    assert route_for(contract_kind="matmul", target_family="cuda", cards=cards) == (
        "a_provider",
        "z_provider",
    )


def test_route_for_unsupported_kind_returns_empty():
    assert route_for(
        contract_kind="totally_made_up_kind", target_family="cuda"
    ) == ()


def test_route_for_unsupported_family_returns_empty():
    assert route_for(
        contract_kind="matmul", target_family="totally_made_up_family"
    ) == ()


def test_supported_kinds_includes_shipped_kinds():
    kinds = supported_kinds()
    assert "matmul" in kinds
    assert "pointwise" in kinds
    assert "attention" in kinds


def test_supported_target_families_includes_shipped():
    families = supported_target_families()
    assert "cuda" in families
    assert "host_cpu" in families


def test_route_matrix_is_non_empty():
    matrix = route_matrix()
    assert ("matmul", "cuda") in matrix
    assert ("matmul", "host_cpu") in matrix


def test_integration_level_rank_enum():
    assert INTEGRATION_LEVEL_RANK["promote"] < INTEGRATION_LEVEL_RANK["verify"]
    assert INTEGRATION_LEVEL_RANK["verify"] < INTEGRATION_LEVEL_RANK["generate"]
    assert INTEGRATION_LEVEL_RANK["generate"] < INTEGRATION_LEVEL_RANK["probe"]
    assert INTEGRATION_LEVEL_RANK["probe"] < INTEGRATION_LEVEL_RANK["card_only"]


def test_kind_preference_table_present():
    assert "matmul" in KIND_PREFERENCE
    assert "attention" in KIND_PREFERENCE
    assert "fused_region" in KIND_PREFERENCE


def test_real_routing_matmul_host_cpu_starts_with_cffi_c():
    """On the shipped cards, matmul × host_cpu must put cffi_c first
    (it's the only card at promote-level for host_cpu)."""

    ids = route_for(contract_kind="matmul", target_family="host_cpu")
    assert ids[0] == "cffi_c"


def test_real_routing_attention_cuda_starts_with_triton():
    """On the shipped cards, attention × cuda must put triton first
    (it's the only card at promote-level for cuda + attention)."""

    ids = route_for(contract_kind="attention", target_family="cuda")
    assert ids[0] == "triton"
