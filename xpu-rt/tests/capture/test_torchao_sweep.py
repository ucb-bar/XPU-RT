"""Tests for the TorchAO scheme catalog (C.1)."""

from __future__ import annotations

import pytest
from compgen.capture.torchao_schemes import (
    TORCHAO_SCHEMES,
    TorchAOScheme,
    list_schemes,
    resolve_config,
    scheme_status,
)


def test_catalog_has_expected_stability_groups():
    # At least 9 stable, 6 prototype, 2 QAT, 1 compgen_custom per the plan.
    stabilities = {s.stability for s in TORCHAO_SCHEMES.values()}
    assert {"stable", "prototype", "qat", "compgen_custom"}.issubset(stabilities)


def test_catalog_minimum_scheme_count():
    # Plan targeted 15+ distinct schemes; we shipped 20+.
    assert len(TORCHAO_SCHEMES) >= 20


@pytest.mark.parametrize("name", sorted(TORCHAO_SCHEMES))
def test_every_scheme_has_status(name):
    s = TORCHAO_SCHEMES[name]
    status = scheme_status(s)
    assert status in ("ok", "schema_only")


@pytest.mark.parametrize("name", sorted(TORCHAO_SCHEMES))
def test_every_scheme_declares_required_fields(name):
    s = TORCHAO_SCHEMES[name]
    assert isinstance(s, TorchAOScheme)
    assert s.name == name
    assert s.config_class_path
    assert s.weight_dtype
    assert s.granularity
    assert s.stability in ("stable", "prototype", "qat", "compgen_custom")
    assert s.target_hardware


def test_list_schemes_filters_by_stability():
    stable_only = list_schemes(stability="stable")
    assert all(s.stability == "stable" for s in stable_only)
    assert len(stable_only) >= 9


def test_list_schemes_filters_by_target():
    npu_only = list_schemes(target_hardware="npu")
    assert all(s.target_hardware == "npu" for s in npu_only)
    # Currently at least the fp8_e4m3_po2_npu custom scheme
    assert len(npu_only) >= 1


def test_resolve_config_returns_none_on_missing():
    """Unknown config paths resolve to None without raising."""
    bogus = TorchAOScheme(
        name="bogus",
        config_class_path="totally_fake.module.path.Config",
        weight_dtype="int8",
        granularity="per_tensor",
        stability="stable",
        target_hardware="any",
    )
    assert resolve_config(bogus) is None
    assert scheme_status(bogus) == "schema_only"
