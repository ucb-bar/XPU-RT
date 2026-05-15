"""Tests for compgen.audit.contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from compgen.audit.contracts import (
    PAPER_CLAIMABLE_LEVELS,
    REALNESS_LEVELS,
    RealnessContract,
    iter_contracts,
    load_contract,
    make_contract,
    validate_contract,
    write_contract,
)
from compgen.audit.errors import RealnessContractError

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "docs" / "realness"

SEED_CONTRACTS = (
    "m26_promotion_bridge",
    "m27_recipe_ir_promote_op",
    "m28_promotion_retrieval",
    "m29_promotion_gates",
    "m30_efficiency_report",
    "m31a_audit_layer",
)


@pytest.mark.parametrize("feature_id", SEED_CONTRACTS)
def test_seed_contract_loads_and_validates(feature_id: str) -> None:
    path = SEED_DIR / f"{feature_id}.yaml"
    contract = load_contract(path)
    assert contract.feature_id == feature_id
    assert contract.realness_level in REALNESS_LEVELS
    assert len(contract.required_evidence) > 0
    assert contract.commit  # non-empty


def test_iter_contracts_finds_six_seeds() -> None:
    contracts = list(iter_contracts(SEED_DIR))
    feature_ids = {c.feature_id for c in contracts}
    assert set(SEED_CONTRACTS).issubset(feature_ids)


def test_paper_claimable_levels_match_spec() -> None:
    assert PAPER_CLAIMABLE_LEVELS == frozenset(
        {"decision_affecting", "production_path", "hardware_backed"}
    )


def test_make_contract_round_trip(tmp_path: Path) -> None:
    contract = make_contract(
        feature_id="dummy_feature",
        claim="A non-trivial claim that should pass the length check.",
        realness_level="production_path",
        forbidden=["mocks", "silent skips"],
        required_evidence=["python/compgen/foo.py", "tests/test_foo.py"],
        commit="deadbeef" * 5,
    )
    assert contract.is_paper_claimable
    out = tmp_path / "dummy_feature.yaml"
    write_contract(contract, out)
    reloaded = load_contract(out)
    assert reloaded.to_dict() == contract.to_dict()


def test_invalid_feature_id_rejected() -> None:
    with pytest.raises(RealnessContractError, match="feature_id"):
        make_contract(
            feature_id="Bad-Name",  # uppercase + dash
            claim="Some non-trivial claim that has length.",
            realness_level="production_path",
            required_evidence=["x"],
            commit="abc",
        )


def test_invalid_level_rejected() -> None:
    with pytest.raises(RealnessContractError, match="realness_level"):
        make_contract(
            feature_id="x_y",
            claim="Some non-trivial claim that has length.",
            realness_level="totally_made_up",
            required_evidence=["x"],
            commit="abc",
        )


def test_empty_claim_rejected() -> None:
    with pytest.raises(RealnessContractError, match="claim"):
        make_contract(
            feature_id="x_y",
            claim="x",
            realness_level="production_path",
            required_evidence=["x"],
            commit="abc",
        )


def test_empty_required_evidence_rejected() -> None:
    with pytest.raises(RealnessContractError, match="required_evidence"):
        make_contract(
            feature_id="x_y",
            claim="A non-trivial claim that should pass the length check.",
            realness_level="production_path",
            required_evidence=[],
            commit="abc",
        )


def test_malformed_yaml_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_mapping_just_a_string")
    with pytest.raises(RealnessContractError):
        load_contract(bad)


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "feature_id": "x_y",
                "claim": "Some non-trivial claim that has length.",
                # missing realness_level
                "required_evidence": ["x"],
                "commit": "abc",
                "created_at_utc": "2026-05-05T00:00:00Z",
            }
        )
    )
    with pytest.raises(RealnessContractError, match="realness_level"):
        load_contract(bad)


def test_seed_contracts_use_recent_commit() -> None:
    """Every seed contract should reference a non-empty commit hash."""
    for feature_id in SEED_CONTRACTS:
        contract = load_contract(SEED_DIR / f"{feature_id}.yaml")
        assert len(contract.commit) >= 7
