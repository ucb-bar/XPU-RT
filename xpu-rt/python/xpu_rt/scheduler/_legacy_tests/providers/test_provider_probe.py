"""provider / dialect probing + matrix report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
)
from compgen.providers.provider_probe import (
    probe_dialect_provider,
    probe_provider,
)
from compgen.providers.provider_reports import write_probe_reports
from compgen.providers.provider_types import (
    BLOCKED_REASONS,
    PROBE_STATUSES,
    ProviderCard,
)


def _card(**kwargs) -> ProviderCard:
    body = {
        "schema_version": "provider_card_v1",
        "provider_id": "test_p",
        "integration_level": "probe",
        "target_families": ["cuda"],
        "contract_kinds": ["matmul"],
        "emits": ["c_source"],
        "entrypoint": "x:Y",
    }
    body.update(kwargs)
    return ProviderCard.from_dict(body)


# ---------------------------------------------------------------------------
# Negative controls — every required-* miss must produce a typed blocked.
# ---------------------------------------------------------------------------


def test_missing_env_var_produces_typed_blocked(monkeypatch):
    monkeypatch.delenv("TOTALLY_FAKE_ENV_VAR_FOR_TESTS", raising=False)
    card = _card(required_env=("TOTALLY_FAKE_ENV_VAR_FOR_TESTS",))
    r = probe_provider(card)
    assert r.status == "blocked"
    assert r.blocked_reason == "env_missing"
    assert r.detail == "TOTALLY_FAKE_ENV_VAR_FOR_TESTS"


def test_empty_env_var_treated_as_missing(monkeypatch):
    monkeypatch.setenv("EMPTY_ENV_VAR_FOR_TESTS", "")
    card = _card(required_env=("EMPTY_ENV_VAR_FOR_TESTS",))
    r = probe_provider(card)
    assert r.status == "blocked"
    assert r.blocked_reason == "env_missing"


def test_present_env_var_passes_env_check(monkeypatch):
    monkeypatch.setenv("PRESENT_ENV_VAR_FOR_TESTS", "1")
    card = _card(required_env=("PRESENT_ENV_VAR_FOR_TESTS",))
    r = probe_provider(card)
    assert r.status == "available"
    assert r.blocked_reason is None


def test_missing_command_produces_typed_blocked():
    card = _card(required_commands=("totally-fake-binary-does-not-exist",))
    r = probe_provider(card)
    assert r.status == "blocked"
    assert r.blocked_reason == "command_missing"
    assert "totally-fake-binary-does-not-exist" in r.detail


def test_missing_python_import_produces_typed_blocked():
    card = _card(required_python_imports=("totally_fake_module_for_tests_xyz",))
    r = probe_provider(card)
    assert r.status == "blocked"
    assert r.blocked_reason == "python_package_missing"
    assert r.detail == "totally_fake_module_for_tests_xyz"


def test_present_python_import_passes():
    card = _card(required_python_imports=("json",))
    r = probe_provider(card)
    assert r.status == "available"


def test_probe_never_raises_on_real_card():
    """Every shipped card must produce a typed result; no exception escapes."""
    for c in iter_provider_cards():
        r = probe_provider(c)
        assert r.status in PROBE_STATUSES
        if r.status != "available":
            assert r.blocked_reason in BLOCKED_REASONS


def test_dialect_probe_never_raises_on_real_card():
    for c in iter_dialect_cards():
        r = probe_dialect_provider(c)
        assert r.status in PROBE_STATUSES
        if r.status != "available":
            assert r.blocked_reason in BLOCKED_REASONS


def test_card_only_dialect_produces_unsupported():
    """A dialect at integration_level=card_only must probe as unsupported."""
    card = DialectProviderCard.from_dict(
        {
            "schema_version": "dialect_provider_card_v1",
            "dialect_provider_id": "test_card_only",
            "dialect_name": "test",
            "integration_level": "card_only",
            "consumes": ["x"],
            "emits": ["y"],
            "entrypoint": "x:Y",
            "required_env": [],
        }
    )
    r = probe_dialect_provider(card)
    assert r.status == "unsupported"
    assert r.blocked_reason == "unsupported_contract_kind"


# ---------------------------------------------------------------------------
# End-to-end report set
# ---------------------------------------------------------------------------


def test_write_probe_reports_produces_full_artifact_set(tmp_path: Path):
    paths = write_probe_reports(tmp_path)
    assert set(paths.keys()) == {
        "provider_status",
        "target_status",
        "dialect_status",
        "pass_tool_status",
        "provider_target_matrix",
        "provider_contract_matrix",
        "probe_summary",
    }
    for rel in paths.values():
        assert (tmp_path / rel).is_file()

    # JSON schemas declared as documented
    p_status = json.loads((tmp_path / "provider_status.json").read_text())
    assert p_status["schema_version"] == "provider_status_v1"
    assert isinstance(p_status["providers"], list)
    assert len(p_status["providers"]) >= 1

    t_status = json.loads((tmp_path / "target_status.json").read_text())
    assert t_status["schema_version"] == "target_status_v1"
    assert len(t_status["targets"]) >= 1

    d_status = json.loads((tmp_path / "dialect_status.json").read_text())
    assert d_status["schema_version"] == "dialect_status_v1"

    pt_status = json.loads((tmp_path / "pass_tool_status.json").read_text())
    assert pt_status["schema_version"] == "pass_tool_status_v1"


def test_no_provider_silently_disappears_from_report(tmp_path: Path):
    """Every shipped card must appear in provider_status.json."""
    write_probe_reports(tmp_path)
    body = json.loads((tmp_path / "provider_status.json").read_text())
    reported = {p["provider_id"] for p in body["providers"]}
    shipped = {c.provider_id for c in iter_provider_cards()}
    assert reported == shipped


def test_baseline_provider_status_is_available_in_this_environment(tmp_path: Path):
    """cffi_c is the correctness anchor — must be available in CI."""
    write_probe_reports(tmp_path)
    body = json.loads((tmp_path / "provider_status.json").read_text())
    by_id = {p["provider_id"]: p for p in body["providers"]}
    assert by_id["cffi_c"]["status"] == "available"


def test_blocked_provider_carries_typed_reason_and_detail(tmp_path: Path):
    """Every non-available provider must carry typed blocked_reason."""
    write_probe_reports(tmp_path)
    body = json.loads((tmp_path / "provider_status.json").read_text())
    for p in body["providers"]:
        if p["status"] != "available":
            assert p["blocked_reason"] in BLOCKED_REASONS, p
            assert p["detail"], f"{p['provider_id']} blocked without detail"


def test_provider_target_matrix_has_header_and_rows(tmp_path: Path):
    write_probe_reports(tmp_path)
    rows = (tmp_path / "provider_target_matrix.csv").read_text().splitlines()
    assert len(rows) >= 2
    header = rows[0].split(",")
    assert "provider_id" in header
    assert "cuda_sm90" in header
    assert "host_cpu" in header


def test_paper_claimable_reflected_in_status(tmp_path: Path):
    """paper_claimable status is faithfully reported (not derived from env)."""
    write_probe_reports(tmp_path)
    body = json.loads((tmp_path / "provider_status.json").read_text())
    by_id = {p["provider_id"]: p for p in body["providers"]}
    assert by_id["cffi_c"]["paper_claimable"] is True
    assert by_id["triton"]["paper_claimable"] is True
    # Blocked SDK-gated provider stays paper_claimable=false even if its
    # card-level integration_level were ever bumped — but at probe level it
    # cannot be paper_claimable (enforces).
    assert by_id["cuda_tile_ir"]["paper_claimable"] is False
