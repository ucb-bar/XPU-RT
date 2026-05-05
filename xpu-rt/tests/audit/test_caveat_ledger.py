"""Tests for compgen.audit.caveat_ledger (M-31A.1)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from compgen.audit.caveat_ledger import (
    CAVEAT_STATUSES,
    DEFAULT_STALE_DAYS,
    Caveat,
    CaveatLedger,
    make_caveat,
)
from compgen.audit.errors import CaveatLedgerError, StaleCaveatError

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_LEDGER = REPO_ROOT / "results" / "audit" / "_seed" / "caveat_ledger.json"


def test_seed_ledger_loads() -> None:
    ledger = CaveatLedger.load(SEED_LEDGER)
    assert len(ledger) >= 1
    portable = ledger.get("portable_gate_single_target")
    assert portable is not None
    assert portable.status == "blocked_by_hardware"
    assert portable.blocks_paper_claim is False


def test_seed_ledger_validates_with_allow_stale() -> None:
    ledger = CaveatLedger.load(SEED_LEDGER)
    # Seed entries are dated 2026-05-05; allow_stale=True ignores age.
    ledger.validate(allow_stale=True)


def test_make_caveat_round_trip(tmp_path: Path) -> None:
    caveat = make_caveat(
        id="example_one",
        claim_affected="some_claim",
        status="open",
        is_bug=False,
        blocks_paper_claim=True,
        required_to_close="ship the dependent feature",
        evidence_paths=["docs/foo.md"],
    )
    ledger = CaveatLedger()
    ledger.add(caveat)
    out = tmp_path / "ledger.json"
    ledger.dump(out)

    reloaded = CaveatLedger.load(out)
    assert len(reloaded) == 1
    assert reloaded.get("example_one") == caveat


def test_byte_stable_round_trip(tmp_path: Path) -> None:
    """Dump -> load -> dump must produce byte-identical output."""
    ledger = CaveatLedger.load(SEED_LEDGER)
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    ledger.dump(out1)
    CaveatLedger.load(out1).dump(out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_duplicate_id_rejected() -> None:
    ledger = CaveatLedger()
    a = make_caveat(
        id="dup",
        claim_affected="x",
        status="open",
        is_bug=False,
        blocks_paper_claim=False,
        required_to_close="close it",
        evidence_paths=["docs/foo.md"],
    )
    ledger.add(a)
    with pytest.raises(CaveatLedgerError, match="already present"):
        ledger.add(a)


def test_invalid_status_rejected() -> None:
    with pytest.raises(CaveatLedgerError, match="status"):
        make_caveat(
            id="bad",
            claim_affected="x",
            status="totally_made_up",
            is_bug=False,
            blocks_paper_claim=False,
            required_to_close="close it",
            evidence_paths=["docs/foo.md"],
        )


def test_missing_evidence_rejected() -> None:
    with pytest.raises(CaveatLedgerError, match="evidence_paths"):
        make_caveat(
            id="bad",
            claim_affected="x",
            status="open",
            is_bug=False,
            blocks_paper_claim=False,
            required_to_close="close it",
            evidence_paths=[],
        )


def test_invalid_id_rejected() -> None:
    with pytest.raises(CaveatLedgerError, match="match"):
        make_caveat(
            id="Bad-Id",
            claim_affected="x",
            status="open",
            is_bug=False,
            blocks_paper_claim=False,
            required_to_close="close it",
            evidence_paths=["docs/foo.md"],
        )


def test_resolved_with_blocks_paper_claim_rejected() -> None:
    bad = Caveat(
        id="contradictory",
        claim_affected="x",
        status="resolved",
        is_bug=False,
        blocks_paper_claim=True,  # contradicts resolved
        required_to_close="close it",
        evidence_paths=("docs/foo.md",),
        created_at_utc="2026-05-05T00:00:00Z",
        last_verified_at_utc="2026-05-05T00:00:00Z",
    )
    with pytest.raises(CaveatLedgerError, match="resolved"):
        bad.validate()


def test_update_status_resolves_caveat(tmp_path: Path) -> None:
    caveat = make_caveat(
        id="will_resolve",
        claim_affected="x",
        status="open",
        is_bug=True,
        blocks_paper_claim=True,
        required_to_close="fix the bug",
        evidence_paths=["docs/foo.md"],
    )
    ledger = CaveatLedger()
    ledger.add(caveat)
    updated = ledger.update_status(
        "will_resolve", status="resolved", evidence_path="commits/abc.diff"
    )
    assert updated.status == "resolved"
    # update_status auto-clears blocks_paper_claim when status=resolved
    assert updated.blocks_paper_claim is False
    assert "commits/abc.diff" in updated.evidence_paths


def test_stale_caveat_detected() -> None:
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=DEFAULT_STALE_DAYS + 5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    caveat = Caveat(
        id="stale_one",
        claim_affected="x",
        status="open",
        is_bug=False,
        blocks_paper_claim=False,
        required_to_close="verify",
        evidence_paths=("docs/foo.md",),
        created_at_utc=old_ts,
        last_verified_at_utc=old_ts,
    )
    assert caveat.is_stale()

    ledger = CaveatLedger(caveats=[caveat])
    with pytest.raises(StaleCaveatError, match="stale"):
        ledger.validate(allow_stale=False)
    # allow_stale skips the check
    ledger.validate(allow_stale=True)


def test_resolved_caveat_never_stale() -> None:
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    caveat = Caveat(
        id="resolved_one",
        claim_affected="x",
        status="resolved",
        is_bug=False,
        blocks_paper_claim=False,
        required_to_close="verify",
        evidence_paths=("docs/foo.md",),
        created_at_utc=old_ts,
        last_verified_at_utc=old_ts,
    )
    assert caveat.is_stale() is False


def test_caveat_statuses_match_spec() -> None:
    assert set(CAVEAT_STATUSES) == {
        "open",
        "blocked_by_hardware",
        "blocked_by_external",
        "resolved",
        "rejected",
    }


def test_seed_ledger_is_json_byte_stable_with_dump(tmp_path: Path) -> None:
    """Loading seed and re-dumping must match the on-disk seed file."""
    ledger = CaveatLedger.load(SEED_LEDGER)
    out = tmp_path / "redump.json"
    ledger.dump(out)
    # Compare structurally (timestamps preserved); byte equality after a
    # canonical round-trip:
    assert json.loads(out.read_text()) == json.loads(SEED_LEDGER.read_text())
