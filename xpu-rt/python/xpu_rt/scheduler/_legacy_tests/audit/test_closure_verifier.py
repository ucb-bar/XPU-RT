"""closure verifier tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev import verify_all_providers as v


def test_verify_all_on_current_repo_passes():
    """The current shipped state must pass every card check."""

    checks = v.verify_all()
    failing = [c for c in checks if not c.passes]
    assert not failing, "\n".join(f"{c.card_id}: {c.detail}" for c in failing)


def test_verify_covers_all_19_provider_cards():
    checks = v.verify_all()
    provider_ids = {c.card_id for c in checks if c.kind == "provider"}
    assert len(provider_ids) == 19


def test_verify_covers_all_10_dialect_cards():
    checks = v.verify_all()
    dialect_ids = {c.card_id for c in checks if c.kind == "dialect"}
    assert len(dialect_ids) == 10


def test_verify_reports_evidence_states_correctly():
    checks = v.verify_all()
    by_id = {c.card_id: c for c in checks}
    # cffi_c was deepened with real kernel + run report.
    assert by_id["cffi_c"].evidence_state == "available"
    # KernelBlaster has a typed blocked_proof.
    assert by_id["kernelblaster"].evidence_state == "blocked"


def test_cli_exits_zero_on_passing_repo(tmp_path: Path):
    out = tmp_path / "verify.json"
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/dev/verify_all_providers.py",
            "--json-out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    ).returncode
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["failing"] == 0
    assert body["total_cards"] >= 29


def test_card_check_to_dict_is_jsonable():
    checks = v.verify_all()
    for c in checks:
        body = c.to_dict()
        # Round-trip through JSON serializer must succeed.
        json.dumps(body)
