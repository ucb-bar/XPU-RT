"""Scanner finds the right facts about a toy vendor repo."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.extensions.vendor_dialect.scanner import scan_repo

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "fake_vendor"


def test_scan_finds_readme_and_license() -> None:
    res = scan_repo(FIXTURE)
    assert res.readme_text
    assert "FakeVendorMLIR" in res.readme_text
    assert res.license_spdx == "Apache-2.0"


def test_scan_finds_td_ops() -> None:
    res = scan_repo(FIXTURE)
    op_names = {op.name for op in res.td_ops}
    assert {"Fake_MatmulOp", "Fake_SoftmaxOp"} <= op_names
    matmul = next(op for op in res.td_ops if op.name == "Fake_MatmulOp")
    assert "Tiled matmul" in matmul.summary


def test_scan_finds_dialect_name() -> None:
    res = scan_repo(FIXTURE)
    assert "fake" in res.dialect_names


def test_scan_finds_cli_tools() -> None:
    res = scan_repo(FIXTURE)
    assert {"fake-opt", "fake-translate"} <= set(res.cli_tools)


def test_scan_collects_tests_and_tutorials() -> None:
    res = scan_repo(FIXTURE)
    assert any("round_trip" in p for p in res.test_examples)
    assert any("tutorial" in p for p in res.tutorial_docs)


def test_scan_missing_repo_raises() -> None:
    with pytest.raises(FileNotFoundError):
        scan_repo("/does/not/exist/vendor-repo")
