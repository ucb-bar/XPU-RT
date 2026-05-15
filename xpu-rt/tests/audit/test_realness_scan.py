"""Tests for compgen.audit.realness_scan.

The contract is: source-level scan finds stub/mock/placeholder markers,
the allowlist tolerates intentional ones, and unallowlisted hits raise
:class:`UnallowlistedStubError`.

The most important invariant: running the scan against the live repo at
its default config produces zero unallowlisted hits. If a contributor
introduces a new mock/stub, they either fix it or add it to the
allowlist with a reason.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from compgen.audit.errors import UnallowlistedStubError
from compgen.audit.realness_scan import (
    Allowlist,
    AllowlistEntry,
    Hit,
    SCAN_PATTERN,
    ScanReport,
    assert_clean,
    scan_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scan_pattern_matches_expected_markers() -> None:
    samples = ["TODO", "FIXME", "XXX", "HACK", "hardcoded", "temporary"]
    for s in samples:
        assert SCAN_PATTERN.search(f"foo {s} bar"), f"pattern should match {s!r}"


def test_scan_pattern_avoids_false_positives() -> None:
    # The narrowed pattern only matches strong residual markers, not
    # domain words. Verify domain terms are NOT flagged.
    no_match_samples = [
        "FX placeholder node",
        "synthetic test data",
        "Mock Trial 5",
        "FakeQuantize",
        "dummy variable",
        "stubborn bug",
    ]
    for s in no_match_samples:
        assert SCAN_PATTERN.search(s) is None, f"pattern should NOT match {s!r}"


def test_live_repo_is_clean() -> None:
    """The whole point: running on origin/main must produce zero unallowlisted hits."""
    report = scan_repo(repo_root=REPO_ROOT, include_tests=False)
    assert report.files_scanned > 0
    bad = report.unallowlisted_hits
    sample_msg = "\n".join(
        f"  {h.path}:{h.line_number}: [{h.marker}] {h.line_text}"
        for h in bad[:25]
    )
    assert not bad, (
        f"realness scan found {len(bad)} unallowlisted hit(s) on the live "
        f"repo. Either fix the code or add an allowlist entry.\n{sample_msg}"
    )


def test_assert_clean_raises_on_unallowlisted_hit(tmp_path: Path) -> None:
    bad_file = tmp_path / "python" / "compgen" / "fake_code.py"
    bad_file.parent.mkdir(parents=True)
    bad_file.write_text("# TODO: write the actual implementation\n")
    report = scan_repo(repo_root=tmp_path, roots=("python/compgen",))
    assert report.files_scanned == 1
    assert any(h.marker == "TODO" for h in report.hits)
    with pytest.raises(UnallowlistedStubError, match="TODO"):
        assert_clean(report)


def test_allowlist_silences_known_path(tmp_path: Path) -> None:
    bad_file = tmp_path / "python" / "compgen" / "llm" / "mock_client.py"
    bad_file.parent.mkdir(parents=True)
    bad_file.write_text("class MockClient: pass\n")
    allowlist = Allowlist(
        entries=(
            AllowlistEntry(
                path="python/compgen/llm/mock_client.py",
                reason="test-only",
                forbidden_in=("graph_compilation.run",),
            ),
        ),
        content_pattern_exemptions=(),
        exclude_paths=(),
    )
    report = scan_repo(
        repo_root=tmp_path,
        roots=("python/compgen",),
        allowlist=allowlist,
    )
    assert all(h.allowlisted for h in report.hits), (
        f"expected all hits to be allowlisted, got: {[h.to_dict() for h in report.hits]}"
    )
    # Should not raise
    assert_clean(report)


def test_excluded_path_is_skipped(tmp_path: Path) -> None:
    excluded_file = tmp_path / "python" / "compgen" / "__pycache__" / "x.pyc"
    excluded_file.parent.mkdir(parents=True)
    excluded_file.write_text("TODO: this should not be scanned")
    allowlist = Allowlist(
        entries=(),
        content_pattern_exemptions=(),
        exclude_paths=("**/__pycache__/**",),
    )
    report = scan_repo(
        repo_root=tmp_path,
        roots=("python/compgen",),
        allowlist=allowlist,
    )
    # __pycache__ should not even be opened
    assert report.files_scanned == 0


def test_tests_dir_skipped_by_default(tmp_path: Path) -> None:
    test_file = tmp_path / "python" / "compgen" / "tests" / "test_foo.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# TODO: refactor this\n")
    report = scan_repo(
        repo_root=tmp_path,
        roots=("python/compgen",),
        include_tests=False,
    )
    # /tests/ inside the scanned root is skipped
    assert report.files_scanned == 0


def test_include_tests_picks_up_test_dir(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_foo.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# FIXME: write a real test\n")
    report = scan_repo(
        repo_root=tmp_path,
        roots=("tests",),
        include_tests=True,
    )
    assert report.files_scanned == 1
    assert any(h.marker == "FIXME" for h in report.hits)


def test_seed_allowlist_loads() -> None:
    allowlist = Allowlist.load()
    # Sanity: the seed allowlist must include mock_client.py
    paths = {e.path for e in allowlist.entries}
    assert "python/compgen/llm/mock_client.py" in paths


def test_for_now_and_raise_not_impl_markers_caught(tmp_path: Path) -> None:
    bad = tmp_path / "python" / "compgen" / "x.py"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "def foo():\n"
        "    raise NotImplementedError('not done')\n"
        "    # hardcoded: for now we just return 1\n"
    )
    report = scan_repo(repo_root=tmp_path, roots=("python/compgen",))
    markers = {h.marker for h in report.hits}
    assert "raise NotImplementedError" in markers
    assert "for now" in markers
    assert "hardcoded" in markers
