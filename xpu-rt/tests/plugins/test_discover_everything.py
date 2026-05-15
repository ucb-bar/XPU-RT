"""Tests for ``compgen.plugins.discover_everything`` — the unified facade.

Locks in:
  * returns a :class:`DiscoveryReport` aggregating entry-point plugins,
    vendor dialects, and user-space ``~/.compgen/extensions/`` state
  * is safe to call when no extensions are installed (no crash, empty lists)
  * honours ``COMPGEN_EXTENSIONS_DIR`` so tests can point at a sandbox
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from compgen.plugins import KNOWN_GROUPS, DiscoveryReport, discover_everything, reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_discover_everything_empty_install_returns_report():
    report = discover_everything()
    assert isinstance(report, DiscoveryReport)
    # Every known entry-point group is present in the dict (even if empty).
    assert set(report.entry_point_plugins.keys()) == set(KNOWN_GROUPS)
    for names in report.entry_point_plugins.values():
        assert isinstance(names, list)
    assert isinstance(report.vendor_dialects, list)
    assert isinstance(report.user_space_tools, list)
    assert isinstance(report.user_space_slots, list)
    assert isinstance(report.user_space_errors, list)


def test_discover_everything_picks_up_user_space_extension(tmp_path: Path, monkeypatch):
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    # Drop a minimal ext that registers no tools/slots — enough to exercise
    # the loader path. Keeping the payload trivial means we don't need to
    # fabricate a Tool/InventSlot object for this smoke test.
    (ext_dir / "minimal.py").write_text(
        textwrap.dedent(
            """
            def register(registry):
                return
            """
        )
    )
    monkeypatch.setenv("COMPGEN_EXTENSIONS_DIR", str(ext_dir))

    # Clear any per-process LLM-registry state so the user-space loader
    # actually imports our fresh file.
    from compgen.llm import registry as llm_registry

    llm_registry.get_registry().clear()

    report = discover_everything()
    assert report.user_space_root == str(ext_dir)
    # The ext registered nothing; the result still succeeds.
    assert report.user_space_errors == []


def test_discover_everything_reports_total():
    report = discover_everything()
    # total() counts all four lists; clean install = 0.
    assert report.total() == sum(len(v) for v in report.entry_point_plugins.values()) + len(
        report.vendor_dialects
    ) + len(report.user_space_tools) + len(report.user_space_slots)


def test_discover_everything_is_idempotent():
    # Running twice must not double-count.
    r1 = discover_everything()
    r2 = discover_everything()
    assert {g: len(v) for g, v in r1.entry_point_plugins.items()} == {
        g: len(v) for g, v in r2.entry_point_plugins.items()
    }
    assert len(r1.vendor_dialects) == len(r2.vendor_dialects)
