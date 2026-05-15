"""Tests for xpu_rt.audit.import_provenance."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from xpu_rt.audit.errors import ForbiddenImportError
from xpu_rt.audit.import_provenance import (
    DEFAULT_FORBIDDEN_MODULES,
    DEFAULT_MOCK_MODULES,
    ImportProvenance,
    ImportSnapshot,
    assert_no_forbidden,
    compute_provenance,
    load_provenance,
    write_provenance,
)


def test_snapshot_filters_to_prefixes() -> None:
    snap = ImportSnapshot.take("after", prefixes=("xpu-rt",))
    # Every entry must start with "xpu-rt" or equal it
    for m in snap.modules:
        assert m == "xpu-rt" or m.startswith("xpu_rt.")


def test_compute_provenance_diffs_snapshots() -> None:
    before = ImportSnapshot(label="before", modules=("xpu-rt",))
    after = ImportSnapshot(
        label="after",
        modules=("xpu-rt", "xpu_rt.foo", "xpu_rt.bar"),
    )
    prov = compute_provenance(
        before=before,
        after=after,
        run_id="r1",
        selection_mode="greedy",
        source_commit="deadbeef",
    )
    # New modules = after - before
    assert prov.production_modules_imported == ["xpu_rt.bar", "xpu_rt.foo"]


def test_provenance_flags_forbidden_module() -> None:
    before = ImportSnapshot(label="before", modules=("xpu-rt",))
    after = ImportSnapshot(
        label="after",
        modules=("xpu-rt", "xpu_rt.llm.mock_client"),
    )
    prov = compute_provenance(
        before=before,
        after=after,
        run_id="r1",
        selection_mode="greedy",
        source_commit="deadbeef",
    )
    assert "xpu_rt.llm.mock_client" in prov.forbidden_modules_imported
    assert prov.evidence_mode == "mocked"
    with pytest.raises(ForbiddenImportError, match="forbidden"):
        assert_no_forbidden(prov)


def test_provenance_clean_run_passes() -> None:
    before = ImportSnapshot(label="before", modules=("xpu-rt",))
    after = ImportSnapshot(
        label="after",
        modules=("xpu-rt", "xpu_rt.graph_compilation.run"),
    )
    prov = compute_provenance(
        before=before,
        after=after,
        run_id="r1",
        selection_mode="greedy",
        source_commit="deadbeef",
    )
    assert prov.forbidden_modules_imported == []
    assert prov.evidence_mode == "real"
    # Should not raise
    assert_no_forbidden(prov)


def test_cache_mode_disabled_under_env() -> None:
    before = ImportSnapshot(label="before", modules=())
    after = ImportSnapshot(label="after", modules=("xpu-rt",))
    with mock.patch.dict(os.environ, {"XPU_RT_DISABLE_RECIPE_MEMORY": "1"}):
        prov = compute_provenance(
            before=before,
            after=after,
            run_id="r1",
            selection_mode="greedy",
            source_commit="deadbeef",
        )
    assert prov.cache_mode == "disabled"
    assert prov.env_overrides.get("XPU_RT_DISABLE_RECIPE_MEMORY") == "1"


def test_cache_mode_disabled_under_kernel_env() -> None:
    before = ImportSnapshot(label="before", modules=())
    after = ImportSnapshot(label="after", modules=("xpu-rt",))
    with mock.patch.dict(os.environ, {"XPU_RT_DISABLE_KERNEL_CACHE": "1"}):
        prov = compute_provenance(
            before=before,
            after=after,
            run_id="r1",
            selection_mode="greedy",
            source_commit="deadbeef",
        )
    assert prov.cache_mode == "disabled"


def test_provenance_round_trip(tmp_path: Path) -> None:
    prov = ImportProvenance(
        run_id="r1",
        selection_mode="greedy",
        source_commit="abc1234",
        cache_mode="cold",
        evidence_mode="real",
        production_modules_imported=["xpu_rt.foo"],
    )
    out = tmp_path / "import_provenance.json"
    out.write_text(__import__("json").dumps(prov.to_dict(), indent=2, sort_keys=True) + "\n")
    reloaded = load_provenance(out)
    assert reloaded.to_dict() == prov.to_dict()


def test_write_provenance_creates_file(tmp_path: Path) -> None:
    prov = ImportProvenance(run_id="r1", selection_mode="greedy", source_commit="abc")
    out = write_provenance(prov, run_dir=tmp_path)
    assert out.exists()
    assert out.name == "import_provenance.json"


def test_default_forbidden_modules_listed() -> None:
    # The DEFAULT_FORBIDDEN_MODULES list is a public contract: changing
    # it requires updating the realness allowlist + this test.
    assert "xpu_rt.llm.mock_client" in DEFAULT_FORBIDDEN_MODULES


def test_default_mock_modules_includes_mock_client() -> None:
    assert "xpu_rt.llm.mock_client" in DEFAULT_MOCK_MODULES


def test_additional_forbidden_extends_check() -> None:
    before = ImportSnapshot(label="before", modules=())
    after = ImportSnapshot(
        label="after",
        modules=("xpu-rt", "xpu_rt.experimental.unfinished"),
    )
    prov = compute_provenance(
        before=before,
        after=after,
        run_id="r1",
        selection_mode="greedy",
        source_commit="deadbeef",
    )
    # Default forbidden list does not catch this one
    assert_no_forbidden(prov)
    # But the additional list does
    with pytest.raises(ForbiddenImportError):
        assert_no_forbidden(prov, additional_forbidden=("xpu_rt.experimental",))
