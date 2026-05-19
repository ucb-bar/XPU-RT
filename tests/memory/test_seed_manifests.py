"""Manifest-shape tests for Gemmini and Saturn.

These are unit-level: they validate that each manifest enumerates the
expected canonical sources and propagates static facts. The end-to-end
integration (run the ingestion against the real Chipyard tree) is
covered separately by an opt-in test gated on the host filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xpu_rt.memory.ingest import IngestPipeline, RouterMock, SourceManifest
from xpu_rt.memory.ingest.router import RouterChunk, RouterResult
from xpu_rt.memory.seeds import gemmini, saturn


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_gemmini_manifest_shape() -> None:
    m = gemmini.manifest()
    assert isinstance(m, SourceManifest)
    assert m.target_id == "gemmini_mx"
    assert m.target_profile_ref == "configs/targets/gemmini_mx.yaml"
    assert m.isa_family == "rocc-systolic"
    loc_set = {s.locator for s in m.sources}
    # Required sources present.
    assert any(s.endswith("README.md") for s in loc_set)
    assert any(s.endswith("Configs.scala") for s in loc_set)
    assert any(s.endswith("gemmini.h") for s in loc_set)
    # No URL sources unless include_urls=True.
    assert all(s.kind != "url" for s in m.sources)
    # Static facts surfaced.
    assert m.static_hardware_facts["dataflow_modes"] == ("weight_stationary", "output_stationary")


def test_gemmini_manifest_include_urls() -> None:
    m = gemmini.manifest(include_urls=True)
    assert any(s.kind == "url" for s in m.sources)


def test_saturn_manifest_shape() -> None:
    m = saturn.manifest()
    assert m.target_id == "saturn_opu_v128"
    assert m.isa_family == "riscv-rvv"
    loc_set = {s.locator for s in m.sources}
    assert any(s.endswith("README.md") for s in loc_set)
    assert any(s.endswith("Parameters.scala") for s in loc_set)
    # docs/ asciidoc directory present.
    docs_refs = [s for s in m.sources if s.kind == "directory" and s.locator.endswith("/docs")]
    assert docs_refs and docs_refs[0].glob == "*.adoc"
    # Curated benchmark coverage spans vec + opu.
    vec_dirs = [s for s in m.sources if "vec-sgemm" in s.locator]
    opu_dirs = [s for s in m.sources if "opu-gemm" in s.locator]
    assert vec_dirs and opu_dirs


def test_source_root_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPU_RT_CHIPYARD_GEMMINI_ROOT", "/override/gemmini")
    monkeypatch.setenv("XPU_RT_CHIPYARD_SATURN_ROOT", "/override/saturn")
    assert gemmini.source_root() == Path("/override/gemmini")
    assert saturn.source_root() == Path("/override/saturn")
    # Manifests reflect the override.
    g = gemmini.manifest()
    s = saturn.manifest()
    assert any("/override/gemmini" in src.locator for src in g.sources)
    assert any("/override/saturn" in src.locator for src in s.sources)


# ---------------------------------------------------------------------------
# Integration: drive the manifest through the pipeline with a mock router
# ---------------------------------------------------------------------------


def _has_gemmini_tree() -> bool:
    return (gemmini.source_root() / "README.md").is_file()


def _has_saturn_tree() -> bool:
    return (saturn.source_root() / "README.md").is_file()


def _routing_table(item: RouterChunk) -> RouterResult:
    """A stub router that always returns a non-skip bucket so the pipeline writes."""
    if item.source_kind == "c":
        return RouterResult(bucket="examples", summary_md="exemplar")
    return RouterResult(bucket="architecture", summary_md=f"chunk from {item.source_locator}")


@pytest.mark.skipif(not _has_gemmini_tree(), reason="chipyard gemmini tree not available")
def test_gemmini_manifest_drives_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path / "targets"))
    monkeypatch.setenv("XPU_RT_INGEST_CACHE_DIR", str(tmp_path / "cache"))
    pipeline = IngestPipeline(router=RouterMock(table=_routing_table), max_chunks_per_source=4)
    card, report = pipeline.run(gemmini.manifest())
    assert card.target_id == "gemmini_mx"
    assert card.hardware_spec.isa_family == "rocc-systolic"
    # Static facts seeded on the card.
    assert len(card.hardware_spec.memory_tiers) >= 2
    # At least the README/Configs.scala/gemmini.h sources processed.
    assert report.sources_seen >= 3
    # Exemplars copied (matmul*.c + conv.c + template.c) — the bareMetalC
    # directory has them.
    assert report.exemplars_copied >= 3


@pytest.mark.skipif(not _has_saturn_tree(), reason="chipyard saturn tree not available")
def test_saturn_manifest_drives_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path / "targets"))
    monkeypatch.setenv("XPU_RT_INGEST_CACHE_DIR", str(tmp_path / "cache"))
    pipeline = IngestPipeline(router=RouterMock(table=_routing_table), max_chunks_per_source=2)
    card, report = pipeline.run(saturn.manifest())
    assert card.target_id == "saturn_opu_v128"
    assert card.hardware_spec.isa_family == "riscv-rvv"
    # AsciiDoc files routed (no parser required — loader normalizes to text).
    assert report.sources_seen > 5
    # vec-sgemm benchmark exemplar copied.
    assert report.exemplars_copied >= 1
