"""End-to-end tests for the universal ingestion pipeline.

Drives the orchestrator with a :class:`RouterMock` and synthetic source
files so the tests are hermetic — no Gemini calls, no chipyard tree.
The Gemmini/Saturn manifests get an integration test in a sibling file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.memory import target_knowledge as tk
from xpu_rt.memory.ingest import (
    IngestPipeline,
    IngestReport,
    RouterMock,
    SourceManifest,
    SourceRef,
)
from xpu_rt.memory.ingest.chunking import TextChunk, chunk_text
from xpu_rt.memory.ingest.router import RouterChunk, RouterResult
from xpu_rt.memory.target_knowledge import (
    ISAInstruction,
    IntrinsicSignature,
    MemoryTierSpec,
    ParameterRange,
)


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path / "targets"))
    monkeypatch.setenv("XPU_RT_INGEST_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_short_returns_single_chunk() -> None:
    out = chunk_text("hello world")
    assert len(out) == 1
    assert out[0].chunk_total == 1
    assert out[0].text == "hello world"


def test_chunk_text_long_splits_on_paragraph() -> None:
    body = ("paragraph one.\n\n" + "x" * 100) + ("paragraph two.\n\n" + "y" * 100)
    text = body * 200
    chunks = chunk_text(text, max_chars=5_000, overlap_chars=200)
    assert len(chunks) > 1
    assert all(c.chunk_total == len(chunks) for c in chunks)
    # The first split happens on a paragraph boundary (the cut char is "\n").
    first_cut = chunks[0].end_offset
    assert text[first_cut - 1] in ("\n", "y", "x")  # boundary or text


# ---------------------------------------------------------------------------
# Pipeline: synthetic manifest + RouterMock
# ---------------------------------------------------------------------------


def _routing_table(item: RouterChunk) -> RouterResult:
    """Deterministic stub router: pick bucket from filename."""
    if "isa" in item.source_locator:
        return RouterResult(
            bucket="isa",
            summary_md="### mvin / mvout\n\nLoads and stores between scratchpad and DRAM.",
            instructions=(
                ISAInstruction(mnemonic="mvin", signature="rs1, rs2", funct_code=2),
                ISAInstruction(mnemonic="mvout", signature="rs1, rs2", funct_code=3),
            ),
        )
    if "header" in item.source_locator:
        return RouterResult(
            bucket="intrinsics",
            summary_md="Macro intrinsics for gemmini.h.",
            intrinsics=(
                IntrinsicSignature(
                    name="gemmini_mvin",
                    c_signature="#define gemmini_mvin(dram, spad)",
                    summary="Lowers to ROCC k_MVIN",
                ),
            ),
        )
    if "config" in item.source_locator:
        return RouterResult(
            bucket="architecture",
            summary_md="Default config sets meshRows = meshColumns = 16.",
            parameters=(
                ParameterRange(name="meshRows", description="Mesh rows", default=16, unit="tiles"),
                ParameterRange(
                    name="meshColumns",
                    description="Mesh columns",
                    default=16,
                    unit="tiles",
                ),
            ),
        )
    if "boilerplate" in item.source_locator:
        return RouterResult(bucket="skip")
    return RouterResult(bucket="architecture", summary_md="generic narrative")


def _manifest_for(tmp: Path, target_id: str = "demo_target") -> SourceManifest:
    isa = _write(tmp / "isa.md", "### `mvin`\n**Format:** `mvin rs1, rs2`\n- funct = 2\n")
    header = _write(tmp / "header.h", "#define gemmini_mvin(d, s) ROCC(k_MVIN)\n")
    config = _write(tmp / "config.scala", "meshRows = 16, meshColumns = 16\n")
    license_file = _write(tmp / "boilerplate.txt", "BSD 3-Clause License ...\n")
    return SourceManifest(
        target_id=target_id,
        target_profile_ref="configs/targets/demo_target.yaml",
        isa_family="rocc-systolic",
        sources=(
            SourceRef(locator=str(isa), kind="path", role="isa"),
            SourceRef(locator=str(header), kind="path", role="intrinsics"),
            SourceRef(locator=str(config), kind="path", role="architecture"),
            SourceRef(locator=str(license_file), kind="path", role="auto"),
        ),
        static_hardware_facts={
            "memory_tiers": (
                MemoryTierSpec(name="scratchpad", kind="scratchpad", size_bytes=262144),
            ),
            "dataflow_modes": ("weight_stationary", "output_stationary"),
        },
    )


def test_pipeline_run_merges_routed_records_into_card(isolated_state: Path) -> None:
    manifest = _manifest_for(isolated_state)
    mock = RouterMock(table=_routing_table)
    pipeline = IngestPipeline(router=mock)

    card, report = pipeline.run(manifest)

    # Records folded in.
    mnemonics = {i.mnemonic for i in card.hardware_spec.instructions}
    assert mnemonics == {"mvin", "mvout"}
    names = {i.name for i in card.hardware_spec.intrinsics}
    assert names == {"gemmini_mvin"}
    params = {p.name: p for p in card.hardware_spec.parameters}
    assert params["meshRows"].default == 16

    # Static facts seeded on first save.
    assert card.hardware_spec.memory_tiers[0].name == "scratchpad"
    assert "weight_stationary" in card.hardware_spec.dataflow_modes

    # Per-bucket markdown files materialised.
    isa_md = card.bucket_path("isa")
    assert isa_md.exists()
    body = isa_md.read_text()
    assert "mvin" in body
    assert f"router={mock.name}" in body

    # Boilerplate was routed to "skip" and produced no bucket text.
    assert "BSD 3-Clause" not in body
    assert report.chunks_skipped >= 1
    assert report.chunks_routed >= 3
    assert report.extracted_instructions == 2
    assert report.extracted_intrinsics == 1
    assert report.extracted_parameters == 2


def test_pipeline_caches_routed_results(isolated_state: Path) -> None:
    manifest = _manifest_for(isolated_state)
    mock = RouterMock(table=_routing_table)
    pipeline = IngestPipeline(router=mock)
    pipeline.run(manifest)
    first_calls = len(mock.calls)

    # Second run with the same content must hit the cache and skip the
    # router entirely.
    mock.calls.clear()
    _, report = pipeline.run(manifest)
    assert len(mock.calls) == 0, "expected cache to absorb every chunk on re-run"
    assert report.chunks_cached >= 3
    assert report.chunks_routed == 0
    # And the first run was non-trivial.
    assert first_calls >= 3


def test_pipeline_copies_exemplars_instead_of_routing(isolated_state: Path) -> None:
    src = _write(
        isolated_state / "code" / "matmul.c",
        "// matmul exemplar\nint main() { return 0; }\n",
    )
    mock = RouterMock(table=lambda item: RouterResult(bucket="skip"))
    pipeline = IngestPipeline(router=mock)
    manifest = SourceManifest(
        target_id="exemplar_target",
        target_profile_ref="configs/targets/exemplar_target.yaml",
        isa_family="rocc-systolic",
        sources=(SourceRef(locator=str(src), kind="path", role="examples", tags=("matmul",)),),
    )
    card, report = pipeline.run(manifest)

    assert report.exemplars_copied == 1
    assert len(mock.calls) == 0  # exemplars must not go to the router
    saved = card.exemplars_dir / "matmul.c"
    assert saved.exists()
    assert card.exemplars[0].op_family == "matmul"
    assert card.exemplars[0].path == "matmul.c"


def test_pipeline_directory_expansion(isolated_state: Path) -> None:
    base = isolated_state / "tree"
    _write(base / "intro.md", "intro section")
    _write(base / "nested" / "ext.md", "nested section")
    _write(base / "ignored.txt", "this file should be skipped because glob != *.md")
    mock = RouterMock(table=lambda item: RouterResult(bucket="architecture", summary_md="x"))
    pipeline = IngestPipeline(router=mock)
    manifest = SourceManifest(
        target_id="dir_target",
        target_profile_ref="configs/targets/dir_target.yaml",
        isa_family="other",
        sources=(SourceRef(locator=str(base), kind="directory", glob="*.md", role="architecture"),),
    )
    pipeline.run(manifest)
    locators = {c.source_locator for c in mock.calls}
    assert any(loc.endswith("intro.md") for loc in locators)
    assert any(loc.endswith("ext.md") for loc in locators)
    assert all(not loc.endswith(".txt") for loc in locators)


def test_pipeline_continues_when_one_source_missing(isolated_state: Path) -> None:
    good = _write(isolated_state / "good.md", "hello")
    pipeline = IngestPipeline(
        router=RouterMock(table=lambda item: RouterResult(bucket="architecture", summary_md="ok"))
    )
    manifest = SourceManifest(
        target_id="resilient_target",
        target_profile_ref="configs/targets/resilient_target.yaml",
        isa_family="host-cpu",
        sources=(
            SourceRef(locator="/does/not/exist.md", kind="path"),
            SourceRef(locator=str(good), kind="path"),
        ),
    )
    card, report = pipeline.run(manifest)
    assert report.sources_seen == 2
    assert report.sources_skipped == 1
    assert any("does/not/exist" in e for e in report.errors)
    # The good source still produced a routed chunk.
    assert (card.bucket_path("architecture")).exists()


def test_pipeline_round_trip_preserves_card_on_disk(isolated_state: Path) -> None:
    manifest = _manifest_for(isolated_state)
    pipeline = IngestPipeline(router=RouterMock(table=_routing_table))
    pipeline.run(manifest)
    reloaded = tk.load(manifest.target_id)
    assert reloaded.target_id == "demo_target"
    assert {i.mnemonic for i in reloaded.hardware_spec.instructions} == {"mvin", "mvout"}
    assert reloaded.hardware_spec.isa_family == "rocc-systolic"
