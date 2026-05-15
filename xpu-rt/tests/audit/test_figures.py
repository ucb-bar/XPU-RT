"""real PNG figures in the evidence pack."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.figures import (
    FigureResult,
    render_all_figures,
    render_blocked_reason_breakdown,
    render_extension_lifecycle,
    render_provider_status_by_family,
    render_provider_target_heatmap,
)


def _make_provider_status(pack: Path) -> None:
    body = {
        "schema_version": "provider_status_v1",
        "generated_at_utc": "x",
        "providers": [
            {
                "provider_id": "cffi_c",
                "integration_level": "promote",
                "status": "available",
                "blocked_reason": None,
                "target_families": ["host_cpu"],
                "contract_kinds": ["matmul", "pointwise"],
            },
            {
                "provider_id": "triton",
                "integration_level": "promote",
                "status": "available",
                "blocked_reason": None,
                "target_families": ["cuda"],
                "contract_kinds": ["matmul", "attention"],
            },
            {
                "provider_id": "cuda_tile_ir",
                "integration_level": "probe",
                "status": "blocked",
                "blocked_reason": "env_missing",
                "target_families": ["cuda"],
                "contract_kinds": ["matmul"],
            },
            {
                "provider_id": "pallas",
                "integration_level": "probe",
                "status": "blocked",
                "blocked_reason": "python_package_missing",
                "target_families": ["tpu"],
                "contract_kinds": ["matmul"],
            },
        ],
    }
    (pack / "provider_status.json").write_text(json.dumps(body))


def _make_snapshots(pack: Path) -> None:
    d = pack / "analysis_snapshots"
    d.mkdir(parents=True, exist_ok=True)
    for level in ("fx_graph", "payload_ir", "tile_ir"):
        (d / f"{level}_analysis.json").write_text(
            json.dumps(
                {
                    "schema_version": "ir_analysis_snapshot_v1",
                    "level": level,
                    "status": "available",
                    "source_artifact": "x",
                    "regions": [{"region_id": "r0", "ops": ["a"]}],
                }
            )
        )


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def test_provider_target_heatmap_produces_png(tmp_path: Path):
    _make_provider_status(tmp_path)
    out = tmp_path / "figures" / "provider_target_heatmap.png"
    r = render_provider_target_heatmap(tmp_path, out)
    assert isinstance(r, FigureResult)
    assert not r.skipped
    assert out.is_file()
    # PNG magic bytes
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 1024


def test_provider_status_by_family_produces_png(tmp_path: Path):
    _make_provider_status(tmp_path)
    out = tmp_path / "figures" / "provider_status_by_family.png"
    r = render_provider_status_by_family(tmp_path, out)
    assert not r.skipped
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_blocked_reason_breakdown_produces_png(tmp_path: Path):
    _make_provider_status(tmp_path)
    out = tmp_path / "figures" / "blocked_reason_breakdown.png"
    r = render_blocked_reason_breakdown(tmp_path, out)
    assert not r.skipped
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_extension_lifecycle_produces_png(tmp_path: Path):
    out = tmp_path / "figures" / "extension_lifecycle.png"
    r = render_extension_lifecycle(tmp_path, out)
    assert not r.skipped
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_skipped_when_no_data(tmp_path: Path):
    out = tmp_path / "figures" / "provider_target_heatmap.png"
    r = render_provider_target_heatmap(tmp_path, out)
    assert r.skipped
    assert "no provider_status.json" in r.reason
    assert not out.exists()


# ---------------------------------------------------------------------------
# Bulk renderer + evidence-pack integration
# ---------------------------------------------------------------------------


def test_render_all_figures_emits_five_artifacts(tmp_path: Path):
    """When all data is present, all 5 figures render."""
    _make_provider_status(tmp_path)
    _make_snapshots(tmp_path)
    results = render_all_figures(tmp_path)
    assert len(results) == 5
    names = {r.name for r in results}
    assert names == {
        "provider_target_heatmap.png",
        "provider_status_by_family.png",
        "extension_lifecycle.png",
        "ir_analysis_levels.png",
        "blocked_reason_breakdown.png",
    }
    for r in results:
        assert not r.skipped, f"{r.name} skipped: {r.reason}"
        assert r.path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_existing_pack_has_real_pngs():
    """The canonical evidence pack must have real PNGs (not markdown)."""
    pack = Path("results/extension_provider_evidence_pack")
    if not pack.is_dir():
        pytest.skip("no canonical evidence pack on disk")
    figures = pack / "figures"
    if not figures.is_dir():
        pytest.skip("no figures dir on disk")
    expected = {
        "provider_target_heatmap.png",
        "provider_status_by_family.png",
        "extension_lifecycle.png",
        "ir_analysis_levels.png",
        "blocked_reason_breakdown.png",
    }
    actual = {p.name for p in figures.glob("*.png")}
    # At least 3 of 5 must be real PNGs (some may be skipped if data
    # is missing on disk at audit time).
    real = expected & actual
    assert len(real) >= 3, f"only {real} are real PNGs in {figures}"
    for name in real:
        p = figures / name
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert p.stat().st_size > 1024


def test_render_replaces_markdown_placeholders(tmp_path: Path):
    """A pack with stale markdown placeholders gets cleaned up."""
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "provider_target_heatmap.md").write_text("stale placeholder")
    _make_provider_status(tmp_path)
    render_all_figures(tmp_path)
    # The .md was deleted.
    assert not (figures / "provider_target_heatmap.md").exists()
    # The .png was rendered.
    assert (figures / "provider_target_heatmap.png").is_file()
