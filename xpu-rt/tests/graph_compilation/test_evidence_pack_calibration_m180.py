"""M-18.0 closure tests for the evidence pack.

Verifies the M-17 evidence pack ingests M-18 calibration correctly:

- New CSV columns (calibration_status, calibration_overall,
  calibration_matched_regions, calibration_total_regions,
  calibration_match_fraction, calibration_suite_scale,
  calibration_suite_mape) are populated when calibration ran.
- Aggregate counts (calibrated_model_count,
  calibration_partial_count, calibration_not_run_count,
  calibration_status_breakdown,
  calibration_mean_match_fraction,
  calibration_total_predicted_us, calibration_total_measured_us,
  calibration_suite_scale_summary) are present in
  graph_section_evidence_tables.json.
- Claim matrix has a "Profiler-calibrated cost preview" entry whose
  observed_metric matches the aggregates.
- Calibration figures (calibration_coverage_by_model.png,
  calibration_suite_scale_by_model.png) emit when at least one model
  has calibration evidence.
- Markdown summary contains the M-18 calibration section AND the
  M-18 limitations box.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path, *, calibrate: bool) -> None:
    env = os.environ.copy()
    if calibrate:
        env["COMPGEN_CALIBRATE_PROFILER"] = "1"
    else:
        env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )


@pytest.fixture(scope="module")
def calibrated_pack(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """Build a small fixture pack with calibration on for one model and
    off for one model, so we can assert both branches."""
    suite = tmp_path_factory.mktemp("m180_fixture") / "suite"
    canonical = suite / "canonical"
    wide = suite / "wide"
    canonical.mkdir(parents=True)
    wide.mkdir(parents=True)

    # canonical: proxy_vla calibrated.
    _run("proxy_vla", canonical / "proxy_vla", calibrate=True)
    # wide: merlin_mlp_wide calibrated.
    _run("merlin_mlp_wide", wide / "merlin_mlp_wide", calibrate=True)
    # wide: tiny_mlp uncalibrated (so calibration_not_run_count > 0).
    _run("tiny_mlp", wide / "tiny_mlp", calibrate=False)

    pack_out = suite / "evidence_pack"
    from compgen.graph_compilation.evidence_pack import build_evidence_pack
    build_evidence_pack(
        canonical_suite_root=canonical,
        wide_suite_root=wide,
        out_dir=pack_out,
    )
    return pack_out


# --------------------------------------------------------------------------- #
# CSV columns
# --------------------------------------------------------------------------- #


def test_model_matrix_has_calibration_columns(calibrated_pack: Path) -> None:
    rows = list(csv.DictReader(
        (calibrated_pack / "graph_section_model_matrix.csv").open(encoding="utf-8"),
    ))
    by_id = {(r["model_id"], r["suite"]): r for r in rows}

    pv = by_id[("proxy_vla", "canonical")]
    for col in (
        "calibration_status", "calibration_overall",
        "calibration_matched_regions", "calibration_total_regions",
        "calibration_match_fraction",
        "calibration_suite_scale", "calibration_suite_mape",
    ):
        assert col in pv, f"missing column {col}"
    assert pv["calibration_overall"] in ("calibrated", "partial")
    assert int(pv["calibration_matched_regions"]) >= 1
    assert int(pv["calibration_total_regions"]) >= 1
    assert float(pv["calibration_match_fraction"]) > 0
    assert float(pv["calibration_suite_scale"]) > 0


def test_uncalibrated_row_is_n_a(calibrated_pack: Path) -> None:
    rows = list(csv.DictReader(
        (calibrated_pack / "graph_section_model_matrix.csv").open(encoding="utf-8"),
    ))
    tm = next(
        r for r in rows
        if r["model_id"] == "tiny_mlp" and r["suite"] == "wide"
    )
    assert tm["calibration_overall"] == "n/a"
    assert tm["calibration_status"] == "n/a"
    assert int(tm["calibration_matched_regions"]) == 0


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def test_evidence_tables_have_calibration_aggregates(
    calibrated_pack: Path,
) -> None:
    agg = _read(calibrated_pack / "graph_section_evidence_tables.json")
    for k in (
        "calibrated_model_count", "calibration_partial_count",
        "calibration_not_run_count", "calibration_status_breakdown",
        "calibration_mean_match_fraction",
        "calibration_total_predicted_us", "calibration_total_measured_us",
        "calibration_suite_scale_summary",
    ):
        assert k in agg, f"missing aggregate {k}"

    # 2 calibrated runs (proxy_vla, merlin_mlp_wide) — exactly 1
    # uncalibrated (tiny_mlp).
    assert (agg["calibrated_model_count"]
            + agg["calibration_partial_count"]) >= 2
    assert agg["calibration_not_run_count"] >= 1
    assert agg["calibration_total_measured_us"] > 0
    assert isinstance(agg["calibration_suite_scale_summary"], list)
    assert len(agg["calibration_suite_scale_summary"]) >= 2


# --------------------------------------------------------------------------- #
# Claim matrix
# --------------------------------------------------------------------------- #


def test_claim_matrix_records_profiler_calibration(
    calibrated_pack: Path,
) -> None:
    cm = _read(calibrated_pack / "graph_section_claim_matrix.json")
    claim = next(
        c for c in cm["claims"]
        if "Profiler-calibrated cost preview" in c["claim"]
    )
    assert claim["status"] == "implemented"
    obs = claim["observed_metric"]
    assert (obs["calibrated_model_count"]
            + obs["calibration_partial_count"]) >= 2
    assert "calibration_status_breakdown" in obs


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def test_calibration_figures_emit_when_calibration_exists(
    calibrated_pack: Path,
) -> None:
    figs = calibrated_pack / "figures"
    expected = [
        "calibration_coverage_by_model.png",
        "calibration_suite_scale_by_model.png",
    ]
    for name in expected:
        path = figs / name
        assert path.exists(), f"missing M-18 figure {name}"
        with path.open("rb") as f:
            head = f.read(8)
        assert head == b"\x89PNG\r\n\x1a\n", f"{name} not a valid PNG"


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def test_summary_md_has_calibration_section(calibrated_pack: Path) -> None:
    text = (calibrated_pack / "graph_section_evidence_summary.md").read_text(
        encoding="utf-8",
    )
    assert "## M-18 profiler calibration" in text
    assert "calibrated models" in text
    assert "aggregate suite scale" in text
    assert "per-model calibration" in text


def test_summary_md_has_m18_limitations_box(calibrated_pack: Path) -> None:
    text = (calibrated_pack / "graph_section_evidence_summary.md").read_text(
        encoding="utf-8",
    )
    assert "M-18 calibration limitations" in text
    for must in (
        "CPU profiler activities only",
        "Fair-share attribution",
        "Single-batch-size",
        "No per-tile-candidate measured cost yet",
        "deterministic baseline is preserved verbatim",
    ):
        assert must in text, f"missing limitation phrase: {must!r}"


def test_summary_md_does_not_claim_cost_model_accurate(
    calibrated_pack: Path,
) -> None:
    """Per the M-18.0 closure note: do not say 'the cost model is
    accurate'. Say it's calibrated and exposes its bias."""
    text = (calibrated_pack / "graph_section_evidence_summary.md").read_text(
        encoding="utf-8",
    )
    forbidden = (
        "The cost model is accurate",
        "the cost model is accurate",
        "cost model accuracy",
    )
    for f in forbidden:
        assert f not in text, f"M-18.0 forbidden phrase present: {f!r}"
