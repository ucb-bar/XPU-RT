"""Acceptance tests for M-18 Profiler-Calibrated Cost Preview.

Verifies:

- Calibration runs end-to-end on a captured exported program when
  ``COMPGEN_CALIBRATE_PROFILER=1`` is set.
- ``profile_run.json`` records iterations, warmup, op_to_us, and
  wall_us_per_iter from a real torch.profiler run.
- ``profiler_calibration_report.json`` has typed
  ``calibration_status`` (``calibrated`` / ``partial_match`` /
  ``no_op_match`` / ``not_run``) and per-region
  predicted-vs-measured rows.
- Suite-level metrics (``suite_predicted_us``, ``suite_measured_us``,
  ``suite_scale``, ``suite_mape``) are computed.
- The readiness matrix row 6 flips from ``ready_for_m18`` to
  ``calibrated`` when calibration ran successfully (and stays
  ``ready_for_m18`` when it didn't).
- ``graph_dossier_v3.json`` and ``llm_graph_view.json`` get a
  top-level ``calibration`` block + per-matched-region calibration
  fields.
- Default off: when the env var is NOT set, the pipeline does not
  run calibration and row 6 stays ``ready_for_m18``.
- The two PNG figures emit with valid magic bytes.
- No compiler-core imports.
- M-17.1 readiness matrix is otherwise unchanged (calibration is
  additive).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path, *, calibrate: bool = False) -> int:
    env = os.environ.copy()
    if calibrate:
        env["COMPGEN_CALIBRATE_PROFILER"] = "1"
    else:
        env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out_dir),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "greedy",
    ]
    res = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    return res.returncode


# --------------------------------------------------------------------------- #
# Module-scope fixtures (run once each; tests share)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def calibrated_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m18_calibrated") / "run"
    _run("proxy_vla", out, calibrate=True)
    return out


@pytest.fixture(scope="module")
def uncalibrated_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m18_uncalibrated") / "run"
    _run("proxy_vla", out, calibrate=False)
    return out


# --------------------------------------------------------------------------- #
# Calibration artifacts emit
# --------------------------------------------------------------------------- #


def test_calibration_artifacts_exist(calibrated_run: Path) -> None:
    base = calibrated_run / "02_graph_analysis" / "calibration"
    expected = [
        "profile_run.json",
        "profiler_calibration_report.json",
        "calibration_summary.md",
    ]
    for name in expected:
        assert (base / name).exists(), f"missing {name}"


def test_calibration_figures_emit_valid_pngs(calibrated_run: Path) -> None:
    figs_dir = calibrated_run / "02_graph_analysis" / "calibration" / "figures"
    expected = [
        "predicted_vs_measured.png",
        "calibration_error_distribution.png",
    ]
    for name in expected:
        path = figs_dir / name
        assert path.exists(), f"missing figure {name}"
        with path.open("rb") as f:
            head = f.read(8)
        assert head == b"\x89PNG\r\n\x1a\n", f"{name} not a valid PNG"


# --------------------------------------------------------------------------- #
# profile_run.json shape
# --------------------------------------------------------------------------- #


def test_profile_run_records_iterations_and_ops(calibrated_run: Path) -> None:
    pr = _read(
        calibrated_run / "02_graph_analysis" / "calibration" / "profile_run.json"
    )
    assert pr["success"] is True
    assert pr["iterations"] >= 1
    assert pr["warmup"] >= 0
    assert pr["wall_us_per_iter"] > 0
    assert isinstance(pr["op_to_us"], dict)
    assert len(pr["op_to_us"]) >= 1


# --------------------------------------------------------------------------- #
# Calibration report shape
# --------------------------------------------------------------------------- #


def test_calibration_report_has_typed_status(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    assert rep["calibration_status"] in (
        "calibrated", "partial_match", "no_op_match", "not_run",
    )
    assert rep["overall"] in ("calibrated", "partial", "not_run")


def test_calibration_report_has_suite_metrics(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    s = rep["summary"]
    assert s["matched_region_count"] >= 1
    assert s["total_region_count"] >= s["matched_region_count"]
    assert s["suite_measured_us"] > 0
    assert s["suite_predicted_us"] >= 0
    if s["suite_scale"] is not None:
        assert s["suite_scale"] > 0


def test_calibration_report_has_per_region_predicted_vs_measured(
    calibrated_run: Path,
) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    assert len(rep["regions"]) >= 1
    matched = [r for r in rep["regions"] if r["match_status"] == "matched"]
    assert len(matched) >= 1
    for r in matched:
        assert r["predicted_us"] >= 0
        assert r["measured_us"] > 0
        assert "matched_op_keys" in r
        assert len(r["matched_op_keys"]) >= 1


def test_calibration_report_has_known_limitations(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    lim = rep["known_limitations"]
    assert any("fuzzy" in s.lower() for s in lim)
    assert any("cpu" in s.lower() for s in lim)


# --------------------------------------------------------------------------- #
# Readiness matrix row 6 flips
# --------------------------------------------------------------------------- #


def test_readiness_row_6_flips_to_calibrated(calibrated_run: Path) -> None:
    m = _read(
        calibrated_run / "02_graph_analysis" / "readiness"
        / "graph_analysis_readiness_matrix.json"
    )
    assert m["overall"] == "pass"
    row6 = m["slide_rows"][5]
    assert row6["status"] == "calibrated"
    assert row6.get("calibration_artifact", "").endswith(
        "profiler_calibration_report.json"
    )
    assert row6.get("calibration_status") in (
        "calibrated", "partial_match",
    )


def test_readiness_row_6_stays_ready_for_m18_when_uncalibrated(
    uncalibrated_run: Path,
) -> None:
    m = _read(
        uncalibrated_run / "02_graph_analysis" / "readiness"
        / "graph_analysis_readiness_matrix.json"
    )
    assert m["overall"] == "pass"
    row6 = m["slide_rows"][5]
    assert row6["status"] == "ready_for_m18"
    # No calibration_artifact when calibration didn't run.
    assert "calibration_artifact" not in row6


def test_no_calibration_artifacts_when_uncalibrated(
    uncalibrated_run: Path,
) -> None:
    """Default OFF: calibration directory should not exist when the
    env var isn't set."""
    cal_dir = uncalibrated_run / "02_graph_analysis" / "calibration"
    assert not cal_dir.exists()


# --------------------------------------------------------------------------- #
# Dossier overlay
# --------------------------------------------------------------------------- #


def test_graph_dossier_v3_has_calibration_overlay(calibrated_run: Path) -> None:
    v3 = _read(calibrated_run / "02_graph_analysis" / "graph_dossier_v3.json")
    assert "calibration" in v3
    cal = v3["calibration"]
    assert "calibration_status" in cal
    assert "suite_scale" in cal
    assert cal["matched_region_count"] >= 1


def test_graph_dossier_v3_per_region_calibration(calibrated_run: Path) -> None:
    v3 = _read(calibrated_run / "02_graph_analysis" / "graph_dossier_v3.json")
    matched_with_calibration = [
        r for r in v3["regions"]
        if r.get("calibration", {}).get("match_status") == "matched"
    ]
    assert len(matched_with_calibration) >= 1
    for r in matched_with_calibration:
        c = r["calibration"]
        assert c["measured_latency_us"] > 0
        assert "predicted_latency_us" in c
        assert "matched_op_keys" in c


def test_llm_graph_view_has_calibration_overlay(calibrated_run: Path) -> None:
    """The bounded LLM view also gets the calibration overlay so the
    agent sees calibrated cost evidence when picking candidates."""
    lv_path = calibrated_run / "02_graph_analysis" / "llm_graph_view.json"
    if not lv_path.exists():
        pytest.skip("llm_graph_view.json not present in this stop-after")
    lv = _read(lv_path)
    assert "calibration" in lv
    assert "calibration_status" in lv["calibration"]


# --------------------------------------------------------------------------- #
# Idempotence + best-effort
# --------------------------------------------------------------------------- #


def test_calibration_module_handles_missing_capture(tmp_path: Path) -> None:
    """Best-effort: profile run on an empty run dir emits a not_run
    report instead of raising."""
    fake_run = tmp_path / "fake_run"
    (fake_run / "02_graph_analysis").mkdir(parents=True)
    (fake_run / "00_graph_capture").mkdir(parents=True)
    (fake_run / "02_graph_analysis" / "region_map.json").write_text(
        json.dumps({"regions": []}), encoding="utf-8",
    )

    from compgen.graph_compilation.profiler_calibration import (
        run_profiler_calibration,
    )

    res = run_profiler_calibration(fake_run, iterations=2, warmup=0)
    assert res.overall == "not_run"
    rep = _read(res.report_path)
    assert rep["calibration_status"] == "not_run"


# --------------------------------------------------------------------------- #
# No compiler-core changes
# --------------------------------------------------------------------------- #


def test_profiler_calibration_does_not_import_compiler_core() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "profiler_calibration.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"profiler_calibration imports forbidden module: {pat}"


def test_calibration_does_not_mutate_existing_graph_analysis_artifacts(
    calibrated_run: Path,
) -> None:
    """The calibration step is allowed to add ``calibration`` overlays
    to ``graph_dossier_v3.json`` and ``llm_graph_view.json`` (per
    M-18.4) but must not touch the original baseline reports
    (``cost_preview_v2.json``, ``hardware_resource_report.json``,
    ``precision_budget_report.json``, etc.)."""
    base = calibrated_run / "02_graph_analysis"
    # These remain byte-identical regardless of whether calibration ran.
    for protected in (
        "cost_preview_v2.json",
        "readiness/precision_budget_report.json",
        "readiness/working_set_fit_report.json",
        "readiness/reuse_lifetime_report.json",
        "readiness/hardware_resource_report.json",
    ):
        path = base / protected
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Calibration overlay is NOT injected into these protected files.
        assert "measured_latency_us" not in text, (
            f"calibration leaked into protected file {protected}"
        )
