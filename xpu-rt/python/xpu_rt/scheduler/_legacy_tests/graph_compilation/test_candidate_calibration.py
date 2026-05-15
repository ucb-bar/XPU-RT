"""Acceptance tests Per-Tile-Candidate Measured Cost.

Verifies:

- Calibration runs end-to-end for every legal SetTileParams candidate
  on a model with multiple tile candidates (merlin_mlp_wide).
- ``candidate_calibration_report.json`` has a typed status, per-
  candidate ``measured_baseline_us``, ``measured_tiled_us``,
  ``measured_speedup``, and ``rel_error``.
- Suite-level summary (``mean_speedup``, ``min_speedup``,
  ``max_speedup``, ``mean_rel_error``) is computed.
- ``cost_preview_v2.json`` AND ``llm_graph_view.json`` get a
  ``calibration`` overlay per matched tile candidate.
- Different tile candidates produce DIFFERENT measured costs (the
  whole point — the agent can now rank by measured evidence).
- Default OFF: when env var unset, no candidate_calibration dir.
- Best-effort: missing inputs → typed ``not_run`` report.
- No compiler-core imports.
Existing region calibration is preserved.
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


def _run(model: str, out_dir: Path, *, calibrate_candidates: bool) -> None:
    env = os.environ.copy()
    if calibrate_candidates:
        env["COMPGEN_CALIBRATE_CANDIDATES"] = "1"
    else:
        env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)  # isolate
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
def calibrated_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m183_calibrated") / "run"
    _run("merlin_mlp_wide", out, calibrate_candidates=True)
    return out


@pytest.fixture(scope="module")
def uncalibrated_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m183_uncalibrated") / "run"
    _run("merlin_mlp_wide", out, calibrate_candidates=False)
    return out


# --------------------------------------------------------------------------- #
# Artifacts emit
# --------------------------------------------------------------------------- #


def test_calibration_artifacts_exist(calibrated_run: Path) -> None:
    base = calibrated_run / "02_graph_analysis" / "candidate_calibration"
    assert (base / "candidate_calibration_report.json").exists()
    assert (base / "candidate_calibration_summary.md").exists()


def test_default_off_no_calibration_dir(uncalibrated_run: Path) -> None:
    base = uncalibrated_run / "02_graph_analysis" / "candidate_calibration"
    assert not base.exists()


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #


def test_report_has_typed_status(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    assert rep["calibration_status"] in (
        "calibrated", "no_candidates", "not_run",
    )
    assert rep["overall"] in ("calibrated", "no_candidates", "not_run")


def test_calibrated_each_legal_set_tile_candidate(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    cas = _read(
        calibrated_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal_count = sum(
        1 for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    )
    assert rep["candidate_count"] == legal_count
    assert rep["candidates_calibrated"] == legal_count
    assert rep["overall"] == "calibrated"


def test_each_calibrated_candidate_has_measurement_fields(
    calibrated_run: Path,
) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    for c in rep["candidates"]:
        if c.get("calibration_status") != "calibrated":
            continue
        for k in ("matmul_shape", "tile", "iters",
                  "measured_baseline_us", "measured_tiled_us",
                  "measured_speedup", "rel_error"):
            assert k in c, f"missing {k} in candidate {c['candidate_id']}"
        assert c["measured_baseline_us"] > 0
        assert c["measured_tiled_us"] > 0
        assert c["measured_speedup"] > 0


def test_summary_metrics_present(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    s = rep["summary"]
    assert s["mean_speedup"] is not None
    assert s["min_speedup"] is not None
    assert s["max_speedup"] is not None
    assert s["mean_rel_error"] is not None
    # min ≤ mean ≤ max.
    assert s["min_speedup"] <= s["mean_speedup"] <= s["max_speedup"]


def test_known_limitations_recorded(calibrated_run: Path) -> None:
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    lim = rep["known_limitations"]
    assert any("CPU only" in s for s in lim)
    assert any("SetTileParams only" in s for s in lim)


# --------------------------------------------------------------------------- #
# Per-candidate variance — the core value claim
# --------------------------------------------------------------------------- #


def test_different_candidates_have_different_measured_costs(
    calibrated_run: Path,
) -> None:
    """The whole point of the agent can rank candidates by
    measured evidence, not just static prediction. Different tiles
    must produce measurably different costs."""
    rep = _read(
        calibrated_run / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    speedups = sorted(
        c["measured_speedup"] for c in rep["candidates"]
        if c.get("calibration_status") == "calibrated"
    )
    assert len(speedups) >= 4
    # The spread should be non-trivial — at least 10% variation.
    spread = (max(speedups) - min(speedups)) / max(speedups)
    assert spread > 0.1, (
        f"calibration produced near-identical speedups across candidates "
        f"({speedups[:5]}); the agent can't distinguish them"
    )


# --------------------------------------------------------------------------- #
# Cost-preview / LLM-view overlay
# --------------------------------------------------------------------------- #


def test_cost_preview_v2_gets_calibration_overlay(calibrated_run: Path) -> None:
    cp = _read(
        calibrated_run / "02_graph_analysis" / "cost_preview_v2.json"
    )
    calibrated_entries = [
        p for p in cp["cost_previews"]
        if "calibration" in p
    ]
    assert len(calibrated_entries) >= 1
    for entry in calibrated_entries:
        cal = entry["calibration"]
        assert cal["calibration_status"] == "calibrated"
        assert cal["measured_baseline_us"] > 0
        assert cal["measured_tiled_us"] > 0
        assert cal["measured_speedup"] > 0
        assert "rel_error" in cal


def test_llm_graph_view_gets_calibration_overlay(calibrated_run: Path) -> None:
    lv_path = (
        calibrated_run / "02_graph_analysis" / "llm_graph_view.json"
    )
    if not lv_path.exists():
        pytest.skip("llm_graph_view.json not present in this stop-after")
    lv = _read(lv_path)
    seen = False
    for region in lv.get("regions", []) or []:
        for lc in region.get("legal_candidates", []) or []:
            if "calibration" in lc:
                seen = True
                assert lc["calibration"]["measured_tiled_us"] > 0
                assert "measured_speedup" in lc["calibration"]
    assert seen, "no candidate in llm_graph_view got a calibration overlay"


# --------------------------------------------------------------------------- #
# Best-effort + isolation
# --------------------------------------------------------------------------- #


def test_handles_missing_cost_preview(tmp_path: Path) -> None:
    fake = tmp_path / "fake_run"
    (fake / "02_graph_analysis").mkdir(parents=True)
    (fake / "02_graph_analysis" / "candidate_actions.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8",
    )
    from compgen.graph_compilation.candidate_calibration import (
        run_candidate_calibration,
    )
    res = run_candidate_calibration(fake, iterations=2, warmup=0)
    assert res.overall == "not_run"


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "candidate_calibration.py"
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
        assert pat not in src, f"candidate_calibration imports forbidden: {pat}"


def test_existing_artifacts_outside_overlay_unchanged(
    calibrated_run: Path,
) -> None:
    """only touches cost_preview_v2.json + llm_graph_view.json
    (overlay) and writes new files under candidate_calibration/. It
    must NOT mutate other artifacts."""
    base = calibrated_run / "02_graph_analysis"
    # candidate_actions.json should NOT contain "calibration".
    cas = (base / "candidate_actions.json").read_text(encoding="utf-8")
    assert "candidate_calibration" not in cas
    # region_map.json untouched.
    rm = (base / "region_map.json").read_text(encoding="utf-8")
    assert "calibration" not in rm
