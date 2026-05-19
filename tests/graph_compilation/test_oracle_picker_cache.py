"""End-to-end test for the production cost-calibration cache.

The oracle picker auto-discovers `.xpu_rt_cache/cost_calibration/`
when no env var override is set. This test writes a synthetic
calibration to that path, runs greedy, and asserts the picker hits
the cached empirical-best instead of the static priority pick.

This is the closure smoke test for the
`cost_model_uncalibrated_across_decisions` caveat: it proves the
production cache is actually consulted on a fresh compile, with no
operator intervention beyond having dropped the JSON in place.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT


@pytest.fixture
def chdir_pkg_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Anchor cwd at xpu-rt/ — model_path is resolved relative to cwd.

    Also ensures no env-var override leaks in from the test env.
    """
    monkeypatch.delenv("XPU_RT_CALIBRATION_DIR", raising=False)
    monkeypatch.delenv("XPU_RT_SUBSYSTEM_MASK", raising=False)
    monkeypatch.chdir(PKG_ROOT)
    return PKG_ROOT


def _read_selection(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "03_recipe_planning" / "candidate_selection.json").read_text()
    )


def test_oracle_picks_from_xpu_rt_cache_default_location(
    chdir_pkg_root: Path, tmp_path: Path,
) -> None:
    """With no env var override but a `.xpu_rt_cache/cost_calibration/
    <model_id>.json` present, greedy must use it and pick the cached
    empirical best (not the static priority pick).
    """
    from xpu_rt.graph_compilation.run import run_graph_compilation

    model_id = "merlin_mlp_wide"
    model_cfg = chdir_pkg_root / "configs" / "models" / f"{model_id}.yaml"
    target_cfg = chdir_pkg_root / "configs" / "targets" / "host_cpu.yaml"

    # First pass: run greedy enumeration to discover real candidate IDs.
    enum_dir = tmp_path / "enum"
    run_graph_compilation(
        model_config_path=model_cfg,
        target_config_path=target_cfg,
        out_dir=enum_dir,
        stop_after="graph-analysis",
    )
    cas = json.loads(
        (enum_dir / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    # Pick two legal candidates: the one greedy would pick statically
    # (lowest static_relative_cost) and a different one to use as the
    # "empirical best." If they happen to be the same, the test still
    # validates the cache is consulted but the assertion below picks
    # any legal candidate that ISN'T the static one.
    legal = sorted(
        (c for c in cas["candidates"] if c["legality"]["ok"]),
        key=lambda c: (
            float(c["cost_preview"].get("static_relative_cost", 1.0)),
            c["candidate_id"],
        ),
    )
    assert len(legal) >= 2, "need at least 2 legal candidates for this test"
    static_pick = legal[0]
    # Pick a *different* candidate as the "calibrated best."
    forced_best = next(c for c in legal if c["candidate_id"] != static_pick["candidate_id"])

    # Drop a synthetic calibration into the cache. Only the forced
    # candidate gets an "ok"+"verified" entry, so the oracle picks it.
    cache_dir = chdir_pkg_root / ".xpu_rt_cache" / "cost_calibration"
    # Clean any pre-existing cache to avoid cross-test leakage.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / f"{model_id}.json"
    cache_file.write_text(json.dumps({
        "schema_version": "model_calibration_v1",
        "model_id": model_id,
        "target_id": "host_cpu",
        "measurements": [
            {
                "candidate_id": forced_best["candidate_id"],
                "candidate_kind": forced_best["kind"],
                "static_relative_cost": float(
                    forced_best["cost_preview"].get("static_relative_cost", 1.0)
                ),
                "typed_outcome": "verified",
                "latency_min_us": 50.0,  # very fast → will win the oracle
                "latency_median_us": 50.0,
                "latency_stddev_us": 1.0,
                "latency_status": "ok",
                "n_iters": 100,
                "run_dir": "",
                "error": "",
            },
        ],
    }))

    try:
        run_dir = tmp_path / "pick"
        run_graph_compilation(
            model_config_path=model_cfg,
            target_config_path=target_cfg,
            out_dir=run_dir,
            stop_after="recipe-planning",
            selection_mode="greedy",
        )
        sel = _read_selection(run_dir)
        assert sel["selected_candidate_id"] == forced_best["candidate_id"], (
            f"expected cached oracle to override static pick; "
            f"got {sel['selected_candidate_id']!r}, "
            f"expected {forced_best['candidate_id']!r} "
            f"(static would have picked {static_pick['candidate_id']!r})"
        )
        assert "calibrated_pick_override" in sel["rationale"]["primary_reason"], (
            f"selection rationale missing oracle marker: "
            f"{sel['rationale']['primary_reason']}"
        )
    finally:
        # Restore the cache directory to its original (absent) state so
        # subsequent tests don't accidentally inherit this fixture.
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def test_oracle_falls_back_to_static_when_cache_missing(
    chdir_pkg_root: Path, tmp_path: Path,
) -> None:
    """When neither the env var nor the cache has a calibration for
    this model, greedy uses the static priority — no oracle marker
    in the rationale.
    """
    from xpu_rt.graph_compilation.run import run_graph_compilation

    cache_dir = chdir_pkg_root / ".xpu_rt_cache" / "cost_calibration"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    model_cfg = chdir_pkg_root / "configs" / "models" / "tiny_mlp.yaml"
    target_cfg = chdir_pkg_root / "configs" / "targets" / "host_cpu.yaml"
    run_dir = tmp_path / "static_path"
    run_graph_compilation(
        model_config_path=model_cfg,
        target_config_path=target_cfg,
        out_dir=run_dir,
        stop_after="recipe-planning",
        selection_mode="greedy",
    )
    sel = _read_selection(run_dir)
    assert "calibrated_pick_override" not in sel["rationale"]["primary_reason"]
