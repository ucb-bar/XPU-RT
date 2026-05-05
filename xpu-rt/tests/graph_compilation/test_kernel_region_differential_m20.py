"""Acceptance tests for M-20 Per-Region Compiled Differential.

Verifies:

- Default OFF (no env var): no region-level artifact emitted.
- With ``COMPGEN_RUN_KERNELS=1`` on a multi-matmul model
  (merlin_mlp_wide has 3 matmul regions): the report covers ALL of
  them, each with a per-region GPU + CPU track.
- Each region's tile is the lowest-cost legal SetTileParams
  candidate (greedy per-region).
- Aggregate counts in the top-level summary match the per-region
  results.
- Per-region directories under ``kernel_execution/regions/`` contain
  their own ``compiled_kernel_run_gpu.json`` + ``compiled_kernel_run_cpu.json``.
- M-15B detector includes ``compiled_kernel_differential_check`` so
  kernel-level fails would trigger retry.
- M-19's single-region artifact is preserved (M-20 layers alongside).
- Existing FX-level reports are unchanged (regression invariant).
- Models with no SetTileParams candidates (e.g. proxy_vla → fusion)
  emit a typed ``no_candidates`` report.
- No compiler-core imports.
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


def _run(model: str, out_dir: Path, *, run_kernels: bool) -> None:
    env = os.environ.copy()
    if run_kernels:
        env["COMPGEN_RUN_KERNELS"] = "1"
    else:
        env.pop("COMPGEN_RUN_KERNELS", None)
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)
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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m20_run") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m20_off") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


@pytest.fixture(scope="module")
def fusion_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """proxy_vla greedy picks fusion; no SetTileParams selected.
    Region-level fan-out should still find SetTileParams candidates
    on matmul regions and run them. (Only the SELECTED candidate is
    fusion; the other matmul regions still have legal SetTileParams.)"""
    out = tmp_path_factory.mktemp("m20_fusion") / "run"
    _run("proxy_vla", out, run_kernels=True)
    return out


# --------------------------------------------------------------------------- #
# Default OFF
# --------------------------------------------------------------------------- #


def test_default_off_no_region_report(no_kernels_run: Path) -> None:
    base = no_kernels_run / "02_graph_analysis" / "kernel_execution"
    assert not base.exists()


# --------------------------------------------------------------------------- #
# Multi-region fan-out
# --------------------------------------------------------------------------- #


def test_region_report_covers_all_set_tile_regions(kernels_run: Path) -> None:
    """merlin_mlp_wide has 3 matmul regions, each with legal
    SetTileParams candidates. M-20 must cover all 3."""
    rep = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    assert rep["status"] in ("pass", "fail"), (
        f"unexpected status: {rep['status']}"
    )
    region_ids = {r["region_id"] for r in rep["regions"]}
    expected = {"matmul_0", "matmul_1", "matmul_2"}
    assert region_ids == expected, f"region drift: {region_ids ^ expected}"


def test_each_region_has_gpu_and_cpu_tracks(kernels_run: Path) -> None:
    rep = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    for region in rep["regions"]:
        assert "gpu" in region and isinstance(region["gpu"], dict)
        assert "cpu" in region and isinstance(region["cpu"], dict)
        for track_name in ("gpu", "cpu"):
            t = region[track_name]
            assert "compile_status" in t
            assert "run_status" in t
            assert "numerical" in t


def test_per_region_subdirectories_have_kernel_artifacts(
    kernels_run: Path,
) -> None:
    """Each region must have its own subdir containing
    compiled_kernel_run_*.json plus the emitted source files."""
    base = kernels_run / "02_graph_analysis" / "kernel_execution" / "regions"
    assert base.is_dir()
    region_dirs = [p for p in base.iterdir() if p.is_dir()]
    assert len(region_dirs) >= 3
    for d in region_dirs:
        assert (d / "compiled_kernel_run_gpu.json").exists()
        assert (d / "compiled_kernel_run_cpu.json").exists()


# --------------------------------------------------------------------------- #
# Greedy per-region tile selection
# --------------------------------------------------------------------------- #


def test_per_region_greedy_picks_lowest_cost(kernels_run: Path) -> None:
    """For each region, M-20 must pick the lowest static_relative_cost
    legal SetTileParams candidate."""
    rep = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    cas = _read(
        kernels_run / "02_graph_analysis" / "candidate_actions.json"
    )
    by_region: dict[str, float] = {}
    for c in cas["candidates"]:
        if c.get("kind") != "set_tile_params":
            continue
        if not (c.get("legality") or {}).get("ok"):
            continue
        rid = c.get("region_id", "")
        cost = float(
            (c.get("cost_preview") or {}).get("static_relative_cost", 1.0),
        )
        by_region[rid] = min(by_region.get(rid, float("inf")), cost)
    for r in rep["regions"]:
        assert r["static_relative_cost"] == by_region[r["region_id"]], (
            f"{r['region_id']} picked cost {r['static_relative_cost']} "
            f"but minimum was {by_region[r['region_id']]}"
        )


# --------------------------------------------------------------------------- #
# Aggregates match per-region results
# --------------------------------------------------------------------------- #


def test_aggregate_counts_match_per_region(kernels_run: Path) -> None:
    rep = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    s = rep["summary"]
    n = len(rep["regions"])
    assert s["region_count"] == n
    expected_gpu = sum(
        1 for r in rep["regions"]
        if (r["gpu"] or {}).get("compile_status") == "compiled"
    )
    expected_cpu = sum(
        1 for r in rep["regions"]
        if (r["cpu"] or {}).get("compile_status") == "compiled"
    )
    assert s["gpu_compiled_count"] == expected_gpu
    assert s["cpu_compiled_count"] == expected_cpu


def test_summary_classification_counts_consistent(kernels_run: Path) -> None:
    rep = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    s = rep["summary"]
    breakdown = s["refinement_breakdown"]
    bit_eq = sum(
        v for k, v in breakdown.items()
        if k.endswith("discharged_compiled_bit_equality")
    )
    tol = sum(
        v for k, v in breakdown.items()
        if k.endswith("discharged_tolerance_eps")
    )
    fail = sum(
        v for k, v in breakdown.items()
        if k.endswith("fail_outside_tolerance")
    )
    assert s["compiled_bit_equality_count"] == bit_eq
    assert s["tolerance_eps_count"] == tol
    assert s["fail_outside_tolerance_count"] == fail


# --------------------------------------------------------------------------- #
# M-15B detector wiring
# --------------------------------------------------------------------------- #


def test_m15b_detector_includes_compiled_kernel_check() -> None:
    """The downstream-retry detector must include the M-20 report so
    kernel-level fails would trigger retry."""
    from compgen.graph_compilation.downstream_retry import _DOWNSTREAM_REPORTS

    stages = {entry[0] for entry in _DOWNSTREAM_REPORTS}
    assert "region_compiled_differential" in stages
    by_stage = {entry[0]: entry for entry in _DOWNSTREAM_REPORTS}
    entry = by_stage["region_compiled_differential"]
    assert entry[1] == (
        "02_graph_analysis/kernel_execution/region_compiled_differential_report.json"
    )
    assert entry[3] == "compiled_kernel_differential_check"


# --------------------------------------------------------------------------- #
# M-19 single-region artifact preserved
# --------------------------------------------------------------------------- #


def test_m19_single_region_artifacts_still_emit(kernels_run: Path) -> None:
    """M-20 layers ALONGSIDE M-19; both artifact sets exist."""
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    # M-19 single-region (the SELECTED candidate's compiled run).
    assert (base / "compiled_kernel_run_gpu.json").exists()
    assert (base / "compiled_kernel_run_cpu.json").exists()
    # M-20 region fan-out.
    assert (base / "region_compiled_differential_report.json").exists()
    assert (base / "regions").is_dir()


# --------------------------------------------------------------------------- #
# FX-level invariants preserved
# --------------------------------------------------------------------------- #


def test_fx_artifacts_unchanged_when_m20_reruns(no_kernels_run: Path) -> None:
    """Calling run_region_compiled_differential on an existing run dir
    must not mutate any FX-level artifact."""
    import hashlib

    protected = [
        "03_recipe_planning/real_verification/real_differential_report.json",
        "02_graph_analysis/cost_preview_v2.json",
        "02_graph_analysis/region_map.json",
        "02_graph_analysis/candidate_actions.json",
    ]

    def _sha(p: Path) -> str:
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    before = {
        rel: _sha(no_kernels_run / rel)
        for rel in protected
        if (no_kernels_run / rel).exists()
    }
    assert before

    from compgen.graph_compilation.kernel_region_differential import (
        run_region_compiled_differential,
    )
    run_region_compiled_differential(no_kernels_run)

    after = {
        rel: _sha(no_kernels_run / rel)
        for rel in protected
        if (no_kernels_run / rel).exists()
    }
    assert before == after, (
        "M-20 mutated FX-level reports: "
        f"{[k for k in before if before[k] != after.get(k)]}"
    )


# --------------------------------------------------------------------------- #
# Fusion run still finds matmul SetTileParams candidates
# --------------------------------------------------------------------------- #


def test_fusion_run_still_runs_on_matmul_regions(fusion_run: Path) -> None:
    """proxy_vla's SELECTED candidate is fusion, but its matmul regions
    still have legal SetTileParams candidates. M-20 must run on them."""
    rep = _read(
        fusion_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    assert rep["overall"] in ("pass", "fail"), rep["overall"]
    region_ids = {r["region_id"] for r in rep["regions"]}
    # proxy_vla has matmul_0, matmul_1, matmul_2 with legal set_tile_params.
    assert "matmul_0" in region_ids


# --------------------------------------------------------------------------- #
# Best-effort
# --------------------------------------------------------------------------- #


def test_handles_missing_inputs(tmp_path: Path) -> None:
    """If candidate_actions or cost_preview_v2 is missing, M-20 emits
    a typed not_run report."""
    fake = tmp_path / "fake_run"
    (fake / "02_graph_analysis").mkdir(parents=True)
    (fake / "00_graph_capture").mkdir(parents=True)

    from compgen.graph_compilation.kernel_region_differential import (
        run_region_compiled_differential,
    )
    res = run_region_compiled_differential(fake)
    assert res.overall == "not_run"


def test_handles_no_set_tile_candidates(tmp_path: Path) -> None:
    """If a model has no legal SetTileParams candidates at all, M-20
    emits no_candidates."""
    fake = tmp_path / "fake_no_tiles"
    ga = fake / "02_graph_analysis"
    ga.mkdir(parents=True)
    (fake / "00_graph_capture").mkdir(parents=True)

    (ga / "candidate_actions.json").write_text(
        json.dumps({"candidates": [
            {"candidate_id": "x", "kind": "create_kernel_contract",
             "legality": {"ok": True}, "region_id": "x_region"}
        ]}),
        encoding="utf-8",
    )
    (ga / "cost_preview_v2.json").write_text(
        json.dumps({"cost_previews": []}), encoding="utf-8",
    )

    from compgen.graph_compilation.kernel_region_differential import (
        run_region_compiled_differential,
    )
    res = run_region_compiled_differential(fake)
    assert res.overall == "no_candidates"


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "kernel_region_differential.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir.payload",
        "import compgen.ir.payload",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"M-20 imports forbidden: {pat}"
