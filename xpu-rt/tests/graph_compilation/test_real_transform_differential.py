"""Tests for M-12 Real Transform Differential Harness.

Path A (executable real transform) is exercised on ``merlin_mlp_wide``;
Path B (blocked) is exercised on every other wide-suite model. Negative
tests run against a copy of a real run dir.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compgen.graph_compilation.real_transform_differential import (
    _generate_cases,
    _tiled_matmul_eval,
    run_real_transform_differential,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WIDE = REPO_ROOT / "results" / "graph_compilation" / "real_transform_differential_suite"


def _need_wide() -> None:
    if not WIDE.is_dir():
        pytest.skip(
            f"fixture suite missing: {WIDE}; run "
            f"`compgen.graph_compilation run-suite --stop-after "
            f"real-transform-differential` first"
        )


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Path A: merlin_mlp_wide passes bit-equality
# --------------------------------------------------------------------------- #


def test_merlin_mlp_wide_path_a_passes_bit_equality() -> None:
    _need_wide()
    rv = WIDE / "merlin_mlp_wide" / "03_recipe_planning" / "real_verification"
    rep = _read(rv / "real_differential_report.json")
    assert rep["status"] == "pass"
    assert rep["mode"] == "executable_real_transform"
    assert rep["cases"]["total"] == 16
    assert rep["cases"]["passed"] == 16
    assert rep["cases"]["failed"] == 0
    assert rep["cases"]["frozen_cases"] == 8
    assert rep["cases"]["generated_cases"] == 8
    assert rep["error"]["max_abs_error"] == 0.0
    assert rep["error"]["max_rel_error"] == 0.0
    assert rep["error"]["refinement_status"] == "discharged_bit_equality"


def test_merlin_mlp_wide_obligation_is_discharged() -> None:
    _need_wide()
    rv = WIDE / "merlin_mlp_wide" / "03_recipe_planning" / "real_verification"
    obs = _read(rv / "real_obligation_status.json")
    assert obs["status"] == "pass"
    assert obs["obligations"][0]["status"] == (
        "discharged_real_transform_differential_check"
    )
    assert obs["obligations"][0]["declared_refinement"] == "bit_equality"
    assert "real_transform_differential_check" in obs["obligations"][0]["discharged"]
    assert obs["obligations"][0]["remaining"] == []


def test_merlin_mlp_wide_emits_all_case_files() -> None:
    _need_wide()
    rv = WIDE / "merlin_mlp_wide" / "03_recipe_planning" / "real_verification"
    n_inputs = len(list((rv / "input_cases").glob("*.pt")))
    n_orig = len(list((rv / "original_outputs").glob("*.pt")))
    n_xform = len(list((rv / "transformed_outputs").glob("*.pt")))
    n_counter = len(list((rv / "counterexamples").glob("*.pt")))
    assert n_inputs == 16
    assert n_orig == 16
    assert n_xform == 16
    assert n_counter == 0  # empty on pass


def test_summary_md_emitted() -> None:
    _need_wide()
    rv = WIDE / "merlin_mlp_wide" / "03_recipe_planning" / "real_verification"
    md = (rv / "real_differential_summary.md").read_text(encoding="utf-8")
    assert "Real Transform Differential" in md
    assert "discharged_bit_equality" in md
    assert "merlin_mlp_wide" in md


# --------------------------------------------------------------------------- #
# Path B: all other models are honestly blocked
# --------------------------------------------------------------------------- #


_BLOCKED_MODELS = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
    "graph_break_mlp", "residual_branch",
    "proxy_qwen_vl", "proxy_llava", "proxy_openvla",
    "proxy_diffusion_vla", "proxy_ocr",
    "merlin_mlp", "merlin_dronet",
)


@pytest.mark.parametrize("model", _BLOCKED_MODELS)
def test_blocked_path_emits_honest_report(model: str) -> None:
    _need_wide()
    rv = WIDE / model / "03_recipe_planning" / "real_verification"
    rep = _read(rv / "real_differential_report.json")
    assert rep["status"] == "blocked"
    assert rep["mode"] == "blocked"
    assert rep["cases"]["total"] == 0
    assert rep["error"]["max_abs_error"] is None
    assert rep["blocked_reason"]
    obs = _read(rv / "real_obligation_status.json")
    assert obs["status"] == "blocked"
    # Obligation must remain pending real_transform_differential_check.
    for ob in obs["obligations"]:
        assert "real_transform_differential_check" in ob["remaining"]
        assert ob["status"] == "remaining"


# --------------------------------------------------------------------------- #
# Tiled evaluator unit tests
# --------------------------------------------------------------------------- #


def test_tiled_evaluator_matches_torch_matmul_for_clean_dims() -> None:
    import torch

    torch.manual_seed(42)
    A = torch.randn(16, 16, dtype=torch.float32)
    B = torch.randn(16, 32, dtype=torch.float32)
    ref = torch.matmul(A, B)
    tiled = _tiled_matmul_eval(A, B, tile_M=16, tile_N=16, tile_K=16)
    assert torch.equal(ref, tiled)


def test_tiled_evaluator_handles_boundary_tiles_under_m16() -> None:
    """M-16: the evaluator no longer asserts divisibility — boundary
    tiles are computed via min()-based slicing. With K=16, tile_K=16 →
    K_iters=1, so accumulation order matches eager; bit-equality
    holds even with M=15 (boundary in M)."""
    import torch

    torch.manual_seed(7)
    A = torch.randn(15, 16, dtype=torch.float32)
    B = torch.randn(16, 16, dtype=torch.float32)
    ref = torch.matmul(A, B)
    tiled = _tiled_matmul_eval(A, B, tile_M=16, tile_N=16, tile_K=16)
    # Single K iteration means accumulation order matches eager → exact.
    assert torch.equal(ref, tiled)


def test_tiled_evaluator_rejects_non_positive_tile_dim() -> None:
    """Non-positive tile dims still raise (caller error)."""
    import torch

    A = torch.randn(16, 16, dtype=torch.float32)
    B = torch.randn(16, 16, dtype=torch.float32)
    # tile_M=0 means range(0, 16, 0) which raises ValueError.
    with pytest.raises((AssertionError, ValueError)):
        _tiled_matmul_eval(A, B, tile_M=0, tile_N=16, tile_K=16)


def test_summarise_boundary_geometry_counts_full_vs_boundary() -> None:
    """The boundary-handling block in M-12 reports must accurately
    count full-tile vs boundary-tile iterations."""
    from compgen.graph_compilation.real_transform_differential import (
        _summarise_boundary_geometry,
    )

    # Clean divides: all full, no boundary.
    s = _summarise_boundary_geometry(
        M=16, N=32, K=16, tile_M=16, tile_N=16, tile_K=16,
    )
    assert s["full_tiles_seen"] == 1 * 2 * 1  # 2
    assert s["boundary_tiles_seen"] == 0
    assert s["boundary_required"] is False

    # tiny_attention shape with tile_32: M=8, N=96, K=32, tile=32.
    # Iters: M=1 (tm=8 boundary), N=3 (3×32 full), K=1 (tk=32 full).
    # Each (i,j,k) tile has M-boundary so all are "boundary tiles".
    s = _summarise_boundary_geometry(
        M=8, N=96, K=32, tile_M=32, tile_N=32, tile_K=32,
    )
    assert s["full_tiles_seen"] == 0
    assert s["boundary_tiles_seen"] == 3
    assert s["boundary_required"] is True


def test_generate_cases_returns_16_cases() -> None:
    cases = _generate_cases(M=16, N=32, K=16)
    assert len(cases) == 16
    case_ids = {c[0] for c in cases}
    assert len(case_ids) == 16  # unique
    for case_id, A, B in cases:
        assert A.shape == (16, 16)
        assert B.shape == (16, 32)
        assert A.dtype == B.dtype  # f32


# --------------------------------------------------------------------------- #
# Negative tests (mutate copy + re-invoke harness)
# --------------------------------------------------------------------------- #


@pytest.fixture
def merlin_mlp_wide_run(tmp_path: Path) -> Path:
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    dst = tmp_path / "merlin_mlp_wide"
    shutil.copytree(src, dst)
    return dst


def test_missing_real_transform_manifest_blocks(tmp_path: Path) -> None:
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    dst = tmp_path / "merlin_mlp_wide"
    shutil.copytree(src, dst)
    (dst / "03_recipe_planning" / "real_lowering" / "real_transform_manifest.json").unlink()
    result = run_real_transform_differential(dst)
    assert result.overall == "blocked"
    rep = _read(result.report_path)
    assert rep["mode"] == "blocked"
    assert "missing M-11B manifest" in rep["blocked_reason"]


def test_non_executable_kind_produces_blocked_not_pass(
    merlin_mlp_wide_run: Path,
) -> None:
    mp = (
        merlin_mlp_wide_run / "03_recipe_planning"
        / "real_lowering" / "real_transform_manifest.json"
    )
    m = _read(mp)
    m["real_transform_kind"] = "non_executable_structural_ir"
    mp.write_text(json.dumps(m), encoding="utf-8")
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "blocked"
    rep = _read(result.report_path)
    assert "non_executable_structural_ir" in rep["blocked_reason"]


def test_boundary_required_no_longer_blocks_after_m16(
    merlin_mlp_wide_run: Path,
) -> None:
    """M-16 reframed: ``boundary_required=true`` no longer blocks when
    the kind is ``executable_with_boundary_handling``. The boundary-
    aware Python evaluator handles partial tiles via ``min()``-based
    slicing, so M-12 runs Path A and the run is mode=
    ``executable_real_transform`` — not blocked."""
    mp = (
        merlin_mlp_wide_run / "03_recipe_planning"
        / "real_lowering" / "real_transform_manifest.json"
    )
    m = _read(mp)
    m["tile_classification"]["boundary_required"] = True
    m["real_transform_kind"] = "executable_with_boundary_handling"
    mp.write_text(json.dumps(m), encoding="utf-8")
    result = run_real_transform_differential(merlin_mlp_wide_run)
    # Not blocked anymore.
    assert result.mode == "executable_real_transform"
    # For merlin_mlp_wide's tile (16,16,16) on matmul (16,32,16), the
    # K loop has exactly 1 iteration → bit-equality preserved.
    rep = _read(result.report_path)
    assert rep["transform"]["real_transform_kind"] == (
        "executable_with_boundary_handling"
    )
    assert "boundary_handling" in rep
    assert rep["boundary_handling"]["enabled"] is True


def test_non_positive_tile_dim_is_blocked(merlin_mlp_wide_run: Path) -> None:
    """M-16 reframed: arbitrary non-divisor tiles are no longer blocked
    (the boundary-aware evaluator handles them). Only pathological
    non-positive dims still block."""
    mp = (
        merlin_mlp_wide_run / "03_recipe_planning"
        / "real_lowering" / "real_transform_manifest.json"
    )
    m = _read(mp)
    m["selected_recipe"]["tile"] = {"M": 0, "N": 11, "K": 13}
    mp.write_text(json.dumps(m), encoding="utf-8")
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "blocked"
    rep = _read(result.report_path)
    # Accept either the legacy phrase or the current ``tile.<axis>
    # missing/zero`` phrasing — the implementation tightened the
    # message to name the offending axis. Both still mean the same
    # block reason: a non-positive tile dim.
    blocked_reason = rep["blocked_reason"]
    assert (
        "tile dimension non-positive" in blocked_reason
        or "missing/zero" in blocked_reason
    ), blocked_reason


def test_corrupted_evaluator_produces_counterexample(
    merlin_mlp_wide_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject an evaluator that adds a constant +1.0 — reference and
    transformed outputs disagree, M-12 emits counterexamples for every
    case and fails."""
    from compgen.graph_compilation import real_transform_differential as mod

    def bad_eval(A, B, *, tile_M, tile_N, tile_K):
        # Deliberately wrong: produce A@B + 1.0 (off by a constant).
        import torch
        return torch.matmul(A, B) + 1.0

    monkeypatch.setattr(mod, "_tiled_matmul_eval", bad_eval)
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "fail"
    assert len(result.counterexamples) >= 1
    rep = _read(result.report_path)
    assert rep["error"]["max_abs_error"] > 0
    # Counterexample files exist on disk.
    counter_dir = result.out_dir / "counterexamples"
    assert any(counter_dir.iterdir())


def test_shape_mismatch_blocks(merlin_mlp_wide_run: Path) -> None:
    mp = (
        merlin_mlp_wide_run / "03_recipe_planning"
        / "real_lowering" / "real_transform_manifest.json"
    )
    m = _read(mp)
    m["matmul_signature"]["lhs_dtype"] = "f64"  # not f32
    mp.write_text(json.dumps(m), encoding="utf-8")
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "blocked"
    rep = _read(result.report_path)
    assert "non-f32" in rep["blocked_reason"]


def test_non_set_tile_params_recipe_blocks(merlin_mlp_wide_run: Path) -> None:
    mp = (
        merlin_mlp_wide_run / "03_recipe_planning"
        / "real_lowering" / "real_transform_manifest.json"
    )
    m = _read(mp)
    m["selected_recipe"]["recipe_kind"] = "FuseProducerConsumer"
    mp.write_text(json.dumps(m), encoding="utf-8")
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "blocked"
    rep = _read(result.report_path)
    assert "recipe_kind=" in rep["blocked_reason"]
    # Obligation must remain remaining; not falsely failed.
    obs = _read(result.obligation_status_path)
    assert obs["status"] == "blocked"


def test_report_claiming_discharge_without_cases_is_overwritten(
    merlin_mlp_wide_run: Path,
) -> None:
    """Plant a forged report claiming discharge with 0 cases. M-12
    re-run must overwrite with the honest counts."""
    rv = merlin_mlp_wide_run / "03_recipe_planning" / "real_verification"
    rv.mkdir(parents=True, exist_ok=True)
    (rv / "real_differential_report.json").write_text(
        json.dumps({
            "schema_version": "real_differential_report_v1",
            "status": "pass",
            "mode": "executable_real_transform",
            "cases": {"total": 0, "passed": 0, "failed": 0,
                      "frozen_cases": 0, "generated_cases": 0},
            "error": {"max_abs_error": 0.0, "max_rel_error": 0.0,
                      "refinement_status": "discharged_bit_equality"},
        }), encoding="utf-8",
    )
    result = run_real_transform_differential(merlin_mlp_wide_run)
    rep = _read(result.report_path)
    # Honest re-emission: 16 cases were actually run.
    assert rep["cases"]["total"] == 16
    assert rep["status"] == "pass"


def test_source_payload_mutation_fails(
    merlin_mlp_wide_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compgen.graph_compilation import real_transform_differential as mod

    pre_count = sum(
        1 for _ in (merlin_mlp_wide_run / "01_payload_lowering").rglob("payload.mlir")
    )
    state = {"call": 0}
    real_sha = mod._sha256_file

    def fake_sha(path: Path) -> str:
        state["call"] += 1
        if state["call"] <= pre_count:
            return real_sha(path)
        return "sha256:" + "deadbeef" * 8

    monkeypatch.setattr(mod, "_sha256_file", fake_sha)
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "fail"
    assert any("source payload mutated" in f for f in result.failures)


def test_counterexample_files_emitted_on_mismatch(
    merlin_mlp_wide_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: with a broken evaluator, counterexamples/ MUST contain
    real .pt files (not just report metadata)."""
    from compgen.graph_compilation import real_transform_differential as mod

    def bad_eval(A, B, *, tile_M, tile_N, tile_K):
        import torch
        return torch.matmul(A, B) * 2.0  # deliberate scale error

    monkeypatch.setattr(mod, "_tiled_matmul_eval", bad_eval)
    result = run_real_transform_differential(merlin_mlp_wide_run)
    assert result.overall == "fail"
    counter_pt = list((result.out_dir / "counterexamples").glob("*.pt"))
    assert len(counter_pt) >= 1
    # Each counterexample file is loadable + contains the expected keys.
    import torch
    body = torch.load(counter_pt[0], weights_only=False)
    assert "A" in body and "B" in body and "expected" in body and "actual" in body
