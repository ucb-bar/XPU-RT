"""Tests for M-16 Real Transform Coverage Expansion.

Verifies that:

- The boundary-aware evaluator runs Path A (mode=executable_real_transform)
  for previously-blocked SetTileParams models — they no longer return
  ``mode=blocked``.
- The ``boundary_handling`` block accurately records full vs boundary
  tile counts.
- ``bit_equality`` is discharged ONLY when ``max_abs_error == 0`` and
  ``max_rel_error == 0`` (i.e. when accumulation order is preserved —
  typically when ``K_iters == 1``).
- A previously-blocked model (tiny_attention with tile_M32_N32_K32)
  passes Path A with discharged_bit_equality.
- M-15B downstream retry still fires when M-12 is forced to fail.
- Non-SetTileParams / non-f32 / pathological cases remain blocked
  with precise reasons.
- merlin_mlp_wide's clean-divides path is unchanged.
"""

from __future__ import annotations

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


def _invoke(
    *, model: str, out_dir: Path,
    selection_mode: str = "greedy",
    response_paths: list[Path] | None = None,
    env_overrides: dict | None = None,
    stop_after: str = "cost-preview-v2",
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out_dir),
        "--stop-after", stop_after,
        "--selection-mode", selection_mode,
    ]
    for p in response_paths or []:
        cmd += ["--agent-decision-response", str(p)]
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )


# --------------------------------------------------------------------------- #
# Path A: previously-blocked models now run the boundary-aware evaluator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", [
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
])
def test_previously_blocked_models_now_run_path_a(
    model: str, tmp_path: Path,
) -> None:
    """Pre-M-16: M-12 returned ``mode=blocked`` for these models.
    Post-M-16: M-12 reaches Path A (mode=executable_real_transform).

    M-37.11 evolution: greedy now derives shape-fit tile candidates
    and prefers clean-divide tiles over boundary-handled ones (the
    boundary-aware path is still emitted as a fallback when no clean
    tile fits). For these three models the shape-fit candidate
    (``tile_M*_N16_K16`` chosen to divide M and K cleanly) wins, so
    Path A runs as ``executable_structured_ir`` (the strictly-better
    refinement). The test still pins Path A; it accepts either
    real_transform_kind because the goal is "Path A reached", not
    "boundary path forced"."""
    out = tmp_path / model
    res = _invoke(model=model, out_dir=out)
    # Pipeline raises via M-15B if downstream fails. We don't care
    # about exit code for this test; we care that the M-12 report
    # exists and shows Path A was attempted.
    rep_path = (
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert rep_path.exists(), f"{model}: M-12 report missing"
    rep = _read(rep_path)
    assert rep["mode"] == "executable_real_transform", (
        f"{model}: expected Path A, got mode={rep.get('mode')}"
    )
    real_kind = rep["transform"]["real_transform_kind"]
    assert real_kind in (
        "executable_structured_ir",            # M-37.11 clean-divide path
        "executable_with_boundary_handling",   # legacy boundary path
    ), f"{model}: unexpected real_transform_kind={real_kind!r}"
    bh = rep["boundary_handling"]
    if real_kind == "executable_with_boundary_handling":
        assert bh["enabled"] is True
        assert bh["boundary_required"] is True
        assert bh["full_tiles_seen"] + bh["boundary_tiles_seen"] >= 1
    else:
        # Clean-divide path: boundary handling is not exercised.
        assert bh.get("boundary_required") is False


def test_merlin_mlp_wide_still_passes_clean_divides_path(
    tmp_path: Path,
) -> None:
    """The pre-M-16 happy path: clean divides → executable_structured_ir
    → bit-equality on 16/16 cases. M-16 must not regress this."""
    out = tmp_path / "merlin_mlp_wide"
    res = _invoke(model="merlin_mlp_wide", out_dir=out)
    assert res.returncode == 0
    rep = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert rep["status"] == "pass"
    assert rep["transform"]["real_transform_kind"] == (
        "executable_structured_ir"
    )
    assert rep["error"]["max_abs_error"] == 0.0
    assert rep["error"]["refinement_status"] == "discharged_bit_equality"
    assert rep["cases"]["passed"] == rep["cases"]["total"] == 16
    # Clean divides → 0 boundary tiles.
    assert rep["boundary_handling"]["boundary_tiles_seen"] == 0
    assert rep["boundary_handling"]["full_tiles_seen"] >= 1


# --------------------------------------------------------------------------- #
# Headline acceptance bar: previously-blocked model passes bit-equality
# --------------------------------------------------------------------------- #


def test_tiny_attention_passes_bit_equality_with_tile_32(
    tmp_path: Path,
) -> None:
    """The headline M-16 bar: a previously-blocked SetTileParams model
    passes Path A with discharged_bit_equality.

    For tiny_attention (M=8, N=96, K=32), greedy picks tile=16x16x16
    (K_iters=2, breaks bit-equality). With agent-file mode + tile=
    32x32x32, K_iters=1 → bit-equality preserved. The evaluator runs
    boundary-aware slicing for M (tm=8 < tile_M=32) and N (3×32, no
    boundary in N at boundary 96=3*32)."""
    # First run a probe to find the tile_32 candidate id. Stop at
    # graph-analysis so M-12 (which would fail on the greedy tile_16
    # pick and trigger M-15B) doesn't run during the probe.
    probe = tmp_path / "probe"
    res = _invoke(
        model="tiny_attention", out_dir=probe,
        stop_after="graph-analysis",
    )
    assert res.returncode == 0, res.stderr
    cas = _read(probe / "02_graph_analysis" / "candidate_actions.json")
    tile32_id = next(
        c["candidate_id"] for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and c.get("label") == "tile_M32_N32_K32"
    )

    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": tile32_id,
        "rationale": {
            "summary": (
                "M-16 demonstration: tile_M32_N32_K32 has K_iters=1 "
                "(tile_K=32=K) so accumulation order matches eager; "
                "bit-equality preserved while boundary handling fires "
                "for the M dim (M=8 < tile_M=32)."
            ),
            "evidence": [
                {"field": "candidate.kind", "value": "set_tile_params",
                 "reason": "Structured tiling on matmul region."},
                {"field": "candidate.label", "value": "tile_M32_N32_K32",
                 "reason": "tile_K equals K so K loop runs once."},
                {"field": "candidate.cost_preview.fits_l2", "value": True,
                 "reason": "Tile fits L2 working set."},
            ],
        },
    }
    response_path = tmp_path / "tile32_response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    out = tmp_path / "tiny_attention_pass"
    res2 = _invoke(
        model="tiny_attention", out_dir=out,
        selection_mode="agent-file",
        response_paths=[response_path],
    )
    assert res2.returncode == 0, res2.stderr
    rep = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert rep["status"] == "pass", (
        f"tiny_attention with tile_32 should discharge bit-equality "
        f"(K_iters=1); got {rep.get('error')}"
    )
    assert rep["transform"]["real_transform_kind"] == (
        "executable_with_boundary_handling"
    )
    assert rep["error"]["max_abs_error"] == 0.0
    assert rep["error"]["max_rel_error"] == 0.0
    assert rep["error"]["refinement_status"] == "discharged_bit_equality"
    assert rep["cases"]["passed"] == 16
    bh = rep["boundary_handling"]
    assert bh["enabled"] is True
    assert bh["boundary_required"] is True
    assert bh["iters_K"] == 1  # the key invariant for bit-equality
    # All tiles are M-boundary (tm=8 < tile_M=32); 3 N-iters × 1 M-iter
    # × 1 K-iter = 3 boundary tiles, 0 full.
    assert bh["boundary_tiles_seen"] == 3
    assert bh["full_tiles_seen"] == 0


# --------------------------------------------------------------------------- #
# Boundary handling counts in the report
# --------------------------------------------------------------------------- #


def test_boundary_handling_block_includes_iter_counts(
    tmp_path: Path,
) -> None:
    """The boundary_handling block in the report must include
    iters_M / iters_N / iters_K so reviewers can see the tile geometry."""
    out = tmp_path / "tiny_mlp_iters"
    _invoke(model="tiny_mlp", out_dir=out)
    rep = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    bh = rep["boundary_handling"]
    for key in ("iters_M", "iters_N", "iters_K", "full_tiles_seen",
                "boundary_tiles_seen", "boundary_required", "enabled"):
        assert key in bh, f"boundary_handling missing {key}"
    # tiny_mlp: M=4 N=128 K=64 tile=16 → iters M=1 N=8 K=4 → 32 total.
    assert bh["iters_M"] == 1
    assert bh["iters_N"] == 8
    assert bh["iters_K"] == 4
    assert bh["boundary_tiles_seen"] + bh["full_tiles_seen"] == 32


# --------------------------------------------------------------------------- #
# Bit-equality only claimed for exact equality
# --------------------------------------------------------------------------- #


def test_bit_equality_not_claimed_when_k_iters_greater_than_one(
    tmp_path: Path,
) -> None:
    """Honest behavior: with K_iters > 1, accumulation order differs
    from eager → max_abs_error > 0. The recipe gate must NOT claim
    bit_equality.

    Pre-M-37.12: M-12 reported ``fail_refinement_mismatch`` because the
    obligation declared ``bit_equality`` (overly optimistic) and the
    differential observed non-zero error.

    Post-M-37.12: the recipe gate downgrades the declaration to
    ``tolerance_eps`` whenever clean_divide AND K_iters > 1 (M-37.12 Fix).
    M-12 then discharges ``tolerance_eps`` honestly. The invariant the
    test checks is: ``refinement_status != discharged_bit_equality``
    AND ``max_abs_error > 0`` (proof that K_iters > 1 reordered
    accumulation). On tiny_mlp (M=4 K=64, tile_M4_N16_K16, K_iters=4)
    this surfaces as ``discharged_tolerance_eps``."""
    out = tmp_path / "tiny_mlp_no_bit_eq"
    _invoke(model="tiny_mlp", out_dir=out)
    rep = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    refinement_status = rep["error"]["refinement_status"]
    # The load-bearing claim: bit-equality is NEVER claimed when
    # K_iters > 1, regardless of whether the differential overall
    # passes or fails.
    assert refinement_status != "discharged_bit_equality", (
        f"bit_equality must not be discharged with K_iters > 1; "
        f"got refinement_status={refinement_status!r}"
    )
    # max_abs_error is not exactly 0 because accumulation reorder
    # caused float rounding to differ.
    assert rep["error"]["max_abs_error"] > 0.0
    # And the post-M-37.12 happy path: status pass, discharged_tolerance_eps.
    if rep["status"] == "pass":
        assert refinement_status == "discharged_tolerance_eps"


# --------------------------------------------------------------------------- #
# Blocked models still honestly blocked
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["proxy_vlm", "proxy_vla", "custom_unsupported_op"])
def test_non_set_tile_params_models_still_blocked_after_m16(
    model: str, tmp_path: Path,
) -> None:
    """Models whose greedy pick is FuseProducerConsumer or
    CreateKernelContract still take Path B (blocked) — M-16 only
    expanded SetTileParams coverage."""
    out = tmp_path / model
    res = _invoke(model=model, out_dir=out)
    assert res.returncode == 0
    rep = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert rep["status"] == "blocked"
    assert rep["mode"] == "blocked"


# --------------------------------------------------------------------------- #
# M-15B downstream retry still fires on REAL M-12 failures under M-16
# --------------------------------------------------------------------------- #


def test_m15b_downstream_retry_fires_on_real_m12_failure_under_m16(
    tmp_path: Path,
) -> None:
    """M-15B downstream-retry plumbing fires when any real downstream
    stage rejects the recipe.

    History:

    - Pre-M-37.11: tile_M16_N16_K16 + K=64 → boundary-aware Path A
      fails bit-equality → M-15B fires on real_transform_differential.
    - M-37.11: shape-fit tile candidates landed; tiny_mlp picks
      tile_M4_N16_K16 (clean-divide). M-11B whitelist became the
      blocker → M-15B fires on real_transform_validation.
    - M-37.12: whitelist relaxed; combined torch.allclose tolerance.
      No canonical-set model produces a natural M-12 failure anymore.
    - M-37.13: coverage restored by tampering an M-12 status=pass
      report into a status=fail report on disk and exercising the
      M-15B detector + emitter end-to-end. The natural-failure path
      itself is unreachable under correct math; the gap was always
      "M-15B plumbing has zero real-failure coverage", and the
      tampered-report test exercises exactly that plumbing.

    See tests/graph_compilation/test_m37_13_negative_controls.py for
    the restored end-to-end coverage. This test now delegates to it.
    """
    from compgen.graph_compilation.downstream_retry import (
        detect_downstream_failure,
    )
    out = tmp_path / "m12_real_fail_under_m16"
    res = _invoke(model="tiny_mlp", out_dir=out)
    assert res.returncode == 0, (
        f"tiny_mlp pipeline must succeed under M-37.12 before tampering"
    )
    report_path = (
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert report_path.exists()
    body = json.loads(report_path.read_text(encoding="utf-8"))
    body["status"] = "fail"
    body["failure_reasons"] = [
        "M-37.13 fault injection — synthetic real M-12 failure",
    ]
    body["error"]["max_abs_error"] = 1e6
    body["error"]["refinement_status"] = "fail_outside_tolerance"
    report_path.write_text(json.dumps(body, indent=2, sort_keys=True))
    failure = detect_downstream_failure(out)
    assert failure is not None
    assert failure.failed_stage == "real_transform_differential"
    assert failure.failed_check == "real_transform_differential_check"
