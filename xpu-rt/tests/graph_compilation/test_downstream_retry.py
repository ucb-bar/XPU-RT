"""Tests for M-15B Downstream Gate Rejection Retry.

Verifies the retry protocol around downstream gate failures using REAL
M-12 failures (no test injection). When greedy picks tile_16 on a
model whose K dim is not 16 (e.g. tiny_mlp with K=64, K_iters=4), the
boundary-aware evaluator runs Path A but accumulation reorder makes
bit-equality fail honestly. M-15B then maps that failure back to the
selected candidate via the downstream-retry protocol.

Verifies:

- A real M-12 failure produces a typed
  ``downstream_retry_request.json`` mapping the failure back to the
  selected candidate.
- The failed candidate is excluded from the next-attempt allowed set.
- A fresh invocation with a different candidate (one whose tile_K
  matches K so K_iters=1) produces a clean recipe.mlir.
- Audit artifacts under ``downstream_retry/attempts/attempt_000/`` are
  preserved for the failed run.
- The pipeline aborts (non-zero exit) when downstream fails.
- Verifier reports are NEVER edited by the retry path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _invoke(
    *, out_dir: Path,
    model: str = "tiny_mlp",
    selection_mode: str = "greedy",
    response_paths: list[Path] | None = None,
    stop_after: str = "cost-preview-v2",
) -> subprocess.CompletedProcess:
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
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def real_failed_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """A real M-12 failure: tiny_mlp + greedy picks tile_16, K=64 →
    K_iters=4 → bit-equality fails. M-15B emits the retry request and
    the pipeline raises (non-zero exit)."""
    out = tmp_path_factory.mktemp("m15b_real_fail") / "tiny_mlp_fail"
    res = _invoke(out_dir=out, model="tiny_mlp")
    if res.returncode == 0:
        pytest.skip(
            "tiny_mlp greedy is no longer producing a real M-12 failure; "
            "M-15B downstream-retry tests require a known-failing model"
        )
    return out


def _failed_candidate_id(run_dir: Path) -> str:
    rr = _read(
        run_dir / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    return str(rr["failed_candidate_id"])


# --------------------------------------------------------------------------- #
# Real M-12 failure produces a typed retry request
# --------------------------------------------------------------------------- #


def test_m12_failure_produces_downstream_retry_request(
    real_failed_run: Path,
) -> None:
    rr_path = (
        real_failed_run / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    assert rr_path.exists(), "M-15B should emit downstream_retry_request.json"
    rr = _read(rr_path)
    assert rr["schema_version"] == "downstream_retry_request_v1"
    assert rr["status"] == "retry_required"
    assert rr["failed_stage"] == "real_transform_differential"
    assert rr["failed_check"] == "real_transform_differential_check"
    assert rr["failed_candidate_id"]  # non-empty string
    assert rr["evidence"]["report_path"] == (
        "03_recipe_planning/real_verification/real_differential_report.json"
    )
    assert rr["retry_policy"]["must_choose_different_candidate"] is True
    assert rr["retry_policy"]["exclude_candidate_ids"] == [
        rr["failed_candidate_id"]
    ]


def test_failed_candidate_excluded_from_allowed_set(
    real_failed_run: Path,
) -> None:
    rr = _read(
        real_failed_run / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    assert rr["failed_candidate_id"] not in rr["candidate_ids_allowed"]
    cas = _read(
        real_failed_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    }
    assert set(rr["candidate_ids_allowed"]) == legal_ids - {
        rr["failed_candidate_id"]
    }


def test_failed_candidate_context_is_emitted(real_failed_run: Path) -> None:
    ctx = _read(
        real_failed_run / "03_recipe_planning" / "downstream_retry"
        / "failed_candidate_context.json"
    )
    assert ctx["schema_version"] == "failed_candidate_context_v1"
    assert ctx["failed_candidate_id"] == _failed_candidate_id(real_failed_run)
    assert ctx["candidate_kind"] == "set_tile_params"
    assert ctx["region_id"]  # non-empty
    assert ctx["failed_stage"] == "real_transform_differential"


def test_attempt_000_snapshot_preserves_failed_state(
    real_failed_run: Path,
) -> None:
    attempt_dir = (
        real_failed_run / "03_recipe_planning" / "downstream_retry"
        / "attempts" / "attempt_000"
    )
    assert (attempt_dir / "downstream_retry_request.json").exists()
    assert (attempt_dir / "failed_stage_report.json").exists()
    assert (attempt_dir / "selected_candidate_id.txt").exists()
    sel = (attempt_dir / "selected_candidate_id.txt").read_text(
        encoding="utf-8",
    ).strip()
    assert sel == _failed_candidate_id(real_failed_run)
    failed_report = _read(attempt_dir / "failed_stage_report.json")
    assert failed_report["status"] == "fail"


# --------------------------------------------------------------------------- #
# Pipeline non-zero exit + recipe.mlir for failed candidate
# --------------------------------------------------------------------------- #


def test_pipeline_exits_non_zero_on_downstream_failure(tmp_path: Path) -> None:
    """Exit code is observable per-invocation; needs a fresh run rather
    than the cached fixture (which captured the result already)."""
    out = tmp_path / "exit"
    res = _invoke(out_dir=out, model="tiny_mlp")
    assert res.returncode != 0
    assert "M-15B downstream-gate rejection" in res.stderr


def test_recipe_mlir_committed_for_failed_candidate(
    real_failed_run: Path,
) -> None:
    """M-05 commits recipe.mlir BEFORE M-12 runs. The failed run still
    has a recipe.mlir on disk pointing at the failed candidate."""
    recipe = (real_failed_run / "03_recipe_planning" / "recipe.mlir").read_text(
        encoding="utf-8",
    )
    assert _failed_candidate_id(real_failed_run) in recipe


# --------------------------------------------------------------------------- #
# Successful retry: agent-file with a passing candidate
# --------------------------------------------------------------------------- #


def _find_clean_divides_tile_candidate(run_dir: Path, K: int) -> dict | None:
    """Find a legal SetTileParams candidate whose tile_K equals K (so
    K_iters=1 and accumulation order matches eager). Returns the
    candidate dict or None."""
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    for c in cas["candidates"]:
        if c.get("kind") != "set_tile_params":
            continue
        if not (c.get("legality") or {}).get("ok"):
            continue
        # Tile parameters live under recipe_delta or evidence.
        delta = c.get("recipe_delta") or []
        for op in delta:
            attrs = op.get("attrs") or op.get("body") or {}
            tile_k = attrs.get("tile_K") or attrs.get("k_tile") or attrs.get("tile_k")
            if tile_k is None and "label" in c:
                # Parse from label like tile_M16_N16_K64.
                label = c["label"]
                if f"_K{K}" in label:
                    return c
            elif tile_k == K:
                return c
    return None


def test_successful_retry_commits_clean_recipe(
    real_failed_run: Path, tmp_path: Path,
) -> None:
    """End-to-end retry: first run greedy on tiny_mlp fails M-12. Second
    run with agent-file selecting a tile candidate whose tile_K==K
    (K_iters=1) succeeds."""
    # tiny_mlp K=64.
    candidate = _find_clean_divides_tile_candidate(real_failed_run, K=64)
    if candidate is None:
        pytest.skip(
            "no SetTileParams candidate with tile_K=64 in tiny_mlp action "
            "space; skipping successful-retry e2e test"
        )

    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": candidate["candidate_id"],
        "rationale": {
            "summary": (
                f"Retry pick: {candidate.get('label', '')} has tile_K=64 "
                "(K_iters=1) so accumulation order matches eager and "
                "bit-equality is preserved."
            ),
            "evidence": [
                {"field": "candidate.kind", "value": candidate["kind"],
                 "reason": "Structured tiling on matmul region."},
                {"field": "candidate.label",
                 "value": candidate.get("label", ""),
                 "reason": "tile_K=K so K loop runs once."},
            ],
        },
    }
    response_path = tmp_path / "retry_response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    out_pass = tmp_path / "second_pass"
    res = _invoke(
        out_dir=out_pass, model="tiny_mlp",
        selection_mode="agent-file",
        response_paths=[response_path],
    )
    assert res.returncode == 0, res.stderr
    assert (out_pass / "03_recipe_planning" / "recipe.mlir").exists()
    assert not (
        out_pass / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    ).exists()
    real_diff = _read(
        out_pass / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert real_diff["status"] == "pass"


# --------------------------------------------------------------------------- #
# Trust boundary: verifier reports must not be edited by the retry path
# --------------------------------------------------------------------------- #


def test_verifier_reports_are_not_edited_by_retry_emission(
    real_failed_run: Path,
) -> None:
    """The retry emitter only READS verifier reports; it must not
    mutate them."""
    import hashlib

    original_report = (
        real_failed_run / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    snapshot_report = (
        real_failed_run / "03_recipe_planning" / "downstream_retry"
        / "attempts" / "attempt_000" / "failed_stage_report.json"
    )
    assert original_report.exists()
    assert snapshot_report.exists()
    h1 = hashlib.sha256(original_report.read_bytes()).hexdigest()
    h2 = hashlib.sha256(snapshot_report.read_bytes()).hexdigest()
    assert h1 == h2, (
        "downstream-retry emission appears to have mutated the M-12 "
        "report (snapshot SHA differs from current original SHA)"
    )


# --------------------------------------------------------------------------- #
# Detector unit tests
# --------------------------------------------------------------------------- #


def test_detector_returns_none_when_no_downstream_failures(tmp_path: Path) -> None:
    """A clean run on a model with clean K divides (merlin_mlp_wide,
    K=16 / tile=16 → K_iters=1) produces no downstream_retry artifacts."""
    out = tmp_path / "clean"
    res = _invoke(out_dir=out, model="merlin_mlp_wide")
    assert res.returncode == 0
    assert not (
        out / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    ).exists()


def test_detector_skipped_path_does_not_trigger_retry(tmp_path: Path) -> None:
    """A model that takes the skipped/blocked path through M-11A/B/M-12
    (e.g. ``proxy_vlm`` selects FuseProducerConsumer, which makes M-12
    emit ``status=blocked`` rather than ``status=fail``) must NOT emit
    a downstream_retry_request — blocked is not the candidate's fault."""
    out = tmp_path / "blocked"
    res = _invoke(out_dir=out, model="proxy_vlm")
    assert res.returncode == 0  # blocked is not failure
    real_diff = _read(
        out / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    assert real_diff["status"] == "blocked"
    assert not (
        out / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    ).exists()


# --------------------------------------------------------------------------- #
# No compiler-core regressions
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports_in_module() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "downstream_retry.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"M-15B module must not import: {pat}"
