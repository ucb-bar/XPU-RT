"""Tests for M-09 Differential / Reference Verification.

Covers:

- ``strip_compgen_metadata`` correctness (positive + adversarial inputs).
- Top-level ``run_differential_verification`` against a real prepared
  run dir copied per-test from the canonical
  ``differential_verification_suite/`` results.
- Negative tests required by M-09:
   * deleting transformed_payload for transform-like models fails.
   * creating transformed_payload for contract-only models fails.
   * mutating non-compgen payload semantics in transformed_payload
     causes the normalized diff to be non-empty and the stage to fail.
   * a hand-rolled report claiming
     ``real_transform_differential_check`` discharged is rejected by
     re-running the stage (it overwrites with the honest status).
   * mutating the source ``payload.mlir`` mid-run causes the source
     payload tree-hash check to fail.
   * missing ``golden_outputs.pt`` is reported as ``skipped``, not
     ``pass``-with-claim.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compgen.graph_compilation.differential_verification import (
    run_differential_verification,
    strip_compgen_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = REPO_ROOT / "results" / "graph_compilation" / "differential_verification_suite"


# --------------------------------------------------------------------------- #
# Stripper unit tests
# --------------------------------------------------------------------------- #


def test_strip_removes_compgen_attrs_only() -> None:
    src = (
        '%8 = linalg.matmul {compgen.region_id = "matmul_0", '
        'compgen.transposed_b = "true", compgen.tile = [16 : i64, 16 : i64, 16 : i64]} '
        'ins(%4, %6 : tensor<4x64xf32>, tensor<64x128xf32>) -> tensor<4x128xf32>'
    )
    out = strip_compgen_metadata(src)
    assert "compgen." not in out
    assert "linalg.matmul" in out
    assert "ins(%4, %6" in out


def test_strip_preserves_non_compgen_attrs() -> None:
    src = '%x = some.op {alignment = 16 : i64, name = "foo"} : tensor<4xf32>'
    out = strip_compgen_metadata(src)
    assert out == src, "non-compgen attrs must round-trip identically"


def test_strip_collapses_empty_block() -> None:
    src = '%x = some.op {compgen.tile = [16 : i64, 16 : i64, 16 : i64]} ins(%y) : tensor<4xf32>'
    out = strip_compgen_metadata(src)
    # Empty {} must be dropped along with the preceding space.
    assert "{}" not in out
    assert "some.op  " not in out, "double space left after collapse"
    assert "some.op ins(%y)" in out


def test_strip_keeps_other_entries_when_some_removed() -> None:
    src = '%x = some.op {alignment = 16 : i64, compgen.tile = [16 : i64], name = "foo"} : tensor<4xf32>'
    out = strip_compgen_metadata(src)
    assert "compgen.tile" not in out
    assert "alignment = 16 : i64" in out
    assert 'name = "foo"' in out


def test_strip_does_not_touch_region_bodies() -> None:
    src = """builtin.module {
  func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = some.op {compgen.tile = [16 : i64]} ins(%a) : tensor<4xf32>
    func.return %0 : tensor<4xf32>
  }
}
"""
    out = strip_compgen_metadata(src)
    assert "func.func @forward" in out
    assert "func.return" in out
    assert "compgen." not in out
    # The multi-line braces around the module/func must survive.
    assert "builtin.module {" in out
    assert "}\n}\n" in out


def test_strip_handles_string_with_brace() -> None:
    src = '%x = some.op {alignment = 16 : i64, message = "}{,"} : tensor<4xf32>'
    out = strip_compgen_metadata(src)
    assert out == src, "quoted braces / commas must not confuse the stripper"


# --------------------------------------------------------------------------- #
# Per-model fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_mlp_run(tmp_path: Path) -> Path:
    src = SUITE / "tiny_mlp"
    if not src.is_dir():
        pytest.skip(f"fixture run dir missing: {src}; run --stop-after differential-verification")
    dst = tmp_path / "tiny_mlp"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def custom_run(tmp_path: Path) -> Path:
    src = SUITE / "custom_unsupported_op"
    if not src.is_dir():
        pytest.skip(f"fixture run dir missing: {src}")
    dst = tmp_path / "custom_unsupported_op"
    shutil.copytree(src, dst)
    return dst


# --------------------------------------------------------------------------- #
# Positive end-to-end tests
# --------------------------------------------------------------------------- #


def test_transform_like_model_passes_metadata_noop(tiny_mlp_run: Path) -> None:
    result = run_differential_verification(tiny_mlp_run)
    assert result.overall == "pass"
    assert result.mode == "metadata_noop_mvp"
    rep = json.loads(result.report_path.read_text(encoding="utf-8"))
    names = {c["name"] for c in rep["checks"]}
    assert "normalized_payloads_equal_after_stripping_compgen_metadata" in names
    assert "no_real_transform_claimed" in names
    # Diff must be empty.
    diff = (
        tiny_mlp_run / "03_recipe_planning" / "differential_verification"
        / "normalized_payload_diff.txt"
    )
    assert diff.read_text(encoding="utf-8") == ""
    # Status must remain "discharged_metadata_noop", with a real_*
    # check still pending.
    statuses = json.loads(result.semantic_status_path.read_text(encoding="utf-8"))[
        "statuses"
    ]
    assert all(s["status"] == "discharged_metadata_noop" for s in statuses)
    assert all(
        any(r.startswith("real_") for r in s["remaining"]) for s in statuses
    )


def test_contract_only_model_passes(custom_run: Path) -> None:
    result = run_differential_verification(custom_run)
    assert result.overall == "pass"
    assert result.mode == "contract_only_mvp"
    cdir = custom_run / "03_recipe_planning" / "differential_verification"
    assert (cdir / "contract_reference_check.json").exists()
    # transformed_payload must NOT exist for contract-only.
    assert not (
        custom_run / "03_recipe_planning" / "post_lowering"
        / "transformed_payload.mlir"
    ).exists()
    # No metadata_noop_equivalence file in contract-only mode.
    assert not (cdir / "metadata_noop_equivalence.json").exists()


# --------------------------------------------------------------------------- #
# Negative tests
# --------------------------------------------------------------------------- #


def test_deleting_transformed_payload_fails(tiny_mlp_run: Path) -> None:
    tp = (
        tiny_mlp_run / "03_recipe_planning" / "post_lowering"
        / "transformed_payload.mlir"
    )
    assert tp.exists()
    tp.unlink()
    result = run_differential_verification(tiny_mlp_run)
    assert result.overall == "fail"
    assert any("transformed_payload" in f for f in result.failures)


def test_creating_transformed_payload_for_contract_only_fails(
    custom_run: Path,
) -> None:
    tp = (
        custom_run / "03_recipe_planning" / "post_lowering"
        / "transformed_payload.mlir"
    )
    tp.write_text("// stray transformed payload\n", encoding="utf-8")
    result = run_differential_verification(custom_run)
    assert result.overall == "fail"
    assert any("contract-only" in f for f in result.failures)


def test_mutating_non_compgen_payload_semantics_fails(tiny_mlp_run: Path) -> None:
    """Strongest negative test: change `linalg.matmul` -> `func.call` in the
    transformed payload. After stripping compgen metadata, the normalized
    payloads differ and M-09 must fail."""
    tp = (
        tiny_mlp_run / "03_recipe_planning" / "post_lowering"
        / "transformed_payload.mlir"
    )
    text = tp.read_text(encoding="utf-8")
    # Inject one real semantic change (replace the op kind on the
    # tile-target line). It still has compgen attrs around it, so this
    # tests that the stripper doesn't accidentally smooth it over.
    new_text = text.replace("linalg.matmul {compgen.region_id", "func.call {compgen.region_id", 1)
    assert new_text != text
    tp.write_text(new_text, encoding="utf-8")

    result = run_differential_verification(tiny_mlp_run)
    assert result.overall == "fail"
    assert any("normalized payloads differ" in f for f in result.failures)
    # The normalized_payload_diff.txt must now be non-empty.
    diff = (
        tiny_mlp_run / "03_recipe_planning" / "differential_verification"
        / "normalized_payload_diff.txt"
    )
    assert diff.read_text(encoding="utf-8") != ""


def test_false_real_transform_discharge_claim_is_overwritten(
    tiny_mlp_run: Path,
) -> None:
    """A previous (or hand-edited) report claiming
    ``discharged_real_transform`` must be rejected when M-09 reruns. The
    stage clears stale outputs and writes the honest discharge level."""
    cdir = tiny_mlp_run / "03_recipe_planning" / "differential_verification"
    cdir.mkdir(parents=True, exist_ok=True)
    bogus = {
        "schema_version": "differential_verification_report_v1",
        "status": "pass",
        "model_id": "tiny_mlp",
        "target_id": "host_cpu",
        "mode": "metadata_noop_mvp",
        "checks": [],
        "semantic_status": [
            {
                "obligation": "obl_recipe_0000",
                "recipe_op_id": "recipe_0000",
                "declared_refinement": "bit_equality",
                "status": "discharged_real_transform_differential_check",
                "discharged": ["real_transform_differential_check"],
                "remaining": [],
            }
        ],
    }
    (cdir / "differential_verification_report.json").write_text(
        json.dumps(bogus), encoding="utf-8"
    )
    result = run_differential_verification(tiny_mlp_run)
    rewritten = json.loads(result.report_path.read_text(encoding="utf-8"))
    # Honest report: status field reverts to discharged_metadata_noop.
    assert rewritten["semantic_status"][0]["status"] == "discharged_metadata_noop"
    # The no_real_transform_claimed check must be present and pass.
    names_pass = {c["name"]: c["status"] for c in rewritten["checks"]}
    assert names_pass.get("no_real_transform_claimed") == "pass"


def test_source_payload_mutation_fails(tiny_mlp_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutate a payload.mlir file under 01_payload_lowering between the
    pre and post hash. The pre/post hash invariant must catch it."""
    from compgen.graph_compilation import differential_verification as dv_mod

    real_sha = dv_mod.sha256_tree
    state = {"call": 0}

    def fake_sha(path: Path) -> str:
        state["call"] += 1
        if state["call"] == 1:
            return real_sha(path)
        return "deadbeef" * 8  # pretend the tree changed after the run

    monkeypatch.setattr(dv_mod, "sha256_tree", fake_sha)
    result = run_differential_verification(tiny_mlp_run)
    assert result.overall == "fail"
    assert any("01_payload_lowering" in f for f in result.failures)


def test_missing_goldens_reports_skipped_not_pass(tiny_mlp_run: Path) -> None:
    gc = tiny_mlp_run / "00_graph_capture"
    for name in ("golden_inputs.pt", "golden_outputs.pt"):
        p = gc / name
        if p.exists():
            p.unlink()
    result = run_differential_verification(tiny_mlp_run)
    # Golden status is reported as skipped, not pass-with-claim.
    golden = json.loads(
        (
            tiny_mlp_run / "03_recipe_planning" / "differential_verification"
            / "golden_reference_check.json"
        ).read_text(encoding="utf-8")
    )
    assert golden["status"] == "skipped"
    assert "skipped_reason" in golden
    # The overall stage still passes — the metadata-noop check is what
    # matters; missing goldens are surfaced as skipped, not pass.
    assert result.overall == "pass"
