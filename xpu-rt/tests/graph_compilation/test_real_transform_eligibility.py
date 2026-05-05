"""Tests for M-11A Real Transform Eligibility Audit.

Read-only audit. Tests verify that:

- The artifact exists for every model that ran the stage.
- ``tiny_mlp`` is eligible (or has a precise rejection reason).
- Tile parsing pulls M/N/K from ``verified_recipe.mlir``.
- Tile mismatch with the verified recipe fails eligibility.
- Opaque regions are rejected.
- Non-SetTileParams recipes are cleanly ineligible (audit status==pass).
- Missing ``payload_ref`` fails eligibility.
- Multiple matching ``compgen.region_id`` occurrences fail eligibility.
- Source payload mutation is detected via the pre/post SHA pair.
- No ``transformed_payload.real.mlir`` is emitted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compgen.graph_compilation.real_transform_eligibility import (
    _find_matmul_for_region,
    _matmul_signature,
    _parse_tile_from_verified_recipe,
    run_real_transform_eligibility,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = (
    REPO_ROOT / "results" / "graph_compilation" / "real_transform_eligibility_suite"
)

_CANONICAL = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
)


def _need_suite() -> None:
    if not SUITE.is_dir():
        pytest.skip(
            f"fixture suite missing: {SUITE}; run "
            f"`compgen.graph_compilation run-suite --stop-after "
            f"real-transform-eligibility` first"
        )


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Positive tests over the canonical suite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", _CANONICAL)
def test_artifact_exists_for_all_six_canonical_models(model: str) -> None:
    _need_suite()
    p = SUITE / model / "03_recipe_planning" / "real_transform_eligibility.json"
    md = SUITE / model / "03_recipe_planning" / "real_transform_eligibility.md"
    assert p.exists(), f"{model}: missing real_transform_eligibility.json"
    assert md.exists(), f"{model}: missing real_transform_eligibility.md"


def test_tiny_mlp_is_eligible() -> None:
    _need_suite()
    a = _read(SUITE / "tiny_mlp" / "03_recipe_planning" / "real_transform_eligibility.json")
    assert a["status"] == "pass"
    assert a["eligible"] is True
    assert a["selected_recipe"]["recipe_kind"] == "SetTileParams"
    assert a["matmul_signature"]["lhs_dtype"] == "f32"
    assert a["matmul_signature"]["rank"] == 2
    assert a["matmul_signature"]["dynamic_dims"] is False
    assert not a["rejection_reasons"]


def test_acceptance_table_matches_user_spec() -> None:
    _need_suite()
    expected = {
        "tiny_mlp": (True, "SetTileParams"),
        "tiny_attention": (True, "SetTileParams"),
        "tiny_conv_block": (True, "SetTileParams"),
        "proxy_vlm": (False, "FuseProducerConsumer"),
        "proxy_vla": (False, "FuseProducerConsumer"),
        "custom_unsupported_op": (False, "CreateKernelContract"),
    }
    for model, (eligible, kind) in expected.items():
        a = _read(SUITE / model / "03_recipe_planning" / "real_transform_eligibility.json")
        assert a["eligible"] == eligible, (
            f"{model}: expected eligible={eligible}, got {a['eligible']} "
            f"with reasons={a['rejection_reasons']}"
        )
        assert a["selected_recipe"]["recipe_kind"] == kind, (
            f"{model}: expected recipe_kind={kind}, got "
            f"{a['selected_recipe']['recipe_kind']}"
        )
        # Audit-side status must always be pass — ineligibility is not a
        # pipeline failure.
        assert a["status"] == "pass"


def test_non_set_tile_params_audit_status_is_pass(tmp_path: Path) -> None:
    """proxy_vlm picks FuseProducerConsumer. The audit must mark it
    cleanly ineligible WITHOUT raising or marking the audit itself as
    fail."""
    _need_suite()
    a = _read(SUITE / "proxy_vlm" / "03_recipe_planning" / "real_transform_eligibility.json")
    assert a["status"] == "pass"
    assert a["eligible"] is False
    assert any("FuseProducerConsumer" in r for r in a["rejection_reasons"])


def test_tile_is_parsed_from_verified_recipe() -> None:
    _need_suite()
    a = _read(SUITE / "tiny_mlp" / "03_recipe_planning" / "real_transform_eligibility.json")
    assert a["selected_recipe"]["tile"] == {"M": 16, "N": 16, "K": 16}


def test_no_transformed_real_payload_emitted() -> None:
    _need_suite()
    for model in _CANONICAL:
        run = SUITE / model
        assert not (
            run / "03_recipe_planning" / "real_lowering" / "transformed_payload.real.mlir"
        ).exists(), f"{model}: transformed_payload.real.mlir must not exist (M-11A is read-only)"


def test_payload_sha_before_equals_after() -> None:
    _need_suite()
    for model in _CANONICAL:
        a = _read(SUITE / model / "03_recipe_planning" / "real_transform_eligibility.json")
        before = a["payload"]["payload_sha256_before"]
        after = a["payload"]["payload_sha256_after"]
        if before:  # may be empty string for ineligible non-SetTileParams models
            assert before == after, (
                f"{model}: payload_sha256_before != _after"
            )


def test_tile_geometry_records_iters_for_eligible_models() -> None:
    _need_suite()
    for model in ("tiny_mlp", "tiny_attention", "tiny_conv_block"):
        a = _read(SUITE / model / "03_recipe_planning" / "real_transform_eligibility.json")
        geom = a["tile_geometry"]
        assert geom is not None
        for k in ("iters_M", "iters_N", "iters_K"):
            assert geom[k] >= 1


# --------------------------------------------------------------------------- #
# Parser unit tests
# --------------------------------------------------------------------------- #


def test_parse_tile_from_verified_recipe() -> None:
    text = (
        "recipe.set_tile_params @recipe_0000 attributes "
        '{ rationale = "x", region = "matmul_0", '
        "tile = { K = 16 : i64, M = 16 : i64, N = 32 : i64 } }"
    )
    assert _parse_tile_from_verified_recipe(text, "recipe_0000") == (16, 32, 16)
    assert _parse_tile_from_verified_recipe(text, "recipe_9999") is None


def test_find_matmul_for_region_extracts_shapes() -> None:
    text = (
        '%8 = linalg.matmul {compgen.region_id = "matmul_0", '
        'compgen.transposed_b = "true"} '
        'ins(%4, %6 : tensor<4x64xf32>, tensor<64x128xf32>) '
        'outs(%7 : tensor<4x128xf32>) -> tensor<4x128xf32>'
    )
    matmuls = _find_matmul_for_region(text, "matmul_0")
    assert len(matmuls) == 1
    sig = _matmul_signature(matmuls[0])
    assert sig is not None
    assert sig["M"] == 4 and sig["N"] == 128 and sig["K"] == 64
    assert sig["lhs_dtype"] == "f32"


def test_find_matmul_returns_empty_for_other_region() -> None:
    text = (
        '%8 = linalg.matmul {compgen.region_id = "matmul_0"} '
        'ins(%4, %6 : tensor<4x64xf32>, tensor<64x128xf32>) '
        'outs(%7 : tensor<4x128xf32>) -> tensor<4x128xf32>'
    )
    assert _find_matmul_for_region(text, "matmul_1") == []


# --------------------------------------------------------------------------- #
# Negative tests (mutate a copy + re-invoke the auditor)
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_mlp_run(tmp_path: Path) -> Path:
    _need_suite()
    src = SUITE / "tiny_mlp"
    if not src.is_dir():
        pytest.skip(f"fixture run dir missing: {src}")
    dst = tmp_path / "tiny_mlp"
    shutil.copytree(src, dst)
    return dst


def test_tile_mismatch_with_verified_recipe_fails_eligibility(tiny_mlp_run: Path) -> None:
    rp = tiny_mlp_run / "03_recipe_planning"
    text = (rp / "verified_recipe.mlir").read_text(encoding="utf-8")
    # Twiddle the tile in the verified recipe text (M=16 -> M=99).
    mutated = text.replace("M = 16 : i64", "M = 99 : i64")
    assert mutated != text
    (rp / "verified_recipe.mlir").write_text(mutated, encoding="utf-8")
    result = run_real_transform_eligibility(tiny_mlp_run)
    assert result.eligible is False
    assert any("tile mismatch" in r for r in result.rejection_reasons)


def test_missing_payload_ref_fails_eligibility(tiny_mlp_run: Path) -> None:
    rm_path = tiny_mlp_run / "02_graph_analysis" / "region_map.json"
    rm = json.loads(rm_path.read_text(encoding="utf-8"))
    for r in rm["regions"]:
        if r["region_id"] == "matmul_0":
            for op in r["payload_ops"]:
                op["payload_ref"] = ""
    rm_path.write_text(json.dumps(rm), encoding="utf-8")
    result = run_real_transform_eligibility(tiny_mlp_run)
    assert result.eligible is False
    assert any("payload_ref" in r for r in result.rejection_reasons)


def test_multiple_matching_region_ids_fails_eligibility(tiny_mlp_run: Path) -> None:
    """Inject a second `linalg.matmul` op carrying the same
    compgen.region_id into the source payload, then re-run."""
    pl = tiny_mlp_run / "01_payload_lowering" / "export_program" / "payload.mlir"
    text = pl.read_text(encoding="utf-8")
    duplicate = (
        '    %999 = linalg.matmul {compgen.region_id = "matmul_0"} '
        'ins(%4, %6 : tensor<4x64xf32>, tensor<64x128xf32>) '
        'outs(%7 : tensor<4x128xf32>) -> tensor<4x128xf32>\n'
    )
    mutated = text.replace(
        "func.return %15 : tensor<4x32xf32>",
        duplicate + "    func.return %15 : tensor<4x32xf32>",
    )
    assert mutated != text
    pl.write_text(mutated, encoding="utf-8")
    result = run_real_transform_eligibility(tiny_mlp_run)
    assert result.eligible is False
    assert any("ambiguous" in r or "found 2" in r for r in result.rejection_reasons)


def test_source_payload_mutation_during_audit_is_detected(
    tiny_mlp_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the post-audit SHA to differ from the pre-audit SHA."""
    from compgen.graph_compilation import real_transform_eligibility as mod

    # The auditor takes one pre-snapshot pass over payload.mlir files
    # (2 files for tiny_mlp), then a post-snapshot pass at the end. The
    # pre-snapshot must use the real hash and the post-snapshot must
    # return a different hash to simulate mutation.
    pre_count = sum(1 for _ in (tiny_mlp_run / "01_payload_lowering").rglob("payload.mlir"))
    state = {"call": 0}
    real_sha = mod._sha256_file

    def fake_sha(path: Path) -> str:
        state["call"] += 1
        if state["call"] <= pre_count:
            return real_sha(path)
        return "sha256:" + "deadbeef" * 8

    monkeypatch.setattr(mod, "_sha256_file", fake_sha)
    result = run_real_transform_eligibility(tiny_mlp_run)
    assert result.overall == "fail"
    assert any(
        "source payload mutated" in r for r in result.rejection_reasons
    )


def test_opaque_region_is_rejected(tiny_mlp_run: Path) -> None:
    """Force the selected region's source_classification to opaque_fallback."""
    rm_path = tiny_mlp_run / "02_graph_analysis" / "region_map.json"
    rm = json.loads(rm_path.read_text(encoding="utf-8"))
    for r in rm["regions"]:
        if r["region_id"] == "matmul_0":
            r["source_classification"] = "opaque_fallback"
            r["kind"] = "opaque_aten_matmul"
    rm_path.write_text(json.dumps(rm), encoding="utf-8")
    result = run_real_transform_eligibility(tiny_mlp_run)
    assert result.eligible is False
    assert any("opaque" in r for r in result.rejection_reasons)
