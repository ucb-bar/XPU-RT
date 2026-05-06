"""Tests for :mod:`compgen.graph_compilation.promotion_retrieval` (M-28)."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.graph_compilation.promotion_retrieval import (
    PromotedCandidate,
    retrieve_for_region,
)


def _write_sidecar(
    library: Path,
    *,
    target_hash: str = "tgt0",
    model_hash: str = "mdl0",
    objective_hash: str = "obj0",
    version: int = 1,
    region_signature: str = "",
    contract_hash: str = "",
    target_class: str = "host_cpu",
    recipe_id: str = "recipe_0",
    gate_level: str = "",
    evidence_summary: dict | None = None,
    applies_when: list | None = None,
    fallback_chain: list | None = None,
) -> Path:
    """Synthesize a recipe directory with a promoted_recipe.json sidecar."""
    key_str = f"{target_hash}_{model_hash}_{objective_hash}_v{version}"
    recipe_dir = library / key_str
    recipe_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "schema_version": "promoted_recipe_v1",
        "key": {
            "target_hash": target_hash,
            "model_hash": model_hash,
            "objective_hash": objective_hash,
            "version": version,
            "region_signature": region_signature,
            "contract_hash": contract_hash,
        },
        "recipe": {
            "recipe_id": recipe_id,
            "recipe_signature": region_signature,
            "recipe_ir_path": "recipe.mlir",
            "evidence_summary": evidence_summary or {},
            "applies_when": applies_when or [],
            "fallback_chain": fallback_chain or [],
            "validity": {"target_class": target_class},
            "gate_level": gate_level,
        },
    }
    (recipe_dir / "promoted_recipe.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Also drop a manifest.json so the recipe looks legit on disk.
    (recipe_dir / "manifest.json").write_text(
        json.dumps({"version": "1.0"}), encoding="utf-8"
    )
    return recipe_dir


def test_retrieves_recipe_by_region_signature(tmp_path: Path) -> None:
    """A recipe whose region_signature matches surfaces as a candidate."""
    library = tmp_path / "library"
    _write_sidecar(library, region_signature="abc123", recipe_id="recipe_A")

    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )

    assert len(out) == 1
    assert out[0].recipe_id == "recipe_A"
    assert out[0].match_kind == "region_pattern"
    assert out[0].region_signature == "abc123"


def test_retrieves_recipe_by_exact_contract_hash(tmp_path: Path) -> None:
    """contract_hash exact match outranks region_signature match."""
    library = tmp_path / "library"
    _write_sidecar(
        library,
        target_hash="tgt0",
        region_signature="abc123",
        contract_hash="ck0",
        recipe_id="recipe_exact",
    )

    out = retrieve_for_region(
        region_signature="abc123",
        contract_hash="ck0",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(out) == 1
    assert out[0].match_kind == "exact_contract"
    assert out[0].contract_hash == "ck0"


def test_exact_contract_ranked_before_region_pattern(tmp_path: Path) -> None:
    """When both kinds match, exact_contract appears first in the list."""
    library = tmp_path / "library"
    _write_sidecar(
        library, target_hash="tgt_exact", region_signature="abc123",
        contract_hash="ck0", recipe_id="recipe_exact",
    )
    _write_sidecar(
        library, target_hash="tgt_pat", region_signature="abc123",
        contract_hash="other_ck", recipe_id="recipe_pattern",
    )

    out = retrieve_for_region(
        region_signature="abc123",
        contract_hash="ck0",
        target_class="host_cpu",
        library_path=library,
    )

    assert len(out) == 2
    assert out[0].match_kind == "exact_contract"
    assert out[1].match_kind == "region_pattern"


def test_filters_by_target_class(tmp_path: Path) -> None:
    """A recipe proven on cuda_sm75 must not surface for host_cpu."""
    library = tmp_path / "library"
    _write_sidecar(
        library, region_signature="abc123",
        target_class="cuda_sm75", recipe_id="recipe_cuda",
    )

    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )
    assert out == []


def test_returns_empty_for_unknown_region_signature(tmp_path: Path) -> None:
    """Cold cache (no matching signatures) returns empty, not error."""
    library = tmp_path / "library"
    _write_sidecar(library, region_signature="aaa111")

    out = retrieve_for_region(
        region_signature="zzz999",
        target_class="host_cpu",
        library_path=library,
    )
    assert out == []


def test_handles_nonexistent_library(tmp_path: Path) -> None:
    """A library that doesn't exist yields empty, never raises."""
    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=tmp_path / "does_not_exist",
    )
    assert out == []


def test_skips_invalid_directories(tmp_path: Path) -> None:
    """Recipes renamed to *.invalid are excluded from results."""
    library = tmp_path / "library"
    _write_sidecar(library, region_signature="abc123", recipe_id="recipe_ok")
    bad = _write_sidecar(library, target_hash="tgtX", region_signature="abc123",
                         recipe_id="recipe_bad")
    bad.rename(bad.parent / (bad.name + ".invalid"))

    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(out) == 1
    assert out[0].recipe_id == "recipe_ok"


def test_skips_recipes_without_sidecar(tmp_path: Path) -> None:
    """Legacy recipes without promoted_recipe.json are silently skipped."""
    library = tmp_path / "library"
    legacy = library / "tgt_legacy_mdl0_obj0_v1"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")

    _write_sidecar(library, region_signature="abc123", recipe_id="recipe_modern")

    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(out) == 1
    assert out[0].recipe_id == "recipe_modern"


def test_pass_id_extracted_from_evidence_summary(tmp_path: Path) -> None:
    """M-37.1: pass_id + candidate_id are extracted from
    recipe.evidence_summary (where the M-26 promotion bridge writes them)
    so the agent can cross-link the promoted recipe to its source pass card."""
    library = tmp_path / "library"
    _write_sidecar(
        library,
        region_signature="rs_m37_passid",
        contract_hash="ch_m37",
        evidence_summary={
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "tile_M16_N16_K16",
            "region_id": "matmul_0",
        },
    )
    matches = retrieve_for_region(
        region_signature="rs_m37_passid",
        contract_hash="ch_m37",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(matches) == 1
    assert matches[0].pass_id == "set_tile_params"
    assert matches[0].candidate_id == "tile_M16_N16_K16"
    # Round-trip through to_dict
    raw = matches[0].to_dict()
    assert raw["pass_id"] == "set_tile_params"
    assert raw["candidate_id"] == "tile_M16_N16_K16"


def test_pass_id_empty_for_pre_m37_sidecar(tmp_path: Path) -> None:
    """A pre-M-37 sidecar (no candidate_kind in evidence_summary)
    yields empty pass_id; agent treats as no cross-link."""
    library = tmp_path / "library"
    _write_sidecar(
        library,
        region_signature="rs_pre_m37",
        contract_hash="ch_pre_m37",
        evidence_summary={"region_id": "matmul_0"},  # no candidate_kind
    )
    matches = retrieve_for_region(
        region_signature="rs_pre_m37",
        contract_hash="ch_pre_m37",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(matches) == 1
    assert matches[0].pass_id == ""
    assert matches[0].candidate_id == ""


def test_pass_id_resolves_against_card_registry(tmp_path: Path) -> None:
    """The pass_id surfaced by promotion retrieval must resolve against
    the live pass-card registry — otherwise the agent would surface a
    promoted recipe whose pass card doesn't exist."""
    from compgen.passes.cards import PassCardRegistry, default_registry_root

    library = tmp_path / "library"
    _write_sidecar(
        library,
        region_signature="rs_card_link",
        contract_hash="ch_card_link",
        evidence_summary={
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "tile_M16_N16_K16",
        },
    )
    matches = retrieve_for_region(
        region_signature="rs_card_link",
        contract_hash="ch_card_link",
        target_class="host_cpu",
        library_path=library,
    )
    registry = PassCardRegistry.load(default_registry_root())
    # Skip the test if the pass_id wasn't surfaced (legacy sidecar)
    if matches[0].pass_id:
        assert matches[0].pass_id in registry, (
            f"pass_id {matches[0].pass_id!r} from promoted recipe does "
            f"not resolve to a card in the registry"
        )


def test_promoted_candidate_to_dict_round_trip(tmp_path: Path) -> None:
    """PromotedCandidate.to_dict() is serializable + complete."""
    library = tmp_path / "library"
    _write_sidecar(
        library,
        region_signature="abc123",
        recipe_id="recipe_full",
        gate_level="verified_kernel",
        evidence_summary={"diff": "pass"},
        applies_when=["fact_a", "fact_b"],
        fallback_chain=["c1", "c2"],
    )

    out = retrieve_for_region(
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(out) == 1
    body = out[0].to_dict()
    # Must JSON-encode without error and preserve every field.
    encoded = json.dumps(body, sort_keys=True)
    assert "verified_kernel" in encoded
    assert body["applies_when"] == ["fact_a", "fact_b"]
    assert body["fallback_chain"] == ["c1", "c2"]
    assert body["evidence_summary"] == {"diff": "pass"}


def test_empty_signatures_returns_empty(tmp_path: Path) -> None:
    """Calling with empty region_signature AND empty contract_hash returns []."""
    library = tmp_path / "library"
    _write_sidecar(library, region_signature="abc123")

    out = retrieve_for_region(
        region_signature="",
        contract_hash="",
        target_class="host_cpu",
        library_path=library,
    )
    assert out == []


def test_cross_model_reuse(tmp_path: Path) -> None:
    """The headline use case: a recipe proven on model A surfaces on model B."""
    library = tmp_path / "library"
    # Run A — promotes a recipe on merlin_mlp_wide.
    _write_sidecar(
        library,
        target_hash="tgt0",
        model_hash="merlin_mlp_wide_hash",
        region_signature="matmul_fp32_16x16_host",
        recipe_id="recipe_run_A",
        target_class="host_cpu",
    )
    # Run B — different model, same region pattern.
    out = retrieve_for_region(
        region_signature="matmul_fp32_16x16_host",
        target_class="host_cpu",
        library_path=library,
    )
    assert len(out) == 1
    assert out[0].recipe_id == "recipe_run_A"
    assert out[0].match_kind == "region_pattern"
