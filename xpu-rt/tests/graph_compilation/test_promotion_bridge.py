"""Tests for :mod:`compgen.graph_compilation.promotion_bridge` (M-26)."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.graph_compilation.promotion_bridge import emit


def _write(path: Path, body: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, dict):
        path.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        path.write_text(body, encoding="utf-8")


def _build_minimal_run_dir(
    tmp_path: Path,
    *,
    selected_candidate_id: str | None = "cand_0001",
    include_differential_pass: bool = True,
    include_post_lowering_pass: bool = True,
) -> Path:
    """Synthesize a Phase B run dir with the minimum artifacts the bridge reads."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    _write(
        run_dir / "run_manifest.json",
        {
            "schema_version": "run_manifest_v1",
            "run_id": "test_run_001",
            "created_at_utc": "2026-05-05T00:00:00Z",
            "model": {
                "config_path": "configs/models/tiny_mlp.yaml",
                "config_sha256": "0" * 64,
                "model_id": "tiny_mlp",
            },
            "target": {
                "config_path": "configs/targets/host_cpu.yaml",
                "config_sha256": "0" * 64,
                "target_id": "host_cpu",
            },
            "seed": 0,
            "stages": [],
        },
    )

    _write(
        run_dir / "01_payload_lowering" / "payload.mlir",
        "module @tiny_mlp { /* canonical payload IR */ }\n",
    )

    if selected_candidate_id is not None:
        _write(
            run_dir / "03_recipe_planning" / "candidate_selection.json",
            {
                "schema_version": "candidate_selection_v1",
                "model_id": "tiny_mlp",
                "target_id": "host_cpu",
                "selected_candidate_id": selected_candidate_id,
                "site_id": "site_0001",
                "region_id": "region_0001",
                "candidate_kind": "set_tile_params",
                "label": "tile=16x16x16",
                "selection_mode": "greedy",
                "selected_at_utc": "2026-05-05T00:00:01Z",
                "source": {},
                "legality": {"ok": True, "reason": ""},
                "rationale": {"primary_reason": "lowest_cost", "evidence": []},
                "recipe_delta": [],
                "cost_preview": {"static_relative_cost": 0.5},
                "evidence": {},
            },
        )
    else:
        _write(
            run_dir / "03_recipe_planning" / "candidate_selection.json",
            {
                "schema_version": "candidate_selection_v1",
                "model_id": "tiny_mlp",
                "target_id": "host_cpu",
                "selected_candidate_id": None,
                "selection_mode": "greedy",
                "rationale": {"primary_reason": "no candidate", "evidence": []},
                "recipe_delta": [],
            },
        )

    _write(
        run_dir / "03_recipe_planning" / "recipe.mlir",
        "module @recipe { /* recipe ir */ }\n",
    )

    _write(
        run_dir / "02_graph_analysis" / "region_dossiers" / "region_0001.json",
        {
            "schema_version": "region_dossier_v2",
            "region_id": "region_0001",
            "module_id": "main",
            "kind": "matmul",
            "source": {
                "fx_nodes": [],
                "fx_targets": [],
                "payload_ops": [],
                "source_classification": "fx_imported",
            },
            "cost": {
                "flops": 0,
                "bytes": 0,
                "arithmetic_intensity": 0.0,
                "estimated_latency_us": {},
                "bottleneck_resource": {},
            },
            "reuse": {"inputs": [], "outputs": []},
            "numerical_sensitivity": {
                "fp32": {
                    "eps_out": 0.0,
                    "budget_remaining": 1.0,
                    "status": "safe",
                },
                "fp16_accum": {
                    "eps_out": 0.0,
                    "budget_remaining": 0.0,
                    "status": "exceeds_budget",
                },
                "fp8_e4m3": {
                    "eps_out": 0.0,
                    "budget_remaining": 0.0,
                    "status": "exceeds_budget",
                },
                "fast_math": {
                    "eps_out": 0.0,
                    "budget_remaining": 0.0,
                    "status": "exceeds_budget",
                },
            },
            "working_set_curve": [{"input_dims": [16, 16]}],
            "placement_envelope": {"devices": []},
            "legality_constraints": [],
        },
    )

    if include_post_lowering_pass:
        _write(
            run_dir / "03_recipe_planning" / "post_lowering_verification_report.json",
            {
                "schema_version": "post_lowering_verification_report_v1",
                "status": "pass",
                "model_id": "tiny_mlp",
                "target_id": "host_cpu",
                "checks": [],
                "semantic_status": [],
                "failure_reasons": [],
            },
        )

    if include_differential_pass:
        _write(
            run_dir / "03_recipe_planning" / "differential_verification_report.json",
            {
                "schema_version": "differential_verification_report_v1",
                "status": "pass",
                "max_abs_error": 1e-6,
            },
        )

    return run_dir


def test_emit_promotes_when_evidence_passes(tmp_path: Path) -> None:
    """Happy path: structural + differential evidence present and pass."""
    run_dir = _build_minimal_run_dir(tmp_path)
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)

    assert result.status == "ok", result.reason
    assert result.recipe_path is not None
    assert result.recipe_path.exists()
    assert (result.recipe_path / "manifest.json").exists()
    assert (result.recipe_path / "promoted_recipe.json").exists()


def test_promoted_recipe_sidecar_carries_two_tier_key(tmp_path: Path) -> None:
    """The sidecar must record region_signature so M-28 retrieval can find it."""
    run_dir = _build_minimal_run_dir(tmp_path)
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)
    assert result.status == "ok"
    assert result.key is not None
    assert result.key.region_signature  # non-empty hash

    sidecar = json.loads(
        (result.recipe_path / "promoted_recipe.json").read_text()
    )
    assert sidecar["key"]["region_signature"] == result.key.region_signature
    assert sidecar["recipe"]["recipe_signature"] == result.key.region_signature
    assert sidecar["recipe"]["validity"]["op_family"] == "matmul"
    assert sidecar["recipe"]["validity"]["target_class"] == "host_cpu"


def test_synthesized_verification_report_lands_under_promotion_dir(
    tmp_path: Path,
) -> None:
    """The verification_report must land under 04_promotion/ to keep R009 monotonic."""
    run_dir = _build_minimal_run_dir(tmp_path)
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)
    assert result.status == "ok"
    # R009 protection: synthesized report does not pollute earlier stage dirs.
    assert not (run_dir / "verification_report.json").exists()
    assert (run_dir / "04_promotion" / "verification_report.json").exists()


def test_emit_skips_when_no_candidate_selected(tmp_path: Path) -> None:
    """Runs that didn't pick a candidate are not promotable — typed not_eligible."""
    run_dir = _build_minimal_run_dir(tmp_path, selected_candidate_id=None)
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)
    assert result.status == "not_eligible"
    assert "no candidate" in result.reason.lower()
    assert not library.exists() or not any(library.iterdir())


def test_emit_skips_when_no_differential_evidence(tmp_path: Path) -> None:
    """Without any Phase B differential evidence the bridge bails honestly."""
    run_dir = _build_minimal_run_dir(
        tmp_path,
        include_differential_pass=False,
        include_post_lowering_pass=False,
    )
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)
    assert result.status == "not_eligible"
    assert "evidence" in result.reason.lower()


def test_emit_skips_when_differential_did_not_pass(tmp_path: Path) -> None:
    """structural-only is not enough — the gate requires differential."""
    run_dir = _build_minimal_run_dir(tmp_path, include_differential_pass=False)
    library = tmp_path / "library"

    result = emit(run_dir, library_path=library)
    assert result.status == "not_eligible"
    assert "differential" in result.reason.lower()


def test_emit_returns_error_for_nonexistent_run_dir(tmp_path: Path) -> None:
    """Bridge never raises; missing inputs become typed not_eligible."""
    result = emit(tmp_path / "does_not_exist", library_path=tmp_path / "lib")
    assert result.status == "not_eligible"


def test_emit_bridges_to_compiler_memory(tmp_path: Path) -> None:
    """When a CompilerMemory is provided, the promotion is indexed in SQLite."""
    run_dir = _build_minimal_run_dir(tmp_path)
    library = tmp_path / "library"

    from compgen.memory.store import CompilerMemory

    memory = CompilerMemory(
        db_path=tmp_path / "memory.db",
        blob_root=tmp_path / "blobs",
    )

    result = emit(run_dir, library_path=library, memory=memory)
    assert result.status == "ok"

    rows = memory.db.fetchall(
        "SELECT promotion_key, region_signature FROM promotions"
    )
    assert len(rows) == 1
    assert rows[0]["region_signature"] == result.key.region_signature
    memory.close()
