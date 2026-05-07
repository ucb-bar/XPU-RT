"""M-44 contract-driven verifier tests.

Layered coverage:
- Schema: generate_obligations produces one obligation per applicable
  contract field.
- Per-verifier: each obligation kind catches the right tampering
  with the right typed failure_kind.
- E2E: a real merlin_mlp_wide contract + tampered metadata yields the
  expected verdict + on-disk validation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.kernels.contract_v3 import KernelContractV3
from compgen.kernels.contract_verifier import (
    generate_obligations,
    verify_kernel,
    write_validation_report,
)


def _build_contract() -> KernelContractV3:
    return KernelContractV3.from_recipe(
        candidate_selection={
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_x",
            "region_id": "matmul_0",
            "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            "recipe_delta": [{"op": "SetTileParams", "M": 16, "N": 16, "K": 16}],
            "target_id": "host_cpu",
        },
        region_dossier={
            "region_id": "matmul_0",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 16], [16, 32]],
                "output_shapes": [[16, 32]],
            },
        },
        target_profile={"target_id": "host_cpu"},
        declared_refinement="bit_equality",
    )


def _correct_metadata(c: KernelContractV3) -> dict:
    io = c.io
    return {
        "inputs": [
            {"dims": list(t.shape.dims), "dtype": t.dtype_class[0],
             "layout": t.layout.value}
            for t in io.inputs
        ],
        "outputs": [
            {"dims": list(t.shape.dims), "dtype": t.dtype_class[0],
             "layout": t.layout.value}
            for t in io.outputs
        ],
        "accumulator_dtype": io.numerics.accumulator_dtype,
        "target_name": (
            c.orchestration.execution.hardware.target_name
            if c.orchestration.execution else ""
        ),
        "signals_emitted": {
            e.name: e.wait_count
            for e in c.orchestration.sync.event_decls
        },
    }


def _correct_claims(c: KernelContractV3) -> dict:
    return {
        "backend": "c_reference",
        "supports_dispatch": [c.orchestration.dispatch.model.value],
        "expected_numerics": "bit_equality",
        "estimated_registers": 0,
        "estimated_smem_bytes": 0,
    }


# --------------------------------------------------------------------------- #
# Schema — obligation generation
# --------------------------------------------------------------------------- #


class TestObligationGeneration:
    def test_obligation_count_matches_contract_fields(self) -> None:
        c = _build_contract()
        obls = generate_obligations(c)
        # 2 inputs × (shape + dtype + layout) = 6
        # 1 output × shape = 1
        # accumulator + differential + deterministic = 3
        # 1 event × signalled_once = 1
        # 2 input_tiers + 1 output_tier = 3
        # dispatch + target_name = 2
        # Total = 16
        assert len(obls) == 16, [o.obl_id for o in obls]

    def test_obligation_ids_are_unique(self) -> None:
        c = _build_contract()
        ids = [o.obl_id for o in generate_obligations(c)]
        assert len(set(ids)) == len(ids)

    def test_dispatch_obligation_present(self) -> None:
        c = _build_contract()
        obls = generate_obligations(c)
        assert any(o.verifier_kind == "dispatch_model_match" for o in obls)


# --------------------------------------------------------------------------- #
# Per-verifier failure modes
# --------------------------------------------------------------------------- #


class TestPerVerifierFailures:
    def test_shape_mismatch_typed_correctly(self, tmp_path: Path) -> None:
        c = _build_contract()
        meta = _correct_metadata(c)
        meta["inputs"][0]["dims"] = [99, 99]  # tampered
        (tmp_path / "kernel_metadata.json").write_text(json.dumps(meta))
        (tmp_path / "provider_claims.json").write_text(
            json.dumps(_correct_claims(c))
        )
        report = verify_kernel(
            contract=c, task_id="kspec_test", contract_hash="abcd",
            kernel_metadata_path=tmp_path / "kernel_metadata.json",
            provider_claims_path=tmp_path / "provider_claims.json",
        )
        assert report.overall == "fail"
        assert report.failure_kind == "shape_mismatch"

    def test_metadata_mismatch_on_wrong_dtype(self, tmp_path: Path) -> None:
        c = _build_contract()
        meta = _correct_metadata(c)
        meta["inputs"][0]["dtype"] = "f16"  # contract says f32
        (tmp_path / "kernel_metadata.json").write_text(json.dumps(meta))
        (tmp_path / "provider_claims.json").write_text(
            json.dumps(_correct_claims(c))
        )
        report = verify_kernel(
            contract=c, task_id="t", contract_hash="abcd",
            kernel_metadata_path=tmp_path / "kernel_metadata.json",
            provider_claims_path=tmp_path / "provider_claims.json",
        )
        assert report.overall == "fail"
        assert report.failure_kind == "metadata_mismatch"

    def test_dispatch_mismatch_typed_correctly(self, tmp_path: Path) -> None:
        c = _build_contract()
        meta = _correct_metadata(c)
        claims = _correct_claims(c)
        claims["supports_dispatch"] = ["async"]  # contract says sync
        (tmp_path / "kernel_metadata.json").write_text(json.dumps(meta))
        (tmp_path / "provider_claims.json").write_text(json.dumps(claims))
        report = verify_kernel(
            contract=c, task_id="t", contract_hash="abcd",
            kernel_metadata_path=tmp_path / "kernel_metadata.json",
            provider_claims_path=tmp_path / "provider_claims.json",
        )
        assert report.overall == "fail"
        assert report.failure_kind == "semantic_contract_violation"

    def test_correct_metadata_passes(self, tmp_path: Path) -> None:
        c = _build_contract()
        (tmp_path / "kernel_metadata.json").write_text(
            json.dumps(_correct_metadata(c))
        )
        (tmp_path / "provider_claims.json").write_text(
            json.dumps(_correct_claims(c))
        )
        report = verify_kernel(
            contract=c, task_id="t", contract_hash="abcd",
            kernel_metadata_path=tmp_path / "kernel_metadata.json",
            provider_claims_path=tmp_path / "provider_claims.json",
        )
        # All obligations either pass or are deferred (no fails).
        assert report.overall in ("pass", "deferred")
        assert report.failure_kind == ""

    def test_missing_metadata_file_returns_metadata_mismatch(
        self, tmp_path: Path,
    ) -> None:
        c = _build_contract()
        report = verify_kernel(
            contract=c, task_id="t", contract_hash="abcd",
            kernel_metadata_path=tmp_path / "missing.json",
            provider_claims_path=tmp_path / "missing_claims.json",
        )
        assert report.overall == "fail"
        assert report.failure_kind == "metadata_mismatch"


# --------------------------------------------------------------------------- #
# E2E — write the validation report
# --------------------------------------------------------------------------- #


def test_write_validation_report_lands_at_expected_path(tmp_path: Path) -> None:
    c = _build_contract()
    (tmp_path / "kernel_metadata.json").write_text(
        json.dumps(_correct_metadata(c))
    )
    (tmp_path / "provider_claims.json").write_text(
        json.dumps(_correct_claims(c))
    )
    report = verify_kernel(
        contract=c, task_id="kspec_e2e", contract_hash="cafe1234",
        kernel_metadata_path=tmp_path / "kernel_metadata.json",
        provider_claims_path=tmp_path / "provider_claims.json",
    )
    out = write_validation_report(
        run_dir=tmp_path, task_id="kspec_e2e", report=report,
    )
    assert out == tmp_path / "04_kernel_codegen" / "validation" / "kspec_e2e.validation.json"
    assert out.exists()
    body = json.loads(out.read_text())
    assert body["task_id"] == "kspec_e2e"
    assert body["contract_hash"] == "cafe1234"
    assert "obligations" in body
    assert "verdicts" in body
    assert len(body["obligations"]) == len(body["verdicts"])
