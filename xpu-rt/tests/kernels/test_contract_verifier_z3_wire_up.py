"""wire-up: ``verify_kernel`` dispatches preconditions to Z3.

These tests prove that when a contract opts in via
``optional_v3_1_fields['z3_proof_required'] = True``, each supported
precondition becomes a real Z3 obligation,
``_verify_predicate_proof_via_z3`` invokes
:mod:`compgen.solve.z3_obligations` through the registered
:class:`Z3Backend`, and the typed response lands in a
``<task>.z3_obligations.json`` report (suitable for
``KernelCertificate.z3_obligation_report_ref``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

z3 = pytest.importorskip("z3")

from compgen.kernels.contract_v3 import KernelContractV3
from compgen.kernels.contract_verifier import (
    generate_obligations,
    verify_kernel,
    write_z3_obligation_report,
)
from compgen.kernels.predicates import ModEq, NumericalWithinEps


def _base_contract() -> KernelContractV3:
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


def _with_z3(contract: KernelContractV3, *, preconditions, z3_proof_required: bool) -> KernelContractV3:
    optional = dict(contract.optional_v3_1_fields)
    optional["z3_proof_required"] = z3_proof_required
    return replace(contract, preconditions=preconditions, optional_v3_1_fields=optional)


def _ok_metadata(c: KernelContractV3) -> dict:
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
        "declared_max_abs_error": 0.0,
        "declared_higham_bound": 1.0,
    }


def _ok_claims(c: KernelContractV3) -> dict:
    return {
        "backend": "c_reference",
        "supports_dispatch": [c.orchestration.dispatch.model.value],
        "expected_numerics": "bit_equality",
        "estimated_registers": 0,
        "estimated_smem_bytes": 0,
    }


def test_z3_obligation_not_added_when_opt_out():
    c = _with_z3(
        _base_contract(),
        preconditions=(ModEq(arg_dim="K", k=16),),
        z3_proof_required=False,
    )
    obls = generate_obligations(c)
    assert not any(o.verifier_kind == "predicate_proof_via_z3" for o in obls)


def test_z3_obligation_added_when_opt_in():
    c = _with_z3(
        _base_contract(),
        preconditions=(ModEq(arg_dim="K", k=16),),
        z3_proof_required=True,
    )
    obls = generate_obligations(c)
    z3_obls = [o for o in obls if o.verifier_kind == "predicate_proof_via_z3"]
    assert len(z3_obls) == 1
    assert z3_obls[0].expected["predicate_index"] == 0


def test_verify_kernel_proves_provable_precondition(tmp_path: Path):
    c = _with_z3(
        _base_contract(),
        # K%16==0 is a tautology under the harness's (no applies_when)
        # encoding only when the precondition is itself K%k==0 for k≤16.
        preconditions=(ModEq(arg_dim="K", k=1),),
        z3_proof_required=True,
    )
    metadata = tmp_path / "kernel_metadata.json"
    claims = tmp_path / "provider_claims.json"
    metadata.write_text(json.dumps(_ok_metadata(c)))
    claims.write_text(json.dumps(_ok_claims(c)))

    z3_report: dict = {
        "schema_version": "z3_obligations_index_v1",
        "task_id": "kt",
        "obligations": [],
    }
    report = verify_kernel(
        contract=c,
        task_id="kt",
        contract_hash="abc",
        kernel_metadata_path=metadata,
        provider_claims_path=claims,
        z3_obligation_report=z3_report,
    )

    z3v = [v for v in report.verdicts if v.verifier_kind == "predicate_proof_via_z3"]
    assert len(z3v) == 1
    assert z3v[0].status == "pass", z3v[0]

    assert len(z3_report["obligations"]) == 1
    entry = z3_report["obligations"][0]
    assert entry["status"] == "proved"
    assert entry["selected_backend"] == "z3"
    assert "formulation_hash" in entry


def test_verify_kernel_rejects_unprovable_precondition(tmp_path: Path):
    """``ModEq(K, 32)`` over K ∈ [1, 65536] without an applies_when is
    NOT a tautology; Z3 returns a counterexample (e.g. K=1) and the
    verifier marks the obligation ``fail``."""

    c = _with_z3(
        _base_contract(),
        preconditions=(ModEq(arg_dim="K", k=32),),
        z3_proof_required=True,
    )
    metadata = tmp_path / "kernel_metadata.json"
    claims = tmp_path / "provider_claims.json"
    metadata.write_text(json.dumps(_ok_metadata(c)))
    claims.write_text(json.dumps(_ok_claims(c)))

    z3_report: dict = {"obligations": []}
    report = verify_kernel(
        contract=c,
        task_id="kt",
        contract_hash="abc",
        kernel_metadata_path=metadata,
        provider_claims_path=claims,
        z3_obligation_report=z3_report,
    )
    z3v = [v for v in report.verdicts if v.verifier_kind == "predicate_proof_via_z3"]
    assert len(z3v) == 1
    assert z3v[0].status == "fail"
    assert "counterexample" in z3v[0].detail.lower()
    entry = z3_report["obligations"][0]
    assert entry["status"] == "sat_counterexample"
    assert entry["counterexample"] is not None


def test_unsupported_predicate_marks_deferred(tmp_path: Path):
    """``NumericalWithinEps`` has no Z3 lowering yet → ``deferred`` honestly."""

    c = _with_z3(
        _base_contract(),
        preconditions=(NumericalWithinEps(out="Y", ref="ref", eps=1e-4),),
        z3_proof_required=True,
    )
    metadata = tmp_path / "kernel_metadata.json"
    claims = tmp_path / "provider_claims.json"
    metadata.write_text(json.dumps(_ok_metadata(c)))
    claims.write_text(json.dumps(_ok_claims(c)))

    z3_report: dict = {"obligations": []}
    report = verify_kernel(
        contract=c,
        task_id="kt",
        contract_hash="abc",
        kernel_metadata_path=metadata,
        provider_claims_path=claims,
        z3_obligation_report=z3_report,
    )
    z3v = [v for v in report.verdicts if v.verifier_kind == "predicate_proof_via_z3"]
    assert len(z3v) == 1
    assert z3v[0].status == "deferred"
    assert z3_report["obligations"] == []


def test_persisted_z3_report_path_matches_cert_field_convention(tmp_path: Path):
    """The persisted JSON path matches the convention used in
    ``KernelCertificate.z3_obligation_report_ref``: relative to
    ``<run_dir>/`` it is
    ``04_kernel_codegen/solver/<task>.z3_obligations.json``."""

    c = _with_z3(
        _base_contract(),
        preconditions=(ModEq(arg_dim="K", k=1),),
        z3_proof_required=True,
    )
    metadata = tmp_path / "kernel_metadata.json"
    claims = tmp_path / "provider_claims.json"
    metadata.write_text(json.dumps(_ok_metadata(c)))
    claims.write_text(json.dumps(_ok_claims(c)))

    z3_report: dict = {
        "schema_version": "z3_obligations_index_v1",
        "task_id": "kt",
        "obligations": [],
    }
    verify_kernel(
        contract=c,
        task_id="kt",
        contract_hash="abc",
        kernel_metadata_path=metadata,
        provider_claims_path=claims,
        z3_obligation_report=z3_report,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = write_z3_obligation_report(run_dir=run_dir, task_id="kt", body=z3_report)
    rel = path.relative_to(run_dir)
    assert str(rel) == "04_kernel_codegen/solver/kt.z3_obligations.json"
    body = json.loads(path.read_text())
    assert body["obligations"][0]["status"] == "proved"
