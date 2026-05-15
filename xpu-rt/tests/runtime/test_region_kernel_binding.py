"""RegionKernelBinding tests.

Round-trip + cross-field validation + run_dir-strict validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.runtime.execution_plan import (
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)


def _minimal_plan(bindings: list[RegionKernelBinding]) -> ExecutionPlan:
    placements = [
        RegionPlacement(region_id=b.region_id, device="host_cpu", queue="q")
        for b in bindings
    ]
    return ExecutionPlan(
        workload="test", target="host_cpu",
        resources=[Resource(id="q", kind="compute", device="host_cpu")],
        region_placement=placements,
        region_kernel_bindings=bindings,
    )


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_from_dict_round_trip(self) -> None:
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="matmul_0", contract_hash="abc",
                certificate_path="04_kernel_codegen/certificates/abc.json",
                kernel_artifact="04_kernel_codegen/artifacts/k/kernel.c",
                dispatch_model="sync",
            ),
        ])
        d = plan.to_dict()
        round_tripped = ExecutionPlan.from_dict(d)
        assert len(round_tripped.region_kernel_bindings) == 1
        b = round_tripped.region_kernel_bindings[0]
        assert b.region_id == "matmul_0"
        assert b.contract_hash == "abc"

    def test_empty_bindings_field_round_trips(self) -> None:
        plan = _minimal_plan([])
        d = plan.to_dict()
        assert d["region_kernel_bindings"] == []
        rt = ExecutionPlan.from_dict(d)
        assert rt.region_kernel_bindings == []


# --------------------------------------------------------------------------- #
# Structural validation (no run_dir needed)
# --------------------------------------------------------------------------- #


class TestStructuralValidation:
    def test_duplicate_region_binding_rejected(self) -> None:
        # One placement, two bindings to the same region — duplicate
        # binding catch fires.
        plan = ExecutionPlan(
            workload="t", target="host_cpu",
            region_placement=[RegionPlacement(region_id="r0", device="host_cpu", queue="q")],
            resources=[Resource(id="q", kind="compute", device="host_cpu")],
            region_kernel_bindings=[
                RegionKernelBinding(
                    region_id="r0", contract_hash="a",
                    certificate_path="cert_a.json",
                ),
                RegionKernelBinding(
                    region_id="r0", contract_hash="b",
                    certificate_path="cert_b.json",
                ),
            ],
        )
        with pytest.raises(ValueError, match="duplicate region_kernel_binding"):
            plan.validate()

    def test_unknown_region_id_rejected(self) -> None:
        plan = ExecutionPlan(
            workload="t", target="host_cpu",
            region_placement=[RegionPlacement(region_id="r0", device="host_cpu", queue="q")],
            resources=[Resource(id="q", kind="compute", device="host_cpu")],
            region_kernel_bindings=[
                RegionKernelBinding(
                    region_id="rogue", contract_hash="a",
                    certificate_path="cert.json",
                ),
            ],
        )
        with pytest.raises(ValueError, match="references unknown region_id"):
            plan.validate()

    def test_empty_contract_hash_rejected(self) -> None:
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="",
                certificate_path="x",
            ),
        ])
        with pytest.raises(ValueError, match="contract_hash must be non-empty"):
            plan.validate()

    def test_empty_certificate_path_rejected(self) -> None:
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="a",
                certificate_path="",
            ),
        ])
        with pytest.raises(ValueError, match="certificate_path must be non-empty"):
            plan.validate()

    def test_unknown_dispatch_model_rejected(self) -> None:
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="a",
                certificate_path="cert.json",
                dispatch_model="warp_megakernel",  # not in (sync|async|persistent|inline)
            ),
        ])
        with pytest.raises(ValueError, match="dispatch_model"):
            plan.validate()


# --------------------------------------------------------------------------- #
# Run-dir-strict validation (loads + checks the certificate file)
# --------------------------------------------------------------------------- #


class TestRunDirValidation:
    def test_missing_certificate_file_rejected(self, tmp_path: Path) -> None:
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="abc",
                certificate_path="04_kernel_codegen/certificates/abc.json",
            ),
        ])
        with pytest.raises(ValueError, match="certificate file missing"):
            plan.validate_with_run_dir(tmp_path)

    def test_certificate_hash_mismatch_rejected(self, tmp_path: Path) -> None:
        cert_dir = tmp_path / "04_kernel_codegen" / "certificates"
        cert_dir.mkdir(parents=True)
        # Cert says contract_hash="abc" but binding says "xyz".
        (cert_dir / "xyz.json").write_text(json.dumps({
            "schema_version": "kernel_certificate_v1",
            "contract_hash": "abc",  # MISMATCHED
            "task_id": "t", "region_id": "r0", "candidate_id": "c",
            "accepted_at_utc": "x", "artifact_hashes": {}, "artifact_paths": {},
            "verifier_report_path": "", "verifier_report_hash": "",
            "claims": {},
        }))
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="xyz",
                certificate_path="04_kernel_codegen/certificates/xyz.json",
            ),
        ])
        with pytest.raises(ValueError, match="contract_hash"):
            plan.validate_with_run_dir(tmp_path)

    def test_well_formed_binding_validates(self, tmp_path: Path) -> None:
        cert_dir = tmp_path / "04_kernel_codegen" / "certificates"
        cert_dir.mkdir(parents=True)
        (cert_dir / "abc.json").write_text(json.dumps({
            "schema_version": "kernel_certificate_v1",
            "contract_hash": "abc",
            "task_id": "t", "region_id": "r0", "candidate_id": "c",
            "accepted_at_utc": "x", "artifact_hashes": {}, "artifact_paths": {},
            "verifier_report_path": "", "verifier_report_hash": "",
            "claims": {},
        }))
        plan = _minimal_plan([
            RegionKernelBinding(
                region_id="r0", contract_hash="abc",
                certificate_path="04_kernel_codegen/certificates/abc.json",
            ),
        ])
        # Must not raise.
        plan.validate_with_run_dir(tmp_path)
