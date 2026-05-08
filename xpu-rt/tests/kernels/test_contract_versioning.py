"""M-64 — Refinement contracts + version migration tests.

Coverage:

- ``TestOptionalV3_1Field`` — the optional_v3_1_fields slot
  round-trips through contract_to_dict + reconstructor without
  affecting the canonical or instance hash.
- ``TestMigration`` — migrate_contract_body_v3_to_v3_1 fills
  defaults for missing keys; idempotent on already-v3.1 bodies.
- ``TestUnknownFieldRejected`` — get_optional_v3_1_field with an
  unrecognised name raises typed ContractRefinementError.
- ``TestPreM64BodyLoadsCleanly`` — a v3 cert body emitted before
  M-64 (no optional_v3_1_fields key) still loads via the
  reconstruct path; canonical hash stays byte-identical to a fresh
  v3.1-emitted body.
- ``TestAuditGate`` — the contract_version_consistency gate flags
  any cert whose canonical hash drifts after migration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_v3_matmul():
    from compgen.kernels.contract_v3 import (
        ConcurrencyUnit, DispatchModel, DispatchSpec, EventDecl,
        ExecutionEnvelope, FusionPolicy, Granularity, HardwareEnvelope,
        IOContract, KernelArchetype, KernelContractV3, LayoutKind, MemorySpec,
        NumericsSpec, ObservabilitySpec, OrchestrationSpec, PaddingPolicy,
        PerformancePriority, ShapeClass, StaticAttr, SyncSpec, TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="host_cpu", vector_lanes=8, scratchpad_bytes=65_536,
        register_bytes=16, native_dtypes=("f32",),
    )
    exe = ExecutionEnvelope(
        hardware=hw, memory_budget_bytes=49_152,
        concurrency_unit=ConcurrencyUnit.WARP,
        padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(name="lhs", shape=ShapeClass(dims=(16, 16)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    B = TensorIO(name="rhs", shape=ShapeClass(dims=(16, 32)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    Y = TensorIO(name="out", shape=ShapeClass(dims=(16, 32)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    return KernelContractV3(
        op_name="linalg.matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(A, B), outputs=(Y,),
            attributes=(StaticAttr(name="t_b", value=False),),
            numerics=NumericsSpec(
                accumulator_dtype="f32", fast_math=False,
                max_relative_error=0.0, deterministic=True,
            ),
        ),
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=exe,
            sync=SyncSpec(event_decls=(EventDecl(name="d"),)),
            memory=MemorySpec(),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.SYNC),
            observability=ObservabilitySpec(emit_completion_event=True),
        ),
    )


# --------------------------------------------------------------------------- #
# Optional field exists + doesn't change hashes
# --------------------------------------------------------------------------- #


class TestOptionalV3_1Field:
    def test_default_is_empty_dict(self) -> None:
        c = _build_v3_matmul()
        assert c.optional_v3_1_fields == {}

    def test_setting_field_does_not_change_hashes(self) -> None:
        from dataclasses import replace

        from compgen.promotion.contract_hash import (
            canonical_contract_hash,
            instance_contract_hash,
        )

        c1 = _build_v3_matmul()
        c2 = replace(c1, optional_v3_1_fields={"prefetch_distance": 16})

        # Both hashes invariant under the new field.
        assert canonical_contract_hash(c1) == canonical_contract_hash(c2)
        assert instance_contract_hash(c1) == instance_contract_hash(c2)

    def test_changing_kernel_facing_still_changes_hash(self) -> None:
        """Sanity: hash invariance is M-64-specific, not a regression."""
        from dataclasses import replace

        from compgen.kernels.contract_v3 import LayoutKind
        from compgen.promotion.contract_hash import canonical_contract_hash

        c1 = _build_v3_matmul()
        c2 = replace(
            c1,
            io=replace(
                c1.io,
                inputs=tuple(
                    replace(t, layout=LayoutKind.COLUMN_MAJOR)
                    for t in c1.io.inputs
                ),
            ),
        )
        # Different layout → different canonical hash.
        assert canonical_contract_hash(c1) != canonical_contract_hash(c2)


# --------------------------------------------------------------------------- #
# Migration helper
# --------------------------------------------------------------------------- #


class TestMigration:
    def test_migrate_fills_defaults(self) -> None:
        from compgen.kernels.contract_migration import (
            migrate_contract_body_v3_to_v3_1,
        )

        # Pre-M-64 body has no optional_v3_1_fields key at all.
        v3_body = {"op_name": "x"}
        migrated = migrate_contract_body_v3_to_v3_1(v3_body)
        assert "optional_v3_1_fields" in migrated
        assert migrated["optional_v3_1_fields"]["prefetch_distance"] == 0
        assert migrated["optional_v3_1_fields"]["pin_inputs_to_cpu"] is False

    def test_migration_is_idempotent(self) -> None:
        from compgen.kernels.contract_migration import (
            migrate_contract_body_v3_to_v3_1,
        )

        v3_body = {
            "op_name": "x",
            "optional_v3_1_fields": {"prefetch_distance": 8},
        }
        migrated = migrate_contract_body_v3_to_v3_1(v3_body)
        # User-supplied value is preserved.
        assert migrated["optional_v3_1_fields"]["prefetch_distance"] == 8
        # Missing field gets the default.
        assert migrated["optional_v3_1_fields"]["pin_inputs_to_cpu"] is False
        # Calling again is a no-op.
        again = migrate_contract_body_v3_to_v3_1(migrated)
        assert again == migrated

    def test_migration_does_not_mutate_input(self) -> None:
        from compgen.kernels.contract_migration import (
            migrate_contract_body_v3_to_v3_1,
        )

        v3_body = {"op_name": "x"}
        snapshot = dict(v3_body)
        migrate_contract_body_v3_to_v3_1(v3_body)
        assert v3_body == snapshot  # defensive copy


# --------------------------------------------------------------------------- #
# Unknown field rejected
# --------------------------------------------------------------------------- #


class TestUnknownFieldRejected:
    def test_unknown_name_raises(self) -> None:
        from compgen.kernels.contract_migration import (
            ContractRefinementError,
            get_optional_v3_1_field,
        )

        c = _build_v3_matmul()
        with pytest.raises(ContractRefinementError, match="unknown v3.1 optional"):
            get_optional_v3_1_field(c, "invent_new_field")

    def test_recognized_field_without_value_returns_default(self) -> None:
        from compgen.kernels.contract_migration import get_optional_v3_1_field

        c = _build_v3_matmul()
        assert get_optional_v3_1_field(c, "prefetch_distance") == 0
        assert get_optional_v3_1_field(c, "pin_inputs_to_cpu") is False

    def test_recognized_field_with_value_returns_value(self) -> None:
        from dataclasses import replace

        from compgen.kernels.contract_migration import get_optional_v3_1_field

        c = _build_v3_matmul()
        c = replace(c, optional_v3_1_fields={"prefetch_distance": 32})
        assert get_optional_v3_1_field(c, "prefetch_distance") == 32


# --------------------------------------------------------------------------- #
# Pre-M-64 bodies still load + same hash
# --------------------------------------------------------------------------- #


def _invoke_pipeline(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-codegen-request",
            "--selection-mode", "greedy",
            "--auction-mode", "disabled",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestPreM64BodyLoadsCleanly:
    def test_body_without_optional_slot_loads_with_defaults(
        self, tmp_path: Path,
    ) -> None:
        """Simulate a pre-M-64 cert body: drop the optional_v3_1_fields
        key + verify the reconstruct path fills it in via migration,
        and the canonical hash stays the same."""
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        contract_path = list(
            (run_dir / "04_kernel_codegen" / "contracts").glob("*.json")
        )[0]
        body_with = json.loads(contract_path.read_text())
        # Strip M-64 slot.
        body_without = dict(body_with)
        body_without.pop("optional_v3_1_fields", None)

        from compgen.graph_compilation.kernel_codegen_response import (
            _reconstruct_contract_from_dict,
        )
        from compgen.promotion.contract_hash import (
            canonical_contract_hash,
            instance_contract_hash,
        )

        c_with = _reconstruct_contract_from_dict(body_with)
        c_without = _reconstruct_contract_from_dict(body_without)

        # Both reconstruct cleanly + share canonical + instance hashes.
        assert canonical_contract_hash(c_with) == canonical_contract_hash(c_without)
        assert instance_contract_hash(c_with) == instance_contract_hash(c_without)

        # Migration filled the slot for the pre-M-64 body.
        assert "prefetch_distance" in c_without.optional_v3_1_fields
        assert c_without.optional_v3_1_fields["prefetch_distance"] == 0


# --------------------------------------------------------------------------- #
# Audit gate
# --------------------------------------------------------------------------- #


class TestAuditGate:
    def test_gate_passes_on_clean_run(self, tmp_path: Path) -> None:
        """contract_version_consistency gate: every cert's canonical
        hash matches a fresh re-derivation."""
        # End-to-end pipeline → at least one cert.
        result = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
                "--out", str(tmp_path / "run"),
                "--stop-after", "execution-plan-emit",
                "--selection-mode", "greedy",
                "--auction-mode", "multi-bidder",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        from compgen.kernels.contract_migration import (
            migrate_contract_body_v3_to_v3_1,
        )
        from compgen.graph_compilation.kernel_codegen_response import (
            _reconstruct_contract_from_dict,
        )
        from compgen.promotion.contract_hash import canonical_contract_hash

        run_dir = tmp_path / "run"
        cert_dir = run_dir / "04_kernel_codegen" / "certificates"
        certs = list(cert_dir.glob("*.json"))
        assert certs

        for cp in certs:
            cert_body = json.loads(cp.read_text())
            recorded = cert_body.get("canonical_contract_hash", "")
            # Re-derive from the cert's source contract.
            contract_rel = cert_body.get("contract_path") or ""
            assert contract_rel
            contract_body = json.loads(
                (run_dir / contract_rel).read_text()
            )
            migrated = migrate_contract_body_v3_to_v3_1(contract_body)
            contract = _reconstruct_contract_from_dict(migrated)
            re_derived = canonical_contract_hash(contract)
            assert recorded == re_derived, (
                f"cert {cp.name} canonical_contract_hash {recorded!r} != "
                f"re-derived {re_derived!r} (M-64 migration would have "
                f"broken the cache)"
            )
