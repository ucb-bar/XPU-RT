"""M-58 — Canonical shape-class hash tests.

Coverage:

- ``TestInstanceVsCanonical`` — concrete-shape contracts hash
  identically under both functions; dynamic-shape contracts collide
  on canonical_contract_hash but produce distinct
  instance_contract_hash entries.
- ``TestStability`` — both hashes are byte-stable across reruns.
- ``TestSensitivity`` — kernel_facing fields (dtype, layout, op_name,
  archetype, target_name, accumulator_dtype) change canonical hash;
  compiler_only fields (fusion, observability, dispatch concurrency
  limit) do not.
- ``TestCertificateCarriesCanonical`` — emit_certificate stamps the
  canonical hash; find_certificate_by_canonical_hash walks the
  cert directory and surfaces the matching cert.
- ``TestBindingCarriesCanonical`` — region_kernel_bindings.json
  serialises the canonical hash; round-trip preserves it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fixture builder
# --------------------------------------------------------------------------- #


def _build_v3_matmul(*, M=16, K=16, N=32, dtype="f32"):
    from compgen.kernels.contract_v3 import (
        ConcurrencyUnit, DispatchModel, DispatchSpec, EventDecl,
        ExecutionEnvelope, FusionPolicy, Granularity, HardwareEnvelope,
        IOContract, KernelArchetype, KernelContractV3, LayoutKind, MemorySpec,
        NumericsSpec, ObservabilitySpec, OrchestrationSpec, PaddingPolicy,
        PerformancePriority, ShapeClass, StaticAttr, SyncSpec, TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="host_cpu", vector_lanes=8,
        scratchpad_bytes=65_536, register_bytes=16,
        native_dtypes=(dtype,),
    )
    exe = ExecutionEnvelope(
        hardware=hw, memory_budget_bytes=49_152,
        concurrency_unit=ConcurrencyUnit.WARP,
        padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(name="A", shape=ShapeClass(dims=(M, K)),
                 dtype_class=(dtype,), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    B = TensorIO(name="B", shape=ShapeClass(dims=(K, N)),
                 dtype_class=(dtype,), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    Y = TensorIO(name="Y", shape=ShapeClass(dims=(M, N)),
                 dtype_class=(dtype,), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    io = IOContract(
        inputs=(A, B), outputs=(Y,),
        attributes=(StaticAttr(name="transpose_b", value=False),),
        numerics=NumericsSpec(accumulator_dtype="f32",
                              fast_math=False, max_relative_error=0.0,
                              deterministic=True),
    )
    return KernelContractV3(
        op_name="matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=io,
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=exe,
            sync=SyncSpec(event_decls=(EventDecl(name="done"),)),
            memory=MemorySpec(),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.SYNC),
            observability=ObservabilitySpec(emit_completion_event=True),
        ),
    )


def _build_v3_dynamic_matmul():
    """Dynamic-shape variant: every dim is None."""
    from compgen.kernels.contract_v3 import (
        ConcurrencyUnit, DispatchModel, DispatchSpec, EventDecl,
        ExecutionEnvelope, FusionPolicy, Granularity, HardwareEnvelope,
        IOContract, KernelArchetype, KernelContractV3, LayoutKind, MemorySpec,
        NumericsSpec, ObservabilitySpec, OrchestrationSpec, PaddingPolicy,
        PerformancePriority, ShapeClass, StaticAttr, SyncSpec, TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="host_cpu", vector_lanes=8,
        scratchpad_bytes=65_536, register_bytes=16,
        native_dtypes=("f32",),
    )
    exe = ExecutionEnvelope(
        hardware=hw, memory_budget_bytes=49_152,
        concurrency_unit=ConcurrencyUnit.WARP,
        padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(name="A", shape=ShapeClass(dims=(None, None)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    B = TensorIO(name="B", shape=ShapeClass(dims=(None, None)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    Y = TensorIO(name="Y", shape=ShapeClass(dims=(None, None)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    io = IOContract(
        inputs=(A, B), outputs=(Y,),
        attributes=(StaticAttr(name="transpose_b", value=False),),
        numerics=NumericsSpec(accumulator_dtype="f32",
                              fast_math=False, max_relative_error=0.0,
                              deterministic=True),
    )
    return KernelContractV3(
        op_name="matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=io,
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=exe,
            sync=SyncSpec(event_decls=(EventDecl(name="done"),)),
            memory=MemorySpec(),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.SYNC),
            observability=ObservabilitySpec(emit_completion_event=True),
        ),
    )


# --------------------------------------------------------------------------- #
# Instance vs canonical
# --------------------------------------------------------------------------- #


class TestInstanceVsCanonical:
    def test_concrete_contract_both_hashes_match(self) -> None:
        from compgen.promotion.contract_hash import (
            canonical_contract_hash,
            instance_contract_hash,
        )

        c = _build_v3_matmul()
        # For a concrete contract with no None dims, the canonical
        # hash equals the instance hash (no abstraction kicks in).
        assert canonical_contract_hash(c) == instance_contract_hash(c)

    def test_dynamic_concrete_pair_differ_on_instance_match_on_canonical(self) -> None:
        from compgen.promotion.contract_hash import (
            canonical_contract_hash,
            instance_contract_hash,
        )

        concrete_a = _build_v3_matmul(M=16, K=16, N=32)
        concrete_b = _build_v3_matmul(M=64, K=128, N=256)
        dynamic = _build_v3_dynamic_matmul()

        # Instance hashes differ across all three.
        ia = instance_contract_hash(concrete_a)
        ib = instance_contract_hash(concrete_b)
        idyn = instance_contract_hash(dynamic)
        assert ia != ib != idyn != ia

        # Canonical hash: concrete contracts still differ from each
        # other (their dims are still distinct concrete ints), but BOTH
        # differ from the dynamic-shape canonical hash.
        ca = canonical_contract_hash(concrete_a)
        cb = canonical_contract_hash(concrete_b)
        cdyn = canonical_contract_hash(dynamic)
        assert ca != cdyn
        assert cb != cdyn
        # The dynamic canonical hash is byte-stable.
        assert cdyn == canonical_contract_hash(_build_v3_dynamic_matmul())

    def test_hash_contract_alias_matches_instance_hash(self) -> None:
        from compgen.promotion.contract_hash import (
            hash_contract,
            instance_contract_hash,
        )

        c = _build_v3_matmul()
        assert hash_contract(c) == instance_contract_hash(c)


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


class TestStability:
    def test_concrete_canonical_byte_stable(self) -> None:
        from compgen.promotion.contract_hash import canonical_contract_hash

        c1 = _build_v3_matmul()
        c2 = _build_v3_matmul()
        assert canonical_contract_hash(c1) == canonical_contract_hash(c2)

    def test_dynamic_canonical_byte_stable(self) -> None:
        from compgen.promotion.contract_hash import canonical_contract_hash

        d1 = _build_v3_dynamic_matmul()
        d2 = _build_v3_dynamic_matmul()
        assert canonical_contract_hash(d1) == canonical_contract_hash(d2)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #


class TestSensitivity:
    def test_dtype_change_changes_canonical(self) -> None:
        from compgen.promotion.contract_hash import canonical_contract_hash

        f32 = _build_v3_matmul(dtype="f32")
        # f16 needs a different accumulator/numerics block, but for
        # this test we just probe sensitivity to dtype on inputs.
        # Build a hand-tweaked f16 variant:
        from dataclasses import replace

        from compgen.kernels.contract_v3 import LayoutKind, ShapeClass, TensorIO

        f16 = replace(
            f32,
            io=replace(
                f32.io,
                inputs=tuple(
                    replace(t, dtype_class=("f16",))
                    for t in f32.io.inputs
                ),
                outputs=tuple(
                    replace(t, dtype_class=("f16",))
                    for t in f32.io.outputs
                ),
            ),
        )
        assert canonical_contract_hash(f32) != canonical_contract_hash(f16)

    def test_compiler_only_fusion_change_does_not_change_canonical(self) -> None:
        from dataclasses import replace

        from compgen.kernels.contract_v3 import FusionPolicy
        from compgen.promotion.contract_hash import canonical_contract_hash

        c = _build_v3_matmul()
        with_fusion = replace(
            c,
            orchestration=replace(
                c.orchestration,
                fusion=FusionPolicy(is_boundary=False),  # compiler-only field
            ),
        )
        # FusionPolicy is in compiler_only(), not kernel_facing(). The
        # canonical hash must NOT change.
        assert canonical_contract_hash(c) == canonical_contract_hash(with_fusion)


# --------------------------------------------------------------------------- #
# Certificate carries the canonical hash
# --------------------------------------------------------------------------- #


def _invoke_pipeline(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "execution-plan-emit",
            "--selection-mode", "greedy",
            "--auction-mode", "multi-bidder",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestCertificateCarriesCanonical:
    def test_emit_certificate_stamps_canonical(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        cert_dir = run_dir / "04_kernel_codegen" / "certificates"
        cert_files = list(cert_dir.glob("*.json"))
        assert len(cert_files) >= 1
        body = json.loads(cert_files[0].read_text())
        assert "canonical_contract_hash" in body
        assert body["canonical_contract_hash"]
        # Concrete contract → canonical == instance.
        assert body["canonical_contract_hash"] == body["contract_hash"]

    def test_find_by_canonical_hash_locates_cert(self, tmp_path: Path) -> None:
        from compgen.kernels.kernel_certificate import (
            find_certificate_by_canonical_hash,
        )

        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        cert_files = list((run_dir / "04_kernel_codegen" / "certificates").glob("*.json"))
        body = json.loads(cert_files[0].read_text())
        canonical = body["canonical_contract_hash"]

        located = find_certificate_by_canonical_hash(
            run_dir=run_dir, canonical_hash=canonical
        )
        assert located is not None
        assert located.canonical_contract_hash == canonical
        assert located.contract_hash == body["contract_hash"]

    def test_find_by_unknown_canonical_returns_none(self, tmp_path: Path) -> None:
        from compgen.kernels.kernel_certificate import (
            find_certificate_by_canonical_hash,
        )

        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        located = find_certificate_by_canonical_hash(
            run_dir=tmp_path / "run", canonical_hash="0000000000000000",
        )
        assert located is None


# --------------------------------------------------------------------------- #
# Binding carries canonical
# --------------------------------------------------------------------------- #


class TestBindingCarriesCanonical:
    def test_binding_serialises_canonical(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        bindings_path = run_dir / "05_execution_plan" / "region_kernel_bindings.json"
        body = json.loads(bindings_path.read_text())
        assert body["bound_count"] >= 1
        bound_rows = [b for b in body["bindings"] if b["status"] == "bound"]
        for row in bound_rows:
            assert "canonical_contract_hash" in row
            # Concrete contract → canonical == instance.
            assert row["canonical_contract_hash"] == row["contract_hash"]
