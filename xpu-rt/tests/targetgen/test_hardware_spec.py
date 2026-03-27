"""Tests for the HardwareSpec data model."""

from __future__ import annotations

from compgen.targetgen.hardware_spec import (
    AddressSpace,
    DtypeSupport,
    EngineGeometrySpec,
    ExecutionModel,
    ExecutionModelSpec,
    HardwareSpec,
    ISAExtension,
    ISASpec,
    MemoryModelSpec,
    NativeOpFamily,
    NativeOpsSpec,
    NumericContractSpec,
    PatchRequirement,
    PatchSpec,
    PlatformSpec,
    TileGeometry,
)


class TestPlatformSpec:
    def test_defaults(self) -> None:
        p = PlatformSpec(vendor="acme", family="test", chip_name="chip1")
        assert p.vendor == "acme"
        assert p.host_arch == "riscv64"
        assert p.deployment_model == "linux_userspace"


class TestExecutionModel:
    def test_all_values(self) -> None:
        assert len(ExecutionModel) == 7
        assert ExecutionModel.SIMT_GPU.value == "simt_gpu"
        assert ExecutionModel.ROCC_COPROCESSOR.value == "rocc_coprocessor"

    def test_spec_defaults(self) -> None:
        s = ExecutionModelSpec(model=ExecutionModel.SIMD_VECTOR)
        assert s.thread_model == "single_thread"
        assert not s.has_scoreboard


class TestISASpec:
    def test_with_extensions(self) -> None:
        isa = ISASpec(
            base_isa="rv64gcv",
            extensions=[ISAExtension(name="Zvl256b")],
        )
        assert len(isa.extensions) == 1
        assert isa.compiler_intrinsics is True


class TestNativeOps:
    def test_with_families(self) -> None:
        ops = NativeOpsSpec(families=[
            NativeOpFamily(name="matmul", ops=["matmul", "batch_matmul"]),
        ])
        assert len(ops.families) == 1
        assert ops.families[0].fallback == "decompose"


class TestEngineGeometry:
    def test_systolic(self) -> None:
        g = EngineGeometrySpec(systolic_array_dim=[16, 16])
        assert g.systolic_array_dim == [16, 16]

    def test_vector(self) -> None:
        g = EngineGeometrySpec(vector_length_bits=256)
        assert g.vector_length_bits == 256

    def test_tiles(self) -> None:
        g = EngineGeometrySpec(tiles=[TileGeometry(name="t0", dimensions=[8, 8])])
        assert g.tiles[0].dimensions == [8, 8]


class TestMemoryModel:
    def test_address_spaces(self) -> None:
        m = MemoryModelSpec(address_spaces=[
            AddressSpace(name="scratchpad", size_bytes=256 * 1024),
            AddressSpace(name="dram", size_bytes=4 * 1024**3),
        ])
        assert len(m.address_spaces) == 2
        assert m.coherence == "coherent"


class TestNumericContract:
    def test_dtypes(self) -> None:
        n = NumericContractSpec(supported_dtypes=[
            DtypeSupport(name="int8", accumulator_dtype="int32"),
            DtypeSupport(name="float32"),
        ])
        assert len(n.supported_dtypes) == 2


class TestPatchSpec:
    def test_with_requirements(self) -> None:
        p = PatchSpec(
            requirements=[PatchRequirement(component="ir", description="Add accel ops")],
            new_dialects_needed=["test_accel"],
        )
        assert len(p.requirements) == 1
        assert p.new_dialects_needed == ["test_accel"]


class TestHardwareSpec:
    def test_minimal(self) -> None:
        spec = HardwareSpec(name="test")
        assert spec.name == "test"
        assert spec.schema_version == "2.0"
        assert spec.execution_model.model == ExecutionModel.SIMD_VECTOR

    def test_full(self) -> None:
        spec = HardwareSpec(
            name="full-test",
            platform=PlatformSpec(vendor="v", family="f", chip_name="c"),
            execution_model=ExecutionModelSpec(model=ExecutionModel.ROCC_COPROCESSOR),
            isa=ISASpec(base_isa="rv64gc"),
            native_ops=NativeOpsSpec(families=[NativeOpFamily(name="matmul", ops=["matmul"])]),
            engine_geometry=EngineGeometrySpec(systolic_array_dim=[16, 16]),
            memory_model=MemoryModelSpec(address_spaces=[AddressSpace(name="sp", size_bytes=1024)]),
            numeric_contract=NumericContractSpec(supported_dtypes=[DtypeSupport(name="int8")]),
        )
        assert spec.execution_model.model == ExecutionModel.ROCC_COPROCESSOR
        assert spec.engine_geometry.systolic_array_dim == [16, 16]
