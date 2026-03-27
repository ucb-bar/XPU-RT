"""Tests for SoC/Zephyr runtime code generation."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.targetgen.hardware_spec import (
    AddressSpace,
    ExecutionModel,
    ExecutionModelSpec,
    HardwareSpec,
    MemoryModelSpec,
    PlatformSpec,
    RuntimeContractSpec,
)
from compgen.targetgen.load import load_hardware_spec
from compgen.targetgen.soc_codegen import (
    SocCodegenResult,
    generate_arena_allocator,
    generate_bare_metal_runtime,
    generate_dma_ops,
    generate_soc_runtime,
    generate_zephyr_project,
)

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


def _with_deployment_model(spec: HardwareSpec, deployment_model: str) -> HardwareSpec:
    """Return a copy of *spec* with a different deployment_model."""
    new_platform = PlatformSpec(
        vendor=spec.platform.vendor,
        family=spec.platform.family,
        chip_name=spec.platform.chip_name,
        host_arch=spec.platform.host_arch,
        deployment_model=deployment_model,
    )
    return HardwareSpec(
        name=spec.name,
        platform=new_platform,
        execution_model=spec.execution_model,
        isa=spec.isa,
        native_ops=spec.native_ops,
        engine_geometry=spec.engine_geometry,
        memory_model=spec.memory_model,
        numeric_contract=spec.numeric_contract,
        runtime_contract=spec.runtime_contract,
        verification_surface=spec.verification_surface,
        patches=spec.patches,
    )


def _make_zephyr_spec() -> HardwareSpec:
    """Create a minimal HW spec with deployment_model='zephyr'."""
    return HardwareSpec(
        name="test-zephyr-target",
        platform=PlatformSpec(
            vendor="test_vendor",
            family="rocc_accelerator",
            chip_name="test_chip",
            host_arch="riscv64",
            deployment_model="zephyr",
        ),
        execution_model=ExecutionModelSpec(
            model=ExecutionModel.ROCC_COPROCESSOR,
            thread_model="single_thread",
            dispatch_model="synchronous",
        ),
        memory_model=MemoryModelSpec(
            address_spaces=[
                AddressSpace(name="scratchpad", id=0, size_bytes=262144, dma_accessible=True),
                AddressSpace(name="accumulator", id=1, size_bytes=65536, dma_accessible=False),
                AddressSpace(name="dram", id=2, size_bytes=4294967296, dma_accessible=True),
            ],
            dma_model="2d",
            double_buffering=True,
            max_outstanding_dma=4,
        ),
        runtime_contract=RuntimeContractSpec(
            calling_convention="custom_abi",
            kernel_launch="function_call",
            synchronization="fence",
        ),
    )


class TestGenerateZephyrProject:
    """Test Zephyr project generation from hardware specs."""

    def test_generates_prj_conf(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        prj = result.output_dir / "prj.conf"
        assert prj.exists()
        content = prj.read_text()
        assert "CONFIG_MAIN_STACK_SIZE=" in content
        assert "CONFIG_HEAP_MEM_POOL_SIZE=" in content
        assert "CONFIG_DMA=y" in content

    def test_generates_app_overlay(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        overlay = result.output_dir / "app.overlay"
        assert overlay.exists()
        content = overlay.read_text()
        assert "scratchpad" in content
        assert "accumulator" in content
        assert "dram" in content

    def test_generates_cmake(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        cmake = result.output_dir / "CMakeLists.txt"
        assert cmake.exists()
        content = cmake.read_text()
        assert "find_package(Zephyr" in content
        assert "src/main.c" in content

    def test_generates_main_c(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        main_c = result.output_dir / "src" / "main.c"
        assert main_c.exists()
        content = main_c.read_text()
        assert "k_thread_create" in content
        assert "k_sem_take" in content
        assert "k_sem_give" in content
        assert "dispatch" in content

    def test_main_c_has_dma_thread(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        main_c = result.output_dir / "src" / "main.c"
        content = main_c.read_text()
        assert "dma_handler" in content
        assert "dma_request" in content
        assert "dma_complete" in content

    def test_result_lists_files(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        assert isinstance(result, SocCodegenResult)
        assert "prj.conf" in result.generated_files
        assert "app.overlay" in result.generated_files
        assert "CMakeLists.txt" in result.generated_files
        assert "src/main.c" in result.generated_files

    def test_no_dma_in_prj_conf_without_dma(self, tmp_path: Path) -> None:
        spec = HardwareSpec(
            name="no-dma-target",
            platform=PlatformSpec(
                vendor="test", family="test", chip_name="test",
                deployment_model="zephyr",
            ),
            execution_model=ExecutionModelSpec(model=ExecutionModel.SIMD_VECTOR),
            memory_model=MemoryModelSpec(
                address_spaces=[
                    AddressSpace(name="sram", id=0, size_bytes=65536),
                ],
                dma_model="none",
            ),
        )
        result = generate_zephyr_project(spec, tmp_path / "zephyr_out")
        content = (result.output_dir / "prj.conf").read_text()
        assert "CONFIG_DMA=y" not in content

    def test_from_rocc_exemplar(self, tmp_path: Path) -> None:
        """Generate Zephyr project from the RoCC exemplar (overriding deployment_model)."""
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        zephyr_spec = _with_deployment_model(spec, "zephyr")
        result = generate_zephyr_project(zephyr_spec, tmp_path / "rocc_zephyr")
        assert (result.output_dir / "prj.conf").exists()
        assert (result.output_dir / "app.overlay").exists()
        assert (result.output_dir / "CMakeLists.txt").exists()
        assert (result.output_dir / "src" / "main.c").exists()

        content = (result.output_dir / "prj.conf").read_text()
        assert "CONFIG_MAIN_STACK_SIZE=" in content
        assert "CONFIG_DMA=y" in content

        main_content = (result.output_dir / "src" / "main.c").read_text()
        assert "k_thread_create" in main_content


class TestGenerateBareMetal:
    """Test bare-metal runtime generation."""

    def test_generates_arena_alloc(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_bare_metal_runtime(spec, tmp_path / "bare_metal")
        arena = result.output_dir / "arena_alloc.c"
        assert arena.exists()
        content = arena.read_text()
        assert "scratchpad_pool" in content
        assert "accumulator_pool" in content
        assert "arena_alloc" in content
        assert "arena_reset_all" in content

    def test_generates_main_c(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_bare_metal_runtime(spec, tmp_path / "bare_metal")
        main_c = result.output_dir / "main.c"
        assert main_c.exists()
        content = main_c.read_text()
        assert "int main(void)" in content
        assert "arena_reset_all" in content
        assert "dispatch_one" in content

    def test_generates_linker_script(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_bare_metal_runtime(spec, tmp_path / "bare_metal")
        linker = result.output_dir / "linker.ld"
        assert linker.exists()
        content = linker.read_text()
        assert "MEMORY" in content
        assert "SECTIONS" in content
        assert "SCRATCHPAD" in content
        assert "DRAM" in content

    def test_generates_dma_ops(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_bare_metal_runtime(spec, tmp_path / "bare_metal")
        dma = result.output_dir / "dma_ops.c"
        assert dma.exists()
        content = dma.read_text()
        assert "dma_transfer" in content
        assert "dma_wait" in content
        assert "DMA_SRC_ADDR" in content
        # RoCC uses 2d DMA
        assert "dma_transfer_2d" in content

    def test_result_lists_files(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_bare_metal_runtime(spec, tmp_path / "bare_metal")
        assert "arena_alloc.c" in result.generated_files
        assert "main.c" in result.generated_files
        assert "linker.ld" in result.generated_files
        assert "dma_ops.c" in result.generated_files

    def test_from_npu_exemplar(self, tmp_path: Path) -> None:
        """Generate bare-metal from NPU text ISA exemplar."""
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        bm_spec = _with_deployment_model(spec, "bare_metal")
        result = generate_bare_metal_runtime(bm_spec, tmp_path / "npu_bare_metal")
        assert (result.output_dir / "arena_alloc.c").exists()
        assert (result.output_dir / "main.c").exists()
        assert (result.output_dir / "linker.ld").exists()
        # NPU uses nd_strided DMA
        assert (result.output_dir / "dma_ops.c").exists()
        dma_content = (result.output_dir / "dma_ops.c").read_text()
        assert "dma_transfer_nd" in dma_content

    def test_npu_polling_sync(self, tmp_path: Path) -> None:
        """NPU uses polling synchronization model."""
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        bm_spec = _with_deployment_model(spec, "bare_metal")
        # Override runtime contract to force polling sync
        bm_spec = HardwareSpec(
            name=bm_spec.name,
            platform=bm_spec.platform,
            execution_model=bm_spec.execution_model,
            isa=bm_spec.isa,
            native_ops=bm_spec.native_ops,
            engine_geometry=bm_spec.engine_geometry,
            memory_model=bm_spec.memory_model,
            numeric_contract=bm_spec.numeric_contract,
            runtime_contract=RuntimeContractSpec(
                calling_convention=spec.runtime_contract.calling_convention,
                kernel_launch=spec.runtime_contract.kernel_launch,
                synchronization="polling",
                memory_allocation=spec.runtime_contract.memory_allocation,
            ),
            verification_surface=bm_spec.verification_surface,
            patches=bm_spec.patches,
        )
        result = generate_bare_metal_runtime(bm_spec, tmp_path / "npu_polling")
        main_content = (result.output_dir / "main.c").read_text()
        assert "dispatch_pending" in main_content


class TestGenerateArenaAllocator:
    """Test standalone arena allocator generation."""

    def test_generates_file(self, tmp_path: Path) -> None:
        mem = MemoryModelSpec(
            address_spaces=[
                AddressSpace(name="sram", id=0, size_bytes=32768),
            ],
        )
        result = generate_arena_allocator(mem, tmp_path / "arena")
        assert (result.output_dir / "arena_alloc.c").exists()

    def test_arena_has_named_functions(self, tmp_path: Path) -> None:
        mem = MemoryModelSpec(
            address_spaces=[
                AddressSpace(name="sram", id=0, size_bytes=32768),
                AddressSpace(name="buffer", id=1, size_bytes=16384),
            ],
        )
        result = generate_arena_allocator(mem, tmp_path / "arena")
        content = (result.output_dir / "arena_alloc.c").read_text()
        assert "sram_alloc" in content
        assert "buffer_alloc" in content
        assert "sram_reset" in content
        assert "buffer_reset" in content


class TestGenerateDmaOps:
    """Test DMA operations generation."""

    def test_no_dma_produces_nothing(self, tmp_path: Path) -> None:
        mem = MemoryModelSpec(dma_model="none")
        result = generate_dma_ops(mem, tmp_path / "dma")
        assert result.generated_files == []

    def test_2d_dma(self, tmp_path: Path) -> None:
        mem = MemoryModelSpec(
            dma_model="2d",
            max_outstanding_dma=4,
            address_spaces=[AddressSpace(name="sram", id=0, size_bytes=32768)],
        )
        result = generate_dma_ops(mem, tmp_path / "dma")
        assert "dma_ops.c" in result.generated_files
        content = (result.output_dir / "dma_ops.c").read_text()
        assert "dma_transfer_2d" in content
        assert "DMA_SRC_STRIDE" in content
        assert "DMA_ROWS" in content

    def test_nd_strided_dma(self, tmp_path: Path) -> None:
        mem = MemoryModelSpec(
            dma_model="nd_strided",
            max_outstanding_dma=2,
            address_spaces=[AddressSpace(name="local", id=0, size_bytes=65536)],
        )
        result = generate_dma_ops(mem, tmp_path / "dma")
        content = (result.output_dir / "dma_ops.c").read_text()
        assert "dma_transfer_nd" in content
        assert "DMA_DIM1_STRIDE" in content
        assert "DMA_DIM1_COUNT" in content


class TestSocRuntimeDispatcher:
    """Test the top-level generate_soc_runtime dispatcher."""

    def test_dispatches_to_zephyr(self, tmp_path: Path) -> None:
        spec = _make_zephyr_spec()
        result = generate_soc_runtime(spec, tmp_path / "soc")
        assert (result.output_dir / "prj.conf").exists()
        assert (result.output_dir / "src" / "main.c").exists()

    def test_dispatches_to_bare_metal(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        result = generate_soc_runtime(spec, tmp_path / "soc")
        assert (result.output_dir / "arena_alloc.c").exists()
        assert (result.output_dir / "linker.ld").exists()

    def test_unsupported_deployment_model_raises(self, tmp_path: Path) -> None:
        spec = HardwareSpec(
            name="bad-deploy",
            platform=PlatformSpec(
                vendor="x", family="x", chip_name="x",
                deployment_model="linux_userspace",
            ),
        )
        with pytest.raises(ValueError, match="Unsupported deployment_model"):
            generate_soc_runtime(spec, tmp_path / "soc")
