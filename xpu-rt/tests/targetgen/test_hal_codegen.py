"""Tests for HAL driver code generation.

For each of the 5 exemplar YAML specs we generate the driver files
and verify they are non-empty, syntactically plausible, and contain
the expected public function names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from compgen.targetgen.hal_codegen import (
    _infer_alloc_strategy,
    _infer_launch_mechanism,
    _infer_sync_mechanism,
    generate_hal_driver,
)
from compgen.targetgen.load import load_hardware_spec

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"

# Mapping: exemplar stem → (expected alloc, expected launch, expected sync)
EXPECTED_STRATEGIES: dict[str, tuple[str, str, str]] = {
    "test_rvv_cpu": ("malloc", "function_call", "none"),
    "test_matrix_ext": ("malloc", "inline_intrinsic", "fence"),
    "test_rocc_accel": ("scratchpad", "rocc_instruction", "fence_dma"),
    "test_npu_text_isa": ("firmware_managed", "mailbox", "polling"),
    "test_gpu_simt": ("device_alloc", "command_queue", "event"),
}

# Public C functions that must appear in the generated driver set.
REQUIRED_FUNCTIONS = [
    "hal_alloc",
    "hal_free",
    "hal_dispatch_kernel",
    "hal_sync",
    "hal_init",
    "hal_shutdown",
    "hal_get_vtable",
]


@pytest.fixture(params=sorted(EXEMPLAR_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def exemplar(request: pytest.FixtureRequest) -> Path:
    """Parametrise over every exemplar YAML."""
    return request.param


class TestStrategyInference:
    """Verify that strategy inference maps each exemplar correctly."""

    @pytest.mark.parametrize(
        "yaml_stem, expected",
        [(k, v) for k, v in EXPECTED_STRATEGIES.items()],
        ids=list(EXPECTED_STRATEGIES),
    )
    def test_alloc_strategy(self, yaml_stem: str, expected: tuple[str, str, str]) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / f"{yaml_stem}.yaml")
        assert _infer_alloc_strategy(spec) == expected[0]

    @pytest.mark.parametrize(
        "yaml_stem, expected",
        [(k, v) for k, v in EXPECTED_STRATEGIES.items()],
        ids=list(EXPECTED_STRATEGIES),
    )
    def test_launch_mechanism(self, yaml_stem: str, expected: tuple[str, str, str]) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / f"{yaml_stem}.yaml")
        assert _infer_launch_mechanism(spec) == expected[1]

    @pytest.mark.parametrize(
        "yaml_stem, expected",
        [(k, v) for k, v in EXPECTED_STRATEGIES.items()],
        ids=list(EXPECTED_STRATEGIES),
    )
    def test_sync_mechanism(self, yaml_stem: str, expected: tuple[str, str, str]) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / f"{yaml_stem}.yaml")
        assert _infer_sync_mechanism(spec) == expected[2]


class TestGenerateHalDriver:
    """Verify generated C files for every exemplar."""

    def test_generates_three_files(self, exemplar: Path, tmp_path: Path) -> None:
        spec = load_hardware_spec(exemplar)
        files = generate_hal_driver(spec, tmp_path / exemplar.stem)
        assert len(files) == 3
        for f in files:
            assert f.exists()
            assert f.stat().st_size > 0

    def test_expected_file_names(self, exemplar: Path, tmp_path: Path) -> None:
        spec = load_hardware_spec(exemplar)
        files = generate_hal_driver(spec, tmp_path / exemplar.stem)
        names = {f.name for f in files}
        assert names == {"hal_driver.c", "hal_allocator.c", "hal_dispatch.c"}

    def test_contains_required_functions(self, exemplar: Path, tmp_path: Path) -> None:
        spec = load_hardware_spec(exemplar)
        out = tmp_path / exemplar.stem
        generate_hal_driver(spec, out)
        combined = ""
        for c_file in out.glob("*.c"):
            combined += c_file.read_text()
        for fn_name in REQUIRED_FUNCTIONS:
            assert fn_name in combined, f"Missing function {fn_name} in {exemplar.name}"

    def test_target_name_in_header(self, exemplar: Path, tmp_path: Path) -> None:
        spec = load_hardware_spec(exemplar)
        out = tmp_path / exemplar.stem
        generate_hal_driver(spec, out)
        driver = (out / "hal_driver.c").read_text()
        assert spec.name in driver

    def test_no_unresolved_placeholders(self, exemplar: Path, tmp_path: Path) -> None:
        """No stray ``{placeholder}`` patterns should remain."""
        spec = load_hardware_spec(exemplar)
        out = tmp_path / exemplar.stem
        generate_hal_driver(spec, out)
        for c_file in out.glob("*.c"):
            text = c_file.read_text()
            unresolved = re.findall(r"\{[a-z_]+\}", text)
            assert not unresolved, (
                f"Unresolved placeholders in {c_file.name}: {unresolved}"
            )


class TestFamilySpecificOutput:
    """Each family produces distinct driver code."""

    def test_rvv_cpu_uses_malloc(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rvv_cpu.yaml")
        generate_hal_driver(spec, tmp_path)
        alloc = (tmp_path / "hal_allocator.c").read_text()
        assert "malloc" in alloc

    def test_vendor_matrix_uses_fence(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_matrix_ext.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "fence" in dispatch

    def test_rocc_uses_scratchpad(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        generate_hal_driver(spec, tmp_path)
        alloc = (tmp_path / "hal_allocator.c").read_text()
        assert "scratchpad" in alloc

    def test_rocc_uses_rocc_instruction(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "ROCC_INSTRUCTION" in dispatch

    def test_npu_uses_mailbox(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "mailbox" in dispatch.lower()

    def test_npu_uses_polling(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "status_reg" in dispatch

    def test_gpu_uses_command_queue(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "queue_enqueue" in dispatch

    def test_gpu_uses_device_alloc(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        generate_hal_driver(spec, tmp_path)
        alloc = (tmp_path / "hal_allocator.c").read_text()
        assert "device_malloc" in alloc

    def test_gpu_uses_event_sync(self, tmp_path: Path) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        generate_hal_driver(spec, tmp_path)
        dispatch = (tmp_path / "hal_dispatch.c").read_text()
        assert "event_synchronize" in dispatch
