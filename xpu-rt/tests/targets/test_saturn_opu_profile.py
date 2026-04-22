"""Tests for the Saturn OPU target profile + HardwareSpec wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.targetgen.classify import TargetFamily, classify_hardware
from compgen.targetgen.generate import generate_target
from compgen.targetgen.load import load_hardware_spec
from compgen.targetgen.validate_spec import validate_hardware_spec
from compgen.targets.backends.saturn_opu import SaturnOPUBackend, SaturnOPUOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SATURN_SPEC = REPO_ROOT / "examples" / "hardware_specs" / "saturn_opu.yaml"


@pytest.fixture
def saturn_spec():
    return load_hardware_spec(SATURN_SPEC)


def test_hardware_spec_validates(saturn_spec) -> None:
    report = validate_hardware_spec(saturn_spec)
    assert report.valid, f"Saturn OPU spec failed validation: {report.errors}"


def test_classifies_as_rvv_cpu_extension(saturn_spec) -> None:
    cls = classify_hardware(saturn_spec)
    # Saturn OPU is an in-vector-unit extension, not a RoCC coprocessor,
    # so the RVV family is the correct lowering path.
    assert cls.family is TargetFamily.RVV_CPU_EXTENSION
    assert cls.confidence >= 0.9


def test_features_include_xopu(saturn_spec) -> None:
    exts = {ext.name for ext in saturn_spec.isa.extensions}
    assert {"Xopu", "XopuMmt4d", "Zvl128b"}.issubset(exts)
    # OPU tile shape lives on the ``opu_outer_product_tile`` tile entry.
    tiles = {t.name: t for t in saturn_spec.engine_geometry.tiles}
    assert "opu_outer_product_tile" in tiles
    assert tiles["opu_outer_product_tile"].dimensions == [16, 16]
    assert saturn_spec.engine_geometry.vector_length_bits == 128


def test_generate_target_end_to_end(saturn_spec, tmp_path) -> None:
    generated = generate_target(SATURN_SPEC, tmp_path / "out")
    assert generated.profile.name == "saturn-opu-v128d64"
    assert len(generated.dialect_stack.stages) >= 4
    dev = generated.profile.devices[0]
    assert "Xopu" in dev.features
    # RVV CPU extension family extracts as device_type="cpu"; the OPU
    # accelerator nature is carried in ``features`` (Xopu, XopuMmt4d).
    assert dev.device_type in {"cpu", "accelerator"}


def test_saturn_backend_supports_target() -> None:
    backend = SaturnOPUBackend()
    assert backend.supports_target("saturn-opu-v128d64")
    assert backend.supports_target("saturn_opu")
    assert not backend.supports_target("nvidia-a100")


def test_saturn_backend_options_match_chipyard_config() -> None:
    opts = SaturnOPUOptions()
    assert opts.vector_length_bits == 128
    assert opts.opu_tile_m == 16 and opts.opu_tile_n == 16 and opts.opu_tile_k == 128
    assert opts.chipyard_config == "OPUV128D64ShuttleConfig"
    assert "+xopu" in opts.mcpu_features
    assert opts.target_triple == "riscv64-unknown-elf"
