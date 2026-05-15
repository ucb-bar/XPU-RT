"""Tests for ``compgen.kernels.envelope_bridge``.

Covers:

  * ``envelope_from_target_profile`` builds a usable envelope from a
    minimal hand-built profile (no filesystem dependency)
  * ``CODEGEN_HINTS`` appear on the envelope when the target name
    matches a known-target key
  * Unknown targets get an empty hint tuple — never raise
  * ``extra_hints`` parameter extends (doesn't replace) the authored list
  * The bridge works against the real YAML exemplars in tests/ so we
    get smoke coverage for at least one realistic profile
  * ``HardwareEnvelope.codegen_hints`` is surfaced through
    ``kernel_facing()`` — available to kernel codegen
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    OrchestrationSpec,
    ShapeClass,
    TensorIO,
)
from compgen.kernels.envelope_bridge import (
    CODEGEN_HINTS,
    envelope_from_target_profile,
)
from compgen.targets.schema import (
    ComputeUnit,
    DeviceSpec,
    MemoryLevel,
    TargetProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _a100_profile() -> TargetProfile:
    """Minimal A100-shaped profile sufficient for the bridge."""
    return TargetProfile(
        name="cuda-a100",
        devices=[
            DeviceSpec(
                device_type="gpu",
                name="A100",
                compute_units=[
                    ComputeUnit(
                        name="tensor_core",
                        count=432,
                        supported_dtypes={"bf16", "f16", "f32", "tf32"},
                        peak_tflops=312.0,
                    ),
                    ComputeUnit(
                        name="cuda_core",
                        count=6912,
                        supported_dtypes={"f32", "f64"},
                    ),
                ],
                memory_hierarchy=[
                    MemoryLevel(name="registers", size_bytes=256),
                    MemoryLevel(name="shared_memory", size_bytes=167936),
                    MemoryLevel(name="l2_cache", size_bytes=41943040),
                    MemoryLevel(name="hbm", size_bytes=80 * 1024**3, bandwidth_gbps=1555.0),
                ],
            )
        ],
    )


def _unknown_target_profile() -> TargetProfile:
    """A profile whose name has no authored hints."""
    return TargetProfile(
        name="exotic-fpga-x7",
        devices=[
            DeviceSpec(
                device_type="fpga",
                name="X7",
                compute_units=[ComputeUnit(name="dsp_block", count=64)],
                memory_hierarchy=[
                    MemoryLevel(name="bram", size_bytes=4 * 1024 * 1024),
                    MemoryLevel(name="ddr4", size_bytes=16 * 1024**3, bandwidth_gbps=25.6),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Bridge produces a well-shaped envelope
# ---------------------------------------------------------------------------


def test_envelope_has_target_and_vector_lanes_from_primary_compute_unit() -> None:
    env = envelope_from_target_profile(_a100_profile())
    assert env.target_name == "cuda-a100"
    # Primary unit preferred = tensor_core (count=432)
    assert env.vector_lanes == 432


def test_envelope_scratchpad_picks_on_chip_tier() -> None:
    """scratchpad_bytes should be the SMEM size (167936), not HBM."""
    env = envelope_from_target_profile(_a100_profile())
    assert env.scratchpad_bytes == 167936


def test_envelope_register_bytes_from_registers_tier() -> None:
    env = envelope_from_target_profile(_a100_profile())
    assert env.register_bytes == 256


def test_envelope_peak_bandwidth_from_largest_memory_tier() -> None:
    """Largest tier (HBM) dictates peak bandwidth."""
    env = envelope_from_target_profile(_a100_profile())
    assert env.peak_bandwidth_gbps == pytest.approx(1555.0)


def test_envelope_native_dtypes_union_across_compute_units() -> None:
    env = envelope_from_target_profile(_a100_profile())
    # Union of tensor_core + cuda_core dtype sets
    assert set(env.native_dtypes) == {"bf16", "f16", "f32", "f64", "tf32"}


# ---------------------------------------------------------------------------
# Codegen hints
# ---------------------------------------------------------------------------


def test_codegen_hints_present_for_known_target() -> None:
    env = envelope_from_target_profile(_a100_profile())
    assert env.codegen_hints
    # At least one hint mentions the canonical bf16 + f32 accumulate pattern.
    assert any("bf16" in h and "f32" in h for h in env.codegen_hints)


def test_codegen_hints_empty_for_unknown_target() -> None:
    """Unknown target name → empty hint tuple, no raise."""
    env = envelope_from_target_profile(_unknown_target_profile())
    assert env.codegen_hints == ()


def test_extra_hints_extends_authored_list() -> None:
    extra = ("Stick to row-major loads; column-major is a 3× penalty here.",)
    env = envelope_from_target_profile(_a100_profile(), extra_hints=extra)
    assert env.codegen_hints[-1] == extra[0]
    # Authored hints are still present (extras append, don't replace).
    assert len(env.codegen_hints) == len(CODEGEN_HINTS["cuda-a100"]) + 1


def test_known_targets_all_have_at_least_one_hint() -> None:
    """Safety: if a key is in CODEGEN_HINTS, it should carry real content."""
    for key, hints in CODEGEN_HINTS.items():
        assert hints, f"CODEGEN_HINTS[{key!r}] is empty"
        for h in hints:
            assert isinstance(h, str) and len(h) > 20, f"CODEGEN_HINTS[{key!r}] has an empty/short hint: {h!r}"


# ---------------------------------------------------------------------------
# Envelope flows through kernel_facing() — kernel codegen can read hints
# ---------------------------------------------------------------------------


def test_hints_reach_kernel_codegen_via_kernel_facing() -> None:
    env = envelope_from_target_profile(_a100_profile())
    contract = KernelContractV3(
        op_name="x",
        archetype=KernelArchetype.ACTIVATION,
        io=IOContract(
            inputs=(TensorIO(name="in", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
        ),
    )
    view = contract.kernel_facing()
    assert view.execution is not None
    assert view.execution.hardware.codegen_hints == env.codegen_hints


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_profile_with_no_devices_raises() -> None:
    profile = TargetProfile(name="empty", devices=[])
    with pytest.raises(ValueError, match="no devices"):
        envelope_from_target_profile(profile)


def test_device_index_out_of_range_raises() -> None:
    with pytest.raises(IndexError, match="device_index"):
        envelope_from_target_profile(_a100_profile(), device_index=3)


# ---------------------------------------------------------------------------
# Smoke — real YAML exemplars load and the bridge succeeds
# ---------------------------------------------------------------------------


EXEMPLAR_ROOT = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars"


@pytest.mark.parametrize(
    "yaml_name",
    [
        "test_gpu_simt.yaml",
    ],
)
def test_bridge_survives_real_yaml_exemplar(yaml_name: str) -> None:
    """The test/exemplar YAMLs that the existing suite uses must all
    survive the bridge without raising. We don't assert *specific* values
    here — just that the pipeline from YAML → envelope succeeds."""
    import tempfile

    from compgen.targetgen.generate import generate_target

    path = EXEMPLAR_ROOT / yaml_name
    pytest.importorskip("yaml")
    if not path.exists():
        pytest.skip(f"exemplar missing: {path}")

    with tempfile.TemporaryDirectory() as out_dir:
        generated = generate_target(path, out_dir)
        env = envelope_from_target_profile(generated.profile)
        assert env.target_name
        assert env.vector_lanes >= 1
        # Unknown-target hints is fine; just check it doesn't crash.
        assert isinstance(env.codegen_hints, tuple)
