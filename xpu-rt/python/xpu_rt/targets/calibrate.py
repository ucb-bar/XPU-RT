"""Hardware calibration and profiling.

Runs microbenchmarks on actual hardware to populate cost model data in
a target profile. Calibration fills in the ``cost_model`` and
``calibration_data`` fields that documentation-only profiles lack.

Invariants:
    - Calibration results are deterministic (or averaged over runs).
    - Results are saved to YAML for reuse across pipeline runs.
    - Calibration never modifies the base profile -- it returns a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

from compgen.targets.schema import TargetProfile

log = structlog.get_logger()

_BENCHMARK_UNITS: dict[str, str] = {
    "hbm_bandwidth": "GB/s",
    "matmul_fp16": "TFLOPS",
    "matmul_fp32": "TFLOPS",
    "l2_bandwidth": "GB/s",
}


@runtime_checkable
class DeviceHandle(Protocol):
    """Protocol for device benchmark execution."""

    def run_benchmark(self, name: str, params: dict[str, Any] | None = None) -> float:
        """Run a named benchmark and return the measured value."""
        ...


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a single calibration benchmark.

    Attributes:
        benchmark: Benchmark name (e.g., "hbm_bandwidth", "matmul_fp16").
        value: Measured value.
        unit: Unit of measurement (e.g., "GB/s", "us", "TFLOPS").
        samples: Number of measurement samples.
        std_dev: Standard deviation across samples.
    """

    benchmark: str
    value: float
    unit: str
    samples: int = 1
    std_dev: float = 0.0


@dataclass(frozen=True)
class CalibratedProfile:
    """A target profile with calibration data filled in.

    Attributes:
        profile: The updated target profile.
        results: List of calibration results.
    """

    profile: TargetProfile
    results: list[CalibrationResult] = field(default_factory=list)


@dataclass
class Calibrator:
    """Hardware calibrator.

    Attributes:
        benchmarks: List of benchmark names to run.
        num_samples: Number of samples per benchmark.
    """

    benchmarks: list[str] = field(default_factory=lambda: ["hbm_bandwidth", "matmul_fp16"])
    num_samples: int = 10

    def calibrate(self, profile: TargetProfile, device_handle: Any = None) -> CalibratedProfile:
        """Run calibration benchmarks and return an updated profile.

        If *device_handle* implements :class:`DeviceHandle`, each benchmark is
        executed on real hardware.  Otherwise synthetic defaults are pulled from
        the profile's ``cost_model``.

        Args:
            profile: Base target profile.
            device_handle: Optional handle to the target device.

        Returns:
            CalibratedProfile with measured data.
        """
        results: list[CalibrationResult] = []
        for bench_name in self.benchmarks:
            if device_handle is not None:
                log.info("calibrator.run_benchmark", benchmark=bench_name, samples=self.num_samples)
                value = device_handle.run_benchmark(bench_name, {"num_samples": self.num_samples})
                unit = _BENCHMARK_UNITS.get(bench_name, "units")
                results.append(
                    CalibrationResult(
                        benchmark=bench_name,
                        value=value,
                        unit=unit,
                        samples=self.num_samples,
                    )
                )
            else:
                # Synthetic defaults from profile cost_model
                default_value = profile.cost_model.get(bench_name, 0.0)
                unit = _BENCHMARK_UNITS.get(bench_name, "units")
                results.append(
                    CalibrationResult(
                        benchmark=bench_name,
                        value=default_value,
                        unit=unit,
                        samples=0,
                    )
                )

        # Update profile calibration_data
        cal_data = dict(profile.calibration_data)
        for r in results:
            cal_data[r.benchmark] = r.value

        updated = TargetProfile(
            name=profile.name,
            schema_version=profile.schema_version,
            devices=profile.devices,
            interconnects=profile.interconnects,
            constraints=profile.constraints,
            cost_model=profile.cost_model,
            calibration_data=cal_data,
            metadata=profile.metadata,
        )
        return CalibratedProfile(profile=updated, results=results)


def calibrate(profile: TargetProfile, device_handle: Any = None) -> CalibratedProfile:
    """Convenience function: calibrate a profile with defaults."""
    return Calibrator().calibrate(profile, device_handle)


__all__ = ["CalibratedProfile", "CalibrationResult", "Calibrator", "DeviceHandle", "calibrate"]
