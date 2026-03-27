"""Hardware calibration and profiling.

Runs microbenchmarks on actual hardware to populate cost model data in
a target profile. Calibration fills in the ``cost_model`` and
``calibration_data`` fields that documentation-only profiles lack.

Invariants:
    - Calibration results are deterministic (or averaged over runs).
    - Results are saved to YAML for reuse across pipeline runs.
    - Calibration never modifies the base profile -- it returns a new one.

TODO: Implement Calibrator with microbenchmark suite.
TODO: Implement memory bandwidth measurement.
TODO: Implement op-level latency measurement (matmul, conv, etc.).
TODO: Support remote calibration (SSH to device, run benchmarks, pull results).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.targets.schema import TargetProfile


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

    TODO: Implement run() that executes benchmarks on the target device.
    TODO: Implement each benchmark type (bandwidth, latency, throughput).
    """

    benchmarks: list[str] = field(default_factory=lambda: ["hbm_bandwidth", "matmul_fp16"])
    num_samples: int = 10

    def calibrate(self, profile: TargetProfile, device_handle: Any = None) -> CalibratedProfile:
        """Run calibration benchmarks and return an updated profile.

        Args:
            profile: Base target profile.
            device_handle: Optional handle to the target device.

        Returns:
            CalibratedProfile with measured data.

        TODO: Run each benchmark, collect results, update profile.
        """
        raise NotImplementedError("Calibrator.calibrate is not yet implemented")


def calibrate(profile: TargetProfile, device_handle: Any = None) -> CalibratedProfile:
    """Convenience function: calibrate a profile with defaults."""
    raise NotImplementedError("calibrate is not yet implemented")


__all__ = ["CalibratedProfile", "CalibrationResult", "Calibrator", "calibrate"]
