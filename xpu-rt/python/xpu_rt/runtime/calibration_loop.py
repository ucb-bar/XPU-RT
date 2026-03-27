"""Calibration loop -- detect cost-model drift and trigger re-solve.

Monitors the gap between cost-model estimates and actual hardware measurements.
When the drift exceeds a configurable threshold (default 20 %), the loop
triggers a re-solve using calibrated costs from
:class:`~compgen.agent.memory.CostCalibration` (EMA-corrected factors).

Invariants:
    - Drift is computed as ``|estimated - measured| / measured``.
    - EMA correction is applied via :meth:`CostCalibration.update`.
    - Re-solve is triggered only when drift exceeds the threshold.
    - The loop is stateless between calls (all state lives in CostCalibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.agent.memory import CostCalibration
from compgen.runtime.planner import ExecutionPlan, ExecutionPlanner
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()

DEFAULT_DRIFT_THRESHOLD: float = 0.20


@dataclass(frozen=True)
class DriftResult:
    """Result of a drift check for a single operation.

    Attributes:
        op_type: The operation type that was measured.
        device_name: Device the operation ran on.
        estimated_us: Cost-model estimate in microseconds.
        measured_us: Actual measured latency in microseconds.
        drift: Relative drift (|est - meas| / meas).
        exceeded: Whether drift exceeds the threshold.
    """

    op_type: str
    device_name: str
    estimated_us: float
    measured_us: float
    drift: float
    exceeded: bool


@dataclass(frozen=True)
class CalibrationResult:
    """Aggregate result of a calibration loop iteration.

    Attributes:
        drift_results: Per-op drift measurements.
        max_drift: Maximum drift observed across all ops.
        threshold: The configured drift threshold.
        re_solve_triggered: Whether a re-solve was triggered.
        new_plan: Updated execution plan (only if re-solve was triggered).
    """

    drift_results: list[DriftResult] = field(default_factory=list)
    max_drift: float = 0.0
    threshold: float = DEFAULT_DRIFT_THRESHOLD
    re_solve_triggered: bool = False
    new_plan: ExecutionPlan | None = None


@dataclass
class CalibrationLoop:
    """Detect cost-model drift and re-solve with calibrated costs.

    Attributes:
        target: Hardware target profile.
        calibration: Persistent cost calibration data (EMA factors).
        drift_threshold: Relative drift threshold to trigger re-solve.
    """

    target: TargetProfile
    calibration: CostCalibration = field(default_factory=CostCalibration)
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD

    def check_drift(
        self,
        measurements: dict[str, dict[str, float]],
        estimates: dict[str, dict[str, float]],
    ) -> list[DriftResult]:
        """Compare estimates to measurements and compute per-op drift.

        Args:
            measurements: ``{device_name: {op_type: measured_us}}``.
            estimates: ``{device_name: {op_type: estimated_us}}``.

        Returns:
            List of :class:`DriftResult`, one per (device, op) pair.
        """
        results: list[DriftResult] = []
        for device_name, ops in measurements.items():
            est_ops = estimates.get(device_name, {})
            for op_type, measured_us in ops.items():
                estimated_us = est_ops.get(op_type, 0.0)
                if measured_us <= 0:
                    continue

                drift = abs(estimated_us - measured_us) / measured_us
                exceeded = drift > self.drift_threshold

                results.append(DriftResult(
                    op_type=op_type,
                    device_name=device_name,
                    estimated_us=estimated_us,
                    measured_us=measured_us,
                    drift=drift,
                    exceeded=exceeded,
                ))

                log.debug(
                    "calibration.drift",
                    device=device_name,
                    op=op_type,
                    drift=round(drift, 4),
                    exceeded=exceeded,
                )

        return results

    def update_calibration(
        self,
        drift_results: list[DriftResult],
    ) -> None:
        """Apply EMA correction to the calibration factors.

        Args:
            drift_results: Results from :meth:`check_drift`.
        """
        for dr in drift_results:
            if dr.estimated_us > 0:
                self.calibration.update(
                    device_name=dr.device_name,
                    op_type=dr.op_type,
                    estimated_us=dr.estimated_us,
                    measured_us=dr.measured_us,
                )

    def maybe_resolve(
        self,
        module: ModuleOp,
        max_drift: float,
        kernels: dict[str, Any] | None = None,
    ) -> ExecutionPlan | None:
        """Re-solve if *max_drift* exceeds the threshold.

        Args:
            module: The xDSL module to re-plan.
            max_drift: Maximum drift observed (pre-computed by caller).
            kernels: Optional generated kernels.

        Returns:
            A new :class:`ExecutionPlan` if re-solve was triggered,
            ``None`` otherwise.
        """
        if max_drift <= self.drift_threshold:
            return None

        log.info(
            "calibration.re_solve",
            max_drift=round(max_drift, 4),
            threshold=self.drift_threshold,
        )

        planner = ExecutionPlanner(target=self.target)
        return planner.plan(module, kernels)

    def run(
        self,
        module: ModuleOp,
        measurements: dict[str, dict[str, float]],
        estimates: dict[str, dict[str, float]],
        kernels: dict[str, Any] | None = None,
    ) -> CalibrationResult:
        """Full calibration loop: check drift, update EMA, optionally re-solve.

        Args:
            module: The xDSL module.
            measurements: ``{device_name: {op_type: measured_us}}``.
            estimates: ``{device_name: {op_type: estimated_us}}``.
            kernels: Optional generated kernels.

        Returns:
            :class:`CalibrationResult` with drift info and optional new plan.
        """
        drift_results = self.check_drift(measurements, estimates)
        self.update_calibration(drift_results)

        max_drift = max((dr.drift for dr in drift_results), default=0.0)

        new_plan = self.maybe_resolve(module, max_drift, kernels)

        return CalibrationResult(
            drift_results=drift_results,
            max_drift=max_drift,
            threshold=self.drift_threshold,
            re_solve_triggered=new_plan is not None,
            new_plan=new_plan,
        )


__all__ = [
    "CalibrationLoop",
    "CalibrationResult",
    "DEFAULT_DRIFT_THRESHOLD",
    "DriftResult",
]
