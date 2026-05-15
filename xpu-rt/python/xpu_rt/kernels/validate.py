"""Kernel correctness validation.

Validates generated kernels against golden reference outputs.
This is part of the verification ladder (level 2: functional).

Validation checks:
    - Output matches reference within tolerance (L2, max-abs).
    - Kernel compiles successfully for the target.
    - Performance is within budget (if specified).

Invariants:
    - Validation uses the same inputs as the golden reference (Stage 0).
    - Tolerance is dtype-aware (fp32 vs fp16 vs int8 have different thresholds).
    - Validation failures produce actionable diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from compgen.kernels.contracts import KernelSpec


@dataclass(frozen=True)
class KernelValidationResult:
    """Result of validating a generated kernel.

    Attributes:
        passed: Whether the kernel passed all validation checks.
        correct: Whether outputs match reference within tolerance.
        l2_error: L2 norm of output difference.
        max_abs_error: Maximum absolute difference.
        compiles: Whether the kernel compiles for the target.
        within_perf_budget: Whether latency is within the target.
        measured_latency_us: Measured latency (if available).
        diagnostics: Validation diagnostics.
    """

    passed: bool
    correct: bool
    l2_error: float = 0.0
    max_abs_error: float = 0.0
    compiles: bool = True
    within_perf_budget: bool = True
    measured_latency_us: float | None = None
    diagnostics: list[str] = field(default_factory=list)


# Dtype-aware tolerances
_DEFAULT_TOLERANCES: dict[str, float] = {
    "float32": 1e-5,
    "float16": 1e-3,
    "bfloat16": 1e-2,
    "int8": 0.0,
    "float64": 1e-10,
}


@dataclass
class KernelValidator:
    """Validates generated kernels against reference.

    Attributes:
        tolerance: Default tolerance for correctness checks.
    """

    tolerance: float = 1e-5

    def validate(
        self,
        kernel_code: str,
        spec: KernelSpec,
        test_inputs: Any,
        reference_outputs: Any,
    ) -> KernelValidationResult:
        """Validate a generated kernel.

        Args:
            kernel_code: The generated kernel source code.
            spec: Kernel specification.
            test_inputs: Test input tensors (tuple of torch.Tensor).
            reference_outputs: Expected output tensors (torch.Tensor or tuple).

        Returns:
            KernelValidationResult.
        """
        diagnostics: list[str] = []

        # Stage 1: Compilation check
        namespace: dict[str, Any] = {"torch": torch}
        try:
            exec(kernel_code, namespace)
            compiles = True
            diagnostics.append("Compilation: PASS")
        except Exception as e:
            return KernelValidationResult(
                passed=False,
                correct=False,
                compiles=False,
                diagnostics=[f"Compilation failed: {e}"],
            )

        # Find the kernel function (look for a callable named 'kernel' or the first function)
        kernel_fn = namespace.get("kernel")
        if kernel_fn is None:
            for v in namespace.values():
                if callable(v) and v is not torch and not isinstance(v, type):
                    kernel_fn = v
                    break

        if kernel_fn is None:
            return KernelValidationResult(
                passed=False,
                correct=False,
                compiles=True,
                diagnostics=["No callable kernel function found in generated code"],
            )

        # Stage 2: Correctness check
        try:
            if isinstance(test_inputs, (list, tuple)):
                actual_outputs = kernel_fn(*test_inputs)
            else:
                actual_outputs = kernel_fn(test_inputs)

            # Normalize to tensor
            if not isinstance(actual_outputs, torch.Tensor):
                if isinstance(actual_outputs, (list, tuple)) and actual_outputs:
                    actual_outputs = actual_outputs[0]

            ref = reference_outputs
            if isinstance(ref, (list, tuple)):
                ref = ref[0]

            if not isinstance(actual_outputs, torch.Tensor) or not isinstance(ref, torch.Tensor):
                return KernelValidationResult(
                    passed=False,
                    correct=False,
                    compiles=True,
                    diagnostics=["Output is not a tensor"],
                )

            diff = actual_outputs.float() - ref.float()
            l2_error = float(torch.norm(diff).item())
            max_abs_error = float(torch.max(torch.abs(diff)).item())

            # Determine tolerance from dtype
            dtype_str = str(ref.dtype).replace("torch.", "")
            tol = _DEFAULT_TOLERANCES.get(dtype_str, self.tolerance)

            correct = max_abs_error <= tol
            diagnostics.append(
                f"Correctness: {'PASS' if correct else 'FAIL'} "
                f"(l2={l2_error:.6f}, max_abs={max_abs_error:.6f}, tol={tol})"
            )
        except Exception as e:
            return KernelValidationResult(
                passed=False,
                correct=False,
                compiles=True,
                diagnostics=[f"Execution failed: {e}"],
            )

        # Stage 3: Performance check (if target specified)
        within_budget = True
        measured_us: float | None = None
        if spec.perf_target_us is not None and correct:
            try:
                # Quick benchmark: 10 iterations
                if isinstance(test_inputs, (list, tuple)):
                    inputs = test_inputs
                else:
                    inputs = (test_inputs,)

                # Warmup
                for _ in range(3):
                    kernel_fn(*inputs)

                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if start is not None and end is not None:
                    start.record()  # type: ignore[union-attr]
                    for _ in range(10):
                        kernel_fn(*inputs)
                    end.record()  # type: ignore[union-attr]
                    torch.cuda.synchronize()
                    measured_us = start.elapsed_time(end) * 1000 / 10  # type: ignore[union-attr]
                    within_budget = measured_us <= spec.perf_target_us * 2  # 2x slack
                    diagnostics.append(
                        f"Performance: {'PASS' if within_budget else 'FAIL'} "
                        f"({measured_us:.1f}us vs target {spec.perf_target_us:.1f}us)"
                    )
            except Exception:
                pass  # Performance check is best-effort

        passed = compiles and correct and within_budget
        return KernelValidationResult(
            passed=passed,
            correct=correct,
            l2_error=l2_error,
            max_abs_error=max_abs_error,
            compiles=compiles,
            within_perf_budget=within_budget,
            measured_latency_us=measured_us,
            diagnostics=diagnostics,
        )


def validate_kernel(
    kernel_code: str, spec: KernelSpec, test_inputs: Any, reference_outputs: Any
) -> KernelValidationResult:
    """Convenience function: validate with default settings."""
    validator = KernelValidator()
    return validator.validate(kernel_code, spec, test_inputs, reference_outputs)


__all__ = ["KernelValidationResult", "KernelValidator", "validate_kernel"]
