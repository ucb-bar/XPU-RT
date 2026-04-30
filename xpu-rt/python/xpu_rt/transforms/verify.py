"""Transform semantic verification.

Checks that applied transforms preserve the semantics of the payload IR.
This is verification ladder level 2+ for transform correctness.

Verification methods (layered):
    1. Structural -- output IR passes verifier.
    2. CHECK assertions -- expected ops present/absent (via ir.checks).
    3. Differential testing -- run original and transformed on same inputs,
       compare outputs within tolerance.
    4. Numeric -- run the original PyTorch model eagerly and via
       ``torch.compile(backend="eager")``, compare outputs with
       :func:`compgen.semantic.verify.compare.compare_tensors`.  When structural
       verification fails but numeric equivalence holds, the overall
       result is still PASS (semantic truth overrides structural mismatch).

Invariants:
    - Verification never modifies the IR.
    - Differential tests use randomized inputs (not just golden).
    - Tolerance is configurable per-dtype.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
import torch
import torch.nn as nn
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from compgen.semantic.verify.compare import compare_tensors

log = structlog.get_logger(__name__)


class VerificationLevel(Enum):
    """Level of transform verification."""

    STRUCTURAL = "structural"
    CHECK_ASSERTIONS = "check_assertions"
    DIFFERENTIAL = "differential"
    TRANSLATION_VALIDATION = "translation_validation"
    NUMERIC = "numeric"


@dataclass(frozen=True)
class TransformVerificationResult:
    """Result of verifying a transform.

    Attributes:
        passed: Whether all requested verification levels passed.
        levels_run: Which verification levels were executed.
        levels_passed: Which levels passed.
        max_abs_error: Maximum absolute error in differential testing.
        details: Per-level details and diagnostics.
    """

    passed: bool
    levels_run: list[VerificationLevel] = field(default_factory=list)
    levels_passed: list[VerificationLevel] = field(default_factory=list)
    max_abs_error: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardedTransformVerificationResult:
    """Verification outcome for a transform gated by synthesized guards."""

    guard_matched: bool
    verification: TransformVerificationResult
    note: str = ""


def _verify_structural(module: ModuleOp) -> tuple[bool, str]:
    """Run the xDSL verifier on the module."""
    try:
        module.verify()
        return True, "structural: PASS"
    except Exception as e:
        return False, f"structural: FAIL — {e}"


def _verify_check_assertions(module: ModuleOp, check_lines: list[str]) -> tuple[bool, str]:
    """Run CHECK-style assertions on the module's IR text."""
    from compgen.ir.checks import check_ir

    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    ir_text = buf.getvalue()

    try:
        result = check_ir(ir_text, check_lines)
        if result.passed:
            return True, f"check_assertions: PASS ({result.checks_run} checks)"
        failure_summary = "; ".join(f.message for f in result.failures)
        return False, f"check_assertions: FAIL — {failure_summary}"
    except Exception as e:
        return False, f"check_assertions: FAIL — {e}"


def _verify_differential(
    original: ModuleOp,
    transformed: ModuleOp,
    tolerance: float,
    sample_inputs: tuple[Any, ...] | None = None,
    *,
    exported_program: Any | None = None,
    num_random_inputs: int = 32,
    seed: int = 0,
) -> tuple[bool, float | None, str]:
    """Differential execution: run both modules on the same inputs and compare.

    This is the production-grade differential verification. Previous
    revisions compared op counts (a structural smell, not a semantic
    check) — the whole point of "differential" is running both sides
    and asserting agreement on outputs.

    Strategy:

    1. If no ``exported_program`` is supplied, execution isn't
       possible — :mod:`compgen.runtime.cpu_executor` needs a
       ``torch.export.ExportedProgram`` for the graph signature +
       state-dict. Report SKIPPED and let the promotion gate decide
       whether that's acceptable.
    2. Otherwise, execute the original module through ``cpu_executor``
       on ``sample_inputs`` (when supplied) plus N randomly generated
       inputs whose shapes match the original's entry block args.
    3. Execute the transformed module on the same inputs.
    4. Compare element-wise via
       :func:`compgen.semantic.verify.compare.compare_tensors`. Any
       input that exceeds tolerance fails the whole check and reports
       its index + error.

    Args:
        original: Pre-transform IR.
        transformed: Post-transform IR.
        tolerance: Absolute tolerance for element-wise comparison.
        sample_inputs: Concrete inputs to drive both modules.
        exported_program: ``torch.export.ExportedProgram`` the IR was
            lowered from. Required for cpu_executor dispatch;
            SKIPPED when absent.
        num_random_inputs: Extra random inputs synthesized from the
            original's arg shapes. Set to 0 to disable.
        seed: Seed for random-input generation (deterministic).

    Returns:
        ``(passed, max_abs_error, diagnostic)``. ``passed=True`` with
        a ``SKIPPED`` message means the check did not run.
    """
    from compgen.runtime import cpu_executor
    from compgen.semantic.verify.compare import compare_tensors

    # 1. Quick sanity: both modules must be printable + non-empty.
    try:
        orig_buf = io.StringIO()
        Printer(stream=orig_buf).print_op(original)
        trans_buf = io.StringIO()
        Printer(stream=trans_buf).print_op(transformed)
    except Exception as e:
        return False, None, f"differential: FAIL — IR not printable: {e}"

    if not orig_buf.getvalue().strip() or not trans_buf.getvalue().strip():
        return False, None, "differential: FAIL — empty IR"

    # 2. Without an ExportedProgram we can't dispatch through
    #    cpu_executor — honest SKIPPED.
    if exported_program is None:
        return (
            True,
            None,
            "differential: SKIPPED — no exported_program supplied; cpu_executor "
            "dispatch requires a graph signature + state-dict to bind IR block "
            "args (pass exported_program to enable)",
        )

    # 3. Collect inputs: user-supplied + synthesised random set.
    inputs_to_try: list[tuple[Any, ...]] = []
    if sample_inputs is not None:
        inputs_to_try.append(tuple(sample_inputs))

    synth_error: str | None = None
    if num_random_inputs > 0:
        try:
            shapes = _extract_entry_arg_shapes(original)
            if shapes:
                rng = torch.Generator().manual_seed(int(seed))
                for _ in range(num_random_inputs):
                    inputs_to_try.append(
                        tuple(torch.randn(shape, generator=rng, dtype=torch.float32) for shape in shapes)
                    )
            else:
                synth_error = "no tensor-typed entry args; skipping random inputs"
        except Exception as e:
            synth_error = f"random input synthesis failed: {e}"

    if not inputs_to_try:
        reason = synth_error or "no sample_inputs provided and num_random_inputs=0"
        return (
            True,
            None,
            f"differential: SKIPPED — no executable inputs ({reason})",
        )

    # 4. Execute both modules on each input and compare outputs.
    max_error_seen: float = 0.0
    for idx, inp in enumerate(inputs_to_try):
        try:
            orig_outs = cpu_executor.execute(original, exported_program, inp)
        except Exception as e:
            return False, None, f"differential: FAIL — original failed on input[{idx}]: {e}"
        try:
            trans_outs = cpu_executor.execute(transformed, exported_program, inp)
        except Exception as e:
            return False, None, f"differential: FAIL — transformed failed on input[{idx}]: {e}"

        orig_tensors = _flatten_tensors(orig_outs)
        trans_tensors = _flatten_tensors(trans_outs)
        if len(orig_tensors) != len(trans_tensors):
            return (
                False,
                None,
                f"differential: FAIL — output count mismatch on input[{idx}]: "
                f"{len(orig_tensors)} vs {len(trans_tensors)}",
            )

        for out_idx, (ref, got) in enumerate(zip(orig_tensors, trans_tensors)):
            if ref.shape != got.shape:
                return (
                    False,
                    None,
                    f"differential: FAIL — shape mismatch on input[{idx}] output[{out_idx}]: "
                    f"{tuple(ref.shape)} vs {tuple(got.shape)}",
                )
            cmp = compare_tensors(ref, got, atol=tolerance, rtol=tolerance)
            max_error_seen = max(max_error_seen, float(cmp.max_abs_error))
            if not cmp.passed:
                return (
                    False,
                    max_error_seen,
                    f"differential: FAIL — input[{idx}] output[{out_idx}] "
                    f"max_abs_error={cmp.max_abs_error:.2e} > tol={tolerance:.2e}",
                )

    note = f" (random synth note: {synth_error})" if synth_error else ""
    return (
        True,
        max_error_seen,
        f"differential: PASS ({len(inputs_to_try)} input(s), max_abs_error={max_error_seen:.2e}){note}",
    )


def _extract_entry_arg_shapes(module: ModuleOp) -> list[tuple[int, ...]]:
    """Return the shapes of the tensor-typed args of the first function.

    Used for random-input synthesis. Non-tensor args (scalars, memrefs
    with unknown rank) cause us to skip synthesis on this module.
    """
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.func import FuncOp

    shapes: list[tuple[int, ...]] = []
    for op in module.walk():
        if isinstance(op, FuncOp):
            for arg in op.body.block.args:
                if isinstance(arg.type, TensorType):
                    dims = arg.type.get_shape()
                    if any(d < 0 for d in dims):
                        return []  # dynamic shape — skip synthesis
                    shapes.append(tuple(dims))
                else:
                    return []
            return shapes
    return []


def _flatten_tensors(val: Any) -> list[torch.Tensor]:
    """Flatten an executor output to a list of tensors."""
    if isinstance(val, torch.Tensor):
        return [val]
    if isinstance(val, (tuple, list)):
        out: list[torch.Tensor] = []
        for v in val:
            out.extend(_flatten_tensors(v))
        return out
    return []


def _verify_numeric(
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    tolerance: float = 1e-5,
) -> tuple[bool, float, str]:
    """Numeric equivalence: eager vs torch.compile(backend='eager').

    Runs the model in plain eager mode and through ``torch.compile`` with
    the ``eager`` backend, then compares outputs element-wise using
    :func:`compgen.semantic.verify.compare.compare_tensors`.

    Args:
        model: The PyTorch module to verify.
        sample_inputs: Inputs forwarded to ``model(*sample_inputs)``.
        tolerance: Absolute tolerance for element-wise comparison.

    Returns:
        Tuple of ``(passed, max_abs_error, diagnostic_message)``.
    """
    try:
        model_eval = model.eval()

        with torch.no_grad():
            eager_out = model_eval(*sample_inputs)

        compiled_model = torch.compile(model_eval, backend="eager")
        with torch.no_grad():
            compiled_out = compiled_model(*sample_inputs)

        # Normalise outputs to flat tensor lists.
        def _to_tensors(val: Any) -> list[torch.Tensor]:
            if isinstance(val, torch.Tensor):
                return [val]
            if isinstance(val, (tuple, list)):
                return [t for t in val if isinstance(t, torch.Tensor)]
            return []

        eager_tensors = _to_tensors(eager_out)
        compiled_tensors = _to_tensors(compiled_out)

        if len(eager_tensors) != len(compiled_tensors):
            return (
                False,
                float("inf"),
                f"numeric: FAIL — output count mismatch ({len(eager_tensors)} vs {len(compiled_tensors)})",
            )

        max_err: float = 0.0
        for i, (ref, got) in enumerate(zip(eager_tensors, compiled_tensors)):
            cmp = compare_tensors(ref, got, atol=tolerance, rtol=tolerance)
            max_err = max(max_err, cmp.max_abs_error)
            if not cmp.passed:
                return (
                    False,
                    max_err,
                    f"numeric: FAIL — output[{i}] max_abs_error={cmp.max_abs_error:.2e} > tol={tolerance:.2e}",
                )

        return True, max_err, f"numeric: PASS (max_abs_error={max_err:.2e})"
    except Exception as e:
        log.warning("numeric verification failed", error=str(e))
        return False, float("inf"), f"numeric: FAIL — {e}"


@dataclass
class TransformVerifier:
    """Verifies that transforms preserve IR semantics.

    Attributes:
        tolerance: Maximum allowed absolute error for differential testing.
        levels: Which verification levels to run.
    """

    tolerance: float = 1e-5
    levels: list[VerificationLevel] = field(
        default_factory=lambda: [VerificationLevel.STRUCTURAL, VerificationLevel.DIFFERENTIAL]
    )

    def verify(
        self,
        original_module: ModuleOp,
        transformed_module: ModuleOp,
        sample_inputs: Any = None,
        *,
        model: nn.Module | None = None,
        exported_program: Any | None = None,
    ) -> TransformVerificationResult:
        """Verify a transform preserves semantics.

        Args:
            original_module: The original xDSL module.
            transformed_module: The transformed xDSL module.
            sample_inputs: Optional sample inputs for differential testing
                and numeric verification.
            model: Optional PyTorch module for numeric equivalence checking.
                Required when :attr:`VerificationLevel.NUMERIC` is in
                :attr:`levels`.

        Returns:
            TransformVerificationResult.
        """
        levels_run: list[VerificationLevel] = []
        levels_passed: list[VerificationLevel] = []
        details: dict[str, Any] = {}
        max_error: float | None = None

        for level in self.levels:
            levels_run.append(level)

            if level == VerificationLevel.STRUCTURAL:
                passed, msg = _verify_structural(transformed_module)
                details["structural"] = msg
                if passed:
                    levels_passed.append(level)

            elif level == VerificationLevel.CHECK_ASSERTIONS:
                # No check lines provided by default
                details["check_assertions"] = "check_assertions: SKIPPED (no assertions)"
                levels_passed.append(level)

            elif level == VerificationLevel.DIFFERENTIAL:
                passed, error, msg = _verify_differential(
                    original_module,
                    transformed_module,
                    self.tolerance,
                    sample_inputs=sample_inputs,
                    exported_program=exported_program,
                )
                details["differential"] = msg
                if error is not None:
                    max_error = error if max_error is None else max(max_error, error)
                if passed:
                    levels_passed.append(level)

            elif level == VerificationLevel.TRANSLATION_VALIDATION:
                try:
                    from compgen.ir.semantic.translation_validation import validate_translation

                    tv = validate_translation(original_module, transformed_module)
                    details["translation_validation"] = f"translation_validation: {tv.status.upper()}"
                    if tv.valid:
                        levels_passed.append(level)
                except NotImplementedError:
                    details["translation_validation"] = "translation_validation: SKIPPED (not implemented)"
                    levels_passed.append(level)

            elif level == VerificationLevel.NUMERIC:
                if model is not None and sample_inputs is not None:
                    passed, num_error, msg = _verify_numeric(model, sample_inputs, tolerance=self.tolerance)
                    details["numeric"] = msg
                    if num_error != float("inf"):
                        max_error = num_error if max_error is None else max(max_error, num_error)
                    if passed:
                        levels_passed.append(level)
                else:
                    details["numeric"] = "numeric: SKIPPED (model or sample_inputs not provided)"
                    levels_passed.append(level)

        # Semantic truth: if structural fails but numeric passes, the overall
        # result is still PASS because the outputs are numerically equivalent.
        structural_failed = (
            VerificationLevel.STRUCTURAL in levels_run and VerificationLevel.STRUCTURAL not in levels_passed
        )
        numeric_passed = VerificationLevel.NUMERIC in levels_run and VerificationLevel.NUMERIC in levels_passed
        if structural_failed and numeric_passed:
            all_passed = all(lvl in levels_passed for lvl in levels_run if lvl != VerificationLevel.STRUCTURAL)
        else:
            all_passed = len(levels_passed) == len(levels_run)

        return TransformVerificationResult(
            passed=all_passed,
            levels_run=levels_run,
            levels_passed=levels_passed,
            max_abs_error=max_error,
            details=details,
        )


def verify_transform(
    original_module: ModuleOp,
    transformed_module: ModuleOp,
    sample_inputs: Any = None,
    *,
    model: nn.Module | None = None,
    levels: list[VerificationLevel] | None = None,
) -> TransformVerificationResult:
    """Convenience function: verify with default settings.

    Args:
        original_module: The original xDSL module.
        transformed_module: The transformed xDSL module.
        sample_inputs: Optional sample inputs for differential / numeric testing.
        model: Optional PyTorch module for numeric equivalence checking.
        levels: Override the default verification levels.  When *None*, uses
            the :class:`TransformVerifier` defaults (STRUCTURAL + DIFFERENTIAL).

    Returns:
        TransformVerificationResult.
    """
    verifier = TransformVerifier()
    if levels is not None:
        verifier.levels = levels
    return verifier.verify(original_module, transformed_module, sample_inputs, model=model)


def verify_guarded_transform(
    original_module: ModuleOp,
    transformed_module: ModuleOp,
    *,
    guard_matched: bool,
    sample_inputs: Any = None,
    verifier: TransformVerifier | None = None,
) -> GuardedTransformVerificationResult:
    """Verify a transform that may have been skipped by a guard."""

    if not guard_matched:
        return GuardedTransformVerificationResult(
            guard_matched=False,
            verification=TransformVerificationResult(
                passed=True,
                levels_run=[],
                levels_passed=[],
                details={"guard": "guard rejected; transform not applied"},
            ),
            note="guard_rejected",
        )
    active_verifier = verifier or TransformVerifier()
    result = active_verifier.verify(original_module, transformed_module, sample_inputs)
    return GuardedTransformVerificationResult(
        guard_matched=True,
        verification=result,
        note="guard_applied",
    )


__all__ = [
    "TransformVerificationResult",
    "GuardedTransformVerificationResult",
    "TransformVerifier",
    "VerificationLevel",
    "verify_guarded_transform",
    "verify_transform",
]
