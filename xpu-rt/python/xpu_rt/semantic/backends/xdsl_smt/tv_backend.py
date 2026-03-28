"""Translation validation backend using Z3.

Checks that a transformed Payload IR region refines the original by:
1. Lowering both regions to Z3 bitvector expressions.
2. Building a refinement formula: the transformed program must produce
   the same outputs for all inputs where the original has defined behavior.
3. Asserting the negation and checking SAT — unsat means the transform
   is correct.

This uses the Z3 Python API directly (not the xdsl-smt subprocess flow)
for tighter integration with CompGen's agent loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import StringIO
from typing import Any

import structlog
from xdsl.dialects.arith import (
    AddiOp,
    CmpiOp,
    ConstantOp,
    DivSIOp,
    DivUIOp,
    MuliOp,
    RemSIOp,
    RemUIOp,
    SelectOp,
    SubiOp,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, SSAValue

from compgen.semantic.backends.xdsl_smt.results import (
    StructuredCounterexample,
    TVResult,
    interpret_z3_result,
    parse_z3_counterexample,
)

log = structlog.get_logger()


def _check_z3_available() -> bool:
    """Check if z3 Python bindings are available."""
    try:
        import z3  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class TranslationValidationBackend:
    """SMT-backed translation validation using Z3 Python API.

    Validates that a target module is a correct refinement of a source
    module by building bitvector expressions and checking with Z3.

    Attributes:
        timeout_ms: Z3 solver timeout in milliseconds.
    """

    timeout_ms: int = 30_000

    def check_refinement(
        self,
        before: ModuleOp,
        after: ModuleOp,
        optimize: bool = True,
        timeout_ms: int | None = None,
    ) -> TVResult:
        """Check that ``after`` refines ``before``.

        Both modules must contain a single ``func.func`` with the same
        signature. The refinement check verifies that for all inputs,
        if the source has defined behavior, the target produces the same
        outputs.

        Args:
            before: Source (original) module.
            after: Target (transformed) module.
            optimize: Whether to apply Z3 simplification.
            timeout_ms: Override solver timeout.

        Returns:
            TVResult with the verification outcome.
        """
        if not _check_z3_available():
            return TVResult(
                ok=False,
                status="unknown",
                solver_stdout="",
                solver_stderr="z3-solver not installed",
                solver_time_ms=0.0,
            )

        import z3

        effective_timeout = timeout_ms or self.timeout_ms

        # Extract func.func from both modules
        before_func = _extract_func(before)
        after_func = _extract_func(after)

        if before_func is None or after_func is None:
            return TVResult(
                ok=False,
                status="unknown",
                solver_stderr="could not extract func.func from both modules",
            )

        try:
            start = time.monotonic()
            result = self._build_and_check(
                before_func, after_func, effective_timeout, optimize
            )
            elapsed = (time.monotonic() - start) * 1000
            return TVResult(
                ok=result["ok"],
                status=result["status"],
                smtlib=result.get("smtlib", ""),
                solver_stdout=result.get("stdout", ""),
                solver_stderr=result.get("stderr", ""),
                solver_time_ms=elapsed,
                counterexample=result.get("counterexample"),
            )
        except Exception as e:
            log.warning("tv.check_refinement.error", error=str(e))
            return TVResult(
                ok=False,
                status="unknown",
                solver_stderr=str(e),
            )

    def _build_and_check(
        self,
        before_func: FuncOp,
        after_func: FuncOp,
        timeout_ms: int,
        optimize: bool,
    ) -> dict[str, Any]:
        """Build the refinement formula and check with Z3."""
        import z3

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        # Build Z3 expressions for both functions
        lowerer = ArithZ3Lowerer()

        before_inputs, before_outputs = lowerer.lower_func(before_func, prefix="src_")
        after_inputs, after_outputs = lowerer.lower_func(after_func, prefix="tgt_")

        # Both functions must use the same input variables
        # Equate corresponding inputs
        for (src_name, src_var), (_, tgt_var) in zip(
            before_inputs.items(), after_inputs.items()
        ):
            solver.add(src_var == tgt_var)

        # Build refinement: for each output, target must equal source
        # Refinement: ¬(∧ᵢ (src_outᵢ == tgt_outᵢ))
        # If this is UNSAT, the refinement holds for all inputs
        refinement_parts = []
        for (src_name, src_val), (_, tgt_val) in zip(
            before_outputs.items(), after_outputs.items()
        ):
            refinement_parts.append(src_val != tgt_val)

        if not refinement_parts:
            return {"ok": True, "status": "valid"}

        # Assert that at least one output differs (negation of refinement)
        solver.add(z3.Or(*refinement_parts))

        # Check
        result = solver.check()

        smtlib = solver.sexpr()

        if result == z3.unsat:
            return {
                "ok": True,
                "status": "valid",
                "smtlib": smtlib,
                "stdout": "unsat",
            }
        elif result == z3.sat:
            model = solver.model()
            cex = _extract_counterexample(model, before_inputs, before_outputs, after_outputs)
            return {
                "ok": False,
                "status": "invalid",
                "smtlib": smtlib,
                "stdout": f"sat\n{model}",
                "counterexample": cex,
            }
        else:
            return {
                "ok": False,
                "status": "unknown",
                "smtlib": smtlib,
                "stdout": "unknown",
            }


class ArithZ3Lowerer:
    """Lowers arith dialect operations to Z3 bitvector expressions.

    Walks a ``func.func`` body and builds Z3 expressions for each SSA
    value. Supports the arith integer operations that have defined
    semantics for translation validation.
    """

    def __init__(self) -> None:
        self._values: dict[SSAValue, Any] = {}

    def lower_func(
        self, func: FuncOp, prefix: str = ""
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Lower a func.func to Z3 expressions.

        Args:
            func: The function to lower.
            prefix: Prefix for variable names (to distinguish source/target).

        Returns:
            (inputs, outputs) where each is a dict of name -> Z3 expression.
        """
        import z3

        self._values = {}
        inputs: dict[str, Any] = {}
        outputs: dict[str, Any] = {}

        # Create Z3 variables for block arguments (function inputs)
        block = func.body.blocks[0]
        for i, arg in enumerate(block.args):
            width = _get_integer_width(arg.type)
            if width is None:
                width = 32  # default for untyped
            name = f"{prefix}arg{i}"
            z3_var = z3.BitVec(name, width)
            self._values[arg] = z3_var
            inputs[name] = z3_var

        # Lower each operation
        for op in block.ops:
            self._lower_op(op, prefix)

        # Collect return values
        for op in block.ops:
            if isinstance(op, ReturnOp):
                for i, operand in enumerate(op.operands):
                    name = f"{prefix}ret{i}"
                    outputs[name] = self._get_value(operand)

        return inputs, outputs

    def _get_value(self, val: SSAValue) -> Any:
        """Get the Z3 expression for an SSA value."""
        import z3

        if val in self._values:
            return self._values[val]
        # Fallback: uninterpreted constant
        width = _get_integer_width(val.type)
        if width is None:
            width = 32
        name = f"unknown_{id(val)}"
        result = z3.BitVec(name, width)
        self._values[val] = result
        return result

    def _lower_op(self, op: Operation, prefix: str) -> None:
        """Lower a single operation to Z3."""
        import z3

        if isinstance(op, ConstantOp):
            if isinstance(op.value, IntegerAttr):
                width = op.value.type.width.data if isinstance(op.value.type, IntegerType) else 32
                val = op.value.value.data
                self._values[op.result] = z3.BitVecVal(val, width)

        elif isinstance(op, AddiOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = lhs + rhs

        elif isinstance(op, SubiOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = lhs - rhs

        elif isinstance(op, MuliOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = lhs * rhs

        elif isinstance(op, DivUIOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = z3.UDiv(lhs, rhs)

        elif isinstance(op, DivSIOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = lhs / rhs  # Z3 signed div

        elif isinstance(op, RemUIOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = z3.URem(lhs, rhs)

        elif isinstance(op, RemSIOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            self._values[op.result] = z3.SRem(lhs, rhs)

        elif isinstance(op, CmpiOp):
            lhs = self._get_value(op.lhs)
            rhs = self._get_value(op.rhs)
            pred = op.predicate.value.data
            cmp = _arith_cmpi_to_z3(pred, lhs, rhs)
            if cmp is not None:
                # CmpI returns i1, encode as 1-bit bitvector
                self._values[op.result] = z3.If(cmp, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))

        elif isinstance(op, SelectOp):
            cond = self._get_value(op.cond)
            true_val = self._get_value(op.lhs)
            false_val = self._get_value(op.rhs)
            # cond is i1 bitvector, compare with 1
            self._values[op.result] = z3.If(cond == z3.BitVecVal(1, 1), true_val, false_val)

        elif isinstance(op, ReturnOp):
            pass  # handled in lower_func

        # Ops without known semantics: leave as uninterpreted


def _extract_func(module: ModuleOp) -> FuncOp | None:
    """Extract the first func.func from a module."""
    for op in module.body.block.ops:
        if isinstance(op, FuncOp):
            return op
    return None


def _get_integer_width(attr: Any) -> int | None:
    """Get the bitwidth of an integer type."""
    if isinstance(attr, IntegerType):
        return attr.width.data
    return None


def _arith_cmpi_to_z3(predicate: int, lhs: Any, rhs: Any) -> Any:
    """Convert arith.cmpi predicate to Z3 comparison."""
    import z3

    # MLIR arith.cmpi predicates:
    # 0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
    mapping = {
        0: lambda a, b: a == b,
        1: lambda a, b: a != b,
        2: lambda a, b: a < b,      # Z3 signed by default for BitVec
        3: lambda a, b: a <= b,
        4: lambda a, b: a > b,
        5: lambda a, b: a >= b,
        6: lambda a, b: z3.ULT(a, b),
        7: lambda a, b: z3.ULE(a, b),
        8: lambda a, b: z3.UGT(a, b),
        9: lambda a, b: z3.UGE(a, b),
    }
    fn = mapping.get(predicate)
    if fn is None:
        return None
    return fn(lhs, rhs)


def _extract_counterexample(
    model: Any,
    inputs: dict[str, Any],
    src_outputs: dict[str, Any],
    tgt_outputs: dict[str, Any],
) -> StructuredCounterexample:
    """Extract a structured counterexample from a Z3 model."""
    input_vals: dict[str, str] = {}
    expected_vals: dict[str, str] = {}
    actual_vals: dict[str, str] = {}

    for name, var in inputs.items():
        val = model.evaluate(var, model_completion=True)
        input_vals[name] = str(val)

    for name, expr in src_outputs.items():
        val = model.evaluate(expr, model_completion=True)
        expected_vals[name] = str(val)

    for name, expr in tgt_outputs.items():
        val = model.evaluate(expr, model_completion=True)
        actual_vals[name] = str(val)

    parts = [f"{k}={v}" for k, v in sorted(input_vals.items())[:3]]
    diffs = [
        f"{k}: expected {expected_vals.get(k, '?')} got {actual_vals.get(k, '?')}"
        for k in expected_vals
        if expected_vals.get(k) != actual_vals.get(k)
    ]
    summary = f"inputs({', '.join(parts)})"
    if diffs:
        summary += f" — {diffs[0]}"

    return StructuredCounterexample(
        inputs=input_vals,
        expected=expected_vals,
        actual=actual_vals,
        summary=summary,
    )


__all__ = ["ArithZ3Lowerer", "TranslationValidationBackend"]
