"""Transfer function verification backend using Z3.

Verifies that dataflow transfer functions are sound over-approximations
of the concrete semantics. A transfer function is sound if for every
concrete input that satisfies the abstract input constraint, the concrete
output satisfies the abstract output constraint.

Example: a known-bits transfer function for ``arith.ori`` says that
``result_known_ones = lhs_known_ones | rhs_known_ones``. This is sound if
for every concrete (lhs, rhs) pair consistent with the known bits, the
concrete OR result is consistent with the predicted known bits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.semantic.backends.xdsl_smt.results import (
    StructuredCounterexample,
    TransferResult,
)
from compgen.semantic.backends.xdsl_smt.tv_backend import _check_z3_available

log = structlog.get_logger()


@dataclass
class TransferVerificationBackend:
    """Verify transfer function soundness using Z3.

    Attributes:
        timeout_ms: Z3 solver timeout.
    """

    timeout_ms: int = 30_000

    def verify_transfer(
        self,
        transfer_module: ModuleOp,
        max_bitwidth: int = 32,
    ) -> TransferResult:
        """Verify transfer functions in a module.

        Args:
            transfer_module: Module containing transfer function definitions.
            max_bitwidth: Maximum bitwidth to verify.

        Returns:
            TransferResult with soundness outcome.
        """
        if not _check_z3_available():
            return TransferResult(sound=False, status="unknown")

        # The full transfer verification from the artifact uses a
        # specialized transfer dialect. For the initial integration,
        # we support the callable-based verification path.
        return TransferResult(sound=False, status="unknown")

    def verify_forward_transfer(
        self,
        concrete_fn: Callable[..., Any],
        transfer_fn: Callable[..., Any],
        abstract_constraint: Callable[..., Any],
        instance_constraint: Callable[..., Any],
        num_operands: int,
        max_bitwidth: int = 32,
    ) -> TransferResult:
        """Verify a forward transfer function via callables.

        A forward transfer function maps abstract inputs to abstract
        outputs. It is sound if: for every concrete input consistent
        with the abstract input, the concrete output is consistent
        with the abstract output predicted by the transfer function.

        Args:
            concrete_fn: (operands: list[BitVec]) -> BitVec
            transfer_fn: (abstract_inputs: list[tuple]) -> tuple
                          where tuple = (zeros, ones) for known-bits.
            abstract_constraint: (concrete, abstract) -> BoolRef
                                  True if concrete is consistent with abstract.
            instance_constraint: (abstract) -> BoolRef
                                  True if abstract is a valid abstract value.
            num_operands: Number of operands.
            max_bitwidth: Maximum bitwidth to verify.

        Returns:
            TransferResult.
        """
        if not _check_z3_available():
            return TransferResult(sound=False, status="unknown")

        import z3

        start = time.monotonic()

        for width in range(1, max_bitwidth + 1):
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)

            # Create concrete operands
            concretes = [z3.BitVec(f"c{i}", width) for i in range(num_operands)]

            # Create abstract operands (e.g., known-bits: zeros, ones)
            abstracts = [(z3.BitVec(f"a{i}_zeros", width), z3.BitVec(f"a{i}_ones", width)) for i in range(num_operands)]

            # Assert abstract values are valid
            for ab in abstracts:
                solver.add(instance_constraint(ab))

            # Assert concrete values are consistent with abstract
            for conc, ab in zip(concretes, abstracts):
                solver.add(abstract_constraint(conc, ab))

            # Compute concrete result
            concrete_result = concrete_fn(concretes)

            # Compute abstract result via transfer
            abstract_result = transfer_fn(abstracts)

            # Assert concrete result is NOT consistent with abstract result
            # (negation of soundness)
            solver.add(z3.Not(abstract_constraint(concrete_result, abstract_result)))

            result = solver.check()
            if result == z3.sat:
                model = solver.model()
                elapsed = (time.monotonic() - start) * 1000

                input_vals = {f"c{i}": str(model.evaluate(c, model_completion=True)) for i, c in enumerate(concretes)}
                cex = StructuredCounterexample(
                    inputs=input_vals,
                    summary=f"transfer unsound at width={width}",
                )
                return TransferResult(
                    sound=False,
                    status="unsound",
                    solver_time_ms=elapsed,
                    counterexample=cex,
                )
            elif result != z3.unsat:
                elapsed = (time.monotonic() - start) * 1000
                return TransferResult(
                    sound=False,
                    status="unknown",
                    solver_time_ms=elapsed,
                )

        elapsed = (time.monotonic() - start) * 1000
        return TransferResult(
            sound=True,
            status="sound",
            solver_time_ms=elapsed,
        )


__all__ = ["TransferVerificationBackend"]
