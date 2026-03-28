"""PDL rewrite verification backend using Z3.

Verifies that DAG-to-DAG peephole rewrites (expressed as pattern →
replacement pairs) are semantics-preserving across all bitwidths.

The approach: for each concrete bitwidth w in [1..max_bitwidth], build a
Z3 formula asserting that there exist inputs where the pattern and
replacement produce different results. If the formula is UNSAT for all
bitwidths, the rewrite is sound.

Limitations (from the paper):
    - DAG-to-DAG rewrites only — no side effects.
    - Pure arith ops only (no memory, control flow).
    - Does not verify type/attribute constraints beyond bitwidths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.semantic.backends.xdsl_smt.results import PDLResult
from compgen.semantic.backends.xdsl_smt.tv_backend import _check_z3_available

log = structlog.get_logger()


@dataclass
class PDLVerificationBackend:
    """Verify PDL rewrite patterns for soundness across bitwidths.

    Attributes:
        timeout_per_bitwidth_ms: Z3 timeout per bitwidth check.
    """

    timeout_per_bitwidth_ms: int = 10_000

    def verify_pattern(
        self,
        pattern_module: ModuleOp,
        max_bitwidth: int = 32,
        optimize: bool = True,
    ) -> PDLResult:
        """Verify a PDL rewrite pattern across all bitwidths [1, max_bitwidth].

        Args:
            pattern_module: Module containing the PDL pattern.
            max_bitwidth: Maximum bitwidth to check.
            optimize: Whether to simplify the Z3 query.

        Returns:
            PDLResult with soundness outcome.
        """
        if not _check_z3_available():
            return PDLResult(
                sound=False,
                status="unknown",
            )

        start = time.monotonic()
        checked: list[int] = []
        unsound: list[int] = []

        # Extract pattern and replacement from PDL module
        pattern_ops, replacement_ops = _extract_pdl_pattern_replacement(pattern_module)
        if pattern_ops is None or replacement_ops is None:
            log.warning("pdl.verify.no_pattern", msg="could not extract pattern/replacement from module")
            return PDLResult(
                sound=False,
                status="unknown",
            )

        for width in range(1, max_bitwidth + 1):
            is_sound = self._check_bitwidth(
                pattern_ops, replacement_ops, width
            )
            checked.append(width)
            if not is_sound:
                unsound.append(width)

        elapsed = (time.monotonic() - start) * 1000

        all_sound = len(unsound) == 0
        return PDLResult(
            sound=all_sound,
            status="sound" if all_sound else "unsound",
            bitwidths_checked=checked,
            unsound_bitwidths=unsound,
            solver_time_ms=elapsed,
        )

    def verify_arith_rewrite(
        self,
        build_pattern: Any,
        build_replacement: Any,
        num_operands: int,
        max_bitwidth: int = 32,
    ) -> PDLResult:
        """Verify an arith rewrite given as Python callables.

        This is the direct-verification path used when a rewrite can be
        expressed as two Z3 expression builders.

        Args:
            build_pattern: Callable(operands: list[z3.BitVec]) -> z3.BitVec
            build_replacement: Callable(operands: list[z3.BitVec]) -> z3.BitVec
            num_operands: Number of symbolic operands.
            max_bitwidth: Maximum bitwidth to check.

        Returns:
            PDLResult with soundness outcome.
        """
        if not _check_z3_available():
            return PDLResult(sound=False, status="unknown")

        import z3

        start = time.monotonic()
        checked: list[int] = []
        unsound: list[int] = []

        for width in range(1, max_bitwidth + 1):
            operands = [z3.BitVec(f"x{i}", width) for i in range(num_operands)]
            pattern_result = build_pattern(operands)
            replacement_result = build_replacement(operands)

            solver = z3.Solver()
            solver.set("timeout", self.timeout_per_bitwidth_ms)
            solver.add(pattern_result != replacement_result)

            result = solver.check()
            checked.append(width)
            if result != z3.unsat:
                unsound.append(width)
                log.debug("pdl.verify.unsound", width=width, result=str(result))

        elapsed = (time.monotonic() - start) * 1000
        all_sound = len(unsound) == 0
        return PDLResult(
            sound=all_sound,
            status="sound" if all_sound else "unsound",
            bitwidths_checked=checked,
            unsound_bitwidths=unsound,
            solver_time_ms=elapsed,
        )

    def _check_bitwidth(
        self,
        pattern_ops: list[Any],
        replacement_ops: list[Any],
        width: int,
    ) -> bool:
        """Check a single bitwidth. Returns True if sound."""
        import z3

        # This is a simplified path; real PDL verification would
        # lower the PDL pattern to SMT. For now we use the callable path.
        return True  # Placeholder — wired via verify_arith_rewrite for real checks


def _extract_pdl_pattern_replacement(module: ModuleOp) -> tuple[Any, Any]:
    """Extract pattern and replacement op lists from a PDL module.

    Returns (pattern_ops, replacement_ops) or (None, None) if extraction
    fails.
    """
    # PDL patterns are complex IR structures. For the initial integration,
    # we support the verify_arith_rewrite() callable path and will extend
    # this to real PDL parsing as CompGen's rewrite export matures.
    return None, None


__all__ = ["PDLVerificationBackend"]
