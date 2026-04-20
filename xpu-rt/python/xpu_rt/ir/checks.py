"""FileCheck-style IR structural assertions.

Implements a lightweight ``// CHECK``-style verification system for MLIR/xDSL
IR text. Used for:

1. **IR snapshot checks** -- verify that a transform produced expected ops.
2. **Target-policy checks** -- verify tiling, memory-space promotion, etc.
3. **Regression checks** -- lock promoted recipe IR against regressions.

This is verification ladder level 1 (structural). It does NOT check semantics.

Invariants:
    - CHECK lines must be unambiguous (unique matches in the IR text).
    - CHECK-NOT lines must not match anywhere in the IR.
    - CHECK-LABEL scopes subsequent checks to a labeled region.
    - All check failures produce actionable diagnostics.

TODO: Implement check_ir() with CHECK, CHECK-NOT, CHECK-LABEL, CHECK-SAME.
TODO: Add CHECK-COUNT for counting op occurrences.
TODO: Support regex patterns in check lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CheckKind(Enum):
    """Kind of check assertion."""

    CHECK = "CHECK"
    CHECK_NOT = "CHECK-NOT"
    CHECK_LABEL = "CHECK-LABEL"
    CHECK_SAME = "CHECK-SAME"
    CHECK_COUNT = "CHECK-COUNT"


@dataclass(frozen=True)
class CheckLine:
    """A single check assertion.

    Attributes:
        kind: The type of check.
        pattern: The pattern to match.
        count: Expected count (for CHECK-COUNT).
    """

    kind: CheckKind
    pattern: str
    count: int | None = None


@dataclass(frozen=True)
class CheckFailure:
    """A single check failure.

    Attributes:
        check: The check that failed.
        line_number: Line number in the IR where the failure was detected (or -1).
        message: Human-readable failure description.
    """

    check: CheckLine
    line_number: int
    message: str


@dataclass(frozen=True)
class CheckResult:
    """Result of running IR checks.

    Attributes:
        passed: Whether all checks passed.
        failures: List of check failures.
        checks_run: Total number of checks run.
    """

    passed: bool
    failures: list[CheckFailure] = field(default_factory=list)
    checks_run: int = 0


class IRChecker:
    """FileCheck-style IR assertion checker."""

    def run(self, ir_text: str, checks: list[CheckLine]) -> CheckResult:
        """Run check assertions against IR text.

        Iterates checks in order. CHECK and CHECK-LABEL search forward from
        the current position. CHECK-NOT fails if the pattern is found anywhere
        between the current position and the next CHECK/CHECK-LABEL. CHECK-LABEL
        resets the search position to the label line. CHECK-SAME requires the
        pattern on the same line as the last match.
        """
        lines = ir_text.splitlines()
        failures: list[CheckFailure] = []
        pos = 0  # current line index in the IR text
        last_match_line = -1

        for check in checks:
            if check.kind == CheckKind.CHECK:
                found = False
                for i in range(pos, len(lines)):
                    if check.pattern in lines[i]:
                        pos = i + 1
                        last_match_line = i
                        found = True
                        break
                if not found:
                    failures.append(
                        CheckFailure(
                            check=check,
                            line_number=pos,
                            message=f"CHECK pattern '{check.pattern}' not found after line {pos}",
                        )
                    )

            elif check.kind == CheckKind.CHECK_NOT:
                for i in range(pos, len(lines)):
                    if check.pattern in lines[i]:
                        failures.append(
                            CheckFailure(
                                check=check,
                                line_number=i + 1,
                                message=f"CHECK-NOT pattern '{check.pattern}' found at line {i + 1}",
                            )
                        )
                        break

            elif check.kind == CheckKind.CHECK_LABEL:
                found = False
                for i in range(len(lines)):
                    if check.pattern in lines[i]:
                        pos = i + 1
                        last_match_line = i
                        found = True
                        break
                if not found:
                    failures.append(
                        CheckFailure(
                            check=check,
                            line_number=-1,
                            message=f"CHECK-LABEL pattern '{check.pattern}' not found anywhere",
                        )
                    )

            elif check.kind == CheckKind.CHECK_SAME:
                if last_match_line >= 0 and check.pattern in lines[last_match_line]:
                    pass  # OK, same line
                else:
                    failures.append(
                        CheckFailure(
                            check=check,
                            line_number=last_match_line,
                            message=f"CHECK-SAME pattern '{check.pattern}' not on same line as last match",
                        )
                    )

            elif check.kind == CheckKind.CHECK_COUNT:
                count = sum(1 for line in lines if check.pattern in line)
                expected = check.count or 0
                if count != expected:
                    failures.append(
                        CheckFailure(
                            check=check,
                            line_number=-1,
                            message=f"CHECK-COUNT expected {expected} occurrences of '{check.pattern}', found {count}",
                        )
                    )

        return CheckResult(
            passed=len(failures) == 0,
            failures=failures,
            checks_run=len(checks),
        )


def _parse_check_line(line: str) -> CheckLine | None:
    """Parse a single check string into a CheckLine."""
    line = line.strip()
    # Strip leading comment markers
    for prefix in ("//", "#", "--"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()

    for kind in CheckKind:
        tag = kind.value + ":"
        if line.startswith(tag):
            pattern = line[len(tag) :].strip()
            if kind == CheckKind.CHECK_COUNT:
                # Format: CHECK-COUNT:N: pattern
                parts = pattern.split(":", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    return CheckLine(kind=kind, pattern=parts[1].strip(), count=int(parts[0].strip()))
            return CheckLine(kind=kind, pattern=pattern)
    return None


def check_ir(ir_text: str, check_lines: list[str]) -> CheckResult:
    """Convenience function: parse check strings and run.

    Args:
        ir_text: The IR text to check.
        check_lines: List of check strings (e.g., "// CHECK: linalg.matmul").

    Returns:
        CheckResult.
    """
    checks: list[CheckLine] = []
    for raw in check_lines:
        parsed = _parse_check_line(raw)
        if parsed is not None:
            checks.append(parsed)
    return IRChecker().run(ir_text, checks)


__all__ = ["CheckFailure", "CheckKind", "CheckLine", "CheckResult", "IRChecker", "check_ir"]
