"""Tests for FileCheck-style IR assertions."""

from __future__ import annotations

from compgen.ir.checks import CheckKind, CheckLine, IRChecker, check_ir

SAMPLE_IR = """\
builtin.module {
  func.func @forward(%arg0: tensor<8x768xf32>) -> tensor<8x768xf32> {
    %0 = linalg.matmul ins(%arg0, %w1) outs(%out1)
    %1 = arith.addf %0, %bias1
    %2 = linalg.generic {gelu} ins(%1)
    %3 = linalg.matmul ins(%2, %w2) outs(%out2)
    %4 = arith.addf %3, %bias2
    func.return %4
  }
}
"""


def test_check_line_construction() -> None:
    check = CheckLine(kind=CheckKind.CHECK, pattern="linalg.matmul")
    assert check.kind == CheckKind.CHECK
    assert check.pattern == "linalg.matmul"


def test_check_not_construction() -> None:
    check = CheckLine(kind=CheckKind.CHECK_NOT, pattern="tensor.empty")
    assert check.kind == CheckKind.CHECK_NOT


def test_check_ir_basic_pass() -> None:
    """CHECK should pass when pattern is found."""
    result = check_ir(SAMPLE_IR, ["// CHECK: linalg.matmul"])
    assert result.passed
    assert result.checks_run == 1
    assert len(result.failures) == 0


def test_check_ir_basic_fail() -> None:
    """CHECK should fail when pattern is NOT found."""
    result = check_ir(SAMPLE_IR, ["// CHECK: linalg.conv2d"])
    assert not result.passed
    assert len(result.failures) == 1
    assert "linalg.conv2d" in result.failures[0].message


def test_check_not_passes() -> None:
    """CHECK-NOT should pass when pattern is NOT found."""
    result = check_ir(SAMPLE_IR, ["// CHECK-NOT: tensor.empty"])
    assert result.passed


def test_check_not_fails_on_match() -> None:
    """CHECK-NOT should fail when pattern is found."""
    result = check_ir(SAMPLE_IR, ["// CHECK-NOT: linalg.matmul"])
    assert not result.passed
    assert len(result.failures) == 1


def test_check_label_scoping() -> None:
    """CHECK-LABEL should reset search position."""
    result = check_ir(SAMPLE_IR, [
        "// CHECK-LABEL: func.func @forward",
        "// CHECK: linalg.matmul",
        "// CHECK: arith.addf",
    ])
    assert result.passed
    assert result.checks_run == 3


def test_check_ordering() -> None:
    """CHECKs must match in order."""
    result = check_ir(SAMPLE_IR, [
        "// CHECK: arith.addf",
        "// CHECK: linalg.matmul",  # second matmul should match
    ])
    assert result.passed


def test_check_count() -> None:
    """CHECK-COUNT should count occurrences."""
    result = check_ir(SAMPLE_IR, ["// CHECK-COUNT:2: linalg.matmul"])
    assert result.passed

    result2 = check_ir(SAMPLE_IR, ["// CHECK-COUNT:3: linalg.matmul"])
    assert not result2.passed


def test_multiple_checks_mixed() -> None:
    """Multiple check types together."""
    result = check_ir(SAMPLE_IR, [
        "// CHECK-LABEL: func.func @forward",
        "// CHECK: linalg.matmul",
        "// CHECK: linalg.generic",
        "// CHECK-NOT: tensor.empty",
        "// CHECK-COUNT:2: arith.addf",
    ])
    assert result.passed
    assert result.checks_run == 5


def test_checker_class_directly() -> None:
    """IRChecker can be used directly with CheckLine objects."""
    checker = IRChecker()
    checks = [
        CheckLine(kind=CheckKind.CHECK, pattern="linalg.matmul"),
        CheckLine(kind=CheckKind.CHECK_NOT, pattern="tensor.empty"),
    ]
    result = checker.run(SAMPLE_IR, checks)
    assert result.passed
