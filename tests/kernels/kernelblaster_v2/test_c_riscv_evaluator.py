"""Tests for the Spike+gemmini-backed :class:`CRiscvEvaluator`.

The full integration test (compile + spike run) is gated on the chipyard
conda toolchain being on disk; CI without it falls through to the
toolchain-missing path and tests that branch deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
    CRiscvEvaluator,
    DEFAULT_CONDA_ROOT,
    ENV_CC,
    ENV_CONDA_ROOT,
    ENV_PK,
    ENV_SPIKE,
    _check_gemmini_extension,
    _check_toolchain,
    _generate_harness,
    _parse_output,
    _resolve_matmul_shape,
    _ToolchainMissing,
)
from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse
from xpu_rt.kernels.provider import KernelContract


def _matmul_contract(M: int = 16, K: int = 64, N: int = 64) -> KernelContract:
    return KernelContract(
        region_id="test_gemm",
        op_family="matmul",
        input_shapes=((M, K), (K, N)),
        output_shapes=((M, N),),
        dtypes=("i8", "i8", "i32"),
        layout="row_major",
        target_name="gemmini_mx",
        objective="latency",
    )


def _toolchain_present() -> bool:
    try:
        _check_toolchain()
    except _ToolchainMissing:
        return False
    return True


SKIP_TOOLCHAIN = pytest.mark.skipif(
    not _toolchain_present(),
    reason="RISC-V conda toolchain not available on this host",
)


# ---------------------------------------------------------------------------
# Pure-function units (no toolchain needed)
# ---------------------------------------------------------------------------


def test_parse_output_matmul_pass() -> None:
    stdout = "M=16 K=64 N=64 ops=131072\nmismatches=0/1024\ncycles=12345\n"
    mismatches, total, cycles = _parse_output(stdout)
    assert (mismatches, total, cycles) == (0, 1024, 12345)


def test_parse_output_matmul_fail() -> None:
    stdout = (
        "M=16 K=64 N=64 ops=131072\n"
        "mismatches=42/1024\n"
        "first_diff_at=3 ref=-1 got=0\n"
        "cycles=99\n"
    )
    mismatches, total, cycles = _parse_output(stdout)
    assert (mismatches, total, cycles) == (42, 1024, 99)


def test_parse_output_no_mismatch_line() -> None:
    """Spike crashes / output garbled → return (None, None, …)."""
    stdout = "An illegal instruction was executed!\n"
    mismatches, total, cycles = _parse_output(stdout)
    assert mismatches is None
    assert total is None
    assert cycles is None


def test_resolve_matmul_shape_happy_path() -> None:
    shape = _resolve_matmul_shape(_matmul_contract(M=8, K=32, N=16))
    assert (shape.M, shape.K, shape.N) == (8, 32, 16)
    assert shape.dtype_in == "i8"


def test_resolve_matmul_shape_mismatched_shapes() -> None:
    bad = KernelContract(
        op_family="matmul",
        input_shapes=((8, 32), (16, 16)),  # K mismatch: A.k=32 vs B.k=16
        output_shapes=((8, 16),),
        dtypes=("i8", "i8", "i32"),
        target_name="gemmini_mx",
    )
    with pytest.raises(NotImplementedError, match="shape inconsistency"):
        _resolve_matmul_shape(bad)


def test_resolve_matmul_rejects_non_i8() -> None:
    bad = KernelContract(
        op_family="matmul",
        input_shapes=((8, 32), (32, 16)),
        output_shapes=((8, 16),),
        dtypes=("fp32", "fp32", "fp32"),
        target_name="gemmini_mx",
    )
    with pytest.raises(NotImplementedError, match="expects an i8 input dtype"):
        _resolve_matmul_shape(bad)


def test_generate_harness_matmul_substitutes_shape() -> None:
    body = _generate_harness(_matmul_contract(M=32, K=128, N=64))
    assert "#define M 32" in body
    assert "#define K 128" in body
    assert "#define N 64" in body
    assert "kernel_under_test" in body
    assert "counter_configure(0, MAIN_LD_ST_EX_CYCLES)" in body


def test_generate_harness_unsupported_op_family() -> None:
    bad = KernelContract(
        op_family="softmax",
        input_shapes=((16, 64),),
        output_shapes=((16, 64),),
        dtypes=("fp16",),
        target_name="gemmini_mx",
    )
    with pytest.raises(NotImplementedError, match="does not yet generate a harness"):
        _generate_harness(bad)


# ---------------------------------------------------------------------------
# Evaluator branches
# ---------------------------------------------------------------------------


def test_evaluator_reports_toolchain_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bogus conda root → clean toolchain_missing path, no crash."""
    monkeypatch.setenv(ENV_CONDA_ROOT, "/nope/not/here")
    monkeypatch.delenv(ENV_CC, raising=False)
    monkeypatch.delenv(ENV_SPIKE, raising=False)
    monkeypatch.delenv(ENV_PK, raising=False)
    evaluator = CRiscvEvaluator(contract=_matmul_contract())
    report = evaluator.evaluate(ProposeResponse(kernel_code="void k(){}", language="c"))
    assert not report.correct
    assert report.score == 0.0
    assert report.metadata.get("reason") == "toolchain_missing"
    assert "/nope/not/here" in report.compile_log


def test_evaluator_target_dispatch_gemmini_uses_extension_flag() -> None:
    """Gemmini-targeted evaluator must emit ``--extension=gemmini`` to
    Spike and ``-march=rv64gc`` to gcc — that's the existing path."""
    e = CRiscvEvaluator(contract=_matmul_contract(), target_id="gemmini_mx")
    assert not e._is_saturn()
    assert e._spike_flag() == ("--extension=gemmini",)
    march = e._march_flags()
    assert "-march=rv64gc" in march
    # Gemmini-tests headers must be on the include search path.
    inc_args = e._target_include_args()
    assert any("gemmini-rocc-tests" in a for a in inc_args)


def test_evaluator_target_dispatch_saturn_uses_isa_flag() -> None:
    """Saturn-targeted evaluator must emit ``--isa=rv64gcv_zvl128b_zicntr``
    (RVV 1.0 + 128-bit vectors + zicntr cycle counter for mcycle) and
    matching ``-march=`` flags. No Gemmini headers leak in."""
    e = CRiscvEvaluator(contract=_matmul_contract(), target_id="saturn_opu_v128")
    assert e._is_saturn()
    assert e._spike_flag() == ("--isa=rv64gcv_zvl128b_zicntr",)
    march = e._march_flags()
    assert "-march=rv64gcv_zvl128b_zicntr" in march
    # No Gemmini include args leak through.
    assert e._target_include_args() == ()


def test_evaluator_reports_unsupported_op_family() -> None:
    """Unsupported op_family → clean failure with the right reason."""
    contract = KernelContract(
        op_family="softmax",
        input_shapes=((16, 64),),
        output_shapes=((16, 64),),
        dtypes=("fp16",),
        target_name="gemmini_mx",
    )
    evaluator = CRiscvEvaluator(contract=contract, require_gemmini_extension=False)
    report = evaluator.evaluate(ProposeResponse(kernel_code="// nope", language="c"))
    assert not report.correct
    # The order in which checks run means we get either toolchain_missing
    # or harness_unsupported_op_family depending on host state.
    assert report.metadata.get("reason") in (
        "harness_unsupported_op_family",
        "toolchain_missing",
    )


@SKIP_TOOLCHAIN
def test_evaluator_compile_failure() -> None:
    """A candidate that doesn't define ``kernel_under_test`` → link error."""
    evaluator = CRiscvEvaluator(
        contract=_matmul_contract(),
        require_gemmini_extension=False,  # we won't get past compile anyway
    )
    report = evaluator.evaluate(
        ProposeResponse(kernel_code="// no kernel_under_test defined\n", language="c"),
    )
    assert not report.correct
    assert report.score == 0.0
    assert report.metadata.get("reason") == "compile_failed"
    assert "kernel_under_test" in report.compile_log


@SKIP_TOOLCHAIN
def test_evaluator_zero_kernel_is_incorrect_but_runs() -> None:
    """Always-zero kernel → mismatches > 0, cycles is set, report.correct=False.

    Demonstrates the evaluator distinguishes "compile-fail" (no kernel) from
    "runs but wrong" (kernel runs, output diffs). Skipped when the gemmini
    extension is absent — the always-zero kernel doesn't issue any RoCC
    instructions, so it would still compile + run on plain spike, but we
    leave the parity gate on so the test exercises the full flow.
    """
    if not _check_gemmini_extension():
        pytest.skip("spike --extension=gemmini not available")
    kernel = """
    #include <stdint.h>
    #include <string.h>
    #define M 16
    #define K 64
    #define N 64
    void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {
        (void)A; (void)B;
        memset(C, 0, M * N * sizeof(int32_t));
    }
    """
    evaluator = CRiscvEvaluator(contract=_matmul_contract(), timeout_s=60)
    report = evaluator.evaluate(ProposeResponse(kernel_code=kernel, language="c"))
    assert report.metadata.get("reason") not in ("compile_failed", "toolchain_missing")
    assert not report.correct
    # mismatches should be > 0 and cycles should be a sensible integer.
    assert report.metadata["mismatches"] > 0
    assert report.cycles is not None and report.cycles >= 0
