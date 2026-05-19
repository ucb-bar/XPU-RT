"""Structural tests for :mod:`xpu_rt.kb_saturn.templates`.

The corresponding Gemmini tests live at
``xpu-rt/tests/kb_gemmini/test_multiop_harness.py`` etc. — this file
mirrors that pattern for the RVV path: verify the emitted C source
contains the right ABI signature, the right CSR-read incantation for
``mcycle``, and the printf protocol KB's parser expects.
"""

from __future__ import annotations

from pathlib import Path

from xpu_rt.kb_saturn.templates import (
    render_driver_c,
    render_init_c,
    stage_contract_dir,
)


def test_init_c_preserves_vanilla_kb_signature() -> None:
    src = render_init_c()
    # Vanilla KB's signature must round-trip — the driver's extern depends on it.
    assert "void launch_gpu_implementation(void *output," in src
    assert "void *input_A," in src
    assert "void *input_B," in src
    assert "int64_t M, int64_t K, int64_t N" in src
    # RVV-flavoured: the starter pulls in riscv_vector.h so the agent's
    # rewrites can reach vsetvli / vle / vmacc straight away.
    assert "#include <riscv_vector.h>" in src


def test_driver_c_uses_mcycle_for_timing() -> None:
    """Saturn's cycle source is the standard mcycle CSR (matches
    chipyard/generators/saturn/benchmarks/common/util.h). Regression
    guard so we don't accidentally regress to a counter API that
    isn't available on Saturn's Spike fork."""
    src = render_driver_c(M=64, K=128, N=64)
    # Counter wiring uses the mcycle CSR via csrr.
    assert "csrr" in src
    assert "mcycle" in src
    # And the printf protocol matches the single-op + multi-op
    # Gemmini harnesses, so the same _parse_output works.
    assert 'printf("mismatches=%d/%d' in src
    assert 'printf("cycles=%lld' in src


def test_driver_c_stamps_per_shape_constants() -> None:
    """The driver allocates static buffers sized to (M, K, N) from the
    contract — those constants must appear in the rendered source."""
    src = render_driver_c(M=32, K=128, N=8)
    assert "#define TEST_M 32LL" in src
    assert "#define TEST_K 128LL" in src
    assert "#define TEST_N 8LL" in src


def test_stage_contract_dir_writes_both_files(tmp_path: Path) -> None:
    out = tmp_path / "saturn_stage"
    p = stage_contract_dir(out, M=64, K=128, N=64)
    assert p == out
    assert (out / "init.cu").is_file()
    assert (out / "driver.cpp").is_file()
    init_c = (out / "init.cu").read_text()
    driver_c = (out / "driver.cpp").read_text()
    assert "launch_gpu_implementation" in init_c
    assert "launch_gpu_implementation" in driver_c
    # Cross-target sanity: the Saturn templates must NOT pull in
    # gemmini.h or the gemmini-counter library — that would mean we
    # accidentally copied the Gemmini template and forgot to swap.
    assert "include/gemmini.h" not in init_c
    assert "include/gemmini.h" not in driver_c
    assert "MAIN_LD_ST_EX_CYCLES" not in driver_c
