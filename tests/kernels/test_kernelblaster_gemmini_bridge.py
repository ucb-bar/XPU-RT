"""Tests for the vanilla-KB Gemmini-prompt-injection bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xpu_rt.kernels.kernelblaster_gemmini_bridge import (
    KBVanillaResult,
    KernelBlasterGemminiBridge,
    _build_kb_prompt,
    _compile_check,
    _intrinsic_use_rate,
    _shape_consistency,
    _starting_scalar_matmul,
    _STRATEGIES,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
    _check_toolchain,
    _ToolchainMissing,
)
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory import target_knowledge as tk


def _matmul_contract() -> KernelContract:
    return KernelContract(
        region_id="kb_bridge_test",
        op_family="matmul",
        input_shapes=((16, 64), (64, 64)),
        output_shapes=((16, 64),),
        dtypes=("i8", "i8", "i32"),
        layout="row_major",
        target_name="gemmini_mx",
        objective="latency",
    )


def _seed_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tk.TargetKnowledgeCard:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path))
    card = tk.TargetKnowledgeCard(
        target_id="gemmini_mx",
        target_profile_ref="configs/targets/gemmini_mx.yaml",
        hardware_spec=tk.HardwareSpec(
            isa_family="rocc-systolic",
            instructions=(
                tk.ISAInstruction(mnemonic="mvin", signature="rs1, rs2", funct_code=2),
                tk.ISAInstruction(mnemonic="mvout", signature="rs1, rs2", funct_code=3),
            ),
            intrinsics=(
                tk.IntrinsicSignature(
                    name="gemmini_mvin",
                    c_signature="#define gemmini_mvin(d, s)",
                    summary="DMA into scratchpad",
                ),
                tk.IntrinsicSignature(
                    name="gemmini_mvout",
                    c_signature="#define gemmini_mvout(d, s)",
                    summary="DMA from scratchpad to DRAM",
                ),
                tk.IntrinsicSignature(
                    name="gemmini_compute_preloaded",
                    c_signature="#define gemmini_compute_preloaded(A, BD)",
                    summary="Run preloaded matmul",
                ),
            ),
            constraints=("DIM=16",),
        ),
    )
    saved = tk.save(card)
    # Materialise the bucket files so _concat_card_buckets has content.
    saved.bucket_path("isa").write_text("## ISA\n- mvin: load tile\n- mvout: store tile\n")
    saved.bucket_path("intrinsics").write_text(
        "## INTRINSICS\n- gemmini_mvin(dram, spad)\n- gemmini_mvout(dram, spad)\n"
        "- gemmini_compute_preloaded(A, BD)\n"
    )
    saved.bucket_path("constraints").write_text("## CONSTRAINTS\n- DIM=16\n")
    return saved


# ---------------------------------------------------------------------------
# Pure-function units
# ---------------------------------------------------------------------------


def test_starting_scalar_matmul_substitutes_dims() -> None:
    code = _starting_scalar_matmul(_matmul_contract())
    assert "#define M 16" in code
    assert "#define K 64" in code
    assert "#define N 64" in code
    assert "void kernel_under_test" in code


def test_intrinsic_use_rate_all_known() -> None:
    card = tk.TargetKnowledgeCard(
        target_id="t",
        target_profile_ref="",
        hardware_spec=tk.HardwareSpec(
            isa_family="rocc-systolic",
            intrinsics=(
                tk.IntrinsicSignature(name="gemmini_mvin", c_signature="x"),
                tk.IntrinsicSignature(name="gemmini_mvout", c_signature="x"),
            ),
        ),
    )
    kernel = "void k() { gemmini_mvin(a, b); gemmini_mvout(c, d); gemmini_mvin(e, f); }"
    rate, matched, total = _intrinsic_use_rate(kernel, card)
    assert (rate, matched, total) == (1.0, 3, 3)


def test_intrinsic_use_rate_mixed_known_and_made_up() -> None:
    card = tk.TargetKnowledgeCard(
        target_id="t",
        target_profile_ref="",
        hardware_spec=tk.HardwareSpec(
            isa_family="rocc-systolic",
            intrinsics=(
                tk.IntrinsicSignature(name="gemmini_mvin", c_signature="x"),
            ),
        ),
    )
    kernel = (
        "void k() { gemmini_mvin(a, b); gemmini_fake_function(c); "
        "gemmini_made_up(d); }"
    )
    rate, matched, total = _intrinsic_use_rate(kernel, card)
    assert matched == 1
    assert total == 3
    assert rate == pytest.approx(1 / 3, rel=1e-6)


def test_intrinsic_use_rate_no_calls_returns_zero() -> None:
    card = tk.TargetKnowledgeCard(
        target_id="t",
        target_profile_ref="",
        hardware_spec=tk.HardwareSpec(isa_family="rocc-systolic"),
    )
    rate, matched, total = _intrinsic_use_rate("// just a comment\n", card)
    assert (rate, matched, total) == (0.0, 0, 0)


def test_shape_consistency_pass_and_fail() -> None:
    contract = _matmul_contract()
    good = "void k(){ int M=16,K=64,N=64; }"
    bad = "void k(){ int M=8,K=32,N=16; }"
    ok_good, miss_good = _shape_consistency(good, contract)
    assert ok_good is True
    assert miss_good == []
    ok_bad, miss_bad = _shape_consistency(bad, contract)
    assert ok_bad is False
    assert set(miss_bad) >= {64}


def test_build_kb_prompt_includes_required_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _seed_card(tmp_path, monkeypatch)
    prompt = _build_kb_prompt(
        contract=_matmul_contract(),
        card=card,
        strategy_name="tile_and_dma_overlap",
        strategy_desc="…",
        prior_attempts=(),
        starting_source="// starting source",
    )
    assert "OPTIMIZATION DATABASE" in prompt
    assert "gemmini_mvin" in prompt
    assert "STRATEGY FOR THIS ROUND: tile_and_dma_overlap" in prompt
    assert "STARTING SCALAR SOURCE" in prompt
    assert "// starting source" in prompt
    assert "kb_bridge_test" in prompt  # region_id from contract


def test_build_kb_prompt_formats_prior_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _seed_card(tmp_path, monkeypatch)
    prompt = _build_kb_prompt(
        contract=_matmul_contract(),
        card=card,
        strategy_name="weight_stationary_dataflow",
        strategy_desc="…",
        prior_attempts=(
            {"strategy": "tile_and_dma_overlap", "compile": False, "intrinsic_use_rate": 0.2, "notes": "missing semicolon"},
        ),
        starting_source="// src",
    )
    assert "PRIOR ATTEMPTS" in prompt
    assert "attempt 1:" in prompt
    assert "tile_and_dma_overlap" in prompt
    assert "missing semicolon" in prompt


# ---------------------------------------------------------------------------
# Integration with a mocked LLM
# ---------------------------------------------------------------------------


class _MockGen:
    """Stand-in for KernelGeneratorLLM that returns scripted emissions."""

    def __init__(self, emissions: list[str], strategy_names: list[str]) -> None:
        self.emissions = emissions
        self.strategy_names = strategy_names
        self.calls: list[Any] = []

    def propose(self, req):  # type: ignore[no-untyped-def]
        idx = len(self.calls)
        self.calls.append(req)
        from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse
        return ProposeResponse(
            kernel_code=self.emissions[idx] if idx < len(self.emissions) else "",
            language="c",
            action=self.strategy_names[idx] if idx < len(self.strategy_names) else "",
        )


def test_bridge_picks_best_compiling_emission_over_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _seed_card(tmp_path, monkeypatch)
    # Round 0: syntactically broken (no semicolon)
    # Round 1: compiles + uses real intrinsics + has shape literals
    e0 = """
#include <stdint.h>
void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {
    int M = 16
}
"""
    e1 = """
#include <stdint.h>
#include "include/gemmini.h"
void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {
    int M = 16, K = 64, N = 64;
    (void)A; (void)B; (void)C;
    gemmini_mvin(0, 0);
    gemmini_mvout(0, 0);
}
"""
    mock = _MockGen([e0, e1], ["tile_and_dma_overlap", "weight_stationary_dataflow"])
    monkeypatch.setattr(
        "xpu_rt.kernels.kernelblaster_gemmini_bridge.KernelGeneratorLLM",
        lambda model: mock,
    )

    bridge = KernelBlasterGemminiBridge(target_card=card, max_rounds=2)
    result = bridge.run(_matmul_contract())

    assert result.rounds == 2
    # Best result is the compiling one, regardless of round order.
    if _toolchain_present():
        assert result.compile is True
        assert result.final_strategy == "weight_stationary_dataflow"
        assert result.intrinsic_use_rate > 0
        assert result.shape_consistency is True


def test_bridge_records_per_attempt_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _seed_card(tmp_path, monkeypatch)
    emissions = ["// r0", "// r1"]
    mock = _MockGen(emissions, [_STRATEGIES[0][0], _STRATEGIES[1][0]])
    monkeypatch.setattr(
        "xpu_rt.kernels.kernelblaster_gemmini_bridge.KernelGeneratorLLM",
        lambda model: mock,
    )

    bridge = KernelBlasterGemminiBridge(target_card=card, max_rounds=2)
    result = bridge.run(_matmul_contract())
    assert len(result.attempts) == 2
    assert result.attempts[0]["strategy"] == _STRATEGIES[0][0]
    assert result.attempts[1]["strategy"] == _STRATEGIES[1][0]
    # Both attempts are "no-real-kernel" stubs; intrinsic_use_rate = 0.
    assert result.intrinsic_use_rate == 0.0


def _toolchain_present() -> bool:
    try:
        _check_toolchain()
    except _ToolchainMissing:
        return False
    return True
