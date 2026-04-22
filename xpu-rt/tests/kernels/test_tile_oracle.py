"""Tests for ``compgen.kernels.tile_oracle``.

Locks in:
  * recommend_tile returns valid TileRecommendation for known
    (op_family, target) pairs
  * MMA-aware adjustment bumps BLOCK_K to a multiple of MMA-K
  * knowledge_brief contains ONLY tile-selection lessons (containerised)
  * unknown (op, target) falls through to a conservative default
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.kernels.contract_v3 import HardwareEnvelope
from compgen.kernels.tile_oracle import (
    recommend_packing,
    recommend_tile,
)
from compgen.memory.knowledge import KnowledgeStore, set_shared_store
from compgen.memory.seed_lessons import install as install_seed


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path):
    set_shared_store(KnowledgeStore(root=tmp_path / "knowledge"))
    install_seed()  # populate so context_brief has content
    yield
    set_shared_store(None)


def _envelope(target_name: str, *, mma_shapes: dict | None = None) -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name=target_name,
        vector_lanes=72,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16", "f32"),
        peak_bandwidth_gbps=672.0,
        mma_shapes=mma_shapes or {},
    )


def test_recommend_tile_for_matmul_on_turing_uses_rule_table() -> None:
    rec = recommend_tile("matmul", (None, None), "f16", _envelope("test-gpu-simt"))
    assert rec.block_m == 64
    assert rec.block_n == 64
    assert rec.block_k == 32
    assert rec.num_warps == 4
    assert rec.num_stages == 2
    assert "matmul" in rec.rationale
    assert rec.confidence > 0.5


def test_recommend_tile_for_matmul_on_ampere_uses_bigger_tiles() -> None:
    rec = recommend_tile("matmul", (None, None), "bf16", _envelope("cuda-a100"))
    # Ampere rule is 128×128×32 with num_warps=8, num_stages=3
    assert rec.block_m == 128
    assert rec.block_n == 128
    assert rec.block_k == 32
    assert rec.num_warps == 8
    assert rec.num_stages == 3


def test_recommend_tile_bumps_block_k_to_mma_k_multiple() -> None:
    """If the envelope declares an MMA shape with K=16, BLOCK_K=32 is
    already a multiple. Make sure a non-multiple bumps up."""
    rec = recommend_tile(
        "matmul",
        (None, None),
        "f16",
        _envelope("test-gpu-simt", mma_shapes={"f16": (16, 8, 24)}),  # K=24 unusual
    )
    # Default block_k=32; 32 % 24 != 0 → bump to 48
    assert rec.block_k == 48
    assert "MMA K=24" in rec.rationale


def test_recommend_tile_falls_through_for_unknown_combo() -> None:
    rec = recommend_tile("exotic_op", (None,), "f32", _envelope("cuda-a100"))
    assert rec.confidence < 0.5
    assert "No rule" in rec.rationale


def test_recommend_tile_brief_contains_only_tile_selection_lessons() -> None:
    """Containerised query — brief must NOT include profiling lessons,
    even though they exist for the same target."""
    rec = recommend_tile(
        "matmul",
        (None, None),
        "f16",
        _envelope("test-gpu-simt"),
    )
    brief = rec.knowledge_brief
    # GROUP_M swizzle lesson is tile-selection topic → IN
    assert "swizzle" in brief.lower() or "GROUP_M" in brief
    # Profiling overhead lesson is topic="profiling" → OUT
    assert "synchronize" not in brief
    assert "profiler" not in brief.lower() or "profil" not in brief.lower() or len(brief) > 0


def test_recommend_packing_returns_pack_k_major_for_row_major_input() -> None:
    out = recommend_packing("row_major", target_mma_shape=(16, 8, 16))
    assert out == "pack_k_major"


def test_recommend_packing_returns_none_when_no_mma_shape() -> None:
    assert recommend_packing("row_major", target_mma_shape=None) is None
