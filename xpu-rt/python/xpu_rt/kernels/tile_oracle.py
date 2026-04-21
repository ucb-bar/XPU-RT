"""Tile + packing recommendations driven by ``HardwareEnvelope`` +
narrow knowledge-store queries.

The oracle's job: given (op-family, shape, dtype, target), return a
first-pass tile recommendation (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
num_stages, GROUP_M). The agent's codegen / refinement loop reads the
recommendation as prompt context — autotune still has the final word
because it measures ground truth.

Containerised knowledge: the oracle queries the knowledge store with
``stage="kernel-gen"`` + ``op_family=op`` + ``topic="tile-selection"``
so it pulls ONLY tile-relevant lessons (not, say, lessons about
profiling overhead) — keeps the prompt tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.kernels.contract_v3 import HardwareEnvelope
from compgen.memory.knowledge import shared_store


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileRecommendation:
    """First-pass tile-shape pick for the codegen prompt.

    Attributes:
        block_m, block_n, block_k: Tile dims; ``None`` when the op
            doesn't have that dim (e.g. POINTWISE has no K).
        num_warps, num_stages, group_m: Triton-style scheduling knobs.
        rationale: Human-readable explanation referencing the lessons /
            envelope facts that drove the pick. Goes into the prompt.
        confidence: 0..1 — how sure the oracle is. Low confidence
            means autotune should sweep aggressively; high means start
            with a tight grid.
        knowledge_brief: Markdown excerpt of relevant lessons from the
            knowledge store (already filtered to tile-selection topic).
    """

    block_m: int | None = None
    block_n: int | None = None
    block_k: int | None = None
    num_warps: int = 4
    num_stages: int = 2
    group_m: int = 8
    rationale: str = ""
    confidence: float = 0.5
    knowledge_brief: str = ""


# ---------------------------------------------------------------------------
# Per-target rule tables (first-pass; autotune corrects)
# ---------------------------------------------------------------------------


# Default tile picks per (op_family, target_class). Authored, not learned.
# Returns the BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages tuple.
_RULES: dict[tuple[str, str], tuple[int, int, int, int, int]] = {
    # Turing (sm_75) — small SMEM, no async copy
    ("matmul", "turing"):       (64, 64, 32, 4, 2),
    ("batch_matmul", "turing"): (64, 64, 32, 4, 2),
    ("softmax", "turing"):      (1, 1024, 0, 4, 1),         # row-per-CTA
    ("rmsnorm", "turing"):      (1, 1024, 0, 4, 1),
    ("silu", "turing"):         (1, 1024, 0, 4, 1),
    # Ampere (sm_80) — bigger tiles + async copy
    ("matmul", "ampere"):       (128, 128, 32, 8, 3),
    ("batch_matmul", "ampere"): (64, 64, 32, 4, 3),
    ("softmax", "ampere"):      (1, 2048, 0, 8, 2),
    ("rmsnorm", "ampere"):      (1, 2048, 0, 8, 2),
    # Hopper (sm_90)
    ("matmul", "hopper"):       (128, 256, 64, 8, 4),
    # Hexagon NPU
    ("matmul", "hexagon"):      (32, 32, 32, 1, 1),
    # CPU (no tensor cores; cache-line-aligned tiles)
    ("matmul", "cpu"):          (32, 32, 64, 1, 1),
}


# Map `target_name` → coarse architecture key used in `_RULES`.
def _arch_key(target_name: str) -> str:
    n = target_name.lower()
    if "turing" in n or "titan-rtx" in n or "test-gpu-simt" in n:
        return "turing"
    if "ampere" in n or "a100" in n or "rtx-30" in n or "rtx-40" in n:
        return "ampere"
    if "hopper" in n or "h100" in n:
        return "hopper"
    if "hexagon" in n or "openq" in n:
        return "hexagon"
    if "cpu" in n or "host" in n:
        return "cpu"
    if "rocm" in n or "mi" in n:
        return "ampere"  # ROCm-Triton tiles roughly like Ampere
    return "ampere"  # safest default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_tile(
    op_family: str,
    shape: tuple[int | None, ...],
    dtype: str,
    envelope: HardwareEnvelope,
) -> TileRecommendation:
    """First-pass tile recommendation for the codegen prompt.

    Args:
        op_family: Op-family name (``matmul``, ``batch_matmul``,
            ``softmax``, ``rmsnorm``, ``silu``, …).
        shape: Concrete or symbolic shape; ``None`` for dynamic dims.
        dtype: One of ``f16, bf16, f32, …``.
        envelope: Target ``HardwareEnvelope`` (carries vector_lanes,
            scratchpad_bytes, mma_shapes, peak_compute_per_dtype, …).

    Returns:
        ``TileRecommendation`` with first-pass dims + a prompt-ready
        ``knowledge_brief`` excerpting only the tile-selection lessons
        relevant to ``(target, op_family)``.
    """
    arch = _arch_key(envelope.target_name)
    key = (op_family, arch)
    rule = _RULES.get(key)

    # Adjust if the op has an MMA-aligned tile in the envelope
    if rule is None:
        # Fall back to a conservative default: small tile, single warp
        rule = (32, 32, 32, 1, 1)
        rationale = (
            f"No rule for ({op_family!r}, {arch!r}); using conservative "
            "32×32×32 first pass — autotune will correct."
        )
        confidence = 0.2
    else:
        bm, bn, bk, nw, ns = rule
        rationale_parts = [
            f"({op_family}, {arch}) → BLOCK_M={bm} BLOCK_N={bn} BLOCK_K={bk} "
            f"num_warps={nw} num_stages={ns}"
        ]
        # MMA-aware adjustment: if envelope declares an MMA shape for this
        # dtype, prefer BLOCK_K to be a multiple of the MMA K-dim.
        mma = envelope.mma_shapes.get(dtype)
        if mma:
            mma_k = mma[2]
            if bk % mma_k != 0:
                bk = ((bk + mma_k - 1) // mma_k) * mma_k
                rationale_parts.append(
                    f"BLOCK_K bumped to {bk} to align with MMA K={mma_k}"
                )
                rule = (bm, bn, bk, nw, ns)
        rationale = "; ".join(rationale_parts)
        confidence = 0.6

    bm, bn, bk, nw, ns = rule

    # Containerised knowledge query: only tile-selection lessons for THIS
    # op_family on THIS target. Caps at 5 to keep the prompt tight.
    brief = shared_store().context_brief(
        envelope.target_name,
        stage="kernel-gen",
        op_family=op_family,
        topic="tile-selection",
        max_lessons=5,
    )

    return TileRecommendation(
        block_m=bm if op_family in ("matmul", "batch_matmul") else None,
        block_n=bn,
        block_k=bk if op_family in ("matmul", "batch_matmul") else None,
        num_warps=nw,
        num_stages=ns,
        group_m=8,                 # safe default; overridable per call
        rationale=rationale,
        confidence=confidence,
        knowledge_brief=brief,
    )


def recommend_packing(
    input_layout: str,
    target_mma_shape: tuple[int, int, int] | None,
) -> str | None:
    """Stub for the packing oracle — returns a packing-kernel hint.

    Returns the *name* of a packing transform to apply ("pack_k_major",
    "pack_n_blocked_32", …) or ``None`` when no packing is needed.

    Today: trivial heuristic. The full version (W2 deeper) consults
    the knowledge store for `topic="memory-layout"` lessons.
    """
    if target_mma_shape is None:
        return None
    if input_layout == "row_major" and target_mma_shape[2] >= 16:
        # MMA wants K-major chunks; if we're row-major, packing helps
        return "pack_k_major"
    return None


__all__ = [
    "TileRecommendation",
    "recommend_packing",
    "recommend_tile",
]
