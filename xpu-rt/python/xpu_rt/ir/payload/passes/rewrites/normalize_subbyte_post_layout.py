"""``normalize_subbyte_post_layout`` -- realign sub-byte pack dims to
``dma_line_bytes``.

Reconstruction of XLA's ``SubByteNormalization`` (the post-layout
pass, complementary to Wave 4's ``normalize_subbyte``). Zero
external references; CompGen owns the rewrite.

Operates on :class:`ExecutionPlan`. Reads the sub-byte canonical
decisions made in Wave 4 via ``compgen.subbyte_canonical`` /
``compgen.subbyte_boundary`` attributes on ops (stored in
``plan.summary["subbyte_ops"]`` after the agent layer copies them
over), and:

1. For each buffer whose producer op has a ``subbyte_canonical``
   tag, compute the packed-stride (``bit_width * innermost_extent
   / 8`` rounded up to ``dma_line_bytes``).
2. Record the realigned packed stride on
   ``plan.summary["subbyte_buffer_strides"]``.
3. When a buffer's size (in bytes) doesn't align to
   ``dma_line_bytes``, pad its ``BufferDescriptor.size_bytes`` up.

The pass does NOT change ownership or memory space decisions made
earlier in the pipeline; it's strictly a byte-level realignment.

Config:

- ``dma_line_bytes`` -- the target DMA line size (default 64,
  matching typical DRAM burst granularity).
- ``subbyte_op_summary_key`` -- where to read tagged ops from on
  the plan summary (default ``"subbyte_ops"``). Populated by the
  pipeline driver that runs Wave 4's normalize_subbyte first.

LLM-tool signature:

    tool_name="normalize_subbyte_post_layout"
    wraps_pass="CompGen:SubByteNormalizationPostLayout"
    invent_slot="runtime/subbyte_realignment"
    policy="AlignPackedStridesToDMALine"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.runtime.execution_plan import ExecutionPlan


@dataclass(frozen=True)
class NormalizeSubbytePostLayoutConfig:
    dma_line_bytes: int = 64
    subbyte_op_summary_key: str = "subbyte_ops"


@dataclass
class NormalizeSubbytePostLayoutStats:
    buffers_realigned: int = 0
    buffers_skipped: int = 0
    strides_by_buffer: dict[str, int] = field(default_factory=dict)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def run_normalize_subbyte_post_layout(
    plan: ExecutionPlan,
    *,
    config: NormalizeSubbytePostLayoutConfig | None = None,
) -> NormalizeSubbytePostLayoutStats:
    cfg = (
        config
        if config is not None
        else NormalizeSubbytePostLayoutConfig()
    )
    stats = NormalizeSubbytePostLayoutStats()

    # subbyte_ops summary format: a list of dicts
    #   { "buffer_id": ..., "bit_width": 4, "pack_dim": 1 }
    # Populated by the pipeline driver after running
    # ``normalize_subbyte`` (Wave 4.4). When the key is absent this
    # pass is a no-op.
    subbyte_ops = plan.summary.get(cfg.subbyte_op_summary_key, [])
    if not subbyte_ops:
        return stats

    strides: dict[str, int] = dict(
        plan.summary.get("subbyte_buffer_strides", {})
    )

    for entry in subbyte_ops:
        bid = entry.get("buffer_id")
        if not bid:
            continue
        buf = None
        for b in plan.buffers:
            if b.buffer_id == bid:
                buf = b
                break
        if buf is None:
            stats.buffers_skipped += 1
            continue

        bw = int(entry.get("bit_width", 8))
        pad_size = _align_up(buf.size_bytes, cfg.dma_line_bytes)
        if pad_size > buf.size_bytes:
            buf.size_bytes = pad_size
        strides[bid] = _align_up(
            (bw * buf.size_bytes + 7) // 8,
            cfg.dma_line_bytes,
        )
        stats.buffers_realigned += 1
        stats.strides_by_buffer[bid] = strides[bid]

    plan.summary["subbyte_buffer_strides"] = strides
    plan.summary["subbyte_dma_line_bytes"] = cfg.dma_line_bytes
    return stats


__all__ = [
    "NormalizeSubbytePostLayoutConfig",
    "NormalizeSubbytePostLayoutStats",
    "run_normalize_subbyte_post_layout",
]
