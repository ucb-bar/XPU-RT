"""Lower Tile IR ops to Exo proc fragments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerAttr, SymbolRefAttr
from xdsl.ir import Operation

from compgen.ir.tile.ops import (
    TileAsyncCopyOp,
    TileBarrierOp,
    TileElementwiseOp,
    TileLoadOp,
    TileMMAOp,
    TileReduceOp,
    TileStoreOp,
)


@dataclass(frozen=True)
class ExoLoweringResult:
    """Result of lowering tile ops to Exo.

    Attributes:
        proc_source: Generated Exo proc body source.
        schedule_hints: Suggested schedule operations for optimization.
        diagnostics: Any warnings or errors during lowering.
    """

    proc_source: str
    schedule_hints: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def lower_tile_to_exo(
    ops: list[Operation],
    target_kit_name: str = "generic",
) -> ExoLoweringResult:
    """Lower tile ops to Exo proc body fragments.

    Args:
        ops: List of tile dialect operations.
        target_kit_name: Name of the Exo target kit for instruction mapping.

    Returns:
        ExoLoweringResult with generated proc source.
    """
    lines: list[str] = []
    schedule_hints: list[str] = []
    diagnostics: list[str] = []

    for op in ops:
        try:
            code, hints = _lower_single_op(op, target_kit_name)
            if code:
                lines.append(code)
            schedule_hints.extend(hints)
        except Exception as e:
            diagnostics.append(f"Error lowering {op.name}: {e}")

    proc_source = "\n".join(lines)
    return ExoLoweringResult(
        proc_source=proc_source,
        schedule_hints=schedule_hints,
        diagnostics=diagnostics,
    )


def _sym(ref: SymbolRefAttr) -> str:
    """Extract the root reference string from a SymbolRefAttr."""
    return ref.root_reference.data


def _dims(op: Any) -> list[int]:
    """Extract integer dimensions from a tile op's shape attribute."""
    return [a.value.data for a in op.shape.dims.data if isinstance(a, IntegerAttr)]


def _lower_single_op(op: Operation, kit: str) -> tuple[str | None, list[str]]:
    """Lower a single tile op to Exo code + schedule hints."""
    hints: list[str] = []

    if isinstance(op, TileLoadOp):
        ref = _sym(op.src_memref)
        mem = op.memory_class.kind.data
        dims = _dims(op)
        dim_str = ", ".join(str(d) for d in dims)
        code = f"# tile.load {ref} [{dim_str}] from {mem}"
        if mem != "global":
            hints.append(f"stage_mem({ref}, ...)")
        return code, hints

    if isinstance(op, TileStoreOp):
        dst = _sym(op.dst_memref)
        frag = _sym(op.fragment_ref)
        return f"# tile.store {frag} -> {dst}", hints

    if isinstance(op, TileMMAOp):
        a, b, c = _sym(op.a_ref), _sym(op.b_ref), _sym(op.c_ref)
        dims = _dims(op)
        code = (
            f"# tile.mma {c} += {a} @ {b}  shape={dims}\n"
            f"for i in seq(0, {dims[0] if dims else 'M'}):\n"
            f"    for j in seq(0, {dims[1] if len(dims) > 1 else 'N'}):\n"
            f"        for k in seq(0, {dims[2] if len(dims) > 2 else 'K'}):\n"
            f"            {c}[i, j] += {a}[i, k] * {b}[k, j]"
        )
        hints.append(f"# Consider: divide_loop, reorder_loops for {c} MMA")
        if kit != "generic":
            hints.append(f"# Consider: replace_all with {kit} compute instruction")
        return code, hints

    if isinstance(op, TileElementwiseOp):
        frag = _sym(op.fragment_ref)
        kind = op.op_kind.data
        dims = _dims(op)
        n = dims[0] if dims else "N"
        code = f"for i in seq(0, {n}):\n    {frag}[i] = {kind}({frag}[i])"
        return code, hints

    if isinstance(op, TileReduceOp):
        frag = _sym(op.fragment_ref)
        kind = op.reduce_kind.data
        axis = op.axis.value.data
        dims = _dims(op)
        n = dims[axis] if axis < len(dims) else "N"
        code = f"# tile.reduce {kind} over axis {axis}\nfor i in seq(0, {n}):\n    result[0] += {frag}[i]"
        return code, hints

    if isinstance(op, TileBarrierOp):
        scope = op.scope.data
        return f"# tile.barrier scope={scope}\nfence()", hints

    if isinstance(op, TileAsyncCopyOp):
        src = _sym(op.src_ref)
        dst = _sym(op.dst_ref)
        src_mem = op.src_memory_class.kind.data
        dst_mem = op.dst_memory_class.kind.data
        code = f"# tile.async_copy {src}({src_mem}) -> {dst}({dst_mem})"
        if kit != "generic":
            hints.append(f"# Consider: replace with {kit} DMA instruction")
        return code, hints

    return None, hints


__all__ = ["ExoLoweringResult", "lower_tile_to_exo"]
