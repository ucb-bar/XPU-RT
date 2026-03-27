"""Lower Tile IR ops to Triton kernel code."""

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
class TritonLoweringResult:
    """Result of lowering tile ops to Triton.

    Attributes:
        kernel_code: Generated Triton kernel source.
        launch_config: Triton launch configuration parameters.
        diagnostics: Any warnings or errors during lowering.
    """

    kernel_code: str
    launch_config: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


# Elementwise op -> Triton code mapping
_TRITON_ELEMENTWISE: dict[str, str] = {
    "relu": "tl.maximum({x}, 0.0)",
    "gelu": "0.5 * {x} * (1.0 + tl.math.tanh(0.7978845608 * ({x} + 0.044715 * {x} * {x} * {x})))",
    "sigmoid": "1.0 / (1.0 + tl.math.exp(-{x}))",
    "tanh": "tl.math.tanh({x})",
    "exp": "tl.math.exp({x})",
    "log": "tl.math.log({x})",
    "sqrt": "tl.math.sqrt({x})",
    "rsqrt": "tl.math.rsqrt({x})",
    "abs": "tl.abs({x})",
    "neg": "-{x}",
    "add": "{x} + {y}",
    "mul": "{x} * {y}",
    "sub": "{x} - {y}",
    "div": "{x} / {y}",
    "max": "tl.maximum({x}, {y})",
    "min": "tl.minimum({x}, {y})",
}

# Reduce kind -> Triton function
_TRITON_REDUCE: dict[str, str] = {
    "sum": "tl.sum",
    "max": "tl.max",
    "min": "tl.min",
    "mean": "tl.sum",  # divide by count after
}


def lower_tile_to_triton(ops: list[Operation]) -> TritonLoweringResult:
    """Lower a sequence of tile ops to Triton kernel source.

    Args:
        ops: List of tile dialect operations.

    Returns:
        TritonLoweringResult with generated kernel code.
    """
    lines: list[str] = []
    diagnostics: list[str] = []

    for op in ops:
        try:
            code = _lower_single_op(op)
            if code:
                lines.append(code)
        except Exception as e:
            diagnostics.append(f"Error lowering {op.name}: {e}")

    kernel_code = "\n".join(lines)
    return TritonLoweringResult(
        kernel_code=kernel_code,
        diagnostics=diagnostics,
    )


def _sym(ref: SymbolRefAttr) -> str:
    """Extract the root reference string from a SymbolRefAttr."""
    return ref.root_reference.data


def _lower_single_op(op: Operation) -> str | None:
    """Lower a single tile op to Triton code."""
    if isinstance(op, TileLoadOp):
        ref = _sym(op.src_memref)
        is_async = op.is_async is not None and op.is_async.value.data
        if is_async:
            return f"{ref}_frag = tl.load({ref}_ptr, mask={ref}_mask, other=0.0, cache_modifier='.cg')"
        return f"{ref}_frag = tl.load({ref}_ptr, mask={ref}_mask, other=0.0)"

    if isinstance(op, TileStoreOp):
        dst = _sym(op.dst_memref)
        frag = _sym(op.fragment_ref)
        return f"tl.store({dst}_ptr, {frag}_frag, mask={dst}_mask)"

    if isinstance(op, TileMMAOp):
        a, b, c = _sym(op.a_ref), _sym(op.b_ref), _sym(op.c_ref)
        return f"{c}_frag = tl.dot({a}_frag, {b}_frag, acc={c}_frag)"

    if isinstance(op, TileElementwiseOp):
        frag = _sym(op.fragment_ref)
        kind = op.op_kind.data
        template = _TRITON_ELEMENTWISE.get(kind)
        if template:
            return f"{frag}_frag = {template.format(x=f'{frag}_frag', y=f'{frag}_rhs')}"
        return f"# unsupported elementwise: {kind}"

    if isinstance(op, TileReduceOp):
        frag = _sym(op.fragment_ref)
        kind = op.reduce_kind.data
        axis = op.axis.value.data
        triton_fn = _TRITON_REDUCE.get(kind, "tl.sum")
        code = f"{frag}_reduced = {triton_fn}({frag}_frag, axis={axis})"
        if kind == "mean":
            dims = [a.value.data for a in op.shape.dims.data if isinstance(a, IntegerAttr)]
            if axis < len(dims):
                code += f"\n{frag}_reduced = {frag}_reduced / {dims[axis]}"
        return code

    if isinstance(op, TileBarrierOp):
        return "# barrier (implicit in Triton)"

    if isinstance(op, TileAsyncCopyOp):
        src = _sym(op.src_ref)
        dst = _sym(op.dst_ref)
        return f"{dst}_frag = tl.load({src}_ptr, mask={src}_mask, other=0.0, cache_modifier='.cg')"

    return None


__all__ = ["TritonLoweringResult", "lower_tile_to_triton"]
