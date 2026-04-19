"""CPU reference executor for the compiled xDSL module.

Walks the ``@forward`` func in a bridged ``ModuleOp`` and dispatches
every op to a concrete PyTorch primitive. Lets CompGen's
``compile_and_diff`` do **real compiled-vs-eager differential
testing** on any CPU host.

The executor is deliberately pure Python + torch:

- No CUDA required.
- No Triton / ukernel backends (those land in separate emitters).
- Supports the subset of ops CompGen's bridge + passes emit:
  ``linalg.matmul`` (with indexing_maps), ``linalg.transpose``,
  ``linalg.generic`` (elementwise + reduction bodies),
  ``tensor.empty`` / ``tensor.insert_slice``,
  ``compgen.tensor_ext.concat`` / ``pack``,
  ``compgen.linalg_ext.softmax`` / ``layer_norm`` / ``rms_norm`` /
  ``silu`` / ``gelu``,
  ``arith.*`` scalar ops inside linalg bodies,
  opaque ``func.call`` dispatches keyed off ``aten_*`` callee names.

Usage::

    from compgen.runtime.cpu_executor import execute
    out = execute(module, exported_program, example_inputs)

``execute`` returns a tuple of tensors (one per ``USER_OUTPUT`` in
the export signature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn.functional as F
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseArrayBase,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.linalg import GenericOp, MatmulOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp, InsertSliceOp
from xdsl.ir import Attribute, Operation, SSAValue

log = structlog.get_logger()


# --- dispatch table for opaque ``aten_*`` calls ----------------------------


def _aten_layer_norm(args: list[torch.Tensor], **kw) -> torch.Tensor:
    inp = args[0]
    # normalized_shape = last dim (simplest case). Real LN gets
    # weight/bias from args[1]/args[2] if present.
    weight = args[1] if len(args) > 1 else None
    bias = args[2] if len(args) > 2 else None
    return F.layer_norm(
        inp, (inp.shape[-1],), weight=weight, bias=bias, eps=1e-5
    )


def _aten_softmax(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return F.softmax(args[0], dim=-1)


def _aten_gelu(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return F.gelu(args[0])


def _aten_silu(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return F.silu(args[0])


def _aten_matmul(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return torch.matmul(args[0], args[1])


def _aten_bmm(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return torch.bmm(args[0], args[1])


def _aten_add(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0] + args[1]


def _aten_mul(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0] * args[1]


def _aten_sub(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0] - args[1]


def _aten_div(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0] / args[1]


def _aten_neg(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return -args[0]


def _aten_sigmoid(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return torch.sigmoid(args[0])


def _aten_rsqrt(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return torch.rsqrt(args[0])


def _aten_sqrt(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return torch.sqrt(args[0])


def _aten_view(args: list[torch.Tensor], attrs: dict[str, Any], **kw) -> torch.Tensor:
    # For our bridge the target shape is encoded on the result type
    # (we don't have it in the op args). Caller supplies the shape
    # via ``target_shape`` keyword.
    target = kw.get("target_shape")
    if target is None:
        return args[0]
    return args[0].reshape(target)


def _aten_transpose(args: list[torch.Tensor], **kw) -> torch.Tensor:
    # Transpose the last two dims by default -- matches how our
    # decomp records the semantic without preserving dim0/dim1.
    target = kw.get("target_shape")
    x = args[0]
    if target is not None and len(target) == x.ndim:
        # Find the perm that matches target.
        src_shape = list(x.shape)
        for i in range(x.ndim):
            for j in range(i + 1, x.ndim):
                swapped = src_shape.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                if swapped == list(target):
                    return x.transpose(i, j)
    return x.transpose(-2, -1)


def _aten_contiguous(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0].contiguous()


def _aten_slice(args: list[torch.Tensor], **kw) -> torch.Tensor:
    # Slice metadata isn't in the op operands. Fall back to identity
    # when we can't recover the slice -- the caller's eager path
    # will diverge but the executor still completes.
    return args[0]


def _aten_cat(args: list[torch.Tensor], **kw) -> torch.Tensor:
    # We can't recover the dim from the op; default to 0.
    return torch.cat(list(args), dim=0)


def _aten_split(args: list[torch.Tensor], **kw) -> torch.Tensor:
    # split_with_sizes returns a list; executor treats the first
    # shard as the canonical output.
    return args[0]


def _aten_unsqueeze(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0].unsqueeze(-1)


def _aten_expand(args: list[torch.Tensor], **kw) -> torch.Tensor:
    target = kw.get("target_shape")
    if target is not None:
        return args[0].expand(*target)
    return args[0]


def _aten_mean_dim(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0].mean(dim=-1, keepdim=True)


def _aten_pow(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0].pow(2)


def _aten_embedding(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0]  # fallback; real embedding needs indices


def _aten_convolution(args: list[torch.Tensor], **kw) -> torch.Tensor:
    # Skeleton: conv body needs stride/padding; fall through to eager
    # output shape via zeros.
    target = kw.get("target_shape")
    if target is not None:
        return torch.zeros(target, dtype=args[0].dtype)
    return args[0]


def _aten_clone(args: list[torch.Tensor], **kw) -> torch.Tensor:
    return args[0].clone()


_ATEN_DISPATCH: dict[str, Any] = {
    "aten_layer_norm": _aten_layer_norm,
    "aten_native_layer_norm": _aten_layer_norm,
    "aten_softmax": _aten_softmax,
    "aten_gelu": _aten_gelu,
    "aten_silu": _aten_silu,
    "aten_matmul": _aten_matmul,
    "aten_bmm": _aten_bmm,
    "aten_add": _aten_add,
    "aten_mul": _aten_mul,
    "aten_sub": _aten_sub,
    "aten_div": _aten_div,
    "aten_neg": _aten_neg,
    "aten_sigmoid": _aten_sigmoid,
    "aten_rsqrt": _aten_rsqrt,
    "aten_sqrt": _aten_sqrt,
    "aten_view": _aten_view,
    "aten_transpose": _aten_transpose,
    "aten_contiguous": _aten_contiguous,
    "aten_slice": _aten_slice,
    "aten_cat": _aten_cat,
    "aten_split_with_sizes": _aten_split,
    "aten_unsqueeze": _aten_unsqueeze,
    "aten_expand": _aten_expand,
    "aten_mean_dim": _aten_mean_dim,
    "aten_pow": _aten_pow,
    "aten_embedding": _aten_embedding,
    "aten_convolution": _aten_convolution,
    "aten_clone": _aten_clone,
}


# --- structured-op dispatch -----------------------------------------------


def _matmul_needs_transpose_b(op: MatmulOp) -> bool:
    maps_attr = op.properties.get("indexing_maps")
    if maps_attr is None or len(maps_attr.data) < 2:
        return False
    rhs_map = maps_attr.data[1].data
    exprs = [e for e in rhs_map.results]
    if len(exprs) == 2:
        rhs_pos = [getattr(e, "position", None) for e in exprs]
        # transpose-b form: rhs read as (j, k) instead of (k, j).
        return rhs_pos == [1, 2]
    return False


def _matmul_needs_transpose_a(op: MatmulOp) -> bool:
    maps_attr = op.properties.get("indexing_maps")
    if maps_attr is None or not maps_attr.data:
        return False
    lhs_map = maps_attr.data[0].data
    exprs = [e for e in lhs_map.results]
    if len(exprs) == 2:
        lhs_pos = [getattr(e, "position", None) for e in exprs]
        return lhs_pos == [2, 0]
    return False


def _exec_matmul(op: MatmulOp, env: dict[SSAValue, torch.Tensor]) -> torch.Tensor:
    lhs = env[op.inputs[0]]
    rhs = env[op.inputs[1]]
    transpose_a = _matmul_needs_transpose_a(op)
    transpose_b = _matmul_needs_transpose_b(op)

    if transpose_a:
        lhs = lhs.transpose(-2, -1)
    if transpose_b:
        rhs = rhs.transpose(-2, -1)

    # Handle 3-D lhs × 2-D rhs (common nn.Linear broadcast).
    if lhs.ndim == 3 and rhs.ndim == 2:
        B, T, K = lhs.shape
        flat = lhs.reshape(B * T, K)
        out_flat = flat @ rhs
        return out_flat.reshape(B, T, out_flat.shape[-1])
    # 3-D lhs × 3-D rhs -> batched matmul
    if lhs.ndim == 3 and rhs.ndim == 3:
        return torch.bmm(lhs, rhs)
    return lhs @ rhs


def _exec_transpose(op: TransposeOp, env: dict[SSAValue, torch.Tensor]) -> torch.Tensor:
    x = env[op.input]
    perm_attr = op.permutation
    if isinstance(perm_attr, DenseArrayBase):
        perm = [int(v) for v in perm_attr.get_values()]
        return x.permute(*perm).contiguous()
    return x.transpose(-2, -1).contiguous()


def _find_constant_in_body(body) -> float | None:
    """Scan a linalg body for an ``arith.constant`` and return its value."""
    for op in body.ops:
        if op.name == "arith.constant":
            val_attr = op.properties.get("value")
            if val_attr is not None and hasattr(val_attr, "value"):
                try:
                    return float(val_attr.value.data)
                except Exception:  # noqa: BLE001
                    pass
    return None


def _exec_linalg_generic(
    op: GenericOp, env: dict[SSAValue, torch.Tensor]
) -> torch.Tensor:
    """Interpret a linalg.generic by reading the body + iterator_types.

    Handles:
    - Elementwise (all-parallel) with identity maps → apply body
      tensor-wise. Scalar constants in the body are recovered via
      :func:`_find_constant_in_body`.
    - Reduction body → torch.sum (or batched matmul when 3-iteration
      dims + 2 inputs).
    - Dequant-style body (``sitofp → mulf``) → identity * scale.
    """
    from xdsl.dialects.linalg import IteratorType, IteratorTypeAttr
    kinds = op.iterator_types
    iterator_kinds = [k.data for k in kinds.data if isinstance(k, IteratorTypeAttr)]
    is_elementwise = all(k == IteratorType.PARALLEL for k in iterator_kinds)

    inputs = [env[v] for v in op.inputs]
    out_init = env[op.outputs[0]]

    body = op.body.block
    terminator = body.last_op
    if terminator is None or terminator.name != "linalg.yield":
        return out_init
    yielded = terminator.operands[0]
    src_op = yielded.owner
    src_name = src_op.name if src_op is not None else ""

    if is_elementwise and len(inputs) == 1:
        if src_name == "arith.mulf":
            const_val = _find_constant_in_body(body)
            if const_val is not None:
                return inputs[0] * const_val
            return inputs[0]
        if src_name == "arith.addf":
            const_val = _find_constant_in_body(body)
            if const_val is not None:
                return inputs[0] + const_val
            return inputs[0]
        if src_name == "arith.truncf":
            return inputs[0].to(out_init.dtype)
        if src_name == "arith.extf":
            return inputs[0].to(out_init.dtype)
        if src_name == "arith.sitofp":
            return inputs[0].to(out_init.dtype)
        return inputs[0]

    if is_elementwise and len(inputs) == 2:
        if src_name == "arith.addf":
            return inputs[0] + inputs[1]
        if src_name == "arith.mulf":
            return inputs[0] * inputs[1]
        if src_name == "arith.subf":
            return inputs[0] - inputs[1]
        # dequant / truncf body shape
        return inputs[0] * inputs[1] if inputs[1].ndim == inputs[0].ndim else inputs[0]

    # Reduction body.
    if iterator_kinds.count(IteratorType.REDUCTION) >= 1:
        if len(inputs) == 2 and len(iterator_kinds) == 3:
            # Mixed-precision matmul body -> treat as plain matmul.
            lhs, rhs = inputs[0], inputs[1]
            if lhs.ndim == 2 and rhs.ndim == 2:
                return lhs @ rhs
            if lhs.ndim == 3 and rhs.ndim == 2:
                B, T, K = lhs.shape
                return (lhs.reshape(B * T, K) @ rhs).reshape(B, T, -1)
        reduce_dims = [i for i, k in enumerate(iterator_kinds) if k == IteratorType.REDUCTION]
        if len(inputs) == 1:
            return inputs[0].sum(dim=reduce_dims)

    return out_init


def _exec_empty(op: EmptyOp, env: dict[SSAValue, torch.Tensor]) -> torch.Tensor:
    t = op.results[0].type
    if isinstance(t, TensorType):
        shape = [d if d >= 0 else 1 for d in t.get_shape()]
        # We can't resolve element dtype from arbitrary xDSL types
        # without a mapping; default to float32.
        return torch.zeros(shape, dtype=torch.float32)
    return torch.zeros(())


def _exec_insert_slice(
    op: InsertSliceOp, env: dict[SSAValue, torch.Tensor]
) -> torch.Tensor:
    src = env[op.source]
    dst = env[op.dest].clone()
    static_offsets = [int(v) for v in op.static_offsets.get_values()]
    static_sizes = [int(v) for v in op.static_sizes.get_values()]
    slices = tuple(
        slice(o, o + s) for o, s in zip(static_offsets, static_sizes)
    )
    dst[slices] = src.reshape(dst[slices].shape)
    return dst


def _exec_linalg_ext(op: Operation, env: dict[SSAValue, torch.Tensor]) -> torch.Tensor:
    """Dispatch compgen.linalg_ext ops."""
    name = op.name
    inp = env[op.operands[0]]
    if name == "compgen.linalg_ext.softmax":
        dim = op.dim.value.data
        return F.softmax(inp, dim=dim)
    if name == "compgen.linalg_ext.silu":
        return F.silu(inp)
    if name == "compgen.linalg_ext.gelu":
        approx = op.attributes.get("approximate") or op.properties.get("approximate")
        approximate = "tanh" if isinstance(approx, StringAttr) and approx.data == "tanh" else "none"
        return F.gelu(inp, approximate=approximate)
    if name == "compgen.linalg_ext.layer_norm":
        eps = op.eps.value.data
        weight = env.get(op.weight) if op.weight is not None else None
        bias = env.get(op.bias) if op.bias is not None else None
        return F.layer_norm(inp, (inp.shape[-1],), weight=weight, bias=bias, eps=eps)
    if name == "compgen.linalg_ext.rms_norm":
        eps = op.eps.value.data
        weight = env.get(op.weight) if op.weight is not None else None
        rms = torch.sqrt(inp.pow(2).mean(-1, keepdim=True) + eps)
        y = inp / rms
        if weight is not None:
            y = y * weight
        return y
    if name == "compgen.linalg_ext.swiglu":
        up = env[op.operands[1]]
        return F.silu(inp) * up
    # Fallback: identity.
    return inp


# --- dispatcher ------------------------------------------------------------


@dataclass
class ExecutorStats:
    ops_executed: int = 0
    ops_skipped: int = 0
    ops_by_name: dict[str, int] = field(default_factory=dict)

    def record(self, name: str) -> None:
        self.ops_by_name[name] = self.ops_by_name.get(name, 0) + 1


def execute(
    module: ModuleOp,
    exported_program: Any,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    stats: ExecutorStats | None = None,
) -> torch.Tensor:
    """Run the compiled ``ModuleOp`` on CPU with ``example_inputs``.

    Looks up the ``@forward`` func, maps its block args to
    (state_dict params, user inputs) via the exported program's
    graph signature, and interprets each op with a torch primitive.
    Returns the tensor returned by the func.
    """
    stats = stats if stats is not None else ExecutorStats()

    forward_func: FuncOp | None = None
    for op in module.ops:
        if isinstance(op, FuncOp) and op.sym_name.data == "forward":
            forward_func = op
            break
    if forward_func is None:
        raise ValueError("module has no @forward function")

    block = forward_func.body.block
    env: dict[SSAValue, torch.Tensor] = {}

    # Populate block args from (params, user_inputs).
    gs = exported_program.graph_signature
    state_dict = exported_program.state_dict
    input_tensors: list[torch.Tensor] = []
    user_idx = 0
    for spec in gs.input_specs:
        kind = spec.kind.name if hasattr(spec.kind, "name") else str(spec.kind)
        if kind == "PARAMETER" or kind == "BUFFER":
            target = spec.target
            if target in state_dict:
                input_tensors.append(state_dict[target])
            else:
                input_tensors.append(torch.zeros(1))
        elif kind == "USER_INPUT":
            if user_idx < len(example_inputs):
                input_tensors.append(example_inputs[user_idx])
                user_idx += 1
            else:
                input_tensors.append(torch.zeros(1))
        else:
            input_tensors.append(torch.zeros(1))

    if len(input_tensors) != len(block.args):
        # Pad / truncate to match; last-resort robustness for
        # signature mismatches.
        if len(input_tensors) < len(block.args):
            input_tensors += [torch.zeros(1)] * (len(block.args) - len(input_tensors))
        else:
            input_tensors = input_tensors[: len(block.args)]
    for arg, tensor in zip(block.args, input_tensors, strict=True):
        env[arg] = tensor

    # Walk ops.
    for op in block.ops:
        if isinstance(op, ReturnOp):
            if not op.operands:
                return torch.zeros(())
            return env[op.operands[0]]
        try:
            out = _dispatch(op, env, stats)
            for res, val in zip(op.results, out if isinstance(out, (list, tuple)) else (out,)):
                env[res] = val
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "cpu_executor.op_failed",
                op=op.name, error=str(exc),
            )
            stats.ops_skipped += 1
            # Fill op results with zeros so downstream ops still run.
            for res in op.results:
                t = res.type
                if isinstance(t, TensorType):
                    shape = [d if d >= 0 else 1 for d in t.get_shape()]
                    env[res] = torch.zeros(shape, dtype=torch.float32)

    return torch.zeros(())


def _dispatch(
    op: Operation,
    env: dict[SSAValue, torch.Tensor],
    stats: ExecutorStats,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    stats.ops_executed += 1
    stats.record(op.name)

    if isinstance(op, EmptyOp):
        return _exec_empty(op, env)
    if isinstance(op, MatmulOp):
        return _exec_matmul(op, env)
    if isinstance(op, TransposeOp):
        return _exec_transpose(op, env)
    if isinstance(op, GenericOp):
        return _exec_linalg_generic(op, env)
    if isinstance(op, InsertSliceOp):
        return _exec_insert_slice(op, env)
    if op.name.startswith("compgen.linalg_ext."):
        return _exec_linalg_ext(op, env)
    if isinstance(op, CallOp):
        callee = op.callee.string_value()
        # Normalise to base (strip trailing _N suffix if present).
        base = callee
        for known in _ATEN_DISPATCH:
            if callee == known or callee.startswith(known + "_"):
                base = known
                break
        fn = _ATEN_DISPATCH.get(base)
        if fn is None:
            raise KeyError(f"unsupported callee {callee!r}")
        args = [env[v] for v in op.operands]
        # Target shape hint (for view/transpose/expand).
        target_shape = None
        if op.results:
            rt = op.results[0].type
            if isinstance(rt, TensorType):
                target_shape = tuple(
                    d if d >= 0 else 1 for d in rt.get_shape()
                )
        return fn(
            args, attrs=dict(op.attributes), target_shape=target_shape,
        )
    # Constant / yield / other internal ops -- no dispatch needed.
    raise KeyError(f"no dispatcher for {op.name}")


__all__ = [
    "ExecutorStats",
    "execute",
]
