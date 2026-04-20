"""Payload IR → ANSI C codegen for the baremetal Hexagon path.

Walks an xDSL payload ModuleOp and emits one C source file per
``func.func`` in it. Output is plain ANSI C that compiles with
``gcc -fsyntax-only`` — the per-op intrinsic lowering is delegated to
``npu_driver.h`` symbols (``npu_call_aten_*`` / ``npu_matmul`` /
``npu_transpose_2d`` / ``npu_batch_matmul``) which the
:class:`~compgen.runtime.baremetal.emitter.BaremetalEmitter` scaffold
produces with conservative ``memcpy``-based stubs the host can run.

Op coverage (every op the captured Gemma decoder produces after
``fx_to_xdsl``):

  Structural
    * ``func.func``       — declaration (extern proto) AND definition
    * ``func.call``       — typed call to ``npu_call_<func>``
    * ``func.return``     — return statement
    * ``arith.constant``  — host-side literal
    * ``tensor.empty``    — stack-allocated buffer of the right size

  Linalg
    * ``linalg.matmul``         → ``npu_matmul(...)``
    * ``linalg.batch_matmul``   → ``npu_batch_matmul(...)``
    * ``linalg.transpose``      → ``npu_transpose_2d(...)``
    * ``linalg.fill``           → ``npu_fill(...)``
    * ``linalg.softmax``        → ``npu_softmax_lastdim(...)``
    * ``linalg.generic``        → emitted as a comment + ``npu_call_generic(...)``

  Arith / math (used by softmax max-stabilisation, RMSNorm decomp)
    * ``arith.addf/subf/mulf/divf/negf/maximumf`` → scalar inline
    * ``arith.cmpf``            → scalar inline
    * ``math.exp/sqrt/rsqrt/tanh`` → ``npu_<fn>(...)``

  Tensor primitives
    * ``tensor.extract_slice / insert_slice / expand_shape / collapse_shape``
      → emitted as ``npu_view_*`` calls (slice metadata in args)

The recipe IR's ``recipe.tile`` / ``recipe.fuse`` decisions surface
through the agent's ``apply_recipe`` mutation of the payload module
*before* this codegen runs — so a different proposal really does
produce different C bytes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Operation, SSAValue

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _c_scalar_for(elem_type: Any) -> str:
    """ANSI C scalar type for a payload element type."""
    if isinstance(elem_type, Float16Type):
        return "uint16_t"   # half stored as raw bits; npu helpers convert
    if isinstance(elem_type, Float32Type):
        return "float"
    if isinstance(elem_type, Float64Type):
        return "double"
    if isinstance(elem_type, IntegerType):
        bits = elem_type.width.data
        if bits == 1:
            return "uint8_t"
        if bits <= 8:
            return "int8_t"
        if bits <= 16:
            return "int16_t"
        if bits <= 32:
            return "int32_t"
        return "int64_t"
    return "float"   # safe default


def _shape_dims(t: TensorType) -> tuple[int, ...]:
    return tuple(int(d) for d in t.get_shape())


def _shape_str(t: TensorType) -> str:
    return "x".join(str(d) for d in _shape_dims(t))


def _elem_count(t: TensorType) -> int:
    n = 1
    for d in _shape_dims(t):
        n *= d if d > 0 else 1
    return n


def _bytes_per_elem(elem_type: Any) -> int:
    if isinstance(elem_type, Float16Type):
        return 2
    if isinstance(elem_type, Float32Type):
        return 4
    if isinstance(elem_type, Float64Type):
        return 8
    if isinstance(elem_type, IntegerType):
        bits = elem_type.width.data
        return max(1, (bits + 7) // 8)
    return 4


# ---------------------------------------------------------------------------
# SSA name plumbing
# ---------------------------------------------------------------------------


@dataclass
class _NameTable:
    """Maps xDSL SSAValues to C identifier names within one function."""

    counter: int = 0
    by_value: dict[int, str] = field(default_factory=dict)

    def name_for(self, value: SSAValue, *, prefix: str = "v") -> str:
        key = id(value)
        if key in self.by_value:
            return self.by_value[key]
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        self.by_value[key] = name
        return name

    def assign(self, value: SSAValue, name: str) -> None:
        self.by_value[id(value)] = name


def _sanitize_callee(name: str) -> str:
    """Make a function symbol safe for C: replace non-identifier chars."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    if out and out[0].isdigit():
        out.insert(0, "_")
    return "".join(out) or "fn"


# ---------------------------------------------------------------------------
# Per-op emitters
# ---------------------------------------------------------------------------


def _emit_param_list(func: FuncOp) -> tuple[str, list[tuple[str, TensorType]]]:
    """Render the C parameter list for a func.func + return per-param info."""
    params: list[str] = []
    info: list[tuple[str, TensorType]] = []
    block = func.body.blocks[0] if func.body.blocks else None
    if block is None:
        # Declaration-only signature pulled from func type.
        for i, in_t in enumerate(func.function_type.inputs):
            assert isinstance(in_t, TensorType)
            scalar = _c_scalar_for(in_t.element_type)
            pname = f"in_{i}"
            params.append(f"const {scalar} *{pname}")
            info.append((pname, in_t))
        return ", ".join(params) if params else "void", info

    for i, arg in enumerate(block.args):
        t = arg.type
        if isinstance(t, TensorType):
            scalar = _c_scalar_for(t.element_type)
            pname = f"in_{i}"
            params.append(f"const {scalar} *{pname}")
            info.append((pname, t))
        else:
            params.append(f"int64_t in_{i}")
    return ", ".join(params) if params else "void", info


def _emit_return_signature(func: FuncOp) -> tuple[str, TensorType | None]:
    """Return (C return-type-string, the TensorType if any)."""
    outs = list(func.function_type.outputs)
    if not outs:
        return "void", None
    out_t = outs[0]
    if isinstance(out_t, TensorType):
        scalar = _c_scalar_for(out_t.element_type)
        return f"{scalar} *", out_t
    return "int64_t", None


def _emit_func_declaration(func: FuncOp) -> str:
    """Emit a C extern prototype for a private func.func."""
    sym = _sanitize_callee(func.sym_name.data)
    params, _ = _emit_param_list(func)
    ret, _ = _emit_return_signature(func)
    return f"extern {ret}npu_call_{sym}({params});\n"


def _emit_op(
    op: Operation, names: _NameTable, lines: list[str],
    *, indent: str = "    ",
) -> None:
    """Emit C source for one op into ``lines``."""

    op_name = op.name

    # ---- func.return ------------------------------------------------------
    if isinstance(op, ReturnOp):
        if op.operands:
            n = names.name_for(op.operands[0])
            lines.append(f"{indent}return {n};")
        else:
            lines.append(f"{indent}return;")
        return

    # ---- func.call --------------------------------------------------------
    if isinstance(op, CallOp):
        callee = _sanitize_callee(op.callee.root_reference.data)
        args = ", ".join(names.name_for(a) for a in op.operands) or ""
        trail = _agent_trail(op)
        if op.results:
            res = op.results[0]
            res_name = names.name_for(res, prefix="t")
            scalar = _c_scalar_for(res.type.element_type) \
                if isinstance(res.type, TensorType) else "int64_t"
            shape = _shape_str(res.type) if isinstance(res.type, TensorType) else ""
            bits = [b for b in (shape, trail) if b]
            comment = f"  /* {' | '.join(bits)} */" if bits else ""
            lines.append(
                f"{indent}{scalar} *{res_name} = "
                f"npu_call_{callee}({args});{comment}"
            )
        else:
            comment = f"  /* {trail} */" if trail else ""
            lines.append(f"{indent}npu_call_{callee}({args});{comment}")
        return

    # ---- tensor.empty -----------------------------------------------------
    if op_name == "tensor.empty":
        if op.results:
            res = op.results[0]
            res_name = names.name_for(res, prefix="buf")
            t = res.type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            n = _elem_count(t) if isinstance(t, TensorType) else 1
            shape = _shape_str(t) if isinstance(t, TensorType) else ""
            lines.append(
                f"{indent}{scalar} *{res_name} = ({scalar} *)"
                f"npu_alloc({n} * sizeof({scalar}));  /* tensor.empty {shape} */"
            )
        return

    # ---- linalg.matmul / batch_matmul ------------------------------------
    if op_name == "linalg.matmul":
        # ins: [a, b], outs: [c]; C is reused as the result (in-place semantics).
        ins = list(op.operands)
        outs = list(op.results)
        if len(ins) >= 3 and outs:
            a, b, c_init = ins[0], ins[1], ins[2]
            res_name = names.name_for(outs[0], prefix="t")
            a_n = names.name_for(a)
            b_n = names.name_for(b)
            c_n = names.name_for(c_init)
            # Pull M/N/K from the result shape + b shape.
            res_t = outs[0].type
            assert isinstance(res_t, TensorType)
            M, N = _shape_dims(res_t)
            b_t = b.type
            assert isinstance(b_t, TensorType)
            _, K = _shape_dims(b_t)[::-1] if len(_shape_dims(b_t)) == 2 else (0, _shape_dims(b_t)[0])
            K = _shape_dims(a.type)[-1] if isinstance(a.type, TensorType) else 0
            scalar = _c_scalar_for(res_t.element_type)
            trail = _agent_trail(op)
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_matmul("
                f"{a_n}, {b_n}, {c_n}, /*M=*/{M}, /*N=*/{N}, /*K=*/{K}"
                f");  /* {trail or 'matmul'} */"
            )
        return

    if op_name == "linalg.batch_matmul":
        ins = list(op.operands)
        outs = list(op.results)
        if len(ins) >= 3 and outs:
            a, b, c_init = ins[0], ins[1], ins[2]
            res_t = outs[0].type
            assert isinstance(res_t, TensorType)
            B, M, N = _shape_dims(res_t)
            K = _shape_dims(a.type)[-1] if isinstance(a.type, TensorType) else 0
            scalar = _c_scalar_for(res_t.element_type)
            res_name = names.name_for(outs[0], prefix="t")
            trail = _agent_trail(op)
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_batch_matmul("
                f"{names.name_for(a)}, {names.name_for(b)}, "
                f"{names.name_for(c_init)}, /*B=*/{B}, /*M=*/{M}, /*N=*/{N}, /*K=*/{K}"
                f");  /* {trail or 'batch_matmul'} */"
            )
        return

    # ---- linalg.transpose ------------------------------------------------
    if op_name == "linalg.transpose":
        ins = list(op.operands)
        outs = list(op.results)
        if len(ins) >= 2 and outs:
            a, c_init = ins[0], ins[1]
            t = outs[0].type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            res_name = names.name_for(outs[0], prefix="t")
            dims = _shape_dims(a.type) if isinstance(a.type, TensorType) else (0,)
            shape_args = ", ".join(str(d) for d in dims)
            perm = op.attributes.get("permutation")
            perm_str = _array_attr_to_c(perm)
            n = len(dims)
            lines.append(
                f"{indent}static const int64_t {res_name}_perm[{n}] = {perm_str};"
            )
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_transpose("
                f"{names.name_for(a)}, {names.name_for(c_init)}, "
                f"(int64_t[]){{ {shape_args} }}, /*ndim=*/{n}, "
                f"{res_name}_perm"
                f");"
            )
        return

    # ---- linalg.fill -----------------------------------------------------
    if op_name == "linalg.fill":
        ins = list(op.operands)
        outs = list(op.results)
        if ins and outs:
            t = outs[0].type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            res_name = names.name_for(outs[0], prefix="t")
            n = _elem_count(t) if isinstance(t, TensorType) else 0
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_fill("
                f"{names.name_for(ins[-1])}, {names.name_for(ins[0])}, "
                f"/*n_elems=*/{n}"
                f");"
            )
        return

    # ---- linalg.softmax --------------------------------------------------
    if op_name == "linalg.softmax":
        ins = list(op.operands)
        outs = list(op.results)
        if ins and outs:
            t = outs[0].type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            res_name = names.name_for(outs[0], prefix="t")
            inner = _shape_dims(t)[-1] if isinstance(t, TensorType) else 0
            outer = _elem_count(t) // max(inner, 1) if isinstance(t, TensorType) else 0
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_softmax_lastdim("
                f"{names.name_for(ins[0])}, "
                f"{names.name_for(ins[-1])}, "
                f"/*outer=*/{outer}, /*inner=*/{inner}"
                f");"
            )
        return

    # ---- linalg.generic --------------------------------------------------
    if op_name == "linalg.generic":
        outs = list(op.results)
        if outs:
            t = outs[0].type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            res_name = names.name_for(outs[0], prefix="t")
            in_args = ", ".join(names.name_for(o) for o in op.operands)
            n = _elem_count(t) if isinstance(t, TensorType) else 0
            lines.append(
                f"{indent}/* linalg.generic — body opaque; deferred to npu_call_generic */"
            )
            lines.append(
                f"{indent}{scalar} *{res_name} = npu_call_generic("
                f"{in_args}, /*n_elems=*/{n}"
                f");"
            )
        return

    # ---- arith / math scalar ops ----------------------------------------
    scalar_op = _SCALAR_OPS.get(op_name)
    if scalar_op is not None:
        if op.results:
            res = op.results[0]
            res_name = names.name_for(res, prefix="s")
            scalar = _c_scalar_for(res.type)
            args = [names.name_for(o) for o in op.operands]
            expr = scalar_op(args)
            lines.append(f"{indent}{scalar} {res_name} = {expr};")
        return

    # ---- arith.constant --------------------------------------------------
    if op_name == "arith.constant":
        if op.results:
            res = op.results[0]
            res_name = names.name_for(res, prefix="c")
            scalar = _c_scalar_for(res.type)
            value = op.attributes.get("value")
            literal = _const_to_c(value, scalar)
            lines.append(f"{indent}{scalar} {res_name} = {literal};")
        return

    # ---- tensor view-like ops -------------------------------------------
    view_call = _TENSOR_VIEW_OPS.get(op_name)
    if view_call is not None:
        if op.results:
            res = op.results[0]
            res_name = names.name_for(res, prefix="v")
            t = res.type
            scalar = _c_scalar_for(t.element_type) if isinstance(t, TensorType) else "float"
            in_args = ", ".join(names.name_for(o) for o in op.operands)
            shape = _shape_str(t) if isinstance(t, TensorType) else ""
            n = _elem_count(t) if isinstance(t, TensorType) else 0
            lines.append(
                f"{indent}{scalar} *{res_name} = {view_call}("
                f"{in_args}, /*n_elems=*/{n});  /* {shape} */"
            )
        return

    # ---- catchall --------------------------------------------------------
    # We refuse to silently skip — emit a compile-time hint (still valid C
    # via /* ... */) so the codegen is honest about what it didn't render.
    lines.append(
        f"{indent}/* unhandled op: {op_name} (results={len(op.results)}) */"
    )


# ---------------------------------------------------------------------------
# Op dispatch tables
# ---------------------------------------------------------------------------


_SCALAR_OPS: dict[str, Any] = {
    "arith.addf":     lambda a: f"{a[0]} + {a[1]}",
    "arith.subf":     lambda a: f"{a[0]} - {a[1]}",
    "arith.mulf":     lambda a: f"{a[0]} * {a[1]}",
    "arith.divf":     lambda a: f"{a[0]} / {a[1]}",
    "arith.negf":     lambda a: f"-({a[0]})",
    "arith.maximumf": lambda a: f"((({a[0]}) > ({a[1]})) ? ({a[0]}) : ({a[1]}))",
    "arith.minimumf": lambda a: f"((({a[0]}) < ({a[1]})) ? ({a[0]}) : ({a[1]}))",
    "arith.cmpf":     lambda a: f"(({a[0]}) == ({a[1]}))",
    "arith.addi":     lambda a: f"{a[0]} + {a[1]}",
    "arith.subi":     lambda a: f"{a[0]} - {a[1]}",
    "arith.muli":     lambda a: f"{a[0]} * {a[1]}",
    "math.exp":       lambda a: f"npu_expf({a[0]})",
    "math.log":       lambda a: f"npu_logf({a[0]})",
    "math.sqrt":      lambda a: f"npu_sqrtf({a[0]})",
    "math.rsqrt":     lambda a: f"npu_rsqrtf({a[0]})",
    "math.tanh":      lambda a: f"npu_tanhf({a[0]})",
    "math.sin":       lambda a: f"npu_sinf({a[0]})",
    "math.cos":       lambda a: f"npu_cosf({a[0]})",
}


_TENSOR_VIEW_OPS: dict[str, str] = {
    "tensor.extract_slice":  "npu_view_extract_slice",
    "tensor.insert_slice":   "npu_view_insert_slice",
    "tensor.expand_shape":   "npu_view_expand_shape",
    "tensor.collapse_shape": "npu_view_collapse_shape",
    "tensor.cast":           "npu_view_cast",
    "tensor.reshape":        "npu_view_reshape",
}


def _attr_str(attr: Any) -> str:
    if attr is None:
        return ""
    if isinstance(attr, StringAttr):
        return attr.data
    return str(attr)


# Attributes the agent's recipe-driven mutator may have stamped on each
# payload op. Surfacing them in the C comment trail makes the agent's
# decisions visible at the kernel level (different proposals → different
# bytes in kernels/*.c).
_AGENT_DECISION_ATTRS: tuple[str, ...] = (
    "compgen.region_id",
    "compgen._pattern_hint",   # op-family role (matmul/softmax/rmsnorm/...) from import_fx
    "compgen.fused_into",
    "compgen.fusion_kind",
    "compgen.tile_sizes_str",
    "compgen.megakernel",
    "compgen.device",
)


def _agent_trail(op: Operation) -> str:
    """Build a short ``key=value`` trail of compgen.* attrs for inline comments."""
    parts: list[str] = []
    for k in _AGENT_DECISION_ATTRS:
        v = op.attributes.get(k)
        if v is None:
            continue
        text = v.data if isinstance(v, StringAttr) else str(v)
        if not text:
            continue
        # Strip the compgen. prefix in the comment for legibility.
        short = k.removeprefix("compgen.")
        parts.append(f"{short}={text}")
    return " ".join(parts)


def _array_attr_to_c(attr: Any) -> str:
    """Render a DenseArrayBaseAttr / ArrayAttr of ints as a C initialiser."""
    if attr is None:
        return "{ 0 }"
    try:
        # DenseArrayBase exposes .get_values()
        vals = attr.get_values()   # type: ignore[attr-defined]
        return "{ " + ", ".join(str(int(v)) for v in vals) + " }"
    except AttributeError:
        try:
            data = attr.data
            return "{ " + ", ".join(str(int(getattr(v, "value", v).data)) for v in data) + " }"
        except Exception:   # noqa: BLE001
            return "{ 0 }"


def _const_to_c(attr: Any, scalar: str) -> str:
    if attr is None:
        return "0"
    try:
        # IntegerAttr
        if hasattr(attr, "value") and hasattr(attr.value, "data"):
            v = attr.value.data
            if scalar in {"float", "double"}:
                return f"{float(v)}f" if scalar == "float" else str(float(v))
            return str(int(v))
        # FloatAttr
        if hasattr(attr, "value"):
            v = attr.value
            return f"{float(v)}f" if scalar == "float" else str(float(v))
    except Exception:   # noqa: BLE001
        pass
    return "0"


# ---------------------------------------------------------------------------
# Per-function emitters
# ---------------------------------------------------------------------------


def emit_function_definition(func: FuncOp, *, header: str = "") -> str:
    """Emit a complete C function body for ``func``.

    Used for the public ``@forward`` entry. Private declaration-only
    funcs are handled by :func:`emit_function_declaration`.
    """
    sym = _sanitize_callee(func.sym_name.data)
    params, _ = _emit_param_list(func)
    ret_str, ret_t = _emit_return_signature(func)

    block = func.body.blocks[0] if func.body.blocks else None
    body_lines: list[str] = []
    names = _NameTable()

    # Bind block arguments to their parameter names (already named in1..inN
    # at the param-list level; tag them under the same key so subsequent
    # ops see the same identifier).
    if block is not None:
        for i, arg in enumerate(block.args):
            names.assign(arg, f"in_{i}")
        for op in block.ops:
            _emit_op(op, names, body_lines)

    body_lines = [
        "    /* prologue: every npu_* helper is declared in npu_driver.h */",
        *body_lines,
    ]
    body = "\n".join(body_lines) if body_lines else "    /* empty body */"
    leader = f"{header}\n" if header else ""
    return (
        f"{leader}{ret_str}npu_call_{sym}({params}) {{\n"
        f"{body}\n"
        f"}}\n"
    )


def emit_function_declaration(func: FuncOp) -> str:
    """Emit a C extern prototype for a private declaration-only func."""
    return _emit_func_declaration(func)


# ---------------------------------------------------------------------------
# Module-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedCFunction:
    sym_name: str          # original MLIR symbol
    c_name: str            # sanitized C identifier
    is_definition: bool    # True for funcs with bodies
    source: str            # the C source string
    pattern_id: str        # used for execution-order metadata


def emit_module(
    module: ModuleOp, *, file_header: str = "",
) -> list[GeneratedCFunction]:
    """Walk a payload ModuleOp; return one GeneratedCFunction per func.func.

    Order matches the module's body order so the agent's recipe-driven
    op reordering surfaces in the emitted file ordering.
    """
    out: list[GeneratedCFunction] = []
    for op in module.body.block.ops:
        if not isinstance(op, FuncOp):
            continue
        sym = op.sym_name.data
        c_name = _sanitize_callee(sym)
        has_body = bool(op.body.blocks and op.body.blocks[0].ops)
        if has_body:
            source = emit_function_definition(op, header=file_header)
            pattern = "forward" if sym == "forward" else c_name
            out.append(GeneratedCFunction(
                sym_name=sym, c_name=c_name,
                is_definition=True, source=source, pattern_id=pattern,
            ))
        else:
            source = emit_function_declaration(op)
            out.append(GeneratedCFunction(
                sym_name=sym, c_name=c_name,
                is_definition=False, source=source, pattern_id="aten_passthrough",
            ))
    return out


def emit_npu_driver_extension_h(
    funcs: list[GeneratedCFunction], *, model_name: str = "model",
) -> str:
    """Header containing prototypes for every npu_* symbol the kernels call.

    Stitched into the BaremetalEmitter's npu_driver.h via a side
    include — let the BaremetalEmitter generate the dispatch enum,
    we only contribute the per-aten + per-linalg helpers + scalar
    math intrinsics.
    """
    lines = [
        "#ifndef NPU_DRIVER_EXT_H",
        "#define NPU_DRIVER_EXT_H",
        "",
        f"/* CompGen-emitted npu helpers for {model_name}. */",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "/* Memory + dispatch helpers (host-side stubs in npu_driver_ext.c). */",
        "void *npu_alloc(size_t bytes);",
        "void  npu_free(void *p);",
        "",
        "/* Linalg primitives. */",
        "float *npu_matmul(const float *a, const float *b, float *c,",
        "                  int64_t M, int64_t N, int64_t K);",
        "float *npu_batch_matmul(const float *a, const float *b, float *c,",
        "                        int64_t B, int64_t M, int64_t N, int64_t K);",
        "float *npu_transpose(const float *a, float *out,",
        "                     const int64_t *shape, int64_t ndim,",
        "                     const int64_t *perm);",
        "float *npu_fill(float *out, const float *value, int64_t n_elems);",
        "float *npu_softmax_lastdim(const float *a, float *out,",
        "                           int64_t outer, int64_t inner);",
        "float *npu_call_generic(const float *a0, /* extra ins, */",
        "                        int64_t n_elems);",
        "",
        "/* Tensor view helpers (zero-copy where possible). */",
        "float *npu_view_extract_slice(const float *a, int64_t n_elems);",
        "float *npu_view_insert_slice(const float *a, int64_t n_elems);",
        "float *npu_view_expand_shape(const float *a, int64_t n_elems);",
        "float *npu_view_collapse_shape(const float *a, int64_t n_elems);",
        "float *npu_view_cast(const float *a, int64_t n_elems);",
        "float *npu_view_reshape(const float *a, int64_t n_elems);",
        "",
        "/* Scalar math intrinsics. */",
        "float npu_expf(float x);",
        "float npu_logf(float x);",
        "float npu_sqrtf(float x);",
        "float npu_rsqrtf(float x);",
        "float npu_tanhf(float x);",
        "float npu_sinf(float x);",
        "float npu_cosf(float x);",
        "",
        "/* Per-aten passthroughs (one per FX-imported func.func private). */",
    ]
    for f in funcs:
        if f.is_definition:
            continue
        # Already an extern proto — embed verbatim.
        lines.append(f.source.rstrip())
    lines += ["", "#endif /* NPU_DRIVER_EXT_H */", ""]
    return "\n".join(lines)


def emit_npu_driver_extension_c(model_name: str = "model") -> str:
    """Implementation of the helpers in npu_driver_ext.h.

    Conservative ``memcpy`` / ``libm``-style stubs the host can run.
    Real Hexagon HVX intrinsic versions slot in here later.
    """
    return f"""\
/* npu_driver_ext.c — host-runnable stubs for {model_name}.
 * Auto-generated by CompGen. Replace with HVX intrinsic versions
 * for on-device performance.
 */
#include "npu_driver_ext.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void *npu_alloc(size_t bytes) {{ return calloc(1, bytes); }}
void  npu_free(void *p)        {{ free(p); }}

float *npu_matmul(const float *a, const float *b, float *c,
                  int64_t M, int64_t N, int64_t K) {{
    for (int64_t i = 0; i < M; ++i) {{
        for (int64_t j = 0; j < N; ++j) {{
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {{
                acc += a[i * K + k] * b[k * N + j];
            }}
            c[i * N + j] = acc;
        }}
    }}
    return c;
}}

float *npu_batch_matmul(const float *a, const float *b, float *c,
                        int64_t B, int64_t M, int64_t N, int64_t K) {{
    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {{
        npu_matmul(a + b_idx * M * K, b + b_idx * K * N,
                   c + b_idx * M * N, M, N, K);
    }}
    return c;
}}

float *npu_transpose(const float *a, float *out,
                     const int64_t *shape, int64_t ndim,
                     const int64_t *perm) {{
    /* Row-major n-d transpose. Naive — replace with HVX gather. */
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) total *= shape[d];
    int64_t out_shape[8] = {{0}};
    for (int64_t d = 0; d < ndim; ++d) out_shape[d] = shape[perm[d]];
    int64_t in_idx[8] = {{0}};
    for (int64_t lin = 0; lin < total; ++lin) {{
        int64_t rem = lin;
        for (int64_t d = ndim - 1; d >= 0; --d) {{
            in_idx[perm[d]] = rem % out_shape[d];
            rem /= out_shape[d];
        }}
        int64_t in_lin = 0;
        for (int64_t d = 0; d < ndim; ++d) {{
            in_lin = in_lin * shape[d] + in_idx[d];
        }}
        out[lin] = a[in_lin];
    }}
    return out;
}}

float *npu_fill(float *out, const float *value, int64_t n_elems) {{
    float v = *value;
    for (int64_t i = 0; i < n_elems; ++i) out[i] = v;
    return out;
}}

float *npu_softmax_lastdim(const float *a, float *out,
                           int64_t outer, int64_t inner) {{
    for (int64_t r = 0; r < outer; ++r) {{
        const float *row = a + r * inner;
        float *o = out + r * inner;
        float m = row[0];
        for (int64_t i = 1; i < inner; ++i) if (row[i] > m) m = row[i];
        float s = 0.0f;
        for (int64_t i = 0; i < inner; ++i) {{
            o[i] = expf(row[i] - m);
            s += o[i];
        }}
        for (int64_t i = 0; i < inner; ++i) o[i] /= s;
    }}
    return out;
}}

float *npu_call_generic(const float *a0, int64_t n_elems) {{
    /* Identity stub — real linalg.generic body would be lowered here. */
    (void)n_elems;
    return (float *)a0;
}}

float *npu_view_extract_slice(const float *a, int64_t n_elems)  {{ (void)n_elems; return (float*)a; }}
float *npu_view_insert_slice(const float *a, int64_t n_elems)   {{ (void)n_elems; return (float*)a; }}
float *npu_view_expand_shape(const float *a, int64_t n_elems)   {{ (void)n_elems; return (float*)a; }}
float *npu_view_collapse_shape(const float *a, int64_t n_elems) {{ (void)n_elems; return (float*)a; }}
float *npu_view_cast(const float *a, int64_t n_elems)           {{ (void)n_elems; return (float*)a; }}
float *npu_view_reshape(const float *a, int64_t n_elems)        {{ (void)n_elems; return (float*)a; }}

float npu_expf(float x)  {{ return expf(x); }}
float npu_logf(float x)  {{ return logf(x); }}
float npu_sqrtf(float x) {{ return sqrtf(x); }}
float npu_rsqrtf(float x){{ return 1.0f / sqrtf(x); }}
float npu_tanhf(float x) {{ return tanhf(x); }}
float npu_sinf(float x)  {{ return sinf(x); }}
float npu_cosf(float x)  {{ return cosf(x); }}
"""


__all__ = [
    "GeneratedCFunction",
    "emit_function_declaration",
    "emit_function_definition",
    "emit_module",
    "emit_npu_driver_extension_c",
    "emit_npu_driver_extension_h",
]
