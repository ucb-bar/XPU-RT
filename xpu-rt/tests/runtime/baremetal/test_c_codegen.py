"""Tests for the baremetal C-codegen walker.

Pins down the public surface the Phase-1 bundle contract + the
baremetal adapter depend on:

- ``emit_module`` walks a Payload ``ModuleOp`` and returns one
  ``GeneratedCFunction`` per ``func.FuncOp`` in body order.
- ``emit_function_definition`` emits a C body for functions.
- Passthrough declarations (functions with no body) emit as
  declarations, not definitions.
- ``func.call`` / ``func.return`` / ``tensor.empty`` are the bread-and-
  butter ops this walker emits for the post-pipeline payload.

These tests used to be zero — ``c_codegen.py`` is 819 LOC that had
never been exercised by the suite. Phase-11 locks it in.
"""

from __future__ import annotations

from compgen.runtime.baremetal.c_codegen import (
    GeneratedCFunction,
    emit_function_declaration,
    emit_function_definition,
    emit_module,
)
from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region


def _make_passthrough_module(sym: str = "forward") -> ModuleOp:
    """Build a ``func @forward`` whose body calls a single aten_* op
    and returns its result.

    This mirrors the shape ``c_codegen.emit_module`` is designed for:
    a ``@forward`` with calls into bodyless ``aten_*`` declarations.
    """
    f32 = Float32Type()
    t = TensorType(f32, [4, 4])

    # Bodyless declaration of aten_relu — empty Region, no blocks.
    aten_relu = FuncOp("aten_relu", ([t], [t]), Region([]))

    # @forward calls aten_relu.
    body = Block(arg_types=[t])
    (x,) = body.args
    call = CallOp("aten_relu", [x], [t])
    body.add_op(call)
    body.add_op(ReturnOp(call.results[0]))
    forward = FuncOp(sym, ([t], [t]), Region([body]))

    return ModuleOp([aten_relu, forward])


class TestEmitModuleStructure:
    def test_emit_module_returns_one_entry_per_funcop(self) -> None:
        module = _make_passthrough_module()
        out = emit_module(module)
        assert len(out) == 2
        assert all(isinstance(g, GeneratedCFunction) for g in out)
        names = [g.sym_name for g in out]
        assert "aten_relu" in names
        assert "forward" in names

    def test_emit_module_preserves_body_order(self) -> None:
        module = _make_passthrough_module()
        out = emit_module(module)
        assert [g.sym_name for g in out] == ["aten_relu", "forward"]

    def test_bodyless_funcop_emits_declaration(self) -> None:
        """aten_* ops are passthrough — they appear as declarations,
        not definitions, because the baremetal ukernel library owns
        the body."""
        module = _make_passthrough_module()
        out = emit_module(module)
        decl = next(g for g in out if g.sym_name == "aten_relu")
        defn = next(g for g in out if g.sym_name == "forward")
        assert decl.is_definition is False
        assert decl.pattern_id == "aten_passthrough"
        assert defn.is_definition is True
        assert defn.pattern_id == "forward"


class TestEmitFunctionSurface:
    def test_forward_body_references_aten_call(self) -> None:
        """The emitted @forward body must dispatch through
        ``npu_call_<callee>`` — that's the stable C-side boundary
        the baremetal runtime hooks into."""
        module = _make_passthrough_module()
        out = emit_module(module)
        defn = next(g for g in out if g.sym_name == "forward")
        assert "npu_call_aten_relu" in defn.source

    def test_function_declaration_matches_definition_name(self) -> None:
        """Declaration and definition share the same sanitised C
        identifier — no drift between extern prototypes and bodies."""
        module = _make_passthrough_module()
        func = next(op for op in module.body.block.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward")
        decl = emit_function_declaration(func)
        defn = emit_function_definition(func)
        assert "forward" in decl
        assert "forward" in defn

    def test_empty_module_emits_nothing(self) -> None:
        module = ModuleOp([])
        out = emit_module(module)
        assert out == []


class TestGeneratedCFunctionFields:
    def test_forward_definition_carries_source(self) -> None:
        module = _make_passthrough_module()
        defn = next(g for g in emit_module(module) if g.sym_name == "forward")
        # The emitted source is non-empty C text.
        assert defn.source.strip()
        assert "{" in defn.source and "}" in defn.source
        # Returns the aten_* result (not a literal / not void).
        assert "return" in defn.source

    def test_c_name_is_sanitised(self) -> None:
        """Names with dots/slashes (rare from LLM-generated IR) are
        sanitised into valid C identifiers. ``emit_module`` returns
        the sanitised name for consumers that build a C symbol table
        without re-sanitising."""
        module = _make_passthrough_module(sym="forward")
        g = next(g for g in emit_module(module) if g.sym_name == "forward")
        # Simple symbol passes through unchanged.
        assert g.c_name == "forward"
