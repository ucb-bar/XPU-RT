"""Tests for PayloadPass base class + auto-registration."""

from __future__ import annotations

from typing import Any, ClassVar

from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import Registry
from xdsl.dialects.builtin import ModuleOp


class _ToyPass(PayloadPass):
    name: ClassVar[str] = "toy_pass"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "TEST:Toy"
    description: ClassVar[str] = "unit-test toy"
    stub: ClassVar[bool] = False

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        # Record the call by tagging the module
        from xdsl.dialects.builtin import IntegerAttr, i64

        prev = module.attributes.get("toy.count")
        prev_val = int(getattr(prev, "value").data) if prev is not None else 0
        module.attributes["toy.count"] = IntegerAttr(prev_val + 1, i64)
        return module


def test_register_builds_real_tool() -> None:
    r = Registry()
    p = _ToyPass()
    p.register(r)
    tool = r.lookup_tool("toy_pass", phase=2)
    assert tool is not None
    assert tool.wraps_pass == "TEST:Toy"
    assert tool.is_stub is False


def test_register_is_idempotent() -> None:
    r = Registry()
    p = _ToyPass()
    p.register(r)
    p.register(r)  # should not raise
    assert r.counts()[2]["tools"] == 1


def test_invoke_with_module_returns_ok() -> None:
    r = Registry()
    _ToyPass().register(r)
    tool = r.lookup_tool("toy_pass", phase=2)
    assert tool is not None

    mod = ModuleOp([])
    result = tool.invoke(module=mod)
    assert result["status"] == "ok"
    assert result["pass_name"] == "toy_pass"
    assert "toy.count" in mod.attributes


def test_invoke_without_module_errors() -> None:
    r = Registry()
    _ToyPass().register(r)
    tool = r.lookup_tool("toy_pass", phase=2)
    result = tool.invoke()
    assert result["status"] == "error"


def test_invoke_with_raising_run_records_error() -> None:
    class _Boom(PayloadPass):
        name: ClassVar[str] = "boom_pass"
        phase: ClassVar[int] = 2
        wraps_pass: ClassVar[str] = "TEST:Boom"
        stub: ClassVar[bool] = False

        def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
            raise RuntimeError("boom")

    r = Registry()
    _Boom().register(r)
    tool = r.lookup_tool("boom_pass", phase=2)
    result = tool.invoke(module=ModuleOp([]))
    assert result["status"] == "error"
    assert "boom" in result["reason"]
