"""Agent IR serialization helpers."""

from __future__ import annotations

import io
from typing import Any

import yaml
from xdsl.context import Context
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.printer import Printer

from compgen.ir.agent.attrs import (
    ConfidenceAttr,
    CreativityPolicyAttr,
    EvaluatorKindAttr,
    FreshnessAttr,
    SearchBudgetAttr,
)
from compgen.ir.agent.dialect import Agent


def agent_to_mlir(module: ModuleOp) -> str:
    """Print Agent IR to canonical MLIR text."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def mlir_to_agent(mlir_text: str) -> ModuleOp:
    """Parse MLIR text into an Agent IR module."""
    ctx = Context()
    ctx.register_dialect("agent", lambda: Agent)
    ctx.register_dialect("builtin", lambda: builtin_dialect.Builtin)
    return Parser(ctx, mlir_text).parse_module()


def agent_module_to_yaml(module: ModuleOp) -> str:
    """Serialize top-level Agent IR ops into deterministic YAML."""
    entries = [_op_to_dict(op) for op in module.body.block.ops]
    return yaml.dump(entries, default_flow_style=False, sort_keys=True)


def _attr_to_python(attr: object) -> Any:
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return attr.value.data
    if isinstance(attr, SymbolRefAttr):
        return f"@{attr.root_reference.data}"
    if isinstance(attr, ArrayAttr):
        return [_attr_to_python(item) for item in attr.data]
    if isinstance(attr, ConfidenceAttr):
        return {"value_milli": attr.value_milli.value.data}
    if isinstance(attr, FreshnessAttr):
        return {"epoch": attr.epoch.value.data, "state": attr.state.data}
    if isinstance(attr, SearchBudgetAttr):
        return {
            "max_candidates": attr.max_candidates.value.data,
            "max_iterations": attr.max_iterations.value.data,
            "timeout_ms": attr.timeout_ms.value.data,
        }
    if isinstance(attr, CreativityPolicyAttr):
        return {
            "mode": attr.mode.data,
            "temperature_milli": attr.temperature_milli.value.data,
        }
    if isinstance(attr, EvaluatorKindAttr):
        return {"kind": attr.kind.data}
    return str(attr)


def _op_to_dict(op: Operation) -> dict[str, Any]:
    data: dict[str, Any] = {"_op": op.name}
    for prop_name, prop_val in op.properties.items():
        data[prop_name] = _attr_to_python(prop_val)
    return data


__all__ = [
    "agent_module_to_yaml",
    "agent_to_mlir",
    "mlir_to_agent",
]
