"""FX graph to xDSL/MLIR conversion.

Converts PyTorch FX graphs (from torch.export) into CompGen's canonical
Payload IR using real xDSL linalg/arith/tensor ops where decompositions
exist, and opaque func.call for ops without known decompositions.

Invariants:
    - Every FX node maps to at least one xDSL op (or a diagnostic).
    - Decomposed ops get ``compgen.region_id`` attributes for Recipe IR targeting.
    - Unsupported ops fall back to ``func.call`` (flagged as opaque).
    - The output module passes the xDSL verifier.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import torch
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.printer import Printer

from compgen.ir.payload.decompositions import DECOMPOSITION_TABLE, reset_region_counters


def _torch_dtype_to_xdsl(dtype: torch.dtype) -> Attribute:
    """Convert a torch dtype to an xDSL element type."""
    mapping = {
        torch.float32: Float32Type,
        torch.float64: Float64Type,
        torch.float16: Float16Type,
    }
    factory = mapping.get(dtype, Float32Type)
    return factory()  # type: ignore[abstract]


def _tensor_type_from_meta(val: Any) -> TensorType | None:
    """Extract a TensorType from an FX node's meta['val']."""
    if val is None:
        return None
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        elem = _torch_dtype_to_xdsl(val.dtype)
        shape = list(val.shape)
        return TensorType(elem, shape)
    return None


@dataclass
class ImportDiagnostic:
    """Diagnostic from an import operation.

    Attributes:
        fx_node: Name of the FX node that produced this diagnostic.
        level: "error", "warning", or "info".
        message: Human-readable description.
    """

    fx_node: str
    level: str
    message: str


@dataclass
class FXImporter:
    """Converts a PyTorch FX graph to an xDSL module.

    Uses the decomposition table from ``decompositions.py`` to produce
    real xDSL ops (linalg.matmul, linalg.transpose, etc.) where possible.
    Falls back to opaque func.call for undecomposed ops.
    """

    diagnostics: list[ImportDiagnostic] = field(default_factory=list)
    decomposed_count: int = 0
    opaque_count: int = 0

    @property
    def decomposition_coverage(self) -> float:
        """Fraction of ops that were decomposed to real xDSL ops."""
        total = self.decomposed_count + self.opaque_count
        return self.decomposed_count / total if total > 0 else 1.0

    def import_graph(self, exported_program: Any) -> ModuleOp:
        """Convert an ExportedProgram's FX graph to an xDSL module."""
        reset_region_counters()
        graph = exported_program.graph
        nodes = list(graph.nodes)

        placeholders = [n for n in nodes if n.op == "placeholder"]
        call_nodes = [n for n in nodes if n.op == "call_function"]
        output_nodes = [n for n in nodes if n.op == "output"]

        # Build xDSL types for each node from meta
        node_types: dict[str, TensorType] = {}
        for node in nodes:
            val = node.meta.get("val")
            tt = _tensor_type_from_meta(val)
            if tt is not None:
                node_types[node.name] = tt

        # All placeholders become func args
        arg_types: list[Attribute] = []
        for p in placeholders:
            tt = node_types.get(p.name)
            if tt is None:
                self.diagnostics.append(ImportDiagnostic(
                    fx_node=p.name, level="warning",
                    message=f"No type info for placeholder {p.name}, using f32[1]",
                ))
                tt = TensorType(Float32Type(), [1])
            arg_types.append(tt)

        # Determine return types
        ret_types: list[Attribute] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in node_types:
                    ret_types.append(node_types[a.name])
        if not ret_types:
            ret_types = [TensorType(Float32Type(), [1])]

        func_type = FunctionType.from_lists(arg_types, ret_types)

        # Build the function body
        block = Block(arg_types=arg_types)
        value_map: dict[str, SSAValue] = {}
        for i, p in enumerate(placeholders):
            value_map[p.name] = block.args[i]

        # Track external function declarations for opaque fallback
        extern_funcs: list[FuncOp] = []
        declared_sigs: dict[str, str] = {}
        name_counters: dict[str, int] = {}

        # Track gelu external declaration
        gelu_declared = False

        # Process call_function nodes
        for node in call_nodes:
            target_str = str(node.target)
            result_type = node_types.get(node.name)
            if result_type is None:
                self.diagnostics.append(ImportDiagnostic(
                    fx_node=node.name, level="warning",
                    message=f"No type info for {node.name}, skipping",
                ))
                continue

            # Resolve operands
            operands: list[SSAValue] = []
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in value_map:
                    operands.append(value_map[arg.name])

            # Try decomposition table first
            decomp_fn = DECOMPOSITION_TABLE.get(target_str)
            if decomp_fn is not None:
                meta = dict(node.meta)
                result = decomp_fn(operands, meta, node.name)

                # Handle extern declarations needed by decompositions (e.g., gelu)
                for op in result.ops:
                    if isinstance(op, CallOp) and not gelu_declared:
                        callee = op.callee.string_value()
                        if callee == "aten_gelu":
                            ext = FuncOp.external("aten_gelu", [operands[0].type], [result_type])
                            extern_funcs.append(ext)
                            gelu_declared = True
                    block.add_op(op)

                if result.result is not None:
                    value_map[node.name] = result.result

                self.decomposed_count += 1
                self.diagnostics.append(ImportDiagnostic(
                    fx_node=node.name, level="info",
                    message=f"Decomposed {target_str} -> {len(result.ops)} ops (regions: {result.region_ids})",
                ))
                continue

            # Fallback: opaque func.call
            base_name = target_str.replace(".", "_")
            operand_types = tuple(str(v.type) for v in operands)
            sig_key = f"{base_name}:{operand_types}:{result_type}"

            if sig_key not in declared_sigs:
                count = name_counters.get(base_name, 0)
                unique_name = base_name if count == 0 else f"{base_name}_{count}"
                name_counters[base_name] = count + 1
                declared_sigs[sig_key] = unique_name
                real_operand_types = [v.type for v in operands]
                ext_func = FuncOp.external(unique_name, real_operand_types, [result_type])
                extern_funcs.append(ext_func)

            func_name = declared_sigs[sig_key]
            call_op = CallOp(func_name, operands, [result_type])
            block.add_op(call_op)
            value_map[node.name] = call_op.res[0]

            self.opaque_count += 1
            self.diagnostics.append(ImportDiagnostic(
                fx_node=node.name, level="info",
                message=f"Opaque: {target_str} -> func.call @{func_name}",
            ))

        # Add return
        ret_values: list[SSAValue] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in value_map:
                    ret_values.append(value_map[a.name])

        if ret_values:
            block.add_op(ReturnOp(ret_values[0]))

        region = Region([block])
        main_func = FuncOp("forward", func_type, region)

        all_ops = list(extern_funcs) + [main_func]
        module = ModuleOp(all_ops)

        try:
            module.verify()
        except Exception as e:
            self.diagnostics.append(ImportDiagnostic(
                fx_node="<module>", level="error",
                message=f"Module verification failed: {e}",
            ))

        return module

    def get_ir_text(self, module: ModuleOp) -> str:
        """Get the IR text representation of a module."""
        stream = io.StringIO()
        Printer(stream=stream).print(module)
        return stream.getvalue()


def fx_to_xdsl(exported_program: Any) -> tuple[ModuleOp, list[ImportDiagnostic]]:
    """Convenience function: export -> xDSL in one call.

    Returns:
        Tuple of (xDSL ModuleOp, list of diagnostics).
    """
    importer = FXImporter()
    module = importer.import_graph(exported_program)
    return module, importer.diagnostics


__all__ = ["FXImporter", "ImportDiagnostic", "fx_to_xdsl"]
