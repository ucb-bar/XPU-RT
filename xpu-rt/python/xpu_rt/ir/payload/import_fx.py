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
    BFloat16Type,
    FlatSymbolRefAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp


def FlatSymbolRefAttr_ref(name: str) -> FlatSymbolRefAttr:
    """Helper: build a FlatSymbolRefAttr from a plain string name."""
    return FlatSymbolRefAttr(name)


from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.printer import Printer

from compgen.ir.payload.decompositions import (
    DECOMPOSITION_TABLE,
    DecompFn,
    reset_region_counters,
)
from compgen.ir.payload.types import Float8E4M3FNType, Float8E5M2Type

# Tags the FX-side graph passes (in ``compgen.transforms.graph_passes``) set
# on ``node.meta``. ``FXImporter`` forwards each onto the emitted xDSL ops
# so downstream Recipe-IR passes don't have to re-detect patterns the FX
# stage already recognized.
_FX_META_FORWARD_KEYS = (
    "_compgen_pattern",
    "_compgen_transpose_absorbed",
    "_compgen_fuse_dequant",
)


def _forward_fx_meta(
    op: Operation,
    fx_meta: dict[str, Any],
    decomp_hint: str | None = None,
) -> None:
    """Copy FX node meta + DecompResult.pattern_hint onto ``op.attributes``.

    - ``_compgen_pattern`` (FX-level tag) -> ``compgen._pattern_hint``
    - ``_compgen_transpose_absorbed`` -> ``compgen.transpose_absorbed`` (bool string)
    - ``_compgen_fuse_dequant`` -> ``compgen.fuse_dequant`` (bool string)
    - ``decomp_hint`` (decomp-side explicit tag) wins when FX didn't set one.

    Idempotent: won't overwrite an existing attribute.
    """
    fx_hint = fx_meta.get("_compgen_pattern") if isinstance(fx_meta, dict) else None
    effective_hint = fx_hint or decomp_hint
    if effective_hint and "compgen._pattern_hint" not in op.attributes:
        op.attributes["compgen._pattern_hint"] = StringAttr(str(effective_hint))

    if isinstance(fx_meta, dict):
        if fx_meta.get("_compgen_transpose_absorbed") and "compgen.transpose_absorbed" not in op.attributes:
            op.attributes["compgen.transpose_absorbed"] = StringAttr("true")
        if fx_meta.get("_compgen_fuse_dequant") and "compgen.fuse_dequant" not in op.attributes:
            op.attributes["compgen.fuse_dequant"] = StringAttr("true")


def _torch_dtype_to_xdsl(dtype: torch.dtype) -> Attribute:
    """Convert a torch dtype to an xDSL element type."""
    mapping: dict[torch.dtype, type] = {
        torch.float32: Float32Type,
        torch.float64: Float64Type,
        torch.float16: Float16Type,
        torch.bfloat16: BFloat16Type,
    }
    # FP8 is a first-class CompGen type (`compgen.float8_e4m3fn`,
    # `compgen.float8_e5m2`) that mirrors MLIR's semantics.  Earlier
    # revisions silently demoted to Float16Type; we now preserve the
    # FP8 semantics so Phase-2 numerics passes see the real type.
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return Float8E4M3FNType()
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return Float8E5M2Type()
    factory = mapping.get(dtype, Float32Type)
    return factory()  # type: ignore[abstract]


def _coerce_static_dim(dim: Any) -> int:
    """Concrete dim → ``int(dim)``; symbolic / data-dependent dim → ``-1``.

    xDSL's ``TensorType`` rejects ``SymInt`` (``"u6 should be of base
    attribute builtin.int"``). For models with dynamic shapes (SmolVLA's
    image-tile counts, etc.) we emit ``-1`` (xDSL's dynamic-dim convention)
    so capture continues; downstream passes that need static shapes will
    short-circuit through their own dynamic-shape paths.
    """
    try:
        return int(dim)
    except Exception:
        return -1


def _tensor_type_from_meta(val: Any) -> TensorType | None:
    """Extract a TensorType from an FX node's meta['val']."""
    if val is None:
        return None
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        elem = _torch_dtype_to_xdsl(val.dtype)
        shape = [_coerce_static_dim(d) for d in val.shape]
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
    allow_opaque_fallback: bool = True
    explicit_blackboxes: set[str] = field(default_factory=set)
    dynamic_decompositions: dict[str, DecompFn] = field(default_factory=dict)

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
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=p.name,
                        level="warning",
                        message=f"No type info for placeholder {p.name}, using f32[1]",
                    )
                )
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

        declared_callee_sig: dict[str, str] = {}

        def ensure_external_decl(call: CallOp) -> None:
            callee = call.callee.string_value()
            operand_types = tuple(str(value.type) for value in call.operands)
            result_types = tuple(str(value.type) for value in call.results)
            sig_key = f"{callee}:{operand_types}:{result_types}"
            if sig_key in declared_sigs:
                # Rewrite the call to use the canonical (possibly
                # disambiguated) name for this signature.
                call.properties["callee"] = FlatSymbolRefAttr_ref(declared_sigs[sig_key])
                return
            # If the callee name already exists with a different
            # signature, generate a unique suffixed name.
            chosen_name = callee
            if callee in declared_callee_sig:
                count = name_counters.get(callee, 1)
                while f"{callee}_{count}" in declared_callee_sig:
                    count += 1
                chosen_name = f"{callee}_{count}"
                name_counters[callee] = count + 1
            declared_callee_sig[chosen_name] = sig_key
            declared_sigs[sig_key] = chosen_name
            if chosen_name != callee:
                # Rewrite the existing CallOp to point at the new name.
                call.properties["callee"] = FlatSymbolRefAttr_ref(chosen_name)
            extern_funcs.append(
                FuncOp.external(
                    chosen_name,
                    [value.type for value in call.operands],
                    [value.type for value in call.results],
                )
            )

        # Process call_function nodes
        for node in call_nodes:
            target_str = str(node.target)
            result_type = node_types.get(node.name)
            if result_type is None:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="warning",
                        message=f"No type info for {node.name}, skipping",
                    )
                )
                continue

            # Resolve operands
            operands: list[SSAValue] = []
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in value_map:
                    operands.append(value_map[arg.name])

            # Try decomposition table first
            decomp_fn = self.dynamic_decompositions.get(target_str, DECOMPOSITION_TABLE.get(target_str))
            if decomp_fn is not None:
                meta = dict(node.meta)
                # Forward FX-level args / kwargs to the decomposition so it
                # can extract scalar properties (group_size, axis, quant_min,
                # quant_max, etc.) that don't show up as SSA operands.
                meta["_fx_args"] = tuple(node.args)
                meta["_fx_kwargs"] = dict(node.kwargs)
                try:
                    result = decomp_fn(operands, meta, node.name)
                except (IndexError, KeyError, TypeError) as decomp_err:
                    # Decomposition failed (e.g. missing operands from scalar constants).
                    # Fall through to opaque fallback instead of crashing.
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=f"Decomposition failed for {target_str}: {decomp_err}; falling back to opaque call",
                        )
                    )
                else:
                    for op in result.ops:
                        if isinstance(op, CallOp):
                            ensure_external_decl(op)
                        _forward_fx_meta(op, meta, result.pattern_hint)
                        block.add_op(op)

                    if result.result is not None:
                        value_map[node.name] = result.result

                    self.decomposed_count += 1
                    hint_suffix = f", hint: {result.pattern_hint}" if result.pattern_hint else ""
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="info",
                            message=(
                                f"Decomposed {target_str} -> {len(result.ops)} ops "
                                f"(regions: {result.region_ids}{hint_suffix})"
                            ),
                        )
                    )
                    continue

            if not self.allow_opaque_fallback and target_str not in self.explicit_blackboxes:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="error",
                        message=f"Unsupported without explicit blackbox approval: {target_str}",
                    )
                )
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
            level = "warning" if target_str in self.explicit_blackboxes else "info"
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node=node.name,
                    level=level,
                    message=f"Opaque: {target_str} -> func.call @{func_name}",
                )
            )

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

        # Reconcile the func signature with the actual return-value types.
        # The original ret_types snapshot (line ~189) was taken from the
        # FX output-node metadata, which can disagree with what the body
        # actually produces — e.g. HF Llama checkpoints declare a bf16
        # output but the attention math upcasts to f32, leaving the
        # declared func.return type at bf16 and the live SSA value at f32.
        # Without this, xDSL's verifier rejects the module on real-scale
        # transformer captures with: "Expected arguments to have the same
        # types as the function output types".
        if ret_values:
            actual_ret_types: list[Attribute] = [v.type for v in ret_values]
            if actual_ret_types != ret_types:
                func_type = FunctionType.from_lists(arg_types, actual_ret_types)

        region = Region([block])
        main_func = FuncOp("forward", func_type, region)

        all_ops = list(extern_funcs) + [main_func]
        module = ModuleOp(all_ops)

        try:
            module.verify()
        except Exception as e:
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node="<module>",
                    level="error",
                    message=f"Module verification failed: {e}",
                )
            )

        return module

    def get_ir_text(self, module: ModuleOp) -> str:
        """Get the IR text representation of a module."""
        stream = io.StringIO()
        Printer(stream=stream).print(module)
        return stream.getvalue()


def fx_to_xdsl(
    exported_program: Any,
    *,
    allow_opaque_fallback: bool = True,
    explicit_blackboxes: set[str] | None = None,
    dynamic_decompositions: dict[str, DecompFn] | None = None,
) -> tuple[ModuleOp, list[ImportDiagnostic]]:
    """Convenience function: export -> xDSL in one call.

    Returns:
        Tuple of (xDSL ModuleOp, list of diagnostics).
    """
    importer = FXImporter(
        allow_opaque_fallback=allow_opaque_fallback,
        explicit_blackboxes=set(explicit_blackboxes or ()),
        dynamic_decompositions=dict(dynamic_decompositions or {}),
    )
    module = importer.import_graph(exported_program)
    return module, importer.diagnostics


__all__ = ["FXImporter", "ImportDiagnostic", "fx_to_xdsl"]
