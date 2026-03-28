"""Pass 3: Introduce virtual layout encodings at kernel boundaries.

For each kernel-boundary op (matmul, conv), introduces ``SetLayoutOp``
on its operands and ``UnsetLayoutOp`` after its results.  The encoding
reflects the op type, operand index, and requested layout from hints.

These virtual encodings have no runtime cost -- they are markers that
downstream passes use to propagate and specialize layouts.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block

from compgen.ir.layout.attrs import LayoutEncodingAttr
from compgen.ir.layout.ops import SetLayoutOp, UnsetLayoutOp

log = structlog.get_logger()

SET_LAYOUT_MARKER = "compgen.has_virtual_encoding"

# Ops that are kernel boundaries (layout decisions matter here)
_KERNEL_BOUNDARY_OPS = frozenset({
    "linalg.matmul",
    "linalg.generic",
    "linalg.conv_2d_nchw_fchw",
    "linalg.batch_matmul",
    "func.call",
})


def _is_ukernel_boundary(op) -> bool:  # type: ignore[no-untyped-def]
    """Check if op is a ukernel call that needs layout handling."""
    return "compgen.ukernel_ref" in op.attributes


def _is_opaque_ukernel(op) -> bool:  # type: ignore[no-untyped-def]
    """Check if a ukernel is opaque (needs materialization boundary)."""
    attr = op.attributes.get("compgen.ukernel_transparency")
    if attr and hasattr(attr, "data"):
        return attr.data == "opaque"
    return True  # Default: opaque


def _layout_from_hint(op) -> str:  # type: ignore[no-untyped-def]
    """Get the layout hint or fall back to encoding attribute."""
    hint_attr = op.attributes.get("compgen.layout_hint")
    if hint_attr and hasattr(hint_attr, "data"):
        return hint_attr.data
    enc_attr = op.attributes.get("compgen.encoding")
    if enc_attr and hasattr(enc_attr, "data"):
        return enc_attr.data
    return "rowmajor"


def _op_type_key(op) -> str:  # type: ignore[no-untyped-def]
    """Extract a short op type key for encoding."""
    name = op.name
    if "matmul" in name:
        return "matmul"
    if "conv" in name:
        return "conv"
    if "generic" in name:
        return "generic"
    return "call"


def set_virtual_encodings(module: ModuleOp) -> ModuleOp:
    """Insert SetLayoutOp/UnsetLayoutOp around kernel boundary ops.

    For each kernel boundary op:
    - Insert SetLayoutOp for each operand (encoding = op_type + index + layout).
    - Insert UnsetLayoutOp after results (materialization boundary marker).
    - Mark the op with ``compgen.has_virtual_encoding = 1``.

    Args:
        module: The xDSL ModuleOp to transform.

    Returns:
        The same ModuleOp with SetLayoutOp/UnsetLayoutOp inserted.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    encodings_set = 0
    counter = 0

    for op in list(module.walk()):
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if op.name not in _KERNEL_BOUNDARY_OPS:
            continue
        if SET_LAYOUT_MARKER in op.attributes:
            continue

        layout_str = _layout_from_hint(op)
        op_type = _op_type_key(op)

        # Create SetLayoutOp for each operand
        for i, _operand in enumerate(op.operands):
            encoding = LayoutEncodingAttr(
                op_type=op_type,
                operand_index=i,
                logical_layout=layout_str,
                tile_dims=[],
                element_types=["f32"],
            )
            ref_name = f"__layout_set_{counter}"
            counter += 1
            set_op = SetLayoutOp.build(
                properties={
                    "encoding": encoding,
                    "source_ref": SymbolRefAttr(ref_name),
                },
            )

            # Insert before the kernel op in its parent block
            parent = op.parent_block()
            if isinstance(parent, Block):
                parent.insert_op_before(set_op, op)
                encodings_set += 1

        # Create UnsetLayoutOp after the kernel op
        for _result in op.results:
            ref_name = f"__layout_unset_{counter}"
            counter += 1
            unset_op = UnsetLayoutOp.build(
                properties={
                    "source_ref": SymbolRefAttr(ref_name),
                },
            )
            parent = op.parent_block()
            if isinstance(parent, Block):
                parent.insert_op_after(unset_op, op)

        op.attributes[SET_LAYOUT_MARKER] = StringAttr("1")

    log.debug("layout.set_virtual_encodings", encodings_set=encodings_set)
    return module


__all__ = ["set_virtual_encodings"]
