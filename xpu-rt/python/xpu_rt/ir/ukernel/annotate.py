"""Annotate IR ops with ukernel information for layout integration.

After kernel selection identifies ops that match registered ukernels,
this module annotates those ops with ``compgen.ukernel_ref``,
``compgen.ukernel_transparency``, and ``compgen.ukernel_tile_family``
attributes. The layout bridge then uses these attributes to decide
whether to propagate layouts through the kernel or materialize at
boundaries.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.ir import Operation

from compgen.ir.ukernel.constraints import ConstraintContext
from compgen.ir.ukernel.registry import UkernelRegistry

log = structlog.get_logger()


def _op_to_family(op: Operation) -> str:
    """Map an xDSL op name to a ukernel op_family string."""
    name = op.name
    if "matmul" in name:
        return "matmul"
    if "conv" in name:
        return "conv"
    return name.split(".")[-1] if "." in name else name


def _shapes_from_op(op: Operation) -> dict[str, int]:
    """Extract M, N, K shape dimensions from an op's operands."""
    shapes: dict[str, int] = {}
    operand_shapes = []
    for operand in op.operands:
        if isinstance(operand.type, TensorType):
            operand_shapes.append(operand.type.get_shape())

    if operand_shapes:
        first = operand_shapes[0]
        if len(first) >= 2:
            shapes["M"] = first[0] if first[0] > 0 else 1
            shapes["K"] = first[1] if first[1] > 0 else 1
        if len(operand_shapes) > 1:
            second = operand_shapes[1]
            if len(second) >= 2:
                shapes["N"] = second[1] if second[1] > 0 else 1

    return shapes


def _dtypes_from_op(op: Operation) -> tuple[str, ...]:
    """Extract dtype strings from an op's operands."""
    dtypes: set[str] = set()
    for operand in op.operands:
        if isinstance(operand.type, TensorType):
            dtypes.add(str(operand.type.element_type))
    return tuple(dtypes) if dtypes else ("f32",)


def annotate_ukernel_ops(
    module: ModuleOp,
    registry: UkernelRegistry,
    target_features: frozenset[str] = frozenset(),
    device_type: str = "",
) -> int:
    """Annotate IR ops with ukernel match information.

    Walks the module and for each op that matches a registered ukernel,
    sets:
    - ``compgen.ukernel_ref``: kernel name
    - ``compgen.ukernel_transparency``: "transparent" or "opaque"
    - ``compgen.ukernel_tile_family``: tile family hint (if any)

    Args:
        module: The xDSL module to annotate.
        registry: Populated UkernelRegistry.
        target_features: Target capability features.
        device_type: Device type string.

    Returns:
        Number of ops annotated.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    annotated = 0

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if not op.results:
            continue
        # Skip ops already annotated
        if "compgen.ukernel_ref" in op.attributes:
            continue

        op_family = _op_to_family(op)
        shapes = _shapes_from_op(op)
        dtypes = _dtypes_from_op(op)

        context = ConstraintContext(
            shapes=shapes,
            dtypes=dtypes,
            target_features=target_features,
            device_type=device_type,
        )

        decl = registry.select_ukernel(op_family, context)
        if decl is None:
            continue

        op.attributes["compgen.ukernel_ref"] = StringAttr(decl.kernel_name)
        op.attributes["compgen.ukernel_transparency"] = StringAttr(decl.transparency)
        if decl.tile_family:
            op.attributes["compgen.ukernel_tile_family"] = StringAttr(decl.tile_family)

        annotated += 1
        log.debug(
            "ukernel.annotate",
            op=op.name,
            kernel=decl.kernel_name,
            transparency=decl.transparency,
            tile_family=decl.tile_family,
        )

    return annotated


__all__ = ["annotate_ukernel_ops"]
