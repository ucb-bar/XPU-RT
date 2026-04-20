"""``expand_tensor_shapes`` -- normalize dynamic-dim carriers.

XLA's ``ExpandTensorShapes``: when a tensor carries symbolic dims
(``-1`` in xDSL's convention), every op that consumes it inherits
the dynamic shape. The pass walks the IR and:

1. Records every op producing a tensor with at least one dynamic
   dim.
2. Tags those ops with ``compgen.dynamic_dim_mask`` -- a string
   of ``'1'``/``'0'`` per dim where ``1`` marks a dynamic extent.
3. For symmetry with ETC's symbolic-shape support (Wave 8), also
   emits a ``compgen.symbolic_shape_template`` attribute holding
   the full shape tuple as a comma-separated string so the tiling
   pass can reason about the template.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType


@dataclass
class ExpandTensorShapesStats:
    ops_seen: int = 0
    ops_with_dynamic_dims: int = 0
    shape_templates_emitted: int = 0


def run_expand_tensor_shapes(
    module: ModuleOp,
) -> ExpandTensorShapesStats:
    stats = ExpandTensorShapesStats()
    for op in module.walk():
        if not op.results:
            continue
        for res in op.results:
            t = res.type
            if not isinstance(t, TensorType):
                continue
            stats.ops_seen += 1
            shape = list(t.get_shape())
            if any(d < 0 for d in shape):
                stats.ops_with_dynamic_dims += 1
                mask = "".join("1" if d < 0 else "0" for d in shape)
                op.attributes["compgen.dynamic_dim_mask"] = StringAttr(mask)
            # Always emit the template for downstream tiling.
            op.attributes["compgen.symbolic_shape_template"] = StringAttr(",".join(str(d) for d in shape))
            stats.shape_templates_emitted += 1
            break  # only tag the first result
    return stats


__all__ = [
    "ExpandTensorShapesStats",
    "run_expand_tensor_shapes",
]
