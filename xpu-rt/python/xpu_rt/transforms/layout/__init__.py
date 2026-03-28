"""Layout transform passes for the data tiling bridge.

Ten passes that progressively resolve virtual layout encodings into
concrete pack/transpose operations:

1. canonicalize_transposes -- fold redundant transposes
2. attach_layout_hints -- bridge analysis plans to IR annotations
3. set_virtual_encodings -- introduce SetLayoutOp at kernel boundaries
4. propagate_layouts -- push encodings through transparent ops
5. hoist_layout_ops -- move SetLayoutOp to dominating positions
6. fuse_layout_into_producers -- eliminate boundaries when producer absorbs layout
7. introduce_prepacking -- insert PackOp for constant operands
8. specialize_layouts -- target-specific encoding resolution
9. materialize_layout_boundaries -- replace virtual ops with real pack/unpack
10. cleanup_layout_artifacts -- remove dead layout ops
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xdsl.dialects.builtin import ModuleOp

if TYPE_CHECKING:
    from compgen.analysis.layout.planner import LayoutPlan
    from compgen.targets.capability import CapabilitySpec
    from compgen.targets.schema import TargetProfile


def run_layout_pipeline(
    module: ModuleOp,
    plans: dict[str, "LayoutPlan"] | None = None,
    resolver: Any | None = None,
    target: "TargetProfile | None" = None,
    capabilities: "CapabilitySpec | None" = None,
    prepack_candidates: list[Any] | None = None,
) -> ModuleOp:
    """Run all 10 layout passes in order."""
    from compgen.transforms.layout.attach_layout_hints import attach_layout_hints
    from compgen.transforms.layout.canonicalize_transposes import canonicalize_transposes
    from compgen.transforms.layout.cleanup_layout_artifacts import cleanup_layout_artifacts
    from compgen.transforms.layout.fuse_layout_into_producers import fuse_layout_into_producers
    from compgen.transforms.layout.hoist_layout_ops import hoist_layout_ops
    from compgen.transforms.layout.introduce_prepacking import introduce_prepacking
    from compgen.transforms.layout.materialize_layout_boundaries import materialize_layout_boundaries
    from compgen.transforms.layout.propagate_layouts import propagate_layouts
    from compgen.transforms.layout.set_virtual_encodings import set_virtual_encodings
    from compgen.transforms.layout.specialize_layouts import specialize_layouts

    module = canonicalize_transposes(module)
    module = attach_layout_hints(module, plans or {})
    module = set_virtual_encodings(module)
    module = propagate_layouts(module)
    module = hoist_layout_ops(module)
    module = fuse_layout_into_producers(module)
    module = introduce_prepacking(module, prepack_candidates or [])
    module = specialize_layouts(module, resolver=resolver, capabilities=capabilities)
    module = materialize_layout_boundaries(module)
    module = cleanup_layout_artifacts(module)
    return module


__all__ = ["run_layout_pipeline"]
