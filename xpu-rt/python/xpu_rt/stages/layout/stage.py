"""Layout stage — resolve virtual layout encodings into concrete pack/transpose ops.

Sits between the Encoding stage (which assigns initial layout attributes)
and the Dispatch stage (which partitions into fusion groups). Runs the
10-pass layout transform pipeline:

Shared passes (target-agnostic):
  1. canonicalize_transposes
  2. attach_layout_hints
  3. set_virtual_encodings
  4. propagate_layouts
  5. hoist_layout_ops
  7. introduce_prepacking
  9. materialize_layout_boundaries
  10. cleanup_layout_artifacts

Target plugin (target-specific):
  6. fuse_layout_into_producers
  8. specialize_layouts
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType

from compgen.stages.base import CompilationStage, IRInvariant, StageContract
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.targets.schema import TargetProfile


def _all_tensors_encoded(module: ModuleOp) -> bool:
    """Check that all tensor-producing ops have encoding attributes."""
    from xdsl.dialects.func import FuncOp, ReturnOp

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if any(isinstance(r.type, TensorType) for r in op.results):
            if ENCODING_ATTR not in op.attributes:
                return False
    return True


def _no_virtual_layout_ops(module: ModuleOp) -> bool:
    """Verify no SetLayoutOp or UnsetLayoutOp remain after the layout stage."""
    from compgen.ir.layout.ops import SetLayoutOp, UnsetLayoutOp

    for op in module.walk():
        if isinstance(op, (SetLayoutOp, UnsetLayoutOp)):
            return False
    return True


def _layout_clean(module: ModuleOp) -> bool:
    """Check the module is marked as layout-clean."""
    return "compgen.layout_clean" in module.attributes


class LayoutStage(CompilationStage):
    """Virtual layout encoding resolution stage.

    Introduces virtual layout encodings at kernel boundaries, propagates
    them through transparent ops, specializes per-target, and materializes
    at true boundaries. Only concrete pack/unpack ops remain after this stage.
    """

    @property
    def name(self) -> str:
        return "layout"

    @property
    def description(self) -> str:
        return "Resolve virtual layout encodings into concrete pack/transpose ops"

    def input_contract(self) -> StageContract:
        return StageContract(
            stage_name="layout",
            preconditions=[
                IRInvariant(
                    name="all_encoded",
                    description="All tensor-producing ops have encoding attribute",
                    custom_check=_all_tensors_encoded,
                ),
            ],
        )

    def output_contract(self) -> StageContract:
        return StageContract(
            stage_name="layout",
            postconditions=[
                IRInvariant(
                    name="no_virtual_layout_ops",
                    description="No SetLayoutOp or UnsetLayoutOp remain",
                    custom_check=_no_virtual_layout_ops,
                ),
                IRInvariant(
                    name="layout_clean",
                    description="Module marked as layout-clean",
                    custom_check=_layout_clean,
                ),
            ],
        )

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Run target-agnostic layout passes."""
        from compgen.transforms.layout.canonicalize_transposes import canonicalize_transposes
        from compgen.transforms.layout.attach_layout_hints import attach_layout_hints
        from compgen.transforms.layout.set_virtual_encodings import set_virtual_encodings
        from compgen.transforms.layout.propagate_layouts import propagate_layouts
        from compgen.transforms.layout.hoist_layout_ops import hoist_layout_ops
        from compgen.transforms.layout.introduce_prepacking import introduce_prepacking
        from compgen.transforms.layout.materialize_layout_boundaries import materialize_layout_boundaries
        from compgen.transforms.layout.cleanup_layout_artifacts import cleanup_layout_artifacts

        # Passes 1-5 (target-agnostic)
        module = canonicalize_transposes(module)
        module = attach_layout_hints(module, {})
        module = set_virtual_encodings(module)
        module = propagate_layouts(module)
        module = hoist_layout_ops(module)

        # Pass 7 (prepacking, target-agnostic)
        module = introduce_prepacking(module)

        # Passes 9-10 (materialization and cleanup)
        module = materialize_layout_boundaries(module)
        module = cleanup_layout_artifacts(module)

        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS.md"


__all__ = ["LayoutStage"]
