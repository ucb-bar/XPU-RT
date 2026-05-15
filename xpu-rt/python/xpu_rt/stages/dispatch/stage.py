"""Dispatch stage — partition the graph into dispatch groups.

Analogous to IREE's Flow dialect.  Determines which ops run together
as a single kernel dispatch.  Assigns a ``compgen.dispatch_id`` attribute
to every op.

Shared passes:
  - Baseline partitioning via solve/partition.py
  - Assign dispatch IDs to ops

Target plugin generates:
  - GPU: group elementwise ops around matmuls for operator fusion
  - NPU: group into hardware dispatch slots
  - Hybrid: partition across device types
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.base import CompilationStage, IRInvariant, StageContract
from compgen.targets.schema import TargetProfile

DISPATCH_ID_ATTR = "compgen.dispatch_id"


def _all_ops_dispatched(module: ModuleOp) -> bool:
    """Check that every non-structural op has a dispatch_id."""
    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if not op.results:
            continue
        if DISPATCH_ID_ATTR not in op.attributes:
            return False
    return True


class DispatchStage(CompilationStage):
    """Graph partitioning into dispatch groups.

    Every target needs this stage.  The shared passes assign sequential
    dispatch IDs (one per op).  Target plugins override with fusion-aware
    grouping.
    """

    @property
    def name(self) -> str:
        return "dispatch"

    @property
    def description(self) -> str:
        return "Partition the graph into dispatch groups (fusion regions)"

    def input_contract(self) -> StageContract:
        return StageContract(
            stage_name="dispatch",
            preconditions=[
                IRInvariant(
                    name="has_encoding",
                    description="All tensor ops should have encoding",
                    # Lenient: we don't strictly require encoding from the previous stage
                    # because some IR (arith-only) doesn't have tensor results
                ),
            ],
        )

    def output_contract(self) -> StageContract:
        return StageContract(
            stage_name="dispatch",
            postconditions=[
                IRInvariant(
                    name="all_dispatched",
                    description="Every op has a dispatch_id attribute",
                    custom_check=_all_ops_dispatched,
                ),
            ],
        )

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Assign sequential dispatch IDs (baseline: one dispatch per op)."""
        dispatch_id = 0
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue
            if DISPATCH_ID_ATTR not in op.attributes:
                op.attributes[DISPATCH_ID_ATTR] = StringAttr(f"d_{dispatch_id}")
                dispatch_id += 1

        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS.md"
