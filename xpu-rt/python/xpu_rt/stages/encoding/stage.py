"""Encoding stage — resolve data layouts and dtype decisions per-target.

Analogous to IREE's Encoding dialect.  Annotates every tensor-typed value
with layout preferences (row-major, column-major, tiled) and dtype
narrowing decisions.

Shared passes:
  - Infer default row-major encoding for all tensors
  - Propagate encoding constraints through data flow

Target plugin generates:
  - GPU Triton: MMA-friendly layouts (tile [128,64] for A100 tensor cores)
  - Accel NPU: DMA-aligned layouts
  - CPU: cache-friendly tiled layouts
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.ir import Operation

from compgen.stages.base import CompilationStage, IRInvariant, StageContract
from compgen.targets.schema import TargetProfile

# Attribute key used to mark encoding decisions
ENCODING_ATTR = "compgen.encoding"


def _has_tensor_results(op: Operation) -> bool:
    """Check if any result has tensor type."""
    return any(isinstance(r.type, TensorType) for r in op.results)


def _all_tensors_encoded(module: ModuleOp) -> bool:
    """Check if all tensor-producing ops have encoding attributes."""
    from xdsl.dialects.func import FuncOp, ReturnOp

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if _has_tensor_results(op) and ENCODING_ATTR not in op.attributes:
            return False
    return True


class EncodingStage(CompilationStage):
    """Data layout and encoding resolution stage.

    Every target needs this stage.  The shared passes assign default
    row-major encoding.  Target plugins override with hardware-specific
    layouts.
    """

    @property
    def name(self) -> str:
        return "encoding"

    @property
    def description(self) -> str:
        return "Resolve data layouts and dtype decisions per-target"

    def input_contract(self) -> StageContract:
        return StageContract(
            stage_name="encoding",
            preconditions=[
                IRInvariant(
                    name="valid_module",
                    description="Module passes xDSL verifier",
                    custom_check=lambda m: _try_verify(m),
                ),
            ],
        )

    def output_contract(self) -> StageContract:
        return StageContract(
            stage_name="encoding",
            postconditions=[
                IRInvariant(
                    name="all_encoded",
                    description="All tensor-producing ops have encoding attribute",
                    custom_check=_all_tensors_encoded,
                ),
            ],
        )

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Assign default row-major encoding to all tensor-producing ops."""
        from xdsl.dialects.func import FuncOp, ReturnOp

        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if _has_tensor_results(op) and ENCODING_ATTR not in op.attributes:
                op.attributes[ENCODING_ATTR] = StringAttr("row_major")

        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS.md"


def _try_verify(module: ModuleOp) -> bool:
    """Try to verify a module, return True if it passes.

    Logs the exception when verification fails — silent failure here makes
    pipeline-stage diagnostics ("valid_module: custom check failed") useless
    on real-scale modules.
    """
    try:
        module.verify()
        return True
    except Exception as exc:
        import structlog

        structlog.get_logger().error(
            "encoding.verify_failed",
            exception=type(exc).__name__,
            message=str(exc)[:500],
        )
        return False
