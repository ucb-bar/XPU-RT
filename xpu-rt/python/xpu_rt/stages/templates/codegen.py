"""Codegen stage template — generate executable source code per dispatch.

For targets that generate kernel source code (Triton, CUDA, LLVM).
Marks each dispatch with a ``compgen.codegen_backend`` attribute
indicating which backend handles it.

Reuses: kernels/autocomp_adapter.py, kernels/validate.py, transforms/apply.py.
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.base import CompilationStage, StageContract
from compgen.targets.schema import TargetProfile

CODEGEN_BACKEND_ATTR = "compgen.codegen_backend"


class CodegenStage(CompilationStage):
    """Executable source generation stage template.

    Shared passes assign a default codegen backend based on op type.
    Target plugins select optimal backends per dispatch group.
    """

    @property
    def name(self) -> str:
        return "codegen"

    @property
    def description(self) -> str:
        return "Generate executable source code for each dispatch group"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="codegen")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="codegen")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Assign default codegen backend (fallback) to all dispatched ops."""
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and CODEGEN_BACKEND_ATTR not in op.attributes:
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("fallback")
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS_codegen.md"
