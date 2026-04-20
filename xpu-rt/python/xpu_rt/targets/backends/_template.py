"""Template: Custom Hardware Target Backend

Copy this file into the ``backends/`` directory and implement the
``TargetBackendProtocol`` for your hardware.

See ``compgen.targets.backend`` for the protocol definition.
See ``compgen.targets.options`` for the base options class.
See ``docs/architecture/target-backend-model.md`` for the Hexagon-inspired architecture.

Steps:
    1. Create a directory: ``backends/my_target/``
    2. Copy this file: ``cp _template.py backends/my_target/__init__.py``
    3. Define your options class extending ``TargetOptions``
    4. Implement the 4 compilation stages (lower, tile, decompose, emit)
    5. Implement validation against golden data
"""

from __future__ import annotations

from dataclasses import dataclass

from compgen.targets.backend import BaseTargetBackend, CompilationStageResult
from compgen.targets.options import TargetOptions


@dataclass(frozen=True)
class TemplateTargetOptions(TargetOptions):
    """Options for the template target.

    Replace with your hardware-specific configuration.
    """

    target_name: str = "my_target"
    # Add your hardware-specific options:
    # matrix_unit_size: int = 16
    # scratchpad_bytes: int = 262144
    # vector_width: int = 128
    # enable_dma: bool = True


class TemplateTargetBackend(BaseTargetBackend):
    """Template target backend.

    Implements the 4-stage compilation pipeline:
    1. lower: Linalg → target-specific ops
    2. tile: Tile operations for target geometry
    3. decompose: Break into hardware microops
    4. emit: Generate target code (ISA, C, assembly)
    """

    def __init__(self) -> None:
        super().__init__(TemplateTargetOptions())

    def supports_target(self, target_name: str) -> bool:
        return target_name == "my_target"

    def compile_stage(
        self,
        stage_name: str,
        ir_text: str,
        options: TargetOptions,
    ) -> CompilationStageResult:
        """Implement each compilation stage.

        Args:
            stage_name: "lower", "tile", "decompose", or "emit".
            ir_text: Current IR state.
            options: Compilation options.
        """
        if stage_name == "lower":
            # TODO: Convert linalg ops to target-specific ops
            return CompilationStageResult(stage_name="lower", ir_text=ir_text)

        if stage_name == "tile":
            # TODO: Tile operations to fit target geometry
            # Example: 32x32 tiles for a matrix unit
            return CompilationStageResult(stage_name="tile", ir_text=ir_text)

        if stage_name == "decompose":
            # TODO: Break tiled ops into hardware microops
            # Like Hexagon's hexkl.matmul → hexkl.micro_hmx_* sequence
            return CompilationStageResult(stage_name="decompose", ir_text=ir_text)

        if stage_name == "emit":
            # TODO: Generate target code (assembly, C, ISA)
            code = f"# Generated code for {options.target_name}\n{ir_text}"
            return CompilationStageResult(stage_name="emit", ir_text=code)

        return CompilationStageResult(stage_name=stage_name, ir_text=ir_text)
