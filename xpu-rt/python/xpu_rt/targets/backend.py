"""Generalized target backend protocol for CompGen.

Defines the interface that any hardware target backend implements to plug
into CompGen's compilation pipeline.  Inspired by Hexagon-MLIR's
``BaseBackend`` pattern but generalized for Python-native backends.

A target backend provides:

1. **Compilation stages** — Like Hexagon's ``add_stages()``, a sequence of
   transformations from Linalg IR to target code.
2. **Options** — Target-specific configuration (tile sizes, memory,
   features).
3. **Validation** — Check compiled artifacts against golden data.

Any hardware (NPU, GPU, FPGA, custom accelerator) implements this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from compgen.targets.options import TargetOptions


@dataclass
class CompiledArtifact:
    """Result of compiling a kernel or model for a target.

    Returned by :meth:`VendorDialectAdapter.emit_artifact` and similar
    backend-emit methods. There is **no** ``artifact_path`` field —
    when the artifact lands on disk (e.g. a ``.tilebc`` bytecode
    written under ``output_dir``), put the path in ``metadata`` (e.g.
    ``metadata["artifact_path"] = str(out / "kernel.tilebc")``) and
    set ``code`` to either the inline source/text or a base64-encoded
    handle. Per bridge #137: agentic-compilation wrappers should
    declare ``format`` explicitly (e.g. ``"cuda-tile-bitcode"``) so
    the host's verify gate knows whether to read ``code`` directly
    or open ``metadata["artifact_path"]``.

    Attributes:
        code: Generated code (assembly, MLIR, Python, C, etc.).
            For binary artifacts: base64 string or text excerpt with
            a path-handle in ``metadata``.
        format: Code format identifier (e.g. ``"python"``,
            ``"mlir-cuda-tile"``, ``"cuda-tile-bitcode"``,
            ``"ptx"``). Distinguishes which downstream consumer can
            handle the code.
        target_name: Which target this was compiled for.
        metadata: Compilation metadata (pass stats, timing, etc.).
            Recommended keys when the artifact is on disk:
            ``"artifact_path"`` (str), ``"artifact_size_bytes"`` (int).
    """

    code: str = ""
    format: str = "python"
    target_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationStageResult:
    """Result from a single compilation stage.

    Attributes:
        stage_name: Name of the stage that ran.
        success: Whether the stage completed successfully.
        ir_text: The IR after this stage (if applicable).
        diagnostics: Any warnings or errors from this stage.
    """

    stage_name: str = ""
    success: bool = True
    ir_text: str = ""
    diagnostics: list[str] = field(default_factory=list)


@runtime_checkable
class TargetBackendProtocol(Protocol):
    """Protocol that any hardware target backend implements.

    Modeled after Hexagon-MLIR's ``BaseBackend`` but adapted for
    CompGen's Python-native pipeline.

    A backend provides:
    - ``supports_target()``: Whether it handles a given target name
    - ``get_options()``: Default compilation options
    - Compilation stages: ``lower_linalg``, ``tile_for_memory``,
      ``decompose_to_microops``, ``emit_code``
    - ``validate()``: Check against golden data
    """

    def supports_target(self, target_name: str) -> bool:
        """Return True if this backend handles the given target.

        Equivalent to Hexagon's ``BaseBackend.supports_target(target)``.
        """
        ...

    def get_options(self) -> TargetOptions:
        """Return default compilation options for this target."""
        ...

    def get_compilation_stages(self) -> list[str]:
        """Return the ordered list of compilation stage names.

        Equivalent to the keys in Hexagon's ``stages`` dict from
        ``add_stages()``.  Example: ``["lower", "tile", "decompose", "emit"]``.
        """
        ...

    def compile_stage(
        self,
        stage_name: str,
        ir_text: str,
        options: TargetOptions,
    ) -> CompilationStageResult:
        """Run a single compilation stage.

        Args:
            stage_name: Which stage to run.
            ir_text: Input IR (MLIR/xDSL text).
            options: Compilation options.

        Returns:
            Stage result with transformed IR.
        """
        ...

    def compile(
        self,
        ir_text: str,
        options: TargetOptions | None = None,
    ) -> CompiledArtifact:
        """Run the full compilation pipeline.

        Runs all stages in order and returns the final artifact.

        Args:
            ir_text: Input linalg IR.
            options: Override options (uses defaults if None).

        Returns:
            Compiled artifact with target code.
        """
        ...

    def validate(
        self,
        artifact: CompiledArtifact,
        golden_inputs: dict[str, Any],
        golden_output: Any,
    ) -> bool:
        """Validate a compiled artifact against golden data.

        Args:
            artifact: The compiled artifact to test.
            golden_inputs: Input tensors (from golden data).
            golden_output: Expected output tensor.

        Returns:
            True if the artifact produces correct output.
        """
        ...


class BaseTargetBackend:
    """Base implementation of the target backend protocol.

    Provides default stage orchestration. Subclasses implement the
    target-specific stages.
    """

    def __init__(self, options: TargetOptions | None = None) -> None:
        self._options = options or TargetOptions()

    def supports_target(self, target_name: str) -> bool:
        return target_name == self._options.target_name

    def get_options(self) -> TargetOptions:
        return self._options

    def get_compilation_stages(self) -> list[str]:
        return ["lower", "tile", "decompose", "emit"]

    def compile_stage(
        self,
        stage_name: str,
        ir_text: str,
        options: TargetOptions,
    ) -> CompilationStageResult:
        """Default: pass-through (no transformation)."""
        return CompilationStageResult(
            stage_name=stage_name,
            success=True,
            ir_text=ir_text,
        )

    def compile(
        self,
        ir_text: str,
        options: TargetOptions | None = None,
    ) -> CompiledArtifact:
        """Run all stages in order."""
        opts = options or self._options
        current_ir = ir_text

        for stage in self.get_compilation_stages():
            result = self.compile_stage(stage, current_ir, opts)
            if not result.success:
                return CompiledArtifact(
                    code=current_ir,
                    format="error",
                    target_name=opts.target_name,
                    metadata={"failed_stage": stage, "diagnostics": result.diagnostics},
                )
            current_ir = result.ir_text

        return CompiledArtifact(
            code=current_ir,
            format=opts.emit_format,
            target_name=opts.target_name,
        )

    def validate(
        self,
        artifact: CompiledArtifact,
        golden_inputs: dict[str, Any],
        golden_output: Any,
    ) -> bool:
        """Default: no validation (override in subclass)."""
        return True


__all__ = [
    "BaseTargetBackend",
    "CompiledArtifact",
    "CompilationStageResult",
    "TargetBackendProtocol",
]
