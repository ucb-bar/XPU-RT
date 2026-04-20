"""Transform application via xDSL.

Takes synthesized transform scripts and applies them to the payload IR.
Application is deterministic — the same script on the same IR always
produces the same result.

Invariants:
    - Application never mutates the input module (returns a new one).
    - Application failures produce diagnostics, not crashes.
    - The output module passes the xDSL verifier.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.transforms.synthesize import TransformScript


@dataclass(frozen=True)
class TransformDiagnostic:
    """Diagnostic from a transform application.

    Attributes:
        transform_name: Name of the transform that produced this diagnostic.
        level: "error", "warning", or "info".
        message: Description.
        op_name: The op being transformed (if applicable).
    """

    transform_name: str
    level: str
    message: str
    op_name: str = ""


@dataclass(frozen=True)
class TransformedIR:
    """Result of applying transforms to the payload IR.

    Attributes:
        module: The transformed xDSL module.
        scripts_applied: List of transform scripts that were successfully applied.
        diagnostics: Diagnostics from the application process.
    """

    module: Any
    scripts_applied: list[TransformScript] = field(default_factory=list)
    diagnostics: list[TransformDiagnostic] = field(default_factory=list)


@dataclass
class TransformApplicator:
    """Applies transform scripts to xDSL modules.

    Applies each script by:
    1. Parsing the Python code
    2. Executing to find a RewritePattern subclass
    3. Applying via PatternRewriteWalker
    4. Verifying the result
    """

    def apply(self, module: ModuleOp, scripts: list[TransformScript]) -> TransformedIR:
        """Apply transform scripts to an xDSL module.

        Args:
            module: The input xDSL module (not mutated).
            scripts: Transform scripts to apply in order.

        Returns:
            TransformedIR with the new module and diagnostics.
        """
        from xdsl.pattern_rewriter import (
            GreedyRewritePatternApplier,
            PatternRewriteWalker,
            RewritePattern,
        )

        result_module = module.clone()
        applied: list[TransformScript] = []
        diagnostics: list[TransformDiagnostic] = []

        for script in scripts:
            try:
                # Parse and validate
                ast.parse(script.content)
            except SyntaxError as e:
                diagnostics.append(
                    TransformDiagnostic(
                        transform_name=script.name,
                        level="error",
                        message=f"Syntax error: {e}",
                    )
                )
                continue

            # Execute to find RewritePattern
            try:
                namespace: dict[str, Any] = {}
                exec(
                    "from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, op_type_rewrite_pattern\n"
                    "from xdsl.dialects import arith, linalg, func\n"
                    "from xdsl.dialects.builtin import ModuleOp\n",
                    namespace,
                )
                exec(script.content, namespace)
            except Exception as e:
                diagnostics.append(
                    TransformDiagnostic(
                        transform_name=script.name,
                        level="error",
                        message=f"Execution error: {e}",
                    )
                )
                continue

            # Find RewritePattern subclass
            patterns = [
                self._instantiate_pattern(v, script.guard_refs)
                for v in namespace.values()
                if isinstance(v, type) and issubclass(v, RewritePattern) and v is not RewritePattern
            ]

            if not patterns:
                diagnostics.append(
                    TransformDiagnostic(
                        transform_name=script.name,
                        level="warning",
                        message="No RewritePattern subclass found in script",
                    )
                )
                continue

            # Apply patterns
            try:
                PatternRewriteWalker(
                    GreedyRewritePatternApplier(patterns),
                    apply_recursively=False,
                ).rewrite_module(result_module)

                # Verify result
                result_module.verify()

                applied.append(script)
                diagnostics.append(
                    TransformDiagnostic(
                        transform_name=script.name,
                        level="info",
                        message="Applied successfully",
                    )
                )
            except Exception as e:
                diagnostics.append(
                    TransformDiagnostic(
                        transform_name=script.name,
                        level="error",
                        message=f"Application failed: {e}",
                    )
                )

        return TransformedIR(
            module=result_module,
            scripts_applied=applied,
            diagnostics=diagnostics,
        )

    def _instantiate_pattern(self, pattern_class: type, guard_refs: tuple[str, ...]) -> Any:
        signature = inspect.signature(pattern_class.__init__)
        kwargs: dict[str, Any] = {}
        if "guard_runtime" in signature.parameters:
            kwargs["guard_runtime"] = None
        if "guard_refs" in signature.parameters:
            kwargs["guard_refs"] = guard_refs
        elif "guard_ref" in signature.parameters:
            kwargs["guard_ref"] = guard_refs[0] if guard_refs else None
        return pattern_class(**kwargs)


def apply_transforms(module: ModuleOp, scripts: list[TransformScript]) -> TransformedIR:
    """Convenience function: apply transforms with defaults."""
    applicator = TransformApplicator()
    return applicator.apply(module, scripts)


__all__ = ["TransformApplicator", "TransformDiagnostic", "TransformedIR", "apply_transforms"]
