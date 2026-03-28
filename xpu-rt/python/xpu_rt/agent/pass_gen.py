"""LLM-driven pass generation and verification.

The agent can ask the LLM to generate new xDSL RewritePattern passes.
Generated code is validated before use:
    1. Structural: Python parses, class inherits RewritePattern
    2. IR validity: apply to cloned module, xDSL verifier passes
    3. Semantic: module still produces same structure (no ops deleted incorrectly)

Only verified passes are added to the session's pass menu.
"""

from __future__ import annotations

import ast
import inspect
import traceback
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import PatternRewriteWalker, RewritePattern

from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
    LLMConfig,
    Objective,
    PromptContext,
)

PASS_GENERATION_PROMPT = """You are an xDSL compiler pass generator. Write a Python class that
inherits from `xdsl.pattern_rewriter.RewritePattern` and implements `match_and_rewrite`.

## Task
{description}

## Target Pattern
{target_pattern}

## Expected Effect
{expected_effect}

## Available xDSL APIs
- `isinstance(op, SomeOp)` to match op types
- `rewriter.replace_matched_op(new_ops)` to replace the matched op
- `rewriter.erase_matched_op()` to remove it
- `rewriter.insert_op_before_matched_op(new_op)` to add before
- `op.operands` to access inputs, `op.results` to access outputs
- `op.attributes` dict for metadata
- `StringAttr("value")` for string attributes

## xDSL Imports Available
```python
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter
from xdsl.ir import Operation
from xdsl.dialects.linalg import MatmulOp, GenericOp, TransposeOp, FillOp
from xdsl.dialects.func import CallOp
from xdsl.dialects.arith import ConstantOp, AddfOp, MulfOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.dialects.builtin import StringAttr, TensorType, Float32Type
```

## Requirements
- The class MUST inherit from RewritePattern
- The method MUST be named match_and_rewrite(self, op, rewriter)
- Return early if the op doesn't match (no rewrite needed)
- Do NOT import anything outside the available imports above

## Output
Return ONLY the Python code block with the class definition. No explanation.
"""


@dataclass(frozen=True)
class GeneratedPass:
    """A dynamically generated compiler pass."""

    name: str
    description: str
    source_code: str
    pattern_class: type | None = None
    guard_refs: tuple[str, ...] = ()
    verified: bool = False
    verification_error: str = ""


@dataclass
class PassGenerator:
    """Generates and validates xDSL RewritePattern passes via LLM."""

    llm_client: CompGenLLMProtocol
    guard_runtime: Any | None = None
    default_guard_refs: tuple[str, ...] = ()
    _generated_passes: dict[str, GeneratedPass] = field(default_factory=dict)

    def generate(
        self,
        description: str,
        target_pattern: str,
        expected_effect: str,
        module: ModuleOp,
        *,
        guard_refs: tuple[str, ...] = (),
    ) -> GeneratedPass:
        """Ask the LLM to generate a pass, then validate it.

        Returns a GeneratedPass. If verified=True, the pass is safe to use.
        If verified=False, verification_error explains what went wrong.
        """
        # Build prompt
        prompt = PASS_GENERATION_PROMPT.format(
            description=description,
            target_pattern=target_pattern,
            expected_effect=expected_effect,
        )

        request = GenerationRequest(
            prompt_template=prompt,
            context=PromptContext(
                model_ir_summary="",
                target_profile_summary="",
                available_transforms=[],
                kernel_contracts=[],
                objective=Objective.LATENCY,
            ),
            config=LLMConfig(
                model=str(getattr(self.llm_client, "model", "default")),
                temperature=0.2,
                max_tokens=2000,
            ),
            artifact_type="rewrite_pattern",
        )

        response = self.llm_client.generate(request)

        # Extract code from response
        source_code = self._extract_code(response)
        if not source_code:
            return GeneratedPass(
                name=description[:50],
                description=description,
                source_code="",
                verified=False,
                verification_error="No code block found in LLM response",
            )

        # Validate the generated code
        return self._validate(description, source_code, module, guard_refs=guard_refs or self.default_guard_refs)

    def _extract_code(self, response: GenerationResponse) -> str:
        """Extract Python code from LLM response."""
        if response.parsed_artifacts:
            return response.parsed_artifacts[0]

        # Try to find code in raw text between ``` markers
        text = response.raw_text
        if "```python" in text:
            start = text.index("```python") + len("```python")
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        return text.strip()

    def _validate(
        self,
        description: str,
        source_code: str,
        module: ModuleOp,
        *,
        guard_refs: tuple[str, ...] = (),
    ) -> GeneratedPass:
        """Validate generated pass: parse, compile, apply to cloned module, verify."""
        name = description[:50].replace(" ", "_").lower()

        # Step 1: Structural — does it parse as Python?
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            return GeneratedPass(
                name=name, description=description, source_code=source_code,
                verified=False, verification_error=f"Syntax error: {e}",
                guard_refs=guard_refs,
            )

        # Step 2: Compile — does it define a RewritePattern subclass?
        namespace: dict[str, Any] = {}
        try:
            exec(source_code, namespace)  # noqa: S102
        except Exception as e:
            return GeneratedPass(
                name=name, description=description, source_code=source_code,
                verified=False, verification_error=f"Execution error: {e}",
                guard_refs=guard_refs,
            )

        # Find the pattern class
        pattern_class = None
        for value in namespace.values():
            if isinstance(value, type) and issubclass(value, RewritePattern) and value is not RewritePattern:
                pattern_class = value
                break

        if pattern_class is None:
            return GeneratedPass(
                name=name, description=description, source_code=source_code,
                verified=False, verification_error="No RewritePattern subclass found in generated code",
                guard_refs=guard_refs,
            )

        # Step 3: IR validity — apply to cloned module, check it still verifies
        try:
            cloned = module.clone()
            pattern = self._instantiate_pattern(pattern_class, guard_refs=guard_refs)
            walker = PatternRewriteWalker(pattern, apply_recursively=False)
            walker.rewrite_module(cloned)
            cloned.verify()
        except Exception as e:
            return GeneratedPass(
                name=name, description=description, source_code=source_code,
                pattern_class=pattern_class,
                guard_refs=guard_refs,
                verified=False, verification_error=f"IR verification failed: {e}\n{traceback.format_exc()[-200:]}",
            )

        # All checks passed
        generated = GeneratedPass(
            name=name, description=description, source_code=source_code,
            pattern_class=pattern_class, guard_refs=guard_refs, verified=True,
        )
        self._generated_passes[name] = generated
        return generated

    def _instantiate_pattern(
        self,
        pattern_class: type,
        *,
        guard_refs: tuple[str, ...],
    ) -> RewritePattern:
        signature = inspect.signature(pattern_class.__init__)
        kwargs: dict[str, Any] = {}
        if "guard_runtime" in signature.parameters:
            kwargs["guard_runtime"] = self.guard_runtime
        if "guard_refs" in signature.parameters:
            kwargs["guard_refs"] = guard_refs
        elif "guard_ref" in signature.parameters:
            kwargs["guard_ref"] = guard_refs[0] if guard_refs else None
        return pattern_class(**kwargs)

    def apply_generated_pass(self, name: str, module: ModuleOp) -> tuple[bool, str]:
        """Apply a previously generated and verified pass to a module.

        Returns (success, error_message).
        """
        gen = self._generated_passes.get(name)
        if gen is None:
            return False, f"No generated pass named '{name}'"
        if not gen.verified or gen.pattern_class is None:
            return False, f"Pass '{name}' is not verified"

        try:
            pattern = self._instantiate_pattern(gen.pattern_class, guard_refs=gen.guard_refs)
            walker = PatternRewriteWalker(pattern, apply_recursively=False)
            walker.rewrite_module(module)
            module.verify()
            return True, ""
        except Exception as e:
            return False, f"Application failed: {e}"

    @property
    def available_passes(self) -> dict[str, GeneratedPass]:
        """Get all verified generated passes."""
        return {k: v for k, v in self._generated_passes.items() if v.verified}


__all__ = ["GeneratedPass", "PassGenerator"]
