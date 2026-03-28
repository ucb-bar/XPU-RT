"""LLM-driven transform script generation.

Thin wrapper around ``agent/pass_gen.py`` (PassGenerator) for the
transforms/ API surface. The actual generation logic lives in PassGenerator;
this module provides the TransformScript types and convenience functions.

Invariants:
    - Generated scripts must parse as valid Python.
    - All generation calls go through the LLM recorder.
    - Multiple candidates can be generated for selection via verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.llm.base import CompGenLLMProtocol, Objective
from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class TransformScript:
    """A generated transform script.

    Attributes:
        name: Script identifier.
        content: The transform script text (Python RewritePattern code).
        parameters: The parameters chosen by the LLM (tile sizes, etc.).
        template_name: Which template was used (if any).
        generation_metadata: LLM call metadata (model, tokens, latency).
    """

    name: str
    content: str
    guard_refs: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    template_name: str | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformSynthesizer:
    """LLM-driven transform script synthesizer.

    Delegates to ``agent.pass_gen.PassGenerator`` for the actual generation.

    Attributes:
        llm_client: The LLM client to use for generation.
        max_candidates: Number of candidates to generate per call.
    """

    llm_client: CompGenLLMProtocol
    max_candidates: int = 5

    def synthesize(
        self,
        ir_summary: str,
        target: TargetProfile,
        module: ModuleOp,
        objective: Objective,
        kernel_contracts: list[str] | None = None,
        prior_feedback: str = "",
        guard_refs: tuple[str, ...] = (),
    ) -> list[TransformScript]:
        """Generate transform script candidates.

        Args:
            ir_summary: Canonical IR summary text.
            target: Target hardware profile.
            objective: Optimization objective.
            kernel_contracts: Serialized kernel contracts (YAML strings).
            prior_feedback: Verification feedback from prior attempts.

        Returns:
            List of TransformScript candidates.
        """
        from compgen.agent.pass_gen import PassGenerator

        generator = PassGenerator(llm_client=self.llm_client)
        description = (
            f"Generate a RewritePattern for {target.name} "
            f"optimizing {objective.value}. "
            f"IR summary: {ir_summary[:500]}"
        )
        if prior_feedback:
            description += f"\nPrior feedback: {prior_feedback}"

        scripts: list[TransformScript] = []
        for i in range(min(self.max_candidates, 1)):  # Generate 1 for now
            result = generator.generate(
                description=description,
                target_pattern="",
                expected_effect="optimization",
                module=module,
                guard_refs=guard_refs,
            )
            if result.verified and result.source_code:
                scripts.append(TransformScript(
                    name=f"transform_{i}",
                    content=result.source_code,
                    guard_refs=result.guard_refs,
                    generation_metadata={"verified": result.verified},
                ))

        return scripts


def synthesize_transforms(
    ir_summary: str,
    target: TargetProfile,
    module: ModuleOp,
    objective: Objective,
    llm_client: CompGenLLMProtocol,
    *,
    guard_refs: tuple[str, ...] = (),
) -> list[TransformScript]:
    """Convenience function: synthesize transforms with defaults."""
    synthesizer = TransformSynthesizer(llm_client=llm_client)
    return synthesizer.synthesize(ir_summary, target, module, objective, guard_refs=guard_refs)


__all__ = ["TransformScript", "TransformSynthesizer", "synthesize_transforms"]
