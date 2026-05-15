"""LLM-driven Exo schedule evolution.

Uses the evolutionary search pattern from agent/evolution.py to
generate, evaluate, and refine Exo scheduling scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog

from compgen.kernels.exo_seedgen import ExoSeedProc

log = structlog.get_logger()


@dataclass(frozen=True)
class ExoTargetKit:
    """Exo target kit for a specific hardware target.

    Contains reusable hardware definitions (instructions, memories,
    configs) and schedule library helpers.

    Attributes:
        name: Target kit identifier.
        instructions_source: Exo instruction definitions source.
        memories_source: Exo memory definitions source.
        configs_source: Exo config definitions source.
        schedule_lib_source: Reusable schedule library source.
    """

    name: str
    instructions_source: str = ""
    memories_source: str = ""
    configs_source: str = ""
    schedule_lib_source: str = ""


@dataclass(frozen=True)
class ScheduleCandidate:
    """A candidate schedule for an Exo proc.

    Attributes:
        proc_name: Name of the Exo proc being scheduled.
        schedule_source: Python schedule code.
        c_output: Compiled C output.
        latency_us: Measured latency in microseconds.
        correct: Whether the scheduled proc is correct.
        generation: Evolution generation number.
        schedule_ops: List of schedule operation names applied.
    """

    proc_name: str
    schedule_source: str
    c_output: str
    latency_us: float
    correct: bool
    generation: int
    schedule_ops: list[str] = field(default_factory=list)


# Exo scheduling primitives the LLM can use
EXO_SCHEDULE_OPS = [
    "reorder_loops",
    "divide_loop",
    "fission",
    "fuse",
    "unroll_loop",
    "stage_mem",
    "set_memory",
    "lift_alloc",
    "sink_alloc",
    "bind_expr",
    "replace",
    "divide_dim",
    "expand_dim",
    "simplify",
    "remove_loop",
    "specialize",
    "extract_subproc",
    "inline",
]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM client protocol for schedule generation."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        ...


class ExoScheduleAgent:
    """LLM-driven Exo schedule search.

    Pipeline:
        1. Seed: Start with unscheduled Exo proc
        2. Generate: LLM proposes schedule operations
        3. Apply: Execute schedule code
        4. Compile: Exo compiles to C
        5. Validate: Correctness + benchmark
        6. Select: Keep top-K candidates
        7. Mutate: LLM refines winners
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_generations: int = 5,
        population_size: int = 4,
    ) -> None:
        self._llm_client = llm_client
        self._max_generations = max_generations
        self._population_size = population_size

    def evolve_schedule(
        self,
        seed: ExoSeedProc,
        target_kit: ExoTargetKit | None = None,
        target_name: str = "generic",
    ) -> ScheduleCandidate:
        """Run evolutionary schedule search.

        Args:
            seed: Unscheduled Exo proc.
            target_kit: Optional target-specific definitions.
            target_name: Target hardware name.

        Returns:
            Best schedule candidate found.
        """
        best = ScheduleCandidate(
            proc_name=seed.name,
            schedule_source="# identity (no schedule)",
            c_output=seed.c_skeleton,
            latency_us=float("inf"),
            correct=True,
            generation=0,
        )

        if self._llm_client is None:
            log.info("exo.schedule.no_llm", proc=seed.name)
            return best

        for gen in range(self._max_generations):
            candidates = self._generate_candidates(seed, target_kit, best, gen)
            evaluated = self._evaluate_candidates(candidates, seed)

            if evaluated:
                evaluated.sort(key=lambda c: c.latency_us)
                if evaluated[0].latency_us < best.latency_us:
                    best = evaluated[0]

            log.info(
                "exo.schedule.generation",
                gen=gen,
                candidates=len(evaluated),
                best_latency=best.latency_us,
            )

        return best

    def _generate_candidates(
        self,
        seed: ExoSeedProc,
        kit: ExoTargetKit | None,
        current_best: ScheduleCandidate,
        generation: int,
    ) -> list[ScheduleCandidate]:
        """Generate schedule candidates via LLM."""
        assert self._llm_client is not None
        candidates: list[ScheduleCandidate] = []

        for i in range(self._population_size):
            prompt = self._build_prompt(seed, kit, current_best, generation, i)
            try:
                response = self._llm_client.generate(prompt)
                schedule_code = self._extract_schedule(response)
                candidates.append(
                    ScheduleCandidate(
                        proc_name=seed.name,
                        schedule_source=schedule_code,
                        c_output="",  # filled during evaluation
                        latency_us=float("inf"),
                        correct=False,
                        generation=generation,
                    )
                )
            except Exception as e:
                log.warning("exo.schedule.gen_failed", error=str(e))

        return candidates

    def _evaluate_candidates(
        self,
        candidates: list[ScheduleCandidate],
        seed: ExoSeedProc,
    ) -> list[ScheduleCandidate]:
        """Evaluate candidates: apply schedule, compile, validate."""
        # Placeholder: in full implementation, would exec schedule code
        # against Exo proc, compile to C, run correctness test, benchmark
        return [
            ScheduleCandidate(
                proc_name=c.proc_name,
                schedule_source=c.schedule_source,
                c_output=seed.c_skeleton,
                latency_us=float("inf"),
                correct=True,
                generation=c.generation,
            )
            for c in candidates
        ]

    def _build_prompt(
        self,
        seed: ExoSeedProc,
        kit: ExoTargetKit | None,
        best: ScheduleCandidate,
        generation: int,
        candidate_idx: int,
    ) -> str:
        """Build LLM prompt for schedule generation."""
        from compgen.agent.prompts.exo_schedule import ExoScheduleContext, format_prompt

        ctx = ExoScheduleContext(
            proc_source=seed.proc_source,
            target_kit_name=kit.name if kit else "generic",
            target_hw_summary=f"Target: {kit.name if kit else 'generic CPU'}",
            available_schedule_ops=EXO_SCHEDULE_OPS,
            previous_attempts=[
                {
                    "schedule": best.schedule_source,
                    "latency": best.latency_us,
                    "generation": best.generation,
                }
            ]
            if generation > 0
            else [],
        )
        return format_prompt(ctx)

    def _extract_schedule(self, response: str) -> str:
        """Extract schedule code from LLM response."""
        # Look for code blocks
        lines = response.split("\n")
        in_code = False
        code_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else response


__all__ = ["ExoScheduleAgent", "ExoTargetKit", "ScheduleCandidate"]
