"""``iterate_kernel`` — bench → distill → re-prompt → codegen loop.

Cheap alternative to autocomp's 8-iteration beam search. The structure
is roughly:

    for attempt in range(max_attempts):
        source = codegen(contract, prior_source_and_diagnosis)
        bench  = run_microbench(source, eager_ref, inputs)
        if bench.passes_target:
            return (source, bench)
        diagnosis = diagnose(contract, bench)
        prior_source_and_diagnosis = (source, diagnosis)
    return best_so_far

The caller supplies:

* ``codegen_callable``  — takes (contract, previous_source, diagnosis) and
  returns a kernel source string. In production this is the Claude Code
  MCP round-trip; for tests and demos it's a Python function.
* ``compile_and_bench`` — takes kernel source, produces a ``BenchResult``.
  This is where Triton JIT happens + timing is measured. Kept pluggable
  so we can test the loop without a GPU.

Cost: **one codegen call per attempt**. With ``max_attempts=3`` and
Claude Code in-session cost ≈ $0, total spend per kernel is ~$0 for the
first attempt + $0.05 per refinement = **~$0.10–0.15 worst case**. Vs
autocomp's ~$15.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from compgen.bench.diagnosis import (
    KernelDiagnosis,
    diagnose,
)
from compgen.bench.kernel_bench import BenchResult
from compgen.bench.refinement import build_refinement_prompt
from compgen.kernels.contract_v3 import KernelContractV3

# ---------------------------------------------------------------------------
# Callable contracts
# ---------------------------------------------------------------------------


# Signature: codegen(contract, previous_source, diagnosis, refinement_prompt)
# previous_source / diagnosis are None on the first attempt.
CodegenCallable = Callable[
    [KernelContractV3, str | None, KernelDiagnosis | None, str | None],
    str,
]

# Compile + bench the kernel. Returns a BenchResult (which also carries
# passed/failed for correctness) OR raises on a hard compile/runtime error.
BenchCallable = Callable[[str, KernelContractV3], BenchResult]


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass
class IterationAttempt:
    attempt: int  # 1-indexed
    kernel_source: str
    bench: BenchResult
    diagnosis: KernelDiagnosis
    prompt_used: str  # the prompt that produced kernel_source (or "" on attempt 1)


@dataclass
class IterationOutcome:
    """Full record of an iterate_kernel call.

    ``best_attempt`` points at the iteration that produced the shipped
    kernel (best perf among those that passed correctness). ``converged``
    is True when we hit ``perf_target_us``; False when we exhausted
    ``max_attempts`` and returned the best so-far.
    """

    contract: KernelContractV3
    attempts: list[IterationAttempt] = field(default_factory=list)
    best_attempt_idx: int = -1
    converged: bool = False
    perf_target_us: float | None = None
    escalate_to_autocomp: bool = False

    @property
    def best(self) -> IterationAttempt | None:
        if not self.attempts or self.best_attempt_idx < 0:
            return None
        return self.attempts[self.best_attempt_idx]

    def summary(self) -> str:
        if not self.attempts:
            return "(no attempts)"
        lines = [
            f"iterate_kernel {self.contract.op_name!r} — {len(self.attempts)} attempt(s), "
            f"converged={self.converged}, escalate={self.escalate_to_autocomp}"
        ]
        for a in self.attempts:
            marker = " ← best" if a.attempt - 1 == self.best_attempt_idx else ""
            lines.append(
                f"  [{a.attempt}] {a.bench.our_us:>7.1f}μs  "
                f"eager={a.bench.eager_us:>7.1f}μs  "
                f"vs_eager={a.bench.us_ratio_vs_eager:>5.2f}x  "
                f"passed={a.bench.passed}  "
                f"bottleneck={a.diagnosis.primary_bottleneck.value}{marker}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def iterate_kernel(
    contract: KernelContractV3,
    codegen: CodegenCallable,
    compile_and_bench: BenchCallable,
    *,
    perf_target_us: float | None = None,
    max_attempts: int = 3,
) -> IterationOutcome:
    """Run the cheap refinement loop.

    ``perf_target_us`` — when provided, the loop converges as soon as an
    attempt passes correctness AND meets the target. When ``None`` the
    loop runs to ``max_attempts`` and returns the best (lowest latency
    that passed correctness).
    """
    outcome = IterationOutcome(contract=contract, perf_target_us=perf_target_us)

    prior_source: str | None = None
    prior_diag: KernelDiagnosis | None = None
    prior_prompt: str | None = None

    for i in range(1, max_attempts + 1):
        # 1. Ask codegen for a kernel — with refinement context when available.
        refinement_prompt = (
            build_refinement_prompt(
                contract,
                prior_source,
                prior_diag,
                perf_target_us=perf_target_us,
            )
            if prior_source is not None and prior_diag is not None
            else None
        )
        source = codegen(contract, prior_source, prior_diag, refinement_prompt)

        # 2. Compile + bench.
        bench = compile_and_bench(source, contract)

        # 3. Distill.
        diag = diagnose(contract, bench)

        outcome.attempts.append(
            IterationAttempt(
                attempt=i,
                kernel_source=source,
                bench=bench,
                diagnosis=diag,
                prompt_used=refinement_prompt or "",
            )
        )

        # 4. Converged?
        passes_correctness = bench.passed
        passes_perf = perf_target_us is None or (passes_correctness and bench.our_us <= perf_target_us)
        if passes_correctness and passes_perf and perf_target_us is not None:
            outcome.converged = True
            outcome.best_attempt_idx = i - 1
            return outcome

        # 5. Set up next round.
        prior_source = source
        prior_diag = diag
        prior_prompt = refinement_prompt

    # Out of attempts — pick the best-so-far.
    passing = [(idx, a) for idx, a in enumerate(outcome.attempts) if a.bench.passed]
    if passing:
        best_idx, _best = min(passing, key=lambda kv: kv[1].bench.our_us)
        outcome.best_attempt_idx = best_idx
    elif outcome.attempts:
        # Nothing passed correctness — return the lowest-latency anyway (caller
        # will see passed=False and route accordingly).
        best_idx, _best = min(
            enumerate(outcome.attempts),
            key=lambda kv: kv[1].bench.our_us,
        )
        outcome.best_attempt_idx = best_idx

    # Escalation decision: if even the best failed correctness OR is still
    # dramatically slower than eager, signal that autocomp should take over.
    if outcome.best is not None:
        still_bad_correctness = not outcome.best.bench.passed
        still_dramatically_slow = outcome.best.bench.us_ratio_vs_eager > 3.0
        outcome.escalate_to_autocomp = still_bad_correctness or still_dramatically_slow

    outcome.converged = (
        outcome.best is not None
        and outcome.best.bench.passed
        and (perf_target_us is None or outcome.best.bench.our_us <= perf_target_us)
    )
    return outcome


__all__ = [
    "BenchCallable",
    "CodegenCallable",
    "IterationAttempt",
    "IterationOutcome",
    "iterate_kernel",
]
