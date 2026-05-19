"""KernelBlaster vanilla (best-effort) for the Gemmini comparison study.

Reproduces vanilla KernelBlaster's **prompting strategy** (strategy-guided
prompt, optimization-database excerpt, source-to-optimize, N rounds with
no inter-round feedback) without dragging in KB's CUDA-tied
orchestration (LangGraph workflow, NCU profiler, docker harness). The
comparison study cares about *whether KB's prompt approach finds a
Gemmini kernel*, not whether KB's orchestration can be retargeted —
forking KB to add a Gemmini evaluator is 1–2 weeks and explicitly out
of scope (see plan 2 "Out of scope").

Inputs that mirror KB:

* The kernel-contract describing the op (what to optimize).
* A "starting source" — a naive scalar implementation. KB rewrites
  this iteratively; here we hand it as ``original_code`` once and ask
  the LLM to rewrite it in one shot per round.
* A "strategy" description chosen from a fixed menu (tiling, dataflow
  selection, reduce-overhead, vectorize). KB picks via its RL bandit;
  this bridge picks via simple round-robin over the menu.
* The optimization-database equivalent: the Target Card's
  ``isa.md`` + ``intrinsics.md`` + ``constraints.md`` concatenated, so
  the LLM sees the same hardware knowledge XPU-RT has.

Routes through ``xpu_rt.llm.factory`` so every call is gated by the
$100 budget cap and recorded by ``observability.gemini_usage``. The
existing OpenAI-SDK instrumentation (see
:func:`xpu_rt.observability.gemini_usage.install_openai_instrumentation`)
already covers KB-style OpenAI-compat clients if a future caller wants
to use them, but the bridge defaults to ``KernelGeneratorLLM`` directly.

Scoring (no NCU, no spike — purely post-emit static + compile):

* ``compile``: 1 if ``riscv64-unknown-linux-gnu-gcc -c`` succeeds.
* ``intrinsic_use_rate``: fraction of ``gemmini_*`` macro calls in the
  emission that the Target Card actually documents (everything else is
  hallucinated).
* ``shape_consistency``: True iff every input/output dim from the
  contract appears as a literal in the kernel source.
* ``tokens_in`` / ``tokens_out`` / ``cost_usd``: pre/post diff of
  ``gemini_usage.load_summary()``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
    _cc_bin,
    _check_toolchain,
    _gemmini_include_args,
    _ToolchainMissing,
)
from xpu_rt.kernels.kernelblaster_v2.generators import (
    KernelGeneratorLLM,
    ProposeRequest,
    ProposeResponse,
)
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import (
    GENERATOR_RESPONSE_SCHEMA,
)
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory.target_knowledge import TargetKnowledgeCard
from xpu_rt.observability import gemini_usage

logger = logging.getLogger(__name__)


# The "strategy menu" — abbreviated translation of KB's technique table
# into Gemmini terms. KB picks via RL bandits; we just round-robin so
# each run sees the same opportunity set.
_STRATEGIES: tuple[tuple[str, str], ...] = (
    (
        "tile_and_dma_overlap",
        "Tile the matmul along M/N/K to fit Gemmini's scratchpad + accumulator. "
        "Overlap A and B mvin lanes (use gemmini_extended_mvin2 for the B operand) "
        "so DMA runs in parallel with the systolic array.",
    ),
    (
        "weight_stationary_dataflow",
        "Use the weight-stationary dataflow (config_ex with WS). Preload B once per "
        "(m, n) output tile and stream A through the array. Reduces re-loads of "
        "the weight matrix from DRAM.",
    ),
    (
        "accumulator_keep_in_place",
        "Keep partial sums in the Gemmini accumulator across the inner-K loop. "
        "Use the accumulate-flag bit on the C scratchpad address for k>0 rounds; "
        "only mvout the final result.",
    ),
    (
        "loop_ws_cisc",
        "Use the CISC loop instruction gemmini_loop_ws which expands into the "
        "matmul-loop sequence. Fewer host instructions; lets Gemmini's reservation "
        "station overlap LD/EX/ST.",
    ),
)


# ---------------------------------------------------------------------------
# Prompt construction — mirror of KB's strategy-guided prompt
# ---------------------------------------------------------------------------


KB_SYSTEM_PROMPT = """\
You are a Gemmini RoCC accelerator kernel optimization expert.

You have access to a comprehensive optimization-knowledge document for
Gemmini (provided as the OPTIMIZATION DATABASE below) and an existing
naive scalar kernel for an op. Your job is to emit one improved C
kernel that uses Gemmini's intrinsics to satisfy the contract.

Conventions:
- The kernel function must be named exactly `kernel_under_test`.
- Argument list matches the contract's (inputs, outputs) order.
- Include `#include "include/gemmini.h"` at the top.
- Use only intrinsics that appear in the OPTIMIZATION DATABASE under
  `intrinsics.md`. Do not invent macros (no made-up `RVV_SEW_8`-style
  constants, no nonexistent `__riscv_vmv_v_i_*` variants).
- Honour every dim from the contract; the shape literals must appear
  in the source.

Return one JSON object matching the schema (no surrounding prose, no
markdown fences).
"""


def _build_kb_prompt(
    *,
    contract: KernelContract,
    card: TargetKnowledgeCard,
    strategy_name: str,
    strategy_desc: str,
    prior_attempts: tuple[dict[str, Any], ...],
    starting_source: str,
) -> str:
    optimization_database = _concat_card_buckets(card)
    contract_summary = _contract_summary(contract)
    prior = _format_prior(prior_attempts)
    body = f"""\
OPTIMIZATION DATABASE (Gemmini ISA + intrinsics + constraints, from the
Target Card; this is the only set of intrinsics you may use):
```
{optimization_database[:8000]}
```

STRATEGY FOR THIS ROUND: {strategy_name}
{strategy_desc}

KERNEL CONTRACT:
{contract_summary}

STARTING SCALAR SOURCE (rewrite into a Gemmini-accelerated kernel; the
function name must stay `kernel_under_test`):
```c
{starting_source}
```
{prior}
Produce the optimized kernel. Return JSON matching the schema."""
    return body


def _concat_card_buckets(card: TargetKnowledgeCard) -> str:
    parts: list[str] = []
    for bucket in ("isa", "intrinsics", "constraints", "architecture"):
        path = card.bucket_path(bucket)
        if not path.exists():
            continue
        parts.append(f"## {bucket}\n\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else "(target card buckets empty)"


def _contract_summary(contract: KernelContract) -> str:
    lines = [
        f"- region_id: {contract.region_id or '(unset)'}",
        f"- op_family: {contract.op_family}",
        f"- dtypes: {', '.join(contract.dtypes)}",
        f"- layout: {contract.layout}",
        f"- target_name: {contract.target_name}",
    ]
    if contract.input_shapes:
        lines.append("- input_shapes:")
        for s in contract.input_shapes:
            lines.append(f"    - {list(s)}")
    if contract.output_shapes:
        lines.append("- output_shapes:")
        for s in contract.output_shapes:
            lines.append(f"    - {list(s)}")
    return "\n".join(lines)


def _format_prior(prior: tuple[dict[str, Any], ...]) -> str:
    if not prior:
        return ""
    lines = ["", "PRIOR ATTEMPTS IN THIS RUN:"]
    for i, a in enumerate(prior, 1):
        lines.append(
            f"  attempt {i}: strategy={a.get('strategy','?')} "
            f"compile={a.get('compile','?')} "
            f"intrinsic_use_rate={a.get('intrinsic_use_rate',-1.0):.2f}"
        )
        if a.get("notes"):
            lines.append(f"    notes: {a['notes']}")
        # The compile log tail is the most actionable signal — it tells
        # the LLM exactly which macro it called wrong and what the real
        # arity is. Quote it verbatim so the next-round prompt teaches
        # the model the fix.
        tail = a.get("compile_log_tail") or ""
        if tail and not a.get("compile"):
            lines.append("    compile_log_tail (verbatim):")
            for ln in tail.splitlines()[-12:]:
                lines.append(f"      {ln.rstrip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Naive scalar starting source (for matmul i8 → i32)
# ---------------------------------------------------------------------------


def _starting_scalar_matmul(contract: KernelContract) -> str:
    A, B = contract.input_shapes
    (C,) = contract.output_shapes
    M, K = A[0], A[1]
    N = B[1]
    return f"""\
#include <stdint.h>

#define M {M}
#define K {K}
#define N {N}

void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {{
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) acc += (int32_t)A[m*K + k] * (int32_t)B[k*N + n];
            C[m*N + n] = acc;
        }}
}}
"""


def _starting_source(contract: KernelContract) -> str:
    op_family = contract.op_family.lower()
    if op_family in ("matmul", "mm", "gemm", "bmm", "linear"):
        return _starting_scalar_matmul(contract)
    raise NotImplementedError(
        f"KernelBlasterGemminiBridge has no scalar starting source for op_family="
        f"{contract.op_family!r}; only matmul is wired for Phase A."
    )


# ---------------------------------------------------------------------------
# Post-emit scoring
# ---------------------------------------------------------------------------


_GEMMINI_CALL_RE = re.compile(r"\b(gemmini_[a-zA-Z0-9_]+)\s*\(")


def _intrinsic_use_rate(kernel_code: str, card: TargetKnowledgeCard) -> tuple[float, int, int]:
    """Return (rate, matched, total). rate=0 when total=0."""
    calls = _GEMMINI_CALL_RE.findall(kernel_code)
    if not calls:
        return 0.0, 0, 0
    known = {i.name for i in card.hardware_spec.intrinsics}
    matched = sum(1 for c in calls if c in known)
    return (matched / len(calls), matched, len(calls))


def _shape_consistency(kernel_code: str, contract: KernelContract) -> tuple[bool, list[int]]:
    """True iff every shape literal in the contract appears in the source."""
    dims: list[int] = []
    for s in (*contract.input_shapes, *contract.output_shapes):
        dims.extend(int(d) for d in s)
    missing = [d for d in dims if str(d) not in kernel_code]
    return (not missing, missing)


def _compile_check(kernel_code: str, timeout_s: int = 30) -> tuple[bool, str]:
    """Run ``riscv64-unknown-linux-gnu-gcc -c`` on the candidate. Return
    (ok, log)."""
    try:
        _check_toolchain()
    except _ToolchainMissing as exc:
        return False, str(exc)
    workdir = Path(tempfile.mkdtemp(prefix="xpu_rt_kb_vanilla_compile_"))
    src = workdir / "kernel.c"
    obj = workdir / "kernel.o"
    src.write_text(kernel_code)
    cmd = [
        str(_cc_bin()),
        "-c",
        "-std=gnu99",
        "-O2",
        "-fno-common",
        "-march=rv64gc",
        "-Wa,-march=rv64gc",
        *_gemmini_include_args(),
        str(src),
        "-o",
        str(obj),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return False, f"compile timeout: {exc}"
    log = (proc.stdout + proc.stderr)[-2000:]
    return proc.returncode == 0, log


# ---------------------------------------------------------------------------
# Result type + runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KBVanillaResult:
    """One run of the vanilla-KB bridge against one contract."""

    contract_hash: str
    region_id: str
    op_family: str
    target_name: str
    rounds: int
    final_kernel_code: str
    final_strategy: str
    compile: bool
    compile_log: str
    intrinsic_use_rate: float
    intrinsic_matched: int
    intrinsic_total: int
    shape_consistency: bool
    shape_missing: list[int]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    wall_s: float
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_hash": self.contract_hash,
            "region_id": self.region_id,
            "op_family": self.op_family,
            "target_name": self.target_name,
            "rounds": self.rounds,
            "compile": self.compile,
            "intrinsic_use_rate": self.intrinsic_use_rate,
            "intrinsic_matched": self.intrinsic_matched,
            "intrinsic_total": self.intrinsic_total,
            "shape_consistency": self.shape_consistency,
            "shape_missing": self.shape_missing,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "wall_s": self.wall_s,
            "final_strategy": self.final_strategy,
            "final_kernel_code_chars": len(self.final_kernel_code),
            "attempts": self.attempts,
        }


@dataclass
class KernelBlasterGemminiBridge:
    """Vanilla-KB-style runner; outputs a :class:`KBVanillaResult` per contract.

    Picks one strategy per round from the fixed menu (round-robin) and
    asks the LLM to rewrite the scalar starting source under that
    strategy. Stops when the emission compiles and uses real intrinsics,
    or after ``max_rounds``.
    """

    target_card: TargetKnowledgeCard
    max_rounds: int = 4
    model: str = "gemini-2.5-flash"
    acceptance_intrinsic_rate: float = 0.5
    """The bridge stops early when a round produces (a) a compiling
    kernel AND (b) ``intrinsic_use_rate >= this threshold``. KB's
    real stop signal is NCU speedup; we use static signals only."""

    def run(self, contract: KernelContract) -> KBVanillaResult:
        starting = _starting_source(contract)
        gen = KernelGeneratorLLM(model=self.model)
        attempts: list[dict[str, Any]] = []
        pre_summary = gemini_usage.load_summary()
        pre_tokens_in = pre_summary.total_prompt_tokens
        pre_tokens_out = pre_summary.total_completion_tokens
        pre_cost = pre_summary.total_cost_usd
        t0 = time.perf_counter()

        best_emission = starting
        best_strategy = "scalar_starting"
        best_compile = False
        best_compile_log = ""
        best_rate = 0.0
        best_matched = 0
        best_total = 0
        best_shape_ok = False
        best_shape_missing: list[int] = []

        for r in range(self.max_rounds):
            strategy_name, strategy_desc = _STRATEGIES[r % len(_STRATEGIES)]
            prompt = _build_kb_prompt(
                contract=contract,
                card=self.target_card,
                strategy_name=strategy_name,
                strategy_desc=strategy_desc,
                prior_attempts=tuple(attempts),
                starting_source=best_emission if best_compile else starting,
            )
            bundle = _bundle(prompt)
            req = ProposeRequest(bundle=bundle, attempt_index=r, state_hash="kb-vanilla")
            resp = gen.propose(req)
            emission = resp.kernel_code or ""
            compile_ok, compile_log = _compile_check(emission)
            rate, matched, total = _intrinsic_use_rate(emission, self.target_card)
            shape_ok, shape_missing = _shape_consistency(emission, contract)
            attempt_row = {
                "round": r,
                "strategy": strategy_name,
                "compile": compile_ok,
                "intrinsic_use_rate": rate,
                "intrinsic_matched": matched,
                "intrinsic_total": total,
                "shape_consistency": shape_ok,
                "shape_missing": shape_missing,
                "compile_log_tail": compile_log[-400:] if compile_log else "",
            }
            attempts.append(attempt_row)
            # Track best emission. Acceptance order:
            #   1. compile=True beats compile=False
            #   2. higher intrinsic_use_rate breaks ties
            #   3. when ALL rounds fail to compile, still record the
            #      best-by-rate (or the latest non-empty emission) so
            #      the report reflects what the LLM actually produced
            #      rather than the scalar fallback.
            replace_best = False
            if compile_ok and not best_compile:
                replace_best = True
            elif compile_ok == best_compile and rate > best_rate:
                replace_best = True
            elif (
                not best_compile
                and best_rate == 0.0
                and emission.strip()
                and total > 0
            ):
                # Initial best is the scalar fallback (no gemmini_* calls).
                # Any LLM emission that names ≥1 gemmini_* macro should
                # supersede it even if it didn't compile.
                replace_best = True
            if replace_best:
                best_emission = emission
                best_strategy = strategy_name
                best_compile = compile_ok
                best_compile_log = compile_log
                best_rate = rate
                best_matched = matched
                best_total = total
                best_shape_ok = shape_ok
                best_shape_missing = shape_missing
            if compile_ok and rate >= self.acceptance_intrinsic_rate:
                break

        wall = time.perf_counter() - t0
        post = gemini_usage.load_summary()
        return KBVanillaResult(
            contract_hash=contract.region_id or "(noid)",
            region_id=contract.region_id,
            op_family=contract.op_family,
            target_name=contract.target_name,
            rounds=len(attempts),
            final_kernel_code=best_emission,
            final_strategy=best_strategy,
            compile=best_compile,
            compile_log=best_compile_log[-1000:],
            intrinsic_use_rate=best_rate,
            intrinsic_matched=best_matched,
            intrinsic_total=best_total,
            shape_consistency=best_shape_ok,
            shape_missing=best_shape_missing,
            tokens_in=post.total_prompt_tokens - pre_tokens_in,
            tokens_out=post.total_completion_tokens - pre_tokens_out,
            cost_usd=post.total_cost_usd - pre_cost,
            wall_s=wall,
            attempts=attempts,
        )


def _bundle(prompt: str):  # type: ignore[no-untyped-def]
    """Wrap the prompt in a :class:`PromptBundle`-shaped object for the
    generator.

    KernelGeneratorLLM only reads ``bundle.system`` and ``bundle.user``,
    not the schema fields, so we synthesise a tiny shim rather than
    pulling in the full PromptBuilder."""
    from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBundle

    return PromptBundle(
        system=KB_SYSTEM_PROMPT,
        user=prompt,
        schema=GENERATOR_RESPONSE_SCHEMA,
        metadata={"runner": "kb_vanilla_gemmini_bridge"},
    )


__all__ = [
    "KBVanillaResult",
    "KernelBlasterGemminiBridge",
]
