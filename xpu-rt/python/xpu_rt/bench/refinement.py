"""Turn a ``KernelDiagnosis`` + the prior kernel source + the contract
into the *refinement prompt* for the next codegen attempt.

This is the inverse of what ``kernel.py``'s ``_render_prompt`` does for
the initial request — that one renders the contract cold; this one
renders "here's what you wrote, here's what happened, here's what to
try next."

Kept tight: the LLM's context is precious. We trim the prior source
if needed and inject only the hypotheses + 3-5 numbers that justify
them.
"""

from __future__ import annotations

import textwrap

from compgen.bench.diagnosis import Bottleneck, KernelDiagnosis
from compgen.kernels.contract_v3 import KernelContractV3


_MAX_PRIOR_SOURCE_CHARS = 4096  # trim oversized kernels before injecting


def build_refinement_prompt(
    contract: KernelContractV3,
    previous_source: str,
    diagnosis: KernelDiagnosis,
    *,
    perf_target_us: float | None = None,
) -> str:
    """Render the refinement prompt.

    The prompt has four sections:

      * **what you wrote**   — trimmed prior kernel source
      * **what happened**    — previous_attempt_summary + metrics
      * **hypotheses**       — ordered list of changes to try
      * **contract reminder**— small re-statement of IO + numerics
    """
    # Trim prior source from the middle (preserve head + tail) to fit budget.
    prior = previous_source.strip()
    if len(prior) > _MAX_PRIOR_SOURCE_CHARS:
        keep = _MAX_PRIOR_SOURCE_CHARS // 2
        prior = prior[:keep] + "\n\n# ... [trimmed for prompt size] ...\n\n" + prior[-keep:]

    # Numbers to surface. Cap to the top-5 most meaningful.
    key_metrics = (
        "our_us", "eager_us", "vs_eager_ratio",
        "bandwidth_efficiency", "compute_efficiency",
        "arithmetic_intensity_flops_per_byte",
    )
    shown: list[str] = []
    for k in key_metrics:
        if k in diagnosis.supporting_metrics:
            v = diagnosis.supporting_metrics[k]
            shown.append(f"  - {k}: {v:.3g}")

    hypos_lines = [
        f"  {i}. {h}" for i, h in enumerate(diagnosis.hypotheses, 1)
    ]

    io = contract.io
    io_summary = (
        f"op={contract.op_name!r} archetype={contract.archetype.value} "
        f"granularity={contract.granularity.value}; "
        f"{len(io.inputs)} in → {len(io.outputs)} out; "
        f"numerics.fast_math={io.numerics.fast_math} "
        f"max_rel_err={io.numerics.max_relative_error:g}"
    )

    target_line = (
        f"\nPerf target: ≤{perf_target_us}μs (your last attempt: "
        f"{diagnosis.supporting_metrics.get('our_us', 0):.1f}μs)"
        if perf_target_us is not None
        else ""
    )

    prompt = textwrap.dedent(f"""\
        You wrote this kernel but it's underperforming. Refine it.

        # 1. Your previous kernel
        ```
        {prior}
        ```

        # 2. What happened
        previous attempt: {diagnosis.previous_attempt_summary}
        compared to     : {diagnosis.compared_to}
        bottleneck      : {diagnosis.primary_bottleneck.value}
        efficiency      : {diagnosis.roofline_efficiency*100:.1f}% of peak roof
        key metrics:
        {chr(10).join(shown) if shown else '  (no metrics collected)'}
        {target_line}

        # 3. Top hypotheses — try IN ORDER, pick the one with the highest expected impact
        {chr(10).join(hypos_lines) if hypos_lines else '  (no hypotheses — the kernel is near-roof already; propose an algorithmic change or stop iterating)'}

        # 4. Contract reminder (don't change the IO contract — only the kernel body)
        {io_summary}

        Respond with ONLY the refined kernel source. No explanation, no markdown
        fences. Keep the same function signature; mutate only the body and the
        autotune configs.
        """
    )
    return prompt


__all__ = ["build_refinement_prompt"]
