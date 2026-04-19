"""Observational metadata: how much does torch.inductor cover each ported pass.

Per the approved P7/P8 plan (and the "port everything; decide per-region
via measurement + cost model" user directive), this module is
**observational, not a filter**. Every pass in
:mod:`compgen.ir.payload.passes` registers for every target. The cost
model and the LLM's tool description consume this metadata to decide
whether firing a pass on a given region is expected to add value over
what inductor already produced in Phase 0.

The table below seeds qualitative biases from the plan matrix. The
discovery script
``user_perspective/scripts/10_probe_inductor_cuda_amd.py`` measures
reality on real hardware and replaces the seed values with
calibrated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Coverage = Literal["full", "partial", "none", "overlap"]
CostWeightBias = Literal["prefer", "neutral", "penalize"]

TargetFamily = Literal[
    "cuda",
    "amd",
    "cpu_inductor",
    "arm_cpu",
    "rvv_cpu",
    "qualcomm_npu",
    "qualcomm_dsp",
    "generic_npu",
]

# Multipliers applied to the generic fusion_cost_model when the cost
# model is considering whether to fire a pass. 1.0 is neutral.
_BIAS_TO_MULTIPLIER: dict[CostWeightBias, float] = {
    "prefer": 0.7,
    "neutral": 1.0,
    "penalize": 1.5,
}


@dataclass(frozen=True)
class InductorCoverage:
    """One row in the coverage table.

    Attributes:
        target_family: Target this row applies to.
        pass_name: Canonical pass name matching :attr:`PayloadPass.name`.
        coverage: Qualitative overlap with inductor on this target.
        cost_weight_bias: Directional bias applied by the cost model.
        autocomp_still_useful: If True, the pass is worth firing even
            when inductor covers the generic case — typically because
            autocomp beats the library on atypical shapes.
        measured_shapes_where_compgen_wins: Concrete shape signatures
            (populated by the discovery script) where CompGen beat
            inductor. Empty until measured.
        notes: Free-form rationale.
        basis: ``"estimated"`` (plan-seeded) or ``"measured"`` (from
            the discovery script).
    """

    target_family: TargetFamily
    pass_name: str
    coverage: Coverage
    cost_weight_bias: CostWeightBias
    autocomp_still_useful: bool = True
    measured_shapes_where_compgen_wins: tuple[str, ...] = ()
    notes: str = ""
    basis: Literal["estimated", "measured"] = "estimated"


# Seed biases from the plan's coverage matrix. Pass names match
# compgen.ir.payload.passes.<name>.name. Only CUDA + AMD + arm_cpu seeds
# are populated; other targets inherit the generic "prefer" default
# (no inductor coverage) via cost_weight_for().
_SEED: tuple[InductorCoverage, ...] = (
    # raise_special_ops
    InductorCoverage("cuda", "raise_special_ops", "partial", "prefer",
                     notes="inductor fuses softmax but does not raise to named kernel; autocomp benefits from named contracts"),
    InductorCoverage("amd", "raise_special_ops", "partial", "prefer",
                     notes="same reasoning as cuda"),
    InductorCoverage("arm_cpu", "raise_special_ops", "none", "prefer"),

    # fuse_dequant_matmul
    InductorCoverage("cuda", "fuse_dequant_matmul", "none", "prefer",
                     notes="inductor rarely fuses non-PyTorch-quant formats into a single Triton kernel"),
    InductorCoverage("amd", "fuse_dequant_matmul", "none", "prefer"),

    # propagate_transposes
    InductorCoverage("cuda", "propagate_transposes", "partial", "neutral",
                     notes="inductor folds many adjacent transposes; tool catches those that cross scheduler boundaries"),
    InductorCoverage("amd", "propagate_transposes", "partial", "neutral"),

    # lower_quantized_matmul / lower_quantized_conv
    InductorCoverage("cuda", "lower_quantized_matmul", "none", "prefer"),
    InductorCoverage("amd", "lower_quantized_matmul", "none", "prefer"),
    InductorCoverage("cuda", "lower_quantized_conv", "none", "prefer"),
    InductorCoverage("amd", "lower_quantized_conv", "none", "prefer"),

    # lower_conv_to_img2col
    InductorCoverage("cuda", "lower_conv_to_img2col", "overlap", "penalize",
                     notes="cuDNN direct conv is usually faster; pass wins on atypical shapes only"),
    InductorCoverage("amd", "lower_conv_to_img2col", "overlap", "penalize",
                     notes="MIOpen direct conv similar story"),

    # decompose_concat
    InductorCoverage("cuda", "decompose_concat", "full", "penalize",
                     autocomp_still_useful=False,
                     notes="inductor decomposes concats; CompGen pass is redundant on CUDA"),
    InductorCoverage("amd", "decompose_concat", "full", "penalize",
                     autocomp_still_useful=False),

    # demote_contraction_inputs
    InductorCoverage("cuda", "demote_contraction_inputs", "partial", "neutral",
                     notes="inductor respects dtype but target-specific accum widths still matter"),
    InductorCoverage("amd", "demote_contraction_inputs", "partial", "neutral"),

    # match_library_call
    InductorCoverage("cuda", "match_library_call", "partial", "prefer",
                     notes="inductor matches cuBLAS/cuDNN; our unified matcher also covers FlashAttention-3, custom epilogues, ONNX-RT"),
    InductorCoverage("amd", "match_library_call", "partial", "prefer",
                     notes="inductor matches rocBLAS/MIOpen; we add more"),

    # set_numerics_policy
    InductorCoverage("cuda", "set_numerics_policy", "full", "neutral",
                     notes="inductor preserves declared dtypes; we still handle fp8/int8 variants it doesn't know"),
    InductorCoverage("amd", "set_numerics_policy", "partial", "neutral"),

    # normalize_subbyte
    InductorCoverage("cuda", "normalize_subbyte", "none", "prefer",
                     notes="no inductor path for int4/int2 packing"),
    InductorCoverage("amd", "normalize_subbyte", "none", "prefer"),

    # fold_transposes_into_dots
    InductorCoverage("cuda", "fold_transposes_into_dots", "full", "penalize",
                     autocomp_still_useful=False,
                     notes="inductor folds; CompGen redundant"),
    InductorCoverage("amd", "fold_transposes_into_dots", "full", "penalize",
                     autocomp_still_useful=False),

    # plan_reduction
    InductorCoverage("cuda", "plan_reduction", "partial", "neutral",
                     notes="inductor picks one reduction strategy per op; target may prefer another"),
    InductorCoverage("amd", "plan_reduction", "partial", "neutral"),

    # fuse_softmax_to_triton
    InductorCoverage("cuda", "fuse_softmax_to_triton", "full", "penalize",
                     autocomp_still_useful=False,
                     notes="inductor fuses SDPA+softmax natively since 2.x"),
    InductorCoverage("amd", "fuse_softmax_to_triton", "partial", "prefer",
                     notes="ROCm Triton less mature; CompGen's fused version often wins"),

    # megakernel_static_schedule (Algorithm 1, Event Tensor Compiler).
    # Inductor never produces a single persistent megakernel that fuses
    # across kernel boundaries -- it preserves them.  Always prefer.
    InductorCoverage("cuda", "megakernel_static_schedule", "none", "prefer",
                     notes="inductor preserves kernel boundaries; megakernel synthesis is gap-fill"),
    InductorCoverage("amd", "megakernel_static_schedule", "none", "prefer",
                     notes="HIP-Triton has no equivalent persistent megakernel codegen"),
)


INDUCTOR_COVERAGE: dict[tuple[TargetFamily, str], InductorCoverage] = {
    (row.target_family, row.pass_name): row for row in _SEED
}


def get_coverage(pass_name: str, target_family: TargetFamily) -> InductorCoverage | None:
    """Return the coverage row or None if not seeded."""
    return INDUCTOR_COVERAGE.get((target_family, pass_name))


def cost_weight_for(pass_name: str, target_family: TargetFamily) -> float:
    """Return the cost-model multiplier for (pass, target).

    Default 1.0 (neutral) when the row is absent — meaning no
    inductor coverage so the pass is not penalized.
    """
    row = get_coverage(pass_name, target_family)
    if row is None:
        return _BIAS_TO_MULTIPLIER["prefer"] if target_family not in ("cuda", "amd") else _BIAS_TO_MULTIPLIER["neutral"]
    return _BIAS_TO_MULTIPLIER[row.cost_weight_bias]


def coverage_notes_for_llm(pass_name: str, target_family: TargetFamily) -> str:
    """Short one-liner the LLM sees as part of tool context."""
    row = get_coverage(pass_name, target_family)
    if row is None:
        return f"no inductor coverage info for ({pass_name}, {target_family}); assume non-overlapping"
    parts = [
        f"inductor coverage on {target_family}: {row.coverage}",
        f"bias: {row.cost_weight_bias}",
    ]
    if not row.autocomp_still_useful:
        parts.append("autocomp redundant")
    if row.notes:
        parts.append(row.notes)
    return "; ".join(parts)


def update_measurement(
    pass_name: str,
    target_family: TargetFamily,
    *,
    coverage: Coverage | None = None,
    cost_weight_bias: CostWeightBias | None = None,
    measured_shapes_where_compgen_wins: tuple[str, ...] = (),
    notes: str | None = None,
) -> None:
    """Replace a seed row with measured data (discovery script calls this).

    Idempotent: repeated calls overwrite. Marks the row ``basis="measured"``.
    """
    existing = INDUCTOR_COVERAGE.get((target_family, pass_name))
    if existing is None:
        # Create from scratch with whatever fields we know
        INDUCTOR_COVERAGE[(target_family, pass_name)] = InductorCoverage(
            target_family=target_family,
            pass_name=pass_name,
            coverage=coverage or "none",
            cost_weight_bias=cost_weight_bias or "prefer",
            measured_shapes_where_compgen_wins=measured_shapes_where_compgen_wins,
            notes=notes or "",
            basis="measured",
        )
        return
    INDUCTOR_COVERAGE[(target_family, pass_name)] = InductorCoverage(
        target_family=target_family,
        pass_name=pass_name,
        coverage=coverage or existing.coverage,
        cost_weight_bias=cost_weight_bias or existing.cost_weight_bias,
        autocomp_still_useful=existing.autocomp_still_useful,
        measured_shapes_where_compgen_wins=(
            measured_shapes_where_compgen_wins or existing.measured_shapes_where_compgen_wins
        ),
        notes=notes if notes is not None else existing.notes,
        basis="measured",
    )


__all__ = [
    "Coverage",
    "CostWeightBias",
    "InductorCoverage",
    "INDUCTOR_COVERAGE",
    "TargetFamily",
    "coverage_notes_for_llm",
    "cost_weight_for",
    "get_coverage",
    "update_measurement",
]
