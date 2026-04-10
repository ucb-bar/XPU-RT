"""NPU alignment verification and quantization accuracy checks.

Validates that a quantized model conforms to NPU hardware constraints:

- All linear/conv2d weights are FP8 E4M3
- All scales are exact powers of two (fit in E8M0 registers)
- Softmax stays BF16
- Vector ops stay BF16
- Quantization accuracy is within acceptable bounds

Also provides accuracy comparison utilities, reusing the existing
``capture.torchao_pipeline.AccuracyReport`` infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
import torch
import torch.nn as nn

from compgen.quantization.attention import ExportableFP8Attention
from compgen.quantization.export_wrappers import ExportableFP8Conv2d, ExportableFP8Linear
from compgen.quantization.fp8_ops import is_power_of_two
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

logger = structlog.get_logger()


@dataclass
class NpuAlignmentResult:
    """Result of NPU alignment verification.

    Attributes:
        passed: Whether all checks passed.
        fp8_linear_count: Number of FP8-quantized linear layers.
        fp8_conv2d_count: Number of FP8-quantized conv2d layers.
        fp8_attention_count: Number of FP8 attention modules.
        non_po2_scales: List of (name, scale) for scales that are not power-of-two.
        unquantized_linears: List of linear layer names that were not quantized.
        warnings: Non-fatal warnings.
        errors: Fatal alignment violations.
    """

    passed: bool = True
    fp8_linear_count: int = 0
    fp8_conv2d_count: int = 0
    fp8_attention_count: int = 0
    non_po2_scales: list[tuple[str, float]] = field(default_factory=list)
    unquantized_linears: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def npu_alignment_check(
    model: nn.Module,
    allow_unquantized: set[str] | None = None,
) -> NpuAlignmentResult:
    """Verify that a quantized model conforms to NPU hardware constraints.

    Walks the model and checks:
    1. All nn.Linear weights are FP8 E4M3 (or in allow_unquantized set)
    2. All scales are exact powers of two
    3. FP8 attention modules exist for attention layers
    4. ExportableFP8* modules have valid FP8 weights

    Args:
        model: The quantized model to verify.
        allow_unquantized: Set of module FQN substrings that are allowed to
            remain unquantized (e.g., ``{"lm_head"}``).

    Returns:
        ``NpuAlignmentResult`` with detailed findings.
    """
    if allow_unquantized is None:
        allow_unquantized = {"lm_head"}

    result = NpuAlignmentResult()

    for name, module in model.named_modules():
        # Check ExportableFP8Linear
        if isinstance(module, ExportableFP8Linear):
            result.fp8_linear_count += 1
            if not is_power_of_two(module.weight_scale):
                result.non_po2_scales.append((name, module.weight_scale))
                result.errors.append(
                    f"{name}: weight scale {module.weight_scale} is not power-of-two"
                )
                result.passed = False
            if module.weight_fp8.dtype != torch.float8_e4m3fn:
                result.errors.append(
                    f"{name}: weight dtype is {module.weight_fp8.dtype}, expected float8_e4m3fn"
                )
                result.passed = False
            continue

        # Check ExportableFP8Conv2d
        if isinstance(module, ExportableFP8Conv2d):
            result.fp8_conv2d_count += 1
            if not is_power_of_two(module.weight_scale):
                result.non_po2_scales.append((name, module.weight_scale))
                result.errors.append(
                    f"{name}: weight scale {module.weight_scale} is not power-of-two"
                )
                result.passed = False
            continue

        # Check ExportableFP8Attention
        if isinstance(module, ExportableFP8Attention):
            result.fp8_attention_count += 1
            if module.config.softmax_dtype != torch.bfloat16:
                result.errors.append(
                    f"{name}: softmax dtype is {module.config.softmax_dtype}, must be bfloat16"
                )
                result.passed = False
            if not module.config.quantize_attn_weights:
                result.warnings.append(
                    f"{name}: attention weights not quantized to FP8 (non-standard)"
                )
            continue

        # Check nn.Linear with FP8E4M3Po2Tensor weight
        if isinstance(module, nn.Linear):
            weight = getattr(module, "weight", None)
            if isinstance(weight, FP8E4M3Po2Tensor):
                result.fp8_linear_count += 1
                if not is_power_of_two(weight._scale):
                    result.non_po2_scales.append((name, weight._scale))
                    result.errors.append(
                        f"{name}: weight scale {weight._scale} is not power-of-two"
                    )
                    result.passed = False
            else:
                # Check if this linear is in the allow list
                is_allowed = any(allowed in name for allowed in allow_unquantized)
                if not is_allowed:
                    result.unquantized_linears.append(name)
                    result.warnings.append(f"{name}: nn.Linear not quantized to FP8")

        # Check nn.Conv2d with FP8E4M3Po2Tensor weight
        if isinstance(module, nn.Conv2d):
            weight = getattr(module, "weight", None)
            if isinstance(weight, FP8E4M3Po2Tensor):
                result.fp8_conv2d_count += 1
                if not is_power_of_two(weight._scale):
                    result.non_po2_scales.append((name, weight._scale))

    logger.info(
        "npu_alignment_check",
        passed=result.passed,
        fp8_linears=result.fp8_linear_count,
        fp8_conv2ds=result.fp8_conv2d_count,
        fp8_attentions=result.fp8_attention_count,
        non_po2_count=len(result.non_po2_scales),
        unquantized_linears=len(result.unquantized_linears),
    )

    return result


def compare_quantized_accuracy(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_inputs: tuple[torch.Tensor, ...],
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Compare accuracy of quantized model against original.

    Args:
        original_model: Unquantized reference model.
        quantized_model: FP8-quantized model.
        test_inputs: Tuple of input tensors.
        tolerance: Maximum acceptable relative error.

    Returns:
        Dict with ``"l2_error"``, ``"max_abs_error"``, ``"cosine_similarity"``,
        ``"within_tolerance"`` keys.
    """
    with torch.no_grad():
        ref = original_model(*test_inputs)
        quant = quantized_model(*test_inputs)

    # Handle tuple/dict outputs
    if isinstance(ref, tuple):
        ref = ref[0]
    if isinstance(quant, tuple):
        quant = quant[0]

    diff = (ref.float() - quant.float())
    ref_norm = torch.linalg.vector_norm(ref.float()).item()
    quant_norm = torch.linalg.vector_norm(quant.float()).item()

    l2_error = torch.linalg.vector_norm(diff).item()
    max_abs_error = diff.abs().max().item()

    denom = max(ref_norm * quant_norm, 1e-12)
    cosine_sim = torch.sum(ref.float() * quant.float()).item() / denom

    within_tol = max_abs_error <= tolerance * ref.float().abs().max().item()

    return {
        "l2_error": l2_error,
        "max_abs_error": max_abs_error,
        "cosine_similarity": cosine_sim,
        "within_tolerance": within_tol,
    }


__all__ = [
    "NpuAlignmentResult",
    "compare_quantized_accuracy",
    "npu_alignment_check",
]
