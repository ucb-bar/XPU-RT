"""TorchAO quantization pipeline integration.

Integrates TorchAO's quantization and sparsity workflows into the CompGen
capture pipeline. Quantization decisions affect kernel contracts, layout
requirements, and verification (quantized paths need separate golden outputs).

Invariants:
    - Quantization config must be serializable (part of the recipe).
    - Accuracy degradation from quantization is measured and reported.
    - Quantized models produce separate golden outputs for verification.

TODO: Implement apply_quantization() with TorchAO quantize_() API.
TODO: Implement verify_quant_accuracy() comparing original vs quantized.
TODO: Support int8, int4, fp8, and structured sparsity configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        scheme: Quantization scheme (e.g., "int8_weight_only", "int4_weight_only", "fp8").
        calibration_samples: Number of calibration samples.
        group_size: Group size for grouped quantization.
        extra_args: Additional scheme-specific arguments.
    """

    scheme: str
    calibration_samples: int = 100
    group_size: int | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccuracyReport:
    """Quantization accuracy report.

    Attributes:
        l2_error: L2 norm of output difference.
        max_abs_error: Maximum absolute output difference.
        cosine_similarity: Cosine similarity between original and quantized outputs.
        within_tolerance: Whether errors are within acceptable bounds.
        tolerance: The tolerance threshold used.
    """

    l2_error: float
    max_abs_error: float
    cosine_similarity: float
    within_tolerance: bool
    tolerance: float


def apply_quantization(model: Any, config: QuantizationConfig) -> Any:
    """Apply TorchAO quantization to a model.

    Args:
        model: PyTorch nn.Module.
        config: Quantization configuration.

    Returns:
        Quantized model (modified in-place or new module).

    """
    try:
        from torchao.quantization import (
            float8_weight_only,
            int4_weight_only,
            int8_weight_only,
            quantize_,
        )
    except ImportError as exc:
        raise RuntimeError("torchao is not installed") from exc

    scheme_map = {
        "int8_weight_only": int8_weight_only,
        "int4_weight_only": int4_weight_only,
        "fp8": float8_weight_only,
    }
    factory = scheme_map.get(config.scheme)
    if factory is None:
        raise ValueError(f"Unsupported TorchAO scheme: {config.scheme}")

    quantizer = factory()
    quantize_(model, quantizer)
    return model


def verify_quant_accuracy(
    original_model: Any,
    quantized_model: Any,
    test_inputs: Any,
    tolerance: float = 0.01,
) -> AccuracyReport:
    """Verify quantization accuracy against the original model.

    """
    with torch.no_grad():
        reference = original_model(*test_inputs)
        candidate = quantized_model(*test_inputs)

    diff = (reference - candidate).float()
    ref_norm = torch.linalg.vector_norm(reference.float()).item()
    cand_norm = torch.linalg.vector_norm(candidate.float()).item()
    l2_error = torch.linalg.vector_norm(diff).item()
    max_abs_error = diff.abs().max().item()

    denom = max(ref_norm * cand_norm, 1e-12)
    cosine_similarity = torch.sum(reference.float() * candidate.float()).item() / denom
    within_tolerance = max_abs_error <= tolerance

    return AccuracyReport(
        l2_error=l2_error,
        max_abs_error=max_abs_error,
        cosine_similarity=float(cosine_similarity),
        within_tolerance=within_tolerance,
        tolerance=tolerance,
    )


__all__ = ["AccuracyReport", "QuantizationConfig", "apply_quantization", "verify_quant_accuracy"]
