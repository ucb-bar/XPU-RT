"""SmolVLA per-component quantization recipe for NPU deployment.

Defines which model components get FP8 quantization and with what settings,
adapted from pi0-quant's ``model_patcher.py`` component tagging system.

SmolVLA architecture components:

    **VISION** -- SigLIP ViT (412M params).  Independent forward pass.
    Includes Conv2d patch embedding + transformer layers.

    **LANGUAGE** -- Gemma 2.5B language model.  Co-attention coupled with
    action expert inside the PaliGemma joint transformer.

    **ACTION_EXPERT** -- Gemma 300M action-specialized transformer.
    Co-attention coupled with the language model.

    **ACTION_HEAD** -- Thin MLP projections at the Pi0 root
    (action_in_proj, action_out_proj, state_proj, time_mlp_*).

The default NPU recipe quantizes all matmuls to FP8 E4M3 (po2 scaling)
and keeps vector ops in BF16, matching the NPU hardware model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import structlog
import torch
import torch.nn as nn

from compgen.quantization.attention import (
    ExportableFP8Attention,
    FP8AttentionConfig,
    replace_sdpa_with_fp8_attention,
)
from compgen.quantization.fp8_config import FP8E4M3Po2Config
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------

class SmolVLAComponent(str, Enum):
    """Architectural component of the SmolVLA model."""

    VISION = "vision"
    LANGUAGE = "language"
    ACTION_EXPERT = "action_expert"
    ACTION_HEAD = "action_head"
    UNKNOWN = "unknown"


# Component tagging rules -- checked in order, first match wins.
# Adapted from pi0-quant model_patcher.py lines 89-106.
_COMPONENT_RULES: list[tuple[str, SmolVLAComponent]] = [
    # Action-head projections live directly on the model root
    ("action_in_proj", SmolVLAComponent.ACTION_HEAD),
    ("action_out_proj", SmolVLAComponent.ACTION_HEAD),
    ("action_time_mlp_in", SmolVLAComponent.ACTION_HEAD),
    ("action_time_mlp_out", SmolVLAComponent.ACTION_HEAD),
    ("state_proj", SmolVLAComponent.ACTION_HEAD),
    ("time_mlp_in", SmolVLAComponent.ACTION_HEAD),
    ("time_mlp_out", SmolVLAComponent.ACTION_HEAD),
    # Gemma action expert (separate transformer from language model)
    ("gemma_expert", SmolVLAComponent.ACTION_EXPERT),
    # Vision tower (SigLIP ViT)
    ("vision_tower", SmolVLAComponent.VISION),
    ("vision_model", SmolVLAComponent.VISION),
    # Language model (Gemma inside PaliGemma)
    ("language_model", SmolVLAComponent.LANGUAGE),
    ("paligemma", SmolVLAComponent.LANGUAGE),
]


def infer_component(module_path: str) -> SmolVLAComponent:
    """Classify a module path into a SmolVLA component.

    Uses substring matching with the same priority rules as pi0-quant.

    Args:
        module_path: Dot-separated module path (e.g.,
            ``"paligemma.vision_tower.encoder.layers.0.self_attn.q_proj"``).

    Returns:
        The SmolVLA component this module belongs to.
    """
    for substring, component in _COMPONENT_RULES:
        if substring in module_path:
            return component
    return SmolVLAComponent.UNKNOWN


# ---------------------------------------------------------------------------
# Per-component quantization config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentQuantConfig:
    """Quantization configuration for a single SmolVLA component.

    Attributes:
        enabled: Whether this component should be quantized at all.
        quantize_linear: Quantize nn.Linear weights to FP8.
        quantize_conv2d: Quantize nn.Conv2d weights to FP8.
        quantize_attention: Replace SDPA with FP8 attention.
        vector_dtype: Dtype for vector ops (always BF16 for NPU).
    """

    enabled: bool = True
    quantize_linear: bool = True
    quantize_conv2d: bool = True
    quantize_attention: bool = True
    vector_dtype: torch.dtype = torch.bfloat16


@dataclass
class SmolVLAQuantRecipe:
    """Complete quantization recipe for the SmolVLA model.

    Attributes:
        vision: Config for the vision tower (SigLIP ViT).
        language: Config for the language model (Gemma 2.5B).
        action_expert: Config for the action expert (Gemma 300M).
        action_head: Config for the action head MLPs.
        scaling_mode: FP8 scaling mode (``"po2"`` for NPU).
        skip_lm_head: Whether to skip lm_head (unused in action inference).
    """

    vision: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    language: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    action_expert: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    action_head: ComponentQuantConfig = field(default_factory=ComponentQuantConfig)
    scaling_mode: str = "po2"
    skip_lm_head: bool = True


def default_npu_recipe() -> SmolVLAQuantRecipe:
    """Return the default SmolVLA quantization recipe for NPU deployment.

    All components are quantized with FP8 E4M3 po2 scaling for matmuls,
    BF16 for vector ops.  This matches the NPU hardware model:
    - MXU: FP8 inputs, BF16 accumulation
    - VPU: BF16 vector ops
    - Softmax: always BF16
    """
    return SmolVLAQuantRecipe(
        vision=ComponentQuantConfig(enabled=True),
        language=ComponentQuantConfig(enabled=True),
        action_expert=ComponentQuantConfig(enabled=True),
        action_head=ComponentQuantConfig(
            enabled=True,
            quantize_conv2d=False,  # No conv2d in action head
            quantize_attention=False,  # No attention in action head
        ),
        scaling_mode="po2",
        skip_lm_head=True,
    )


# ---------------------------------------------------------------------------
# Apply quantization recipe
# ---------------------------------------------------------------------------

def _make_component_filter(
    recipe: SmolVLAQuantRecipe,
) -> Callable[[nn.Module, str], bool]:
    """Build a filter function for ``quantize_()`` based on the recipe.

    Returns a function that returns True for ``nn.Linear`` modules that
    should be quantized according to the recipe.
    """
    component_configs = {
        SmolVLAComponent.VISION: recipe.vision,
        SmolVLAComponent.LANGUAGE: recipe.language,
        SmolVLAComponent.ACTION_EXPERT: recipe.action_expert,
        SmolVLAComponent.ACTION_HEAD: recipe.action_head,
    }

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # Skip lm_head
        if recipe.skip_lm_head and "lm_head" in fqn:
            return False
        component = infer_component(fqn)
        config = component_configs.get(component)
        if config is None or not config.enabled or not config.quantize_linear:
            return False
        return True

    return filter_fn


def _quantize_conv2d_modules(
    model: nn.Module,
    recipe: SmolVLAQuantRecipe,
) -> list[str]:
    """Replace Conv2d weights with FP8E4M3Po2Tensor in-place.

    Only targets Conv2d modules in components where ``quantize_conv2d=True``.

    Returns:
        List of quantized module paths.
    """
    component_configs = {
        SmolVLAComponent.VISION: recipe.vision,
        SmolVLAComponent.LANGUAGE: recipe.language,
        SmolVLAComponent.ACTION_EXPERT: recipe.action_expert,
        SmolVLAComponent.ACTION_HEAD: recipe.action_head,
    }
    quantized: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        component = infer_component(name)
        config = component_configs.get(component)
        if config is None or not config.enabled or not config.quantize_conv2d:
            continue
        # Quantize weight
        fp8_weight = FP8E4M3Po2Tensor.from_float(module.weight)
        module.weight = nn.Parameter(fp8_weight, requires_grad=False)
        quantized.append(name)
        logger.debug("quantized_conv2d", module=name, component=component.value)

    return quantized


def apply_smolvla_quantization(
    model: nn.Module,
    recipe: SmolVLAQuantRecipe | None = None,
) -> nn.Module:
    """Apply the SmolVLA quantization recipe to a model.

    This is the main entry point for quantizing a SmolVLA model for NPU
    deployment.  It:

    1. Quantizes nn.Linear weights to FP8 using ``quantize_()``
    2. Quantizes nn.Conv2d weights to FP8 (SigLIP patch embedding)
    3. Adds ExportableFP8Attention modules for SDPA replacement

    Args:
        model: The SmolVLA model (any nn.Module).
        recipe: Quantization recipe.  Defaults to ``default_npu_recipe()``.

    Returns:
        The quantized model (modified in-place).
    """
    if recipe is None:
        recipe = default_npu_recipe()

    # Step 1: Quantize nn.Linear weights via torchAO
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "torchao is required for SmolVLA quantization. "
            "Install with: pip install torchao>=0.16"
        ) from exc

    config = FP8E4M3Po2Config(scaling_mode=recipe.scaling_mode)
    filter_fn = _make_component_filter(recipe)
    quantize_(model, config, filter_fn=filter_fn)

    # Count quantized linears
    n_linear = sum(
        1 for _, p in model.named_parameters()
        if isinstance(p, FP8E4M3Po2Tensor)
    )
    logger.info("quantized_linears", count=n_linear, scaling_mode=recipe.scaling_mode)

    # Step 2: Quantize Conv2d weights
    conv_quantized = _quantize_conv2d_modules(model, recipe)
    if conv_quantized:
        logger.info("quantized_conv2d_modules", count=len(conv_quantized))

    # Step 3: Add FP8 attention modules
    attn_config = FP8AttentionConfig(
        quantize_qkv=True,
        quantize_attn_weights=True,
        softmax_dtype=torch.bfloat16,
    )
    attn_patched = replace_sdpa_with_fp8_attention(model, attn_config)
    if attn_patched:
        logger.info("patched_attention_modules", count=len(attn_patched))

    return model


__all__ = [
    "ComponentQuantConfig",
    "SmolVLAComponent",
    "SmolVLAQuantRecipe",
    "apply_smolvla_quantization",
    "default_npu_recipe",
    "infer_component",
]
