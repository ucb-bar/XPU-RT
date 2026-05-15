"""Tests for SmolVLA per-component quantization recipe."""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor
from compgen.quantization.smolvla_recipe import (
    ComponentQuantConfig,
    SmolVLAComponent,
    SmolVLAQuantRecipe,
    apply_smolvla_quantization,
    default_npu_recipe,
    infer_component,
)

# ---------------------------------------------------------------------------
# Component inference
# ---------------------------------------------------------------------------


class TestInferComponent:
    def test_vision_tower(self) -> None:
        assert infer_component("paligemma.vision_tower.encoder.layers.0.self_attn.q_proj") == SmolVLAComponent.VISION

    def test_vision_model(self) -> None:
        assert infer_component("vision_model.encoder.layers.3.mlp.fc1") == SmolVLAComponent.VISION

    def test_language_model(self) -> None:
        assert infer_component("language_model.layers.0.mlp.gate_proj") == SmolVLAComponent.LANGUAGE

    def test_paligemma_fallback(self) -> None:
        assert infer_component("paligemma.some_other_module.linear") == SmolVLAComponent.LANGUAGE

    def test_gemma_expert(self) -> None:
        assert infer_component("gemma_expert.layers.2.self_attn.v_proj") == SmolVLAComponent.ACTION_EXPERT

    def test_action_in_proj(self) -> None:
        assert infer_component("action_in_proj") == SmolVLAComponent.ACTION_HEAD

    def test_action_out_proj(self) -> None:
        assert infer_component("action_out_proj") == SmolVLAComponent.ACTION_HEAD

    def test_state_proj(self) -> None:
        assert infer_component("state_proj") == SmolVLAComponent.ACTION_HEAD

    def test_time_mlp(self) -> None:
        assert infer_component("time_mlp_in") == SmolVLAComponent.ACTION_HEAD
        assert infer_component("time_mlp_out") == SmolVLAComponent.ACTION_HEAD

    def test_action_time_mlp(self) -> None:
        assert infer_component("action_time_mlp_in") == SmolVLAComponent.ACTION_HEAD
        assert infer_component("action_time_mlp_out") == SmolVLAComponent.ACTION_HEAD

    def test_unknown(self) -> None:
        assert infer_component("some_random_module") == SmolVLAComponent.UNKNOWN

    def test_priority_order(self) -> None:
        """action_in_proj should match ACTION_HEAD even if it contains 'paligemma'."""
        # action_in_proj is checked before paligemma in _COMPONENT_RULES
        assert infer_component("action_in_proj") == SmolVLAComponent.ACTION_HEAD

    def test_nested_vision_tower(self) -> None:
        """Deep nested paths should still match."""
        path = "model.paligemma_with_expert.paligemma.vision_tower.encoder.layers.11.self_attn.out_proj"
        assert infer_component(path) == SmolVLAComponent.VISION


# ---------------------------------------------------------------------------
# Default recipe
# ---------------------------------------------------------------------------


class TestDefaultRecipe:
    def test_all_components_enabled(self) -> None:
        recipe = default_npu_recipe()
        assert recipe.vision.enabled is True
        assert recipe.language.enabled is True
        assert recipe.action_expert.enabled is True
        assert recipe.action_head.enabled is True

    def test_scaling_mode_po2(self) -> None:
        assert default_npu_recipe().scaling_mode == "po2"

    def test_skip_lm_head(self) -> None:
        assert default_npu_recipe().skip_lm_head is True

    def test_action_head_no_conv2d(self) -> None:
        recipe = default_npu_recipe()
        assert recipe.action_head.quantize_conv2d is False

    def test_action_head_no_attention(self) -> None:
        recipe = default_npu_recipe()
        assert recipe.action_head.quantize_attention is False

    def test_vector_ops_bf16(self) -> None:
        recipe = default_npu_recipe()
        for comp in [recipe.vision, recipe.language, recipe.action_expert, recipe.action_head]:
            assert comp.vector_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Apply recipe to mock model
# ---------------------------------------------------------------------------


class _MockSmolVLA(nn.Module):
    """Minimal mock of SmolVLA architecture for testing."""

    def __init__(self) -> None:
        super().__init__()
        # Vision tower
        self.vision_tower = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Patch embedding
            nn.Linear(16, 32),
        )
        # Language model
        self.language_model = nn.Sequential(
            nn.Linear(32, 64),
            nn.Linear(64, 32),
        )
        # Action expert
        self.gemma_expert = nn.Sequential(
            nn.Linear(32, 64),
            nn.Linear(64, 32),
        )
        # Action head
        self.action_in_proj = nn.Linear(32, 16)
        self.action_out_proj = nn.Linear(16, 8)
        # lm_head (should be skipped)
        self.lm_head = nn.Linear(32, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestApplyQuantization:
    def test_linear_weights_become_fp8(self) -> None:
        model = _MockSmolVLA().to(torch.bfloat16)
        apply_smolvla_quantization(model)

        # Vision linear should be quantized
        assert isinstance(model.vision_tower[1].weight, FP8E4M3Po2Tensor)
        # Language linears should be quantized
        assert isinstance(model.language_model[0].weight, FP8E4M3Po2Tensor)
        assert isinstance(model.language_model[1].weight, FP8E4M3Po2Tensor)
        # Expert linears should be quantized
        assert isinstance(model.gemma_expert[0].weight, FP8E4M3Po2Tensor)
        # Action head should be quantized
        assert isinstance(model.action_in_proj.weight, FP8E4M3Po2Tensor)
        assert isinstance(model.action_out_proj.weight, FP8E4M3Po2Tensor)

    def test_lm_head_skipped(self) -> None:
        model = _MockSmolVLA().to(torch.bfloat16)
        apply_smolvla_quantization(model)
        assert not isinstance(model.lm_head.weight, FP8E4M3Po2Tensor)

    def test_conv2d_vision_quantized(self) -> None:
        model = _MockSmolVLA().to(torch.bfloat16)
        apply_smolvla_quantization(model)
        assert isinstance(model.vision_tower[0].weight, FP8E4M3Po2Tensor)

    def test_custom_recipe_disable_vision(self) -> None:
        model = _MockSmolVLA().to(torch.bfloat16)
        recipe = SmolVLAQuantRecipe(
            vision=ComponentQuantConfig(enabled=False),
        )
        apply_smolvla_quantization(model, recipe)
        # Vision should NOT be quantized
        assert not isinstance(model.vision_tower[1].weight, FP8E4M3Po2Tensor)
        # Language should still be quantized
        assert isinstance(model.language_model[0].weight, FP8E4M3Po2Tensor)

    def test_forward_still_works(self) -> None:
        model = _MockSmolVLA().to(torch.bfloat16)
        apply_smolvla_quantization(model)
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        out = model(x)
        assert out.shape == (2, 32)
