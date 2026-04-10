"""Tests for torchAO FP8 E4M3 Po2 custom quantization config and tensor subclass."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from compgen.quantization.fp8_ops import FP8_E4M3_DTYPE, is_power_of_two
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

torchao = pytest.importorskip("torchao")
from torchao.quantization import quantize_  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_mlp(in_f: int = 32, hidden: int = 64, out_f: int = 16) -> nn.Module:
    """Create a simple two-layer MLP in bfloat16."""
    return nn.Sequential(
        nn.Linear(in_f, hidden, bias=True),
        nn.ReLU(),
        nn.Linear(hidden, out_f, bias=False),
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# FP8E4M3Po2Tensor basics
# ---------------------------------------------------------------------------

class TestFP8E4M3Po2Tensor:
    def test_from_float(self) -> None:
        w = torch.randn(16, 32, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        assert isinstance(fp8_w, FP8E4M3Po2Tensor)
        assert fp8_w.shape == w.shape
        assert fp8_w._quantized_data.dtype == FP8_E4M3_DTYPE
        assert is_power_of_two(fp8_w._scale)
        assert fp8_w._source_dtype == torch.bfloat16

    def test_dequantize(self) -> None:
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        w_deq = fp8_w.dequantize()
        assert w_deq.dtype == torch.bfloat16
        assert w_deq.shape == w.shape
        # Should be close to original
        assert torch.allclose(w.float(), w_deq.float(), atol=5.0)

    def test_dequantize_custom_dtype(self) -> None:
        w = torch.randn(8, 8, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        w_deq = fp8_w.dequantize(target_dtype=torch.float32)
        assert w_deq.dtype == torch.float32

    def test_transpose(self) -> None:
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        fp8_wt = fp8_w.t()
        assert isinstance(fp8_wt, FP8E4M3Po2Tensor)
        assert fp8_wt.shape == (16, 8)

    def test_detach(self) -> None:
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        fp8_d = fp8_w.detach()
        assert isinstance(fp8_d, FP8E4M3Po2Tensor)

    def test_clone(self) -> None:
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        fp8_c = fp8_w.clone()
        assert isinstance(fp8_c, FP8E4M3Po2Tensor)
        assert fp8_c._scale == fp8_w._scale

    def test_tensor_flatten_unflatten(self) -> None:
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        # Flatten
        names, metadata = fp8_w.__tensor_flatten__()
        assert "_quantized_data" in names
        # Unflatten
        tensor_dict = {name: getattr(fp8_w, name) for name in names}
        fp8_w2 = FP8E4M3Po2Tensor.__tensor_unflatten__(tensor_dict, metadata, fp8_w.shape, fp8_w.stride())
        assert isinstance(fp8_w2, FP8E4M3Po2Tensor)
        assert fp8_w2._scale == fp8_w._scale
        assert fp8_w2.shape == fp8_w.shape


# ---------------------------------------------------------------------------
# Matmul dispatch
# ---------------------------------------------------------------------------

class TestFP8Matmul:
    def test_mm_dispatch(self) -> None:
        """torch.mm with FP8 weight should produce correct-shaped result."""
        activation = torch.randn(4, 32, dtype=torch.bfloat16)
        weight = FP8E4M3Po2Tensor.from_float(
            torch.randn(16, 32, dtype=torch.bfloat16)
        )
        result = torch.mm(activation, weight.t())
        assert result.shape == (4, 16)
        assert result.dtype == torch.bfloat16

    def test_linear_dispatch(self) -> None:
        """F.linear with FP8 weight should work."""
        activation = torch.randn(4, 32, dtype=torch.bfloat16)
        weight = FP8E4M3Po2Tensor.from_float(
            torch.randn(16, 32, dtype=torch.bfloat16)
        )
        bias = torch.randn(16, dtype=torch.bfloat16)
        result = torch.nn.functional.linear(activation, weight, bias)
        assert result.shape == (4, 16)

    def test_linear_module_forward(self) -> None:
        """nn.Linear with FP8 weight should produce correct output."""
        layer = nn.Linear(32, 16, bias=True, dtype=torch.bfloat16)
        layer.weight = nn.Parameter(
            FP8E4M3Po2Tensor.from_float(layer.weight), requires_grad=False
        )
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (4, 16)
        assert out.dtype == torch.bfloat16

    def test_matmul_accuracy(self) -> None:
        """FP8 matmul should be close to BF16 matmul."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, dtype=torch.bfloat16)
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        b = torch.randn(32, dtype=torch.bfloat16)

        ref = torch.nn.functional.linear(x, w, b)

        fp8_w = FP8E4M3Po2Tensor.from_float(w)
        fp8_out = torch.nn.functional.linear(x, fp8_w, b)

        # FP8 should be within reasonable tolerance
        rel_err = (ref.float() - fp8_out.float()).abs() / (ref.float().abs() + 1e-8)
        assert rel_err.mean() < 0.2, f"Mean relative error {rel_err.mean():.4f}"


# ---------------------------------------------------------------------------
# quantize_() integration
# ---------------------------------------------------------------------------

class TestQuantizeIntegration:
    def test_quantize_replaces_weights(self) -> None:
        """quantize_() should replace nn.Linear weights with FP8E4M3Po2Tensor."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        config = FP8E4M3Po2Config()
        quantize_(model, config)

        # Check that linear weights are now FP8
        assert isinstance(model[0].weight, FP8E4M3Po2Tensor)
        assert isinstance(model[2].weight, FP8E4M3Po2Tensor)

    def test_quantized_model_forward(self) -> None:
        """Quantized model should produce correct-shaped output."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        quantize_(model, config=FP8E4M3Po2Config())

        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = model(x)
        assert out.shape == (4, 16)
        assert out.dtype == torch.bfloat16

    def test_quantized_scales_are_po2(self) -> None:
        """All scales in quantized model should be powers of two."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        quantize_(model, config=FP8E4M3Po2Config())

        for name, param in model.named_parameters():
            if isinstance(param, FP8E4M3Po2Tensor):
                assert is_power_of_two(param._scale), (
                    f"Scale for {name} is {param._scale}, not a power of two"
                )

    def test_quantize_with_filter_fn(self) -> None:
        """quantize_() with filter should only quantize matching layers."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        # Only quantize first linear
        quantize_(
            model,
            config=FP8E4M3Po2Config(),
            filter_fn=lambda mod, fqn: fqn == "0",
        )

        assert isinstance(model[0].weight, FP8E4M3Po2Tensor)
        assert not isinstance(model[2].weight, FP8E4M3Po2Tensor)

    def test_idempotent_quantize(self) -> None:
        """Quantizing an already-quantized model should be a no-op."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        config = FP8E4M3Po2Config()
        quantize_(model, config)
        scale_before = model[0].weight._scale

        # Quantize again
        quantize_(model, config)
        scale_after = model[0].weight._scale
        assert scale_before == scale_after

    def test_extra_repr_updated(self) -> None:
        """Module repr should show FP8 quantization info."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config

        model = _simple_mlp()
        quantize_(model, config=FP8E4M3Po2Config())
        repr_str = repr(model[0])
        assert "float8_e4m3fn" in repr_str
