"""FP8 E4M3 tensor subclass with power-of-two scaling for torchAO.

Implements a ``torch.Tensor`` subclass that stores FP8 E4M3 quantized data
alongside a per-tensor po2 scale factor.  The subclass intercepts matmul and
linear dispatches to implement FP8 input quantization + BF16 accumulation,
matching the NPU's ``vmatmul`` (FP8 weight slots, BF16 accumulators).

Inherits from ``TorchAOBaseTensor`` for automatic ``__tensor_flatten__``,
``__tensor_unflatten__``, ``_apply_fn_to_data``, and ``__repr__``.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

from compgen.quantization.fp8_ops import (
    FP8_E4M3_DTYPE,
    dequantize_fp8_e4m3,
    quantize_fp8_e4m3_po2,
)

# ATen ops we intercept for FP8 matmul dispatch
aten = torch.ops.aten


class FP8E4M3Po2Tensor(TorchAOBaseTensor):
    """Tensor subclass wrapping FP8 E4M3 quantized data with a po2 scale.

    This tensor stores weights in ``torch.float8_e4m3fn`` format alongside a
    scalar power-of-two scale factor.  When used as a weight in
    ``F.linear`` / ``torch.mm`` / ``torch.addmm``, it dynamically quantizes
    the activation input to FP8, dequantizes both to BF16, performs the matmul
    in BF16 (matching NPU accumulation), and returns the result.

    Attributes:
        tensor_data_names: Names of tensor-valued attributes.
        tensor_attribute_names: Names of non-tensor metadata attributes.
    """

    tensor_data_names = ["_quantized_data"]
    tensor_attribute_names = ["_scale", "_source_dtype"]

    @staticmethod
    def __new__(
        cls,
        quantized_data: torch.Tensor,
        scale: float,
        source_dtype: torch.dtype = torch.bfloat16,
    ) -> FP8E4M3Po2Tensor:
        """Create a new FP8E4M3Po2Tensor wrapper.

        Args:
            quantized_data: The FP8 E4M3 quantized weight tensor.
            scale: Per-tensor po2 scale factor.
            source_dtype: Original dtype before quantization (for dequantize).
        """
        return torch.Tensor._make_wrapper_subclass(
            cls,
            quantized_data.shape,
            dtype=source_dtype,
            device=quantized_data.device,
        )

    def __init__(
        self,
        quantized_data: torch.Tensor,
        scale: float,
        source_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._quantized_data = quantized_data
        self._scale = scale
        self._source_dtype = source_dtype

    @classmethod
    def from_float(
        cls,
        weight: torch.Tensor,
    ) -> FP8E4M3Po2Tensor:
        """Quantize a floating-point weight tensor to FP8 E4M3 with po2 scaling.

        Args:
            weight: Weight tensor in any float dtype.

        Returns:
            A new ``FP8E4M3Po2Tensor`` wrapping the quantized weight.
        """
        source_dtype = weight.dtype
        fp8_data, scale = quantize_fp8_e4m3_po2(weight)
        return cls(fp8_data, scale, source_dtype)

    def dequantize(self, target_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize back to a standard tensor.

        Args:
            target_dtype: Output dtype.  Defaults to ``self._source_dtype``.

        Returns:
            Plain ``torch.Tensor`` with dequantized values.
        """
        out_dtype = target_dtype if target_dtype is not None else self._source_dtype
        return dequantize_fp8_e4m3(self._quantized_data, self._scale, out_dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> object:  # noqa: N805
        """Dispatch ATen operations on FP8E4M3Po2Tensor.

        Intercepts ``aten.linear``, ``aten.mm``, ``aten.addmm`` to implement
        FP8 matmul.  For other ops, falls back to dequantizing.
        """
        if kwargs is None:
            kwargs = {}

        # --- mm: activation @ weight^T (weight is this tensor) ---------------
        if func is aten.mm.default:
            return _fp8_mm(args[0], args[1])

        # --- addmm: bias + activation @ weight^T ----------------------------
        if func is aten.addmm.default:
            bias, activation, weight = args[0], args[1], args[2]
            result = _fp8_mm(activation, weight)
            if isinstance(bias, FP8E4M3Po2Tensor):
                bias = bias.dequantize()
            return result + bias

        # --- linear: F.linear(input, weight, bias) ---------------------------
        if func is aten.linear.default:
            activation, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            result = _fp8_mm(activation, weight.t())
            if bias is not None:
                if isinstance(bias, FP8E4M3Po2Tensor):
                    bias = bias.dequantize()
                result = result + bias
            return result

        # --- transpose: propagate subclass -----------------------------------
        if func is aten.t.default:
            (tensor,) = args
            new_data = tensor._quantized_data.t()
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- detach: propagate subclass --------------------------------------
        if func is aten.detach.default:
            (tensor,) = args
            new_data = tensor._quantized_data.detach()
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- clone: propagate subclass ---------------------------------------
        if func is aten.clone.default:
            (tensor,) = args
            new_data = tensor._quantized_data.clone()
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- slice: propagate subclass (needed for model surgery) ------------
        if func is aten.slice.Tensor:
            tensor = args[0]
            new_data = func(tensor._quantized_data, *args[1:], **kwargs)
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- view / reshape: propagate subclass ------------------------------
        if func in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            tensor = args[0]
            new_data = func(tensor._quantized_data, *args[1:], **kwargs)
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- expand: propagate subclass --------------------------------------
        if func is aten.expand.default:
            tensor = args[0]
            new_data = tensor._quantized_data.expand(*args[1:], **kwargs)
            return cls(new_data, tensor._scale, tensor._source_dtype)

        # --- Fallback: dequantize all FP8 tensors and run in source dtype ----
        def unwrap(t: object) -> object:
            if isinstance(t, FP8E4M3Po2Tensor):
                return t.dequantize()
            return t

        new_args = torch.utils._pytree.tree_map(unwrap, args)
        new_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
        return func(*new_args, **new_kwargs)


def _fp8_mm(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Perform FP8-aware matmul: quantize activation, dequantize both, matmul in BF16.

    This simulates the NPU's MXU pipeline:
    1. Activation dynamically quantized to FP8 E4M3 (po2 scale)
    2. Both operands dequantized to BF16
    3. Matmul in BF16 (matching NPU accumulator precision)

    Args:
        activation: Input activation (any dtype).  May already be FP8E4M3Po2Tensor.
        weight: Weight tensor.  Expected to be FP8E4M3Po2Tensor.

    Returns:
        BF16 result tensor.
    """
    # Quantize activation to FP8 then dequantize to BF16 (bakes in FP8 noise)
    if isinstance(activation, FP8E4M3Po2Tensor):
        act_bf16 = activation.dequantize(torch.bfloat16)
    else:
        act_fp8, act_scale = quantize_fp8_e4m3_po2(activation)
        act_bf16 = dequantize_fp8_e4m3(act_fp8, act_scale, torch.bfloat16)

    # Dequantize weight to BF16
    if isinstance(weight, FP8E4M3Po2Tensor):
        wt_bf16 = weight.dequantize(torch.bfloat16)
    else:
        wt_fp8, wt_scale = quantize_fp8_e4m3_po2(weight)
        wt_bf16 = dequantize_fp8_e4m3(wt_fp8, wt_scale, torch.bfloat16)

    # Matmul in BF16 (NPU accumulation precision)
    return torch.mm(act_bf16, wt_bf16)


__all__ = ["FP8E4M3Po2Tensor"]
