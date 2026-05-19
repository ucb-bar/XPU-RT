"""Adapter for Moondream2 (vikhyatk/moondream2).

Moondream's HfMoondream class exposes its real forward through the
``encode_image(pil_image)`` method (which routes to the underlying
MoondreamModel.encode_image). Standard ``model(input_ids=..., pixel_values=...)``
is not the canonical call.

For admission, we wrap the model into a small ``nn.Module`` whose
``forward(pixel_values)`` calls ``encode_image`` on a single PIL image
constructed from the input tensor. This gives the probe a
torch-friendly forward to trace / dynamo / compile.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _MoondreamForward(nn.Module):
    """Standard-shape forward over Moondream2's ``encode_image`` path."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> Any:
        # Moondream encode_image accepts a PIL.Image; we convert from tensor.
        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if pixel_values.dim() == 4:
            pixel_values = pixel_values[0]
        arr = (pixel_values.detach().to(torch.float32).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        if arr.shape[0] in (1, 3) and arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        img = Image.fromarray(arr)
        return self.model.encode_image(img)


def build(model: nn.Module, processor: Any | None) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
    wrapped = _MoondreamForward(model)
    pixel = torch.rand(1, 3, 384, 384)
    return wrapped, (pixel,), {}
