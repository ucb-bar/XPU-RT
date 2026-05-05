"""Adapter for DeepSeek-OCR (deepseek-ai/DeepSeek-OCR).

DeepSeek-OCR's ``forward`` expects an idiosyncratic input shape::

    images = [(crop_tensor, ori_tensor)]

where ``crop_tensor`` is a stack of multiple resized image patches and
``ori_tensor`` is the original-resolution image. We construct synthetic
versions of both and feed them through the canonical forward signature.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _DeepSeekOCRForward(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        images_crop: torch.Tensor,
        images_ori: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        images_seq_mask: torch.Tensor,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            images=[(images_crop, images_ori)],
            images_spatial_crop=images_spatial_crop,
            images_seq_mask=images_seq_mask,
        )


def build(model: nn.Module, processor: Any | None) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
    wrapped = _DeepSeekOCRForward(model)
    # DeepSeek-OCR processes 4 sub-image crops at 384x384 plus a single
    # original at 1024x1024. images_spatial_crop describes the grid layout
    # (here a single 2x2 = 4-crop layout). images_seq_mask flags which
    # tokens correspond to image embeddings (none for this synthetic probe).
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    images_crop = torch.rand(4, 3, 384, 384)
    images_ori = torch.rand(1, 3, 1024, 1024)
    # crop_shape is iterated per-image inside forward; each entry must be a
    # 1-D tensor [width_crop_num, height_crop_num].
    images_spatial_crop = torch.tensor([[2, 2]], dtype=torch.long)
    images_seq_mask = torch.zeros(1, 8, dtype=torch.bool)
    return wrapped, (input_ids, images_crop, images_ori, images_spatial_crop, images_seq_mask), {}
