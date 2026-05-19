"""Adapter for Microsoft Florence-2.

Florence-2 is an encoder-decoder VLM: forward expects ``pixel_values``,
``input_ids`` (encoder prompt like ``<OD>``), and ``decoder_input_ids``
(decoder start). The standard VLM input cascade only produces
``pixel_values + input_ids`` and Florence-2's forward then complains that
``decoder_input_ids`` is missing. The adapter supplies both.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _Florence2Forward(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> Any:
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )


def build(model: nn.Module, processor: Any | None) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
    wrapped = _Florence2Forward(model)
    # Florence-2 expects 224x224 or 768x768 pixel_values depending on variant;
    # base uses ~768. Encoder prompt = task token (e.g. <OD>). Decoder start
    # = BOS or task token.
    pixel_values = torch.rand(1, 3, 768, 768)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
    return wrapped, (pixel_values, input_ids, decoder_input_ids), {}
