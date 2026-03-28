"""Eager-mode reference execution wrapper.

Provides :class:`EagerReference` -- a thin wrapper around an ``nn.Module``
that captures example inputs and produces deterministic CPU-side reference
outputs.  Used as the ground-truth source for verification runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EagerReference:
    """Frozen wrapper holding a module plus its reference inputs and outputs.

    Attributes:
        module: The original ``nn.Module`` (moved to CPU, eval mode).
        example_inputs: Tuple of tensors used for the reference forward pass.
        reference_outputs: Cached outputs from the reference forward pass.
    """

    module: nn.Module
    example_inputs: tuple[Any, ...]
    reference_outputs: Any

    def __call__(self) -> Any:
        """Re-run the reference forward pass (CPU, no-grad)."""
        with torch.no_grad():
            return self.module(*self.example_inputs)


def build_eager_reference(
    model: nn.Module,
    example_inputs: tuple[Any, ...] | list[Any],
) -> EagerReference:
    """Build an eager-mode reference from a module and example inputs.

    The model is cloned to CPU, set to eval mode, and run once to cache
    reference outputs.

    Args:
        model: The ``nn.Module`` to wrap.
        example_inputs: Inputs forwarded to ``model(*example_inputs)``.

    Returns:
        An :class:`EagerReference` ready for use as a verification baseline.
    """
    cpu_model = model.cpu().eval()
    inputs = tuple(t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in example_inputs)
    with torch.no_grad():
        ref_out = cpu_model(*inputs)

    return EagerReference(
        module=cpu_model,
        example_inputs=inputs,
        reference_outputs=ref_out,
    )


__all__ = [
    "EagerReference",
    "build_eager_reference",
]
