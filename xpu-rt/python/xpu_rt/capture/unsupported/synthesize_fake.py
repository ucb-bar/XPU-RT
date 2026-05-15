"""Fake/meta-kernel synthesis for operators missing Meta dispatch keys.

When an operator has no registered Meta kernel, ``torch.export`` and
shape-propagation passes cannot reason about its output shapes.  This
module synthesises a lightweight *fake kernel* from the example
input/output shapes recorded in the operator dossier.  The fake kernel
returns empty tensors with the correct shapes and dtypes, which is
sufficient for the export trace to proceed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
import torch

from compgen.capture.unsupported.introspect import ExampleTensorInfo, UnsupportedOpDossier

log = structlog.get_logger()

# Map from dtype name strings to torch dtypes.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


@dataclass(frozen=True)
class SynthesizedFakeKernel:
    """A synthesized fake kernel for shape propagation.

    Attributes:
        target: Fully qualified op target (e.g. ``aten.custom_op.default``).
        output_shape: Expected output shape.
        output_dtype: Expected output dtype name.
        fake_fn: Callable that accepts any args/kwargs and returns an
            empty tensor with the correct shape and dtype on the ``meta``
            device.
    """

    target: str
    output_shape: tuple[int, ...]
    output_dtype: str
    fake_fn: Callable[..., torch.Tensor]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a dtype name string to a ``torch.dtype``."""
    return _DTYPE_MAP.get(dtype_name, torch.float32)


def _build_fake_fn(
    output_shape: tuple[int, ...],
    output_dtype: torch.dtype,
) -> Callable[..., torch.Tensor]:
    """Build a fake kernel callable that returns an empty meta tensor.

    The returned callable ignores its arguments and produces an empty
    tensor with the recorded shape and dtype on whichever device the
    first tensor argument lives on (falling back to ``meta``).
    """

    def _fake(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Try to infer device from the first tensor argument.
        device: str | torch.device = "meta"
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        return torch.empty(output_shape, dtype=output_dtype, device=device)

    return _fake


def synthesize_fake_kernel(
    target: str,
    dossier: UnsupportedOpDossier,
) -> SynthesizedFakeKernel | None:
    """Synthesize a fake kernel for shape propagation from the dossier.

    Uses the ``example_output`` recorded in the dossier to build a
    minimal fake that returns an empty tensor with the correct shape and
    dtype.  Returns ``None`` when the dossier lacks sufficient shape
    information to produce a reliable fake.

    Args:
        target: Fully qualified op target.
        dossier: Introspection dossier for the operator.

    Returns:
        A ``SynthesizedFakeKernel`` or ``None`` if shape info is
        insufficient.
    """
    example_output: ExampleTensorInfo | None = dossier.example_output

    if example_output is None:
        log.debug("synthesize_fake_kernel: no example output for %s, cannot synthesize", target)
        return None

    if not example_output.shape:
        log.debug("synthesize_fake_kernel: empty output shape for %s, cannot synthesize", target)
        return None

    # Reject outputs with any zero-sized dimensions as they are likely
    # sentinel values rather than real shapes.
    if any(d <= 0 for d in example_output.shape):
        log.debug("synthesize_fake_kernel: non-positive dimensions in output shape for %s", target)
        return None

    output_dtype = _resolve_dtype(example_output.dtype)
    output_shape = example_output.shape

    fake_fn = _build_fake_fn(output_shape, output_dtype)

    log.debug(
        "synthesize_fake_kernel: synthesized fake for %s with shape=%s dtype=%s",
        target,
        output_shape,
        example_output.dtype,
    )

    return SynthesizedFakeKernel(
        target=target,
        output_shape=output_shape,
        output_dtype=example_output.dtype,
        fake_fn=fake_fn,
    )


__all__ = ["SynthesizedFakeKernel", "synthesize_fake_kernel"]
