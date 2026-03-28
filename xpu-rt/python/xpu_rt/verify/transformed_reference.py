"""Helpers for running CompGen-transformed callables.

Provides utilities to wrap a transformation function (e.g. a compiled or
optimised forward pass) so it can be fed into the verification harness
alongside an :class:`~compgen.verify.eager_reference.EagerReference`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TransformedCallable:
    """A callable produced by applying a transformation to a model.

    Attributes:
        name: Human-readable label describing the transformation.
        fn: Zero-argument callable that returns output tensor(s).
    """

    name: str
    fn: Callable[[], Any]

    def __call__(self) -> Any:
        """Invoke the underlying callable."""
        return self.fn()


def wrap_transformed(
    transform_fn: Callable[..., Any],
    example_inputs: tuple[Any, ...],
    *,
    name: str = "transformed",
) -> TransformedCallable:
    """Wrap a transformed model/function for harness consumption.

    The returned :class:`TransformedCallable` captures *example_inputs* so
    the harness can invoke it with no arguments.

    Args:
        transform_fn: A callable that accepts ``*example_inputs`` and returns
            output tensor(s).
        example_inputs: Inputs to bind into the closure.
        name: Label for the transformation.

    Returns:
        A :class:`TransformedCallable` ready for
        :func:`~compgen.verify.harness.verify_callable_against_reference`.
    """
    inputs = tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in example_inputs)

    def _run() -> Any:
        with torch.no_grad():
            return transform_fn(*inputs)

    return TransformedCallable(name=name, fn=_run)


def identity_transform(model: torch.nn.Module) -> Callable[..., Any]:
    """Return a transform function that simply runs the model as-is.

    Useful for smoke-testing the verification pipeline itself.

    Args:
        model: The module to wrap.

    Returns:
        A callable with signature ``(*inputs) -> outputs``.
    """
    model_copy = model.cpu().eval()

    def _forward(*args: Any) -> Any:
        with torch.no_grad():
            return model_copy(*args)

    return _forward


__all__ = [
    "TransformedCallable",
    "identity_transform",
    "wrap_transformed",
]
