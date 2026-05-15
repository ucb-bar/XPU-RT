"""Export-side decomposition synthesis for unsupported operators.

Maintains a conservative allow-list of ATen operators with known
decompositions into simpler ATen ops.  For ops on the allow-list the
function returns a callable decomposition; for everything else it
returns ``None`` so the pipeline falls through to other recovery
strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
import torch

from compgen.capture.unsupported.introspect import UnsupportedOpDossier

log = structlog.get_logger()


@dataclass(frozen=True)
class SynthesizedDecomposition:
    """A synthesized export-level decomposition for one operator target.

    Attributes:
        target: Fully qualified op target (e.g. ``aten.addmm.default``).
        description: Human-readable description of the decomposition.
        decomp_fn: Callable that takes the same args as the original op
            and returns the decomposed result.
    """

    target: str
    description: str
    decomp_fn: Callable[..., Any]


def _decomp_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, **_: Any) -> torch.Tensor:
    """addmm(bias, mat1, mat2) -> mm(mat1, mat2) + bias."""
    return torch.mm(mat1, mat2) + bias


def _decomp_baddbmm(
    self: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    **_: Any,
) -> torch.Tensor:
    """baddbmm(self, batch1, batch2) -> beta * self + alpha * bmm(batch1, batch2)."""
    return beta * self + alpha * torch.bmm(batch1, batch2)


def _decomp_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **_: Any,
) -> torch.Tensor:
    """linear(input, weight, bias) -> mm(input, weight^T) + bias."""
    result = torch.mm(input, weight.t()) if input.dim() == 2 else torch.matmul(input, weight.t())
    if bias is not None:
        result = result + bias
    return result


def _decomp_silu(input: torch.Tensor, **_: Any) -> torch.Tensor:
    """silu(input) -> input * sigmoid(input)."""
    return input * torch.sigmoid(input)


def _decomp_mish(input: torch.Tensor, **_: Any) -> torch.Tensor:
    """mish(input) -> input * tanh(softplus(input))."""
    return input * torch.tanh(torch.nn.functional.softplus(input))


def _decomp_hardswish(input: torch.Tensor, **_: Any) -> torch.Tensor:
    """hardswish(input) -> input * relu6(input + 3) / 6."""
    return input * torch.clamp(input + 3.0, min=0.0, max=6.0) / 6.0


def _decomp_leaky_relu(input: torch.Tensor, negative_slope: float = 0.01, **_: Any) -> torch.Tensor:
    """leaky_relu(input) -> max(0, input) + negative_slope * min(0, input)."""
    return torch.where(input >= 0, input, input * negative_slope)


# Conservative allow-list: only ops with well-understood, shape-preserving
# decompositions into simpler ATen primitives.
_DECOMPOSITION_ALLOW_LIST: dict[str, tuple[Callable[..., Any], str]] = {
    "aten.addmm.default": (_decomp_addmm, "mm(mat1, mat2) + bias"),
    "aten.baddbmm.default": (_decomp_baddbmm, "beta * self + alpha * bmm(batch1, batch2)"),
    "aten.linear.default": (_decomp_linear, "matmul(input, weight^T) + bias"),
    "aten.silu.default": (_decomp_silu, "input * sigmoid(input)"),
    "aten.mish.default": (_decomp_mish, "input * tanh(softplus(input))"),
    "aten.hardswish.default": (_decomp_hardswish, "input * relu6(input + 3) / 6"),
    "aten.leaky_relu.default": (_decomp_leaky_relu, "max(0, x) + negative_slope * min(0, x)"),
}


def synthesize_export_decomposition(
    target: str,
    dossier: UnsupportedOpDossier,
) -> SynthesizedDecomposition | None:
    """Synthesize an export-level decomposition for a known ATen operator.

    Looks up ``target`` in a conservative allow-list of ATen ops whose
    decompositions into simpler primitives are well-understood.  Returns
    ``None`` for any op not on the list so the caller can fall through to
    other recovery strategies.

    Args:
        target: Fully qualified op target (e.g. ``aten.addmm.default``).
        dossier: Introspection dossier for the operator.

    Returns:
        A ``SynthesizedDecomposition`` with a callable ``decomp_fn``, or
        ``None`` if the operator is not on the allow-list.
    """
    if not dossier.is_aten:
        log.debug("synthesize_export_decomposition: %s is not an ATen op, skipping", target)
        return None

    entry = _DECOMPOSITION_ALLOW_LIST.get(target)
    if entry is None:
        log.debug("synthesize_export_decomposition: %s not in allow-list", target)
        return None

    decomp_fn, description = entry
    log.debug("synthesize_export_decomposition: found decomposition for %s: %s", target, description)
    return SynthesizedDecomposition(
        target=target,
        description=description,
        decomp_fn=decomp_fn,
    )


__all__ = ["SynthesizedDecomposition", "synthesize_export_decomposition"]
