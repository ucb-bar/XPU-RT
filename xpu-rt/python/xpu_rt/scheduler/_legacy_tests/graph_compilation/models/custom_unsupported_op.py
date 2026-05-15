"""Model with a registered custom op that has no Payload decomposition.

The point of this fixture is to drive the **opaque-call** + **unsupported-op**
inventories in Payload Lowering. We register a real custom op
(``crgtoy::affine_gelu``) so:

- ``torch.export`` traces it as ``torch.ops.crgtoy.affine_gelu.default``
  (no entry in ``compgen.ir.payload.decompositions``), forcing
  ``FXImporter`` into its opaque ``func.call`` fallback.
- The opaque/unsupported records carry a precise, human-readable
  target name a downstream gap-discovery pass can match on.

The op is a real (non-trivial) computation: ``y = gelu(x @ w + b)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Use the newer torch.library.custom_op decorator API. It registers a
# functional schema, an eager CPU implementation, and a meta (shape
# inference) implementation in one declaration. The decorator is
# idempotent across re-imports because ``custom_op`` looks up the
# existing op in the registry when the qualified name already exists.
def _register() -> None:
    qname = "crgtoy::affine_gelu"
    # Skip if already defined (re-import in long-running test sessions).
    try:
        torch.ops.crgtoy.affine_gelu
        return
    except (AttributeError, RuntimeError):
        pass

    @torch.library.custom_op(qname, mutates_args=())
    def affine_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(F.linear(x, w, b))

    @affine_gelu.register_fake
    def _(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape[:-1] + (w.shape[0],), dtype=x.dtype, device=x.device)


_register()


class CustomUnsupportedOpModel(nn.Module):
    """A tiny model whose forward calls our unknown custom op."""

    def __init__(self, in_dim: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.crgtoy.affine_gelu(x, self.weight, self.bias)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = CustomUnsupportedOpModel().eval()
    x = torch.randn(2, 16)
    return model, (x,)
