"""Multi-output model — exercises non-critical-path detection.

Topology:

::

    x ─┬─ linear_a ── tanh_a ── out_a
       └─ linear_b ── relu_b ── out_b

Two outputs share the input but are otherwise independent. So:

- ``linear_a`` and ``tanh_a`` feed *only* ``out_a`` — removing either
  breaks ``out_a`` but leaves ``out_b`` intact. Under
  ``_is_critical`` they should be reported as **non-critical** because
  the output node still has at least one fully-reachable producer
  (``out_b``).
- Symmetric for ``linear_b`` and ``relu_b``.

Wait — re-read ``_is_critical``: "critical iff the output node has at
least one non-computable producer". With a two-tuple output, blocking
``linear_a`` makes ``out_a`` non-computable, which means
``all(p in computable for p in producers)`` is False, which means
the algorithm marks it **critical**. That's deliberate: gap discovery
should consider an op critical if removing it breaks *any* output
value the user asked for.

So this model demonstrates the conservative semantic, not the
non-critical case. None of the realistic torch models we want to
support actually have non-critical ops in the strict sense — once an
op produces a value the user returns, removing it loses the value.

Kept as ``residual_branch.yaml`` for backward-compat naming; the model
is now a two-output proof of the algorithm's behavior on branchy DAGs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoOutputBranches(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear_a = nn.Linear(dim, dim)
        self.linear_b = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_a = torch.tanh(self.linear_a(x))
        out_b = F.relu(self.linear_b(x))
        return out_a, out_b


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = TwoOutputBranches().eval()
    x = torch.randn(2, 16)
    return model, (x,)
