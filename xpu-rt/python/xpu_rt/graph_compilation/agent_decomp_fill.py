"""Deterministic agent fill: known-good decompositions for canonical test ops.

This is **not** a real agent. It's a fixture-style fill that proves the
materialize → fill → verify → register loop works end-to-end without
introducing an LLM dependency. The first milestone of Extension Closure
ships this; LLM-driven fills land in a follow-up milestone.

Each known target maps to a Python function that decomposes the op
into supported primitives. The fill writes that function body into
``extension.py`` (the only file the agent is allowed to edit).

If the gap's ``fx_target`` is not in :data:`KNOWN_FILLS`, the fill
function raises — the deterministic agent has nothing to say about an
op it has never seen, and lying to verify isn't acceptable.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Known-good decompositions (canonical test targets)
# --------------------------------------------------------------------------- #


_AFFINE_GELU_BODY = '''"""Agent-filled extension for ``crgtoy.affine_gelu``.

Decomposition: ``y = gelu(linear(x, w, b))`` — this is the literal
implementation of the custom op (see
``tests/graph_compilation/models/custom_unsupported_op.py``), so the
extension matches the reference within floating-point tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extension(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``x @ wᵀ + b → gelu``. Decomposed to F.linear + F.gelu.

    Both primitives have entries in compgen.ir.payload.decompositions,
    so re-running Payload Lowering after this extension is registered
    leaves zero crgtoy.affine_gelu opaque calls.
    """
    return F.gelu(F.linear(x, w, b))
'''


KNOWN_FILLS: dict[str, str] = {
    # Both Dynamo (pre-decomp) and export (.default suffix) names map to
    # the same body — the decomposition is identical.
    "crgtoy.affine_gelu": _AFFINE_GELU_BODY,
    "crgtoy.affine_gelu.default": _AFFINE_GELU_BODY,
}


class UnknownTargetError(ValueError):
    """The deterministic agent has no fill for this fx_target."""


def deterministic_fill(workspace: Path, fx_target: str) -> Path:
    """Write ``extension.py`` for ``fx_target`` into ``workspace``.

    Returns the path written. Raises :class:`UnknownTargetError` if
    the target is outside the canonical test set — the deterministic
    agent only handles ``crgtoy.affine_gelu`` for this milestone.
    """
    body = KNOWN_FILLS.get(fx_target)
    if body is None:
        raise UnknownTargetError(
            f"deterministic agent has no fill for {fx_target!r}; "
            f"known: {sorted(KNOWN_FILLS)}"
        )
    extension_path = Path(workspace) / "extension.py"
    extension_path.write_text(body, encoding="utf-8")
    return extension_path
