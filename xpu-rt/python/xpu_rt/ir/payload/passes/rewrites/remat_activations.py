"""``remat_activations`` -- tag activations for rematerialization
during backward pass.

XLA's ``HloRematerialization``: when peak live memory exceeds a
threshold, tag certain intermediate activations as "recomputable"
so the runtime can re-run their producers at backward time instead
of storing them.

Selection policy:
- Only large float tensors (> ``min_size_bytes``).
- Produced by a single op with pure float semantics
  (``compgen._pattern_hint`` must be in the allowlist).
- Not already marked as ``persistent`` / ``shared_readonly``.

No structural rewrite; the tag is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType

_RECOMPUTABLE_HINTS: frozenset[str] = frozenset(
    {
        "layer_norm",
        "rms_norm",
        "softmax",
        "silu",
        "gelu",
        "sub",
        "mul",
        "div",
        "add",
        "sigmoid",
        "pow_tensor_scalar",
    }
)


@dataclass(frozen=True)
class RematActivationsConfig:
    min_size_bytes: int = 8192
    hint_allowlist: frozenset[str] = _RECOMPUTABLE_HINTS
    assume_elem_bytes: int = 4  # default f32


@dataclass
class RematActivationsStats:
    ops_seen: int = 0
    ops_tagged: int = 0
    total_recomputable_bytes: int = 0


def _tensor_size_bytes(t: TensorType, elem_bytes: int) -> int:
    size = elem_bytes
    for d in t.get_shape():
        if d < 0:
            return -1
        size *= d
    return size


def run_remat_activations(
    module: ModuleOp,
    *,
    config: RematActivationsConfig | None = None,
) -> RematActivationsStats:
    cfg = config if config is not None else RematActivationsConfig()
    stats = RematActivationsStats()
    for op in module.walk():
        hint_attr = op.attributes.get("compgen._pattern_hint")
        if not isinstance(hint_attr, StringAttr):
            continue
        if hint_attr.data not in cfg.hint_allowlist:
            continue
        stats.ops_seen += 1
        if "compgen.rematerialize" in op.attributes:
            continue
        if not op.results:
            continue
        rt = op.results[0].type
        if not isinstance(rt, TensorType):
            continue
        size = _tensor_size_bytes(rt, cfg.assume_elem_bytes)
        if size < cfg.min_size_bytes:
            continue
        op.attributes["compgen.rematerialize"] = StringAttr("true")
        op.attributes["compgen.remat_size_bytes"] = StringAttr(str(size))
        stats.ops_tagged += 1
        stats.total_recomputable_bytes += size
    return stats


__all__ = [
    "RematActivationsConfig",
    "RematActivationsStats",
    "run_remat_activations",
]
