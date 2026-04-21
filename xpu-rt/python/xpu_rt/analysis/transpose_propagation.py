"""Detect transpose chains and identify cancellations.

Two transposes back-to-back with the same permutation cancel; a
transpose absorbed into an adjacent matmul (transpose-A or transpose-B
variant) saves a kernel launch.

This module is a *finder*, not a rewriter. It returns:
  * ``TransposeChain`` records — runs of transposes between non-transpose
    consumers, with the composed permutation.
  * ``TransposeCancellation`` proposals — concrete (op_a, op_b, reason)
    triples the caller can apply via the existing payload mutator.

The actual rewrite lives in ``ir/recipe/payload_mutators.py`` (extension
point); this module produces the verdicts the agent / oracle reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation


def _is_transpose_op(op: Operation) -> bool:
    """Recognise transpose / permute ops by name + pattern hint."""
    name = op.name.lower()
    if "transpose" in name or "permute" in name:
        return True
    attrs = getattr(op, "attributes", {})
    hint = attrs.get("compgen._pattern_hint") if attrs else None
    if hint is not None and hasattr(hint, "data"):
        if hint.data in ("transpose", "permute"):
            return True
    return False


@dataclass
class TransposeChain:
    """A consecutive run of transposes feeding one non-transpose consumer.

    Attributes:
        ops: The transpose ops in dataflow order (producer → consumer).
        consumer: The first non-transpose op consuming the chain's output.
            None when the chain ends at a func.return.
        composed_permutation: The composition of all permutations in the
            chain — when this is the identity ``(0, 1, ..., n-1)`` the
            entire chain is a no-op and can be deleted.
    """

    ops: list[Operation] = field(default_factory=list)
    consumer: Operation | None = None
    composed_permutation: tuple[int, ...] = ()

    @property
    def is_identity(self) -> bool:
        n = len(self.composed_permutation)
        return n > 0 and self.composed_permutation == tuple(range(n))


@dataclass
class TransposeCancellation:
    """A concrete cancellation proposal: delete ``ops_to_remove``."""

    chain: TransposeChain
    ops_to_remove: tuple[Operation, ...]
    reason: str = ""


def _permutation_of(op: Operation) -> tuple[int, ...] | None:
    """Best-effort: pull the permutation from the op's attributes."""
    attrs = getattr(op, "attributes", {})
    perm = attrs.get("permutation") if attrs else None
    if perm is None:
        return None
    if hasattr(perm, "data") and isinstance(perm.data, list | tuple):
        try:
            return tuple(int(getattr(d, "value", d)) for d in perm.data)
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _compose(p1: tuple[int, ...], p2: tuple[int, ...]) -> tuple[int, ...]:
    """Compose ``p2 ∘ p1``: result[i] = p1[p2[i]]."""
    if len(p1) != len(p2):
        return ()
    return tuple(p1[p2[i]] for i in range(len(p1)))


def detect_transpose_chains(module: ModuleOp) -> list[TransposeChain]:
    """Walk ``module`` and build a TransposeChain for each consecutive
    transpose run.

    Limitation: this is a *linear* chain finder. Branching dataflow
    (transpose feeding multiple consumers) is reported as separate
    chains for each consumer.
    """
    chains: list[TransposeChain] = []
    seen: set[Operation] = set()
    for op in module.walk():
        if not _is_transpose_op(op) or op in seen:
            continue

        # Walk forward through transposes only.
        chain_ops: list[Operation] = []
        cur: Operation | None = op
        composed: tuple[int, ...] | None = None
        while cur is not None and _is_transpose_op(cur) and cur not in seen:
            chain_ops.append(cur)
            seen.add(cur)
            perm = _permutation_of(cur)
            if perm is not None:
                if composed is None:
                    composed = perm
                else:
                    composed = _compose(composed, perm)
            # Find the single consumer of this op's first result; stop on branch.
            if not cur.results:
                cur = None
                break
            users = list(cur.results[0].uses)
            if len(users) != 1:
                cur = None
                break
            cur = users[0].operation

        consumer = cur if cur is not None and not _is_transpose_op(cur) else None
        chains.append(TransposeChain(
            ops=chain_ops,
            consumer=consumer,
            composed_permutation=composed or (),
        ))
    return chains


def propose_transpose_cancellations(
    module: ModuleOp,
) -> list[TransposeCancellation]:
    """Identify chains where the composed permutation is the identity —
    those can be wholesale removed."""
    out: list[TransposeCancellation] = []
    for chain in detect_transpose_chains(module):
        if chain.is_identity and chain.ops:
            out.append(TransposeCancellation(
                chain=chain,
                ops_to_remove=tuple(chain.ops),
                reason=(
                    f"chain of {len(chain.ops)} transposes composes to identity "
                    f"{chain.composed_permutation} — delete all"
                ),
            ))
    return out


__all__ = [
    "TransposeCancellation",
    "TransposeChain",
    "detect_transpose_chains",
    "propose_transpose_cancellations",
]
