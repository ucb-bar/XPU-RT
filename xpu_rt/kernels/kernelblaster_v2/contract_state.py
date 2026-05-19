"""KernelContract → StateVector.

The legacy KernelBlaster derived state from raw NCU profile metrics
(``memory_bandwidth_limited``, ``compute_throughput_limited``, …) — a
CUDA-only signal. KB v2 derives state from the typed
:class:`KernelContract` (and optional v3 extension) so the same agent
loop works on Saturn, Gemmini, host CPU, or any future target.

The :class:`StateVector` is the cache/strategy key. Two contracts that
hash to the same StateVector share lessons and reuse strategies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from xpu_rt.kernels.provider import KernelContract


@dataclass(frozen=True)
class StateVector:
    """Compact, hashable summary of "what kind of contract is this".

    Built from contract fields, target hardware facts, and any
    coarsened shape information available. Used as the key into the
    Target Card's ``strategies.json`` and to filter ``lessons.jsonl``.
    """

    op_family: str
    archetype: str  # COMPUTE_TILED / POINTWISE / REDUCE / MEMORY / ACTIVATION / TYPE_CONV_INDEX / unknown
    dtype_class: str  # primary input dtype canonicalized (i8, bf16, fp16, fp32, mixed)
    layout_kind: str
    target_id: str
    granularity: str = "normal"  # micro / normal / mega
    shape_signature: str = ""  # bucketed shape fingerprint

    def key(self) -> str:
        """Stable string used as the strategy DB primary key."""
        return "|".join(
            (
                self.op_family,
                self.archetype,
                self.dtype_class,
                self.layout_kind,
                self.target_id,
                self.granularity,
                self.shape_signature,
            )
        )

    def hash(self) -> str:
        return hashlib.sha256(self.key().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_family": self.op_family,
            "archetype": self.archetype,
            "dtype_class": self.dtype_class,
            "layout_kind": self.layout_kind,
            "target_id": self.target_id,
            "granularity": self.granularity,
            "shape_signature": self.shape_signature,
            "hash": self.hash(),
        }


# Common op-family normalizations so contracts that say "matmul" or
# "gemm" or "mm" all hit the same lesson rows.
_OP_FAMILY_CANONICAL: dict[str, str] = {
    "matmul": "matmul",
    "mm": "matmul",
    "gemm": "matmul",
    "bmm": "matmul",
    "linear": "matmul",
    "conv": "conv",
    "conv2d": "conv",
    "conv1d": "conv",
    "convolution": "conv",
    "reduce": "reduce",
    "sum": "reduce",
    "mean": "reduce",
    "softmax": "softmax",
    "relu": "activation",
    "gelu": "activation",
    "elementwise": "pointwise",
    "binary": "pointwise",
}

# Dtype normalization — strips precision suffixes so 'i8/int8/sint8'
# collapse and the strategy DB stays compact.
_DTYPE_CANONICAL: dict[str, str] = {
    "i8": "i8",
    "int8": "i8",
    "sint8": "i8",
    "uint8": "u8",
    "u8": "u8",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "f32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "i32": "i32",
    "int32": "i32",
    "f8e4m3": "f8",
    "f8e5m2": "f8",
}

_LAYOUT_CANONICAL: dict[str, str] = {
    "row_major": "row_major",
    "column_major": "col_major",
    "col_major": "col_major",
    "nchw": "nchw",
    "nhwc": "nhwc",
    "blocked": "blocked",
    "packed_k_major": "packed_k",
    "opaque": "opaque",
}


def _canonical_op_family(op: str) -> str:
    if not op:
        return "unknown"
    return _OP_FAMILY_CANONICAL.get(op.lower(), op.lower())


def _canonical_dtype(dtypes: tuple[str, ...]) -> str:
    if not dtypes:
        return "unknown"
    primary = dtypes[0].lower()
    if primary in _DTYPE_CANONICAL:
        primary = _DTYPE_CANONICAL[primary]
    # If inputs and accumulators differ (e.g. i8 in, i32 acc), record as
    # mixed so the strategy DB doesn't collapse them.
    distinct = {_DTYPE_CANONICAL.get(d.lower(), d.lower()) for d in dtypes}
    return "mixed" if len(distinct) > 1 else primary


def _canonical_layout(layout: str) -> str:
    if not layout:
        return "opaque"
    return _LAYOUT_CANONICAL.get(layout.lower(), layout.lower())


def _shape_signature(contract: KernelContract) -> str:
    """Bucketed shape fingerprint that keeps similar shapes together.

    Each dimension is mapped to one of: ``"k"`` (small ≤ 32), ``"m"``
    (32-512), ``"l"`` (>512), or ``"?"`` (unknown). The full signature
    is the per-input string joined with ``"x"``. Symbolic / wildcard
    shapes get ``"?"`` and still produce a stable, comparable key.
    """
    if not contract.input_shapes:
        return ""

    def bucket(d: int | None) -> str:
        if d is None:
            return "?"
        if d <= 0:
            return "?"
        if d <= 32:
            return "k"
        if d <= 512:
            return "m"
        return "l"

    parts: list[str] = []
    for shape in contract.input_shapes:
        parts.append("".join(bucket(d) for d in shape))
    return "x".join(parts)


def derive_state(
    contract: KernelContract,
    *,
    target_id: str | None = None,
    archetype_hint: str = "",
    granularity_hint: str = "",
) -> StateVector:
    """Derive a :class:`StateVector` from a kernel contract.

    Args:
        contract: The legacy KernelContract from
            :mod:`xpu_rt.kernels.provider`.
        target_id: Override the target id when the contract's
            ``target_name`` doesn't match the knowledge-card id (e.g.
            ``"saturn"`` in the contract but ``"saturn_opu_v128"`` in
            the card). Falls back to ``contract.target_name`` then
            ``"unknown"``.
        archetype_hint: Optional v3 archetype string. Defaults to a
            heuristic from the op family.
        granularity_hint: Optional v3 granularity (micro/normal/mega).
    """
    op_family = _canonical_op_family(contract.op_family)
    archetype = archetype_hint or _archetype_heuristic(op_family)
    dtype_class = _canonical_dtype(contract.dtypes)
    layout_kind = _canonical_layout(contract.layout)
    resolved_target = target_id or contract.target_name or "unknown"
    granularity = granularity_hint or "normal"
    return StateVector(
        op_family=op_family,
        archetype=archetype,
        dtype_class=dtype_class,
        layout_kind=layout_kind,
        target_id=resolved_target,
        granularity=granularity,
        shape_signature=_shape_signature(contract),
    )


_ARCHETYPE_HEURISTIC: dict[str, str] = {
    "matmul": "COMPUTE_TILED",
    "conv": "COMPUTE_TILED",
    "reduce": "REDUCE",
    "softmax": "REDUCE",
    "pointwise": "POINTWISE",
    "activation": "ACTIVATION",
    "gemv": "COMPUTE_TILED",
    "memory": "MEMORY",
    "memcpy": "MEMORY",
}


def _archetype_heuristic(op_family: str) -> str:
    return _ARCHETYPE_HEURISTIC.get(op_family, "unknown")
