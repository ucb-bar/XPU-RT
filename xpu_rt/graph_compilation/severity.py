"""Calibrated severity scoring for gap records.

Replaces the binary "on critical path → critical_path else coverage_gap"
heuristic with a four-bucket classifier that combines:

- ``critical_path_member`` — does removing the node disconnect every
  placeholder→output route in the FX graph?
- ``cost_fraction_estimate`` — fraction of the model's opaque-op compute
  this gap is responsible for, estimated from the op family weight and
  the input/output element counts in the gap's shape signature.
- ``blocks_lowering`` — whether this kind of gap halts payload lowering
  (today: every ``unsupported_op`` does, by definition).
- ``low_cost_fallback`` — a view/reshape-style op that can be lowered
  to a plain dispatch even if it shows up as opaque.

The buckets are:

| bucket | meaning                                                            |
|--------|--------------------------------------------------------------------|
| ``critical_path``       | on critical path AND non-trivial cost          |
| ``performance_blocker`` | non-trivial cost OFF the critical path         |
| ``coverage_gap``        | small cost gap that must still be filled       |
| ``noncritical``         | view/reshape-shaped gap, fallback is cheap     |

The thresholds and the op-family table are deliberately conservative:
matmul/conv/attention always score ``critical_path`` or
``performance_blocker``; a stray ``aten.view`` opaque doesn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Op family token tables. Match by simple substring against a normalised
# form of the FX target string (``aten.matmul.default`` →
# ``"aten matmul default"``). Heavy/medium/light/view weights are flop
# proxies, not absolute FLOPs.
_HEAVY_TOKENS: tuple[str, ...] = (
    "matmul", "mm", "bmm", "addmm",
    "linear",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "attention",
    "softmax", "_softmax", "log_softmax",
    "native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training",
    "native_layer_norm", "layer_norm", "rms_norm",
    "einsum",
    "scaled_mm",
)
_MEDIUM_TOKENS: tuple[str, ...] = (
    "gelu", "relu", "tanh", "sigmoid", "silu", "elu", "leaky_relu",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "max_pool",
    "embedding", "dropout",
    "mean", "sum", "var", "std", "norm",
    "scatter", "gather",
)
_LIGHT_TOKENS: tuple[str, ...] = (
    "add", "sub", "mul", "div", "sqrt", "rsqrt", "pow", "exp", "log",
    "sin", "cos", "abs", "neg", "clamp", "where", "minimum", "maximum",
    "ge", "gt", "le", "lt", "eq", "ne",
    "sign", "trunc", "ceil", "floor", "round",
)
_VIEW_TOKENS: tuple[str, ...] = (
    "view", "reshape", "transpose", "permute", "expand", "expand_as",
    "squeeze", "unsqueeze", "flatten", "contiguous",
    "slice", "select", "index_select", "getitem", "__getitem__",
    "stack", "cat", "split", "chunk", "repeat", "tile",
    "to", "type_as", "detach",
)

_HEAVY_W = 1.0
_MEDIUM_W = 0.3
_LIGHT_W = 0.05
_VIEW_W = 0.01
_UNKNOWN_W = _MEDIUM_W  # custom/unknown ops default to medium

# Severity thresholds (cost fraction relative to total model opaque cost).
THRESHOLD_HIGH = 0.20
THRESHOLD_MED = 0.05
THRESHOLD_LOW = 0.02

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _normalise(target: str) -> set[str]:
    """Tokenise ``target`` into a set of identifier-like words."""
    return {t.lower() for t in _TOKEN_RE.findall(target)}


def op_family(fx_target: str) -> tuple[str, float]:
    """Return ``(family, weight)`` for an FX target string.

    Order matters — view-shaped ops are checked **before** light-token
    ops to keep a stray ``cat`` from being miscounted as light. Custom
    ops (``crgtoy.affine_gelu``) fall through to ``unknown``, which
    gets the same weight as a medium op so the closure isn't penalised
    for being a research workload.
    """
    tokens = _normalise(fx_target)
    if tokens & set(_HEAVY_TOKENS):
        return ("heavy", _HEAVY_W)
    if tokens & set(_VIEW_TOKENS):
        return ("view", _VIEW_W)
    if tokens & set(_MEDIUM_TOKENS):
        return ("medium", _MEDIUM_W)
    if tokens & set(_LIGHT_TOKENS):
        return ("light", _LIGHT_W)
    return ("unknown", _UNKNOWN_W)


def _shape_numel_total(shape_sig: dict[str, Any]) -> int:
    """Sum of element counts over every input and output tensor.

    Dynamic dims (strings) and zero/negative dims contribute 0 to the
    tensor-numel — we don't want to overweight a gap whose shape
    signature is a placeholder.
    """
    total = 0
    for key in ("inputs", "outputs"):
        for shape in shape_sig.get(key, []):
            n = 1
            valid = True
            for dim in shape:
                if not isinstance(dim, int) or dim <= 0:
                    valid = False
                    break
                n *= dim
            if valid:
                total += n
    return total


@dataclass(frozen=True)
class SeverityVerdict:
    bucket: str          # "critical_path"|"performance_blocker"|"coverage_gap"|"noncritical"
    score: float         # 0.0..1.0 (rounded to 3 places)
    reasons: tuple[str, ...]
    cost_fraction: float
    family: str
    raw_cost: float


def estimate_raw_cost(fx_target: str, shape_signature: dict[str, Any]) -> tuple[float, str]:
    """Per-gap raw cost = family_weight × max(numel, 1)."""
    family, weight = op_family(fx_target)
    numel = _shape_numel_total(shape_signature)
    return (weight * max(numel, 1), family)


def classify(
    *,
    on_critical_path: bool,
    cost_fraction: float,
    family: str,
    blocks_lowering: bool,
) -> SeverityVerdict:
    """Calibrated severity classification.

    The score is a soft sum used for downstream sorting (highest first
    in the queue). The bucket is the categorical decision the validator
    keys on. Reasons enumerate which inputs drove the verdict so the
    paper can report ``severity_reasons`` per gap.
    """
    reasons: list[str] = []
    score = 0.0

    if on_critical_path:
        reasons.append("on_critical_path")
        score += 0.4

    if cost_fraction >= THRESHOLD_HIGH:
        reasons.append("high_estimated_cost_fraction")
        score += 0.4
    elif cost_fraction >= THRESHOLD_MED:
        score += 0.2

    if blocks_lowering:
        reasons.append("blocks_lowering")
        score += 0.2

    # ``low_cost_fallback`` triggers when the gap is cheap enough that
    # a deterministic fallback is acceptable. View-shaped ops also
    # qualify even at moderate cost — re-implementing a transpose is
    # never worth the engineering cost.
    is_low_cost = cost_fraction < THRESHOLD_LOW
    is_view = family == "view"
    if is_low_cost or is_view:
        reasons.append("low_cost_fallback")
        score -= 0.4

    score = max(0.0, min(1.0, score))

    if is_view:
        bucket = "noncritical"
    elif cost_fraction >= THRESHOLD_HIGH and on_critical_path:
        bucket = "critical_path"
    elif cost_fraction >= THRESHOLD_HIGH and not on_critical_path:
        bucket = "performance_blocker"
    elif on_critical_path and cost_fraction >= THRESHOLD_MED:
        bucket = "critical_path"
    elif is_low_cost:
        bucket = "noncritical"
    else:
        bucket = "coverage_gap"

    return SeverityVerdict(
        bucket=bucket,
        score=round(score, 3),
        reasons=tuple(reasons),
        cost_fraction=round(cost_fraction, 4),
        family=family,
        raw_cost=0.0,  # caller sets this
    )
