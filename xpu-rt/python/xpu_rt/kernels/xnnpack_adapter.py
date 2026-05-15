"""Adapter layer between XPU-RT kernel contracts and the libxpu_rt XNNPACK bridge.

This module is **pure Python**: a declarative catalogue mapping
``(contract.op_kind, contract.dtype, contract.layout)`` to a wire-stable
``XnnOpKind`` integer that the libxpu_rt bridge speaks
(``runtime/native/libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.h``).

The :class:`xpu_rt.kernels.providers.xnnpack.XnnpackProvider` consults
this catalogue in :py:meth:`can_bid` (to decide whether to bid) and in
:py:meth:`propose` (to emit the C kernel artifact with the right
``xpu_rt_xnn_create()`` call).

The catalogue is deliberately code-shaped, not config-shaped: the
typed Python dataclasses give IDE completion and let the audit
framework discover the supported op surface declaratively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class XnnOpKind(IntEnum):
    """Mirror of the ``xpu_rt_xnn_op_kind`` enum in xnnpack_bridge.h.

    KEEP IN SYNC with that enum. Both ends are wire-stable: never
    renumber, only append.
    """

    UNKNOWN                          = 0

    # ----- f32 NHWC core -------------------------------------------
    FULLY_CONNECTED_F32              = 1
    CONVOLUTION2D_NHWC_F32           = 2
    DEPTHWISE_CONVOLUTION2D_NHWC_F32 = 3
    BATCH_MATMUL_F32                 = 4
    AVERAGE_POOLING2D_NHWC_F32       = 5
    MAX_POOLING2D_NHWC_F32           = 6
    GLOBAL_AVERAGE_POOLING_NHWC_F32  = 7
    SOFTMAX_NC_F32                   = 8
    UNARY_F32                        = 9
    BINARY_F32                       = 10
    REDUCE_F32                       = 11
    TRANSPOSE                        = 12
    RESIZE_BILINEAR2D_NHWC_F32       = 13
    DECONVOLUTION2D_NHWC_F32         = 14
    PRELU_NHWC_F32                   = 15
    LEAKY_RELU_NHWC_F32              = 16
    STATIC_SLICE                     = 17

    # ----- f16 (reserved +100) -------------------------------------
    FULLY_CONNECTED_F16              = 101
    CONVOLUTION2D_NHWC_F16           = 102
    BATCH_MATMUL_F16                 = 104

    # ----- qs8 / qu8 / qd8→f32 (reserved +200 / +300 / +400) -------
    FULLY_CONNECTED_QS8              = 201
    CONVOLUTION2D_NHWC_QS8           = 202
    DEPTHWISE_CONVOLUTION2D_NHWC_QS8 = 203

    FULLY_CONNECTED_QU8              = 301
    CONVOLUTION2D_NHWC_QU8           = 302

    FULLY_CONNECTED_QD8_F32          = 401
    CONVOLUTION2D_NHWC_QD8_F32       = 402
    BATCH_MATMUL_QD8_F32             = 404


# Activation sub-kinds for XnnOpKind.UNARY_F32.
class XnnUnaryKind(IntEnum):
    RELU      = 1
    SIGMOID   = 2
    TANH      = 3
    HSWISH    = 4
    GELU      = 5
    ELU       = 6
    ABS       = 7
    NEG       = 8
    FLOOR     = 9
    CEIL      = 10
    SQUARE    = 11
    SQRT      = 12
    LOG       = 13
    EXP       = 14


class XnnBinaryKind(IntEnum):
    ADD                    = 1
    SUB                    = 2
    MUL                    = 3
    DIV                    = 4
    MAX                    = 5
    MIN                    = 6
    SQUARED_DIFFERENCE     = 7


class XnnReduceKind(IntEnum):
    SUM       = 1
    MEAN      = 2
    MAX       = 3
    MIN       = 4


# Map from human op-kind names commonly emitted by the
# match_library_call pass (and used in KernelContractV3.op_kind) to
# the XNNPACK bridge integer.
@dataclass(frozen=True)
class OpEntry:
    """Declarative entry: one (op_kind, dtype, layout) → XNNPACK op."""

    contract_op_kinds: tuple[str, ...]  # e.g. ("matmul", "fully_connected", "linear")
    dtype: str                          # "f32" | "f16" | "qs8" | "qu8" | "qd8_f32"
    layout: str                         # "NHWC" | "NC" | "any"
    xnn_kind: XnnOpKind
    # confidence the provider should advertise on a clean hit; tuned
    # so XNNPACK wins over cffi_c on host_cpu but loses to a hardware-
    # specific kernel if one declares for the same target.
    base_confidence: float = 0.85
    # Whether the provider needs static_weights (FC, conv): if so, the
    # contract must carry a packed weights buffer.
    needs_static_weights: bool = False
    # Notes shown in BidPreview.detail when this entry wins.
    note: str = ""


# Master table. Order matters for ties: earlier wins. Keep grouped by
# op-kind family.
CATALOGUE: tuple[OpEntry, ...] = (
    # ===== Fully-connected / matmul / linear =================
    OpEntry(("matmul", "fully_connected", "linear", "dense"),
            "f32", "NC", XnnOpKind.FULLY_CONNECTED_F32,
            base_confidence=0.90, needs_static_weights=True,
            note="xnn_create_fully_connected_nc_f32"),
    OpEntry(("matmul", "fully_connected", "linear", "dense"),
            "f16", "NC", XnnOpKind.FULLY_CONNECTED_F16,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("matmul", "fully_connected", "linear", "dense"),
            "qs8", "NC", XnnOpKind.FULLY_CONNECTED_QS8,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("matmul", "fully_connected", "linear", "dense"),
            "qd8_f32", "NC", XnnOpKind.FULLY_CONNECTED_QD8_F32,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("batch_matmul", "bmm"), "f32", "any",
            XnnOpKind.BATCH_MATMUL_F32, base_confidence=0.85),
    OpEntry(("batch_matmul", "bmm"), "f16", "any",
            XnnOpKind.BATCH_MATMUL_F16, base_confidence=0.80),
    OpEntry(("batch_matmul", "bmm"), "qd8_f32", "any",
            XnnOpKind.BATCH_MATMUL_QD8_F32, base_confidence=0.80),

    # ===== 2D convolution =====================================
    OpEntry(("conv", "conv2d", "convolution", "convolution2d"),
            "f32", "NHWC", XnnOpKind.CONVOLUTION2D_NHWC_F32,
            base_confidence=0.95, needs_static_weights=True,
            note="xnn_create_convolution2d_nhwc_f32"),
    OpEntry(("conv", "conv2d", "convolution", "convolution2d"),
            "f16", "NHWC", XnnOpKind.CONVOLUTION2D_NHWC_F16,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("conv", "conv2d", "convolution", "convolution2d"),
            "qs8", "NHWC", XnnOpKind.CONVOLUTION2D_NHWC_QS8,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("conv", "conv2d", "convolution", "convolution2d"),
            "qu8", "NHWC", XnnOpKind.CONVOLUTION2D_NHWC_QU8,
            base_confidence=0.80, needs_static_weights=True),
    OpEntry(("conv", "conv2d", "convolution", "convolution2d"),
            "qd8_f32", "NHWC", XnnOpKind.CONVOLUTION2D_NHWC_QD8_F32,
            base_confidence=0.80, needs_static_weights=True),

    # ===== Depthwise conv =====================================
    OpEntry(("depthwise_conv", "depthwise_conv2d", "depthwise"),
            "f32", "NHWC", XnnOpKind.DEPTHWISE_CONVOLUTION2D_NHWC_F32,
            base_confidence=0.95, needs_static_weights=True),
    OpEntry(("depthwise_conv", "depthwise_conv2d", "depthwise"),
            "qs8", "NHWC", XnnOpKind.DEPTHWISE_CONVOLUTION2D_NHWC_QS8,
            base_confidence=0.85, needs_static_weights=True),

    # ===== Deconv =============================================
    OpEntry(("deconv", "deconv2d", "transposed_conv"),
            "f32", "NHWC", XnnOpKind.DECONVOLUTION2D_NHWC_F32,
            base_confidence=0.85, needs_static_weights=True),

    # ===== Pooling ============================================
    OpEntry(("avg_pool", "average_pool", "average_pool2d"),
            "f32", "NHWC", XnnOpKind.AVERAGE_POOLING2D_NHWC_F32,
            base_confidence=0.85),
    OpEntry(("max_pool", "max_pool2d"),
            "f32", "NHWC", XnnOpKind.MAX_POOLING2D_NHWC_F32,
            base_confidence=0.85),
    OpEntry(("global_avg_pool", "adaptive_avg_pool", "adaptive_average_pool2d"),
            "f32", "NHWC", XnnOpKind.GLOBAL_AVERAGE_POOLING_NHWC_F32,
            base_confidence=0.85),

    # ===== Softmax ============================================
    OpEntry(("softmax",), "f32", "NC", XnnOpKind.SOFTMAX_NC_F32,
            base_confidence=0.90),

    # ===== Elementwise unary / binary =========================
    OpEntry(("unary", "relu", "sigmoid", "tanh", "hswish", "gelu",
             "elu", "abs", "neg", "floor", "ceil", "square", "sqrt",
             "log", "exp"),
            "f32", "any", XnnOpKind.UNARY_F32, base_confidence=0.80),
    OpEntry(("binary", "add", "sub", "mul", "div", "max", "min",
             "squared_difference"),
            "f32", "any", XnnOpKind.BINARY_F32, base_confidence=0.80),

    # ===== Reductions =========================================
    OpEntry(("reduce", "sum", "mean", "max_reduce", "min_reduce"),
            "f32", "any", XnnOpKind.REDUCE_F32, base_confidence=0.75),

    # ===== Activations standalone =============================
    OpEntry(("prelu",), "f32", "NHWC", XnnOpKind.PRELU_NHWC_F32,
            base_confidence=0.85, needs_static_weights=True),
    OpEntry(("leaky_relu",), "f32", "NHWC",
            XnnOpKind.LEAKY_RELU_NHWC_F32, base_confidence=0.85),

    # ===== Misc ===============================================
    OpEntry(("transpose", "permute"), "f32", "any",
            XnnOpKind.TRANSPOSE, base_confidence=0.70),
    OpEntry(("resize_bilinear", "interpolate_bilinear",
             "upsample_bilinear"),
            "f32", "NHWC", XnnOpKind.RESIZE_BILINEAR2D_NHWC_F32,
            base_confidence=0.85),
    OpEntry(("slice", "static_slice"), "f32", "any",
            XnnOpKind.STATIC_SLICE, base_confidence=0.70),
)


# Distilled "supported op-kind / target_family" view for the provider
# card and the routing table. Keeps the YAML card from going stale.
def supported_contract_kinds() -> tuple[str, ...]:
    seen: set[str] = set()
    for entry in CATALOGUE:
        for k in entry.contract_op_kinds:
            seen.add(k)
    return tuple(sorted(seen))


# NHWC-requiring kinds. The provider uses this to detect when the
# contract's layout is wrong and emit a typed `requires_nhwc_layout`
# decline reason that the layout-norm pass can react to.
def kinds_requiring_nhwc() -> tuple[str, ...]:
    seen: set[str] = set()
    for entry in CATALOGUE:
        if entry.layout == "NHWC":
            for k in entry.contract_op_kinds:
                seen.add(k)
    return tuple(sorted(seen))


def kinds_requiring_static_weights() -> tuple[str, ...]:
    seen: set[str] = set()
    for entry in CATALOGUE:
        if entry.needs_static_weights:
            for k in entry.contract_op_kinds:
                seen.add(k)
    return tuple(sorted(seen))


# ----- Lookup --------------------------------------------------

@dataclass(frozen=True)
class LookupResult:
    entry: OpEntry
    canonical_op_kind: str          # the entry's first contract_op_kinds entry
    matched_via: str                # which contract field aliased to canonical


def lookup(
    contract_op_kind: str,
    dtype: str,
    layout: str | None,
) -> LookupResult | None:
    """Find the OpEntry that handles ``(op_kind, dtype, layout)``.

    Returns ``None`` when no entry matches. The caller (the provider's
    ``can_bid``) uses the absence to either decline outright (truly
    unsupported) or to flag a typed decline reason (e.g. dtype is
    supported only in a layout we can't accept).
    """

    op_kind_norm = (contract_op_kind or "").lower()
    dtype_norm = (dtype or "").lower()
    layout_norm = (layout or "any").upper()

    for entry in CATALOGUE:
        if op_kind_norm not in entry.contract_op_kinds:
            continue
        if entry.dtype != dtype_norm:
            continue
        if entry.layout != "any" and layout_norm != "ANY" and entry.layout.upper() != layout_norm:
            continue
        return LookupResult(
            entry=entry,
            canonical_op_kind=entry.contract_op_kinds[0],
            matched_via=op_kind_norm,
        )
    return None


def lookup_any_layout(contract_op_kind: str, dtype: str) -> Iterable[OpEntry]:
    """Return every entry matching (op_kind, dtype) regardless of layout.

    Used by the provider to detect "we'd support this if it were NHWC"
    and emit the right typed decline reason.
    """

    op_kind_norm = (contract_op_kind or "").lower()
    dtype_norm = (dtype or "").lower()
    return tuple(
        entry for entry in CATALOGUE
        if op_kind_norm in entry.contract_op_kinds and entry.dtype == dtype_norm
    )
