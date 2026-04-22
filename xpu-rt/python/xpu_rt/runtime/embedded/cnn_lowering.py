"""CNN → embedded C lowering.

Walks a ResNet-style ``nn.Module`` (Conv/BN/ReLU/residual/pool/linear)
and emits a deterministic, bit-reproducible C forward pass plus a
packed float32 weights blob. BN is folded into the preceding Conv at
lowering time — standard inference-time trick — so the runtime only
needs generic conv2d/relu/add/pool/linear primitives.

Scope limit (honest): this lowerer is intentionally topology-aware.
It recognises the ConvNet family used by the Saturn OPU bring-up
(``tests/fixtures/saturn_opu_convnet``): a stem, three stages of
basic residual blocks, global avgpool, linear head. Anything else
raises :class:`NotImplementedError` with a specific message. Generic
support comes from the FX / payload-IR path; this module exists so
the bring-up has a path that works *today* for the real model.

Memory plan
-----------

The forward function consumes four 128 KiB activation buffers from the
arena (``A``, ``B``, ``R``, ``S``). That's enough for every tensor the
ConvNet produces — peak is the 32 × 32 × 32 activation after the stem
(128 KiB). Conservatively sized so later CNNs of similar scale fit
without re-planning.

Ping-pong convention: the "current" activation lives in ``A`` or ``B``
alternately; each residual block writes its output back to whichever
buffer held the block input. No hidden reshapes; no transposes. NCHW
layout matches PyTorch bit-for-bit so the weights need no relayout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

_BN_EPS = 1e-5  # PyTorch default; assert at lowering time.


def _fold_bn_into_conv(
    conv_w: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a ``Conv2d`` (no bias) + ``BatchNorm2d`` into a biased Conv.

    Math (PyTorch inference-time semantics):

        y = (x - μ) / √(σ² + ε) * γ + β
        so for Conv output z: y = (z - μ) / √(σ² + ε) * γ + β
        Folded: z'_{oc} = α_{oc} * (Σ w_{oc,:,:,:} * x) + δ_{oc}
          α = γ / √(σ² + ε)    δ = β - α * μ

    Returns ``(W_folded, b_folded)``.
    """
    std = torch.sqrt(bn_var + eps)
    alpha = bn_weight / std  # [Cout]
    w_f = conv_w * alpha.view(-1, 1, 1, 1)
    b_f = bn_bias - alpha * bn_mean
    return w_f.contiguous(), b_f.contiguous()


# ---------------------------------------------------------------------------
# Blob packing
# ---------------------------------------------------------------------------


@dataclass
class _BlobRef:
    name: str
    offset_floats: int  # into the float32 view of compgen_model_blob
    num_floats: int


class _BlobBuilder:
    """Pack float32 tensors into one contiguous blob; track offsets."""

    def __init__(self) -> None:
        self._flats: list[torch.Tensor] = []
        self._refs: dict[str, _BlobRef] = {}
        self._total = 0

    def add(self, name: str, tensor: torch.Tensor) -> _BlobRef:
        assert name not in self._refs, name
        flat = tensor.detach().to(torch.float32).contiguous().flatten()
        ref = _BlobRef(name=name, offset_floats=self._total, num_floats=flat.numel())
        self._refs[name] = ref
        self._flats.append(flat)
        self._total += flat.numel()
        return ref

    def bytes(self) -> bytes:
        full = torch.cat(self._flats) if self._flats else torch.zeros(0, dtype=torch.float32)
        return full.numpy().astype("<f4").tobytes()

    def refs(self) -> dict[str, _BlobRef]:
        return dict(self._refs)


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


_BUFFER_NAMES = ("A", "B", "R", "S")  # activation, activation-alt, residual, shortcut


def _act_bytes(shape: tuple[int, int, int]) -> int:
    C, H, W = shape
    return 4 * C * H * W


def _fmt_conv(
    cur_in: str,
    cur_out: str,
    in_shape: tuple[int, int, int],
    out_shape: tuple[int, int, int],
    stride: int,
    pad: int,
    kernel: int,
    w_off: int,
    b_off: int,
) -> str:
    ci, hi, wi = in_shape
    co, ho, wo = out_shape
    return (
        f"    cg_conv2d_f32({cur_in}, {cur_out}, "
        f"{ci}, {hi}, {wi}, {co}, {ho}, {wo}, "
        f"{kernel}, {kernel}, {stride}, {stride}, {pad}, {pad}, "
        f"blob_f32({w_off}), blob_f32({b_off}));"
    )


def _fmt_relu(buf: str, n: int) -> str:
    return f"    cg_relu_f32({buf}, {n});"


def _fmt_add(a: str, b: str, out: str, n: int) -> str:
    return f"    cg_add_f32({a}, {b}, {out}, {n});"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class LoweredModel:
    """Artifacts a successful lowering produces.

    Attributes:
        name: Model identifier baked into the emitted C.
        forward_c_source: The model-specific straight-line forward C.
            Depends on the CompGen foundational runtime's headers
            (``compgen/ops.h``, ``compgen/types.h``) and links against
            ``libcompgen_runtime.a`` (see ``runtime/Makefile``).
        weights_blob: Packed float32 weights (BN already folded).
        input_bytes / output_bytes: Fixed I/O buffer sizes.
        arena_bytes: Minimum arena size the forward needs.
        num_params: Total float32 parameters in ``weights_blob``.
        op_counts: Breakdown of ops emitted, for diagnostics.
    """

    name: str
    forward_c_source: str
    weights_blob: bytes
    input_bytes: int
    output_bytes: int
    arena_bytes: int
    num_params: int
    op_counts: dict[str, int] = field(default_factory=dict)


def _is_conv_bn_pair(seq: Any) -> bool:
    return (
        isinstance(seq, nn.Sequential)
        and len(seq) == 2
        and isinstance(seq[0], nn.Conv2d)
        and isinstance(seq[1], nn.BatchNorm2d)
    )


def _validate_block(block: nn.Module) -> None:
    for attr in ("conv1", "bn1", "conv2", "bn2", "shortcut"):
        if not hasattr(block, attr):
            raise NotImplementedError(
                f"cnn_lowering: block {type(block).__name__} is missing attribute "
                f"'{attr}' — this lowerer only handles ConvBlock-shaped residual blocks."
            )
    if not isinstance(block.conv1, nn.Conv2d):
        raise NotImplementedError("conv1 must be nn.Conv2d")
    if not isinstance(block.bn1, nn.BatchNorm2d):
        raise NotImplementedError("bn1 must be nn.BatchNorm2d")
    if not isinstance(block.conv2, nn.Conv2d):
        raise NotImplementedError("conv2 must be nn.Conv2d")
    if not isinstance(block.bn2, nn.BatchNorm2d):
        raise NotImplementedError("bn2 must be nn.BatchNorm2d")
    if not (isinstance(block.shortcut, nn.Identity) or _is_conv_bn_pair(block.shortcut)):
        raise NotImplementedError(
            "ConvBlock shortcut must be nn.Identity or nn.Sequential(Conv2d, BatchNorm2d)"
        )


def lower_cnn_to_c(
    model: nn.Module,
    *,
    sample_input_shape: tuple[int, int, int] = (3, 64, 64),
    model_name: str = "compgen_convnet",
) -> LoweredModel:
    """Lower a ResNet-style ConvNet to portable C + a weights blob.

    Expected module structure (validated at entry):

    ``model.stem``  = ``nn.Sequential(Conv2d, BatchNorm2d, ReLU)``
    ``model.stage1``, ``model.stage2``, ``model.stage3``
                    = ``nn.Sequential`` of 2 ``ConvBlock``-shaped modules each
    ``model.pool``  = ``nn.AdaptiveAvgPool2d(1)``
    ``model.head``  = ``nn.Linear``

    Args:
        model: The PyTorch module. Must be in ``eval()`` mode.
        sample_input_shape: ``(C, H, W)`` — used to compute activation shapes.
        model_name: Identifier baked into the emitted C (logs, symbols).

    Returns:
        :class:`LoweredModel` with C source, weights blob, byte sizes.

    Raises:
        NotImplementedError: The model topology differs from the
            expected ConvNet shape. The message pinpoints the offending
            attribute so callers can see exactly what's missing.
    """
    if model.training:
        raise ValueError("cnn_lowering: model must be in eval() mode (BN frozen)")

    # Topology validation up-front — fail loud, not deep in emission.
    for attr in ("stem", "stage1", "stage2", "stage3", "pool", "head"):
        if not hasattr(model, attr):
            raise NotImplementedError(
                f"cnn_lowering: model missing attribute '{attr}' — "
                "this lowerer targets the ConvNet family only."
            )
    if not isinstance(model.stem, nn.Sequential):
        raise NotImplementedError("model.stem must be nn.Sequential(Conv, BN, ReLU)")
    if len(model.stem) != 3 or not (
        isinstance(model.stem[0], nn.Conv2d)
        and isinstance(model.stem[1], nn.BatchNorm2d)
        and isinstance(model.stem[2], nn.ReLU)
    ):
        raise NotImplementedError("unexpected stem layout")
    if not isinstance(model.pool, nn.AdaptiveAvgPool2d):
        raise NotImplementedError("model.pool must be nn.AdaptiveAvgPool2d")
    if not isinstance(model.head, nn.Linear):
        raise NotImplementedError("model.head must be nn.Linear")
    for stage_name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, stage_name)
        if not isinstance(stage, nn.Sequential):
            raise NotImplementedError(f"model.{stage_name} must be nn.Sequential")
        for blk in stage:
            _validate_block(blk)

    # Discover shapes by tracing.
    # We could infer from conv params; running a dummy tensor through the
    # model is more robust to padding/stride quirks and is cheap.
    with torch.no_grad():
        dummy = torch.zeros(1, *sample_input_shape)
        stem_out = model.stem(dummy)
        stage1_outs = [dummy.new_zeros(0)]  # placeholder
        a = stem_out
        stage_outputs: list[list[torch.Tensor]] = []
        for stage_name in ("stage1", "stage2", "stage3"):
            stage = getattr(model, stage_name)
            this_stage: list[torch.Tensor] = []
            for blk in stage:
                a = blk(a)
                this_stage.append(a)
            stage_outputs.append(this_stage)
        pooled = model.pool(a).flatten(1)
        final_out = model.head(pooled)

    # ----------------------------------------------------------------
    # Pack weights (BN folded into each Conv).
    # ----------------------------------------------------------------
    bb = _BlobBuilder()
    op_counts: dict[str, int] = {"conv2d": 0, "relu": 0, "add": 0, "linear": 0, "avgpool": 0}
    num_params = 0

    def _pack_conv_bn(prefix: str, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[_BlobRef, _BlobRef]:
        assert conv.bias is None, "conv before BN must have bias=False"
        assert abs(bn.eps - _BN_EPS) < 1e-9, f"unexpected BN eps {bn.eps}"
        w_f, b_f = _fold_bn_into_conv(
            conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps,
        )
        wref = bb.add(f"{prefix}_w", w_f)
        bref = bb.add(f"{prefix}_b", b_f)
        return wref, bref

    stem_conv: nn.Conv2d = model.stem[0]
    stem_bn: nn.BatchNorm2d = model.stem[1]
    stem_w, stem_b = _pack_conv_bn("stem", stem_conv, stem_bn)
    num_params += stem_w.num_floats + stem_b.num_floats

    block_refs: list[dict[str, Any]] = []
    for stage_idx, stage_name in enumerate(("stage1", "stage2", "stage3")):
        stage = getattr(model, stage_name)
        for block_idx, blk in enumerate(stage):
            prefix = f"s{stage_idx + 1}b{block_idx}"
            w1, b1 = _pack_conv_bn(f"{prefix}_c1", blk.conv1, blk.bn1)
            w2, b2 = _pack_conv_bn(f"{prefix}_c2", blk.conv2, blk.bn2)
            refs = {
                "c1_w": w1, "c1_b": b1, "c2_w": w2, "c2_b": b2,
                "stride": blk.conv1.stride[0],
                "in_ch": blk.conv1.in_channels,
                "out_ch": blk.conv1.out_channels,
                "has_shortcut": not isinstance(blk.shortcut, nn.Identity),
            }
            if refs["has_shortcut"]:
                sc_conv: nn.Conv2d = blk.shortcut[0]
                sc_bn: nn.BatchNorm2d = blk.shortcut[1]
                assert sc_conv.kernel_size == (1, 1), "shortcut conv must be 1x1"
                wsc, bsc = _pack_conv_bn(f"{prefix}_sc", sc_conv, sc_bn)
                refs["sc_w"] = wsc
                refs["sc_b"] = bsc
                num_params += wsc.num_floats + bsc.num_floats
            num_params += w1.num_floats + b1.num_floats + w2.num_floats + b2.num_floats
            block_refs.append(refs)

    head_w = bb.add("head_w", model.head.weight)
    head_b = bb.add("head_b", model.head.bias) if model.head.bias is not None else None
    num_params += head_w.num_floats + (head_b.num_floats if head_b else 0)

    weights_blob = bb.bytes()

    # ----------------------------------------------------------------
    # Emit forward C
    # ----------------------------------------------------------------
    # Buffer allocation: compute peak activation size × 4 buffers.
    peak_floats = max(
        stem_out.numel(),
        *(t.numel() for stage in stage_outputs for t in stage),
    )
    buf_bytes = 4 * peak_floats
    # Round up to 16-byte alignment for safety.
    buf_bytes = (buf_bytes + 15) & ~15
    arena_bytes = 4 * buf_bytes + 1024  # 4 buffers + headroom for pooled/head slack

    lines: list[str] = [
        "/* SPDX-License-Identifier: Apache-2.0 */",
        "/* Auto-generated by compgen.runtime.embedded.cnn_lowering. */",
        f"/* model: {model_name}   params: {num_params} ({len(weights_blob)} B weights) */",
        "",
        '#include "compgen/ops.h"',
        '#include "compgen/types.h"',
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <string.h>",
        "",
        "extern const uint8_t compgen_model_blob[];",
        "",
        "static inline const float *blob_f32(size_t offset_floats) {",
        "    return (const float *)compgen_model_blob + offset_floats;",
        "}",
        "",
        f"/* Arena layout: 4 × {buf_bytes}-byte buffers (A, B, R, S). */",
        f"#define COMPGEN_BUF_BYTES {buf_bytes}",
        "",
        "cg_status_t compgen_model_forward(const float *input_f32,",
        "                                  float *output_f32,",
        "                                  void *arena, size_t arena_size)",
        "{",
        f"    if (arena_size < (size_t)({arena_bytes}u)) return CG_STATUS_RESOURCE_EXHAUSTED;",
        "    uint8_t *base = (uint8_t *)arena;",
        "    float *A = (float *)(base + 0 * COMPGEN_BUF_BYTES);",
        "    float *B = (float *)(base + 1 * COMPGEN_BUF_BYTES);",
        "    float *R = (float *)(base + 2 * COMPGEN_BUF_BYTES);",
        "    float *S = (float *)(base + 3 * COMPGEN_BUF_BYTES);",
        "    (void)S; /* reserved for future shortcut buffers */",
        "",
    ]

    def emit(s: str) -> None:
        lines.append(s)

    Cin, Hin, Win = sample_input_shape
    # Stem: conv(3→32, 3x3, s=2, p=1) + ReLU → A
    so = stem_out.shape[1:]
    assert tuple(so) == (32, 32, 32), f"unexpected stem out shape {so}"
    emit(f"    /* stem: Conv2d({Cin}→32, 3x3, s=2, p=1) + BN + ReLU */")
    emit(_fmt_conv("input_f32", "A", (Cin, Hin, Win), (32, 32, 32),
                   stride=2, pad=1, kernel=3,
                   w_off=stem_w.offset_floats, b_off=stem_b.offset_floats))
    op_counts["conv2d"] += 1
    emit(_fmt_relu("A", 32 * 32 * 32))
    op_counts["relu"] += 1
    emit("")
    current = "A"
    other = "B"
    cur_shape = (32, 32, 32)

    # Stages
    stage_shapes = [t.shape[1:] for stage in stage_outputs for t in stage]
    block_idx = 0
    for stage_idx in range(3):
        stage = getattr(model, f"stage{stage_idx + 1}")
        for within_stage, blk in enumerate(stage):
            refs = block_refs[block_idx]
            out_shape = tuple(stage_outputs[stage_idx][within_stage].shape[1:])
            stride = refs["stride"]
            in_ch = refs["in_ch"]
            assert in_ch == cur_shape[0], f"block input channel mismatch: {in_ch} vs {cur_shape[0]}"
            mid_shape = out_shape  # conv1 + conv2 outputs have same spatial as the block output
            emit(f"    /* stage{stage_idx + 1} block{within_stage}: "
                 f"in={cur_shape} out={out_shape} stride={stride} "
                 f"{'res=1x1conv' if refs['has_shortcut'] else 'res=identity'} */")
            # h1 = conv1(current) -> other
            emit(_fmt_conv(current, other, cur_shape, mid_shape,
                           stride=stride, pad=1, kernel=3,
                           w_off=refs["c1_w"].offset_floats,
                           b_off=refs["c1_b"].offset_floats))
            op_counts["conv2d"] += 1
            emit(_fmt_relu(other, mid_shape[0] * mid_shape[1] * mid_shape[2]))
            op_counts["relu"] += 1
            # h2 = conv2(other) -> R
            emit(_fmt_conv(other, "R", mid_shape, out_shape,
                           stride=1, pad=1, kernel=3,
                           w_off=refs["c2_w"].offset_floats,
                           b_off=refs["c2_b"].offset_floats))
            op_counts["conv2d"] += 1
            if refs["has_shortcut"]:
                # sc = conv1x1(current, stride) -> S
                emit(_fmt_conv(current, "S", cur_shape, out_shape,
                               stride=stride, pad=0, kernel=1,
                               w_off=refs["sc_w"].offset_floats,
                               b_off=refs["sc_b"].offset_floats))
                op_counts["conv2d"] += 1
                # current = relu(R + S) -> current (OK: current buffer may have larger shape; fine because we're writing less)
                emit(_fmt_add("R", "S", current,
                              out_shape[0] * out_shape[1] * out_shape[2]))
            else:
                # current = relu(R + current) -> current
                emit(_fmt_add("R", current, current,
                              out_shape[0] * out_shape[1] * out_shape[2]))
            op_counts["add"] += 1
            emit(_fmt_relu(current, out_shape[0] * out_shape[1] * out_shape[2]))
            op_counts["relu"] += 1
            emit("")
            cur_shape = out_shape
            block_idx += 1

    # Global avgpool: current [C, H, W] -> other [C]
    emit(f"    /* pool: AdaptiveAvgPool2d(1) over {cur_shape} -> [{cur_shape[0]}] */")
    emit(f"    cg_global_avgpool_f32({current}, {other}, "
         f"{cur_shape[0]}, {cur_shape[1]}, {cur_shape[2]});")
    op_counts["avgpool"] += 1
    emit("")

    # Linear head: other -> output_f32
    assert head_w.num_floats == model.head.in_features * model.head.out_features
    emit(f"    /* head: Linear({model.head.in_features} -> {model.head.out_features}) */")
    if head_b is not None:
        emit(f"    cg_linear_f32({other}, output_f32, "
             f"{model.head.in_features}, {model.head.out_features}, "
             f"blob_f32({head_w.offset_floats}), blob_f32({head_b.offset_floats}));")
    else:
        emit(f"    cg_linear_f32({other}, output_f32, "
             f"{model.head.in_features}, {model.head.out_features}, "
             f"blob_f32({head_w.offset_floats}), (const float *)0);")
    op_counts["linear"] += 1
    emit("    return CG_STATUS_OK;")
    emit("}")
    emit("")

    forward_c = "\n".join(lines)

    input_bytes = 4 * Cin * Hin * Win
    output_bytes = 4 * model.head.out_features

    return LoweredModel(
        name=model_name,
        forward_c_source=forward_c,
        weights_blob=weights_blob,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        arena_bytes=arena_bytes,
        num_params=num_params,
        op_counts=op_counts,
    )


__all__ = ["LoweredModel", "lower_cnn_to_c"]
