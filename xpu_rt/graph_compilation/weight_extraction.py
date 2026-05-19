"""Extract static-weight tensor data from a captured graph.

Bridge background
-----------------

The XNNPACK provider's emitted C kernel artifact carries an embedded
``static const float kWeights[] = { ... }`` array that gets passed to
``xpu_rt_xnn_create`` as the operator's static weights. For
``XPU_RT_XNN_OP_FULLY_CONNECTED_F32`` the bridge expects the layout

  weight[out_c][in_c] (row-major)  ||  bias[out_c]   (optional)

— i.e. ``out_c * in_c + out_c`` fp32 values total when bias is
present, packed back-to-back. See
``runtime/native/libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.c``'s
``XPU_RT_XNN_OP_FULLY_CONNECTED_F32`` arm for the exact convention.

What this module does
---------------------

This module exposes pure-Python helpers that take a captured
``torch.export.ExportedProgram`` (or its underlying
``state_dict``) plus a region's metadata describing which FX
``get_attr`` nodes are the region's weight + bias, and returns the
packed fp32 list that goes into

    KernelCodegenRequest.extras["static_weights_f32"]

— ready for ``XnnpackProvider.propose()`` to bake into the emitted
kernel C. Tests for the Phase C cross-compile orchestrator and the
Phase E parity check hand-pack these values; this helper makes the
production-pipeline path do the same thing deterministically.

The helpers are intentionally narrow:

  * v1 supports only fp32 FC (``nn.Linear``-shaped) regions. Any other
    op-kind returns ``None`` from
    :func:`extract_static_weights_for_region` so the caller can fall
    back to the legacy placeholder path without crashing.
  * No state_dict introspection beyond direct lookup by attribute
    name. The pipeline already records the FX node's ``target``
    string (e.g. ``"linear.weight"``); we use that verbatim.

Production wiring
-----------------

The intended call site is the compile pipeline's
``KernelCodegenRequest`` builder. A future change to that builder
should look something like::

    extras: dict[str, Any] = {}
    weights = extract_static_weights_for_region(
        region, exported_program, shape_dims=shape_dims,
    )
    if weights is not None:
        extras["static_weights_f32"] = weights
        extras["shape_dims"] = shape_dims
        extras["int_params"] = [0]
        extras["float_params"] = [-1e30, 1e30]
    request = KernelCodegenRequest(
        task_id=..., contract=..., target=...,
        artifact_dir=..., extras=extras,
    )

Until that wiring lands, the helper is callable from any provider
that wants to opportunistically pull weights from the captured graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RegionWeightSpec:
    """Minimal region descriptor the extraction helper needs.

    The compile pipeline carries richer per-region info; this
    dataclass extracts just what weight extraction cares about so the
    helper isn't coupled to the larger ``RegionInfo`` schema.

    Attributes
    ----------
    op_kind
        Contract op kind string. Only ``"matmul"`` / ``"fully_connected"``
        / ``"linear"`` are supported in v1; other kinds short-circuit.
    weight_attr_name
        FX node ``target`` for the ``get_attr`` node loading the weight
        tensor (e.g. ``"weight"`` for a single ``nn.Linear``, or
        ``"linear.weight"`` for a module-nested one). Must match a key
        in ``state_dict`` when stripped of any leading ``_orig_mod.``
        prefix the export pass may have added.
    bias_attr_name
        Same shape as ``weight_attr_name`` but for the bias. ``None``
        when the layer has ``bias=False``.
    in_c
        Input channel dimension expected by the kernel (the K
        dimension of an FC matmul).
    out_c
        Output channel dimension expected by the kernel.
    """

    op_kind: str
    weight_attr_name: str
    bias_attr_name: str | None
    in_c: int
    out_c: int


_FC_OP_KINDS = frozenset({"matmul", "fully_connected", "linear", "dense"})


def pack_linear_weights(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> list[float]:
    """Pack ``nn.Linear``-shape weight + bias into the bridge's
    fp32 row-major layout.

    Parameters
    ----------
    weight
        Float tensor of shape ``(out_c, in_c)``. Any floating dtype is
        coerced to ``torch.float32``; integer dtypes raise
        :class:`TypeError`.
    bias
        Optional float tensor of shape ``(out_c,)``. ``None`` means no
        bias (XNNPACK FC f32 supports this).

    Returns
    -------
    Flat list of fp32 values in the order
    ``weight[0][0..in_c], weight[1][0..in_c], ..., bias[0..out_c]``.
    """
    if not torch.is_tensor(weight):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}")
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2-D (out_c, in_c); got shape {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise TypeError(
            f"weight must be a floating-point tensor; got dtype {weight.dtype}"
        )
    out_c, in_c = weight.shape
    w = weight.detach().contiguous().to(torch.float32)
    flat: list[float] = []
    # Row-major: for each output channel, append all input-channel weights.
    for o in range(out_c):
        for i in range(in_c):
            flat.append(float(w[o, i].item()))
    if bias is not None:
        if not torch.is_tensor(bias):
            raise TypeError(f"bias must be a torch.Tensor, got {type(bias)}")
        if bias.ndim != 1 or bias.shape[0] != out_c:
            raise ValueError(
                f"bias shape mismatch: expected ({out_c},), got "
                f"{tuple(bias.shape)}"
            )
        b = bias.detach().contiguous().to(torch.float32)
        for o in range(out_c):
            flat.append(float(b[o].item()))
    return flat


def _resolve_state_dict_key(
    state_dict: dict[str, torch.Tensor],
    attr_name: str,
) -> str | None:
    """Best-effort match an FX ``get_attr`` target to a state_dict key.

    torch.export occasionally rewrites parameter names with a leading
    ``_orig_mod.`` or ``L__self___`` prefix; we try a few common
    variants before giving up.
    """
    if attr_name in state_dict:
        return attr_name
    candidates = [
        attr_name,
        f"_orig_mod.{attr_name}",
        f"L__self___{attr_name}",
        attr_name.replace(".", "_"),
    ]
    for cand in candidates:
        if cand in state_dict:
            return cand
    # Last-resort: substring match against the *full* attr_name —
    # useful when torch.export's name-mangling prepends a prefix we
    # don't know. NB: deliberately matches only suffixes that contain
    # all of attr_name (no tail-only match) — a bare ``weight`` lookup
    # against a state_dict containing ``other.module.weight`` should
    # NOT match, since the contract identified a specific qualified
    # name.
    for k in state_dict:
        if k.endswith("." + attr_name) or k == attr_name:
            return k
    return None


def extract_static_weights_for_region(
    spec: RegionWeightSpec,
    state_dict: dict[str, torch.Tensor],
) -> list[float] | None:
    """Extract + pack static weights for an FC region.

    Parameters
    ----------
    spec
        Minimal region descriptor; see :class:`RegionWeightSpec`.
    state_dict
        Result of ``exported_program.module().state_dict()`` or
        equivalent; maps FX attribute names to their tensors.

    Returns
    -------
    Packed fp32 list ready for
    ``KernelCodegenRequest.extras["static_weights_f32"]``, or ``None``
    when the region is not a supported FC variant or the requested
    tensors aren't in the state_dict.
    """
    if spec.op_kind.lower() not in _FC_OP_KINDS:
        log.info(
            "weight_extraction: op_kind not FC, skipping",
            op_kind=spec.op_kind,
        )
        return None

    weight_key = _resolve_state_dict_key(state_dict, spec.weight_attr_name)
    if weight_key is None:
        log.warning(
            "weight_extraction: weight attr not in state_dict",
            attr=spec.weight_attr_name,
            available=list(state_dict.keys())[:8],
        )
        return None
    weight = state_dict[weight_key]
    if weight.shape != (spec.out_c, spec.in_c):
        log.warning(
            "weight_extraction: weight shape mismatch — expected "
            f"({spec.out_c}, {spec.in_c}), got {tuple(weight.shape)}",
            attr=weight_key,
        )
        return None

    bias: torch.Tensor | None = None
    if spec.bias_attr_name is not None:
        bias_key = _resolve_state_dict_key(state_dict, spec.bias_attr_name)
        if bias_key is None:
            log.warning(
                "weight_extraction: bias attr not in state_dict",
                attr=spec.bias_attr_name,
            )
            # Without a bias the bridge falls back to NULL-bias path,
            # which the FC f32 case supports. We could continue here,
            # but a missing bias is usually a sign the contract is
            # mis-labelled, so be defensive and bail.
            return None
        bias = state_dict[bias_key]

    return pack_linear_weights(weight, bias)


def load_state_dict_from_bundle(bundle_dir: Path | str) -> dict[str, torch.Tensor]:
    """Load the captured graph's state_dict from a bundle on disk.

    Looks under ``<bundle>/00_graph_capture/exported_program.pt2``
    (the canonical capture location) and falls back to
    ``<bundle>/exported_program.pt2`` for flat bundle layouts. Returns
    an empty dict when neither exists or torch.export.load fails — the
    caller can then decide to skip weight extraction and emit a
    placeholder kernel.
    """
    bundle_dir = Path(bundle_dir)
    for candidate in (
        bundle_dir / "00_graph_capture" / "exported_program.pt2",
        bundle_dir / "exported_program.pt2",
    ):
        if not candidate.is_file():
            continue
        try:
            ep = torch.export.load(str(candidate))
        except Exception as exc:
            log.warning(
                "weight_extraction: torch.export.load failed",
                path=str(candidate),
                error=repr(exc),
            )
            continue
        try:
            return dict(ep.module().state_dict())
        except Exception as exc:
            log.warning(
                "weight_extraction: state_dict access failed",
                error=repr(exc),
            )
    return {}


def populate_provider_extras(
    extras: dict[str, Any],
    spec: RegionWeightSpec,
    state_dict: dict[str, torch.Tensor],
    *,
    int_params: list[int] | None = None,
    float_params: list[float] | None = None,
) -> dict[str, Any]:
    """Mutate / return an ``extras`` dict suitable for
    :class:`xpu_rt.providers.kernel_provider.KernelCodegenRequest`.

    Fields populated when extraction succeeds:

      * ``shape_dims``         — ``[in_c, out_c]`` for FC.
      * ``int_params``         — defaults to ``[0]``.
      * ``float_params``       — defaults to wide ``[-1e30, 1e30]`` clamps.
      * ``static_weights_f32`` — packed fp32 list.

    When extraction returns ``None`` (unsupported op or missing
    tensors) the dict is left untouched so the provider falls back to
    its v1 placeholder emit. The caller can detect this by checking
    ``"static_weights_f32" in extras`` afterwards.
    """
    weights = extract_static_weights_for_region(spec, state_dict)
    if weights is None:
        return extras
    extras["shape_dims"] = [spec.in_c, spec.out_c]
    extras["int_params"] = list(int_params or [0])
    extras["float_params"] = list(float_params or [-1.0e30, 1.0e30])
    extras["static_weights_f32"] = weights
    return extras
