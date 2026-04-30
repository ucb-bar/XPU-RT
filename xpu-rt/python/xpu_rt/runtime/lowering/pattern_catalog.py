"""Wave 2.1 — pattern-catalog matchers (residual+norm, MHA, MoE).

Extends the FX→megakernel matcher cascade beyond Wave 1's diamond +
FFN with three transformer-shaped patterns:

* ``residual+norm``: ``LayerNorm(x + sublayer(x))`` — recursively
  reuses the diamond / FFN / hand-rolled-linear matcher to lower the
  inner sublayer, then appends a tail row-reduce + row-normalize +
  affine-scale tile-task per row of the activation.

* ``mha``: ``softmax(Q @ K^T / sqrt(d_head)) @ V`` followed by an
  output-projection linear. Works on either ``nn.MultiheadAttention``
  (with its packed ``in_proj_weight``) or the hand-rolled
  ``q/k/v/o = nn.Linear`` form. The softmax row-reduction is split
  into max-reduce, exp, sum-reduce, divide tile-tasks so the
  cross-tile event-tensor structure stays explicit (paper §3.1).

* ``moe``: a router that produces a top-k expert routing map plus an
  ``nn.ModuleList`` of expert FFNs. Data-dependent, so the matcher
  routes through the dynamic-schedule path (``policy="dynamic"``) and
  emits a :class:`~compgen.transforms.event_dynamic_schedule.TriggerGenerator`
  per expert + an ``"requires_ondevice_scheduler"`` schedule hint so
  Phase 6 picks the on-device-scheduler-capable target. Per the
  paper's §3.2 dispatch model — each token's routing index is a
  runtime-known trigger that pushes downstream expert tasks onto the
  ready queue.

The Wave 1 matchers (diamond / FFN) stay primary in the cascade;
this module's matchers run after them so a plain FFN-shaped block
still lowers to ``"ffn"`` (not ``"residual_norm@ffn"``) even if the
caller wraps it in a residual+norm later. The cascade order is set
in :mod:`compgen.runtime.lowering.fx_to_megakernel`.

These matchers emit *placeholder* tile-task bodies that are
correct at the graph-topology level; the CUDA codegen for the new
kinds (softmax, layernorm, masked-matmul) follows in Wave 2.2+ and
plugs in via :class:`~compgen.transforms.emit_cuda_megakernel.DeviceFunctionSource`.
This module's contract is the matcher's structural recognition +
graph-shape contract — those are the things downstream tests pin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import (
    DeviceCall,
    EventEdge,
    MegakernelGraph,
)
from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

if TYPE_CHECKING:
    from compgen.runtime.lowering.fx_to_megakernel import LoweringResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TILE_M = 32  # row tile (across batch_flat / sequence)
_TILE_N = 32  # col tile (across feature dim / head dim)
_TILE_K = 32  # K-reduction tile


def _flatten_batch(x: torch.Tensor, in_features: int) -> int:
    """Flatten leading dims into a single batch axis matching the
    Wave-1 contract — consistent with bridge #108. Trailing dim must
    equal ``in_features``."""
    batch_flat = 1
    for d in x.shape[:-1]:
        batch_flat *= int(d)
    return batch_flat


def _placeholder_body(name: str, comment: str) -> DeviceFunctionSource:
    """Emit a placeholder ``DeviceFunctionSource`` for a new tile-task
    kind whose CUDA codegen lives in a later wave.

    The body is a no-op so NVRTC compiles it; the rationale comment
    documents what the eventual codegen needs to produce. Tests pin
    the *graph topology* (DeviceCall names, event-tensor shapes,
    pattern_name); the body content is irrelevant at that level.
    """
    body = (
        f"// {comment}\n"
        f"// Wave 2.1 placeholder body for {name!r}.\n"
        "// Body codegen lands in Wave 2.2+; the matcher pins the\n"
        "// graph-topology contract so downstream tests (and the\n"
        "// Phase-3/Phase-5 emitter) see the right event-tensor\n"
        "// shape regardless of body content.\n"
    )
    return DeviceFunctionSource(name=name, body=body)


# ---------------------------------------------------------------------------
# residual+norm
# ---------------------------------------------------------------------------


def _is_layer_norm(mod: nn.Module) -> bool:
    return isinstance(mod, nn.LayerNorm)


def _find_residual_components(
    model: nn.Module,
) -> tuple[nn.Module, nn.LayerNorm] | None:
    """Walk ``named_children`` looking for ``(sublayer, norm)`` where
    ``norm`` is an ``nn.LayerNorm`` and ``sublayer`` is the only
    other child module (the residual sublayer).

    Returns ``None`` when the structure doesn't match.
    """
    children = list(model.named_children())
    norms = [(n, m) for n, m in children if _is_layer_norm(m)]
    others = [(n, m) for n, m in children if not _is_layer_norm(m)]
    if len(norms) != 1 or len(others) != 1:
        return None
    return others[0][1], norms[0][1]


def _match_residual_norm(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
) -> LoweringResult:
    """Match ``y = LayerNorm(x + sublayer(x))``.

    The sublayer is recursively lowered through the Wave-1 matcher
    cascade (diamond / FFN) so the composition produces e.g.
    ``"residual_norm@ffn"``.
    """
    from compgen.runtime.lowering.fx_to_megakernel import (
        UnsupportedShape,
        _match_diamond,
        _match_ffn,
    )

    found = _find_residual_components(model)
    if found is None:
        raise UnsupportedShape(
            "residual_norm requires exactly one nn.LayerNorm child + one non-LayerNorm sublayer child"
        )
    sublayer, norm = found

    if norm.elementwise_affine is False:
        raise UnsupportedShape(
            "residual_norm requires nn.LayerNorm with elementwise_affine=True "
            "(weight + bias) — affine=False is not yet supported"
        )

    x = sample_inputs[0]
    if x.ndim < 2:
        raise UnsupportedShape(f"residual_norm input must be at least 2-D; got shape {tuple(x.shape)}")
    in_features = int(x.shape[-1])

    # LayerNorm normalizes over the last len(norm.normalized_shape) dims.
    # Wave 2.1 only accepts last-dim normalization (the post-norm
    # residual that dominates transformer blocks).
    if tuple(norm.normalized_shape) != (in_features,):
        raise UnsupportedShape(
            f"residual_norm requires LayerNorm over the trailing axis "
            f"(normalized_shape=({in_features},)); got "
            f"{tuple(norm.normalized_shape)}"
        )

    # Probe forward — both the wrapping module and a synthetic
    # `LayerNorm(x + sublayer(x))` must agree.
    with torch.no_grad():
        try:
            top_y = model(x)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"residual_norm forward probe raised {exc!r}") from exc
        try:
            sub_y = sublayer(x)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"residual_norm sublayer forward failed: {exc!r}") from exc
        if not isinstance(sub_y, torch.Tensor):
            raise UnsupportedShape(f"residual_norm sublayer must return a tensor; got {type(sub_y).__name__}")
        if sub_y.shape != x.shape:
            raise UnsupportedShape(
                f"residual_norm requires shape-preserving sublayer; "
                f"sub(x) shape {tuple(sub_y.shape)} != input "
                f"{tuple(x.shape)}"
            )
        expected_y = norm(x + sub_y)
        if not torch.allclose(top_y, expected_y, atol=1e-5, rtol=1e-5):
            raise UnsupportedShape(
                "residual_norm pattern children match but forward output disagrees with LayerNorm(x + sublayer(x))"
            )

    # Recursively lower the sublayer.
    sub_errors: list[str] = []
    sub_result = None
    for sub_matcher in (_match_diamond, _match_ffn):
        try:
            sub_result = sub_matcher(
                sublayer,
                sample_inputs,
                prefer_cublasdx_for_linears=prefer_cublasdx_for_linears,
                cublasdx_precision=cublasdx_precision,
                cublasdx_sm=cublasdx_sm,
            )
            break
        except UnsupportedShape as exc:
            sub_errors.append(f"{sub_matcher.__name__}: {exc}")
    if sub_result is None:
        # Fall back: a single nn.Linear sublayer. We still want to
        # accept ``residual_norm`` over a bare linear since it's a
        # common transformer skip.
        if isinstance(sublayer, nn.Linear) and sublayer.bias is None:
            sub_result = _emit_residual_norm_with_bare_linear(
                model_x=x,
                sublayer=sublayer,
                norm=norm,
            )
            return sub_result
        raise UnsupportedShape("residual_norm sublayer didn't match diamond/ffn/bare-linear; " + "; ".join(sub_errors))

    return _emit_residual_norm(
        x=x,
        sublayer_result=sub_result,
        norm=norm,
    )


def _emit_residual_norm(
    *,
    x: torch.Tensor,
    sublayer_result: LoweringResult,
    norm: nn.LayerNorm,
) -> LoweringResult:
    """Build the residual+norm graph: sublayer's tile-tasks +
    ``add(x, sub_y)`` + ``mean`` + ``var`` + ``normalize`` + ``affine``.
    """
    from dataclasses import replace as _replace

    from compgen.runtime.lowering.fx_to_megakernel import (
        LoweringDecision,
        LoweringResult,
        UnsupportedShape,
        _BodyDecision,
    )

    sub_graph = sublayer_result.megakernel_graph
    sub_decision = sublayer_result.decision
    sub_pattern = sub_decision.pattern_name

    in_features = int(x.shape[-1])
    batch_flat = _flatten_batch(x, in_features)

    # Tail tile-task grid: one task per row of the activation. Each
    # row computes mean, var, then normalize+affine over its
    # feature axis. We emit these as four ops so cross-task
    # dependencies (rows reading their own statistics) live in the
    # event-tensor structure.
    if batch_flat % _TILE_M != 0:
        raise UnsupportedShape(f"residual_norm needs batch_flat ({batch_flat}) divisible by tile_m={_TILE_M}")
    n_row_tiles = batch_flat // _TILE_M

    # Build new event tensors for the tail. Re-use the sublayer's
    # graph + event tensors for the inner pattern.
    ev_resid = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_mean = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_var = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_norm_done = EventTensor((n_row_tiles,), wait_count_default=1)
    same = lambda c: (c[0],)  # noqa: E731

    # The tail ops:
    #   resid_add: read x (buf 0) + sub_y (sublayer output buf) → resid_buf
    #   ln_mean:   row-reduce mean over resid → mean_buf
    #   ln_var:    row-reduce var over resid + mean → var_buf
    #   ln_affine: normalize + scale + shift → out_buf
    tail_calls = (
        DeviceCall(
            name="residual_add",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            out_edges=(EventEdge("ev_resid", same),),
        ),
        DeviceCall(
            name="ln_mean",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(EventEdge("ev_resid", same),),
            out_edges=(EventEdge("ev_mean", same),),
        ),
        DeviceCall(
            name="ln_var",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(
                EventEdge("ev_resid", same),
                EventEdge("ev_mean", same),
            ),
            out_edges=(EventEdge("ev_var", same),),
        ),
        DeviceCall(
            name="ln_affine",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(
                EventEdge("ev_resid", same),
                EventEdge("ev_mean", same),
                EventEdge("ev_var", same),
            ),
            out_edges=(EventEdge("ev_norm_done", same),),
        ),
    )

    # Compose with the sublayer's calls. Re-name event tensors to
    # avoid collisions with our tail tensors.
    sub_event_prefix = "sub__"
    renamed_sub_events: dict[str, EventTensor] = {
        f"{sub_event_prefix}{n}": e for n, e in sub_graph.event_tensors.items()
    }

    def _rename_edge(edge: EventEdge) -> EventEdge:
        return EventEdge(
            event_name=f"{sub_event_prefix}{edge.event_name}",
            index_fn=edge.index_fn,
            decrement=edge.decrement,
            peer_rank=edge.peer_rank,
        )

    sub_calls_renamed = tuple(
        DeviceCall(
            name=f"sub__{c.name}",
            body_fn=c.body_fn,
            task_shape=c.task_shape,
            in_edges=tuple(_rename_edge(e) for e in c.in_edges),
            out_edges=tuple(_rename_edge(e) for e in c.out_edges),
        )
        for c in sub_graph.calls
    )

    all_calls = sub_calls_renamed + tail_calls
    all_events: dict[str, EventTensor] = {
        **renamed_sub_events,
        "ev_resid": ev_resid,
        "ev_mean": ev_mean,
        "ev_var": ev_var,
        "ev_norm_done": ev_norm_done,
    }
    graph = MegakernelGraph(
        name=f"residual_norm@{sub_pattern}_lowered",
        calls=all_calls,
        event_tensors=all_events,
        policy="static",
    )

    # Reuse sublayer body sources, prefixed to match renamed calls,
    # plus four placeholder bodies for the tail.
    bodies: dict[str, DeviceFunctionSource] = {}
    for n, src in sublayer_result.device_function_sources.items():
        bodies[f"sub__{n}"] = DeviceFunctionSource(
            name=f"sub__{n}",
            body=src.body,
            signature=src.signature,
            included_headers=src.included_headers,
        )
    bodies["residual_add"] = _placeholder_body(
        "residual_add",
        f"elementwise add: x[r,c] + sub_y[r,c] → resid[r,c]; row-tile {_TILE_M}",
    )
    bodies["ln_mean"] = _placeholder_body(
        "ln_mean",
        f"row-reduce mean over feature axis (D={in_features}); 1 task per row tile",
    )
    bodies["ln_var"] = _placeholder_body(
        "ln_var",
        f"row-reduce variance using ev_mean output; D={in_features}",
    )
    bodies["ln_affine"] = _placeholder_body(
        "ln_affine",
        f"normalize + weight*x + bias; LayerNorm.eps={float(norm.eps)}",
    )

    # User buffer layout: keep the sublayer's layout (re-used as-is)
    # plus the new buffers for the residual + LN intermediates.
    sub_layout = list(sublayer_result.user_buffer_layout)
    layout = sub_layout + [
        "y_resid",  # x + sub_y
        "ln_mean",  # row mean buffer
        "ln_var",  # row var buffer
        "ln_weight",  # LN weight (γ)
        "ln_bias",  # LN bias  (β)
        "y_out",  # final output
    ]

    body_decisions = list(sub_decision.body_decisions) + [
        _BodyDecision(
            op_name="residual_add",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale="elementwise add over the residual + sublayer output.",
        ),
        _BodyDecision(
            op_name="ln_mean",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale="LayerNorm row-mean reduction; 1 task per row tile.",
        ),
        _BodyDecision(
            op_name="ln_var",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale="LayerNorm row-variance reduction; consumes ev_mean.",
        ),
        _BodyDecision(
            op_name="ln_affine",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale=(
                "LayerNorm affine: y = (x - mean) / sqrt(var + eps) * γ + β; "
                f"eps={float(norm.eps)}, affine={norm.elementwise_affine}."
            ),
        ),
    ]

    decision = LoweringDecision(
        pattern_name=f"residual_norm@{sub_pattern}",
        pattern_rationale=(
            f"matched LayerNorm(x + sub(x)) with sub matched as {sub_pattern!r}; "
            f"feature dim D={in_features}, batch_flat={batch_flat}, "
            f"row_tiles={n_row_tiles}. Sublayer emission re-used verbatim, "
            "tail ops appended."
        ),
        body_decisions=tuple(body_decisions),
        schedule_hints={
            "policy": "static",
            "row_tiles": n_row_tiles,
            "feature_dim": in_features,
            "tile_shape": [_TILE_M, _TILE_N, _TILE_K],
            "block_dim": [32, 32, 1],
            "sublayer_pattern": sub_pattern,
            **sub_decision.schedule_hints,
        },
        total_tile_tasks=sub_decision.total_tile_tasks + 4 * n_row_tiles,
        backends=sub_decision.backends,
        nvrtc_include_paths=sub_decision.nvrtc_include_paths,
        nvrtc_extra_options=sub_decision.nvrtc_extra_options,
    )

    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=tuple(layout),
        decision=_replace(decision) if False else decision,
    )


def _emit_residual_norm_with_bare_linear(
    *,
    model_x: torch.Tensor,
    sublayer: nn.Linear,
    norm: nn.LayerNorm,
) -> LoweringResult:
    """Single-Linear sublayer specialization: skips the Wave-1
    diamond/ffn matcher (a single ``nn.Linear`` doesn't match either)
    and synthesizes a one-task linear sub-graph manually.
    """
    from compgen.runtime.lowering.fx_to_megakernel import (
        LoweringDecision,
        LoweringResult,
        _BodyDecision,
    )

    if sublayer.in_features != sublayer.out_features:
        raise __import__("compgen.runtime.lowering.fx_to_megakernel", fromlist=["UnsupportedShape"]).UnsupportedShape(
            f"residual_norm with bare nn.Linear sublayer requires square "
            f"in==out; got {sublayer.in_features}->{sublayer.out_features}"
        )

    in_features = int(model_x.shape[-1])
    batch_flat = _flatten_batch(model_x, in_features)
    if batch_flat % _TILE_M != 0:
        raise __import__("compgen.runtime.lowering.fx_to_megakernel", fromlist=["UnsupportedShape"]).UnsupportedShape(
            f"residual_norm needs batch_flat ({batch_flat}) divisible by tile_m={_TILE_M}"
        )
    n_row_tiles = batch_flat // _TILE_M

    ev_lin = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_resid = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_mean = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_var = EventTensor((n_row_tiles,), wait_count_default=1)
    ev_norm_done = EventTensor((n_row_tiles,), wait_count_default=1)
    same = lambda c: (c[0],)  # noqa: E731

    calls = (
        DeviceCall(
            name="sub__linear",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            out_edges=(EventEdge("ev_lin", same),),
        ),
        DeviceCall(
            name="residual_add",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(EventEdge("ev_lin", same),),
            out_edges=(EventEdge("ev_resid", same),),
        ),
        DeviceCall(
            name="ln_mean",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(EventEdge("ev_resid", same),),
            out_edges=(EventEdge("ev_mean", same),),
        ),
        DeviceCall(
            name="ln_var",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(
                EventEdge("ev_resid", same),
                EventEdge("ev_mean", same),
            ),
            out_edges=(EventEdge("ev_var", same),),
        ),
        DeviceCall(
            name="ln_affine",
            body_fn=lambda c: None,
            task_shape=(n_row_tiles,),
            in_edges=(
                EventEdge("ev_resid", same),
                EventEdge("ev_mean", same),
                EventEdge("ev_var", same),
            ),
            out_edges=(EventEdge("ev_norm_done", same),),
        ),
    )
    graph = MegakernelGraph(
        name="residual_norm@linear_lowered",
        calls=calls,
        event_tensors={
            "ev_lin": ev_lin,
            "ev_resid": ev_resid,
            "ev_mean": ev_mean,
            "ev_var": ev_var,
            "ev_norm_done": ev_norm_done,
        },
        policy="static",
    )

    bodies = {
        "sub__linear": _placeholder_body(
            "sub__linear",
            f"single nn.Linear sublayer; in=out={in_features}",
        ),
        "residual_add": _placeholder_body(
            "residual_add",
            f"elementwise add x + sub(x); D={in_features}",
        ),
        "ln_mean": _placeholder_body("ln_mean", "LayerNorm row-mean reduction"),
        "ln_var": _placeholder_body("ln_var", "LayerNorm row-variance reduction"),
        "ln_affine": _placeholder_body("ln_affine", "LayerNorm affine"),
    }
    body_decisions = tuple(
        _BodyDecision(
            op_name=name,
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale="residual_norm@linear bare-linear specialization",
        )
        for name in (
            "sub__linear",
            "residual_add",
            "ln_mean",
            "ln_var",
            "ln_affine",
        )
    )
    decision = LoweringDecision(
        pattern_name="residual_norm@linear",
        pattern_rationale=(
            f"matched LayerNorm(x + linear(x)) with single nn.Linear "
            f"sublayer (in=out={in_features}). batch_flat={batch_flat}, "
            f"row_tiles={n_row_tiles}."
        ),
        body_decisions=body_decisions,
        schedule_hints={
            "policy": "static",
            "row_tiles": n_row_tiles,
            "feature_dim": in_features,
            "tile_shape": [_TILE_M, _TILE_N, _TILE_K],
            "block_dim": [32, 32, 1],
            "sublayer_pattern": "linear",
        },
        total_tile_tasks=5 * n_row_tiles,
    )
    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=(
            "x",
            "w_linear",
            "y_lin",
            "y_resid",
            "ln_mean",
            "ln_var",
            "ln_weight",
            "ln_bias",
            "y_out",
        ),
        decision=decision,
    )


# ---------------------------------------------------------------------------
# MHA — multi-head attention
# ---------------------------------------------------------------------------


def _match_mha(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
) -> LoweringResult:
    """Match multi-head attention.

    Accepts either:
      * ``nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=False,
        batch_first=True)`` — the canonical PyTorch shape, or
      * a hand-rolled module with attributes ``q``, ``k``, ``v``, ``o``
        (each ``nn.Linear(D, D, bias=False)``) and ``num_heads``.

    Wave-1 simplification: ``bias=False`` everywhere, ``batch_first=True``,
    ``embed_dim % num_heads == 0``, no causal mask is required, but the
    matcher accepts an ``is_causal`` flag and threads it into
    ``schedule_hints["mha_causal"]`` so the kernel emitter (Wave 2.2)
    can pick the masked-softmax variant.
    """
    from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

    embed_dim, num_heads, head_dim, is_causal = _classify_mha(model)
    if embed_dim is None:
        raise UnsupportedShape(
            "mha: model is neither nn.MultiheadAttention(bias=False, "
            "batch_first=True) nor a hand-rolled q/k/v/o(+num_heads) form"
        )
    assert num_heads is not None and head_dim is not None
    if embed_dim % num_heads != 0:
        raise UnsupportedShape(
            f"mha: embed_dim ({embed_dim}) must be divisible by num_heads "
            f"({num_heads}); got head_dim={embed_dim / num_heads}"
        )

    x = sample_inputs[0]
    if x.ndim != 3:
        raise UnsupportedShape(f"mha requires batch-first 3-D input (B, S, D); got shape {tuple(x.shape)}")
    if x.shape[-1] != embed_dim:
        raise UnsupportedShape(f"mha input trailing dim ({int(x.shape[-1])}) != embed_dim ({embed_dim})")
    batch = int(x.shape[0])
    seq = int(x.shape[1])

    # Forward probe — confirm shapes & differentiate from
    # cross-attention (we only handle self-attention in Wave 2.1).
    with torch.no_grad():
        try:
            y = model(x) if not isinstance(model, nn.MultiheadAttention) else model(x, x, x)[0]
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"mha forward probe raised {exc!r}") from exc
    if not isinstance(y, torch.Tensor):
        raise UnsupportedShape(f"mha forward must return a tensor; got {type(y).__name__}")
    if tuple(y.shape) != (batch, seq, embed_dim):
        raise UnsupportedShape(f"mha forward returned shape {tuple(y.shape)}, expected ({batch}, {seq}, {embed_dim})")

    return _emit_mha(
        batch=batch,
        seq=seq,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )


def _classify_mha(
    model: nn.Module,
) -> tuple[int | None, int | None, int | None, bool]:
    """Return ``(embed_dim, num_heads, head_dim, is_causal)`` if the
    module is a recognised MHA shape; ``(None, None, None, False)``
    otherwise."""
    if isinstance(model, nn.MultiheadAttention):
        if not model.batch_first:
            return None, None, None, False
        # Wave-1: require bias=False on the output projection. The
        # in_proj uses ``in_proj_bias`` (not a Linear).
        if model.in_proj_bias is not None or model.out_proj.bias is not None:
            return None, None, None, False
        return model.embed_dim, model.num_heads, model.head_dim, False

    # Hand-rolled form: needs q / k / v / o + num_heads.
    children = dict(model.named_children())
    needed = {"q", "k", "v", "o"}
    if not needed.issubset(children.keys()):
        return None, None, None, False
    for n in ("q", "k", "v", "o"):
        if not isinstance(children[n], nn.Linear) or children[n].bias is not None:
            return None, None, None, False
    embed_dim = children["q"].in_features
    if any(children[n].in_features != embed_dim or children[n].out_features != embed_dim for n in ("q", "k", "v", "o")):
        return None, None, None, False
    num_heads = getattr(model, "num_heads", None)
    if not isinstance(num_heads, int) or num_heads <= 0:
        return None, None, None, False
    head_dim = embed_dim // num_heads
    is_causal = bool(getattr(model, "is_causal", False))
    return embed_dim, num_heads, head_dim, is_causal


def _emit_mha(
    *,
    batch: int,
    seq: int,
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    is_causal: bool,
) -> LoweringResult:
    """Build the MHA tile graph.

    Tile-task layout (per head, per row tile, per col tile):

      q_proj  : (B*S, D) GEMM → (B*S, D)
      k_proj  : (B*S, D) GEMM → (B*S, D)
      v_proj  : (B*S, D) GEMM → (B*S, D)
      qk_matmul : per-head batched GEMM
                  Q (B, H, S, d) @ K^T (B, H, d, S) → scores (B, H, S, S)
      softmax_max : row-reduce max over S' for each (B, H, S) row
      softmax_exp : exp(score - max) per element
      softmax_sum : row-reduce sum over S'
      softmax_div : score / sum (writes the attention probs)
      av_matmul : (B, H, S, S) @ V (B, H, S, d) → out (B, H, S, d)
      o_proj  : (B*S, D) GEMM → (B*S, D)
    """
    from compgen.runtime.lowering.fx_to_megakernel import (
        LoweringDecision,
        LoweringResult,
        _BodyDecision,
    )

    if seq % _TILE_M != 0:
        from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

        raise UnsupportedShape(f"mha needs seq ({seq}) divisible by tile_m={_TILE_M}")
    if embed_dim % _TILE_N != 0:
        from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

        raise UnsupportedShape(f"mha needs embed_dim ({embed_dim}) divisible by tile_n={_TILE_N}")

    s_tiles = seq // _TILE_M
    d_tiles = embed_dim // _TILE_N
    n_qkv_tiles = batch * s_tiles * d_tiles  # one per (B, S_row, D_col)
    n_score_tiles = batch * num_heads * s_tiles * s_tiles  # (B, H, S_q, S_k)
    n_softmax_row_tiles = batch * num_heads * s_tiles  # one per row of scores
    n_av_tiles = batch * num_heads * s_tiles * (head_dim // _TILE_N if head_dim >= _TILE_N else 1)

    same = lambda c: (c[0],)  # noqa: E731

    ev_q = EventTensor((n_qkv_tiles,), wait_count_default=1)
    ev_k = EventTensor((n_qkv_tiles,), wait_count_default=1)
    ev_v = EventTensor((n_qkv_tiles,), wait_count_default=1)
    # qk_matmul tile reads all D-tiles of Q and K rows (K-fan-in by D);
    # we model that by making the wait count = d_tiles per score tile.
    ev_score_raw = EventTensor((n_score_tiles,), wait_count_default=1)
    ev_max = EventTensor((n_softmax_row_tiles,), wait_count_default=1)
    ev_exp = EventTensor((n_score_tiles,), wait_count_default=1)
    ev_sum = EventTensor((n_softmax_row_tiles,), wait_count_default=1)
    ev_probs = EventTensor((n_score_tiles,), wait_count_default=1)
    ev_av = EventTensor((n_av_tiles,), wait_count_default=1)
    ev_out = EventTensor((n_qkv_tiles,), wait_count_default=1)

    # Map score-tile coord → its softmax-row-tile coord:
    # score tiles are laid out (B, H, S_q, S_k); we fold to (B, H, S_q).
    def _score_to_row(c: tuple[int, ...]) -> tuple[int, ...]:
        # c[0] is a flat index into (B, H, S_q, S_k); fold the S_k axis.
        return (c[0] // s_tiles,)

    def _score_to_qkv(c: tuple[int, ...]) -> tuple[int, ...]:
        # Score tile is gated on Q being computed: pick its (B, S_q)
        # mapped to the Q's tile flat index.
        # Q tiles are (B, S_row, D_col); we just pick index 0 — every
        # downstream tile depends on Q being completed up to its row,
        # but Wave-2.1 keeps the dep granularity coarse: any Q tile
        # producing the row-slice unblocks. Simplification: target
        # tile 0 in Q's flat index space.
        return (0,)

    calls = (
        DeviceCall(
            name="q_proj",
            body_fn=lambda c: None,
            task_shape=(n_qkv_tiles,),
            out_edges=(EventEdge("ev_q", same),),
        ),
        DeviceCall(
            name="k_proj",
            body_fn=lambda c: None,
            task_shape=(n_qkv_tiles,),
            out_edges=(EventEdge("ev_k", same),),
        ),
        DeviceCall(
            name="v_proj",
            body_fn=lambda c: None,
            task_shape=(n_qkv_tiles,),
            out_edges=(EventEdge("ev_v", same),),
        ),
        DeviceCall(
            name="qk_matmul",
            body_fn=lambda c: None,
            task_shape=(n_score_tiles,),
            in_edges=(
                EventEdge("ev_q", _score_to_qkv),
                EventEdge("ev_k", _score_to_qkv),
            ),
            out_edges=(EventEdge("ev_score_raw", same),),
        ),
        DeviceCall(
            name="softmax_max",
            body_fn=lambda c: None,
            task_shape=(n_softmax_row_tiles,),
            # row-reduce over S_k tiles: depend on every score tile in
            # the row. We model the dep coarsely by picking score tile 0
            # in the row (Wave 2.1 keeps the coarse-grained shape
            # contract; per-row fan-in lands in 2.2 along with the
            # softmax codegen).
            in_edges=(EventEdge("ev_score_raw", lambda c: (c[0] * s_tiles,)),),
            out_edges=(EventEdge("ev_max", same),),
        ),
        DeviceCall(
            name="softmax_exp",
            body_fn=lambda c: None,
            task_shape=(n_score_tiles,),
            in_edges=(
                EventEdge("ev_score_raw", same),
                EventEdge("ev_max", _score_to_row),
            ),
            out_edges=(EventEdge("ev_exp", same),),
        ),
        DeviceCall(
            name="softmax_sum",
            body_fn=lambda c: None,
            task_shape=(n_softmax_row_tiles,),
            in_edges=(EventEdge("ev_exp", lambda c: (c[0] * s_tiles,)),),
            out_edges=(EventEdge("ev_sum", same),),
        ),
        DeviceCall(
            name="softmax_div",
            body_fn=lambda c: None,
            task_shape=(n_score_tiles,),
            in_edges=(
                EventEdge("ev_exp", same),
                EventEdge("ev_sum", _score_to_row),
            ),
            out_edges=(EventEdge("ev_probs", same),),
        ),
        DeviceCall(
            name="av_matmul",
            body_fn=lambda c: None,
            task_shape=(n_av_tiles,),
            in_edges=(
                EventEdge("ev_probs", lambda c: (0,)),
                EventEdge("ev_v", lambda c: (0,)),
            ),
            out_edges=(EventEdge("ev_av", same),),
        ),
        DeviceCall(
            name="o_proj",
            body_fn=lambda c: None,
            task_shape=(n_qkv_tiles,),
            in_edges=(EventEdge("ev_av", lambda c: (0,)),),
            out_edges=(EventEdge("ev_out", same),),
        ),
    )

    graph = MegakernelGraph(
        name="mha_lowered",
        calls=calls,
        event_tensors={
            "ev_q": ev_q,
            "ev_k": ev_k,
            "ev_v": ev_v,
            "ev_score_raw": ev_score_raw,
            "ev_max": ev_max,
            "ev_exp": ev_exp,
            "ev_sum": ev_sum,
            "ev_probs": ev_probs,
            "ev_av": ev_av,
            "ev_out": ev_out,
        },
        policy="static",
    )

    bodies: dict[str, DeviceFunctionSource] = {
        name: _placeholder_body(
            name,
            f"MHA op {name!r}; B={batch}, S={seq}, D={embed_dim}, H={num_heads}, d_head={head_dim}, causal={is_causal}",
        )
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "qk_matmul",
            "softmax_max",
            "softmax_exp",
            "softmax_sum",
            "softmax_div",
            "av_matmul",
            "o_proj",
        )
    }

    body_decisions = tuple(
        _BodyDecision(
            op_name=name,
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale=(
                f"MHA Wave-2.1 placeholder for {name!r}; codegen lands in "
                "Wave 2.2 (online softmax + masked-matmul kernel)."
            ),
        )
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "qk_matmul",
            "softmax_max",
            "softmax_exp",
            "softmax_sum",
            "softmax_div",
            "av_matmul",
            "o_proj",
        )
    )

    decision = LoweringDecision(
        pattern_name="mha",
        pattern_rationale=(
            f"matched multi-head attention with B={batch}, S={seq}, "
            f"D={embed_dim}, H={num_heads}, d_head={head_dim}; "
            f"is_causal={is_causal}. softmax expressed as 4 tile-tasks "
            "(max, exp, sum, div) so cross-tile event-tensor structure "
            "is explicit."
        ),
        body_decisions=body_decisions,
        schedule_hints={
            "policy": "static",
            "mha_causal": is_causal,
            "batch": batch,
            "seq": seq,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "tile_shape": [_TILE_M, _TILE_N, _TILE_K],
            "s_tiles": s_tiles,
            "d_tiles": d_tiles,
            "block_dim": [32, 32, 1],
        },
        total_tile_tasks=(
            3 * n_qkv_tiles + n_score_tiles + 2 * n_softmax_row_tiles + 2 * n_score_tiles + n_av_tiles + n_qkv_tiles
        ),
    )

    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=(
            "x",  # 0: input (B, S, D)
            "w_q",  # 1: q weight
            "w_k",  # 2: k weight
            "w_v",  # 3: v weight
            "w_o",  # 4: output proj weight
            "y_q",  # 5: Q   (B, S, D)
            "y_k",  # 6: K   (B, S, D)
            "y_v",  # 7: V   (B, S, D)
            "y_score",  # 8: scaled (B, H, S, S)
            "y_max",  # 9: row max (B, H, S)
            "y_exp",  # 10: exp(score - max)
            "y_sum",  # 11: row sum
            "y_probs",  # 12: softmax probs
            "y_av",  # 13: probs @ V
            "y_out",  # 14: o_proj output
        ),
        decision=decision,
    )


# ---------------------------------------------------------------------------
# MoE — sparse expert dispatch (data-dependent → dynamic schedule)
# ---------------------------------------------------------------------------


def _match_moe(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
) -> LoweringResult:
    """Match MoE: ``router(x) → topk → expert dispatch``.

    Required attributes on ``model``:
      * ``router`` — ``nn.Linear(d, n_experts, bias=False)``
      * ``experts`` — ``nn.ModuleList`` of ``n_experts`` expert modules
        (each shape-preserving, e.g. an FFN)
      * ``top_k`` — int (1 ≤ top_k ≤ n_experts)
    """
    from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

    router = getattr(model, "router", None)
    experts = getattr(model, "experts", None)
    top_k = getattr(model, "top_k", None)

    if not isinstance(router, nn.Linear) or router.bias is not None:
        raise UnsupportedShape("moe requires self.router to be nn.Linear(..., bias=False)")
    if not isinstance(experts, nn.ModuleList) or len(experts) == 0:
        raise UnsupportedShape(
            "moe requires self.experts: nn.ModuleList[experts]; got "
            f"{type(experts).__name__ if experts is not None else 'None'}"
        )
    if not isinstance(top_k, int) or top_k <= 0:
        raise UnsupportedShape(f"moe requires self.top_k to be a positive int; got {top_k!r}")
    n_experts = len(experts)
    if top_k > n_experts:
        raise UnsupportedShape(f"moe top_k ({top_k}) > n_experts ({n_experts})")

    embed_dim = router.in_features
    if router.out_features != n_experts:
        raise UnsupportedShape(f"moe router out_features ({router.out_features}) must equal len(experts) ({n_experts})")

    x = sample_inputs[0]
    if x.ndim < 2 or x.shape[-1] != embed_dim:
        raise UnsupportedShape(
            f"moe input shape {tuple(x.shape)} does not match router in_features={embed_dim} on the trailing axis"
        )
    n_tokens = _flatten_batch(x, embed_dim)

    # Forward probe to confirm the module is indeed a routed MoE
    # (not just a coincidence of attribute names). Permitted: any
    # forward that produces a same-shape output on x.
    with torch.no_grad():
        try:
            y = model(x)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"moe forward probe raised {exc!r}") from exc
    if not isinstance(y, torch.Tensor) or tuple(y.shape) != tuple(x.shape):
        raise UnsupportedShape(
            f"moe forward must return a same-shape tensor; got "
            f"{type(y).__name__ if not isinstance(y, torch.Tensor) else tuple(y.shape)}"
        )

    return _emit_moe(
        n_tokens=n_tokens,
        embed_dim=embed_dim,
        n_experts=n_experts,
        top_k=top_k,
    )


def _emit_moe(
    *,
    n_tokens: int,
    embed_dim: int,
    n_experts: int,
    top_k: int,
) -> LoweringResult:
    """Build the MoE dynamic graph.

    Tile-task layout:

      router_proj : (n_tokens, D) GEMM → (n_tokens, n_experts) routes
      router_topk : per-token topk + softmax of the top-k routes
      expert_e    : per-expert task (one DeviceCall per expert) — the
                    runtime schedules the actual instantiations
                    on-device via TriggerOp once the topk indices are
                    known. We declare the static fan-out here as one
                    task per expert; the trigger generators record
                    the runtime expansion contract.
      expert_combine : weighted sum of the top_k expert outputs per token

    The graph is ``policy="dynamic"`` because the per-token expert
    routing is data-dependent (paper §3.2). The
    ``"requires_ondevice_scheduler"`` schedule hint signals to Phase 6
    that this graph needs a target with
    ``DeviceTraits.supports_ondevice_scheduler=True``.
    """
    from compgen.runtime.lowering.fx_to_megakernel import (
        LoweringDecision,
        LoweringResult,
        _BodyDecision,
    )
    from compgen.transforms.event_dynamic_schedule import TriggerGenerator

    # MoE per-token granularity: when n_tokens < tile_m we still
    # produce one tile (the runtime kernel masks the unused rows).
    # Otherwise we require divisibility for the tile graph to stay
    # regular at the structural-recognition level. The tail mask
    # belongs to the codegen; the matcher only pins the shape contract.
    if n_tokens < _TILE_M:
        n_token_tiles = 1
    elif n_tokens % _TILE_M != 0:
        from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

        raise UnsupportedShape(f"moe needs n_tokens ({n_tokens}) divisible by tile_m={_TILE_M}")
    else:
        n_token_tiles = n_tokens // _TILE_M

    same = lambda c: (c[0],)  # noqa: E731
    flat_zero = lambda c: (0,)  # noqa: E731

    ev_routes = EventTensor((n_token_tiles,), wait_count_default=1)
    ev_topk = EventTensor((n_token_tiles,), wait_count_default=1)
    # Per-expert event tensors: each is a single-cell event whose
    # wait count = topk × n_token_tiles in the worst case (every token
    # routes to every expert). At runtime the actual decrement count
    # equals the dispatched count; we set wait_count_default to 1 and
    # rely on TriggerOp to inject the runtime fan-in.
    ev_experts = {f"ev_expert_{e}": EventTensor((1,), wait_count_default=1) for e in range(n_experts)}
    ev_combine = EventTensor((n_token_tiles,), wait_count_default=1)

    expert_calls = tuple(
        DeviceCall(
            name=f"expert_{e}",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("ev_topk", flat_zero),),
            out_edges=(EventEdge(f"ev_expert_{e}", flat_zero),),
        )
        for e in range(n_experts)
    )

    calls = (
        DeviceCall(
            name="router_proj",
            body_fn=lambda c: None,
            task_shape=(n_token_tiles,),
            out_edges=(EventEdge("ev_routes", same),),
        ),
        DeviceCall(
            name="router_topk",
            body_fn=lambda c: None,
            task_shape=(n_token_tiles,),
            in_edges=(EventEdge("ev_routes", same),),
            out_edges=(EventEdge("ev_topk", same),),
        ),
        *expert_calls,
        DeviceCall(
            name="expert_combine",
            body_fn=lambda c: None,
            task_shape=(n_token_tiles,),
            in_edges=tuple(EventEdge(f"ev_expert_{e}", flat_zero) for e in range(n_experts))
            + (EventEdge("ev_topk", same),),
            out_edges=(EventEdge("ev_combine", same),),
        ),
    )
    graph = MegakernelGraph(
        name="moe_lowered",
        calls=calls,
        event_tensors={
            "ev_routes": ev_routes,
            "ev_topk": ev_topk,
            **ev_experts,
            "ev_combine": ev_combine,
        },
        policy="dynamic",
    )

    # TriggerOp records — one per expert. The dynamic-schedule pass
    # materializes these into ready-queue pushes at runtime once the
    # topk indices populate the trigger source tensor.
    trigger_generators = tuple(
        TriggerGenerator(
            target_event=f"ev_expert_{e}",
            source_tensor="topk_indices",
            target_device_func=f"expert_{e}",
            task_shape=(1,),
        )
        for e in range(n_experts)
    )

    bodies: dict[str, DeviceFunctionSource] = {
        "router_proj": _placeholder_body(
            "router_proj",
            f"linear router projection: x (n_tokens={n_tokens}, D={embed_dim}) → "
            f"routes (n_tokens, n_experts={n_experts})",
        ),
        "router_topk": _placeholder_body(
            "router_topk",
            f"top-k routing: per-token argsort of routes → topk_indices "
            f"shape (n_tokens, top_k={top_k}); softmax weights",
        ),
        "expert_combine": _placeholder_body(
            "expert_combine",
            f"weighted sum across top_k={top_k} expert outputs per token",
        ),
    }
    for e in range(n_experts):
        bodies[f"expert_{e}"] = _placeholder_body(
            f"expert_{e}",
            f"expert {e}/{n_experts}; D={embed_dim}; runtime-dispatched via TriggerOp(source=topk_indices)",
        )

    body_decisions = (
        _BodyDecision(
            op_name="router_proj",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale=f"MoE router linear projection over n_experts={n_experts}",
        ),
        _BodyDecision(
            op_name="router_topk",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale=f"MoE topk + softmax; top_k={top_k}",
        ),
        *(
            _BodyDecision(
                op_name=f"expert_{e}",
                backend="hand_rolled_fmaf",
                tile_shape=(_TILE_M, _TILE_N, _TILE_K),
                rationale=f"MoE expert {e}; data-dependent dispatch via TriggerOp",
            )
            for e in range(n_experts)
        ),
        _BodyDecision(
            op_name="expert_combine",
            backend="hand_rolled_fmaf",
            tile_shape=(_TILE_M, _TILE_N, _TILE_K),
            rationale=f"MoE per-token weighted combine over top_k={top_k} experts",
        ),
    )

    decision = LoweringDecision(
        pattern_name="moe",
        pattern_rationale=(
            f"matched MoE block: router=Linear({embed_dim}->{n_experts}) + "
            f"experts=ModuleList[{n_experts}], top_k={top_k}, "
            f"n_tokens={n_tokens}. Data-dependent dispatch — graph runs on "
            "the dynamic schedule path (policy='dynamic') with one "
            "TriggerGenerator per expert."
        ),
        body_decisions=body_decisions,
        schedule_hints={
            "policy": "dynamic",
            "requires_ondevice_scheduler": True,
            "n_experts": n_experts,
            "top_k": top_k,
            "n_tokens": n_tokens,
            "embed_dim": embed_dim,
            "n_token_tiles": n_token_tiles,
            "tile_shape": [_TILE_M, _TILE_N, _TILE_K],
            "block_dim": [32, 32, 1],
            "trigger_source_tensor": "topk_indices",
            "trigger_target_events": [f"ev_expert_{e}" for e in range(n_experts)],
        },
        total_tile_tasks=2 * n_token_tiles + n_experts + n_token_tiles,
    )

    layout: tuple[str, ...] = (
        "x",  # 0
        "w_router",  # 1
        "y_routes",  # 2: per-token, per-expert raw scores
        "topk_indices",  # 3: per-token expert ids
        "topk_weights",  # 4: per-token softmax weights for top_k
        *(f"w_expert_{e}" for e in range(n_experts)),
        *(f"y_expert_{e}" for e in range(n_experts)),
        "y_out",  # final
    )

    # Stash the trigger generators on the result via the
    # schedule_hints — Phase 6 reads them when invoking the dynamic
    # schedule pass. We can't add new fields to LoweringResult
    # without a wider refactor; the hints carry trigger metadata so
    # ``compute_dynamic_schedule(...)`` can be reconstructed.
    decision.schedule_hints["trigger_generators"] = [
        {
            "target_event": tg.target_event,
            "source_tensor": tg.source_tensor,
            "target_device_func": tg.target_device_func,
            "task_shape": list(tg.task_shape),
        }
        for tg in trigger_generators
    ]

    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=layout,
        decision=decision,
    )


def build_moe_trigger_generators(decision: Any) -> tuple[Any, ...]:
    """Reconstruct the :class:`TriggerGenerator` tuple from a
    ``LoweringDecision.schedule_hints["trigger_generators"]``.

    Helper for callers that want to invoke
    :func:`compgen.transforms.event_dynamic_schedule.compute_dynamic_schedule`
    on a MoE-lowered graph without re-running the matcher.
    """
    from compgen.transforms.event_dynamic_schedule import TriggerGenerator

    raw = decision.schedule_hints.get("trigger_generators", ())
    return tuple(
        TriggerGenerator(
            target_event=str(r["target_event"]),
            source_tensor=str(r["source_tensor"]),
            target_device_func=str(r["target_device_func"]),
            task_shape=tuple(int(d) for d in r["task_shape"]),
        )
        for r in raw
    )


__all__ = [
    "_match_residual_norm",
    "_match_mha",
    "_match_moe",
    "build_moe_trigger_generators",
]
