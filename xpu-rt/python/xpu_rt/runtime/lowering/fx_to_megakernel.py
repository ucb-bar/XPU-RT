"""FX-graph → MegakernelGraph pattern matcher.

Replaces the conformance-harness's hand-built workload factories
with an automatic lowering: a torch ``nn.Module`` + sample inputs
goes in, a :class:`compgen.runtime.megakernel.MegakernelGraph`
plus the matching device-function bodies come out, ready for the
Phase-2/3/5 pipeline.

**Round-1 patterns** (covered):

- ``Diamond``: ``y = (linear_a(x) + linear_b(x)).relu()``. Two
  ``nn.Linear`` (no bias), one elementwise add, one elementwise
  relu. Tile-level task graph (32×32 tiles).

Future patterns (round 2+):

- ``Ffn``: ``y = down(relu(up(x)))`` — decoder_layer's FFN.
- ``RowParallelLinear``: ``y = x @ W`` with W row-sharded — gemm_rs.

When no pattern matches, :class:`UnsupportedShape` is raised. The
caller decides whether to fall back to the legacy single-kernel
path or surface the error to the user.

Decision logging — every matcher records its match decision
(pattern name, op-by-op routing, tile shape, why this backend
won) on the returned :class:`LoweringDecision`. The MCP-driven
flow exposes these via the ``compgen_compile_decisions`` tool so
remote agents can audit the compiler's choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import (
    DeviceCall,
    EventEdge,
    MegakernelGraph,
)
from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


class UnsupportedShape(RuntimeError):
    """No FX-graph pattern matched the input model.

    Callers should treat this as recoverable: either try a
    different lowering path (legacy compile_model) or surface a
    typed error to the user.
    """


@dataclass(frozen=True)
class _BodyDecision:
    """One per device function: which backend produced the body."""

    op_name: str
    backend: str  # e.g. "hand_rolled_fmaf" — round 2 adds "cublasdx", "tile_ir"
    tile_shape: tuple[int, int, int]  # (M, N, K) of the per-task tile
    rationale: str


@dataclass(frozen=True)
class _BackendAvailability:
    """Probe results for the optional codegen backends.

    The matcher records what's reachable so the agent can audit
    why a body went to fmaf instead of cuBLASDx (e.g. the
    nvidia-mathdx wheel isn't installed, or libcudacxx headers
    aren't on NVRTC's include path) without source-reading.
    """

    cublasdx_include: str | None
    cublasdx_status: str  # "available" | "missing" | "<reason>"
    libcudacxx_include: str | None = None
    libcudacxx_status: str = "unprobed"  # "available" | "missing"
    cutlass_include: str | None = None
    cutlass_status: str = "unprobed"  # "available" | "missing"


@dataclass(frozen=True)
class LoweringDecision:
    """Compile-time decisions, surfaced for MCP audit.

    Carries enough information that a remote agent can ask
    "why did the compiler pick X for op Y?" and get an answer
    grounded in the matcher's decision tree, not folk knowledge.
    """

    pattern_name: str
    pattern_rationale: str
    body_decisions: tuple[_BodyDecision, ...]
    schedule_hints: dict[str, Any] = field(default_factory=dict)
    # Total number of tile-tasks (all op kinds combined) the
    # scheduler will see. Useful for back-of-the-envelope sanity:
    # 32 tasks fits 188 SMs comfortably; 1000 would not.
    total_tile_tasks: int = 0
    # Backend availability snapshot at compile time. ``None`` for
    # round-1 decisions that pre-date the probe; populated by
    # round-2+ matchers.
    backends: _BackendAvailability | None = None
    # NVRTC ``-I`` paths the compile pipeline must thread through
    # CudaModule.extra_include_paths so the bodies' #includes
    # resolve. Non-empty when at least one body uses a backend that
    # ships header-only code (cuBLASDx / CUTLASS / etc.).
    nvrtc_include_paths: tuple[str, ...] = ()
    # NVRTC compiler options the body emitter requires.
    # ``-default-device`` is needed when any body calls cuBLASDx
    # static constexpr functions (cuBLASDx 0.4.0 doesn't annotate
    # them with __host__ __device__ so NVRTC rejects them as
    # host-only without this flag). Empty tuple when no body needs
    # extra options.
    nvrtc_extra_options: tuple[str, ...] = ()
    # Wave 1.8 — when the matcher accepted via the submodule
    # fallback (``_try_submodule_match``), this carries the dotted
    # path from the top-level model down to the matched sub-block
    # (e.g. ``"ffn"``, ``"transformer.layers.0.mlp"``). Empty
    # string means the top-level model itself matched. Used by the
    # compile/dispatch boundary to pickle the right effective
    # model so weight extraction works at runtime — per bridge
    # #108: composition compile worked but dispatch raised
    # ``AttributeError: 'ThinWrapper' object has no attribute 'up'``
    # because the runtime saw the wrapper, not the matched ffn.
    submodule_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "pattern_rationale": self.pattern_rationale,
            "body_decisions": [
                {
                    "op_name": d.op_name,
                    "backend": d.backend,
                    "tile_shape": list(d.tile_shape),
                    "rationale": d.rationale,
                }
                for d in self.body_decisions
            ],
            "schedule_hints": dict(self.schedule_hints),
            "total_tile_tasks": self.total_tile_tasks,
            "backends": (
                {
                    "cublasdx_include": self.backends.cublasdx_include,
                    "cublasdx_status": self.backends.cublasdx_status,
                    "libcudacxx_include": self.backends.libcudacxx_include,
                    "libcudacxx_status": self.backends.libcudacxx_status,
                    "cutlass_include": self.backends.cutlass_include,
                    "cutlass_status": self.backends.cutlass_status,
                }
                if self.backends is not None
                else None
            ),
            "nvrtc_include_paths": list(self.nvrtc_include_paths),
            "nvrtc_extra_options": list(self.nvrtc_extra_options),
            "submodule_path": self.submodule_path,
        }


@dataclass(frozen=True)
class LoweringResult:
    """The output of :func:`lower_torch_to_megakernel`."""

    megakernel_graph: MegakernelGraph
    device_function_sources: dict[str, DeviceFunctionSource]
    user_buffer_layout: tuple[str, ...]
    decision: LoweringDecision


# ---------------------------------------------------------------------------
# User-registered lowerings (Wave 2.3 entrypoint)
# ---------------------------------------------------------------------------


def _registered_user_lowerings() -> list[Any]:
    """Return user-registered lowerings, ordered by registration.

    Late-binds the ``compgen.plugins`` import so the matcher stays
    importable on minimal installs. Returns an empty list when the
    plugins module isn't available or no user lowerings are
    registered.

    The user registers a lowering via either:

    1. ``compgen.plugins.register(GROUP_LOWERINGS, name, fn)`` —
       programmatic, in-process. Useful when the agent loads a
       dialect at session start.
    2. A ``compgen.runtime.lowerings`` entry-point in the user's
       package's ``pyproject.toml`` — picked up automatically by
       :func:`compgen.plugins.discover_all`.

    Each registered ``fn`` must match the same contract as the
    built-in matchers: ``fn(model, sample_inputs, *,
    backend_choice=None) -> LoweringResult`` or raise
    :class:`UnsupportedShape`.
    """
    try:
        from compgen.plugins import GROUP_LOWERINGS, registry
    except ImportError:
        return []
    return registry().get(GROUP_LOWERINGS)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


_VALID_CUBLASDX_PRECISIONS = ("fp32", "bf16_fp32")


# Wave 1.14 — moved to ``targets/gpu/nvidia/common/sm_tag.py``.
# Re-exported here under the original private name for one round
# of backward compatibility; callers should import from the new
# location going forward.
from compgen.targets.gpu.nvidia.common.sm_tag import (  # noqa: E402
    arch_to_cublasdx_sm as _arch_to_cublasdx_sm,
)


def lower_torch_to_megakernel(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    backend_choice: Any = None,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
    target_arch: str = "sm_100",
    allow_generic_fallback: bool = True,
    fuse_epilogue: bool = False,
) -> LoweringResult:
    """Walk ``model`` + ``sample_inputs`` and emit a MegakernelGraph.

    Args:
        model: PyTorch ``nn.Module``. Must match a registered pattern;
            otherwise :class:`UnsupportedShape` is raised.
        sample_inputs: Concrete inputs. Used to infer tensor shapes
            for the pattern matcher and the body bodies.
        backend_choice: Optional :class:`compgen.runtime.autotune.BackendChoice`
            from :func:`compgen.runtime.autotune.probe_device`. When
            provided, every individual flag is **overridden** by the
            choice's fields — the agentic-compilation flow uses this
            so a PyPI user only passes ``compile_model(model)`` and
            CompGen probes everything internally. When ``None``, the
            individual flags below are honored (backwards-compatible
            path for tests + advanced callers who want fine control).
        prefer_cublasdx_for_linears: When True AND cuBLASDx +
            libcudacxx + CUTLASS are all reachable (per
            :func:`_probe_backends`), emit linear bodies that call
            ``cublasdx::BLAS().execute(...)`` with
            ``Arrangement<row_major, row_major, row_major>`` and
            accumulate over the K dimension. Default False: keep
            the hand_rolled_fmaf body so non-cuBLASDx hosts still
            compile. When True but cuBLASDx isn't reachable, the
            matcher silently falls back to fmaf — the decision log
            records that fact so the agent can audit.
        target_arch: NVRTC arch flag form. Used here to map the
            cuBLASDx ``SM<...>`` template tag — e.g. ``sm_100`` →
            ``SM<1000>`` so Blackwell datacenter hardware engages
            ``tcgen05.mma`` tensor cores. Default ``"sm_100"``
            (paper-faithful Blackwell). The matcher does not
            validate that the host machine is actually that arch;
            NVRTC will catch a mismatch at compile time.
        cublasdx_precision: Which cuBLASDx ``Precision<...>`` tag
            to instantiate when the cuBLASDx body is selected.
            ``"fp32"`` (default): ``Precision<float>`` — fp32 SIMT,
            no tensor cores. Bit-tight vs ``torch.matmul`` but caps
            at fp32 SIMT throughput. ``"bf16_fp32"``:
            ``Precision<__nv_bfloat16, __nv_bfloat16, float>`` —
            bf16 inputs, fp32 accumulator, **engages Blackwell
            tensor cores**. Smem loads cast fp32 gmem to bf16
            on the way in; accumulator + output stay fp32 so the
            user-side torch model needs no dtype change. Loses ~3-4
            mantissa bits vs end-to-end fp32 but ≈10× perf headroom
            on Blackwell. Ignored when ``prefer_cublasdx_for_linears``
            is False.

    Returns:
        :class:`LoweringResult` with the graph, bodies, buffer
        layout, and decision log.

    Raises:
        UnsupportedShape: No pattern matched.
    """
    if backend_choice is not None:
        # Source of truth — overrides every individual flag. The
        # agentic-compilation entry point packages all four
        # configuration knobs into one BackendChoice object so we
        # don't have to pass them around individually.
        prefer_cublasdx_for_linears = backend_choice.use_cublasdx_for_linears
        cublasdx_precision = backend_choice.cublasdx_precision
        target_arch = backend_choice.target_arch
        cublasdx_sm = backend_choice.cublasdx_sm
    if cublasdx_precision not in _VALID_CUBLASDX_PRECISIONS:
        raise ValueError(f"cublasdx_precision={cublasdx_precision!r} not in {_VALID_CUBLASDX_PRECISIONS}")
    if backend_choice is None:
        cublasdx_sm = _arch_to_cublasdx_sm(target_arch)
    matcher_errors: list[str] = []

    # User-registered lowerings get tried FIRST (Wave 2.3 — the
    # cuda-tile / custom-MLIR-dialect entrypoint). Each is a callable
    # registered via ``compgen.plugins.register(GROUP_LOWERINGS, name,
    # fn)`` or via a ``compgen.runtime.lowerings`` entry-point in
    # the user's pyproject. The contract matches the built-in
    # matchers — ``(model, sample_inputs, *, backend_choice=...)``
    # → ``LoweringResult`` or ``UnsupportedShape``.
    user_lowerings = _registered_user_lowerings()
    for plugin in user_lowerings:
        try:
            return plugin.object(
                model,
                sample_inputs,
                backend_choice=backend_choice,
            )
        except UnsupportedShape as exc:
            matcher_errors.append(f"user:{plugin.name}: {exc}")
        except TypeError:
            # Older user lowerings may not accept backend_choice yet.
            try:
                return plugin.object(model, sample_inputs)
            except UnsupportedShape as exc:
                matcher_errors.append(f"user:{plugin.name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                matcher_errors.append(f"user:{plugin.name}: raised {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            matcher_errors.append(f"user:{plugin.name}: raised {type(exc).__name__}: {exc}")

    # Built-in matchers run after user lowerings. The Wave-2.1
    # additions (residual+norm, MHA, MoE) follow the Wave-1
    # diamond/FFN matchers — order matters: residual+norm wraps
    # diamond/FFN sublayers, and we want the inner matchers to
    # have first crack at a plain block before residual_norm
    # composes them.
    from compgen.runtime.lowering.pattern_catalog import (
        _match_mha,
        _match_moe,
        _match_residual_norm,
    )

    matchers = (
        _match_diamond,
        _match_ffn,
        _match_residual_norm,
        _match_mha,
        _match_moe,
    )
    for matcher in matchers:
        kwargs: dict[str, Any] = {
            "prefer_cublasdx_for_linears": prefer_cublasdx_for_linears,
            "cublasdx_precision": cublasdx_precision,
            "cublasdx_sm": cublasdx_sm,
        }
        # Wave 2.5: epilogue fusion only applies to the FFN matcher
        # today (folds relu into linear_up's MMA tail). Other matchers
        # don't accept the kwarg.
        if matcher is _match_ffn:
            kwargs["fuse_epilogue"] = fuse_epilogue
        try:
            return matcher(model, sample_inputs, **kwargs)
        except UnsupportedShape as exc:
            matcher_errors.append(f"{matcher.__name__}: {exc}")

    # Wave 1.8 — composition fallback. When none of the matchers
    # accepted the top-level model, walk submodules and check if
    # the model is a thin wrapper around a single matchable
    # sub-block (e.g. ``class M: def forward(self, x): return
    # self.ffn(x)``). Per bridge #102: paper-shape decoder layers
    # often wrap an FFN inside a larger module; without this fallback
    # the matcher rejects them as "got 0 linears at top level."
    try:
        return _try_submodule_match(
            model,
            sample_inputs,
            prefer_cublasdx_for_linears=prefer_cublasdx_for_linears,
            cublasdx_precision=cublasdx_precision,
            cublasdx_sm=cublasdx_sm,
            fuse_epilogue=fuse_epilogue,
            matcher_errors=matcher_errors,
        )
    except UnsupportedShape as exc:
        matcher_errors.append(f"submodule_match: {exc}")

    # Wave 2.2 — generic FX-trace fallback. When no registered
    # pattern + no submodule match took the model, fall back to a
    # serial-chain MegakernelGraph that handles any FX-traceable
    # combination of supported ops (linear / relu / add). Gives a
    # runnable bundle for arbitrary models — slower than
    # pattern-matched paths but correct.
    # Opt-in with ``allow_generic_fallback`` (default True). The
    # generic path raises UnsupportedShape with a list of unhandled
    # ops if the FX trace contains anything outside the supported
    # family — surfaces as one more matcher_errors entry, which
    # the agent can read to know why even the fallback failed.
    if allow_generic_fallback:
        try:
            from compgen.runtime.lowering.fx_generic import lower_generic_fx

            return lower_generic_fx(
                model,
                sample_inputs,
                backend_choice=backend_choice,
            )
        except UnsupportedShape as exc:
            matcher_errors.append(f"generic_fx_chain: {exc}")
        except Exception as exc:  # noqa: BLE001
            matcher_errors.append(f"generic_fx_chain: raised {type(exc).__name__}: {exc}")

    # No matcher took the model. Surface every matcher's reason so
    # the caller can see *why* none matched — "bias", "divisible",
    # whatever — rather than a generic "no pattern matched". Crucial
    # for the MCP-driven flow's audit story.
    detail = "; ".join(matcher_errors)
    raise UnsupportedShape(f"no FX-graph pattern matched model {type(model).__name__}: {detail}")


# ---------------------------------------------------------------------------
# Composition fallback (Wave 1.8 — bridge #102 unblocker)
# ---------------------------------------------------------------------------


def _try_submodule_match(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool,
    cublasdx_precision: str,
    cublasdx_sm: int,
    matcher_errors: list[str],
    fuse_epilogue: bool = False,
) -> Any:
    """Try matching a sub-module when the top-level model wraps
    a single matchable block.

    Catches the bridge #102 case: ``class StackedFFN: def forward
    (self, x): return self.ffn(x)`` (or any thin-wrapper variant)
    whose ``named_children()`` returns the wrapper's own children
    (an ``nn.ModuleList``, an ``nn.Sequential``) instead of the
    eventual ``nn.Linear``s the matchers look for.

    Algorithm: compute the wrapper's forward output, walk
    ``named_modules()`` in increasing-depth order, try each
    submodule as a candidate. If a submodule's forward produces
    the same output (within fp32 ULP) AND a built-in matcher
    accepts it, return that LoweringResult tagged with the
    submodule's path so the agent's audit query knows which
    sub-block was lowered.

    Multi-block composition (DecoderBlock with attention + FFN
    where both contribute to the output) is out of scope here —
    that needs Wave 2.1's MHA matcher + a stitcher pass. This
    helper handles only the single-block-wrapped case.
    """
    x = sample_inputs[0]
    try:
        with torch.no_grad():
            top_y = model(x)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedShape(f"top-level forward failed: {exc!r}") from exc

    # Iterate submodules. Skip the model itself (already tried) and
    # container modules whose children we'll visit separately.
    for name, sub in model.named_modules():
        if name == "":
            continue
        if isinstance(sub, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            # Containers are searched by visiting their children;
            # don't try the container itself as a matchable model.
            continue
        if isinstance(sub, nn.Linear):
            # Single Linear can't match diamond/FFN.
            continue
        # Probe the submodule with the same input. If it accepts the
        # input shape and produces the same output as the top-level
        # forward, the wrapper is just delegating to this sub-block.
        try:
            with torch.no_grad():
                sub_y = sub(x)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sub_y, torch.Tensor):
            continue
        if sub_y.shape != top_y.shape:
            continue
        if not torch.allclose(sub_y, top_y, atol=1e-5, rtol=1e-5):
            continue

        # Sub-block reproduces the wrapper's forward — try matching it.
        from compgen.runtime.lowering.pattern_catalog import (
            _match_mha,
            _match_moe,
            _match_residual_norm,
        )

        for matcher in (
            _match_diamond,
            _match_ffn,
            _match_residual_norm,
            _match_mha,
            _match_moe,
        ):
            kwargs: dict[str, Any] = {
                "prefer_cublasdx_for_linears": prefer_cublasdx_for_linears,
                "cublasdx_precision": cublasdx_precision,
                "cublasdx_sm": cublasdx_sm,
            }
            if matcher is _match_ffn:
                kwargs["fuse_epilogue"] = fuse_epilogue
            try:
                result = matcher(sub, sample_inputs, **kwargs)
                # Tag the decision so the agent's audit query sees
                # which sub-block path was taken.
                from dataclasses import replace as _replace

                tagged = _replace(
                    result.decision,
                    pattern_name=f"{result.decision.pattern_name}@{name}",
                    pattern_rationale=(
                        f"matched {result.decision.pattern_name} on submodule "
                        f"{name!r} of {type(model).__name__}; the wrapper's "
                        "forward output equals the submodule's forward "
                        "(thin-wrapper case from bridge #102). " + result.decision.pattern_rationale
                    ),
                    submodule_path=name,
                )
                return _replace(result, decision=tagged)
            except UnsupportedShape as exc:
                matcher_errors.append(f"submodule@{name}/{matcher.__name__}: {exc}")

    raise UnsupportedShape("no submodule reproduces the wrapper's forward + matches a known pattern")


# ---------------------------------------------------------------------------
# Pattern matchers
# ---------------------------------------------------------------------------


def _match_diamond(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
) -> LoweringResult:
    """Match ``y = (linear_a(x) + linear_b(x)).relu()``.

    Recognised by structure: the model has exactly two ``nn.Linear``
    children with the same in/out shape, no bias on either, and the
    forward computes ``relu(self.<a>(x) + self.<b>(x))``. We don't
    inspect the FX graph yet — for round 1, structural inspection
    of the module's attributes + a single forward-shape probe is
    enough to identify diamond.
    """
    linears = [(name, m) for name, m in model.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise UnsupportedShape(f"diamond requires exactly 2 nn.Linear children; got {len(linears)}")
    name_a, lin_a = linears[0]
    name_b, lin_b = linears[1]
    if (lin_a.in_features, lin_a.out_features) != (lin_b.in_features, lin_b.out_features):
        raise UnsupportedShape(
            "diamond requires both nn.Linear to share (in_features, out_features); "
            f"got {lin_a.in_features}→{lin_a.out_features} and "
            f"{lin_b.in_features}→{lin_b.out_features}"
        )
    if lin_a.bias is not None or lin_b.bias is not None:
        raise UnsupportedShape("diamond round-1 matcher requires bias=False on both linears")

    # Probe the forward to confirm the actual shape it returns.
    x = sample_inputs[0]
    # Accept ND inputs (B, ..., in_features) by flattening leading
    # dims into the batch axis — torch.nn.Linear does the same, and
    # the matcher's tile graph operates on a 2D (batch_flat, in)
    # contract. Per bridge #108: ``compile_to_megakernel`` callers
    # routinely pass (1, B, in_features) shapes; the strict ndim==2
    # check forced 2D-only and made the Python API less ergonomic
    # than the MCP tool surface.
    if x.ndim < 2 or x.shape[-1] != lin_a.in_features:
        raise UnsupportedShape(
            f"diamond input shape {tuple(x.shape)} does not match "
            f"linear in_features={lin_a.in_features} on the trailing axis"
        )
    with torch.no_grad():
        try:
            y = model(x)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"diamond forward probe raised {exc!r}") from exc
    # Output shape: trailing dim becomes out_features, leading dims
    # preserved (torch.nn.Linear convention).
    expected_y_shape = tuple(x.shape[:-1]) + (lin_a.out_features,)
    if tuple(y.shape) != expected_y_shape:
        raise UnsupportedShape(f"diamond forward returned shape {tuple(y.shape)}, expected {expected_y_shape}")
    # Verify the actual computation matches `(linear_a(x) + linear_b(x)).relu()`
    # within fp32 ULP. Catches modules with the right children but a
    # different forward (e.g. concat instead of add).
    with torch.no_grad():
        expected = (lin_a(x) + lin_b(x)).relu()
    if not torch.allclose(y, expected, atol=1e-5, rtol=1e-5):
        raise UnsupportedShape(
            "diamond pattern children match but forward output disagrees with (linear_a(x) + linear_b(x)).relu()"
        )

    # Flatten leading dims into a single batch axis for the tile graph.
    # The dispatch path's _workload_buffers builds (batch_flat, in_dim)
    # buffers — same as what the matcher sees here.
    batch_flat = 1
    for d in x.shape[:-1]:
        batch_flat *= int(d)

    return _emit_diamond(
        x_shape=(batch_flat, int(x.shape[-1])),
        linear_a=(name_a, lin_a),
        linear_b=(name_b, lin_b),
        prefer_cublasdx_for_linears=prefer_cublasdx_for_linears,
        cublasdx_precision=cublasdx_precision,
        cublasdx_sm=cublasdx_sm,
    )


# ---------------------------------------------------------------------------
# Lowering: diamond
# ---------------------------------------------------------------------------


_TILE_M = 32
_TILE_N = 32
_TILE_K = 32

# cuBLASDx tile size for the tensor-core path. Bumped from
# 32×32×32 per bridge #095: at the smaller tile cuBLASDx's dispatcher
# silently picks the SIMT fma.rn.f32 path; at 64×64×16 it switches
# to mma.sync (PTX dump confirmed). Smaller K-dim is intentional —
# cuBLASDx prefers wider M/N tiles with shallower K accumulation
# per call when targeting MMA atoms.
_TILE_M_CUBLASDX = 64
_TILE_N_CUBLASDX = 64
_TILE_K_CUBLASDX = 16


def _select_tile_shape(
    use_cublasdx: bool,
) -> tuple[int, int, int]:
    """Return ``(TM, TN, TK)`` for the selected backend.

    cuBLASDx wants 64×64×16 (per bridge #095) to engage mma.sync.
    The hand_rolled_fmaf path stays at 32×32×32 since it doesn't
    care about MMA atom granularity and the smaller tile keeps the
    matcher's existing shape constraints (multiples of 32) lenient.
    """
    if use_cublasdx:
        return (_TILE_M_CUBLASDX, _TILE_N_CUBLASDX, _TILE_K_CUBLASDX)
    return (_TILE_M, _TILE_N, _TILE_K)


def _probe_backends() -> _BackendAvailability:
    """Cheap availability probe for the optional codegen backends.

    Round-2a probes cuBLASDx via the public discovery helper. Round
    3 will add a Tile IR probe (``cuda.bindings.tile_ir`` import).

    The probe is best-effort and never raises — a missing or
    misconfigured backend translates to ``status="missing"`` plus a
    ``cublasdx_include=None``, and the matcher falls back to its
    hand-rolled bodies cleanly.
    """
    try:
        from compgen.runtime.native.cuda import (
            discover_cublasdx_include,
            discover_cutlass_include,
            discover_libcudacxx_include,
        )

        cublasdx_path = discover_cublasdx_include()
        libcudacxx_path = discover_libcudacxx_include()
        cutlass_path = discover_cutlass_include()
    except Exception as exc:  # noqa: BLE001
        return _BackendAvailability(
            cublasdx_include=None,
            cublasdx_status=f"probe-failed: {exc!r}",
        )

    libcudacxx_status = (
        "available"
        if libcudacxx_path is not None
        else "missing (install nvidia-cuda-cccl-cu13 or set $LIBCUDACXX_INCLUDE_PATH)"
    )
    cutlass_status = (
        "available"
        if cutlass_path is not None
        else "missing (nvidia-mathdx vendors CUTLASS under external/cutlass/include — install nvidia-mathdx, or set $CUTLASS_INCLUDE_PATH)"
    )

    if cublasdx_path is None:
        return _BackendAvailability(
            cublasdx_include=None,
            cublasdx_status="missing (install nvidia-mathdx for cuBLASDx-backed bodies)",
            libcudacxx_include=libcudacxx_path,
            libcudacxx_status=libcudacxx_status,
            cutlass_include=cutlass_path,
            cutlass_status=cutlass_status,
        )
    missing_deps: list[str] = []
    if libcudacxx_path is None:
        missing_deps.append("libcudacxx")
    if cutlass_path is None:
        missing_deps.append("CUTLASS")
    if missing_deps:
        return _BackendAvailability(
            cublasdx_include=cublasdx_path,
            cublasdx_status=(
                f"available-but-deps-missing ({', '.join(missing_deps)} "
                "not found; cuBLASDx bodies will NVRTC-fail until each "
                "include path resolves)"
            ),
            libcudacxx_include=libcudacxx_path,
            libcudacxx_status=libcudacxx_status,
            cutlass_include=cutlass_path,
            cutlass_status=cutlass_status,
        )
    return _BackendAvailability(
        cublasdx_include=cublasdx_path,
        cublasdx_status="available",
        libcudacxx_include=libcudacxx_path,
        libcudacxx_status=libcudacxx_status,
        cutlass_include=cutlass_path,
        cutlass_status=cutlass_status,
    )


def _emit_diamond(
    *,
    x_shape: tuple[int, ...],
    linear_a: tuple[str, nn.Linear],
    linear_b: tuple[str, nn.Linear],
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
) -> LoweringResult:
    """Emit the diamond MegakernelGraph + bodies.

    Mirrors the hand-built ``compgen.testing.workloads.diamond_dag``
    factory bit-for-bit, just driven by the matched torch module
    instead of being typed in by hand.
    """
    batch = int(x_shape[0])
    in_dim = linear_a[1].in_features
    out_dim = linear_a[1].out_features

    backends = _probe_backends()
    use_cublasdx = (
        prefer_cublasdx_for_linears
        and backends.cublasdx_status == "available"
        and backends.libcudacxx_status == "available"
        and backends.cutlass_status == "available"
    )
    tile_m, tile_n, tile_k = _select_tile_shape(use_cublasdx)

    # Tile-task counts: per-op task_shape = (NUM_TILES,) where
    # NUM_TILES = (batch / TM) * (out_dim / TN). The cuBLASDx path
    # uses 64×64×16 tiles so it engages mma.sync (per #095); the
    # fmaf path stays at 32×32×32.
    if batch % tile_m != 0 or out_dim % tile_n != 0:
        raise UnsupportedShape(
            f"diamond needs batch ({batch}) and out_dim ({out_dim}) "
            f"both divisible by tile size {tile_m}/{tile_n} "
            f"({'cuBLASDx' if use_cublasdx else 'fmaf'} path)"
        )
    if in_dim % tile_k != 0:
        raise UnsupportedShape(
            f"diamond needs in_dim ({in_dim}) divisible by K-tile {tile_k} "
            f"({'cuBLASDx' if use_cublasdx else 'fmaf'} path)"
        )

    tiles_per_row = out_dim // tile_n
    num_tiles = (batch // tile_m) * tiles_per_row

    # Event tensors — one cell per tile per producer→consumer edge.
    ev_a = EventTensor((num_tiles,), wait_count_default=1)
    ev_b = EventTensor((num_tiles,), wait_count_default=1)
    ev_add = EventTensor((num_tiles,), wait_count_default=1)
    ev_done = EventTensor((num_tiles,), wait_count_default=1)
    same_cell = lambda c: (c[0],)  # noqa: E731

    calls = (
        DeviceCall(
            name="linear_a",
            body_fn=lambda c: None,
            task_shape=(num_tiles,),
            out_edges=(EventEdge("ev_a", same_cell),),
        ),
        DeviceCall(
            name="linear_b",
            body_fn=lambda c: None,
            task_shape=(num_tiles,),
            out_edges=(EventEdge("ev_b", same_cell),),
        ),
        DeviceCall(
            name="add_op",
            body_fn=lambda c: None,
            task_shape=(num_tiles,),
            in_edges=(
                EventEdge("ev_a", same_cell),
                EventEdge("ev_b", same_cell),
            ),
            out_edges=(EventEdge("ev_add", same_cell),),
        ),
        DeviceCall(
            name="relu_op",
            body_fn=lambda c: None,
            task_shape=(num_tiles,),
            in_edges=(EventEdge("ev_add", same_cell),),
            out_edges=(EventEdge("ev_done", same_cell),),
        ),
    )
    graph = MegakernelGraph(
        name="diamond_lowered",
        calls=calls,
        event_tensors={"ev_a": ev_a, "ev_b": ev_b, "ev_add": ev_add, "ev_done": ev_done},
        policy="static",
    )

    bodies = _diamond_bodies(
        batch=batch,
        in_dim=in_dim,
        out_dim=out_dim,
        tiles_per_row=tiles_per_row,
        use_cublasdx=use_cublasdx,
        cublasdx_precision=cublasdx_precision,
        cublasdx_sm=cublasdx_sm,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    if use_cublasdx:
        linear_backend = "cublasdx_bf16_fp32" if cublasdx_precision == "bf16_fp32" else "cublasdx_fp32"
        if cublasdx_precision == "bf16_fp32":
            body_rationale = (
                "round-2d: cuBLASDx Precision<bf16, bf16, fp32> with "
                "Arrangement<row_major, row_major, row_major>. fp32 gmem "
                "→ __float2bfloat16 cast at smem load, fp32 accumulator + "
                "fp32 output. Engages Blackwell tensor cores; loses ~3-4 "
                "mantissa bits vs end-to-end fp32. discovery state: "
                f"{backends.cublasdx_status}."
            )
        else:
            body_rationale = (
                "round-2c+: cuBLASDx fp32 GEMM with "
                "Arrangement<row_major, row_major, row_major> per K-tile, "
                "validated bit-exact against torch.matmul on bwell #083. "
                f"discovery state: {backends.cublasdx_status}."
            )
    else:
        linear_backend = "hand_rolled_fmaf"
        prefer_note = (
            " (prefer_cublasdx_for_linears requested but backend unreachable)" if prefer_cublasdx_for_linears else ""
        )
        body_rationale = (
            f"shared-memory tiled fp32 fmaf body{prefer_note}. cuBLASDx "
            f"discovery: {backends.cublasdx_status}. Pass "
            "``prefer_cublasdx_for_linears=True`` to lower_torch_to_megakernel "
            "(or set the kwarg via compgen_compile_torch_model) to swap "
            "linear bodies to the cuBLASDx path."
        )
    # Plumb both cuBLASDx + libcudacxx include dirs into the
    # decision's NVRTC list. The matcher emits them eagerly even
    # though bodies are still hand_rolled_fmaf — the agent gets to
    # see what NVRTC will receive on the round-2c path, and the
    # round-2c body emission inherits a properly-populated list.
    paths: list[str] = []
    if backends.cublasdx_include is not None:
        paths.append(backends.cublasdx_include)
    if backends.libcudacxx_include is not None:
        paths.append(backends.libcudacxx_include)
    if backends.cutlass_include is not None:
        paths.append(backends.cutlass_include)
    nvrtc_include_paths = tuple(paths)
    nvrtc_extra_options: tuple[str, ...] = ("-default-device",) if use_cublasdx else ()
    diamond_backends = {
        "linear_a": linear_backend,
        "linear_b": linear_backend,
        "add_op": "hand_rolled_fmaf",  # elementwise — never cuBLASDx
        "relu_op": "hand_rolled_fmaf",  # elementwise — never cuBLASDx
    }
    decision = LoweringDecision(
        pattern_name="diamond",
        pattern_rationale=(
            "model has 2 nn.Linear children with matching shapes, no bias; "
            "forward output matches (linear_a(x) + linear_b(x)).relu() within "
            "fp32 ULP across one probe input."
        ),
        body_decisions=tuple(
            _BodyDecision(
                op_name=name,
                backend=diamond_backends[name],
                tile_shape=(tile_m, tile_n, tile_k),
                rationale=body_rationale,
            )
            for name in ("linear_a", "linear_b", "add_op", "relu_op")
        ),
        schedule_hints={
            "tile_grid": [batch // tile_m, out_dim // tile_n],
            "k_tiles": in_dim // tile_k,
            "block_dim": [32, 32, 1],
            "tile_shape": [tile_m, tile_n, tile_k],
        },
        total_tile_tasks=num_tiles * 4,
        backends=backends,
        nvrtc_include_paths=nvrtc_include_paths,
        nvrtc_extra_options=nvrtc_extra_options,
    )

    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=("x", "wa", "wb", "ya", "yb", "yadd", "yout"),
        decision=decision,
    )


def _diamond_bodies(
    *,
    batch: int,
    in_dim: int,
    out_dim: int,
    tiles_per_row: int,
    use_cublasdx: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
    tile_m: int = _TILE_M,
    tile_n: int = _TILE_N,
    tile_k: int = _TILE_K,
) -> dict[str, DeviceFunctionSource]:
    """Single-tile GEMM + elementwise bodies, parameterised by the
    matcher-derived shape so we don't re-hardcode (B, IN, OUT)."""
    common_dims = (
        f"const int B = {batch};\n"
        f"const int IN = {in_dim};\n"
        f"const int OUT = {out_dim};\n"
        f"const int TM = {tile_m}, TN = {tile_n}, TK = {tile_k};\n"
        f"const int TILES_PER_ROW = {tiles_per_row};\n"
    )

    def _gemm_body(weight_buf: int, out_buf: int) -> str:
        return (
            common_dims
            + r"""
const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x = (const float *)buffers[0];
const float *w = (const float *)buffers[__WEIGHT_BUF__];
float       *y = (float *)buffers[__OUT_BUF__];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < IN; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < IN)
        ? x[a_row * IN + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < OUT && w_k < IN)
        ? w[w_n * IN + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < OUT) {
    y[out_row * OUT + out_col] = acc;
}
""".replace("__WEIGHT_BUF__", str(weight_buf)).replace("__OUT_BUF__", str(out_buf))
        )

    # Tile-aware elementwise prelude. With BlockDim<32,32,1>=1024
    # threads each thread covers ceil(TM*TN/1024) elements via a
    # strided loop. Auto-degrades to 1 iteration when TM*TN=1024
    # (the 32-tile fmaf path) and runs 4 iterations when TM*TN=4096
    # (the 64-tile cuBLASDx path). Per bridge #097 — without this,
    # add_op + relu_op only cover 1024/4096 of the output tile.
    elementwise_elems = tile_m * tile_n
    elementwise_per_thread = (elementwise_elems + 1023) // 1024
    elementwise_prelude = common_dims + (
        "const int row_tile_idx = coord_x / TILES_PER_ROW;\n"
        "const int col_tile_idx = coord_x % TILES_PER_ROW;\n"
        "const int row_start = row_tile_idx * TM;\n"
        "const int col_start = col_tile_idx * TN;\n"
        "const int linear = threadIdx.y * 32 + threadIdx.x;\n"
    )

    def _elementwise_loop_open() -> str:
        return (
            f"#pragma unroll\nfor (int p = 0; p < {elementwise_per_thread}; ++p) {{\n"
            f"    int t_idx = p * 1024 + linear;\n"
            f"    if (t_idx < {elementwise_elems}) {{\n"
            f"        int dy = t_idx / TN;\n"
            f"        int dx = t_idx % TN;\n"
            f"        int row = row_start + dy;\n"
            f"        int col = col_start + dx;\n"
            f"        if (row < B && col < OUT) {{\n"
            f"            int idx = row * OUT + col;\n"
        )

    elementwise_loop_close = "        }\n    }\n}\n"

    if use_cublasdx:
        linear_a_src = _cublasdx_gemm_body(
            name="linear_a",
            b_dim=batch,
            k_dim=in_dim,
            n_dim=out_dim,
            n_tiles_per_row=tiles_per_row,
            x_buf=0,
            w_buf=1,
            out_buf=3,
            precision=cublasdx_precision,
            sm=cublasdx_sm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
        linear_b_src = _cublasdx_gemm_body(
            name="linear_b",
            b_dim=batch,
            k_dim=in_dim,
            n_dim=out_dim,
            n_tiles_per_row=tiles_per_row,
            x_buf=0,
            w_buf=2,
            out_buf=4,
            precision=cublasdx_precision,
            sm=cublasdx_sm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
    else:
        linear_a_src = DeviceFunctionSource(name="linear_a", body=_gemm_body(1, 3))
        linear_b_src = DeviceFunctionSource(name="linear_b", body=_gemm_body(2, 4))

    add_body = (
        elementwise_prelude
        + "\nconst float *ya = (const float *)buffers[3];\n"
        + "const float *yb = (const float *)buffers[4];\n"
        + "float       *yadd = (float *)buffers[5];\n"
        + _elementwise_loop_open()
        + "            yadd[idx] = ya[idx] + yb[idx];\n"
        + elementwise_loop_close
    )
    relu_body = (
        elementwise_prelude
        + "\nconst float *yadd = (const float *)buffers[5];\n"
        + "float       *yout = (float *)buffers[6];\n"
        + _elementwise_loop_open()
        + "            float v = yadd[idx];\n"
        + "            yout[idx] = v > 0.0f ? v : 0.0f;\n"
        + elementwise_loop_close
    )

    return {
        "linear_a": linear_a_src,
        "linear_b": linear_b_src,
        "add_op": DeviceFunctionSource(name="add_op", body=add_body),
        "relu_op": DeviceFunctionSource(name="relu_op", body=relu_body),
    }


def _cublasdx_gemm_body(
    *,
    name: str,
    b_dim: int,
    k_dim: int,
    n_dim: int,
    n_tiles_per_row: int,
    x_buf: int,
    w_buf: int,
    out_buf: int,
    precision: str = "fp32",
    sm: int = 1000,
    tile_m: int = _TILE_M,
    tile_n: int = _TILE_N,
    tile_k: int = _TILE_K,
    apply_relu: bool = False,
) -> DeviceFunctionSource:
    """Emit a tile-shape-parameterised GEMM body using cuBLASDx as
    the inner tile multiplier.

    The body's outer K-loop iterates ``k_dim/tile_k`` times, calling
    ``BLAS().execute(1.0, A_tile, B_tile, beta_iter, C_tile)``
    with ``beta_iter`` = 0.0 on the first iteration and 1.0
    thereafter so the K-tiles accumulate into the same C tile.

    Buffer convention: same as the hand_rolled path —
    buffers[x_buf] = activations (B, K) row-major fp32,
    buffers[w_buf] = nn.Linear weight (N, K) row-major fp32,
    buffers[out_buf] = output (B, N) row-major fp32. The weight load
    transposes per-element so cuBLASDx's row-major B_tile holds
    ``B_tile[k, n] = W[n, k]``.

    Arrangement is fixed at ``<row_major, row_major, row_major>``
    per bwell #083.

    Tile-shape contract:
    - Block-level: ``BlockDim<32, 32, 1>`` = 1024 threads.
    - A_tile: ``tile_m × tile_k`` elements row-major.
    - B_tile: ``tile_k × tile_n`` elements row-major (transposed
      access from W).
    - C_tile: ``tile_m × tile_n`` elements row-major.
    - Each thread cooperates on ``ceil(elems/1024)`` elements per
      smem buffer per K-iter.

    Args:
        precision: ``"fp32"`` → fp32 SIMT (Precision<float>);
            ``"bf16_fp32"`` → tensor-core path
            (Precision<bf16, bf16, float>).
        sm: cuBLASDx ``SM<...>`` tag (1000 = Blackwell tcgen05).
        tile_m, tile_n, tile_k: per-call tile shape. Bridge #095
            confirmed 64×64×16 engages mma.sync; 32×32×32 stays
            on SIMT regardless of arch.
    """
    if precision == "bf16_fp32":
        precision_tag = "cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>"
        smem_dtype_ab = "__nv_bfloat16"
        smem_dtype_c = "float"
        a_cast_open = "__float2bfloat16("
        a_cast_close = ")"
        a_zero = "__float2bfloat16(0.0f)"
        b_cast_open = "__float2bfloat16("
        b_cast_close = ")"
        b_zero = "__float2bfloat16(0.0f)"
        extra_includes: tuple[str, ...] = (
            "#include <cuda_bf16.h>",
            "#include <cublasdx.hpp>",
        )
    elif precision == "fp32":
        precision_tag = "cublasdx::Precision<float>"
        smem_dtype_ab = "float"
        smem_dtype_c = "float"
        a_cast_open = ""
        a_cast_close = ""
        a_zero = "0.0f"
        b_cast_open = ""
        b_cast_close = ""
        b_zero = "0.0f"
        extra_includes = ("#include <cublasdx.hpp>",)
    else:
        raise ValueError(f"_cublasdx_gemm_body: unsupported precision={precision!r}")

    a_elems = tile_m * tile_k
    b_elems = tile_k * tile_n
    c_elems = tile_m * tile_n
    threads = 1024

    # Number of cooperative passes each thread does per smem buffer.
    a_per_thread = (a_elems + threads - 1) // threads
    b_per_thread = (b_elems + threads - 1) // threads
    c_per_thread = (c_elems + threads - 1) // threads

    a_load_loop = f"""
#pragma unroll
for (int p = 0; p < {a_per_thread}; ++p) {{
    int idx = p * {threads} + linear;
    if (idx < {a_elems}) {{
        int m = idx / {tile_k};
        int k_off = idx % {tile_k};
        int a_row = row_start + m;
        int a_col = k_tile + k_off;
        smem_a[idx] = (a_row < B && a_col < IN)
            ? {a_cast_open}x[a_row * IN + a_col]{a_cast_close}
            : {a_zero};
    }}
}}
"""

    # B is loaded transposed from W: B_tile[k, n] = W[col_start+n, k_tile+k].
    b_load_loop = f"""
#pragma unroll
for (int p = 0; p < {b_per_thread}; ++p) {{
    int idx = p * {threads} + linear;
    if (idx < {b_elems}) {{
        int k_off = idx / {tile_n};
        int n = idx % {tile_n};
        int w_n = col_start + n;
        int w_k = k_tile + k_off;
        smem_b[idx] = (w_n < OUT && w_k < IN)
            ? {b_cast_open}w[w_n * IN + w_k]{b_cast_close}
            : {b_zero};
    }}
}}
"""

    # Store smem_c → gmem y at (row_start, col_start). Each thread
    # writes ``c_per_thread`` elements; bounds-check against B/OUT.
    # When apply_relu, fold relu into the store epilogue (Wave 2.5
    # — eliminates the separate relu_up pointwise pass + y_up
    # round-trip through gmem).
    store_expr = "fmaxf(smem_c[idx], 0.0f)" if apply_relu else "smem_c[idx]"
    c_store_loop = f"""
#pragma unroll
for (int p = 0; p < {c_per_thread}; ++p) {{
    int idx = p * {threads} + linear;
    if (idx < {c_elems}) {{
        int m = idx / {tile_n};
        int n = idx % {tile_n};
        int out_row = row_start + m;
        int out_col = col_start + n;
        if (out_row < B && out_col < OUT) {{
            y[out_row * OUT + out_col] = {store_expr};
        }}
    }}
}}
"""

    body = (
        f"const int B = {b_dim};\n"
        f"const int IN = {k_dim};\n"
        f"const int OUT = {n_dim};\n"
        f"const int TM = {tile_m}, TN = {tile_n}, TK = {tile_k};\n"
        f"const int TILES_PER_ROW = {n_tiles_per_row};\n"
        f"\nusing BLAS = decltype(\n"
        f"      cublasdx::Size<{tile_m}, {tile_n}, {tile_k}>()\n"
        f"    + {precision_tag}()\n"
        f"    + cublasdx::Type<cublasdx::type::real>()\n"
        f"    + cublasdx::Function<cublasdx::function::MM>()\n"
        f"    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()\n"
        f"    + cublasdx::Block()\n"
        f"    + cublasdx::BlockDim<32, 32, 1>()\n"
        f"    + cublasdx::SM<{sm}>()\n"
        f");\n\n"
        r"""const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x = (const float *)buffers[__X_BUF__];
const float *w = (const float *)buffers[__W_BUF__];
float       *y = (float *)buffers[__OUT_BUF__];

"""
        + f"__shared__ {smem_dtype_ab} smem_a[{a_elems}];\n"
        + f"__shared__ {smem_dtype_ab} smem_b[{b_elems}];\n"
        + f"__shared__ {smem_dtype_c} smem_c[{c_elems}];\n"
        + r"""
const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int linear = ty * 32 + tx;

float beta_iter = 0.0f;
for (int k_tile = 0; k_tile < IN; k_tile += TK) {
"""
        + a_load_loop
        + b_load_loop
        + r"""    __syncthreads();

    BLAS().execute(1.0f, smem_a, smem_b, beta_iter, smem_c);
    __syncthreads();

    beta_iter = 1.0f;
}
"""
        + c_store_loop
    )
    body = body.replace("__X_BUF__", str(x_buf)).replace("__W_BUF__", str(w_buf)).replace("__OUT_BUF__", str(out_buf))
    return DeviceFunctionSource(
        name=name,
        body=body,
        included_headers=extra_includes,
    )


# ---------------------------------------------------------------------------
# Pattern matcher + emit: FFN  (y = down(relu(up(x))))
# ---------------------------------------------------------------------------


def _match_ffn(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
    fuse_epilogue: bool = False,
) -> LoweringResult:
    """Match ``y = down(relu(up(x)))``.

    Two ``nn.Linear`` children chained through a relu, no bias on
    either, with ``up.out_features == down.in_features`` and
    distinct shapes (otherwise diamond would have matched first).
    Tile graph has K-fan-in on the relu_up→linear_down edge so the
    cross-K dependency is explicit in the event-tensor structure.

    Wave 2.5 (per bridge #137): when ``fuse_epilogue=True``, fold
    the relu into ``linear_up``'s MMA epilogue. Eliminates the
    ``relu_up`` pointwise pool (and the ``ev_up`` event tensor +
    ``y_up`` intermediate buffer that fed it), collapsing the
    bipartite layer structure that kept ``coop_share`` low at
    paper shapes.
    """
    linears = [(name, m) for name, m in model.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != 2:
        raise UnsupportedShape(f"ffn requires exactly 2 nn.Linear children; got {len(linears)}")
    name_up, lin_up = linears[0]
    name_down, lin_down = linears[1]
    if lin_up.bias is not None or lin_down.bias is not None:
        raise UnsupportedShape("ffn matcher requires bias=False on both linears")
    if lin_up.out_features != lin_down.in_features:
        raise UnsupportedShape(
            f"ffn requires up.out_features ({lin_up.out_features}) == "
            f"down.in_features ({lin_down.in_features}) — hidden mismatch"
        )

    x = sample_inputs[0]
    # Accept ND inputs — flatten leading dims into the batch axis for
    # the tile graph (per bridge #108).
    if x.ndim < 2 or x.shape[-1] != lin_up.in_features:
        raise UnsupportedShape(
            f"ffn input shape {tuple(x.shape)} does not match up.in_features={lin_up.in_features} on the trailing axis"
        )
    with torch.no_grad():
        try:
            y = model(x)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedShape(f"ffn forward probe raised {exc!r}") from exc
    expected_y_shape = tuple(x.shape[:-1]) + (lin_down.out_features,)
    if tuple(y.shape) != expected_y_shape:
        raise UnsupportedShape(f"ffn forward returned shape {tuple(y.shape)}, expected {expected_y_shape}")
    with torch.no_grad():
        expected = lin_down(torch.relu(lin_up(x)))
    if not torch.allclose(y, expected, atol=1e-5, rtol=1e-5):
        raise UnsupportedShape("ffn pattern children match but forward output disagrees with down(relu(up(x)))")

    batch_flat = 1
    for d in x.shape[:-1]:
        batch_flat *= int(d)

    return _emit_ffn(
        x_shape=(batch_flat, int(x.shape[-1])),
        linear_up=(name_up, lin_up),
        linear_down=(name_down, lin_down),
        prefer_cublasdx_for_linears=prefer_cublasdx_for_linears,
        cublasdx_precision=cublasdx_precision,
        cublasdx_sm=cublasdx_sm,
        fuse_epilogue=fuse_epilogue,
    )


def _emit_ffn(
    *,
    x_shape: tuple[int, ...],
    linear_up: tuple[str, nn.Linear],
    linear_down: tuple[str, nn.Linear],
    prefer_cublasdx_for_linears: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
    fuse_epilogue: bool = False,
) -> LoweringResult:
    """Emit the FFN MegakernelGraph + bodies.

    Tile graph topology (default, ``fuse_epilogue=False``) — three
    ops, K-fan-in on the second edge:

        linear_up  → ev_up  (B*H tiles, wait=1)
        relu_up    : reads ev_up cell c[0]; notifies ev_relu folded to b
        ev_relu    (B tiles, wait=H_TILES)
        linear_down: reads ev_relu cell c[0]/O_TILES (folded to b);
                     notifies ev_done cell c[0]

    With ``fuse_epilogue=True`` (Wave 2.5) — two ops, no ev_up,
    no y_up buffer:

        linear_up_relu : applies relu in the MMA epilogue, writes
                         y_relu directly; notifies ev_relu folded to b
        ev_relu    (B tiles, wait=H_TILES)
        linear_down: reads ev_relu cell c[0]/O_TILES; notifies ev_done
    """
    batch = int(x_shape[0])
    in_dim = linear_up[1].in_features
    hidden = linear_up[1].out_features
    out_dim = linear_down[1].out_features

    backends = _probe_backends()
    use_cublasdx = (
        prefer_cublasdx_for_linears
        and backends.cublasdx_status == "available"
        and backends.libcudacxx_status == "available"
        and backends.cutlass_status == "available"
    )
    tile_m, tile_n, tile_k = _select_tile_shape(use_cublasdx)
    backend_label = "cuBLASDx" if use_cublasdx else "fmaf"

    if batch % tile_m != 0:
        raise UnsupportedShape(f"ffn needs batch ({batch}) divisible by TM={tile_m} ({backend_label} path)")
    if hidden % tile_n != 0 or hidden % tile_k != 0:
        raise UnsupportedShape(
            f"ffn needs hidden ({hidden}) divisible by both "
            f"TN={tile_n} (relu_up tiles) and TK={tile_k} "
            f"(K-fan-in count) — {backend_label} path"
        )
    if out_dim % tile_n != 0:
        raise UnsupportedShape(f"ffn needs out_dim ({out_dim}) divisible by TN={tile_n} ({backend_label} path)")
    if in_dim % tile_k != 0:
        raise UnsupportedShape(f"ffn needs in_dim ({in_dim}) divisible by TK={tile_k} ({backend_label} path)")

    b_tiles = batch // tile_m
    h_tiles = hidden // tile_n
    o_tiles = out_dim // tile_n
    n_up = b_tiles * h_tiles
    n_down = b_tiles * o_tiles

    # K-fan-in: each linear_down (b, o) tile reads all H_TILES of
    # the producer's (b, *) row. One cell per row stripe with
    # wait_count = H_TILES so it falls to zero only after every
    # producer tile in that stripe has notified.
    ev_relu = EventTensor((b_tiles,), wait_count_default=h_tiles)
    ev_done = EventTensor((n_down,), wait_count_default=1)

    # Notifier folds (b_tile, h_tile) -> b_tile; the linear_down
    # reader folds (b_tile, o_tile) -> b_tile. Different task layouts
    # on each side, same target cell.
    relu_to_ev_relu = lambda c: (c[0] // h_tiles,)  # noqa: E731
    down_from_ev_relu = lambda c: (c[0] // o_tiles,)  # noqa: E731
    same_cell = lambda c: (c[0],)  # noqa: E731

    if fuse_epilogue:
        # Two-op topology: linear_up_relu writes y_relu directly,
        # then linear_down reads it. No ev_up, no y_up.
        calls = (
            DeviceCall(
                name="linear_up_relu",
                body_fn=lambda c: None,
                task_shape=(n_up,),
                out_edges=(EventEdge("ev_relu", relu_to_ev_relu),),
            ),
            DeviceCall(
                name="linear_down",
                body_fn=lambda c: None,
                task_shape=(n_down,),
                in_edges=(EventEdge("ev_relu", down_from_ev_relu),),
                out_edges=(EventEdge("ev_done", same_cell),),
            ),
        )
        event_tensors = {"ev_relu": ev_relu, "ev_done": ev_done}
    else:
        ev_up = EventTensor((n_up,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="linear_up",
                body_fn=lambda c: None,
                task_shape=(n_up,),
                out_edges=(EventEdge("ev_up", same_cell),),
            ),
            DeviceCall(
                name="relu_up",
                body_fn=lambda c: None,
                task_shape=(n_up,),
                in_edges=(EventEdge("ev_up", same_cell),),
                out_edges=(EventEdge("ev_relu", relu_to_ev_relu),),
            ),
            DeviceCall(
                name="linear_down",
                body_fn=lambda c: None,
                task_shape=(n_down,),
                in_edges=(EventEdge("ev_relu", down_from_ev_relu),),
                out_edges=(EventEdge("ev_done", same_cell),),
            ),
        )
        event_tensors = {
            "ev_up": ev_up,
            "ev_relu": ev_relu,
            "ev_done": ev_done,
        }
    graph = MegakernelGraph(
        name="ffn_lowered",
        calls=calls,
        event_tensors=event_tensors,
        policy="static",
    )

    bodies = _ffn_bodies(
        batch=batch,
        in_dim=in_dim,
        hidden=hidden,
        out_dim=out_dim,
        h_tiles=h_tiles,
        o_tiles=o_tiles,
        use_cublasdx=use_cublasdx,
        cublasdx_precision=cublasdx_precision,
        cublasdx_sm=cublasdx_sm,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        fuse_epilogue=fuse_epilogue,
    )

    if use_cublasdx:
        linear_backend = "cublasdx_bf16_fp32" if cublasdx_precision == "bf16_fp32" else "cublasdx_fp32"
        if cublasdx_precision == "bf16_fp32":
            body_rationale = (
                "round-2d FFN: cuBLASDx Precision<bf16, bf16, fp32> with "
                "Arrangement<row_major, row_major, row_major>. fp32 gmem "
                "cast to bf16 at smem load, fp32 accumulator + fp32 output. "
                f"Engages Blackwell tensor cores. discovery: {backends.cublasdx_status}."
            )
        else:
            body_rationale = (
                "round-2c+: cuBLASDx fp32 GEMM with "
                "Arrangement<row_major, row_major, row_major> per K-tile, "
                "validated bit-exact against torch.matmul on bwell #083. "
                f"discovery state: {backends.cublasdx_status}."
            )
    else:
        linear_backend = "hand_rolled_fmaf"
        prefer_note = (
            " (prefer_cublasdx_for_linears requested but backend unreachable)" if prefer_cublasdx_for_linears else ""
        )
        body_rationale = (
            f"round-2 FFN default: shared-memory tiled fp32 fmaf{prefer_note}. "
            f"cuBLASDx discovery: {backends.cublasdx_status}. Pass "
            "``prefer_cublasdx_for_linears=True`` to swap linear_up + "
            "linear_down to cuBLASDx; relu_up stays fmaf either way."
        )
    paths: list[str] = []
    if backends.cublasdx_include is not None:
        paths.append(backends.cublasdx_include)
    if backends.libcudacxx_include is not None:
        paths.append(backends.libcudacxx_include)
    if backends.cutlass_include is not None:
        paths.append(backends.cutlass_include)
    nvrtc_include_paths = tuple(paths)
    nvrtc_extra_options: tuple[str, ...] = ("-default-device",) if use_cublasdx else ()
    if fuse_epilogue:
        ffn_backends = {
            "linear_up_relu": linear_backend,
            "linear_down": linear_backend,
        }
        op_order = ("linear_up_relu", "linear_down")
        total_tile_tasks = n_up + n_down
        user_buffer_layout = ("x", "w_up", "w_down", "y_relu", "y_out")
        pattern_rationale_extra = " — relu fused into linear_up's MMA epilogue (Wave 2.5)"
    else:
        ffn_backends = {
            "linear_up": linear_backend,
            "relu_up": "hand_rolled_fmaf",  # elementwise — never cuBLASDx
            "linear_down": linear_backend,
        }
        op_order = ("linear_up", "relu_up", "linear_down")
        total_tile_tasks = n_up * 2 + n_down
        user_buffer_layout = ("x", "w_up", "w_down", "y_up", "y_relu", "y_out")
        pattern_rationale_extra = ""
    schedule_hints: dict[str, Any] = {
        "tile_grid_up": [b_tiles, h_tiles],
        "tile_grid_down": [b_tiles, o_tiles],
        "k_tiles_up": in_dim // tile_k,
        "k_tiles_down": hidden // tile_k,
        "k_fan_in": h_tiles,
        "block_dim": [32, 32, 1],
        "tile_shape": [tile_m, tile_n, tile_k],
    }
    if fuse_epilogue:
        schedule_hints["epilogue_fusion"] = "relu_into_linear_up"
    decision = LoweringDecision(
        pattern_name="ffn",
        pattern_rationale=(
            "model has 2 nn.Linear children chained as down(relu(up(x))); "
            f"shapes up={in_dim}->{hidden}, down={hidden}->{out_dim}, no bias; "
            "forward output matches the canonical FFN within fp32 ULP across "
            f"one probe input{pattern_rationale_extra}."
        ),
        body_decisions=tuple(
            _BodyDecision(
                op_name=name,
                backend=ffn_backends[name],
                tile_shape=(tile_m, tile_n, tile_k),
                rationale=body_rationale,
            )
            for name in op_order
        ),
        schedule_hints=schedule_hints,
        total_tile_tasks=total_tile_tasks,
        backends=backends,
        nvrtc_include_paths=nvrtc_include_paths,
        nvrtc_extra_options=nvrtc_extra_options,
    )

    return LoweringResult(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=user_buffer_layout,
        decision=decision,
    )


def _ffn_bodies(
    *,
    batch: int,
    in_dim: int,
    hidden: int,
    out_dim: int,
    h_tiles: int,
    o_tiles: int,
    use_cublasdx: bool = False,
    cublasdx_precision: str = "fp32",
    cublasdx_sm: int = 1000,
    tile_m: int = _TILE_M,
    tile_n: int = _TILE_N,
    tile_k: int = _TILE_K,
    fuse_epilogue: bool = False,
) -> dict[str, DeviceFunctionSource]:
    """Body emission for FFN.

    Default (``fuse_epilogue=False``): three bodies — linear_up GEMM
    (in→hidden), elementwise relu_up, linear_down GEMM (hidden→out).
    Buffer layout: 0=x, 1=w_up, 2=w_down, 3=y_up, 4=y_relu, 5=y_out.

    Fused (``fuse_epilogue=True``, Wave 2.5): two bodies — single
    linear_up_relu GEMM that applies relu in the store epilogue,
    then linear_down GEMM. Buffer layout: 0=x, 1=w_up, 2=w_down,
    3=y_relu, 4=y_out (no y_up).
    """

    def _gemm_body(
        *,
        name: str,
        b_dim: int,
        k_dim: int,
        n_dim: int,
        n_tiles_per_row: int,
        x_buf: int,
        w_buf: int,
        out_buf: int,
        apply_relu: bool = False,
    ) -> DeviceFunctionSource:
        body = (
            f"const int B = {b_dim};\n"
            f"const int IN = {k_dim};\n"
            f"const int OUT = {n_dim};\n"
            f"const int TM = {_TILE_M}, TN = {_TILE_N}, TK = {_TILE_K};\n"
            f"const int TILES_PER_ROW = {n_tiles_per_row};\n"
            r"""
const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x = (const float *)buffers[__X_BUF__];
const float *w = (const float *)buffers[__W_BUF__];
float       *y = (float *)buffers[__OUT_BUF__];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < IN; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < IN)
        ? x[a_row * IN + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < OUT && w_k < IN)
        ? w[w_n * IN + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < OUT) {
    y[out_row * OUT + out_col] = __EPILOGUE__;
}
"""
        )
        epilogue = "(acc > 0.0f ? acc : 0.0f)" if apply_relu else "acc"
        body = (
            body.replace("__X_BUF__", str(x_buf))
            .replace("__W_BUF__", str(w_buf))
            .replace("__OUT_BUF__", str(out_buf))
            .replace("__EPILOGUE__", epilogue)
        )
        return DeviceFunctionSource(name=name, body=body)

    # relu_up handles a TM×TN tile. With BlockDim<32,32,1>=1024
    # threads, each thread covers ceil(TM*TN/1024) elements. For
    # the fmaf path (TM=TN=32), that's 1 elt/thread (the 1-iteration
    # loop). For cuBLASDx (TM=TN=64), 4 elts/thread.
    relu_elems = tile_m * tile_n
    relu_per_thread = (relu_elems + 1023) // 1024
    relu_body = (
        f"const int B = {batch};\n"
        f"const int OUT = {hidden};\n"
        f"const int TM = {tile_m}, TN = {tile_n};\n"
        f"const int TILES_PER_ROW = {h_tiles};\n"
        r"""
const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;
const int linear = threadIdx.y * 32 + threadIdx.x;

const float *y_up = (const float *)buffers[3];
float       *y_relu = (float *)buffers[4];

"""
        + f"#pragma unroll\nfor (int p = 0; p < {relu_per_thread}; ++p) {{\n"
        + "    int t_idx = p * 1024 + linear;\n"
        + f"    if (t_idx < {relu_elems}) {{\n"
        + "        int m = t_idx / TN;\n"
        + "        int n = t_idx % TN;\n"
        + r"""        int row = row_start + m;
        int col = col_start + n;
        if (row < B && col < OUT) {
            int g_idx = row * OUT + col;
            float v = y_up[g_idx];
            y_relu[g_idx] = v > 0.0f ? v : 0.0f;
        }
    }
}
"""
    )

    if fuse_epilogue:
        # Fused buffer layout: 0=x, 1=w_up, 2=w_down, 3=y_relu, 4=y_out.
        # linear_up_relu writes y_relu directly with relu epilogue.
        if use_cublasdx:
            linear_up_relu_src = _cublasdx_gemm_body(
                name="linear_up_relu",
                b_dim=batch,
                k_dim=in_dim,
                n_dim=hidden,
                n_tiles_per_row=h_tiles,
                x_buf=0,
                w_buf=1,
                out_buf=3,
                precision=cublasdx_precision,
                sm=cublasdx_sm,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                apply_relu=True,
            )
            linear_down_src = _cublasdx_gemm_body(
                name="linear_down",
                b_dim=batch,
                k_dim=hidden,
                n_dim=out_dim,
                n_tiles_per_row=o_tiles,
                x_buf=3,
                w_buf=2,
                out_buf=4,
                precision=cublasdx_precision,
                sm=cublasdx_sm,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
            )
        else:
            linear_up_relu_src = _gemm_body(
                name="linear_up_relu",
                b_dim=batch,
                k_dim=in_dim,
                n_dim=hidden,
                n_tiles_per_row=h_tiles,
                x_buf=0,
                w_buf=1,
                out_buf=3,
                apply_relu=True,
            )
            linear_down_src = _gemm_body(
                name="linear_down",
                b_dim=batch,
                k_dim=hidden,
                n_dim=out_dim,
                n_tiles_per_row=o_tiles,
                x_buf=3,
                w_buf=2,
                out_buf=4,
            )
        return {
            "linear_up_relu": linear_up_relu_src,
            "linear_down": linear_down_src,
        }

    if use_cublasdx:
        linear_up_src = _cublasdx_gemm_body(
            name="linear_up",
            b_dim=batch,
            k_dim=in_dim,
            n_dim=hidden,
            n_tiles_per_row=h_tiles,
            x_buf=0,
            w_buf=1,
            out_buf=3,
            precision=cublasdx_precision,
            sm=cublasdx_sm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
        linear_down_src = _cublasdx_gemm_body(
            name="linear_down",
            b_dim=batch,
            k_dim=hidden,
            n_dim=out_dim,
            n_tiles_per_row=o_tiles,
            x_buf=4,
            w_buf=2,
            out_buf=5,
            precision=cublasdx_precision,
            sm=cublasdx_sm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
    else:
        linear_up_src = _gemm_body(
            name="linear_up",
            b_dim=batch,
            k_dim=in_dim,
            n_dim=hidden,
            n_tiles_per_row=h_tiles,
            x_buf=0,
            w_buf=1,
            out_buf=3,
        )
        linear_down_src = _gemm_body(
            name="linear_down",
            b_dim=batch,
            k_dim=hidden,
            n_dim=out_dim,
            n_tiles_per_row=o_tiles,
            x_buf=4,
            w_buf=2,
            out_buf=5,
        )

    return {
        "linear_up": linear_up_src,
        "relu_up": DeviceFunctionSource(name="relu_up", body=relu_body),
        "linear_down": linear_down_src,
    }
