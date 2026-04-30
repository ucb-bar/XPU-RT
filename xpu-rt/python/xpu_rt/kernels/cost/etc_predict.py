"""ETC-vs-eager dispatch cost prediction.

The per-bundle perf gate: predict whether the Event Tensor Compiler
bundle will beat eager PyTorch by ``threshold`` (default 1.2×). If
it won't, surface a typed :class:`WontWinError` so the agent can
either accept the bundle as a tracing/debugging artifact or fall
through to eager dispatch.

Per bridge #099 diagnosis: ETC's per-task scheduling overhead is
**fixed cost per task**. Eager amortizes its single kernel launch
across all SMs. So ETC wins when:

    per_task_gemm_us > scheduling_overhead_us / target_threshold

…i.e., when the per-task GEMM is large enough that the megakernel's
event-tensor decrement + cooperative dispatch + per-tile setup is
small in comparison.

This module gives a compile-time prediction so the agent doesn't
ship a bundle that's predictably slower than eager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Empirical scheduling-overhead constant for the megakernel's
# per-task dispatch. Measured on bwell during the #099 round at the
# 32×32×32 tile (most data-dense): ~1.0 µs per (event-tensor wait +
# task body invoke + event-tensor notify) cycle on Blackwell sm_120
# in the static-schedule path. Conservative — actual overhead may
# be smaller at the 64-tile cuBLASDx path since fewer events fire.
#
# This constant is the load-bearing assumption of the cost model.
# When bwell's per-task instrumentation (#099's request 2) lands
# we'll calibrate it; for now it's the bridge-derived empirical.
_SCHEDULING_OVERHEAD_US_PER_TASK = 1.0

# Eager PyTorch's single-launch overhead. Includes cuBLAS handle
# selection + driver kernel launch + grid setup. Empirically
# dominated by ~5-10 µs on Blackwell; we use the conservative end.
_EAGER_LAUNCH_OVERHEAD_US = 10.0

# Per-arch peak throughput tables.
#
# Wave 1.14c — moved to the per-arch leaf modules under
# ``targets/gpu/nvidia/{blackwell,hopper,ampere}/cost.py``. The
# universal predictor queries the leaves through
# :func:`_lookup_arch_tflops` so adding a new arch is one
# leaf-file change rather than touching this module.
#
# These local fallback tables stay so the predictor still works
# when the leaves aren't importable (e.g. tests that mock the
# registry, or older installs). Both old + new paths must agree
# for the ones already-shipped in the leaves; future-only arches
# (sm_130 etc.) only need the new path.
_FP32_SIMT_TFLOPS_PER_SM = {
    "100": 4.5,  # B200 datacenter
    "120": 4.5,  # workstation Blackwell
    "90": 4.0,  # H100
    "89": 3.0,  # Ada
    "86": 2.0,  # Ampere consumer
    "80": 3.0,  # A100
}

_BF16_TC_TFLOPS_PER_SM = {
    "100": 50.0,
    "120": 50.0,
    "90": 40.0,
    "89": 22.0,
    "86": 8.0,
    "80": 16.0,
}


def _lookup_arch_tflops(arch_key: str, *, tensor_core: bool) -> float:
    """Per-arch peak throughput. Looks up first in the per-arch
    leaf (``targets/gpu/nvidia/<arch>/cost.py``), falls back to
    the local table when the leaf isn't importable.

    Args:
        arch_key: ``"100"`` / ``"90"`` / ``"86"`` etc. — the
            stripped form of an NVRTC arch tag.
        tensor_core: True → bf16+fp32-acc tensor-core peak; False
            → fp32 SIMT peak.

    Returns: TFLOPS/s/SM at the requested precision.
    """
    # Map arch_key → leaf module path. Leaves only exist for the
    # arches we've explicitly added; others fall back.
    arch_to_leaf = {
        "100": "compgen.targets.gpu.nvidia.blackwell.cost",
        "120": "compgen.targets.gpu.nvidia.blackwell.cost",
        "90": "compgen.targets.gpu.nvidia.hopper.cost",
        "80": "compgen.targets.gpu.nvidia.ampere.cost",
        "86": "compgen.targets.gpu.nvidia.ampere.cost",
    }
    leaf_path = arch_to_leaf.get(arch_key)
    if leaf_path is not None:
        try:
            mod = __import__(leaf_path, fromlist=["*"])
            attr = "PEAK_BF16_TC_TFLOPS_PER_SM" if tensor_core else "PEAK_FP32_TFLOPS_PER_SM"
            value = getattr(mod, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        except ImportError:
            pass

    # Fallback to local table.
    table = _BF16_TC_TFLOPS_PER_SM if tensor_core else _FP32_SIMT_TFLOPS_PER_SM
    default = 50.0 if tensor_core else 4.0
    return table.get(arch_key, default)


# Empirical eager-cuBLAS throughputs per arch, per dtype (TFLOPS/s/SM).
# These are NOT silicon peaks — they bake in driver overhead, tile-shape
# inefficiency, and dtype-routing cost. Calibrated from bridge #118
# measurements. Used only for the eager-vs-ETC speedup prediction; the
# ETC-side rates remain peak / TC because the megakernel can engage
# tensor cores fully when the tile shape and precision align.
_FALLBACK_EAGER_FP32_TFLOPS_PER_SM = {
    "100": 0.47,
    "120": 0.47,
    "90": 0.40,
    "89": 0.30,
    "86": 0.20,
    "80": 0.20,
}
_FALLBACK_EAGER_BF16_TFLOPS_PER_SM = {
    "100": 2.65,
    "120": 2.65,
    "90": 2.30,
    "89": 1.50,
    "86": 1.20,
    "80": 1.20,
}
_FALLBACK_EAGER_FP8_TFLOPS_PER_SM = {
    "100": 5.30,
    "120": 5.30,
    "90": 4.50,
    "89": 1.50,
    "86": 1.20,
    "80": 1.20,
}
_FALLBACK_COOPERATIVE_SYNC_US_PER_WAVE = {
    "100": 70.0,
    "120": 70.0,
    "90": 60.0,
    "89": 50.0,
    "86": 50.0,
    "80": 50.0,
}


def _lookup_eager_tflops(arch_key: str, *, dtype: str) -> float:
    """Per-arch *empirical* eager-cuBLAS throughput per SM.

    Distinct from :func:`_lookup_arch_tflops` (silicon peak) — this
    is what cuBLAS actually delivers at typical shapes, baked from
    bridge #118's MLP-1 measurement. The cost model uses this for
    the eager side of the speedup gate so the prediction tracks
    observed wall-clock instead of theoretical peak.

    Args:
        arch_key: stripped NVRTC arch tag (``"100"`` / ``"90"`` / ...)
        dtype: ``"fp32"``, ``"bf16"``, ``"fp16"``, or ``"fp8"``.

    Returns: TFLOPS/s/SM at the requested dtype.
    """
    arch_to_leaf = {
        "100": "compgen.targets.gpu.nvidia.blackwell.cost",
        "120": "compgen.targets.gpu.nvidia.blackwell.cost",
        "90": "compgen.targets.gpu.nvidia.hopper.cost",
        "80": "compgen.targets.gpu.nvidia.ampere.cost",
        "86": "compgen.targets.gpu.nvidia.ampere.cost",
    }
    leaf_path = arch_to_leaf.get(arch_key)
    attr_by_dtype = {
        "fp32": "PEAK_EAGER_FP32_TFLOPS_PER_SM",
        "bf16": "PEAK_EAGER_BF16_TFLOPS_PER_SM",
        "fp16": "PEAK_EAGER_BF16_TFLOPS_PER_SM",  # fp16 ~ bf16 throughput
        "fp8": "PEAK_EAGER_FP8_TFLOPS_PER_SM",
        "fp4": "PEAK_EAGER_FP8_TFLOPS_PER_SM",  # fp4 routed via fp8 path
    }
    attr = attr_by_dtype.get(dtype, "PEAK_EAGER_FP32_TFLOPS_PER_SM")
    if leaf_path is not None:
        try:
            mod = __import__(leaf_path, fromlist=["*"])
            value = getattr(mod, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        except ImportError:
            pass

    fallback_table = {
        "fp32": _FALLBACK_EAGER_FP32_TFLOPS_PER_SM,
        "bf16": _FALLBACK_EAGER_BF16_TFLOPS_PER_SM,
        "fp16": _FALLBACK_EAGER_BF16_TFLOPS_PER_SM,
        "fp8": _FALLBACK_EAGER_FP8_TFLOPS_PER_SM,
        "fp4": _FALLBACK_EAGER_FP8_TFLOPS_PER_SM,
    }.get(dtype, _FALLBACK_EAGER_FP32_TFLOPS_PER_SM)
    default = 0.45  # conservative fp32 default
    return fallback_table.get(arch_key, default)


def _analytic_eager_flops(
    schedule_hints: dict[str, Any],
    *,
    tm: int,
    tn: int,
    tk: int,
    num_linear_ops: int,
) -> float:
    """Compute total eager-side GEMM FLOPs from the schedule's tile grid.

    Per bridge #124: the prior approach used a single ``per_task_gemm_flops
    × num_linear_tasks`` formula, which collapsed FFN's ``linear_up``
    (K = in_dim) and ``linear_down`` (K = hidden_dim) into a single
    ``k_per_task`` value — under-counting the linear with the larger K by
    the ratio of the two K dims. At MLP-1 that drove the predicted
    eager_us 4× over measured.

    This helper instead derives FLOPs per-op analytically from the
    schedule's tile grid:

    - **FFN**: ``tile_grid_up = [b_tiles, h_tiles]``,
      ``tile_grid_down = [b_tiles, o_tiles]`` plus their respective
      ``k_tiles_up`` / ``k_tiles_down``. Each linear contributes
      ``2 × M × N × K`` FLOPs where ``M = b_tiles × tm``,
      ``N = h_tiles × tn`` (or ``o_tiles × tn``),
      ``K = k_tiles × tk``.
    - **Diamond**: shared ``tile_grid`` across both linears, single
      ``k_tiles``. Sum is ``num_linear_ops × 2 × M × N × K``.
    - **No grid info**: returns ``0.0`` so callers can fall back to the
      legacy per-task method.

    Args:
        schedule_hints: ``decision["schedule_hints"]``.
        tm, tn, tk: tile shape from ``backend_choice``.
        num_linear_ops: linear op count from ``decision["body_decisions"]``.

    Returns: total FLOPs across all linear ops, or ``0.0`` when the
        schedule_hints don't carry a recognized tile grid.
    """
    if "tile_grid_up" in schedule_hints and "tile_grid_down" in schedule_hints:
        gu = schedule_hints["tile_grid_up"]
        gd = schedule_hints["tile_grid_down"]
        ku = int(schedule_hints.get("k_tiles_up", 0))
        kd = int(schedule_hints.get("k_tiles_down", 0))
        flops = 0.0
        if ku > 0:
            flops += 2.0 * (gu[0] * tm) * (gu[1] * tn) * (ku * tk)
        if kd > 0:
            flops += 2.0 * (gd[0] * tm) * (gd[1] * tn) * (kd * tk)
        return flops
    if "tile_grid" in schedule_hints:
        g = schedule_hints["tile_grid"]
        k = int(schedule_hints.get("k_tiles", 0))
        if k > 0:
            return num_linear_ops * 2.0 * (g[0] * tm) * (g[1] * tn) * (k * tk)
    return 0.0


def _lookup_cooperative_sync_us(arch_key: str) -> float:
    """Per-arch cooperative-grid sync cost (microseconds per wave).

    Cooperative-launch grid-sync between waves is the dominant ETC
    cost when tasks_per_sm > 1 (per bridge #118). At MLP-1 we
    measured 230 ms ETC vs 26 ms eager — the gap is dominated by
    this term, not per-task scheduling.
    """
    arch_to_leaf = {
        "100": "compgen.targets.gpu.nvidia.blackwell.cost",
        "120": "compgen.targets.gpu.nvidia.blackwell.cost",
        "90": "compgen.targets.gpu.nvidia.hopper.cost",
        "80": "compgen.targets.gpu.nvidia.ampere.cost",
        "86": "compgen.targets.gpu.nvidia.ampere.cost",
    }
    leaf_path = arch_to_leaf.get(arch_key)
    if leaf_path is not None:
        try:
            mod = __import__(leaf_path, fromlist=["*"])
            value = getattr(mod, "COOPERATIVE_SYNC_US_PER_WAVE", None)
            if isinstance(value, (int, float)):
                return float(value)
        except ImportError:
            pass
    return _FALLBACK_COOPERATIVE_SYNC_US_PER_WAVE.get(arch_key, 60.0)


@dataclass(frozen=True)
class EtcCostPrediction:
    """Compile-time prediction for ETC vs. eager.

    Components are exposed individually for the agent's audit query —
    when a bundle fails the gate, the agent can ask "what specifically
    is too slow?" and get a number for each step.

    Attributes:
        etc_us: Predicted total ETC dispatch time per forward.
            ``num_tasks * (per_task_gemm_us +
            scheduling_overhead_us)``.
        eager_us: Predicted total eager dispatch time. Single-launch
            overhead + roofline of the full-shape GEMM at the same
            arch's tensor-core throughput.
        speedup: ``eager_us / etc_us``. ETC wins when > 1.0; the
            perf gate fires at >= ``threshold`` (default 1.2).
        threshold: The pass gate.
        passes_gate: True iff ``speedup >= threshold``.
        components: Per-component breakdown for audit (per-task GEMM,
            scheduling cost, eager GEMM, eager launch).
        reason: Human-readable explanation. Goes into WontWinError's
            message + the bundle's verification_report.
    """

    etc_us: float
    eager_us: float
    speedup: float
    threshold: float
    passes_gate: bool
    components: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class WontWinError(RuntimeError):
    """The cost model predicts ETC won't clear the perf gate.

    Raised when the agent asked for a bundle that beats eager but
    the predicted speedup is below threshold. The :attr:`prediction`
    field carries the full :class:`EtcCostPrediction` for the audit
    query.
    """

    def __init__(self, prediction: EtcCostPrediction) -> None:
        super().__init__(
            f"ETC predicted {prediction.speedup:.2f}× vs eager "
            f"(threshold {prediction.threshold:.2f}×); " + (prediction.reason or "")
        )
        self.prediction = prediction


def predict_etc_dispatch(
    *,
    sample_input_shape: tuple[int, ...],
    decision: dict[str, Any],
    backend_choice: dict[str, Any],
    threshold: float = 1.2,
    model_dtype: str | None = None,
) -> EtcCostPrediction:
    """Predict whether the ETC bundle will beat eager by ``threshold``.

    Inputs are dicts (the JSON-serializable forms of LoweringDecision
    and BackendChoice) so the agent can feed them in from the
    bundle's compile_context.json without re-importing the matcher.

    Args:
        sample_input_shape: The forward's input shape — e.g.
            ``(64, 64)`` for a (B=64, in_dim=64) diamond.
        decision: ``LoweringDecision.to_dict()``. Used for: pattern
            name, body decisions (which ops are cuBLASDx vs fmaf),
            schedule hints (tile_grid, k_tiles, total tile-tasks).
        backend_choice: ``BackendChoice.to_dict()``. Used for:
            target_arch, tile_shape, use_cublasdx_for_linears,
            cublasdx_precision.
        threshold: Speedup gate. Default 1.2× per the conformance
            harness. Pass 1.0 to require non-regression only;
            higher values for more aggressive gates.

    Returns:
        :class:`EtcCostPrediction`. Caller decides whether to raise
        :class:`WontWinError` based on ``passes_gate``.
    """
    # Extract the canonical numbers we need.
    arch = backend_choice.get("target_arch", "sm_100")
    arch_key = arch.lower().lstrip("sm_").rstrip("a")
    tile_shape = backend_choice.get("tile_shape", [32, 32, 32])
    tm, tn, tk = tile_shape
    use_cublasdx = bool(backend_choice.get("use_cublasdx_for_linears", False))
    precision = backend_choice.get("cublasdx_precision", "fp32")

    schedule_hints = decision.get("schedule_hints", {})
    total_tile_tasks = decision.get("total_tile_tasks", 0)
    if total_tile_tasks == 0:
        # Older decisions didn't populate; derive from tile_grid.
        grid = schedule_hints.get("tile_grid", [1, 1])
        # Fallback estimate: 4 ops × tile_grid product.
        total_tile_tasks = max(1, grid[0] * grid[1] * 4)

    # Number of distinct kinds of ops (for the eager comparison: FFN
    # is 3 ops, diamond is 4, etc — cuBLAS issues one launch per
    # GEMM-ish op, plus pointwise fusions).
    body_decisions = decision.get("body_decisions", [])
    num_linear_ops = sum(1 for b in body_decisions if "linear" in b.get("op_name", "")) or 1

    # ETC per-task GEMM roofline. Uses _lookup_arch_tflops so the
    # per-arch leaves under targets/gpu/nvidia/{arch}/cost.py are
    # the source of truth (with local-table fallback per Wave 1.14c).
    use_tc = use_cublasdx and precision == "bf16_fp32"
    flops_per_sm_per_s = _lookup_arch_tflops(arch_key, tensor_core=use_tc) * 1e12

    # Per-task GEMM FLOPs: 2 * tm * tn * (k_per_task) where
    # k_per_task = full IN dim (we accumulate over K inside the
    # task). For diamond/FFN, IN is the inner contraction dim; we
    # lift it from schedule_hints if present, fall back to
    # sample_input_shape[1].
    k_per_task = (
        schedule_hints.get("k_tiles", 0) * tk
        or schedule_hints.get("k_tiles_up", 0) * tk
        or schedule_hints.get("k_tiles_down", 0) * tk
        or (sample_input_shape[1] if len(sample_input_shape) >= 2 else tk)
    )
    per_task_gemm_flops = 2 * tm * tn * k_per_task
    # One task occupies one SM, so the effective FLOPS the task
    # sees is one SM's peak — not the whole device's.
    per_task_gemm_us = per_task_gemm_flops / flops_per_sm_per_s * 1e6

    # ETC scheduling overhead per task.
    per_task_overhead_us = _SCHEDULING_OVERHEAD_US_PER_TASK

    # Pointwise tasks (relu, add, etc.) are very cheap individually
    # but each still pays the overhead. Tally them in.
    #
    # Per bridge #124: pool classification has to handle FFN's two
    # distinct linear ops (`linear_up` + `linear_down`) which carry
    # SEPARATE tile grids (``tile_grid_up`` / ``tile_grid_down``),
    # not a single shared ``tile_grid``. Diamond uses ``tile_grid``
    # for its two parallel linears (matching shape). When neither
    # variant is present we conservatively assume each linear op
    # contributes 1 tile-task (i.e. the rest are pointwise) — which
    # is wrong only at very large shapes, but the WontWinError gate
    # still fires correctly in that regime.
    if "tile_grid_up" in schedule_hints and "tile_grid_down" in schedule_hints:
        # FFN — sum tiles across both linears.
        gu = schedule_hints["tile_grid_up"]
        gd = schedule_hints["tile_grid_down"]
        num_linear_tile_tasks = (gu[0] * gu[1]) + (gd[0] * gd[1])
    elif "tile_grid" in schedule_hints:
        # Diamond — both linears share the same tile grid.
        g = schedule_hints["tile_grid"]
        num_linear_tile_tasks = num_linear_ops * (g[0] * g[1])
    else:
        # No grid info — fall back to the prior heuristic (1 tile-task
        # per linear op). Wrong at scale but keeps the predictor
        # honest about its uncertainty.
        num_linear_tile_tasks = num_linear_ops

    num_pointwise_tasks = max(0, total_tile_tasks - num_linear_tile_tasks)
    num_linear_tasks = total_tile_tasks - num_pointwise_tasks
    if num_linear_tasks <= 0:
        # total_tile_tasks doesn't decompose into the schedule's tile
        # grid. Likely a custom user-dialect bundle or a malformed
        # schedule; treat every task as linear-equivalent so we don't
        # over-credit the pointwise pool.
        num_linear_tasks = total_tile_tasks
        num_pointwise_tasks = 0

    etc_total_us = (
        num_linear_tasks * (per_task_gemm_us + per_task_overhead_us) + num_pointwise_tasks * per_task_overhead_us
    )

    # Eager: assume cuBLAS on the full GEMM shape. SM count from the
    # arch; eager rate is dtype-aware (per bridge #118 — using the
    # tensor-core peak for fp32 workloads was wrong by ~2700×).
    sm_count = {"100": 132, "120": 188, "90": 132, "89": 128, "86": 84, "80": 108}.get(arch_key, 80)
    # Pick the dtype eager actually runs at. Per bridge #121:
    # the prior code keyed off ``cublasdx_precision`` (compgen's
    # internal compute path), which on Blackwell defaults to
    # ``bf16_fp32`` even when the user's model is fp32. eager runs
    # the user's model directly — its rate depends on the model's
    # parameter dtype, not compgen's compute path. When the caller
    # passes ``model_dtype``, use it; otherwise fall back to
    # mapping ``cublasdx_precision`` for backward compat.
    if model_dtype is not None:
        eager_dtype = model_dtype
    else:
        # Map cuBLASDx-precision string → dtype label.
        # ``"bf16_fp32"`` = bf16 inputs, fp32 accumulator.
        # ``"fp32"`` = fp32 inputs → cuBLAS fp32 path (TF32 on
        # Blackwell/Hopper, SIMT on older).
        precision_to_dtype = {
            "fp32": "fp32",
            "bf16_fp32": "bf16",
            "fp16_fp32": "fp16",
            "fp8_fp32": "fp8",
            "fp4_fp32": "fp4",
        }
        eager_dtype = precision_to_dtype.get(precision, "fp32")
    eager_per_sm_flops_per_s = _lookup_eager_tflops(arch_key, dtype=eager_dtype) * 1e12
    # Eager processes the same total FLOPs as ETC's linear ops do.
    # Per bridge #124: derive per-op FLOPs analytically from the tile
    # grid (so FFN's two-different-K linears are counted correctly)
    # rather than ``per_task_gemm_flops × num_linear_tasks`` which
    # collapsed FFN's K dims into one value and under-counted the
    # larger-K linear by ~3× at MLP-1.
    #
    # The tile_shape that pairs with schedule_hints' tile_grid is the
    # one the matcher emitted INTO the schedule (e.g. 32×32×32 for the
    # fmaf path), NOT necessarily the same as backend_choice.tile_shape
    # (which is the cuBLASDx 64×64×16 when use_cublasdx=True). Read
    # the schedule's tile_shape when present so the analytic FLOPs
    # match the geometry the kernel actually generates.
    sched_tile = schedule_hints.get("tile_shape")
    if sched_tile and len(sched_tile) >= 3:
        atm, atn, atk = sched_tile[0], sched_tile[1], sched_tile[2]
    else:
        atm, atn, atk = tm, tn, tk
    eager_total_flops = _analytic_eager_flops(
        schedule_hints,
        tm=atm,
        tn=atn,
        tk=atk,
        num_linear_ops=num_linear_ops,
    )
    if eager_total_flops <= 0.0:
        # Fall back to the legacy per-task estimate when the schedule
        # doesn't carry recognized tile-grid info (e.g. user-dialect
        # bundles, mock schedules in tests).
        eager_total_flops = per_task_gemm_flops * num_linear_tasks
    eager_gemm_us = eager_total_flops / (eager_per_sm_flops_per_s * sm_count) * 1e6
    # cuBLAS issues one kernel launch per linear op — overhead scales
    # with ``num_linear_ops``, NOT with eager_gemm_us. Bug fix per
    # bridge #124: the prior code multiplied eager_gemm_us by
    # num_linear_ops (double-counting FLOPs already summed above).
    eager_total_us = eager_gemm_us + num_linear_ops * _EAGER_LAUNCH_OVERHEAD_US

    # Cooperative-grid sync between waves — the dominant ETC cost
    # when tasks-per-SM > 1 (bridge #118: 230ms ETC at MLP-1 was
    # 90% sync, 10% per-task scheduling). num_waves = ceil(tasks /
    # sm_count); each wave pays one cooperative_grid_sync.
    #
    # Per bridge #121 fix: pointwise tasks fan out across SMs the
    # same way linear tasks do; both pools contribute waves and
    # each wave pays the cooperative sync. Without summing both
    # pools, MLP-1's 57344 pointwise tasks land at num_waves=1 and
    # the predictor reports coop_share=0% — the wrong reason
    # string surfaces at exactly the shape where cluster-launch
    # would matter.
    num_linear_waves = max(1, (num_linear_tasks + sm_count - 1) // sm_count)
    num_pointwise_waves = max(0, (num_pointwise_tasks + sm_count - 1) // sm_count) if num_pointwise_tasks > 0 else 0
    num_waves = num_linear_waves + num_pointwise_waves
    cooperative_sync_us_per_wave = _lookup_cooperative_sync_us(arch_key)
    cooperative_sync_us = num_waves * cooperative_sync_us_per_wave
    etc_total_us = etc_total_us + cooperative_sync_us

    speedup = eager_total_us / etc_total_us if etc_total_us > 0 else 0.0
    passes = speedup >= threshold

    components = {
        "per_task_gemm_us": per_task_gemm_us,
        "per_task_overhead_us": per_task_overhead_us,
        "num_linear_tasks": num_linear_tasks,
        "num_pointwise_tasks": num_pointwise_tasks,
        "eager_gemm_us": eager_gemm_us,
        "eager_launch_overhead_us": _EAGER_LAUNCH_OVERHEAD_US,
        "sm_count": sm_count,
        "flops_per_sm_per_s": flops_per_sm_per_s,
        "eager_flops_per_sm_per_s": eager_per_sm_flops_per_s,
        "eager_dtype": eager_dtype,
        "k_per_task": k_per_task,
        "tile_shape": [tm, tn, tk],
        "use_cublasdx": use_cublasdx,
        "precision": precision,
        "arch": arch,
        # Cooperative-grid sync — bridge #118 added. The leading term
        # at MLP-1 once tasks_per_sm > 1. Per bridge #121, num_waves
        # is split between the linear pool (GEMM tile-tasks) and the
        # pointwise pool (relu / add). Each pool contributes its own
        # ceil(N_tasks / sm_count) waves; each wave pays one grid sync.
        "cooperative_sync_us": cooperative_sync_us,
        "cooperative_sync_us_per_wave": cooperative_sync_us_per_wave,
        "num_waves": num_waves,
        "num_linear_waves": num_linear_waves,
        "num_pointwise_waves": num_pointwise_waves,
        # Per bridge #127 nit: surface the per-pool tile-grid shapes
        # the predictor used. ``None`` when the matcher didn't emit
        # them (older diamond-only schedules pre-Wave-1.8).
        "tile_grid_up": (list(schedule_hints["tile_grid_up"]) if "tile_grid_up" in schedule_hints else None),
        "tile_grid_down": (list(schedule_hints["tile_grid_down"]) if "tile_grid_down" in schedule_hints else None),
        "tile_grid": (list(schedule_hints["tile_grid"]) if "tile_grid" in schedule_hints else None),
    }

    if passes:
        reason = (
            f"ETC predicted {etc_total_us:.1f}µs vs eager "
            f"{eager_total_us:.1f}µs ({speedup:.2f}× speedup, "
            f"≥ {threshold}× gate)"
        )
    else:
        reason = _format_loss_reason(
            etc_total_us=etc_total_us,
            eager_total_us=eager_total_us,
            speedup=speedup,
            components=components,
            threshold=threshold,
        )

    return EtcCostPrediction(
        etc_us=etc_total_us,
        eager_us=eager_total_us,
        speedup=speedup,
        threshold=threshold,
        passes_gate=passes,
        components=components,
        reason=reason,
    )


def _format_loss_reason(
    *,
    etc_total_us: float,
    eager_total_us: float,
    speedup: float,
    components: dict[str, Any],
    threshold: float,
) -> str:
    """Explain WHY ETC won't beat eager — the agent's audit query.

    Per bridge #099, the typical loss mode is "per-task scheduling
    dominates per-task GEMM." We surface that ratio so the agent
    can decide whether to abandon the bundle or chase a fix.
    """
    per_task_gemm = components["per_task_gemm_us"]
    per_task_overhead = components["per_task_overhead_us"]
    num_linear = components["num_linear_tasks"]
    num_pointwise = components.get("num_pointwise_tasks", 0)
    coop_sync = components.get("cooperative_sync_us", 0.0)
    num_waves = components.get("num_waves", 1)
    num_linear_waves = components.get("num_linear_waves", 0)
    num_pointwise_waves = components.get("num_pointwise_waves", 0)
    # Per bridge #121: overhead_share must include BOTH pools, not
    # just linear tasks. At MLP-1 the pointwise pool dwarfs linears
    # (57k vs 512), so excluding it makes overhead_share appear ~1%
    # and the predictor falls back to "larger tile" instead of
    # surfacing the real driver (per-task overhead × pointwise count).
    overhead_total_us = (num_linear + num_pointwise) * per_task_overhead
    overhead_share = overhead_total_us / etc_total_us if etc_total_us > 0 else 0.0
    coop_share = coop_sync / etc_total_us if etc_total_us > 0 else 0.0
    # Combined share — when sync + per-task overhead together dominate
    # but neither individually crosses 50%, the predictor would have
    # fallen through to the "larger tile" branch even though the real
    # driver is wave/task fanout. Surface that case cleanly.
    structural_share = overhead_share + coop_share

    lines = [
        f"ETC predicted {etc_total_us:.1f}µs vs eager {eager_total_us:.1f}µs ({speedup:.2f}× ; gate {threshold}×)",
        f"  per-task GEMM:        {per_task_gemm:.2f}µs",
        f"  per-task overhead:    {per_task_overhead:.2f}µs",
        f"  #linear tasks:        {num_linear}  ({num_linear_waves} waves)",
        f"  #pointwise tasks:     {num_pointwise}  ({num_pointwise_waves} waves)",
        f"  cooperative sync:     {coop_sync:.1f}µs ({num_waves} waves)",
        f"  overhead share:       {overhead_share:.0%} of ETC total",
        f"  coop-sync share:      {coop_share:.0%} of ETC total",
    ]
    # Cooperative-grid sync between waves dominates once tasks-per-SM
    # > 1 (bridge #118 disconfirmed Wave 1.2's row-strip hypothesis;
    # the lever is cluster-cooperative bodies, not fewer tasks).
    if coop_share > 0.5:
        lines.append(
            "  → cooperative-grid sync between waves dominates. "
            "Cluster-cooperative bodies (cute::cluster_sync, Wave "
            "1.6b) replace per-wave grid sync with intra-cluster "
            "sync — typically ~10× cheaper. Reduce wave count "
            "before reducing per-task overhead."
        )
    elif overhead_share > 0.5 and num_pointwise > num_linear * 4:
        # Pointwise-overhead dominant. Real lever is fusing the
        # pointwise op into the GEMM epilogue (so num_pointwise_tasks
        # collapses to 0) AND/OR cluster-launch (cheaper sync per wave).
        lines.append(
            "  → pointwise-task overhead dominates "
            f"({num_pointwise} pointwise tasks vs {num_linear} linear). "
            "Fuse the pointwise op into the GEMM epilogue (relu/add "
            "post-MMA) so the pointwise pool collapses, then "
            "cluster-cooperative bodies (Wave 1.6b) cut the "
            "remaining wave-sync cost."
        )
    elif overhead_share > 0.5:
        lines.append(
            "  → per-task scheduling dominates. Larger tile (so "
            "fewer tasks) or cluster-launch (so fewer waves) "
            "amortizes the per-task fixed cost."
        )
    elif structural_share > 0.5:
        # Sync + overhead together dominate but neither alone is 50%
        # — the workload is wave/sync-bound, not GEMM-bound.
        lines.append(
            "  → sync + scheduling overhead together dominate "
            f"({structural_share:.0%}). Cluster-cooperative bodies "
            "(Wave 1.6b) attack both: cheaper per-wave sync AND "
            "fewer waves needed for the same fan-out."
        )
    elif per_task_gemm < per_task_overhead:
        lines.append(
            "  → per-task GEMM smaller than scheduling cost. "
            "Larger tile shape pushes per-task GEMM above the "
            "scheduling-cost floor."
        )
    elif not components.get("use_cublasdx"):
        lines.append("  → fmaf path; bf16+fp32-acc cuBLASDx engages tensor cores per #095. Check probe_device output.")
    else:
        lines.append(
            "  → ETC's per-task GEMM amortizes scheduling, but eager's "
            "device-wide parallelism still wins at this shape."
        )
    return "\n".join(lines)
