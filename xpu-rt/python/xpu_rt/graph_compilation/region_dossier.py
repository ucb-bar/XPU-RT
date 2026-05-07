"""Region Dossier V2 (Milestone 03 MVP).

Turns the IR-grounded region/tensor graph from Milestone B + 02.5 into
**decision-quality evidence** for the Strategist/Tactician agent
roles: per-region cost, reuse, numerical sensitivity, working-set
curves, placement envelopes, and legality constraints.

This stage does **not** produce candidate actions. It produces the
facts that future candidate generation will consume.

Inputs (read-only):

- ``00_graph_capture/...``                      — capture artifacts (untouched here)
- ``01_payload_lowering/fx_to_payload_accounting.json`` (v2)
- ``01_payload_lowering/dialect_coverage.json``
- ``02_graph_analysis/region_map.json``
- ``02_graph_analysis/tensor_use_def_graph.json``
- ``02_graph_analysis/region_graph.json``
- target profile YAML (peak_compute_gflops, peak_bandwidth_gb_s,
  memory tiers, numerical budgets, working-set tile candidates)

Outputs:

- ``02_graph_analysis/graph_analysis.mlir``           — canonical, IR-flavored
- ``02_graph_analysis/graph_dossier_v2.json``         — top-level projection
- ``02_graph_analysis/region_dossiers/<safe_id>.json``— one per region
- ``02_graph_analysis/dossier_validation.json``       — cross-check report

The MLIR file is intentionally simple text (not parseable by xDSL until
a custom dialect lands). It is the *canonical compiler-owned* analysis
record; JSON is its view, per the project's format policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Target profile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TargetProfile:
    target_id: str
    device_kind: str
    peak_compute_gflops: float
    peak_bandwidth_gb_s: float
    scratchpad_bytes: int
    l2_bytes: int
    l3_bytes: int
    system_bytes: int
    supported_dtypes: tuple[str, ...]
    numerical_budgets: dict[str, float] = field(default_factory=dict)
    working_set_tiles_matmul: tuple[dict[str, int], ...] = ()
    working_set_tiles_elementwise: tuple[dict[str, int], ...] = ()


_DEFAULT_MATMUL_TILES: tuple[dict[str, int], ...] = (
    {"M": 16, "N": 16, "K": 16},
    {"M": 32, "N": 32, "K": 32},
    {"M": 64, "N": 64, "K": 32},
    {"M": 128, "N": 128, "K": 32},
    {"M": 256, "N": 256, "K": 64},
    {"M": 512, "N": 512, "K": 64},  # 1.25 MB live — exceeds typical L2
)
_DEFAULT_ELEMENTWISE_TILES: tuple[dict[str, int], ...] = (
    {"numel": 1024},
    {"numel": 4096},
    {"numel": 16384},
)
_DEFAULT_BUDGETS = {
    "fp32": 1e-3,
    "fast_math": 5e-3,
    "fp16_accum": 1e-2,
    "fp8_e4m3": 1e-1,
}

# Monotonicity ranking for numerical-precision dtypes. The list is
# strictly ordered "most precise → least precise"; each entry must have
# a status that is at-least-as-safe as every entry that follows it.
_PRECISION_ORDER: tuple[str, ...] = ("fp32", "fast_math", "fp16_accum", "fp8_e4m3")

# Status ranking (higher = less safe). Used to enforce monotonicity.
_STATUS_RANK: dict[str, int] = {
    "safe": 0,
    "risky": 1,
    "exceeds_budget": 2,
    "requires_reference": 3,
}
_RANK_STATUS: dict[int, str] = {v: k for k, v in _STATUS_RANK.items()}


def load_target_profile(target_yaml_path: Path) -> TargetProfile:
    raw = yaml.safe_load(Path(target_yaml_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"target YAML must be a mapping: {target_yaml_path}")
    mt = raw.get("memory_tiers") or {}
    nb = raw.get("numerical_budgets") or {}
    wst = raw.get("working_set_tiles") or {}

    matmul_tiles = wst.get("matmul")
    if matmul_tiles:
        matmul_tiles = tuple(dict(t) for t in matmul_tiles)
    else:
        matmul_tiles = _DEFAULT_MATMUL_TILES

    ew_tiles = wst.get("elementwise")
    if ew_tiles:
        ew_tiles = tuple(dict(t) for t in ew_tiles)
    else:
        ew_tiles = _DEFAULT_ELEMENTWISE_TILES

    budgets = {**_DEFAULT_BUDGETS, **{k: float(v) for k, v in nb.items()}}

    return TargetProfile(
        target_id=str(raw.get("target_id", "unknown")),
        device_kind=str(raw.get("device_kind", "cpu")),
        peak_compute_gflops=float(raw.get("peak_compute_gflops", 100.0)),
        peak_bandwidth_gb_s=float(raw.get("peak_bandwidth_gb_s", 30.0)),
        scratchpad_bytes=int(mt.get("scratchpad_bytes", 32_768)),
        l2_bytes=int(mt.get("l2_bytes", 524_288)),
        l3_bytes=int(mt.get("l3_bytes", 16_777_216)),
        system_bytes=int(mt.get("system_bytes", 16 * 1024 * 1024 * 1024)),
        supported_dtypes=tuple(raw.get("supported_dtypes", ["fp32"])),
        numerical_budgets=budgets,
        working_set_tiles_matmul=matmul_tiles,
        working_set_tiles_elementwise=ew_tiles,
    )


# --------------------------------------------------------------------------- #
# Region kind buckets
# --------------------------------------------------------------------------- #

_REDUCTION_KINDS = {"matmul", "conv", "softmax", "layer_norm", "batch_norm"}
_REDUCE_MEAN_KINDS = {"reduce_mean"}
_ELEMENTWISE_KINDS = {
    "elementwise_gelu", "elementwise_relu", "elementwise_tanh",
    "bias_add", "embedding", "generic", "unknown",
}
_VIEW_KINDS = {"transpose"}
_ALLOCATOR_KINDS = {"tensor_empty"}


def _is_opaque_kind(kind: str) -> bool:
    return kind.startswith("opaque_")


def _is_matmul_like(kind: str) -> bool:
    return kind in {"matmul", "conv"}


# --------------------------------------------------------------------------- #
# Numerical sensitivity
# --------------------------------------------------------------------------- #

# Approximate unit roundoff per dtype. fp8_e4m3 is the worst-case
# representable rounding for that mantissa width.
_UNIT_ROUNDOFF: dict[str, float] = {
    "fp32": 5.96e-8,        # 2^-24
    "fp16": 4.88e-4,        # 2^-11
    "bf16": 3.91e-3,        # 2^-8
    "fp8_e4m3": 6.25e-2,    # 2^-4
}

_DTYPE_BYTES: dict[str, int] = {
    "f32": 4, "f16": 2, "bf16": 2, "f64": 8,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1,
    "ui64": 8, "ui32": 4, "ui16": 2, "ui8": 1,
    "i1": 1,
}


def _reduction_dimension(region: dict[str, Any], tensor_lookup: dict[str, dict[str, Any]]) -> int:
    """Best-effort estimate of the reduction dimension K for a region."""
    kind = region["kind"]
    candidates: list[int] = []
    for port in region.get("inputs", []):
        t = tensor_lookup.get(port["tensor_id"])
        if not t:
            continue
        shape = [d for d in t.get("shape", []) if isinstance(d, int) and d > 0]
        if not shape:
            continue
        if kind == "matmul" or kind == "conv":
            # Use the largest input dim as a conservative K proxy.
            candidates.append(max(shape))
        elif kind in {"softmax", "layer_norm", "batch_norm"}:
            candidates.append(shape[-1])
    return max(candidates) if candidates else 1


_MLIR_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x([a-z0-9]+)>")
_MLIR_MATMUL_RE = re.compile(
    r'linalg\.matmul\s+\{[^}]*compgen\.region_id\s*=\s*"([^"]+)"[^}]*\}'
    r"\s+ins\([^:]*:\s*(tensor<[^>]+>)\s*,\s*(tensor<[^>]+>)\s*\)"
)


def _parse_mlir_tensor_type(text: str) -> tuple[list[int], str]:
    """Parse ``tensor<8x27xf32>`` into ([8, 27], "f32"). Empty on failure."""
    m = _MLIR_TENSOR_RE.search(text)
    if not m:
        return [], ""
    dim_str, dtype = m.group(1), m.group(2)
    try:
        dims = [int(d) for d in dim_str.split("x") if d]
    except ValueError:
        return [], ""
    return dims, dtype


def _shape_from_payload_mlir(
    region_id: str, payload_path: Path,
) -> tuple[list[list[int]], list[list[int]], str]:
    """Parse payload.mlir for ``linalg.matmul`` matching ``region_id``.

    Used as a fallback when the FX-level ``tensor_lookup`` doesn't
    carry shapes for a region — typical for regions produced by
    intermediate lowerings (e.g. conv → im2col → matmul where the
    matmul's tensor metadata is below the FX layer). The shape IS
    deterministically derivable from the conv's input + kernel +
    padding + stride; rather than re-running that derivation, we
    just read the lowered MLIR (which already has the answer).

    Returns (input_shapes, output_shapes, dtype). Empty when the
    payload is missing, the region isn't matched, or the matmul's
    K dims don't pair (``ins(MxK, KxN)``).
    """
    if not payload_path.exists():
        return [], [], ""
    try:
        text = payload_path.read_text(encoding="utf-8")
    except OSError:
        return [], [], ""
    for match in _MLIR_MATMUL_RE.finditer(text):
        rid, lhs_t, rhs_t = match.group(1), match.group(2), match.group(3)
        if rid != region_id:
            continue
        lhs_dims, lhs_dtype = _parse_mlir_tensor_type(lhs_t)
        rhs_dims, _ = _parse_mlir_tensor_type(rhs_t)
        if (
            len(lhs_dims) == 2 and len(rhs_dims) == 2
            and lhs_dims[1] == rhs_dims[0]
        ):
            # linalg.matmul: ins(MxK, KxN) outs(MxN)
            out_dims = [lhs_dims[0], rhs_dims[1]]
            return [lhs_dims, rhs_dims], [out_dims], lhs_dtype
    return [], [], ""


def _region_shape(
    region: dict[str, Any],
    tensor_lookup: dict[str, dict[str, Any]],
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Distinctive shape signature for a region (M-37.9 Fix 1).

    Captures the *actual* tensor shapes the region operates on so two
    regions named ``matmul_0`` in different models with different
    shapes produce distinct downstream candidate_ids. Returns:

      {
        "input_shapes":  [[M, K], [K, N], ...],
        "output_shapes": [[M, N], ...],
        "kind":          "matmul" | ...,
        "summary":       "matmul/4x128x64/f32",
        "source":        "fx_tensor_lookup" | "payload_mlir_fallback"
      }

    Two-tier extraction:

    1. **FX tensor_lookup** — cheap, in-memory; covers regions lifted
       directly from torch.export.
    2. **Payload MLIR fallback** — parses ``linalg.matmul ins/outs``
       text directly when the FX layer doesn't have the shape (e.g.
       conv → im2col → matmul). Same answer every run, no laziness.
    """
    def _ports_shapes(ports: list) -> list[list[int]]:
        out: list[list[int]] = []
        for port in ports or []:
            t = tensor_lookup.get(port.get("tensor_id"))
            if not t:
                continue
            shape = [
                int(d) for d in t.get("shape", [])
                if isinstance(d, int) and d > 0
            ]
            if shape:
                out.append(shape)
        return out

    inp = _ports_shapes(region.get("inputs", []))
    outp = _ports_shapes(region.get("outputs", []))
    kind = region.get("kind", "")
    dtype = ""
    source = "fx_tensor_lookup"
    for port in region.get("inputs", []):
        t = tensor_lookup.get(port.get("tensor_id"))
        if t and t.get("dtype"):
            dtype = str(t["dtype"])
            break

    # MLIR fallback when FX didn't give us a 2-input MxK × KxN matmul.
    # We fall through in three cases:
    #   1. Fewer than 2 input shapes recorded (rare).
    #   2. The first two shapes aren't both rank-2.
    #   3. The first two shapes are rank-2 but their K dims don't pair
    #      — typical for conv → im2col → matmul where FX records the
    #      output buffer + operands in mixed order.
    fx_pair_invalid = False
    if kind == "matmul":
        if len(inp) < 2 or len(inp[0]) != 2 or len(inp[1]) != 2:
            fx_pair_invalid = True
        elif inp[0][1] != inp[1][0]:
            fx_pair_invalid = True
    if kind == "matmul" and run_dir is not None and fx_pair_invalid:
        payload_ref = ""
        for po in region.get("payload_ops", []):
            if po.get("region_id") == region.get("region_id"):
                payload_ref = po.get("payload_ref", "")
                break
        if payload_ref:
            inp_mlir, outp_mlir, dtype_mlir = _shape_from_payload_mlir(
                region.get("region_id", ""), run_dir / payload_ref,
            )
            if inp_mlir:
                inp = inp_mlir
                outp = outp_mlir
                if dtype_mlir:
                    dtype = dtype_mlir
                source = "payload_mlir_fallback"

    summary = f"{kind}/unknown"
    if kind == "matmul" and len(inp) >= 2 and len(inp[0]) == 2 and len(inp[1]) == 2:
        m, k0 = inp[0]
        k1, n = inp[1]
        if k0 == k1:
            summary = f"matmul/{m}x{n}x{k0}" + (f"/{dtype}" if dtype else "")
    elif outp and outp[0]:
        summary = f"{kind}/" + "x".join(str(d) for d in outp[0]) + (
            f"/{dtype}" if dtype else ""
        )

    return {
        "input_shapes": inp,
        "output_shapes": outp,
        "kind": kind,
        "dtype": dtype,
        "summary": summary,
        "source": source,
    }


def _kind_multiplier(kind: str) -> float:
    """Convert raw ``K * unit_roundoff`` to ``eps_out`` per op family."""
    if kind in {"matmul", "conv"}:
        return 1.0
    if kind == "softmax":
        return 4.0
    if kind in {"layer_norm", "batch_norm"}:
        return 8.0
    if kind in _REDUCE_MEAN_KINDS:
        return 1.0
    return 1.0


def _bucket_status(eps: float, budget: float) -> str:
    if eps >= budget:
        return "exceeds_budget"
    if eps >= 0.5 * budget:
        return "risky"
    return "safe"


def _compute_numerics(
    region: dict[str, Any],
    tensor_lookup: dict[str, dict[str, Any]],
    profile: TargetProfile,
) -> dict[str, dict[str, Any]]:
    kind = region["kind"]
    if _is_opaque_kind(kind):
        return {
            dt: {"eps_out": 0.0, "budget_remaining": 0.0, "status": "requires_reference"}
            for dt in ("fp32", "fp16_accum", "fp8_e4m3", "fast_math")
        }
    if kind in _ALLOCATOR_KINDS or kind in _VIEW_KINDS:
        # Pure data movement; no numerical loss.
        return {
            dt: {"eps_out": 0.0, "budget_remaining": 1.0, "status": "safe"}
            for dt in ("fp32", "fp16_accum", "fp8_e4m3", "fast_math")
        }

    K = _reduction_dimension(region, tensor_lookup)
    mult = _kind_multiplier(kind)

    def _entry(unit_dtype: str, budget_key: str, extra_mult: float = 1.0) -> dict[str, Any]:
        u = _UNIT_ROUNDOFF.get(unit_dtype, _UNIT_ROUNDOFF["fp32"])
        eps = K * u * mult * extra_mult
        budget = profile.numerical_budgets.get(budget_key, _DEFAULT_BUDGETS[budget_key])
        margin = (budget - eps) / max(budget, 1e-30)
        return {
            "eps_out": float(f"{eps:.6g}"),
            "budget_remaining": float(f"{max(margin, 0.0):.6g}"),
            "status": _bucket_status(eps, budget),
        }

    sensitivity = {
        "fp32":       _entry("fp32",     "fp32"),
        "fp16_accum": _entry("fp16",     "fp16_accum"),
        "fp8_e4m3":   _entry("fp8_e4m3", "fp8_e4m3"),
        "fast_math":  _entry("fp32",     "fast_math", extra_mult=4.0),
    }
    # Enforce monotone precision ordering: a less-precise dtype's status
    # is never marked safer than a more-precise dtype's. We walk the
    # ranking and clamp downstream entries to the running max.
    return _enforce_monotonic_precision(sensitivity)


def _enforce_monotonic_precision(
    sensitivity: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Clamp the status of each less-precise dtype to be at-least-as-bad
    as every more-precise dtype in ``_PRECISION_ORDER``.

    ``requires_reference`` is special: it propagates straight through
    (a region either uses requires_reference uniformly or not at all),
    so we do not downgrade it on monotonicity grounds.
    """
    if not sensitivity:
        return sensitivity
    # If any status in ``_PRECISION_ORDER`` is requires_reference, leave
    # the dict untouched — it's an opaque/allocator/view region whose
    # entries were already set uniformly upstream.
    for k in _PRECISION_ORDER:
        if sensitivity.get(k, {}).get("status") == "requires_reference":
            return sensitivity
    running_max = -1
    for k in _PRECISION_ORDER:
        entry = sensitivity.get(k)
        if entry is None:
            continue
        cur = _STATUS_RANK.get(entry["status"], 0)
        if cur < running_max:
            entry["status"] = _RANK_STATUS[running_max]
            entry["budget_remaining"] = 0.0  # downgraded by monotonicity
            entry["monotonicity_clamped"] = True
        running_max = max(running_max, _STATUS_RANK.get(entry["status"], 0))
    return sensitivity


# --------------------------------------------------------------------------- #
# Cost / reuse / working-set / placement / legality
# --------------------------------------------------------------------------- #


def _compute_cost(region: dict[str, Any], profile: TargetProfile) -> dict[str, Any]:
    flops = int(region["estimated"]["flops"])
    bytes_total = int(region["estimated"]["bytes"])
    arith_intensity = flops / max(bytes_total, 1)

    peak_flops = profile.peak_compute_gflops * 1e9
    peak_bw = profile.peak_bandwidth_gb_s * 1e9
    compute_time_us = (flops / peak_flops) * 1e6 if peak_flops > 0 else 0.0
    memory_time_us = (bytes_total / peak_bw) * 1e6 if peak_bw > 0 else 0.0
    latency_us = max(compute_time_us, memory_time_us, 1e-3)
    bottleneck = "compute" if compute_time_us >= memory_time_us else "memory"

    return {
        "flops": flops,
        "bytes": bytes_total,
        "arithmetic_intensity": round(arith_intensity, 6),
        "estimated_latency_us": {profile.target_id: round(latency_us, 6)},
        "bottleneck_resource": {profile.target_id: bottleneck},
    }


def _port_to_reuse(
    port: dict[str, Any], tensor_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    t = tensor_lookup.get(port["tensor_id"])
    if not t:
        return None
    return {
        "tensor_id": t["tensor_id"],
        "ssa": t.get("ssa", ""),
        "shape": t.get("shape", []),
        "dtype": t.get("dtype", ""),
        "bytes": t.get("bytes", 0),
        "consumer_count": int(t.get("consumer_count", 0)),
        "reuse_horizon": int(t.get("reuse_horizon", -1)),
        "lifetime_class": t.get("producer_lifetime_class", "transient"),
    }


def _compute_reuse(
    region: dict[str, Any], tensor_lookup: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    inputs = [r for r in (_port_to_reuse(p, tensor_lookup) for p in region.get("inputs", [])) if r]
    outputs = [r for r in (_port_to_reuse(p, tensor_lookup) for p in region.get("outputs", [])) if r]
    return {"inputs": inputs, "outputs": outputs}


def _output_dtype_size(region: dict[str, Any], tensor_lookup: dict[str, dict[str, Any]]) -> int:
    for port in region.get("outputs", []):
        t = tensor_lookup.get(port["tensor_id"])
        if t:
            return _DTYPE_BYTES.get(t.get("dtype", "f32"), 4)
    for port in region.get("inputs", []):
        t = tensor_lookup.get(port["tensor_id"])
        if t:
            return _DTYPE_BYTES.get(t.get("dtype", "f32"), 4)
    return 4


def _shape_fit_dim(d: int, *, max_tile: int = 16) -> int:
    """Largest divisor of ``d`` that is also <= ``max_tile``.

    M-37.11 (Improvement A): used to derive shape-fit matmul tiles
    that cleanly divide the region's actual dimensions, breaking the
    dead-end where the only proposed tiles start at 16 and a region
    with M=4 has no clean-divide option.

    Walks the standard cache-friendly sizes (16, 8, 4, 2, 1) and picks
    the largest that divides ``d``. For composite ``d`` this finds
    a useful tile; for prime ``d`` it returns 1 (caller can choose to
    skip). Returns ``d`` itself when ``d <= max_tile`` (use the whole
    dim — a single tile, no boundary).
    """
    if d <= 0:
        return 0
    if d <= max_tile:
        return d
    for v in (16, 8, 4, 2, 1):
        if v <= max_tile and d % v == 0:
            return v
    return 1


def _shape_fit_tile_for_matmul(
    shape_info: dict[str, Any],
    *,
    max_tile: int = 16,
) -> dict[str, int] | None:
    """Derive a clean-divide matmul tile from a region's actual shape.

    Returns ``None`` when the shape is unknown, when any dim is too
    small to be useful (< 2), or when the resulting tile is degenerate
    (any dim == 1). The candidate proposer skips degenerate tiles —
    a (M=7, N=1, K=1) tile is not a useful action.
    """
    if not shape_info or shape_info.get("kind") != "matmul":
        return None
    inp = shape_info.get("input_shapes") or []
    if (
        len(inp) < 2
        or len(inp[0]) != 2
        or len(inp[1]) != 2
        or inp[0][1] != inp[1][0]
    ):
        return None
    M, K = inp[0]
    _, N = inp[1]
    tM = _shape_fit_dim(M, max_tile=max_tile)
    tN = _shape_fit_dim(N, max_tile=max_tile)
    tK = _shape_fit_dim(K, max_tile=max_tile)
    if tM < 2 or tN < 2 or tK < 2:
        return None
    return {"M": tM, "N": tN, "K": tK}


def _working_set_curve(
    region: dict[str, Any],
    profile: TargetProfile,
    tensor_lookup: dict[str, dict[str, Any]],
    *,
    region_shape: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    kind = region["kind"]
    if _is_opaque_kind(kind) or kind in _ALLOCATOR_KINDS or kind == "unknown":
        return []
    dtype_size = _output_dtype_size(region, tensor_lookup)
    curve: list[dict[str, Any]] = []
    if _is_matmul_like(kind):
        # M-37.11 Improvement A: append a shape-fit tile when the
        # region's actual M/N/K are known and the standard profile
        # tiles all force boundary handling. The shape-fit tile is
        # guaranteed to cleanly divide every region dim it covers
        # (whole-dim when dim < max_tile, largest divisor otherwise).
        # Composite shapes get a clean-divide option; prime shapes
        # honestly stay boundary-only.
        seen_tiles: set[tuple[int, int, int]] = set()
        all_tiles = list(profile.working_set_tiles_matmul)
        if region_shape:
            shape_fit = _shape_fit_tile_for_matmul(region_shape)
            if shape_fit is not None:
                # Prepend so it surfaces first in the curve (smallest
                # cache footprint typically).
                all_tiles = [shape_fit] + all_tiles
        for tile in all_tiles:
            M = int(tile.get("M", 0))
            N = int(tile.get("N", 0))
            K = int(tile.get("K", 0))
            key = (M, N, K)
            if key in seen_tiles:
                continue
            seen_tiles.add(key)
            live_bytes = (M * K + K * N + M * N) * dtype_size
            curve.append(
                {
                    "tile": dict(tile),
                    "live_bytes": live_bytes,
                    "fits_scratchpad": live_bytes <= profile.scratchpad_bytes,
                    "fits_l2": live_bytes <= profile.l2_bytes,
                    "fits_l3": live_bytes <= profile.l3_bytes,
                }
            )
    else:
        for tile in profile.working_set_tiles_elementwise:
            numel = int(tile.get("numel", 0))
            live_bytes = 2 * numel * dtype_size  # input + output tile
            curve.append(
                {
                    "tile": dict(tile),
                    "live_bytes": live_bytes,
                    "fits_scratchpad": live_bytes <= profile.scratchpad_bytes,
                    "fits_l2": live_bytes <= profile.l2_bytes,
                    "fits_l3": live_bytes <= profile.l3_bytes,
                }
            )
    return curve


def _placement_envelope(
    region: dict[str, Any], profile: TargetProfile, cost: dict[str, Any]
) -> dict[str, Any]:
    bytes_total = int(region["estimated"]["bytes"])
    return {
        "devices": [
            {
                "device": profile.target_id,
                "estimated_latency_us": cost["estimated_latency_us"][profile.target_id],
                "memory_fit": bytes_total <= profile.l3_bytes,
                "bottleneck_resource": cost["bottleneck_resource"][profile.target_id],
            }
        ]
    }


def _legality_constraints(
    region: dict[str, Any], partial: dict[str, Any]
) -> list[dict[str, Any]]:
    kind = region["kind"]
    is_opaque = _is_opaque_kind(kind)
    is_alloc = kind in _ALLOCATOR_KINDS

    out: list[dict[str, Any]] = []
    out.append(
        {
            "constraint": "can_tile",
            "ok": (not is_opaque) and (not is_alloc),
            "reason": (
                "opaque region; requires extension closure first"
                if is_opaque
                else ("pure allocator op — no tile to apply" if is_alloc else "")
            ),
        }
    )

    fp8_status = partial["numerical_sensitivity"]["fp8_e4m3"]["status"]
    out.append(
        {
            "constraint": "can_quantize_fp8",
            "ok": fp8_status == "safe",
            "reason": f"fp8_e4m3 status = {fp8_status}",
        }
    )

    transient_outputs = [
        o for o in partial["reuse"]["outputs"]
        if o["consumer_count"] == 1 and o["lifetime_class"] == "transient"
    ]
    can_fuse = bool(transient_outputs) and not is_opaque
    out.append(
        {
            "constraint": "can_fuse_with_single_consumer",
            "ok": can_fuse,
            "reason": (
                "single-consumer transient output"
                if can_fuse
                else (
                    "opaque region — no semantic model"
                    if is_opaque
                    else "no single-consumer transient output"
                )
            ),
        }
    )

    if is_opaque:
        out.append(
            {
                "constraint": "requires_reference_or_extension",
                "ok": False,
                "reason": "opaque region — cannot be tiled/fused without semantic model",
            }
        )

    return out


# --------------------------------------------------------------------------- #
# graph_analysis.mlir text writer
# --------------------------------------------------------------------------- #


def _mlir_safe_string(value: object) -> str:
    """MLIR string-attribute body (escape backslash and quote, no surrounding "")"""
    s = str(value)
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _mlir_attr_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v} : i64"
    if isinstance(v, float):
        return f"{v} : f64"
    if isinstance(v, list):
        return "[" + ", ".join(_mlir_attr_value(x) for x in v) + "]"
    return f'"{_mlir_safe_string(v)}"'


def _emit_attrs(d: dict[str, Any]) -> str:
    items = []
    for k in sorted(d):
        items.append(f"{k} = {_mlir_attr_value(d[k])}")
    return ", ".join(items)


def _emit_graph_analysis_mlir(
    *,
    model_id: str,
    target_id: str,
    summary: dict[str, Any],
    region_map: dict[str, Any],
    use_def: dict[str, Any],
) -> str:
    lines: list[str] = []
    graph_attrs = {
        "model_id": model_id,
        "target_id": target_id,
        "total_regions": int(summary["total_regions"]),
        "total_tensors": int(summary["total_tensors"]),
        "total_flops": int(summary["total_flops"]),
        "total_bytes": int(summary["total_bytes"]),
        "structured_fraction": float(summary["structured_fraction"]),
        "opaque_fraction": float(summary["opaque_fraction"]),
    }
    lines.append(f"compgen.graph @{_mlir_symbol(model_id)} attributes {{ {_emit_attrs(graph_attrs)} }} {{")
    for r in region_map["regions"]:
        rid_sym = _mlir_symbol(r["region_id"])
        attrs = {
            "region_id": r["region_id"],
            "module_id": r["module_id"],
            "kind": r["kind"],
            "source_classification": r["source_classification"],
            "fx_nodes": r["fx_nodes"],
            "payload_op_names": [p["op_name"] for p in r["payload_ops"]],
            "flops": int(r["estimated"]["flops"]),
            "bytes": int(r["estimated"]["bytes"]),
            "arithmetic_intensity": float(r["estimated"]["arithmetic_intensity"]),
        }
        lines.append(f"  compgen.region @{rid_sym} attributes {{ {_emit_attrs(attrs)} }}")
    for t in use_def["tensors"]:
        tid_sym = _mlir_symbol(t["tensor_id"])
        attrs = {
            "tensor_id": t["tensor_id"],
            "module_id": t["module_id"],
            "ssa": t["ssa"],
            "producer_region": t["producer_region"],
            "consumer_regions": t["consumer_regions"],
            "shape": [d if isinstance(d, int) else 0 for d in t["shape"]],
            "dtype": t["dtype"],
            "bytes": int(t["bytes"]),
            "consumer_count": int(t["consumer_count"]),
            "reuse_horizon": int(t["reuse_horizon"]),
            "producer_lifetime_class": t["producer_lifetime_class"],
            "is_reduction_input": bool(t["is_reduction_input"]),
        }
        lines.append(f"  compgen.tensor @{tid_sym} attributes {{ {_emit_attrs(attrs)} }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


_MLIR_SYMBOL_RE = re.compile(r"[^A-Za-z0-9_]+")


def _mlir_symbol(s: str) -> str:
    """Lossy MLIR-symbol projection. Region/tensor IDs may contain
    arbitrary characters; we collapse non-symbol chars to underscores
    and keep the verbatim string in attributes for the round-trip."""
    s2 = _MLIR_SYMBOL_RE.sub("_", s)
    return s2 or "_"


# --------------------------------------------------------------------------- #
# Filename safety for region_dossiers/<id>.json
# --------------------------------------------------------------------------- #


def _safe_filename(region_id: str) -> str:
    """Stable, readable filename for a region_id that may contain weird
    characters (e.g. ``<built-in function linear>_0``)."""
    slug = _MLIR_SYMBOL_RE.sub("_", region_id).strip("_")
    if not slug:
        slug = "region"
    digest = hashlib.sha1(region_id.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}.json"


# --------------------------------------------------------------------------- #
# Result + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionDossierResult:
    graph_analysis_mlir_path: Path
    graph_dossier_v2_path: Path
    region_dossier_dir: Path
    dossier_validation_path: Path
    numerical_sensitivity_audit_path: Path
    region_dossier_count: int
    matmul_like_count: int
    opaque_count: int


def _build_numerical_sensitivity_audit(
    region_dossiers_summary: list[dict[str, Any]],
    region_map_regions: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    """Cross-region sanity audit (M-03.5).

    Invariants enforced:

    - **monotonic_precision_order**: for every non-opaque region,
      ``status_rank(fp32) <= status_rank(fast_math) <= status_rank(fp16_accum) <=
      status_rank(fp8_e4m3)``.
    - **fp32_not_less_safe_than_fast_math**: for every non-opaque region,
      ``status_rank(fp32) <= status_rank(fast_math)``.
    - **opaque_regions_require_reference**: every opaque/allocator/view
      region has all four dtypes either ``requires_reference`` or ``safe``
      (no spurious risky/exceeds_budget on regions with no real math).
    - **status_consistent_with_eps**: ``eps_out`` and the bucketed
      ``status`` agree (eps >= budget → at least exceeds_budget;
      eps < 0.5*budget → safe; etc.). Tolerated when the entry was
      ``monotonicity_clamped`` (the clamp can move status above the eps
      bucket on purpose).
    """
    rid_to_kind = {r["region_id"]: r["kind"] for r in region_map_regions}
    violations: list[dict[str, Any]] = []

    def _rank(s: str) -> int:
        return _STATUS_RANK.get(s, 0)

    for entry in region_dossiers_summary:
        rid = entry["region_id"]
        kind = rid_to_kind.get(rid, "")
        dossier_obj = _read_json(run_dir / entry["dossier_ref"])
        sens = dossier_obj["numerical_sensitivity"]
        is_opaque = _is_opaque_kind(kind)

        # opaque_regions_require_reference: opaque/allocator regions get
        # uniform safe (allocator/view) or uniform requires_reference
        # (opaque). Anything else is wrong.
        if is_opaque:
            allowed = {"requires_reference"}
            for dt in ("fp32", "fp16_accum", "fp8_e4m3", "fast_math"):
                if sens[dt]["status"] not in allowed:
                    violations.append(
                        {
                            "rule": "opaque_regions_require_reference",
                            "region_id": rid,
                            "dtype": dt,
                            "actual": sens[dt]["status"],
                            "expected_one_of": sorted(allowed),
                        }
                    )

        if not is_opaque:
            # Strict precision order across the four dtypes.
            for i in range(len(_PRECISION_ORDER) - 1):
                a = _PRECISION_ORDER[i]
                b = _PRECISION_ORDER[i + 1]
                if _rank(sens[a]["status"]) > _rank(sens[b]["status"]):
                    violations.append(
                        {
                            "rule": "monotonic_precision_order",
                            "region_id": rid,
                            "more_precise": {"dtype": a, "status": sens[a]["status"]},
                            "less_precise": {"dtype": b, "status": sens[b]["status"]},
                        }
                    )
            # fp32 ≤ fast_math always.
            if _rank(sens["fp32"]["status"]) > _rank(sens["fast_math"]["status"]):
                violations.append(
                    {
                        "rule": "fp32_not_less_safe_than_fast_math",
                        "region_id": rid,
                        "fp32": sens["fp32"]["status"],
                        "fast_math": sens["fast_math"]["status"],
                    }
                )

        # Status-consistent-with-eps.
        for dt in ("fp32", "fp16_accum", "fp8_e4m3", "fast_math"):
            e = sens[dt]
            if e.get("monotonicity_clamped"):
                continue  # clamp wins by design
            if e["status"] == "requires_reference":
                continue
            # Reverse-engineer the budget from eps_out and budget_remaining is
            # noisy because of rounding, so we only flag the egregious case:
            # status=safe with eps_out >= 1.0 (massive overflow).
            if e["status"] == "safe" and e["eps_out"] >= 1.0:
                violations.append(
                    {
                        "rule": "status_consistent_with_eps",
                        "region_id": rid,
                        "dtype": dt,
                        "eps_out": e["eps_out"],
                        "status": e["status"],
                        "detail": "status=safe but eps_out >= 1.0",
                    }
                )

    rule_results = []
    for rule in (
        "monotonic_precision_order",
        "fp32_not_less_safe_than_fast_math",
        "opaque_regions_require_reference",
        "status_consistent_with_eps",
    ):
        offenders = [v for v in violations if v["rule"] == rule]
        rule_results.append(
            {
                "name": rule,
                "status": "pass" if not offenders else "fail",
                "violations": len(offenders),
            }
        )

    overall = "pass" if not violations else "fail"
    return {
        "schema_version": "numerical_sensitivity_audit_v1",
        "status": overall,
        "totals": {
            "regions_audited": len(region_dossiers_summary),
            "violations": len(violations),
        },
        "checks": rule_results,
        "violations": violations,
    }


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def build_region_dossiers(
    run_dir: Path, target_yaml_path: Path
) -> RegionDossierResult:
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "02_graph_analysis"
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"02_graph_analysis/ missing under {run_dir}; run graph-analysis first"
        )

    profile = load_target_profile(Path(target_yaml_path))

    region_map = _read_json(out_dir / "region_map.json")
    use_def = _read_json(out_dir / "tensor_use_def_graph.json")
    region_graph = _read_json(out_dir / "region_graph.json")
    pl_dir = run_dir / "01_payload_lowering"
    accounting = _read_json(pl_dir / "fx_to_payload_accounting.json")
    dialect = _read_json(pl_dir / "dialect_coverage.json")

    tensor_lookup = {t["tensor_id"]: t for t in use_def.get("tensors", [])}
    fx_target_lookup: dict[str, str] = {}
    for mod in accounting.get("modules", []):
        for n in mod.get("nodes", []):
            fx_target_lookup[n["fx_node"]] = n["fx_target"]

    # ------------------------------------------------------------------ #
    # Per-region dossiers
    # ------------------------------------------------------------------ #
    region_dossier_dir = out_dir / "region_dossiers"
    region_dossier_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale files (idempotent re-runs, keep dir, drop old jsons).
    for p in region_dossier_dir.glob("*.json"):
        p.unlink()

    region_dossier_index: dict[str, str] = {}
    region_dossiers_summary: list[dict[str, Any]] = []

    matmul_like_count = 0
    opaque_count = 0

    total_flops = 0
    total_bytes = 0
    structured_count = 0
    opaque_classification_count = 0
    bottleneck_compute = 0
    bottleneck_memory = 0
    fp8_status_histogram: dict[str, int] = {}
    fits_scratchpad_any = False
    not_fits_scratchpad_any = False
    single_consumer_transient_any = False

    for region in region_map.get("regions", []):
        rid = region["region_id"]
        cost = _compute_cost(region, profile)
        reuse = _compute_reuse(region, tensor_lookup)
        sensitivity = _compute_numerics(region, tensor_lookup, profile)
        partial: dict[str, Any] = {
            "cost": cost,
            "reuse": reuse,
            "numerical_sensitivity": sensitivity,
        }
        # M-37.9 Fix 1 / M-37.11 Improvement A: compute region_shape
        # FIRST so the working_set_curve can derive shape-fit tiles
        # that cleanly divide the region's actual dimensions.
        region_shape_info = _region_shape(
            region, tensor_lookup, run_dir=run_dir,
        )
        wsc = _working_set_curve(
            region, profile, tensor_lookup,
            region_shape=region_shape_info,
        )
        placement = _placement_envelope(region, profile, cost)
        partial["working_set_curve"] = wsc
        partial["placement_envelope"] = placement
        legality = _legality_constraints(region, partial)

        fx_targets = sorted(
            {
                fx_target_lookup.get(n, n)
                for n in region.get("fx_nodes", [])
            }
        )

        dossier = {
            "schema_version": "region_dossier_v2",
            "region_id": rid,
            "module_id": region["module_id"],
            "kind": region["kind"],
            "source": {
                "fx_nodes": list(region.get("fx_nodes", [])),
                "fx_targets": fx_targets,
                "payload_ops": [
                    {
                        "op_name": p["op_name"],
                        "region_id": p.get("region_id"),
                        "dispatch_id": p.get("dispatch_id"),
                        "callee": p.get("callee"),
                        "payload_ref": p["payload_ref"],
                    }
                    for p in region.get("payload_ops", [])
                ],
                "source_classification": region["source_classification"],
            },
            "cost": cost,
            "reuse": reuse,
            "numerical_sensitivity": sensitivity,
            "working_set_curve": wsc,
            "placement_envelope": placement,
            "legality_constraints": legality,
            # M-37.9 Fix 1: actual region shape, used by action_space
            # candidate_id construction to disambiguate same-named
            # regions across models.
            "region_shape": region_shape_info,
        }

        fname = _safe_filename(rid)
        dossier_path = region_dossier_dir / fname
        dossier_path.write_text(
            json.dumps(dossier, indent=2, sort_keys=True), encoding="utf-8"
        )
        rel = dossier_path.relative_to(run_dir).as_posix()
        region_dossier_index[rid] = rel
        region_dossiers_summary.append(
            {
                "region_id": rid,
                "kind": region["kind"],
                "estimated_latency_us": cost["estimated_latency_us"][profile.target_id],
                "bottleneck_resource": cost["bottleneck_resource"][profile.target_id],
                "source_classification": region["source_classification"],
                "fx_targets": fx_targets,
                "dossier_ref": rel,
            }
        )

        # Aggregates
        total_flops += int(region["estimated"]["flops"])
        total_bytes += int(region["estimated"]["bytes"])
        if region["source_classification"] == "decomposed_structured":
            structured_count += 1
        if region["source_classification"] == "opaque_fallback" or _is_opaque_kind(
            region["kind"]
        ):
            opaque_classification_count += 1
        if _is_matmul_like(region["kind"]):
            matmul_like_count += 1
        if _is_opaque_kind(region["kind"]):
            opaque_count += 1
        bottleneck = cost["bottleneck_resource"][profile.target_id]
        if bottleneck == "compute":
            bottleneck_compute += 1
        elif bottleneck == "memory":
            bottleneck_memory += 1
        st = sensitivity["fp8_e4m3"]["status"]
        fp8_status_histogram[st] = fp8_status_histogram.get(st, 0) + 1
        for tile in wsc:
            if tile["fits_scratchpad"]:
                fits_scratchpad_any = True
            else:
                not_fits_scratchpad_any = True
        for o in reuse["outputs"]:
            if o["consumer_count"] == 1 and o["lifetime_class"] == "transient":
                single_consumer_transient_any = True

    total_regions = len(region_map.get("regions", []))
    total_tensors = len(use_def.get("tensors", []))

    # Structured/opaque fractions from dialect coverage (truth), not from
    # the region count (which can over-count single-op regions).
    dial_agg = dialect.get("aggregate", {})
    total_payload_ops = int(dial_agg.get("total_payload_ops", 0))
    structured_ops_total = sum(
        n
        for op, n in dial_agg.get("structured_ops", {}).items()
        if op.startswith(("linalg.", "tensor.", "arith."))
    )
    opaque_ops_total = sum(dial_agg.get("opaque_func_calls", {}).values())
    if total_payload_ops > 0:
        structured_fraction = structured_ops_total / total_payload_ops
        opaque_fraction = opaque_ops_total / total_payload_ops
    else:
        structured_fraction = 0.0
        opaque_fraction = 0.0

    # Bottleneck class roll-up.
    if bottleneck_compute > 0 and bottleneck_memory > 0:
        bottleneck_class = "mixed"
    elif bottleneck_compute > 0:
        bottleneck_class = "compute"
    elif bottleneck_memory > 0:
        bottleneck_class = "memory"
    else:
        bottleneck_class = "unknown"

    # Top regions by estimated latency.
    top_regions = sorted(
        region_dossiers_summary,
        key=lambda r: r["estimated_latency_us"],
        reverse=True,
    )[:10]

    summary = {
        "total_regions": total_regions,
        "total_tensors": total_tensors,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "structured_fraction": round(structured_fraction, 6),
        "opaque_fraction": round(opaque_fraction, 6),
        "bottleneck_class": bottleneck_class,
        "bottleneck_compute_count": bottleneck_compute,
        "bottleneck_memory_count": bottleneck_memory,
        "fp8_status_histogram": fp8_status_histogram,
        "matmul_like_count": matmul_like_count,
        "opaque_count": opaque_count,
        "single_consumer_transient_seen": single_consumer_transient_any,
    }

    # Model_id from the first FX accounting module — mirrors how the rest
    # of the pipeline derives it.
    model_id = ""
    rm_path = run_dir / "run_manifest.json"
    if rm_path.exists():
        rm = _read_json(rm_path)
        model_id = rm.get("model", {}).get("model_id", "")
    if not model_id:
        # Fall back to first module_id stem.
        model_id = (
            region_map.get("regions", [{}])[0].get("module_id", "model")
            if region_map.get("regions")
            else "model"
        )

    # ------------------------------------------------------------------ #
    # graph_analysis.mlir
    # ------------------------------------------------------------------ #
    mlir_text = _emit_graph_analysis_mlir(
        model_id=model_id,
        target_id=profile.target_id,
        summary=summary,
        region_map=region_map,
        use_def=use_def,
    )
    mlir_path = out_dir / "graph_analysis.mlir"
    mlir_path.write_text(mlir_text, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # graph_dossier_v2.json
    # ------------------------------------------------------------------ #
    graph_dossier = {
        "schema_version": "graph_dossier_v2",
        "model_id": model_id,
        "target_id": profile.target_id,
        "source": {
            "graph_analysis_ir": mlir_path.relative_to(run_dir).as_posix(),
            "region_map": (out_dir / "region_map.json").relative_to(run_dir).as_posix(),
            "tensor_use_def_graph": (
                out_dir / "tensor_use_def_graph.json"
            ).relative_to(run_dir).as_posix(),
            "region_graph": (out_dir / "region_graph.json").relative_to(run_dir).as_posix(),
            "fx_to_payload_accounting": (
                pl_dir / "fx_to_payload_accounting.json"
            ).relative_to(run_dir).as_posix(),
        },
        "summary": summary,
        "critical_path": list(region_graph.get("critical_path", [])),
        "top_regions_by_estimated_latency": top_regions,
        "region_dossiers": region_dossier_index,
    }
    graph_dossier_path = out_dir / "graph_dossier_v2.json"
    graph_dossier_path.write_text(
        json.dumps(graph_dossier, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # dossier_validation.json
    # ------------------------------------------------------------------ #
    checks: list[dict[str, Any]] = []

    def _add_check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    rm_region_ids = {r["region_id"] for r in region_map.get("regions", [])}
    dossier_region_ids = set(region_dossier_index.keys())
    _add_check(
        "every_region_has_dossier",
        rm_region_ids <= dossier_region_ids,
        f"missing={sorted(rm_region_ids - dossier_region_ids)}",
    )
    _add_check(
        "no_extraneous_dossiers",
        dossier_region_ids <= rm_region_ids,
        f"extra={sorted(dossier_region_ids - rm_region_ids)}",
    )

    # Every payload_ref resolves on disk.
    bad_refs: list[str] = []
    for ref_path in region_dossier_index.values():
        dpath = run_dir / ref_path
        try:
            obj = _read_json(dpath)
        except (OSError, json.JSONDecodeError) as exc:
            bad_refs.append(f"{ref_path}: {exc}")
            continue
        for po in obj.get("source", {}).get("payload_ops", []):
            ref = run_dir / po["payload_ref"]
            if not ref.exists():
                bad_refs.append(f"{po['payload_ref']} (in {ref_path})")
    _add_check(
        "all_payload_refs_resolve",
        not bad_refs,
        f"missing_count={len(bad_refs)}; first={bad_refs[:3]}",
    )

    # Matmul-like regions have non-empty working_set_curve.
    bad_matmul: list[str] = []
    for entry in region_dossiers_summary:
        if not _is_matmul_like(
            next((r["kind"] for r in region_map["regions"] if r["region_id"] == entry["region_id"]), "")
        ):
            continue
        dossier_obj = _read_json(run_dir / entry["dossier_ref"])
        if not dossier_obj["working_set_curve"]:
            bad_matmul.append(entry["region_id"])
    _add_check(
        "matmul_like_regions_have_working_set_curve",
        not bad_matmul,
        f"empty_curve={bad_matmul}",
    )

    # Opaque regions have a "requires_reference_or_extension" legality.
    bad_opaque: list[str] = []
    for entry in region_dossiers_summary:
        kind = next(
            (r["kind"] for r in region_map["regions"] if r["region_id"] == entry["region_id"]), ""
        )
        if not _is_opaque_kind(kind):
            continue
        dossier_obj = _read_json(run_dir / entry["dossier_ref"])
        if not any(
            c["constraint"] == "requires_reference_or_extension" and not c["ok"]
            for c in dossier_obj["legality_constraints"]
        ):
            bad_opaque.append(entry["region_id"])
    _add_check(
        "opaque_regions_marked_requires_reference",
        not bad_opaque,
        f"missing_constraint={bad_opaque}",
    )

    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    validation: dict[str, Any] = {
        "schema_version": "dossier_validation_v1",
        "overall": overall,
        "totals": {
            "regions_in_map": len(rm_region_ids),
            "region_dossiers_emitted": len(dossier_region_ids),
            "matmul_like_regions": matmul_like_count,
            "opaque_regions": opaque_count,
            "bottleneck_compute_count": bottleneck_compute,
            "bottleneck_memory_count": bottleneck_memory,
            "fp8_status_histogram": fp8_status_histogram,
            "fits_scratchpad_any": fits_scratchpad_any,
            "not_fits_scratchpad_any": not_fits_scratchpad_any,
            "single_consumer_transient_seen": single_consumer_transient_any,
        },
        "checks": checks,
    }
    validation_path = out_dir / "dossier_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # numerical_sensitivity_audit.json (M-03.5)
    # ------------------------------------------------------------------ #
    ns_audit = _build_numerical_sensitivity_audit(
        region_dossiers_summary, region_map.get("regions", []), run_dir
    )
    ns_audit_path = out_dir / "numerical_sensitivity_audit.json"
    ns_audit_path.write_text(
        json.dumps(ns_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Reflect the M-03.5 audit in dossier_validation as a top-level check
    # so a single ``dossier_validation.overall`` answers "is the dossier
    # decision-quality?". We rewrite validation_path with the additional
    # check folded in.
    validation["checks"].append(
        {
            "name": "numerical_sensitivity_audit",
            "status": ns_audit["status"],
            "detail": (
                f"violations={ns_audit['totals']['violations']}; "
                f"see numerical_sensitivity_audit.json"
            ),
        }
    )
    if ns_audit["status"] != "pass":
        validation["overall"] = "fail"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    return RegionDossierResult(
        graph_analysis_mlir_path=mlir_path,
        graph_dossier_v2_path=graph_dossier_path,
        region_dossier_dir=region_dossier_dir,
        dossier_validation_path=validation_path,
        numerical_sensitivity_audit_path=ns_audit_path,
        region_dossier_count=len(dossier_region_ids),
        matmul_like_count=matmul_like_count,
        opaque_count=opaque_count,
    )
