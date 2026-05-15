"""TinyLlama end-to-end solver evaluation.

Drives the four Phase E planners (placement, overlap schedule,
memory MILP, Z3 obligations) against the topology of
TinyLlama-1.1B (decoder-only LLM, 22 layers, hidden=2048,
intermediate=5632, 32 attention heads / 4 KV heads).

The evaluation is **topology-derived**, not full-graph-compilation:
we read the HuggingFace ``config.json`` directly, enumerate every
matmul (Q/K/V/O projections + gate/up/down) per layer, and feed the
real operator shapes into the solver planners. This is honest:

- The planners see the exact tensor sizes of a 1-token decode step
  (seq_len=1) plus an attention pass with a configurable KV-cache
  length.
- No torch.export, no GPU required, no full forward pass.
- All artifacts produced are real solver responses with
  ``formulation_hash``, ``selected_backend``, ``status``, ``time_ms``.

Usage::

    uv run python scripts/dev/eval_tinyllama_solvers.py \\
        --out /tmp/tinyllama_solver_eval \\
        --kv-len 128 \\
        --num-devices 2 \\
        --num-layers 22 \\
        --z3-proof-required

The output directory contains the same layout the trust gates
audit: ``05_execution_plan/solver/*_{request,response}.json`` and
``04_kernel_codegen/solver/*z3_obligations.json``, plus a paper-facing
``tinyllama_solver_eval_report.{json,md}`` summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.solve.backend_registry import default_registry
from compgen.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    _build_formulation as _build_mem_formulation,
    plan_memory,
)
from compgen.solve.overlap_planner import (
    Dependency,
    Operation,
    OverlapPlanInput,
    Resource,
    _build_formulation as _build_overlap_formulation,
    plan_overlap,
)
from compgen.solve.placement_planner import (
    Device,
    Edge,
    PlacementPlanInput,
    Region,
    _build_formulation as _build_placement_formulation,
    plan_placement,
)
from compgen.solve.reports import write_solver_request, write_solver_response
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)
from compgen.solve.z3_obligations import (
    OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
)


# ---------------------------------------------------------------------------
# Topology extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LlamaConfig:
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    vocab: int


def _find_tinyllama_config() -> Path | None:
    hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    base = Path(hf_cache) / "hub"
    if not base.is_dir():
        return None
    for cfg in base.rglob("config.json"):
        if "TinyLlama-1.1B-Chat-v1.0" in str(cfg):
            return cfg
    return None


def _load_config(path: Path) -> _LlamaConfig:
    body = json.loads(path.read_text())
    return _LlamaConfig(
        hidden=int(body["hidden_size"]),
        intermediate=int(body["intermediate_size"]),
        n_heads=int(body["num_attention_heads"]),
        n_kv_heads=int(body["num_key_value_heads"]),
        n_layers=int(body["num_hidden_layers"]),
        vocab=int(body["vocab_size"]),
    )


@dataclass(frozen=True)
class _Region:
    region_id: str
    layer: int
    family: str  # attn_qkv | attn_proj | mlp_gate | mlp_up | mlp_down | rmsnorm | rope | attention_score
    m: int  # rows
    n: int  # cols
    k: int  # inner dim (0 for non-matmul)
    bytes_io: int  # IO bytes (rough)


def _enumerate_regions(cfg: _LlamaConfig, *, num_layers: int, kv_len: int) -> list[_Region]:
    """Enumerate one decode step (seq_len=1)."""

    out: list[_Region] = []
    head_dim = cfg.hidden // cfg.n_heads
    q_dim = head_dim * cfg.n_heads
    kv_dim = head_dim * cfg.n_kv_heads
    bytes_per_elem = 4  # we plan in f32

    for layer in range(num_layers):
        # input_layernorm: shape (1, hidden)
        out.append(_Region(
            region_id=f"layer{layer}.input_norm",
            layer=layer, family="rmsnorm",
            m=1, n=cfg.hidden, k=0,
            bytes_io=1 * cfg.hidden * bytes_per_elem,
        ))
        # Q/K/V projections
        out.append(_Region(
            region_id=f"layer{layer}.q_proj",
            layer=layer, family="attn_qkv",
            m=1, n=q_dim, k=cfg.hidden,
            bytes_io=(1 * cfg.hidden + cfg.hidden * q_dim + 1 * q_dim) * bytes_per_elem,
        ))
        out.append(_Region(
            region_id=f"layer{layer}.k_proj",
            layer=layer, family="attn_qkv",
            m=1, n=kv_dim, k=cfg.hidden,
            bytes_io=(1 * cfg.hidden + cfg.hidden * kv_dim + 1 * kv_dim) * bytes_per_elem,
        ))
        out.append(_Region(
            region_id=f"layer{layer}.v_proj",
            layer=layer, family="attn_qkv",
            m=1, n=kv_dim, k=cfg.hidden,
            bytes_io=(1 * cfg.hidden + cfg.hidden * kv_dim + 1 * kv_dim) * bytes_per_elem,
        ))
        # Rotary embedding (cheap pointwise on Q + K)
        out.append(_Region(
            region_id=f"layer{layer}.rope",
            layer=layer, family="rope",
            m=1, n=q_dim + kv_dim, k=0,
            bytes_io=(q_dim + kv_dim) * bytes_per_elem,
        ))
        # Attention scores (Q @ K^T) and weighted sum (scores @ V).
        out.append(_Region(
            region_id=f"layer{layer}.attn_scores",
            layer=layer, family="attention_score",
            m=cfg.n_heads, n=kv_len, k=head_dim,
            bytes_io=(q_dim + kv_dim * kv_len + cfg.n_heads * kv_len) * bytes_per_elem,
        ))
        out.append(_Region(
            region_id=f"layer{layer}.attn_combine",
            layer=layer, family="attention_score",
            m=cfg.n_heads, n=head_dim, k=kv_len,
            bytes_io=(cfg.n_heads * kv_len + kv_dim * kv_len + q_dim) * bytes_per_elem,
        ))
        # Output projection
        out.append(_Region(
            region_id=f"layer{layer}.o_proj",
            layer=layer, family="attn_proj",
            m=1, n=cfg.hidden, k=q_dim,
            bytes_io=(1 * q_dim + q_dim * cfg.hidden + 1 * cfg.hidden) * bytes_per_elem,
        ))
        # post_attention_layernorm
        out.append(_Region(
            region_id=f"layer{layer}.post_attn_norm",
            layer=layer, family="rmsnorm",
            m=1, n=cfg.hidden, k=0,
            bytes_io=1 * cfg.hidden * bytes_per_elem,
        ))
        # MLP gate + up + down with SiLU between
        out.append(_Region(
            region_id=f"layer{layer}.mlp_gate",
            layer=layer, family="mlp_gate",
            m=1, n=cfg.intermediate, k=cfg.hidden,
            bytes_io=(1 * cfg.hidden + cfg.hidden * cfg.intermediate + 1 * cfg.intermediate) * bytes_per_elem,
        ))
        out.append(_Region(
            region_id=f"layer{layer}.mlp_up",
            layer=layer, family="mlp_up",
            m=1, n=cfg.intermediate, k=cfg.hidden,
            bytes_io=(1 * cfg.hidden + cfg.hidden * cfg.intermediate + 1 * cfg.intermediate) * bytes_per_elem,
        ))
        out.append(_Region(
            region_id=f"layer{layer}.mlp_down",
            layer=layer, family="mlp_down",
            m=1, n=cfg.hidden, k=cfg.intermediate,
            bytes_io=(1 * cfg.intermediate + cfg.intermediate * cfg.hidden + 1 * cfg.hidden) * bytes_per_elem,
        ))
    return out


def _enumerate_edges(regions: list[_Region]) -> list[tuple[str, str, int]]:
    """Build the per-layer producer→consumer chain."""

    edges: list[tuple[str, str, int]] = []
    by_layer: dict[int, list[_Region]] = {}
    for r in regions:
        by_layer.setdefault(r.layer, []).append(r)

    for layer, ops in by_layer.items():
        names = [r.region_id for r in ops]
        # Linear dependency within a layer (input_norm → qkv → rope →
        # scores → combine → o_proj → post_norm → mlp_gate →
        # mlp_up → mlp_down). q/k/v share input_norm.
        in_norm = f"layer{layer}.input_norm"
        for proj in (f"layer{layer}.q_proj", f"layer{layer}.k_proj", f"layer{layer}.v_proj"):
            edges.append((in_norm, proj, 1))
        edges.append((f"layer{layer}.q_proj", f"layer{layer}.rope", 1))
        edges.append((f"layer{layer}.k_proj", f"layer{layer}.rope", 1))
        edges.append((f"layer{layer}.rope", f"layer{layer}.attn_scores", 1))
        edges.append((f"layer{layer}.v_proj", f"layer{layer}.attn_combine", 1))
        edges.append((f"layer{layer}.attn_scores", f"layer{layer}.attn_combine", 1))
        edges.append((f"layer{layer}.attn_combine", f"layer{layer}.o_proj", 1))
        edges.append((f"layer{layer}.o_proj", f"layer{layer}.post_attn_norm", 1))
        edges.append((f"layer{layer}.post_attn_norm", f"layer{layer}.mlp_gate", 1))
        edges.append((f"layer{layer}.post_attn_norm", f"layer{layer}.mlp_up", 1))
        edges.append((f"layer{layer}.mlp_gate", f"layer{layer}.mlp_down", 1))
        edges.append((f"layer{layer}.mlp_up", f"layer{layer}.mlp_down", 1))
        # Next-layer input_norm depends on this-layer mlp_down (residual is
        # outside our compute regions but the sequencing holds).
        next_layer_norm = f"layer{layer + 1}.input_norm"
        if any(r.region_id == next_layer_norm for r in regions):
            edges.append((f"layer{layer}.mlp_down", next_layer_norm, 1))
    return edges


def _est_us(region: _Region, *, device: str) -> float:
    """Rough analytical cost per device. CPU peak ~ 50 GFLOPs/s, GPU ~ 5 TFLOPs/s.

    Returns microseconds. Cheap models are still cheap on both — the
    point is to give the placement solver a real cost discriminator
    when CPU and GPU are co-resident.
    """

    if region.k > 0:
        flops = 2.0 * region.m * region.n * region.k
    else:
        flops = region.m * region.n  # pointwise / norm proxy
    cpu_gflops = 50.0e9
    gpu_gflops = 5.0e12
    rate = cpu_gflops if device == "cpu" else gpu_gflops
    return max(flops / rate * 1e6, 0.01)


# ---------------------------------------------------------------------------
# Solver invocations
# ---------------------------------------------------------------------------


def _build_placement(
    regions: list[_Region], edges: list[tuple[str, str, int]], devices_def: list[tuple[str, int]]
) -> PlacementPlanInput:
    device_ids = tuple(name for name, _ in devices_def)
    out_regions: list[Region] = []
    for r in regions:
        out_regions.append(Region(
            region_id=r.region_id,
            allowed_devices=device_ids,
            memory_bytes=r.bytes_io,
            compute_cost_by_device={d: _est_us(r, device=d) for d in device_ids},
        ))
    out_devices = tuple(
        Device(device_id=name, memory_capacity=cap, target_class=name)
        for name, cap in devices_def
    )
    transfer_costs: dict[tuple[str, str], float] = {}
    for i, (a, _) in enumerate(devices_def):
        for j, (b, _) in enumerate(devices_def):
            if i == j:
                continue
            # Rough PCIe: ~32 GB/s = 32 ns/byte; per-byte cost in us.
            transfer_costs[(a, b)] = 1.0 / (32.0 * 1024 * 1024 * 1024) * 1e6
    out_edges = []
    for src, dst, _ in edges:
        out_edges.append(Edge(
            src_region=src,
            dst_region=dst,
            bytes_=next((r.bytes_io for r in regions if r.region_id == src), 0) // 4,
            transfer_cost_by_device_pair=transfer_costs,
        ))
    return PlacementPlanInput(
        regions=tuple(out_regions),
        devices=out_devices,
        edges=tuple(out_edges),
        time_budget_ms=10_000,
    )


def _build_overlap(
    regions: list[_Region],
    edges: list[tuple[str, str, int]],
    placement: dict[str, str],
) -> OverlapPlanInput:
    ops: list[Operation] = []
    for r in regions:
        # Quantize to ticks: 1 tick = 0.1us, capped.
        dev = placement.get(r.region_id, "cpu")
        dur_us = _est_us(r, device=dev)
        ticks = max(1, int(dur_us * 10))
        ops.append(Operation(
            op_id=r.region_id, duration=ticks, resource_id=dev,
            kind="compute",
        ))
    deps = tuple(Dependency(src_op=s, dst_op=d) for s, d, _ in edges)
    resources = tuple(Resource(resource_id=d) for d in {placement.get(r.region_id, "cpu") for r in regions})
    return OverlapPlanInput(
        operations=tuple(ops),
        dependencies=deps,
        resources=resources,
        time_budget_ms=15_000,
    )


def _build_memory(
    regions: list[_Region],
    placement: dict[str, str],
    schedule: dict[str, tuple[int, int]],
    devices_def: list[tuple[str, int]],
) -> MemoryPlanInput:
    """One buffer per region; lifetime taken from schedule.

    Tier id = device id (single-tier per device for this evaluation).
    """

    tier_capacities = tuple(
        TierCapacity(tier_id=name, capacity_bytes=cap, weight=1.0)
        for name, cap in devices_def
    )
    # Memory lifetime model:
    # - Per-layer activations live during their layer's tick range
    #   (layer N spans ticks [N*10, (N+1)*10]).
    # - This produces clean disjoint lifetimes across layers, which is
    #   the realistic single-token-decode model and what unlocks stage
    #   decomposition. The overlap planner's tick start_times measure
    #   *issue order* on resources, not buffer lifetime — those are
    #   different things.
    # Every region is allowed on its placement-assigned tier AND on a
    # "host" fallback tier, so the rule-based heuristic can route
    # large/long-lived buffers to host honestly.
    buffers: list[BufferSpec] = []
    layer_span = 10
    placement_devices = {placement.get(r.region_id, devices_def[0][0]) for r in regions}
    fallback_tier = next(
        (name for name, _ in devices_def if name not in placement_devices),
        devices_def[0][0],
    )
    for r in regions:
        layer_start = r.layer * layer_span
        layer_end = (r.layer + 1) * layer_span
        primary_tier = placement.get(r.region_id, devices_def[0][0])
        allowed = tuple(dict.fromkeys((primary_tier, fallback_tier)))
        buffers.append(BufferSpec(
            buffer_id=r.region_id,
            size_bytes=max(r.bytes_io, 64),
            lifetime_start=layer_start,
            lifetime_end=layer_end,
            allowed_tiers=allowed,
        ))
    # Declare alias candidates: every region's output may alias the
    # next region's input on the same device if their lifetimes are
    # disjoint. Conservative — only same-layer same-family pairs.
    by_layer_family: dict[tuple[int, str], list[str]] = {}
    for r in regions:
        by_layer_family.setdefault((r.layer, r.family), []).append(r.region_id)
    alias_pairs: list[AliasCandidate] = []
    for ids in by_layer_family.values():
        if len(ids) >= 2:
            alias_pairs.append(AliasCandidate(buffer_a=ids[0], buffer_b=ids[-1]))
    return MemoryPlanInput(
        buffers=tuple(buffers),
        tier_capacities=tier_capacities,
        alias_candidates=tuple(alias_pairs),
        objective_lambda=1e-9,
        time_budget_ms=60_000,
    )


# ---------------------------------------------------------------------------
# Z3 proofs
# ---------------------------------------------------------------------------


def _run_z3_proofs(
    regions: list[_Region], *, out_dir: Path, head_dim: int, tile: int = 16
) -> dict:
    """Prove, for each unique (m,n,k) signature of a matmul region,
    that the contract precondition ``k mod tile == 0`` holds OR
    surface a counterexample. This is the production-path
    integration of the Z3 obligation harness against real TinyLlama
    shapes.
    """

    registry = default_registry()
    z3_backend = registry.get_backend(SolverBackendName.Z3)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": "z3_obligations_index_v1",
        "tile": tile,
        "obligations": [],
    }
    if z3_backend is None or z3_backend.probe().availability is not BackendAvailabilityStatus.AVAILABLE:
        report["overall"] = "skipped"
        report["skipped_reason"] = "z3 backend unavailable"
        (out_dir / "tinyllama_z3_obligations.json").write_text(
            json.dumps(report, sort_keys=True, indent=2)
        )
        return report

    # Distinct k dims observed.
    distinct_ks = sorted({r.k for r in regions if r.k > 0})
    for k_val in distinct_ks:
        # Real obligation: prove K mod tile == 0 under the premise
        # that K equals the concrete tinyllama k.
        problem_id = f"k_eq_{k_val}_implies_mod_{tile}"
        request = SolverRequest(
            problem_id=problem_id,
            problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
            formulation={
                "obligation_kind": OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
                "params": {
                    "variables": {"K": {"min": k_val, "max": k_val}},
                    "applies_when": [{"op": "equal", "a": "K", "b": k_val}],
                    "precondition": {"op": "divisible_by", "var": "K", "k": tile},
                },
            },
        )
        response = z3_backend.solve(request)
        write_solver_request(request, out_dir / f"{problem_id}.request.json")
        write_solver_response(response, out_dir / f"{problem_id}.response.json")
        report["obligations"].append({
            "k": k_val,
            "tile": tile,
            "status": response.status.value,
            "selected_backend": response.selected_backend.value,
            "formulation_hash": response.formulation_hash,
            "time_ms": response.time_ms,
            "counterexample": response.counterexample,
        })
    report["overall"] = (
        "pass"
        if all(o["status"] == "proved" for o in report["obligations"])
        else "honest_residual"
    )
    (out_dir / "tinyllama_z3_obligations.json").write_text(
        json.dumps(report, sort_keys=True, indent=2)
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _emit_summary_md(report: dict) -> str:
    lines = [
        "# TinyLlama solver evaluation",
        "",
        f"- **generated_at**: {report['generated_at']}",
        f"- **config**: hidden={report['config']['hidden']}, "
        f"intermediate={report['config']['intermediate']}, "
        f"layers={report['layers_used']}, "
        f"kv_len={report['kv_len']}, "
        f"devices={report['devices']}",
        f"- **regions_total**: {report['regions_total']}",
        f"- **edges_total**: {report['edges_total']}",
        "",
        "## Solver outcomes",
        "",
        "| stage | backend | status | objective | time_ms | formulation_hash |",
        "|---|---|---|---|---|---|",
    ]
    for stage in ("placement", "overlap", "memory"):
        row = report["solvers"][stage]
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {:.1f} | `{}` |".format(
                stage,
                row["backend"], row["status"],
                row.get("objective_value") if row.get("objective_value") is not None else "-",
                row["time_ms"],
                row.get("formulation_hash") or "-",
            )
        )
    lines += [
        "",
        "## Placement assignments by device",
        "",
        "| device | region count |",
        "|---|---|",
    ]
    for d, n in report["placement_counts"].items():
        lines.append(f"| `{d}` | {n} |")
    lines += [
        "",
        "## Memory peak per tier",
        "",
        "| tier | bytes |",
        "|---|---|",
    ]
    for t, b in report["tier_peak_usage"].items():
        lines.append(f"| `{t}` | {b:,} |")
    if report["z3"]:
        lines += [
            "",
            "## Z3 obligations (matmul ``K mod 16 == 0`` per distinct K)",
            "",
            f"- **tile**: {report['z3']['tile']}",
            f"- **overall**: {report['z3'].get('overall', 'unknown')}",
            "",
            "| K | status | backend | time_ms | counterexample |",
            "|---|---|---|---|---|",
        ]
        for o in report["z3"]["obligations"]:
            cex = "-" if not o.get("counterexample") else str(o["counterexample"])
            lines.append(
                f"| {o['k']} | `{o['status']}` | `{o['selected_backend']}` | {o['time_ms']:.2f} | {cex} |"
            )
    lines.append("")
    lines.append("## Honest residuals")
    for r in report.get("honest_residuals", []):
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/tinyllama_solver_eval"))
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=22, help="cap layers for faster solves")
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--z3-proof-required", action="store_true")
    parser.add_argument(
        "--use-solver-hints", action="store_true",
        help=(
            "Pass LLM/rule-based hints to the MILP memory planner. "
            "Stage decomposition + tier-fixing typically reduces "
            "MOSEK solve time by an order of magnitude on 4+ layers."
        ),
    )
    parser.add_argument(
        "--hint-mode", default="rule_based",
        choices=["rule_based", "llm_file", "merged"],
        help="Hint source: deterministic heuristic, LLM JSON file, or merged.",
    )
    parser.add_argument(
        "--llm-hint-path", type=Path, default=None,
        help="Path to LLM-produced hint JSON (mode=llm_file|merged).",
    )
    args = parser.parse_args(argv)

    cfg_path = _find_tinyllama_config()
    if cfg_path is None:
        print("TinyLlama config not found in HF cache; aborting honestly.", file=sys.stderr)
        return 2
    cfg = _load_config(cfg_path)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    solver_dir = out_dir / "05_execution_plan" / "solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    z3_dir = out_dir / "04_kernel_codegen" / "solver"
    z3_dir.mkdir(parents=True, exist_ok=True)

    num_layers = min(args.num_layers, cfg.n_layers)
    regions = _enumerate_regions(cfg, num_layers=num_layers, kv_len=args.kv_len)
    edges = _enumerate_edges(regions)

    # Devices (rough memory capacity, chosen so the MILP isn't infeasible).
    bytes_per_layer = sum(r.bytes_io for r in regions if r.layer == 0)
    per_device_capacity = max(bytes_per_layer * num_layers * 2, 1 << 30)
    devices_def = [(f"d{i}", per_device_capacity) for i in range(args.num_devices)]
    if args.num_devices >= 2:
        devices_def[0] = ("cpu", per_device_capacity)
        devices_def[1] = ("gpu", per_device_capacity)

    report: dict[str, Any] = {
        "schema_version": "tinyllama_solver_eval_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "hidden": cfg.hidden,
            "intermediate": cfg.intermediate,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "n_layers": cfg.n_layers,
        },
        "layers_used": num_layers,
        "kv_len": args.kv_len,
        "devices": [d for d, _ in devices_def],
        "regions_total": len(regions),
        "edges_total": len(edges),
        "solvers": {},
        "placement_counts": {},
        "tier_peak_usage": {},
        "honest_residuals": [],
        "z3": None,
    }

    # ---- Placement ----------------------------------------------------
    placement_input = _build_placement(regions, edges, devices_def)
    p_response, p_plan = plan_placement(
        placement_input, problem_id="tinyllama_placement"
    )
    write_solver_request(
        SolverRequest(
            problem_id="tinyllama_placement",
            problem_kind=SolverProblemKind.PLACEMENT,
            formulation=_build_placement_formulation(placement_input),
            time_budget_ms=placement_input.time_budget_ms,
        ),
        solver_dir / "placement_solver_request.json",
    )
    write_solver_response(p_response, solver_dir / "placement_solver_response.json")
    if p_plan is not None:
        (out_dir / "05_execution_plan" / "placement_plan.solved.json").write_text(
            json.dumps(p_plan.to_dict(), sort_keys=True, indent=2)
        )
    report["solvers"]["placement"] = {
        "backend": p_response.selected_backend.value,
        "status": p_response.status.value,
        "objective_value": p_response.objective_value,
        "time_ms": p_response.time_ms,
        "formulation_hash": p_response.formulation_hash,
    }

    placement_map: dict[str, str] = {}
    if p_plan is not None:
        for a in p_plan.assignments:
            placement_map[a.region_id] = a.device_id
        counts: dict[str, int] = {}
        for d in placement_map.values():
            counts[d] = counts.get(d, 0) + 1
        report["placement_counts"] = counts
    else:
        report["honest_residuals"].append(
            f"placement solver returned {p_response.status.value}; "
            f"overlap + memory ran with default device=cpu for every region"
        )
        placement_map = {r.region_id: devices_def[0][0] for r in regions}
        report["placement_counts"] = {devices_def[0][0]: len(regions)}

    # ---- Overlap schedule --------------------------------------------
    overlap_input = _build_overlap(regions, edges, placement_map)
    o_response, o_sched = plan_overlap(
        overlap_input, problem_id="tinyllama_overlap"
    )
    write_solver_request(
        SolverRequest(
            problem_id="tinyllama_overlap",
            problem_kind=SolverProblemKind.OVERLAP_PLANNING,
            formulation=_build_overlap_formulation(overlap_input),
            time_budget_ms=overlap_input.time_budget_ms,
        ),
        solver_dir / "overlap_solver_request.json",
    )
    write_solver_response(o_response, solver_dir / "overlap_solver_response.json")
    if o_sched is not None:
        (out_dir / "05_execution_plan" / "overlap_schedule.solved.json").write_text(
            json.dumps(o_sched.to_dict(), sort_keys=True, indent=2)
        )
    report["solvers"]["overlap"] = {
        "backend": o_response.selected_backend.value,
        "status": o_response.status.value,
        "objective_value": o_response.objective_value,
        "time_ms": o_response.time_ms,
        "formulation_hash": o_response.formulation_hash,
    }

    schedule_map: dict[str, tuple[int, int]] = {}
    if o_sched is not None:
        for s in o_sched.schedule:
            schedule_map[s.op_id] = (s.start_tick, s.end_tick)
    else:
        report["honest_residuals"].append(
            f"overlap solver returned {o_response.status.value}; "
            f"memory planner using per-layer lifetimes as fallback"
        )
        for r in regions:
            schedule_map[r.region_id] = (r.layer, r.layer + 1)

    # ---- Memory MILP --------------------------------------------------
    memory_input = _build_memory(regions, placement_map, schedule_map, devices_def)
    memory_hints = None
    if args.use_solver_hints:
        from compgen.solve.llm_hint_provider import get_memory_hints

        memory_hints = get_memory_hints(
            memory_input,
            mode=args.hint_mode,
            llm_hint_path=args.llm_hint_path,
        )
        report["memory_hints"] = memory_hints.to_dict()
    m_response, m_plan = plan_memory(
        memory_input, problem_id="tinyllama_memory", hints=memory_hints,
    )
    write_solver_request(
        SolverRequest(
            problem_id="tinyllama_memory",
            problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
            formulation=_build_mem_formulation(memory_input),
            time_budget_ms=memory_input.time_budget_ms,
        ),
        solver_dir / "memory_solver_request.json",
    )
    write_solver_response(m_response, solver_dir / "memory_solver_response.json")
    if m_plan is not None:
        (out_dir / "05_execution_plan" / "memory_plan.solved.json").write_text(
            json.dumps(m_plan.to_dict(), sort_keys=True, indent=2)
        )
        report["tier_peak_usage"] = m_plan.tier_peak_usage
    report["solvers"]["memory"] = {
        "backend": m_response.selected_backend.value,
        "status": m_response.status.value,
        "objective_value": m_response.objective_value,
        "time_ms": m_response.time_ms,
        "formulation_hash": m_response.formulation_hash,
    }
    if m_plan is None:
        report["honest_residuals"].append(
            f"memory planner returned {m_response.status.value}: "
            f"{m_response.infeasibility_reason}"
        )

    # ---- Z3 obligations (matmul K mod tile) -------------------------
    if args.z3_proof_required:
        z3_report = _run_z3_proofs(regions, out_dir=z3_dir, head_dim=cfg.hidden // cfg.n_heads, tile=16)
        report["z3"] = z3_report

    # ---- Persist top-level summary ----------------------------------
    json_path = out_dir / "tinyllama_solver_eval_report.json"
    md_path = out_dir / "tinyllama_solver_eval_report.md"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2))
    md_path.write_text(_emit_summary_md(report))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
