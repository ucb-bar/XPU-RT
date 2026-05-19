"""Cross-target × cross-backend matrix driver.

Top-level entry point for the comparison study described in
``floofy-foraging-matsumoto.md``. Iterates a configurable subset of
the (backend × target × workload) matrix and persists one JSON per
cell under ``results/comparison/cross_target/per_cell/``.

Backends:

  * ``kb-vanilla`` — KB-vanilla. Today wired live on Gemmini only
    (via :mod:`xpu_rt.kb_gemmini.kb_pipeline_driver`); the Saturn
    fork was scope-narrowed and reports ``status="deferred"`` per
    the plan.
  * ``kb-v2`` — XPU-RT-native KB v2 (agentic, with the FusionPlanner
    + MegaContractEmitter machinery). Wired for both Gemmini and
    Saturn via :class:`xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv.CRiscvEvaluator`'s
    target dispatch + the per-target HardwareEnvelope.
  * ``autocomp`` — autocomp's own Gemmini / Saturn backends. Wired
    via :func:`xpu_rt.kernels.autocomp_adapter.resolve_autocomp_target`.
    Requires per-target env vars (``INT8_16PE_CHIPYARD_PATH`` /
    ``SATURN_CHIPYARD_PATH``); cells report ``status="env_missing"``
    with the missing var names when the env isn't set up.

Workloads:

  * ``smolvla_matmuls`` — the existing 14 SmolVLA single-kernel
    contracts from ``benchmarks/smolvla_subset``.
  * ``smolvla_mlp_block`` — one MLP block per layer from
    ``benchmarks/smolvla_block_enumerator``; what the pipeline-level
    fusion comparison measures.

Modes:

  * ``plan`` (default) — structural numbers + planner verdicts +
    env-readiness check. No LLM spend, no Spike execution. The
    deliverable today.
  * ``full`` — wires the live LLM + Spike eval. Gated behind the
    same budget check as the prior matmul study; aborts cleanly
    if the projected per-cell cost would push past the soft cap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.run_pipeline_comparison import envelope_for_target


logger = logging.getLogger("xpu_rt.benchmarks.run_cross_target_comparison")


# ---------------------------------------------------------------------------
# Per-cell result schema
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """One (backend, target, workload) cell of the matrix.

    A cell is "ready" when its status is ``"ok"``; everything else
    surfaces a clean reason (``"deferred"``, ``"env_missing"``,
    ``"budget_exceeded"``, ``"error"``) so the aggregator can render
    the matrix without dropping data.
    """

    backend: str
    target: str
    workload: str
    status: str  # ok | deferred | env_missing | budget_exceeded | error
    notes: str = ""
    n_kernels_vanilla: int = 0
    n_kernels_agentic: int = 0
    planner_estimated_speedup: float = 1.0
    correctness_rate: float = math.nan
    geomean_cycles: float = math.nan
    rounds_per_kernel: float = math.nan
    cost_usd: float = 0.0
    wall_s: float = 0.0
    env_missing: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cell runners (one per backend)
# ---------------------------------------------------------------------------


def _cell_dir(out_dir: Path, backend: str, target: str, workload: str) -> Path:
    p = out_dir / "per_cell" / f"{backend}__{target}__{workload}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _persist_cell(out_dir: Path, cell: CellResult) -> None:
    d = _cell_dir(out_dir, cell.backend, cell.target, cell.workload)
    (d / "status.json").write_text(json.dumps(dataclasses.asdict(cell), indent=2))


# --- kb-v2 (agentic) ---


def _run_kb_v2_cell(
    *,
    target: str,
    workload: str,
    mode: str,
    out_dir: Path,
) -> CellResult:
    """Run the KB-v2 agentic flow for one cell.

    In ``plan`` mode this re-uses
    :func:`xpu_rt.benchmarks.run_pipeline_comparison.run` to get the
    planner-level numbers without invoking the LLM. In ``full`` mode
    it would additionally drive the KB v2 agent loop + the Spike
    evaluator — wired conditional on the toolchain + Gemini key.
    """
    from xpu_rt.benchmarks.run_pipeline_comparison import run as pipeline_run

    start = time.time()
    cell = CellResult(backend="kb-v2", target=target, workload=workload, status="ok")

    blocks = _load_workload_blocks(workload)
    envelope = envelope_for_target(target)
    sub_out = _cell_dir(out_dir, cell.backend, cell.target, cell.workload) / "pipeline"
    report = pipeline_run(sub_out, mode="plan", blocks=blocks, envelope=envelope)

    cell.n_kernels_vanilla = report.total_vanilla_kernels
    cell.n_kernels_agentic = report.total_agentic_kernels
    cell.planner_estimated_speedup = report.aggregate_planner_speedup_geomean

    if mode == "full":
        missing = _kb_v2_full_mode_env_missing(target)
        if missing:
            cell.status = "env_missing"
            cell.env_missing = tuple(missing)
            cell.notes = (
                "KB v2 full-mode run needs the riscv-tools conda env on "
                "PATH + GOOGLE_API_KEY set + the Saturn / Gemmini "
                "Target Card seeded."
            )
            cell.wall_s = time.time() - start
            return cell

        # Live invocation. Drive KB-v2 per shape per repeat, persist
        # canonical rows. Each repeat re-derives the StateVector from
        # the contract so the LLM sees the same prompt structure
        # (only the gemini-side seed differs, which the API picks at
        # request-time).
        repeats = int(_KB_V2_REPEATS_OVERRIDE or 1)
        samples = _kb_v2_live_run(
            target=target, workload=workload, out_dir=out_dir,
            repeats=repeats,
        )
        from xpu_rt.benchmarks.canonical_metrics import write_jsonl

        cell_dir = _cell_dir(out_dir, cell.backend, cell.target, cell.workload)
        n = write_jsonl(samples, cell_dir / "samples.jsonl")
        cell.extra["n_samples_persisted"] = n
        cell.extra["n_correct"] = sum(1 for s in samples if s.correctness)
        cell.extra["requested_repeats"] = repeats
        cell.notes = (
            f"KB-v2 live run: {n} samples persisted, "
            f"{cell.extra['n_correct']} correct."
        )

    cell.wall_s = time.time() - start
    return cell


# Module-level override for repeats — set by the matrix driver before
# invoking each cell so the cell function doesn't have to thread the
# arg through the BACKEND_RUNNERS signature (which is fixed).
_KB_V2_REPEATS_OVERRIDE: int = 1


def _kb_v2_live_run(
    *,
    target: str,
    workload: str,
    out_dir: Path,
    repeats: int,
):
    """Drive the KB-v2 agent loop live for every (shape, repeat) in
    one cell. Returns a list of :class:`CanonicalCellRow`.

    Shapes are taken from the cached KB-vanilla report for matmuls
    (the same 14 SmolVLA shapes the prior batch ran), and from the
    block enumerator for ``smolvla_mlp_block`` (per-block MEGA
    contracts).

    Budget-gated via ``gemini_usage.check_pre_call`` — each shape ×
    repeat asks for up to ``$0.05`` projected (4 rounds × ~$0.005
    plus headroom); the wider study cap ($10) is enforced by the
    matrix runner.
    """
    from xpu_rt.benchmarks.canonical_metrics import (
        CanonicalCellRow, shape_id_for_matmul,
    )
    from xpu_rt.benchmarks.loaders.kb_v2_loader import (
        CostSnapshot, load_kb_v2_row,
    )
    from xpu_rt.benchmarks.loaders.kb_vanilla_loader import (
        DEFAULT_REPORT_PATH, load_kb_vanilla_rows,
    )
    from xpu_rt.kernels.kernelblaster_v2 import (
        AgentLoopConfig, KernelBlasterV2,
    )
    from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
        CRiscvEvaluator,
    )
    from xpu_rt.kernels.kernelblaster_v2.generators import KernelGeneratorLLM
    from xpu_rt.kernels.provider import KernelContract
    from xpu_rt.memory import target_knowledge as tk
    from xpu_rt.observability import gemini_usage

    rows: list[CanonicalCellRow] = []

    # Materialise the shape list.
    shapes: list[tuple[int, int, int]] = []
    if workload == "smolvla_matmuls":
        for r in load_kb_vanilla_rows(DEFAULT_REPORT_PATH):
            sh = _parse_matmul_shape_id(r.shape_id)
            if sh is not None:
                shapes.append(sh)
    elif workload == "smolvla_mlp_block":
        # For the MLP-block workload, run KB-v2 on the dominant
        # matmul of one MLP block (gate_proj of action_expert layer 0).
        # Full MEGA-contract live wiring is a follow-up.
        shapes = [(64, 720, 1440)]
    else:
        raise ValueError(f"unknown workload {workload!r}")

    try:
        card = tk.load(target)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "kb-v2 live: failed to load target card for %s (%s); "
            "falling back to a minimal hand-built card",
            target, exc,
        )
        card = _fallback_target_card(target)

    for (M, K, N) in shapes:
        shape_id = shape_id_for_matmul(M, K, N)
        for repeat in range(repeats):
            row = _kb_v2_one_run(
                target=target, workload=workload,
                M=M, K=K, N=N, shape_id=shape_id, repeat=repeat,
                card=card,
            )
            rows.append(row)
    return rows


def _kb_v2_one_run(
    *,
    target: str,
    workload: str,
    M: int, K: int, N: int,
    shape_id: str,
    repeat: int,
    card,
):
    """One (shape, repeat) of KB-v2 live invocation. Returns a
    :class:`CanonicalCellRow`."""
    from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow
    from xpu_rt.benchmarks.loaders.kb_v2_loader import (
        CostSnapshot, load_kb_v2_row,
    )
    from xpu_rt.kernels.kernelblaster_v2 import (
        AgentLoopConfig, KernelBlasterV2,
    )
    from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
        CRiscvEvaluator,
    )
    from xpu_rt.kernels.kernelblaster_v2.generators import KernelGeneratorLLM
    from xpu_rt.kernels.provider import KernelContract
    from xpu_rt.observability import gemini_usage

    contract = KernelContract(
        region_id=f"kb_v2_live_{target}_{M}x{K}x{N}_r{repeat}",
        op_family="matmul",
        input_shapes=((M, K), (K, N)),
        output_shapes=((M, N),),
        dtypes=("i8", "i8", "i32"),
        layout="row_major",
        target_name=target,
    )

    try:
        gemini_usage.check_pre_call(
            projected_cost_usd=0.05,
            source=f"kb-v2.{target}.{workload}.{shape_id}.r{repeat}",
        )
    except Exception as exc:  # noqa: BLE001 — typed budget breach
        logger.warning(
            "kb-v2 live: budget gate aborted %s repeat=%d (%s)",
            shape_id, repeat, exc,
        )
        return _kb_v2_failed_row(
            target=target, workload=workload, shape_id=shape_id,
            repeat=repeat, note=f"budget_gate: {exc}",
        )

    pre = gemini_usage.load_summary()
    pre_t = time.time()
    generator = KernelGeneratorLLM(model="gemini-2.5-flash")
    evaluator = CRiscvEvaluator(
        contract=contract, target_id=target,
        require_gemmini_extension=(target.startswith("gemmini")),
    )
    loop = KernelBlasterV2(
        card=card, generator=generator, evaluator=evaluator,
        config=AgentLoopConfig(max_iterations=4),
    )
    try:
        result = loop.run(contract)
    except Exception as exc:  # noqa: BLE001
        logger.exception("kb-v2 live: agent loop crashed on %s repeat=%d", shape_id, repeat)
        return _kb_v2_failed_row(
            target=target, workload=workload, shape_id=shape_id,
            repeat=repeat,
            note=f"agent_loop_crash: {type(exc).__name__}: {exc}",
        )

    post = gemini_usage.load_summary()
    cost_usd = post.total_cost_usd - pre.total_cost_usd
    cost = CostSnapshot(
        cost_usd=cost_usd,
        tokens_in=post.total_prompt_tokens - pre.total_prompt_tokens,
        tokens_out=post.total_completion_tokens - pre.total_completion_tokens,
        wall_s=time.time() - pre_t,
    )
    return load_kb_v2_row(
        result, target=target, workload=workload,
        shape_id=shape_id, repeat=repeat, cost=cost,
    )


def _kb_v2_failed_row(*, target, workload, shape_id, repeat, note):
    from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow
    return CanonicalCellRow(
        backend="kb-v2", target=target, workload=workload, shape_id=shape_id,
        repeat=repeat, correctness=False, cycles=None, rounds_used=0,
        tokens_in=0, tokens_out=0, cost_usd=0.0, wall_s=0.0,
        cycle_source="none", notes=note,
    )


def _parse_matmul_shape_id(s: str) -> tuple[int, int, int] | None:
    from xpu_rt.benchmarks.canonical_metrics import parse_matmul_shape_id
    return parse_matmul_shape_id(s)


def _fallback_target_card(target: str):
    """Minimal hand-built TargetKnowledgeCard for environments where
    the on-disk card hasn't been seeded yet (Saturn seed needs
    user-side ``saturn_hand_card`` run)."""
    from xpu_rt.memory import target_knowledge as tk

    isa_family = "rocc-systolic" if target.startswith("gemmini") else "riscv-rvv"
    return tk.TargetKnowledgeCard(
        target_id=target,
        target_profile_ref=f"configs/targets/{target}.yaml",
        hardware_spec=tk.HardwareSpec(isa_family=isa_family),
    )


def _kb_v2_full_mode_env_missing(target: str) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMMINI_API"):
        missing.append("GOOGLE_API_KEY (or GEMMINI_API)")
    if not _conda_riscv_tools_present():
        missing.append("riscv-tools conda env (XPU_RT_RISCV_CONDA_ROOT)")
    return missing


def _conda_riscv_tools_present() -> bool:
    p = Path(
        os.environ.get(
            "XPU_RT_RISCV_CONDA_ROOT", "/scratch2/agustin/chipyard/.conda-env/riscv-tools"
        )
    )
    return (p / "bin" / "spike").is_file()


# --- kb-vanilla ---


def _run_kb_vanilla_cell(
    *, target: str, workload: str, mode: str, out_dir: Path
) -> CellResult:
    """KB-vanilla is wired live on Gemmini only.

    The Saturn fork would need a parallel kb_pipeline_driver
    (the 437-line port called out in the plan). That's deferred —
    this runner reports ``status="deferred"`` for Saturn cells and
    points the operator at the prior single-kernel report for
    Gemmini cells (where KB-vanilla has been measured at 7/14
    correct against the same SmolVLA matmuls)."""
    start = time.time()
    cell = CellResult(backend="kb-vanilla", target=target, workload=workload, status="ok")

    if target.lower().startswith("saturn") or target.lower().startswith("opu"):
        cell.status = "deferred"
        cell.notes = (
            "KB-vanilla on Saturn requires a parallel kb_pipeline_driver "
            "fork (~437 lines mirroring kb_gemmini). Scope-narrowed for "
            "this study; KB-v2 is the agentic path being measured."
        )
        cell.wall_s = time.time() - start
        return cell

    # Gemmini path — reference the existing report rather than re-run
    # (we already paid $0.16 for the 14-shape matmul batch).
    if workload == "smolvla_matmuls":
        prior_report = (
            Path("/scratch2/agustin/xpu-rt-integration/results/comparison/vanilla_kb_gemmini/report.md")
        )
        if prior_report.is_file():
            cell.status = "ok"
            cell.notes = (
                f"Reuses the prior single-kernel Gemmini batch at "
                f"{prior_report.relative_to(prior_report.parents[1])} — "
                "7/14 correct, $0.16 Gemini spend."
            )
            cell.correctness_rate = 7.0 / 14.0
            cell.cost_usd = 0.16
            # Pipe the cached report through Phase A's loader so the
            # fair-comparison report has real canonical rows to
            # aggregate — without this the Phase E builder reports
            # "0 cells" even when the cache is fully populated.
            from xpu_rt.benchmarks.canonical_metrics import write_jsonl
            from xpu_rt.benchmarks.loaders.kb_vanilla_loader import load_kb_vanilla_rows

            rows = load_kb_vanilla_rows(prior_report)
            cell_dir = _cell_dir(out_dir, cell.backend, cell.target, cell.workload)
            write_jsonl(rows, cell_dir / "samples.jsonl")
            cell.extra["n_samples_persisted"] = len(rows)
        else:
            cell.status = "error"
            cell.notes = f"Prior Gemmini matmul report not found at {prior_report}"
    elif workload == "smolvla_mlp_block":
        cell.status = "deferred"
        cell.notes = (
            "KB-vanilla on the MLP-block workload requires the multi-op "
            "Spike harness (xpu_rt.kb_gemmini.multiop_harness) wired "
            "into KB-vanilla's pipeline driver — separate from the "
            "single-kernel matmul harness it currently ships with."
        )
    else:
        cell.status = "error"
        cell.notes = f"unknown workload {workload!r}"

    cell.wall_s = time.time() - start
    return cell


# --- autocomp ---


def _run_autocomp_cell(
    *, target: str, workload: str, mode: str, out_dir: Path
) -> CellResult:
    """Run autocomp via :func:`resolve_autocomp_target`.

    Today this only verifies the env is ready (importable backend +
    chipyard env var set) and emits ``status="env_missing"`` /
    ``"deferred"`` accordingly. The live invocation is gated behind
    the same readiness check the live-mode KB v2 path uses."""
    from xpu_rt.kernels.autocomp_adapter import resolve_autocomp_target

    start = time.time()
    cell = CellResult(backend="autocomp", target=target, workload=workload, status="ok")

    bindings = resolve_autocomp_target(target)
    missing = list(bindings.missing_env())
    # Check the autocomp package itself is importable too.
    try:
        bindings.resolve()
    except ImportError as exc:
        missing.append(f"autocomp package import ({exc})")

    if missing:
        cell.status = "env_missing"
        cell.env_missing = tuple(missing)
        cell.notes = (
            f"Autocomp on {target!r} needs: {', '.join(missing)}. See "
            f"third_party/autocomp/{bindings.setup_doc_relpath}."
        )
        cell.wall_s = time.time() - start
        return cell

    # Env ready. In plan mode (default) we still don't spend tokens —
    # report status=deferred so the caveats ledger picks up the row.
    if mode != "full":
        cell.status = "deferred"
        cell.notes = (
            f"Autocomp env is ready (resolved {bindings.agent_class} / "
            f"{bindings.eval_class}); pass --mode full to invoke "
            f"`AutocompAdapter.search_kernel_for_target` live."
        )
        cell.wall_s = time.time() - start
        return cell

    # full mode: run the live invocation, persist canonical rows.
    from xpu_rt.benchmarks.canonical_metrics import write_jsonl
    from xpu_rt.benchmarks.loaders.autocomp_loader import load_autocomp_row
    from xpu_rt.kernels.autocomp_adapter import AutocompAdapter, AutocompEnvMissing

    # Autocomp's existing problem set doesn't have SmolVLA shapes; we
    # pick gemm prob_id=2 (256×256×256) as the default smoke shape so
    # the cell at least exercises the live path. The operator should
    # override (or author SmolVLA harnesses) for a faithful comparison.
    prob_type = "gemm"
    prob_id = 2
    shape_id = f"autocomp_{prob_type}_{prob_id}"
    cell_dir = _cell_dir(out_dir, cell.backend, cell.target, cell.workload)
    autocomp_out = cell_dir / "autocomp_search"
    adapter = AutocompAdapter()
    try:
        adapter.search_kernel_for_target(
            target_id=target,
            prob_type=prob_type,
            prob_id=prob_id,
            output_dir=autocomp_out,
            iterations=6,  # bounded per the budget plan
        )
    except AutocompEnvMissing as exc:
        cell.status = "env_missing"
        cell.env_missing = (str(exc),)
        cell.notes = f"AutocompEnvMissing: {exc}"
        cell.wall_s = time.time() - start
        return cell
    except Exception as exc:  # noqa: BLE001
        cell.status = "error"
        cell.notes = f"{type(exc).__name__}: {exc}"
        cell.wall_s = time.time() - start
        return cell

    row = load_autocomp_row(
        autocomp_out, target=target, workload=workload,
        shape_id=shape_id, repeat=0,
    )
    write_jsonl([row], cell_dir / "samples.jsonl")
    cell.extra["n_samples_persisted"] = 1
    cell.extra["autocomp_prob_type"] = prob_type
    cell.extra["autocomp_prob_id"] = prob_id
    cell.notes = (
        f"Live autocomp run on {prob_type}/test{prob_id}.c "
        f"(SmolVLA-shape harnesses not authored yet — note shape mismatch caveat)."
    )
    cell.wall_s = time.time() - start
    return cell


# ---------------------------------------------------------------------------
# Workload loaders
# ---------------------------------------------------------------------------


def _load_workload_blocks(workload: str) -> list:
    """Materialise the blocks for one workload.

    Loads SmolVLA via the existing loader. The matmuls workload
    is currently re-exposed as a single-block-per-matmul (the
    pipeline driver still walks each block as a 1-node graph),
    keeping the per-cell shape uniform across workloads.
    """
    from xpu_rt.benchmarks.smolvla_block_enumerator import (
        BlockEnumeratorConfig,
        enumerate_blocks,
    )

    # Load SmolVLA. In a real run this is the actual SmolVLA model;
    # the matrix driver runs the same loader path the pipeline
    # driver uses so the per-cell numbers stay comparable.
    try:
        from xpu_rt.benchmarks.smolvla_subset import SubsetSelector

        selector = SubsetSelector(seq_len=64)
        model = selector.load()
    except Exception as exc:  # noqa: BLE001 — loader is heavy, fall back to stub
        logger.warning("SmolVLA loader failed (%s); falling back to MLP stub", exc)
        model = _stub_smolvla()

    if workload == "smolvla_mlp_block":
        cfg = BlockEnumeratorConfig(
            kinds=("mlp",), components=("action_expert",), layer_indices=(0,)
        )
    elif workload == "smolvla_matmuls":
        cfg = BlockEnumeratorConfig(
            kinds=("mlp", "head"),
            components=("action_expert", "action_head"),
            layer_indices=(0, 1, 2, 3),
        )
    else:
        raise ValueError(f"unknown workload {workload!r}")

    return enumerate_blocks(model, cfg)


def _stub_smolvla():
    """Fallback SmolVLA-shaped stub for environments where the real
    loader isn't available (e.g. CI). Mirrors the
    test_run_pipeline_comparison `_SmolVLAStub` shape."""
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(720, 1440, bias=False)
            self.up_proj = nn.Linear(720, 1440, bias=False)
            self.down_proj = nn.Linear(1440, 720, bias=False)

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _MLP()

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm_with_expert = nn.Module()
            self.vlm_with_expert.lm_expert = nn.Module()
            self.vlm_with_expert.lm_expert.layers = nn.ModuleList(_Layer() for _ in range(4))
            self.action_in_proj = nn.Linear(720, 320, bias=False)
            self.action_out_proj = nn.Linear(320, 7, bias=False)

    return _Stub()


# ---------------------------------------------------------------------------
# Top-level matrix driver
# ---------------------------------------------------------------------------


_BACKENDS = {
    "kb-vanilla": _run_kb_vanilla_cell,
    "kb-v2": _run_kb_v2_cell,
    "autocomp": _run_autocomp_cell,
}
_DEFAULT_TARGETS = ("gemmini", "saturn_opu_v128")
_DEFAULT_WORKLOADS = ("smolvla_matmuls", "smolvla_mlp_block")


def run(
    out_dir: Path,
    *,
    backends: tuple[str, ...] = tuple(_BACKENDS.keys()),
    targets: tuple[str, ...] = _DEFAULT_TARGETS,
    workloads: tuple[str, ...] = _DEFAULT_WORKLOADS,
    mode: str = "plan",
    repeats: int = 1,
) -> list[CellResult]:
    # Stash repeats into the module-level slot KB-v2 cells consult.
    global _KB_V2_REPEATS_OVERRIDE
    _KB_V2_REPEATS_OVERRIDE = max(1, int(repeats))
    """Run every (backend, target, workload) cell in the matrix.

    Persists each cell's :class:`CellResult` under
    ``<out_dir>/per_cell/<backend>__<target>__<workload>/status.json``
    and aggregates them via the report writer at the end.

    Args:
        repeats: How many times to run each cell. Only meaningful in
            ``full`` mode — ``plan`` mode is deterministic so
            ``repeats>1`` adds no information. The Phase-D wiring
            iterates ``repeats`` for each cell, persisting one
            :class:`CanonicalCellRow` per repeat into
            ``<cell_dir>/samples.jsonl``. Today's cells emit a
            single ``status="deferred"`` or ``"env_missing"`` row
            and do **not** iterate repeats — once Task 54 lands the
            live wiring the loop fires.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CellResult] = []
    for backend in backends:
        if backend not in _BACKENDS:
            logger.warning("skipping unknown backend %r", backend)
            continue
        runner = _BACKENDS[backend]
        for target in targets:
            for workload in workloads:
                try:
                    cell = runner(target=target, workload=workload, mode=mode, out_dir=out_dir)
                except Exception as exc:  # noqa: BLE001
                    cell = CellResult(
                        backend=backend, target=target, workload=workload,
                        status="error", notes=f"{type(exc).__name__}: {exc}",
                    )
                # Annotate the cell with the requested repeats count
                # so the aggregator can tell at-a-glance how the
                # sample budget was set (this is the *requested* N;
                # the *actual* number of CanonicalCellRows persisted
                # is what samples.jsonl carries).
                cell.extra["requested_repeats"] = repeats
                _persist_cell(out_dir, cell)
                results.append(cell)
                logger.info(
                    "cell %s × %s × %s → %s%s",
                    backend, target, workload, cell.status,
                    f" ({cell.notes[:80]})" if cell.notes else "",
                )

    from xpu_rt.benchmarks.cross_target_report import write_reports
    write_reports(out_dir, results, mode=mode)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/comparison/cross_target"),
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(_BACKENDS.keys()), choices=list(_BACKENDS.keys()),
    )
    parser.add_argument(
        "--targets", nargs="+", default=list(_DEFAULT_TARGETS),
    )
    parser.add_argument(
        "--workloads", nargs="+", default=list(_DEFAULT_WORKLOADS),
        choices=list(_DEFAULT_WORKLOADS),
    )
    parser.add_argument("--mode", choices=("plan", "full"), default="plan")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of N=N statistical repeats per non-cached cell. Only "
        "meaningful in --mode full. Default 1; the plan recommends 3 for the "
        "fair-comparison study so the aggregator can show median ± min/max.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    run(
        args.out_dir,
        backends=tuple(args.backends),
        targets=tuple(args.targets),
        workloads=tuple(args.workloads),
        repeats=args.repeats,
        mode=args.mode,
    )
    print(f"wrote {args.out_dir / 'report.md'}", file=sys.stderr)
    print(f"wrote {args.out_dir / 'report.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CellResult", "main", "run"]
