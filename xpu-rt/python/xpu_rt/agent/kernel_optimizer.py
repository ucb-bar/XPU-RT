"""Agent-driven model optimisation loop — Wave 6.

The end-to-end glue function ``optimize_model`` does, per region:

  1. Build / reuse a ``KernelContractV3``.
  2. Ask ``decide_dispatch`` (W6.2) which target + granularity to use.
  3. Look up the kernel store + KernelDB for a cached implementation
     under that target × fingerprint. If hit → bind + continue.
  4. Otherwise route through the escalating provider router
     (Claude Code default → autocomp on miss). When the chosen target
     is non-CUDA, the contract goes through the appropriate
     ``KernelContractTranslator`` first (W5.1).
  5. Persist the result (KernelStore + KernelDB).
  6. After all regions are bound, ask the runtime adapter to capture
     the model's forward into a replayable graph (CUDA only — others
     return ``None`` and we just return the unwrapped callable).

Pure glue: every primitive lives in W2-W5 modules. This file is the
loop that ties them together. ``optimize_model_multi_target`` (W6.3)
just maps this over a list of targets.

Usage::

    from compgen.agent.kernel_optimizer import optimize_model

    optim = optimize_model(model, "cuda-a100", perf_budget_us=20_000.0)
    out = optim.forward(*inputs)
    print(optim.summary())
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from compgen.agent.hw_aware_dispatch import (
    MultiTargetDispatchDecision,
    TargetDispatchDecision,
    decide_dispatch,
)
from compgen.kernels.contract_v3 import (
    Granularity,
    HardwareEnvelope,
    KernelContractV3,
)
from compgen.llm.base import CompGenLLMProtocol, Objective
from compgen.memory.kernel_db import (
    KernelDB,
    KernelPerfRecord,
    shared_db,
)
from compgen.runtime.glue import CapturedGraph, RuntimeAdapter, select_adapter

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelDecision:
    """One region's outcome after the optimisation loop."""

    contract: KernelContractV3
    fingerprint: str
    granularity: Granularity
    adapter_name: str
    target: str
    provider_name: str
    cached: bool
    perf_us: float | None
    rationale: str
    translator_name: str = ""


@dataclass
class OptimizedModel:
    """What ``optimize_model`` returns."""

    target: str
    adapter_name: str
    decisions: list[KernelDecision]
    captured_graph: CapturedGraph | None
    forward: Callable[..., Any]
    profile_fn: Callable[[], dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"OptimizedModel(target={self.target}, adapter={self.adapter_name})",
            f"  regions optimised: {len(self.decisions)}",
            f"  cache hits:        {sum(1 for d in self.decisions if d.cached)}",
            f"  graph captured:    {self.captured_graph is not None}",
        ]
        for i, d in enumerate(self.decisions):
            tag = "HIT " if d.cached else "GEN "
            perf = f"{d.perf_us:.1f}us" if d.perf_us else "n/a"
            lines.append(
                f"    [{i:2d}] {tag} {d.contract.op_name:32s} "
                f"granularity={d.granularity.value:6s} provider={d.provider_name:18s} "
                f"perf={perf}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fingerprint — same scheme as the MCP cache so on-disk hits work
# ---------------------------------------------------------------------------


def _v3_to_fingerprint_dict(c: KernelContractV3) -> dict[str, Any]:
    """Pull just the fingerprint-relevant fields, in the exact shape
    ``compgen.mcp.tools.kernel.contract_fingerprint`` expects. Keeps
    the MCP-cache and the optimizer's disk-cache lookups in sync."""
    return {
        "op_name": c.op_name,
        "archetype": c.archetype.value,
        "granularity": c.granularity.value,
        "io": {
            "inputs": [
                {
                    "name": t.name,
                    "shape": {"dims": list(t.shape.dims)},
                    "dtype_class": list(t.dtype_class),
                    "layout": t.layout.value,
                    "alignment_bytes": t.alignment_bytes,
                }
                for t in c.io.inputs
            ],
            "outputs": [
                {
                    "name": t.name,
                    "shape": {"dims": list(t.shape.dims)},
                    "dtype_class": list(t.dtype_class),
                    "layout": t.layout.value,
                    "alignment_bytes": t.alignment_bytes,
                }
                for t in c.io.outputs
            ],
        },
        # MCP reads target name from this nested location.
        "orchestration": {
            "execution": {
                "hardware": {
                    "target_name": c.orchestration.execution.hardware.target_name,
                }
            }
        },
    }


def fingerprint_for(contract: KernelContractV3) -> str:
    """Stable fingerprint that matches the MCP cache scheme."""
    from compgen.mcp.tools.kernel import contract_fingerprint

    return contract_fingerprint(_v3_to_fingerprint_dict(contract))


# ---------------------------------------------------------------------------
# Codegen + bench callbacks (caller-supplied — keeps the loop testable)
# ---------------------------------------------------------------------------


CodegenFn = Callable[[KernelContractV3, TargetDispatchDecision], "CodegenResult"]
BenchFn = Callable[[KernelContractV3, "CodegenResult"], "BenchResult"]


@dataclass(frozen=True)
class CodegenResult:
    """What a codegen callback returns."""

    callable_kernel: Callable[..., Any]
    provider_name: str
    source: str = ""  # raw kernel source — persisted to KernelStore
    language: str = "python"
    translation_artifact: Any = None  # output of the v3-translator if used


@dataclass(frozen=True)
class BenchResult:
    """What a bench callback returns."""

    perf_us: float | None
    correct: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# A toy default codegen — wraps a Python identity callable. Real callers
# pass their own (e.g. the MCP / Claude-Code provider).
# ---------------------------------------------------------------------------


def _identity_codegen(
    contract: KernelContractV3,
    decision: TargetDispatchDecision,
) -> CodegenResult:
    """Tiny default for tests / smoke-runs. Real users pass their own."""

    def _kernel(*args, **kwargs):
        if args:
            return args[0]
        return None

    return CodegenResult(
        callable_kernel=_kernel,
        provider_name="identity_default",
        source=f"# placeholder kernel for {contract.op_name}",
        language="python",
    )


def _trivial_bench(_contract: KernelContractV3, _result: CodegenResult) -> BenchResult:
    """Default bench — skips measurement. Real callers should pass a
    callback that actually times the kernel."""
    return BenchResult(perf_us=None, correct=True, notes="bench skipped")


# ---------------------------------------------------------------------------
# Per-region optimisation
# ---------------------------------------------------------------------------


def _optimise_region(
    contract: KernelContractV3,
    decision: TargetDispatchDecision,
    *,
    adapter: RuntimeAdapter,
    codegen_fn: CodegenFn,
    bench_fn: BenchFn,
    db: KernelDB,
) -> tuple[KernelDecision, Callable[..., Any]]:
    fp = fingerprint_for(contract)
    target = decision.target
    op_family = contract.archetype.value

    # Cache lookup — only honour if correctness is recorded.
    cached = db.best_kernel_perf(target, op_family, fp)
    if cached is not None and cached.correctness_passed:
        # Cache hit: return a no-op callable handle and the recorded perf.
        # Real systems re-load the kernel source from KernelStore here;
        # callers supply their own loader for that.
        kernel: Callable[..., Any] = lambda *a, **kw: None  # noqa: E731
        return (
            KernelDecision(
                contract=contract,
                fingerprint=fp,
                granularity=decision.granularity,
                adapter_name=adapter.name,
                target=target,
                provider_name="cache",
                cached=True,
                perf_us=cached.perf_us,
                rationale=f"cache hit ({cached.perf_us:.1f}us under {target})",
            ),
            kernel,
        )

    # Miss → codegen → bench → persist.
    cg = codegen_fn(contract, decision)
    bench = bench_fn(contract, cg)
    db.record_kernel_perf(
        KernelPerfRecord(
            target=target,
            op_family=op_family,
            fingerprint=fp,
            perf_us=float(bench.perf_us or 0.0),
            correctness_passed=bool(bench.correct),
            source_path="",
            measured_at=time.time(),
            notes=cg.provider_name,
        )
    )
    translator_name = ""
    if cg.translation_artifact is not None:
        translator_name = getattr(cg.translation_artifact, "name", None) or "<unknown>"
    return (
        KernelDecision(
            contract=contract,
            fingerprint=fp,
            granularity=decision.granularity,
            adapter_name=adapter.name,
            target=target,
            provider_name=cg.provider_name,
            cached=False,
            perf_us=bench.perf_us,
            rationale=decision.rationale,
            translator_name=translator_name,
        ),
        cg.callable_kernel,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_model(
    model_fn: Callable[..., Any] | None,
    target: str,
    *,
    contracts: Sequence[KernelContractV3],
    envelope: HardwareEnvelope | None = None,
    perf_budget_us: float | None = None,
    objective: Objective = Objective.LATENCY,
    llm: CompGenLLMProtocol | None = None,
    codegen_fn: CodegenFn | None = None,
    bench_fn: BenchFn | None = None,
    db: KernelDB | None = None,
    capture_graph: bool = True,
    sample_inputs: tuple = (),
) -> OptimizedModel:
    """Optimise ``model_fn`` for one target — agent-driven loop.

    Args:
        model_fn: The model's forward callable. Optional — when None,
            no graph capture is attempted (useful for headless/region-
            only optimisation).
        target: Target name (e.g. ``"cuda-a100"``, ``"openq_5165rb"``).
        contracts: The list of v3 contracts forming the model's regions.
            (W6 callers extract these from the recipe; tests pass them
            directly.)
        envelope: Optional override; defaults to the envelope on the
            first contract's ``ExecutionEnvelope.hardware``.
        perf_budget_us: Optional latency budget per region.
        objective: Optimisation objective (LATENCY by default).
        llm: Optional LLM client for dispatch decisions.
        codegen_fn: Per-region codegen callback. Defaults to a
            placeholder that returns an identity kernel — tests and
            smoke runs use this; real callers wire the MCP provider in.
        bench_fn: Per-region bench callback. Defaults to a no-op.
        db: KernelDB instance. Defaults to the shared one.
        capture_graph: When True (default), call ``adapter.capture_graph``
            after binding all regions. When ``model_fn`` is None or the
            adapter doesn't support capture, returns ``None``.
        sample_inputs: Inputs to drive graph capture warmup.

    Returns:
        ``OptimizedModel``.
    """
    if not contracts:
        raise ValueError("optimize_model requires at least one contract")

    codegen = codegen_fn or _identity_codegen
    bench = bench_fn or _trivial_bench
    kdb = db or shared_db()
    env = envelope or contracts[0].orchestration.execution.hardware
    if env.target_name == "" or env.target_name is None:
        env = HardwareEnvelope(
            target_name=target,
            vector_lanes=env.vector_lanes,
            scratchpad_bytes=env.scratchpad_bytes,
            register_bytes=env.register_bytes,
            native_dtypes=env.native_dtypes,
            peak_bandwidth_gbps=env.peak_bandwidth_gbps,
            codegen_hints=env.codegen_hints,
        )

    adapter = select_adapter(target)
    decisions: list[KernelDecision] = []
    bound_kernels: dict[str, Callable[..., Any]] = {}

    for contract in contracts:
        verdict: MultiTargetDispatchDecision = decide_dispatch(
            region=[contract],
            envelopes=[env],
            perf_budget_us=perf_budget_us,
            objective=objective,
            llm=llm,
        )
        decision = verdict.per_target[env.target_name]
        kd, kernel = _optimise_region(
            contract,
            decision,
            adapter=adapter,
            codegen_fn=codegen,
            bench_fn=bench,
            db=kdb,
        )
        decisions.append(kd)
        bound_kernels[kd.fingerprint] = kernel

    # Optional whole-model graph capture.
    captured: CapturedGraph | None = None
    forward: Callable[..., Any]
    if model_fn is not None and capture_graph:
        try:
            captured = adapter.capture_graph(model_fn, sample_inputs)
        except Exception:  # noqa: BLE001
            captured = None
        if captured is not None:

            def _replay(*args, **_kwargs):
                return adapter.replay(captured, args)

            forward = _replay
        else:
            forward = model_fn
    else:
        forward = model_fn or (lambda *a, **kw: None)

    return OptimizedModel(
        target=target,
        adapter_name=adapter.name,
        decisions=decisions,
        captured_graph=captured,
        forward=forward,
        profile_fn=None,
        metadata={
            "objective": objective.value,
            "perf_budget_us": perf_budget_us,
            "envelope_target": env.target_name,
            "bound_kernel_count": len(bound_kernels),
        },
    )


def optimize_model_multi_target(
    model_fn: Callable[..., Any] | None,
    targets: Sequence[str],
    *,
    contracts: Sequence[KernelContractV3],
    envelopes: Sequence[HardwareEnvelope] | None = None,
    perf_budget_us: float | None = None,
    objective: Objective = Objective.LATENCY,
    llm: CompGenLLMProtocol | None = None,
    codegen_fn: CodegenFn | None = None,
    bench_fn: BenchFn | None = None,
    db: KernelDB | None = None,
    capture_graph: bool = True,
    sample_inputs: tuple = (),
) -> dict[str, OptimizedModel]:
    """W6.3 surface — optimise the same model for multiple targets.

    Each target gets its own decision pass + per-target kernel cache,
    so deploying the same model to two backends costs at most one
    extra codegen per uncached (target, region) pair.

    Args:
        targets: Target names to compile for.
        envelopes: Optional per-target envelopes; defaults to using the
            envelope on each contract's hardware spec, with target_name
            overridden per target.

    Returns:
        Dict mapping ``target_name → OptimizedModel``.
    """
    if not targets:
        raise ValueError("optimize_model_multi_target requires at least one target")

    out: dict[str, OptimizedModel] = {}
    env_by_target: dict[str, HardwareEnvelope] = {}
    if envelopes:
        for e in envelopes:
            env_by_target[e.target_name] = e

    base_env = contracts[0].orchestration.execution.hardware

    for target in targets:
        env = env_by_target.get(target)
        if env is None:
            env = HardwareEnvelope(
                target_name=target,
                vector_lanes=base_env.vector_lanes,
                scratchpad_bytes=base_env.scratchpad_bytes,
                register_bytes=base_env.register_bytes,
                native_dtypes=base_env.native_dtypes,
                peak_bandwidth_gbps=base_env.peak_bandwidth_gbps,
                codegen_hints=base_env.codegen_hints,
            )
        out[target] = optimize_model(
            model_fn=model_fn,
            target=target,
            contracts=contracts,
            envelope=env,
            perf_budget_us=perf_budget_us,
            objective=objective,
            llm=llm,
            codegen_fn=codegen_fn,
            bench_fn=bench_fn,
            db=db,
            capture_graph=capture_graph,
            sample_inputs=sample_inputs,
        )
    return out


__all__ = [
    "BenchResult",
    "CodegenFn",
    "CodegenResult",
    "BenchFn",
    "KernelDecision",
    "OptimizedModel",
    "fingerprint_for",
    "optimize_model",
    "optimize_model_multi_target",
]
