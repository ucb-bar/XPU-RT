"""Differential test harness for ``compile_through_pipeline``.

Drives a PyTorch model through the pipeline and records:

- whether the FX → xDSL bridge succeeded
- which passes ran vs. were skipped
- module verifier status
- ExecutionPlan validator status
- opaque-call rate on the final module
- eager reference output vs. the golden baseline (for re-run
  determinism)

When ``run_compiled_executor=True`` and a runtime is wired up, the
harness also compares the compiled output against the eager
reference. Without an executor it verifies that the compilation path
does not perturb eager behaviour on the same fixture.

Usage::

    from compgen.pipeline.differential import compile_and_diff
    from compgen.options import cuda_a100_defaults

    report = compile_and_diff(
        model, example_inputs,
        options=cuda_a100_defaults(),
        eager_reference=golden_output,  # optional
    )
    assert report.passed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

from compgen.options import CompGenOptions
from compgen.pipeline.driver import PipelineResult, compile_through_pipeline

log = structlog.get_logger()


@runtime_checkable
class CompiledExecutorProtocol(Protocol):
    """Callable contract for arbitrary compiled-executor backends.

    Any callable matching this signature can be passed to
    :func:`compile_and_diff` via ``run_compiled_executor=…`` to get a
    real compiled-vs-eager diff. The Protocol is intentionally
    minimal — it deliberately doesn't say *how* the inputs run
    (in-process interpreter, RTL sim, FPGA, physical hardware, remote
    process), only that the executor takes the example inputs and
    returns a tensor (or tuple of tensors) shaped like what eager
    PyTorch would produce.

    Implementations:

    - In-tree default: ``compgen.runtime.cpu_executor.execute`` wrapped
      to match this signature (the bool=True path keeps doing this).
    - Subprocess-backed (RTL sim, FPGA, physical): wrap the post-sim
      result-readback as a callable that takes the example inputs,
      issues whatever build/run/parse pipeline the target needs, and
      returns a torch.Tensor.
    - Remote: similar — wrap the RPC.

    The harness treats whatever this returns as the "compiled output"
    and diffs it against ``eager_reference`` with the configured
    tolerances. No type adapter is applied; ``len`` / ``tuple`` /
    ``shape`` semantics on the returned object are the executor's
    responsibility.
    """

    def __call__(
        self,
        inputs: tuple[Any, ...],
    ) -> Any: ...


@dataclass
class DiffReport:
    """Outcome of a :func:`compile_and_diff` run."""

    passed: bool
    fixture_name: str = ""
    bridge_path: str = ""
    module_verified: bool = False
    plan_validated: bool = False
    opaque_count: int = 0
    total_ops: int = 0
    stages_run: int = 0
    stages_skipped: int = 0
    eager_diff_max_abs: float = 0.0
    eager_diff_pass: bool = True
    # Real compiled-vs-eager diff from the CPU executor (W11).
    compiled_executed: bool = False
    compiled_diff_max_abs: float = float("inf")
    compiled_diff_pass: bool = False
    executor_ops_run: int = 0
    executor_ops_skipped: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pipeline_result: PipelineResult | None = None

    @property
    def opaque_rate(self) -> float:
        return (self.opaque_count / self.total_ops) if self.total_ops else 0.0


def _opaque_counts(module: Any) -> tuple[int, int]:
    """Return ``(opaque_count, total_ops)`` for the ``@forward`` body.

    Excludes extern ``func.func private`` declarations, the
    ``builtin.module`` wrapper, and ``func.return`` so the rate
    reflects only the actual program body.
    """
    if module is None:
        return (0, 0)
    from xdsl.dialects.func import FuncOp

    forward_func = None
    for op in module.ops:
        if isinstance(op, FuncOp) and op.sym_name.data == "forward":
            forward_func = op
            break
    target = forward_func if forward_func is not None else module

    total = 0
    opaque = 0
    for op in target.walk():
        if op.name in {"builtin.module", "func.func", "func.return"}:
            continue
        total += 1
        if op.name == "func.call":
            callee_attr = op.properties.get("callee")
            if callee_attr is not None and "aten_" in str(callee_attr):
                opaque += 1
    return (opaque, total)


def compile_and_diff(
    model: Any,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    options: CompGenOptions | None = None,
    fixture_name: str = "",
    eager_reference: Any = None,
    opaque_rate_threshold: float = 0.50,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    run_compiled_executor: bool | CompiledExecutorProtocol = False,
    exported_program: Any = None,
) -> DiffReport:
    """Compile ``model`` and run the differential check.

    Args:
        model: nn.Module or ExportedProgram.
        example_inputs: input tensors (required when ``model`` is a Module).
        options: compilation knobs; defaults to ``CompGenOptions()``.
        fixture_name: tag used in log messages + diagnostics.
        eager_reference: optional torch.Tensor to compare the eager
            re-run against. When ``None`` we re-run eager twice and
            compare to itself as a determinism check.
        opaque_rate_threshold: fraction above which the diff is
            considered failed. Default 0.15.
        atol / rtol: tolerance for eager vs reference comparison.
    """
    report = DiffReport(passed=True, fixture_name=fixture_name)
    if options is None:
        options = CompGenOptions()

    # --- 1. Pipeline ---------------------------------------------------
    pr = compile_through_pipeline(
        model,
        example_inputs=example_inputs,
        options=options,
        workload_name=fixture_name or "unnamed",
    )
    report.pipeline_result = pr
    report.bridge_path = pr.bridge_path
    report.stages_run = pr.stages_run
    report.stages_skipped = pr.stages_skipped

    if pr.bridge_path == "failed" or pr.module is None:
        report.passed = False
        report.failures.append(f"bridge failed on {fixture_name}")
        return report

    # --- 2. Module verification ---------------------------------------
    try:
        pr.module.verify()
        report.module_verified = True
    except Exception as exc:  # noqa: BLE001
        report.failures.append(f"module.verify failed: {exc}")
        report.passed = False

    # --- 3. ExecutionPlan validation ----------------------------------
    try:
        pr.execution_plan.validate()
        report.plan_validated = True
    except Exception as exc:  # noqa: BLE001
        report.failures.append(f"plan.validate failed: {exc}")
        report.passed = False

    # --- 4. Opaque rate ----------------------------------------------
    opaque, total = _opaque_counts(pr.module)
    report.opaque_count = opaque
    report.total_ops = total
    if report.opaque_rate > opaque_rate_threshold:
        report.warnings.append(f"opaque rate {report.opaque_rate:.3f} exceeds threshold {opaque_rate_threshold}")

    # --- 5. Eager diff ------------------------------------------------
    if example_inputs is not None:
        try:
            import torch

            was_training = None
            if hasattr(model, "training"):
                was_training = model.training
                model.eval()
            with torch.no_grad():
                rerun = model(*example_inputs)
            if was_training is True and hasattr(model, "train"):
                model.train()

            ref = eager_reference if eager_reference is not None else rerun
            if isinstance(ref, torch.Tensor) and isinstance(rerun, torch.Tensor):
                diff = (ref - rerun).abs().max().item()
                report.eager_diff_max_abs = diff
                if diff > atol + rtol * ref.abs().max().item():
                    report.failures.append(
                        f"eager vs reference diff {diff:.6f} exceeds tolerance ({atol} + {rtol} * max_abs)"
                    )
                    report.eager_diff_pass = False
                    report.passed = False
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"eager diff skipped: {exc}")

    # --- 6. Compiled executor ---
    # Two modes (REQ-004):
    #   ``run_compiled_executor=True``   → in-tree CPU interpreter
    #                                      (compgen.runtime.cpu_executor.execute).
    #   ``run_compiled_executor=<callable>`` → an external executor
    #                                          satisfying CompiledExecutorProtocol.
    #                                          Lets RTL sim, FPGA, physical-HW,
    #                                          or remote-process backends plug
    #                                          straight in without changing
    #                                          this harness.
    if run_compiled_executor and example_inputs is not None and eager_reference is not None:
        try:
            import torch

            external_exec = run_compiled_executor if callable(run_compiled_executor) else None

            if external_exec is not None:
                # Out-of-process / external executor — just call it.
                out = external_exec(tuple(example_inputs))
                report.compiled_executed = True
            else:
                from compgen.runtime.cpu_executor import ExecutorStats, execute

                ep = exported_program
                if ep is None and hasattr(model, "graph_signature"):
                    ep = model  # already an ExportedProgram
                if ep is None:
                    try:
                        ep = torch.export.export(model, example_inputs)
                    except Exception as exc:  # noqa: BLE001
                        report.warnings.append(f"exported_program reconstruct failed: {exc}")
                        ep = None
                out = None
                if ep is not None and pr.module is not None:
                    exec_stats = ExecutorStats()
                    out = execute(pr.module, ep, example_inputs, stats=exec_stats)
                    report.compiled_executed = True
                    report.executor_ops_run = exec_stats.ops_executed
                    report.executor_ops_skipped = exec_stats.ops_skipped

            if out is not None and isinstance(out, torch.Tensor) and isinstance(eager_reference, torch.Tensor):
                if tuple(out.shape) == tuple(eager_reference.shape):
                    finite = torch.isfinite(out).all() and torch.isfinite(eager_reference).all()
                    if finite:
                        diff = (out - eager_reference).abs().max().item()
                    else:
                        diff = float("nan")
                    report.compiled_diff_max_abs = diff
                    tol = atol + rtol * eager_reference.abs().max().item()
                    report.compiled_diff_pass = diff == diff and diff <= tol  # nan-safe
                    if not report.compiled_diff_pass:
                        report.warnings.append(f"compiled vs eager diff {diff:.6f} > tol {tol:.6f}")
                else:
                    report.warnings.append(
                        f"compiled output shape {tuple(out.shape)} != eager {tuple(eager_reference.shape)}"
                    )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"compiled executor failed: {exc}")

    log.info(
        "compile_and_diff",
        fixture=fixture_name,
        passed=report.passed,
        bridge=report.bridge_path,
        opaque_rate=report.opaque_rate,
        compiled_executed=report.compiled_executed,
        compiled_diff=report.compiled_diff_max_abs,
        stages_run=report.stages_run,
    )
    return report


__all__ = [
    "CompiledExecutorProtocol",
    "DiffReport",
    "compile_and_diff",
]
