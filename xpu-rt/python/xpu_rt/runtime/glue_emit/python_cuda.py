"""Python CUDA plan executor emitter (M-52).

Phase C M-52: third emitted-glue milestone. When the plan's target
starts with ``cuda`` (or matches the GPU target taxonomy understood
by :func:`compgen.runtime.glue.select_adapter`), emit
``06_glue_emit/generated_plan_executor_cuda.py`` alongside the M-47
SYNC and M-51 ASYNC executors. The CUDA executor:

- Exposes ``compgen_run_cuda(io, kernels, runtime, *, mode,
  capture=False)`` where ``mode in {"sync", "async"}``.
- ``mode="sync"`` runs each region sequentially with
  ``runtime.synchronize()`` between dispatches (a synchronous CUDA
  stream model).
- ``mode="async"`` mirrors M-51's per-region threading + EventTensor
  handshake but routes the dispatch through the
  :class:`CudaRuntimeAdapter` (which fire-and-forgets on the CUDA
  stream and synchronizes at the end).
- ``capture=True`` wraps a single-region plan's region in
  :class:`CudaGraphCaptureWrapper` and emits ``captured_graph``
  metadata into the plan-executor manifest.

The emitter is GPU-host-conditional in the sense that the *runtime*
side requires CUDA: the emitted code is plain Python and parses /
imports cleanly on a CPU-only host. Tests verify emit + import + the
non-GPU code paths (mode="sync" without GPU is rejected by the
adapter, not by the emitter); the actual GPU dispatch path is
exercised by tests marked ``requires_gpu`` so a CPU-only CI run does
not silently pass.

Hard rules:

- Emitter only fires when ``plan.target`` matches the CUDA family.
  Non-CUDA targets skip the emit (``overall=skipped``); the M-47 SYNC
  and (when applicable) M-51 ASYNC executors are the artifacts.
- The captured_graph payload is opt-in (``capture=True``); the default
  is plain dispatch so a CPU-only test can import and call
  ``compgen_run_cuda(mode="sync")`` against a stub adapter that
  emulates ``runtime.dispatch`` / ``runtime.synchronize`` without a
  GPU.
- The CUDA emitter REUSES M-48's typed PLAN_VIOLATION classes and
  M-51's per-region EventTensor handshake template — duplication is
  forbidden by the realness contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.runtime.execution_plan import ExecutionPlan


_GLUE_EMIT_CUDA_SCHEMA_VERSION = "plan_executor_cuda_manifest_v1"
_DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class CudaGlueEmitResult:
    out_dir: Path
    executor_path: Path
    manifest_path: Path
    overall: str  # "pass" | "skipped"
    cuda_target: str
    skipped_reason: str = ""


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_yaml_or_json_plan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
            return yaml.safe_load(text) or {}
        except ImportError:
            return json.loads(text)
    return json.loads(text)


def _topological_region_order(plan: ExecutionPlan) -> list[str]:
    region_ids = [rp.region_id for rp in plan.region_placement]
    if not plan.dependency_edges:
        return region_ids
    in_degree = {r: 0 for r in region_ids}
    edges_by_from: dict[str, list[str]] = {r: [] for r in region_ids}
    for edge in plan.dependency_edges:
        if edge.from_region not in in_degree or edge.to_region not in in_degree:
            continue
        in_degree[edge.to_region] += 1
        edges_by_from[edge.from_region].append(edge.to_region)
    queue = [r for r in region_ids if in_degree[r] == 0]
    out: list[str] = []
    while queue:
        queue.sort()
        r = queue.pop(0)
        out.append(r)
        for n in edges_by_from.get(r, []):
            in_degree[n] -= 1
            if in_degree[n] == 0:
                queue.append(n)
    if len(out) != len(region_ids):
        return region_ids
    return out


def _is_cuda_target(target: str) -> bool:
    """Mirror :func:`compgen.runtime.glue.select_adapter`'s CUDA
    classification — a single source of truth would be ideal but the
    factory takes a string and returns an instance; here we need the
    decision before we build the adapter."""
    n = (target or "").lower()
    return n.startswith("cuda") or "titan-rtx" in n or "test-gpu-simt" in n


def _event_name_for(region_id: str) -> str:
    return f"{region_id}_done"


def _consumer_count_per_event(plan: ExecutionPlan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in plan.dependency_edges:
        ev = _event_name_for(edge.from_region)
        counts[ev] = counts.get(ev, 0) + 1
    return counts


def _emit_cuda_executor_source(
    *,
    plan: ExecutionPlan,
    plan_path: str,
    region_assertions_body: str,
) -> str:
    bindings_by_region = {b.region_id: b for b in plan.region_kernel_bindings}
    region_order = _topological_region_order(plan)
    consumer_counts = _consumer_count_per_event(plan)
    deps_in: dict[str, list[str]] = {r: [] for r in region_order}
    for edge in plan.dependency_edges:
        if edge.to_region in deps_in and edge.from_region in bindings_by_region:
            deps_in[edge.to_region].append(_event_name_for(edge.from_region))

    bindings_block_json = json.dumps(
        {b.region_id: {
            "contract_hash": b.contract_hash,
            "certificate_path": b.certificate_path,
            "kernel_artifact": b.kernel_artifact,
            "dispatch_model": b.dispatch_model,
        } for b in plan.region_kernel_bindings},
        indent=4, sort_keys=True,
    )
    event_specs = []
    for region_id in region_order:
        if region_id not in bindings_by_region:
            continue
        ev = _event_name_for(region_id)
        wcount = max(1, consumer_counts.get(ev, 0))
        event_specs.append((ev, wcount))
    event_specs_repr = "{" + ", ".join(
        f"{name!r}: {{'wait_count_default': {wc}}}"
        for name, wc in event_specs
    ) + "}"
    terminal_region = None
    for region_id in reversed(region_order):
        if region_id in bindings_by_region:
            terminal_region = region_id
            break

    # Bound region dispatch lines for SYNC mode.
    sync_lines: list[str] = []
    for region_id in region_order:
        binding = bindings_by_region.get(region_id)
        if binding is None:
            sync_lines.append(
                f"    # {region_id}: UNBOUND — refused at assert_plan time."
            )
            continue
        sync_lines.append(
            f"    # Region {region_id!r} — CUDA {binding.dispatch_model.upper()} dispatch"
        )
        sync_lines.append(
            f"    out_{region_id} = runtime.dispatch("
        )
        sync_lines.append(f"        contract={region_id!r},")
        sync_lines.append(f"        callable_kernel=kernels[{region_id!r}],")
        sync_lines.append(f"        args=tuple(io.values()),")
        sync_lines.append("        kwargs={},")
        sync_lines.append("    )")
        sync_lines.append("    runtime.synchronize()")
        sync_lines.append(f"    last_out = out_{region_id}.output")
    if not sync_lines:
        sync_lines.append("    last_out = ()  # no regions to dispatch")

    # ASYNC worker bodies (mirrors M-51).
    async_workers: list[str] = []
    for region_id in region_order:
        binding = bindings_by_region.get(region_id)
        if binding is None:
            async_workers.append(f"    # {region_id}: UNBOUND.")
            continue
        wait_list = deps_in.get(region_id, [])
        ev_name = _event_name_for(region_id)
        async_workers.append(f"    def _worker_{region_id}():")
        async_workers.append("        try:")
        for w in wait_list:
            async_workers.append(
                f"            events[{w!r}].wait((0,), timeout_s=timeout_s)"
            )
        async_workers.append(
            f"            _disp = runtime.dispatch("
        )
        async_workers.append(f"                contract={region_id!r},")
        async_workers.append(
            f"                callable_kernel=kernels[{region_id!r}],"
        )
        async_workers.append(f"                args=tuple(io.values()),")
        async_workers.append("                kwargs={},")
        async_workers.append("            )")
        async_workers.append(f"            results[{region_id!r}] = _disp.output")
        async_workers.append(f"            events[{ev_name!r}].notify((0,))")
        async_workers.append("        except BaseException as exc:")
        async_workers.append(f"            errors[{region_id!r}] = exc")
        async_workers.append("            for _ev in events.values():")
        async_workers.append("                _ev._cancel(exc)")
        async_workers.append(
            f"            events[{ev_name!r}].notify((0,))  # release waiters"
        )
        async_workers.append("")

    from compgen.runtime.glue_emit.plan_assertions import (
        render_plan_violation_classes,
    )
    plan_violation_classes = render_plan_violation_classes()

    workers_block = "\n".join(async_workers)
    workers_dict = "\n".join(
        f"        {b.region_id!r}: _worker_{b.region_id},"
        for b in plan.region_kernel_bindings
    )

    return f'''"""Auto-generated by M-52 (compgen.runtime.glue_emit.python_cuda).

Workload: {plan.workload}
Target  : {plan.target}
Source  : {plan_path}

DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.

The executor exposes ``compgen_run_cuda(io, kernels, runtime, *,
mode, capture=False, timeout_s=30.0)``.

Modes:

- ``mode="sync"``: each region dispatches sequentially with
  ``runtime.synchronize()`` between launches.
- ``mode="async"``: per-region threading + EventTensor handshake
  (mirrors M-51's CPU async pattern but routed through
  CudaRuntimeAdapter).

When ``capture=True`` and the plan has exactly one bound region, the
region is wrapped in :class:`CudaGraphCaptureWrapper` and a
``captured_graph`` payload is recorded on the returned dict. Multi-
region capture is honestly not supported in this milestone (the
graph-capture wrapper expects a single forward function).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.glue import (
    CudaRuntimeAdapter,
    RuntimeAdapter,
    select_adapter,
)


PLAN_WORKLOAD = {plan.workload!r}
PLAN_TARGET = {plan.target!r}
PLAN_REGION_ORDER = {region_order!r}
KERNEL_BINDINGS = {bindings_block_json}
EVENT_SPECS = {event_specs_repr}
TERMINAL_REGION = {terminal_region!r}


# M-48 typed PlanViolation classes (one per check kind).
{plan_violation_classes}


def assert_plan(io):
    if not isinstance(io, dict):
        raise PLAN_VIOLATION_IO_TYPE(
            f"expected dict, got {{type(io).__name__}}"
        )
    for region_id in PLAN_REGION_ORDER:
        if region_id in KERNEL_BINDINGS:
            continue
        raise PLAN_VIOLATION_UNBOUND_REGION(
            f"region {{region_id!r}} has no certified kernel binding; "
            f"M-46 emitted the plan with "
            f"bound_count={{len(KERNEL_BINDINGS)}} of "
            f"{{len(PLAN_REGION_ORDER)}} regions; refusing to dispatch"
        )

{region_assertions_body}


def _run_sync(io, kernels, runtime):
{chr(10).join(sync_lines)}
    return last_out


def _run_async(io, kernels, runtime, *, timeout_s):
    events = {{
        name: EventTensor(
            shape=(1,),
            wait_count_default=spec["wait_count_default"],
            sym_name=name,
        )
        for name, spec in EVENT_SPECS.items()
    }}
    results: dict[str, Any] = {{}}
    errors: dict[str, BaseException] = {{}}

{workers_block}

    _WORKERS = {{
{workers_dict}
    }}

    threads = []
    for region_id in PLAN_REGION_ORDER:
        if region_id not in KERNEL_BINDINGS:
            continue
        worker = _WORKERS[region_id]
        t = threading.Thread(
            target=worker, name=f'compgen-cuda-async-{{region_id}}',
        )
        threads.append(t)
        t.start()
    deadline = time.monotonic() + timeout_s
    for t in threads:
        remaining = max(0.0, deadline - time.monotonic())
        t.join(timeout=remaining)
        if t.is_alive():
            for _ev in events.values():
                _ev._cancel(TimeoutError(t.name))
            for tt in threads:
                tt.join(timeout=0.5)
            for _ev in events.values():
                _ev._uncancel()
            raise TimeoutError(
                f'compgen_run_cuda(mode="async") timed out after '
                f'{{timeout_s}}s; worker {{t.name}} still running'
            )
    for _ev in events.values():
        _ev._uncancel()
    if errors:
        first = next(iter(errors.items()))
        raise first[1]
    return results.get(TERMINAL_REGION, ())


def compgen_run_cuda(
    io: dict[str, Any],
    kernels: dict[str, Callable[..., Any]],
    runtime: RuntimeAdapter | None = None,
    *,
    mode: str = "sync",
    capture: bool = False,
    timeout_s: float = {_DEFAULT_TIMEOUT_S!r},
) -> Any:
    """Per-workload emitted CUDA plan executor (M-52).

    Args:
        io: model inputs keyed by name.
        kernels: dict mapping region_id to a callable the runtime
            adapter dispatches.
        runtime: a compgen.runtime.glue.RuntimeAdapter. If None,
            ``select_adapter(PLAN_TARGET)`` returns a CudaRuntimeAdapter.
        mode: "sync" (default) or "async".
        capture: when True and the plan has one bound region, wrap
            that region's dispatch via CudaGraphCaptureWrapper and
            return ``{{"output": <out>, "captured_graph": <handle>}}``.
            For multi-region plans this raises ValueError honestly
            instead of silently degrading.
        timeout_s: per-launch deadline for the async path.

    Returns:
        The terminal region's output (or capture payload when
        capture=True).
    """
    if runtime is None:
        runtime = select_adapter(PLAN_TARGET)
    if mode not in ("sync", "async"):
        raise ValueError(
            f"compgen_run_cuda: mode must be 'sync' or 'async', got {{mode!r}}"
        )
    assert_plan(io)

    if capture:
        bound = [r for r in PLAN_REGION_ORDER if r in KERNEL_BINDINGS]
        if len(bound) != 1:
            raise ValueError(
                f"compgen_run_cuda(capture=True) requires exactly one "
                f"bound region; this plan has {{len(bound)}}. M-52 does "
                f"not support multi-region graph capture honestly."
            )
        sample_inputs = tuple(io.values())
        captured = runtime.capture_graph(
            model_fn=kernels[bound[0]],
            sample_inputs=sample_inputs,
        )
        if captured is None:
            # GPU not available; honestly report unavailable.
            return {{
                "output": runtime.dispatch(
                    contract=bound[0],
                    callable_kernel=kernels[bound[0]],
                    args=sample_inputs, kwargs={{}},
                ).output,
                "captured_graph": None,
                "capture_status": "unavailable_no_cuda",
            }}
        out = runtime.replay(captured, sample_inputs)
        return {{
            "output": out, "captured_graph": captured,
            "capture_status": "captured",
        }}

    if mode == "sync":
        out = _run_sync(io, kernels, runtime)
    else:
        out = _run_async(io, kernels, runtime, timeout_s=timeout_s)
    runtime.synchronize()
    return out


def main() -> None:
    print(
        f"compgen_run_cuda generated by M-52 for "
        f"workload={{PLAN_WORKLOAD!r}} target={{PLAN_TARGET!r}}; "
        f"{{len(KERNEL_BINDINGS)}} bound region(s) over "
        f"{{len(PLAN_REGION_ORDER)}} planned region(s)."
    )


if __name__ == "__main__":
    main()
'''


def emit_python_cuda_executor(run_dir: Path) -> CudaGlueEmitResult:
    """Read the M-46 plan; emit the CUDA executor when the target is
    in the CUDA family. Non-CUDA targets skip emission (overall=skipped)."""
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "05_execution_plan" / "execution_plan.yaml"
    if not plan_path.exists():
        plan_path = run_dir / "05_execution_plan" / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"M-46 execution plan not found at "
            f"{run_dir / '05_execution_plan'}; run --stop-after "
            f"execution-plan-emit first"
        )
    plan_dict = _read_yaml_or_json_plan(plan_path)
    plan = ExecutionPlan.from_dict(plan_dict)

    out_dir = run_dir / "06_glue_emit"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "plan_executor_cuda_manifest.json"

    if not _is_cuda_target(plan.target):
        manifest_path.write_text(
            json.dumps({
                "schema_version": _GLUE_EMIT_CUDA_SCHEMA_VERSION,
                "generated_at_utc": _utcnow(),
                "workload": plan.workload,
                "target": plan.target,
                "executor_kind": "python_cuda",
                "overall": "skipped",
                "skipped_reason": (
                    f"target {plan.target!r} is not CUDA-class; M-52 "
                    f"emit applies only to targets matching the CUDA "
                    f"family (cuda*, titan-rtx, test-gpu-simt)"
                ),
                "cuda_target": "",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return CudaGlueEmitResult(
            out_dir=out_dir,
            executor_path=Path("/dev/null"),
            manifest_path=manifest_path,
            overall="skipped",
            cuda_target="",
            skipped_reason=(
                f"target {plan.target!r} is not CUDA-class"
            ),
        )

    from compgen.runtime.glue_emit.plan_assertions import (
        collect_region_assertions,
        render_assert_plan_body,
    )
    bindings_dicts = [
        {
            "region_id": b.region_id, "status": "bound",
            "contract_hash": b.contract_hash,
            "certificate_path": b.certificate_path,
        }
        for b in plan.region_kernel_bindings
    ]
    region_assertions = collect_region_assertions(
        run_dir=run_dir, bindings=bindings_dicts,
    )
    region_assertions_body = render_assert_plan_body(region_assertions)

    executor_path = out_dir / "generated_plan_executor_cuda.py"
    executor_path.write_text(
        _emit_cuda_executor_source(
            plan=plan,
            plan_path=str(plan_path.relative_to(run_dir)),
            region_assertions_body=region_assertions_body,
        ),
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps({
            "schema_version": _GLUE_EMIT_CUDA_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "workload": plan.workload,
            "target": plan.target,
            "executor_kind": "python_cuda",
            "executor_path": str(executor_path.relative_to(run_dir)),
            "source_plan_path": str(plan_path.relative_to(run_dir)),
            "overall": "pass",
            "cuda_target": plan.target,
            "supports_modes": ["sync", "async"],
            "supports_capture": True,
            "default_timeout_s": _DEFAULT_TIMEOUT_S,
            "kernel_bindings": [
                {
                    "region_id": b.region_id,
                    "contract_hash": b.contract_hash,
                    "certificate_path": b.certificate_path,
                    "kernel_artifact": b.kernel_artifact,
                    "dispatch_model": b.dispatch_model,
                }
                for b in plan.region_kernel_bindings
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return CudaGlueEmitResult(
        out_dir=out_dir,
        executor_path=executor_path,
        manifest_path=manifest_path,
        overall="pass",
        cuda_target=plan.target,
    )
