"""Python ASYNC plan executor emitter.

Phase C second emitted-glue milestone. When the plan's bindings
include any region with ``dispatch_model="async"``, emit
``06_glue_emit/generated_plan_executor_async.py`` alongside the
SYNC executor. The async executor allocates one
:class:`compgen.runtime.event_tensor.EventTensor` per producer region
(named after the region's ``event_decls``), runs each region in its
own thread, and wires producer/consumer pairs through the
``dependency_edges`` from the plan.

Generated shape:

::

    def compgen_run_async(io, kernels, runtime, *, timeout_s=30.0):
        assert_plan(io)
        events = {
            "matmul_0_done": EventTensor(shape=(1,), wait_count_default=N0, ...),
            ...
        }
        results = {}

        def _worker_<region_id>():
            for w in WAIT_ON[<region_id>]:
                events[w].wait((0,), timeout_s=timeout_s)
            out = runtime.dispatch(...)
            results[<region_id>] = out.output
            events["<region_id>_done"].notify((0,))

        threads = [threading.Thread(target=_worker_<r>, ...) for r in REGION_ORDER]
        for t in threads: t.start()
        for t in threads: t.join(timeout_s)
        runtime.synchronize()
        return results[<terminal_region_id>]

The ASYNC executor:

- Produces the SAME output as the SYNC executor on dependency-free or
  K_iters=1 plans. The dependency graph determines the wait/notify
  edges; on a chain, this is equivalent to SYNC topological order.
- Times out deterministically when a producer fails to ``notify`` its
  completion event within ``timeout_s`` (default 30s).
- The static event-writer assertion (every event name has exactly one
  declaring region) lives 's ``assert_plan``; this emitter does
  not bypass it.

Hard rules:

- Emitter only fires when at least one binding has
  ``dispatch_model == "async"``. SYNC-only plans skip the emit (the
  executor is the only one written).
- Worker threads catch the kernel exception, mark the EventTensor
  cancelled (so siblings stop waiting), and re-raise to the main thread
  via the standard ``persistent_launch`` cancellation primitive. We
  reuse :class:`EventTensor._cancel` rather than re-implementing
  cross-thread cancellation.
- ``runtime.synchronize()`` is called once at the end, after all
  regions notify, so the SYNC adapter's no-op contract is preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.runtime.execution_plan import ExecutionPlan


_GLUE_EMIT_ASYNC_SCHEMA_VERSION = "plan_executor_async_manifest_v1"
_DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class AsyncGlueEmitResult:
    out_dir: Path
    executor_path: Path
    manifest_path: Path
    overall: str  # "pass" | "skipped"
    async_regions: tuple[str, ...]
    sync_regions: tuple[str, ...]


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


def _event_name_for(region_id: str) -> str:
    """Convention: each producer region's completion event is
    ``<region_id>_done``. This matches the default
    ``EventDecl(name="matmul_done")`` widened across regions and lets
    the static event-writer check confirm uniqueness.
    """
    return f"{region_id}_done"


def _load_overlap_schedule(run_dir: Path) -> dict[str, int] | None:
    """Read ``05_execution_plan/overlap_schedule.solved.json`` if present.

    Returns ``{region_id: start_tick}`` or ``None`` when no solved
    schedule exists. wires the overlap planner's output into the
    ASYNC spawn order so threads start in the planner's preferred
    sequence; EventTensor dependencies still enforce correctness, but
    schedule-ordered spawn avoids OS-scheduler roulette on independent
    producers.
    """

    candidate = run_dir / "05_execution_plan" / "overlap_schedule.solved.json"
    if not candidate.is_file():
        return None
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    sched = body.get("schedule")
    if not isinstance(sched, list):
        return None
    return {
        str(entry["op_id"]): int(entry["start_tick"])
        for entry in sched
        if isinstance(entry, dict) and "op_id" in entry and "start_tick" in entry
    }


def _apply_overlap_schedule(
    region_order: list[str], schedule: dict[str, int] | None
) -> list[str]:
    """Re-order ``region_order`` by the solved schedule's start_tick.

    Regions not in the schedule keep their topo-order position
    (appended after the scheduled ones). Stable across ties.
    """

    if not schedule:
        return region_order
    in_schedule = [r for r in region_order if r in schedule]
    not_in_schedule = [r for r in region_order if r not in schedule]
    in_schedule.sort(key=lambda r: (schedule[r], region_order.index(r)))
    return in_schedule + not_in_schedule


def _consumer_count_per_event(plan: ExecutionPlan) -> dict[str, int]:
    """Count how many regions consume each producer's completion
    event. ``wait_count_default`` is set to ``max(1, consumer_count)``
    so a producer with no consumers (terminal region) still notifies
    once and any waiter on it (debug introspection) gets through."""
    counts: dict[str, int] = {}
    for edge in plan.dependency_edges:
        ev = _event_name_for(edge.from_region)
        counts[ev] = counts.get(ev, 0) + 1
    return counts


def _collect_async_bindings(plan: ExecutionPlan) -> list[Any]:
    return [b for b in plan.region_kernel_bindings if b.dispatch_model == "async"]


def _emit_async_executor_source(
    *,
    plan: ExecutionPlan,
    plan_path: str,
    region_assertions_body: str,
    overlap_schedule: dict[str, int] | None = None,
) -> str:
    bindings_by_region = {b.region_id: b for b in plan.region_kernel_bindings}
    region_order = _topological_region_order(plan)
    # wire-up: when an ``overlap_schedule.solved.json`` is present,
    # threads spawn in the planner's preferred order. Dependencies are
    # still enforced by EventTensor wait/notify — this only orders the
    # initial spawn, never the wait edges.
    region_order = _apply_overlap_schedule(region_order, overlap_schedule)
    consumer_counts = _consumer_count_per_event(plan)
    deps_in: dict[str, list[str]] = {r: [] for r in region_order}
    for edge in plan.dependency_edges:
        if edge.to_region in deps_in and edge.from_region in bindings_by_region:
            deps_in[edge.to_region].append(_event_name_for(edge.from_region))

    # Build per-region worker bodies.
    worker_blocks: list[str] = []
    bound_meta: list[dict[str, Any]] = []
    for region_id in region_order:
        binding = bindings_by_region.get(region_id)
        if binding is None:
            worker_blocks.append(
                f"    # {region_id}: UNBOUND — refused at assert_plan(io) time."
            )
            continue
        bound_meta.append({
            "region_id": region_id,
            "contract_hash": binding.contract_hash,
            "certificate_path": binding.certificate_path,
            "kernel_artifact": binding.kernel_artifact,
            "dispatch_model": binding.dispatch_model,
        })
        wait_list = deps_in.get(region_id, [])
        ev_name = _event_name_for(region_id)
        worker_blocks.append(f"    def _worker_{region_id}():")
        worker_blocks.append(f"        try:")
        for w in wait_list:
            worker_blocks.append(
                f"            events[{w!r}].wait((0,), timeout_s=timeout_s)"
            )
        worker_blocks.append(
            f"            _disp = runtime.dispatch("
        )
        worker_blocks.append(
            f"                contract={region_id!r},"
        )
        worker_blocks.append(
            f"                callable_kernel=kernels[{region_id!r}],"
        )
        worker_blocks.append(
            f"                args=tuple(io.values()),"
        )
        worker_blocks.append("                kwargs={},")
        worker_blocks.append("            )")
        worker_blocks.append(f"            results[{region_id!r}] = _disp.output")
        worker_blocks.append(f"            events[{ev_name!r}].notify((0,))")
        worker_blocks.append("        except BaseException as exc:")
        worker_blocks.append(f"            errors[{region_id!r}] = exc")
        worker_blocks.append("            for _ev in events.values():")
        worker_blocks.append("                _ev._cancel(exc)")
        worker_blocks.append(
            f"            events[{ev_name!r}].notify((0,))  # release waiters"
        )
        worker_blocks.append("")

    # Bindings block for top-level metadata.
    bindings_block_json = json.dumps(
        {b.region_id: {
            "contract_hash": b.contract_hash,
            "certificate_path": b.certificate_path,
            "kernel_artifact": b.kernel_artifact,
            "dispatch_model": b.dispatch_model,
        } for b in plan.region_kernel_bindings},
        indent=4, sort_keys=True,
    )

    # Event spec block.
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

    # Terminal region — last bound region in topo order.
    terminal_region = None
    for region_id in reversed(region_order):
        if region_id in bindings_by_region:
            terminal_region = region_id
            break

    from compgen.runtime.glue_emit.plan_assertions import (
        render_plan_violation_classes,
    )
    plan_violation_classes = render_plan_violation_classes()

    spawn_block = "\n".join(
        [
            "    threads = []",
            "    for region_id in PLAN_REGION_ORDER:",
            "        if region_id not in KERNEL_BINDINGS:",
            "            continue",
            "        worker = _WORKERS[region_id]",
            "        t = threading.Thread(target=worker, name=f'compgen-async-{region_id}')",
            "        threads.append(t)",
            "        t.start()",
            "    deadline = time.monotonic() + timeout_s",
            "    for t in threads:",
            "        remaining = max(0.0, deadline - time.monotonic())",
            "        t.join(timeout=remaining)",
            "        if t.is_alive():",
            "            for _ev in events.values():",
            "                _ev._cancel(TimeoutError(t.name))",
            "            for tt in threads:",
            "                tt.join(timeout=0.5)",
            "            for _ev in events.values():",
            "                _ev._uncancel()",
            "            raise TimeoutError(",
            "                f'compgen_run_async timed out after {timeout_s}s; '",
            "                f'worker {t.name} still running'",
            "            )",
            "    for _ev in events.values():",
            "        _ev._uncancel()",
            "    if errors:",
            "        # First worker to raise wins.",
            "        first = next(iter(errors.items()))",
            "        raise first[1]",
        ]
    )

    return f'''"""Auto-generated by M-51 (compgen.runtime.glue_emit.python_async).

Workload: {plan.workload}
Target  : {plan.target}
Source  : {plan_path}

DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.

The executor exposes ``compgen_run_async(io, kernels, runtime, *,
timeout_s=30.0)`` per the M-51 protocol. Each region runs in its own
thread; producer→consumer dependencies are wired through EventTensor
notify/wait pairs. The typed PLAN_VIOLATION_<KIND> classes apply
unchanged. Output matches the M-47 SYNC executor on K_iters=1 plans.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.glue import (
    CpuRuntimeAdapter,
    RuntimeAdapter,
    select_adapter,
)


PLAN_WORKLOAD = {plan.workload!r}
PLAN_TARGET = {plan.target!r}
PLAN_REGION_ORDER = {region_order!r}
KERNEL_BINDINGS = {bindings_block_json}
EVENT_SPECS = {event_specs_repr}
TERMINAL_REGION = {terminal_region!r}


# Typed PlanViolation classes (one per check kind).
{plan_violation_classes}


def assert_plan(io):
    """Plan invariants — generated from contract fields.

    Each check fires a typed PLAN_VIOLATION_<KIND> subclass naming
    the failed invariant. The unbound-region check (M-46 carryover)
    fires before any per-input check.
    """
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


def compgen_run_async(
    io: dict[str, Any],
    kernels: dict[str, Callable[..., Any]],
    runtime: RuntimeAdapter | None = None,
    *,
    timeout_s: float = {_DEFAULT_TIMEOUT_S!r},
) -> Any:
    """Per-workload emitted ASYNC plan executor (M-51).

    Args:
        io: model inputs keyed by name.
        kernels: dict mapping region_id to a callable the runtime
            adapter dispatches.
        runtime: a compgen.runtime.glue.RuntimeAdapter. If None,
            ``select_adapter(PLAN_TARGET)`` picks the right one.
        timeout_s: per-launch deadline (default 30s). EventTensor
            ``wait`` and per-thread ``join`` both honour this — a
            missing notify deterministically times out instead of
            hanging.

    Returns:
        The terminal region's output. Mirrors the M-47 SYNC executor's
        ``last_out`` semantics.
    """
    if runtime is None:
        runtime = select_adapter(PLAN_TARGET)
    assert_plan(io)

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

{chr(10).join(worker_blocks)}

    _WORKERS = {{
{chr(10).join(f"        {b.region_id!r}: _worker_{b.region_id}," for b in plan.region_kernel_bindings)}
    }}

{spawn_block}

    runtime.synchronize()
    return results.get(TERMINAL_REGION, ())


def main() -> None:
    print(
        f"compgen_run_async generated by M-51 for "
        f"workload={{PLAN_WORKLOAD!r}} target={{PLAN_TARGET!r}}; "
        f"{{len(KERNEL_BINDINGS)}} async-bound region(s) over "
        f"{{len(PLAN_REGION_ORDER)}} planned region(s)."
    )


if __name__ == "__main__":
    main()
'''


def emit_python_async_executor(run_dir: Path) -> AsyncGlueEmitResult:
    """Read the plan from disk; emit the ASYNC executor when at
    least one binding has ``dispatch_model == "async"``. SYNC-only
    plans skip the emit (overall="skipped"); the SYNC file is the
    only artifact in that case.
    """
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

    async_bindings = _collect_async_bindings(plan)
    sync_bindings = [
        b for b in plan.region_kernel_bindings if b.dispatch_model == "sync"
    ]
    out_dir = run_dir / "06_glue_emit"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "plan_executor_async_manifest.json"

    if not async_bindings:
        manifest_path.write_text(
            json.dumps({
                "schema_version": _GLUE_EMIT_ASYNC_SCHEMA_VERSION,
                "generated_at_utc": _utcnow(),
                "workload": plan.workload,
                "target": plan.target,
                "executor_kind": "python_async",
                "overall": "skipped",
                "skipped_reason": (
                    "no binding has dispatch_model=async; M-47 SYNC "
                    "executor is the only artifact for this plan"
                ),
                "async_regions": [],
                "sync_regions": [b.region_id for b in sync_bindings],
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return AsyncGlueEmitResult(
            out_dir=out_dir,
            executor_path=Path("/dev/null"),
            manifest_path=manifest_path,
            overall="skipped",
            async_regions=(),
            sync_regions=tuple(b.region_id for b in sync_bindings),
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

    overlap_schedule = _load_overlap_schedule(run_dir)
    executor_path = out_dir / "generated_plan_executor_async.py"
    executor_path.write_text(
        _emit_async_executor_source(
            plan=plan,
            plan_path=str(plan_path.relative_to(run_dir)),
            region_assertions_body=region_assertions_body,
            overlap_schedule=overlap_schedule,
        ),
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps({
            "schema_version": _GLUE_EMIT_ASYNC_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "workload": plan.workload,
            "target": plan.target,
            "executor_kind": "python_async",
            "executor_path": str(executor_path.relative_to(run_dir)),
            "source_plan_path": str(plan_path.relative_to(run_dir)),
            "overall": "pass",
            "async_regions": [b.region_id for b in async_bindings],
            "sync_regions": [b.region_id for b in sync_bindings],
            "default_timeout_s": _DEFAULT_TIMEOUT_S,
            # wire-up evidence: when the overlap planner has run
            # and persisted a solved schedule, record that the emit
            # consumed it. ``solver_schedule_consumed=false`` is the
            # honest default for runs without a solver step.
            "solver_schedule_consumed": overlap_schedule is not None,
            "solver_schedule_path": (
                "05_execution_plan/overlap_schedule.solved.json"
                if overlap_schedule is not None else None
            ),
            "solver_schedule_region_order": (
                _apply_overlap_schedule(
                    _topological_region_order(plan), overlap_schedule
                )
                if overlap_schedule is not None else None
            ),
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

    return AsyncGlueEmitResult(
        out_dir=out_dir,
        executor_path=executor_path,
        manifest_path=manifest_path,
        overall="pass",
        async_regions=tuple(b.region_id for b in async_bindings),
        sync_regions=tuple(b.region_id for b in sync_bindings),
    )
