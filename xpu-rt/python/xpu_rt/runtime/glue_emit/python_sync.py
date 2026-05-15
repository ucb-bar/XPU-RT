"""Python SYNC plan executor emitter.

Phase C first emitted-glue milestone. Generate a
``compgen_run(io, kernels, runtime)`` Python module from
``05_execution_plan/execution_plan.yaml``. CPU/SYNC only via the
existing :class:`compgen.runtime.glue.CpuRuntimeAdapter`. widens
to ASYNC + EventTensor; adds CUDA + graph capture.

Generated shape (per Phase C plan):

::

    def compgen_run(io, kernels, runtime):
        assert_plan(io)
        b0 = runtime.allocate_buffer(...) # wires per-region buffer specs
        out0 = runtime.dispatch(contract_0, kernels["region_000"], args_0, {})
        runtime.synchronize()
        return out0

The emitted module is a STANDALONE python file under ``06_glue_emit/``.
It imports ``compgen.runtime.glue.CpuRuntimeAdapter`` and the
``ExecutionPlan`` schema; it does NOT import test-only mocks (the
realness-scan gate enforces this on the emitted file).

Hard rules at this layer:

- The generated executor calls one ``runtime.dispatch`` per region in
  a topologically-sorted order derived from
  ``ExecutionPlan.dependency_edges``.
Each region's kernel callable is provided by the operator (
  wires the load-from-artifact path). ships the protocol; the
  callable surface is dict[str, Callable].
``assert_plan(io)`` is a stub at ; generates the typed
  ``PLAN_VIOLATION_<KIND>`` checks from the contract fields.
- Bound regions are dispatched; unbound regions raise a typed
  ``PlanViolation`` at runtime so 's "honest unbound" cannot
  silently elide a region.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.runtime.execution_plan import ExecutionPlan


_GLUE_EMIT_SCHEMA_VERSION = "plan_executor_manifest_v1"


@dataclass(frozen=True)
class GlueEmitResult:
    out_dir: Path
    executor_path: Path
    manifest_path: Path
    overall: str  # "pass" | "skipped"
    bound_regions: tuple[str, ...]
    unbound_regions: tuple[str, ...]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_yaml_or_json_plan(path: Path) -> dict[str, Any]:
    """Load the execution plan from yaml (preferred) or json (fallback)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
            return yaml.safe_load(text) or {}
        except ImportError:
            return json.loads(text)
    return json.loads(text)


def _topological_region_order(plan: ExecutionPlan) -> list[str]:
    """Return region_ids in a topologically-valid execution order
    derived from ``dependency_edges``. Falls back to placement order
    when the dependency graph is empty."""
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
        # Deterministic sort within a tier.
        queue.sort()
        r = queue.pop(0)
        out.append(r)
        for n in edges_by_from.get(r, []):
            in_degree[n] -= 1
            if in_degree[n] == 0:
                queue.append(n)
    if len(out) != len(region_ids):
        # Cycle — fall back to placement order; will reject the plan.
        return region_ids
    return out


def _emit_executor_source(
    *,
    plan: ExecutionPlan,
    plan_path: str,
    run_dir_relpath: str,
    region_assertions_body: str = "",
) -> str:
    """Render the generated_plan_executor.py file body."""
    bindings_by_region = {
        b.region_id: b for b in plan.region_kernel_bindings
    }
    region_order = _topological_region_order(plan)

    bound_lines: list[str] = []
    bound_meta: list[dict[str, Any]] = []
    for region_id in region_order:
        binding = bindings_by_region.get(region_id)
        if binding is None:
            bound_lines.append(
                f"    # {region_id}: UNBOUND — refused at runtime by assert_plan."
            )
            continue
        bound_meta.append({
            "region_id": region_id,
            "contract_hash": binding.contract_hash,
            "certificate_path": binding.certificate_path,
            "kernel_artifact": binding.kernel_artifact,
            "dispatch_model": binding.dispatch_model,
        })
        bound_lines.append(
            f"    # Region {region_id!r} — {binding.dispatch_model.upper()} "
            f"dispatch (cert: {binding.contract_hash[:8]}...)"
        )
        bound_lines.append(
            f"    out_{region_id} = runtime.dispatch("
        )
        bound_lines.append(
            f"        contract={region_id!r},  # M-49 will resolve to KernelContractV3"
        )
        bound_lines.append(
            f"        callable_kernel=kernels[{region_id!r}],"
        )
        bound_lines.append(
            f"        args=tuple(io.values()),"
        )
        bound_lines.append(
            "        kwargs={},"
        )
        bound_lines.append(
            "    )"
        )
        bound_lines.append(
            f"    last_out = out_{region_id}.output"
        )
    if not bound_lines:
        bound_lines.append("    last_out = ()  # no regions to dispatch")

    bindings_block_json = json.dumps(
        {b.region_id: {
            "contract_hash": b.contract_hash,
            "certificate_path": b.certificate_path,
            "kernel_artifact": b.kernel_artifact,
            "dispatch_model": b.dispatch_model,
        } for b in plan.region_kernel_bindings},
        indent=4, sort_keys=True,
    )

    # render PlanViolation subclasses + assert_plan body.
    from compgen.runtime.glue_emit.plan_assertions import (
        render_plan_violation_classes,
    )
    plan_violation_classes = render_plan_violation_classes()

    return f'''"""Auto-generated by M-47 (compgen.runtime.glue_emit.python_sync).

Workload: {plan.workload}
Target  : {plan.target}
Source  : {plan_path}

DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.

The executor exposes ``compgen_run(io, kernels, runtime)`` per the
M-47 protocol. ``kernels`` is a dict mapping region_id to a callable
the runtime adapter dispatches. ``runtime`` is a
``compgen.runtime.glue.RuntimeAdapter`` (default:
``CpuRuntimeAdapter``). M-48 generates the typed assert_plan body
from contract fields; M-49 wires the load-kernel-from-artifact path.
"""
from __future__ import annotations

from typing import Any, Callable

from compgen.runtime.glue import (
    CpuRuntimeAdapter,
    RuntimeAdapter,
    select_adapter,
)


# Plan summary — frozen at emit time. Re-emit if the plan changes.
PLAN_WORKLOAD = {plan.workload!r}
PLAN_TARGET = {plan.target!r}
PLAN_REGION_ORDER = {region_order!r}
PLAN_RUN_DIR_RELPATH = {run_dir_relpath!r}
KERNEL_BINDINGS = {bindings_block_json}


# M-48 typed PlanViolation classes (one per check kind).
{plan_violation_classes}


def assert_plan(io):
    """Plan invariants — generated from contract fields by M-48.

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


def compgen_run(
    io: dict[str, Any],
    kernels: dict[str, Callable[..., Any]],
    runtime: RuntimeAdapter | None = None,
) -> Any:
    """Per-workload emitted plan executor (M-47).

    Args:
        io: model inputs keyed by name.
        kernels: dict mapping region_id to a callable the runtime
            adapter dispatches. M-49 wires the load-from-artifact path
            so this dict is built from KERNEL_BINDINGS' kernel_artifact
            paths automatically.
        runtime: a compgen.runtime.glue.RuntimeAdapter. If None,
            ``select_adapter(PLAN_TARGET)`` picks the right one.

    Returns:
        The last region's output(s). M-49 wires the proper
        torch-tensor output protocol.
    """
    if runtime is None:
        runtime = select_adapter(PLAN_TARGET)
    assert_plan(io)
    last_out: Any = ()

{chr(10).join(bound_lines)}

    runtime.synchronize()
    return last_out


def main() -> None:
    """Demonstration entry point — operators call this after wiring the
    kernels dict from KERNEL_BINDINGS' kernel_artifact paths. M-49
    will replace this with a real driver that runs the differential."""
    print(
        f"compgen_run(io, kernels, runtime) generated by M-47 for "
        f"workload={{PLAN_WORKLOAD!r}} target={{PLAN_TARGET!r}}; "
        f"{{len(KERNEL_BINDINGS)}} bound region(s) over "
        f"{{len(PLAN_REGION_ORDER)}} planned region(s)."
    )


if __name__ == "__main__":
    main()
'''


def emit_python_sync_executor(run_dir: Path) -> GlueEmitResult:
    """Read the plan from disk, render the SYNC executor, and
    persist it under ``06_glue_emit/``.

    Caller invariant: has already emitted
    ``05_execution_plan/execution_plan.yaml`` (or .json).
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

    bound = tuple(b.region_id for b in plan.region_kernel_bindings)
    placement_regions = tuple(rp.region_id for rp in plan.region_placement)
    unbound = tuple(r for r in placement_regions if r not in bound)

    out_dir = run_dir / "06_glue_emit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # build the per-region assertion body from contract files.
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

    executor_path = out_dir / "generated_plan_executor.py"
    executor_path.write_text(
        _emit_executor_source(
            plan=plan,
            plan_path=str(plan_path.relative_to(run_dir)),
            run_dir_relpath="..",
            region_assertions_body=region_assertions_body,
        ),
        encoding="utf-8",
    )

    manifest_path = out_dir / "plan_executor_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": _GLUE_EMIT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "workload": plan.workload,
            "target": plan.target,
            "executor_kind": "python_sync",
            "executor_path": str(executor_path.relative_to(run_dir)),
            "source_plan_path": str(plan_path.relative_to(run_dir)),
            "bound_regions": list(bound),
            "unbound_regions": list(unbound),
            "region_order": _topological_region_order(plan),
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

    overall = "pass" if bound else "skipped"
    return GlueEmitResult(
        out_dir=out_dir,
        executor_path=executor_path,
        manifest_path=manifest_path,
        overall=overall,
        bound_regions=bound,
        unbound_regions=unbound,
    )
