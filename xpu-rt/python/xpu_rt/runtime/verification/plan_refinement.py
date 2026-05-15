"""Plan-refinement check (D6, Phase G).

Mechanical post-emit gate: every plan op corresponds to one launch
in the emit, in the declared order. Three failure kinds:

- ``count_mismatch``    — number of ``cg_rt_command_buffer_dispatch``
                          calls in the emit does not equal the bound
                          region count.
- ``missing_dispatch``  — a region in the plan does not appear as a
                          ``compgen_kernel_<region_id>`` dispatch.
- ``unknown_dispatch``  — the emit dispatches a kernel the plan
                          doesn't declare (extra launch).
- ``order_mismatch``    — kernel dispatch sites appear in an order
                          inconsistent with the plan's region order.

The check reads:

``06_glue_emit/plan_executor_c11_manifest.json`` **or**
  ``06_glue_emit/plan_executor_cpp_host_manifest.json`` for
  the declared region order; and
- the emitted ``.c`` or ``.cpp`` source for the actual dispatch
  sequence (parsed by greping for
  ``cg_rt_command_buffer_dispatch(... compgen_kernel_<id>``).

Both inputs are byte-stable, so the gate is deterministic. The check
returns a typed :class:`PlanRefinementReport`; on failure it raises
:class:`compgen.runtime.errors.RuntimeRefinementError`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from compgen.runtime.errors import RuntimeRefinementError


_DISPATCH_RE = re.compile(
    r"cg_rt_command_buffer_dispatch\s*\(\s*\w+\s*,\s*"
    r"(compgen_kernel_[A-Za-z0-9_]+)"
)


@dataclass(frozen=True)
class PlanRefinementReport:
    """Outcome of the plan-refinement check."""

    overall: str  # "pass" | "fail"
    emit_path: str
    plan_region_order: tuple[str, ...] = field(default_factory=tuple)
    observed_dispatch_order: tuple[str, ...] = field(default_factory=tuple)
    failures: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "runtime_refinement_v1",
            "overall": self.overall,
            "emit_path": self.emit_path,
            "plan_region_order": list(self.plan_region_order),
            "observed_dispatch_order": list(self.observed_dispatch_order),
            "failures": [
                {"kind": k, "detail": d} for k, d in self.failures
            ],
        }


def _parse_dispatch_order(src: str) -> tuple[str, ...]:
    """Return the kernel symbol invoked at each dispatch site, in
    source order."""
    return tuple(
        m.group(1).removeprefix("compgen_kernel_")
        for m in _DISPATCH_RE.finditer(src)
    )


def check_plan_refinement(
    emit_dir: Path,
    *,
    raise_on_fail: bool = True,
) -> PlanRefinementReport:
    """Run the plan-refinement check on a glue-emit directory.

    Locates the (.c + c11 manifest) or (.cpp + cpp_host
    manifest) emit, parses the dispatch sequence, and compares
    against the manifest's declared region order.
    """
    emit_dir = Path(emit_dir).resolve()
    candidates = [
        ("plan_executor_c11_manifest.json", "generated_plan_executor.c"),
        ("plan_executor_cpp_host_manifest.json",
         "generated_plan_executor.cpp"),
    ]
    manifest_path: Path | None = None
    emit_path: Path | None = None
    for manifest_name, src_name in candidates:
        m = emit_dir / manifest_name
        s = emit_dir / src_name
        if m.exists() and s.exists():
            manifest_path = m
            emit_path = s
            break

    if manifest_path is None or emit_path is None:
        report = PlanRefinementReport(
            overall="fail",
            emit_path=str(emit_dir),
            failures=(("missing_emit",
                       "no C11 or C++ emit found under 06_glue_emit/"),),
        )
        if raise_on_fail:
            raise RuntimeRefinementError(
                "missing_emit", report.failures[0][1],
            )
        return report

    manifest = json.loads(manifest_path.read_text())
    plan_region_order = tuple(manifest.get("region_order", []))
    bound_regions = set(manifest.get("bound_regions", []))

    src = emit_path.read_text()
    observed = _parse_dispatch_order(src)

    failures: list[tuple[str, str]] = []

    expected_bound = [r for r in plan_region_order if r in bound_regions]
    if len(observed) != len(expected_bound):
        failures.append((
            "count_mismatch",
            f"emit dispatches {len(observed)} kernel(s); plan has "
            f"{len(expected_bound)} bound region(s)",
        ))
    for region in expected_bound:
        if region not in observed:
            failures.append((
                "missing_dispatch",
                f"plan declares bound region {region!r} but the emit "
                f"does not call compgen_kernel_{region}",
            ))
    for region in observed:
        if region not in bound_regions:
            failures.append((
                "unknown_dispatch",
                f"emit dispatches {region!r} which the plan does not "
                f"declare as a bound region",
            ))

    # Order check: filter observed down to plan-known regions, then
    # ensure that filtered list matches the bound region order.
    observed_known = [r for r in observed if r in bound_regions]
    if observed_known != expected_bound:
        failures.append((
            "order_mismatch",
            f"emit dispatch order {observed_known!r} differs from "
            f"plan's region order {expected_bound!r}",
        ))

    overall = "pass" if not failures else "fail"
    report = PlanRefinementReport(
        overall=overall,
        emit_path=str(emit_path),
        plan_region_order=plan_region_order,
        observed_dispatch_order=observed,
        failures=tuple(failures),
    )
    if overall == "fail" and raise_on_fail:
        kind, detail = failures[0]
        raise RuntimeRefinementError(kind, detail)
    return report
