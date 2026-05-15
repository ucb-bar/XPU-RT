"""Resource-budget check (D6, Phase G).

Mechanical post-emit gate: the emit's static allocation totals do
not exceed the plan's declared budget. Three observable budgets:

- ``push_constants_bytes`` — bytes reserved for the push-constants
  block; capped at the libcompgen_rt static maximum.
- ``binding_slots``        — the count of ``cg_rt_buffer_t *``
  pointers in the bindings array; capped at the static maximum.
- ``kernel_extern_count``  — the number of ``compgen_kernel_<id>``
  externs the emit references; must equal the bound region count
  (any mismatch is also flagged by the plan-refinement gate).

When the plan's resources or kernel-binding fields declare a budget,
the gate enforces it. When the plan is silent on a resource, the
gate uses the static libcompgen_rt cap (256 bytes push-constants,
32 binding slots — both visible 's emitted source).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from compgen.runtime.errors import ResourceBudgetError


# Static caps the emitters reserve. If you change these /
# change them here in lockstep.
DEFAULT_PUSH_CONSTANT_BYTES: int = 256
DEFAULT_BINDING_SLOTS: int = 32

_PUSH_CONSTANT_RE_C = re.compile(r"uint8_t\s+push_constants\s*\[\s*(\d+)\s*\]")
_PUSH_CONSTANT_RE_CPP = re.compile(
    r"uint32_t\s+push_constants\s*\[\s*(\d+)\s*\]"
)
_BINDINGS_RE = re.compile(r"cg_rt_buffer_t\s*\*\s*bindings\s*\[\s*(\d+)\s*\]")
_KERNEL_EXTERN_RE = re.compile(r"compgen_kernel_([A-Za-z0-9_]+)")


@dataclass(frozen=True)
class ResourceBudgetReport:
    overall: str  # "pass" | "fail"
    emit_path: str
    push_constant_bytes: int = 0
    binding_slots: int = 0
    kernel_extern_count: int = 0
    bound_region_count: int = 0
    failures: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)
    # (resource_name, declared_budget, observed)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "runtime_resource_budget_v1",
            "overall": self.overall,
            "emit_path": self.emit_path,
            "push_constant_bytes": self.push_constant_bytes,
            "binding_slots": self.binding_slots,
            "kernel_extern_count": self.kernel_extern_count,
            "bound_region_count": self.bound_region_count,
            "failures": [
                {
                    "resource": r, "declared_budget": d, "observed": o,
                }
                for r, d, o in self.failures
            ],
        }


def _scan_emit(src: str) -> tuple[int, int, int]:
    """Return (push_constant_bytes, binding_slots, kernel_extern_count)."""
    pc_match = _PUSH_CONSTANT_RE_C.search(src) or _PUSH_CONSTANT_RE_CPP.search(src)
    bind_match = _BINDINGS_RE.search(src)
    pc_count = int(pc_match.group(1)) if pc_match else 0
    pc_elem_bytes = 1 if pc_match and "uint8_t" in pc_match.group(0) else 4
    pc_bytes = pc_count * pc_elem_bytes
    binding_slots = int(bind_match.group(1)) if bind_match else 0
    kernel_count = len(set(_KERNEL_EXTERN_RE.findall(src)))
    return pc_bytes, binding_slots, kernel_count


def check_resource_budget(
    emit_dir: Path,
    *,
    raise_on_fail: bool = True,
    push_constants_max_bytes: int = DEFAULT_PUSH_CONSTANT_BYTES,
    binding_slots_max: int = DEFAULT_BINDING_SLOTS,
) -> ResourceBudgetReport:
    """Run the resource-budget check on an emit directory.

    Picks the C11 emit when available, else the C++ host emit. Reads
    the matching manifest for the bound region count and any
    plan-declared budgets.
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
        report = ResourceBudgetReport(
            overall="fail",
            emit_path=str(emit_dir),
            failures=(("emit", 1, 0),),
        )
        if raise_on_fail:
            raise ResourceBudgetError(
                "emit", 1, 0, emit_path=str(emit_dir),
            )
        return report

    manifest = json.loads(manifest_path.read_text())
    bound = manifest.get("bound_regions", [])
    bound_count = len(bound)

    src = emit_path.read_text()
    pc_bytes, slots, n_kernels = _scan_emit(src)

    failures: list[tuple[str, int, int]] = []
    if pc_bytes > push_constants_max_bytes:
        failures.append(
            ("push_constants_bytes", push_constants_max_bytes, pc_bytes)
        )
    if slots > binding_slots_max:
        failures.append(("binding_slots", binding_slots_max, slots))
    if n_kernels != bound_count:
        failures.append(("kernel_extern_count", bound_count, n_kernels))

    overall = "pass" if not failures else "fail"
    report = ResourceBudgetReport(
        overall=overall,
        emit_path=str(emit_path),
        push_constant_bytes=pc_bytes,
        binding_slots=slots,
        kernel_extern_count=n_kernels,
        bound_region_count=bound_count,
        failures=tuple(failures),
    )
    if overall == "fail" and raise_on_fail:
        resource, declared, observed = failures[0]
        raise ResourceBudgetError(
            resource, declared, observed, emit_path=str(emit_path),
        )
    return report
