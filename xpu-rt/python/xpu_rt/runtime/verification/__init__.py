"""Runtime-as-verification-target harness (, Phase G — §12 Dream 6).

Three mechanical post-emit checks on a Layer-1 plan executor:

- :mod:`compgen.runtime.verification.plan_refinement` — every plan op
  corresponds to one launch in the emit, in the declared order.
- :mod:`compgen.runtime.verification.abi_conformance` — the emit
  calls only ``cg_rt_*`` and ``compgen_kernel_*`` externs; nothing
  else (no ``cudaMalloc``, no ``hipLaunchKernel``, no ``vkCmd*``).
- :mod:`compgen.runtime.verification.resource_budget` — static
  allocation totals (push-constant bytes, binding-slot count, kernel
  count) do not exceed the plan's declared budget.

Each check returns a typed report and raises a named typed error
from :mod:`compgen.runtime.errors` on failure. Callers compose the
three into a single :func:`run_runtime_verification` aggregator.
"""

from __future__ import annotations

from compgen.runtime.verification.abi_conformance import (
    AbiConformanceReport,
    check_abi_conformance,
)
from compgen.runtime.verification.plan_refinement import (
    PlanRefinementReport,
    check_plan_refinement,
)
from compgen.runtime.verification.resource_budget import (
    ResourceBudgetReport,
    check_resource_budget,
)
from compgen.runtime.verification.runner import (
    RuntimeVerificationReport,
    run_runtime_verification,
)


__all__ = [
    "AbiConformanceReport",
    "PlanRefinementReport",
    "ResourceBudgetReport",
    "RuntimeVerificationReport",
    "check_abi_conformance",
    "check_plan_refinement",
    "check_resource_budget",
    "run_runtime_verification",
]
