"""Phase C +: per-workload emitted glue (host-side dispatch code).

Each submodule is a backend-specific emitter that consumes a
:class:`compgen.runtime.execution_plan.ExecutionPlan` and produces an
importable Python module under ``06_glue_emit/``. The emitted module
exposes ``compgen_run(io, kernels, runtime)`` plus an
``assert_plan(io)`` invariant block.
"""

from compgen.runtime.glue_emit.c11_baremetal import (
    C11GlueEmitResult,
    emit_c11_baremetal_executor,
)
from compgen.runtime.glue_emit.cpp_host import (
    CppHostGlueEmitResult,
    emit_cpp_host_executor,
)
from compgen.runtime.glue_emit.dispatch_table import (
    DispatchEmitResult,
    PlanDispatchEntry,
    PlanDispatchSpec,
    emit_dispatch_table,
    plan_dispatch_spec_from_recipe_op,
    select_plan,
)
from compgen.runtime.glue_emit.python_async import (
    AsyncGlueEmitResult,
    emit_python_async_executor,
)
from compgen.runtime.glue_emit.python_cuda import (
    CudaGlueEmitResult,
    emit_python_cuda_executor,
)
from compgen.runtime.glue_emit.python_sync import (
    GlueEmitResult,
    emit_python_sync_executor,
)

__all__ = [
    "AsyncGlueEmitResult",
    "C11GlueEmitResult",
    "CppHostGlueEmitResult",
    "CudaGlueEmitResult",
    "DispatchEmitResult",
    "GlueEmitResult",
    "PlanDispatchEntry",
    "PlanDispatchSpec",
    "emit_c11_baremetal_executor",
    "emit_cpp_host_executor",
    "emit_dispatch_table",
    "emit_python_async_executor",
    "emit_python_cuda_executor",
    "emit_python_sync_executor",
    "plan_dispatch_spec_from_recipe_op",
    "select_plan",
]
