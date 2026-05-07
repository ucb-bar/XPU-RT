"""Phase C M-47+: per-workload emitted glue (host-side dispatch code).

Each submodule is a backend-specific emitter that consumes a
:class:`compgen.runtime.execution_plan.ExecutionPlan` and produces an
importable Python module under ``06_glue_emit/``. The emitted module
exposes ``compgen_run(io, kernels, runtime)`` plus an
``assert_plan(io)`` invariant block.
"""

from compgen.runtime.glue_emit.python_sync import (
    GlueEmitResult,
    emit_python_sync_executor,
)

__all__ = ["GlueEmitResult", "emit_python_sync_executor"]
