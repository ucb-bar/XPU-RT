"""Conformance-harness workload factories.

Each module in this subpackage builds the structures Phase-7's
``_compile_and_evaluate`` consumes for one workload:

- ``build(dtype, num_gpus)`` — returns a :class:`Workload`
  containing the eager model, sample inputs, the
  :class:`xpu_rt.runtime.megakernel.MegakernelGraph` factory, the
  CUDA C++ device-function bodies, and the eager-output runner.

Workloads are registered in
:data:`xpu_rt.testing.workloads.WORKLOAD_FACTORIES` so the
conformance harness can look them up by their
:class:`xpu_rt.testing.etc_conformance.ConformanceWorkload` value.
"""

from __future__ import annotations

from xpu_rt.testing.workloads import (
    decoder_layer,
    diamond_dag,
    gemm_reduce_scatter,
)

# Maps the string value of
# :class:`~xpu_rt.testing.etc_conformance.ConformanceWorkload`
# (e.g. ``"diamond_dag"``) to its ``build(dtype, num_gpus)`` factory.
# Workloads not yet implemented are simply absent — the harness
# raises a typed error rather than silently routing through a stub.
WORKLOAD_FACTORIES = {
    "diamond_dag": diamond_dag.build,
    "decoder_layer": decoder_layer.build,
    "gemm_rs": gemm_reduce_scatter.build,
}

__all__ = ["WORKLOAD_FACTORIES"]
