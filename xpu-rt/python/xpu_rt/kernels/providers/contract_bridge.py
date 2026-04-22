"""Bridge KernelContract v3 → KernelProvider protocol's v1 KernelContract.

Existing providers (autocomp, exo, megakernel, …) consume the v1
``compgen.kernels.provider.KernelContract`` dataclass. v3 lives in
``compgen.kernels.contract_v3.KernelContractV3`` and carries the
sharp-boundary IO + orchestration + execution envelope.

This bridge produces a v1 contract that:

* preserves all v1 fields (op_family, shapes, dtypes, layout, target,
  hardware_key, objective) so existing providers keep working
* attaches the v3 ``kernel_facing()`` view inside ``constraints`` under
  key ``"kernel_facing_view"`` — the new ``ClaudeCodeKernelProvider``
  (and any v3-aware provider) reads it from there for the rich prompt

So the same router can serve both v3-aware and legacy providers without
a fork.
"""

from __future__ import annotations

from typing import Any

from compgen.kernels.contract_v3 import (
    KernelContractV3,
    LayoutKind,
)
from compgen.kernels.provider import KernelContract as KernelContractV1


def _v3_layout_to_v1(layout: LayoutKind) -> str:
    """Map v3 LayoutKind to v1's stringly-typed layout."""
    return {
        LayoutKind.ROW_MAJOR: "row_major",
        LayoutKind.COLUMN_MAJOR: "column_major",
        LayoutKind.BLOCKED: "blocked",
        LayoutKind.PACKED_K_MAJOR: "packed_k_major",
        LayoutKind.OPAQUE: "opaque",
    }[layout]


def _shape_to_tuple(shape) -> tuple[int, ...]:
    """v3 ShapeClass → tuple of int (None dims become -1, the v1 dynamic-dim
    convention used elsewhere in the repo)."""
    return tuple(d if d is not None else -1 for d in shape.dims)


def v3_to_v1_contract(
    v3: KernelContractV3,
    *,
    region_id: str = "",
    extra_constraints: dict[str, Any] | None = None,
) -> KernelContractV1:
    """Project a v3 contract down to the v1 surface that providers consume.

    The full v3 ``kernel_facing()`` view is attached as
    ``constraints['kernel_facing_view']`` so v3-aware providers (e.g.
    ``ClaudeCodeKernelProvider``) can read the rich spec without a
    schema migration on the provider side.
    """
    # Pull canonical fields from the IO block.
    input_shapes = tuple(_shape_to_tuple(t.shape) for t in v3.io.inputs)
    output_shapes = tuple(_shape_to_tuple(t.shape) for t in v3.io.outputs)

    # Dtype set: union across all IO. Each TensorIO declares a dtype_class
    # (set of accepted dtypes); we surface the union so the v1 selector
    # has the right superset.
    dtypes_set: set[str] = set()
    for t in (*v3.io.inputs, *v3.io.outputs):
        dtypes_set.update(t.dtype_class)
    dtypes = tuple(sorted(dtypes_set))

    # Use the FIRST input's layout as the v1 "layout" (v1 only carries
    # one). The kernel_facing view inside constraints carries the full
    # per-operand layout info for v3-aware providers.
    layout_v1 = _v3_layout_to_v1(v3.io.inputs[0].layout) if v3.io.inputs else "row_major"

    # Target / hardware identity from the execution envelope (v3) or
    # selection metadata (v3 fallback).
    env = v3.orchestration.execution
    target_name = env.hardware.target_name if env else ""
    hardware_key = target_name  # one-to-one for now

    constraints: dict[str, Any] = dict(extra_constraints or {})
    constraints["kernel_facing_view"] = v3.kernel_facing()
    constraints["archetype"] = v3.archetype.value
    constraints["granularity"] = v3.granularity.value

    return KernelContractV1(
        region_id=region_id or v3.op_name,
        op_family=v3.op_name,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        dtypes=dtypes,
        layout=layout_v1,
        target_name=target_name,
        hardware_key=hardware_key,
        objective="latency",
        constraints=constraints,
    )


__all__ = ["v3_to_v1_contract"]
