"""Recipe IR Family P: Propose operations (LLM invent-slots).

Each op here is a **plan container** the LLM emits when running in
invent mode. Unlike the single-action ``recipe.tile``, ``recipe.fuse``,
etc. in ``ops_candidate.py``, a propose-op records the LLM's full
deliberation:

    candidates + chosen + target_feature_justification + gate_result

This mirrors the typed shape in
``user_perspective/prototypes/invent_slots/*/schema.yaml`` and lets the
verification gate + recorder + promotion subsystems treat every
invention uniformly.

Design note: the payload (candidates/chosen/gate_result) is stored as a
``StringAttr`` holding JSON. This keeps the xDSL layer simple and
matches what the LLM emits natively. A future refactor may promote the
JSON to a custom ``PlanPayloadAttr`` for schema-level validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.recipe.attrs import ProvenanceAttr


# ---------------------------------------------------------------------------
# Shared payload dataclass (Python-side)
# ---------------------------------------------------------------------------


SelectVsInvent = Literal["select", "invent"]


@dataclass(frozen=True)
class ProposePayload:
    """Typed Python-side view of a propose-op's JSON payload.

    Fields match ``user_perspective/prototypes/schemas/recipe_semantic_global.schema.yaml``
    decisions[] entries.
    """

    candidates: list[dict[str, Any]] = field(default_factory=list)
    chosen: dict[str, Any] = field(default_factory=dict)
    target_feature_justification: str = ""
    gate_result: dict[str, Any] = field(default_factory=dict)
    select_vs_invent: SelectVsInvent = "invent"
    llm_turn_id: str = ""
    baseline_seed_source: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "candidates": self.candidates,
                "chosen": self.chosen,
                "target_feature_justification": self.target_feature_justification,
                "gate_result": self.gate_result,
                "select_vs_invent": self.select_vs_invent,
                "llm_turn_id": self.llm_turn_id,
                "baseline_seed_source": self.baseline_seed_source,
            },
            sort_keys=True,
            default=str,
        )

    @classmethod
    def from_json(cls, blob: str) -> ProposePayload:
        d = json.loads(blob)
        return cls(
            candidates=list(d.get("candidates", [])),
            chosen=dict(d.get("chosen", {})),
            target_feature_justification=str(d.get("target_feature_justification", "")),
            gate_result=dict(d.get("gate_result", {})),
            select_vs_invent=d.get("select_vs_invent", "invent"),
            llm_turn_id=str(d.get("llm_turn_id", "")),
            baseline_seed_source=str(d.get("baseline_seed_source", "")),
        )


# ---------------------------------------------------------------------------
# Base verify() helper — all propose ops share these invariants
# ---------------------------------------------------------------------------


def _verify_payload_shape(payload_str: str, op_name: str) -> None:
    """Shared verify: payload is valid JSON with at least a ``chosen`` entry."""
    try:
        parsed = json.loads(payload_str)
    except json.JSONDecodeError as e:
        raise VerifyException(f"{op_name}.payload is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise VerifyException(f"{op_name}.payload must decode to a dict")
    if "chosen" not in parsed:
        raise VerifyException(f"{op_name}.payload missing required key 'chosen'")
    if "select_vs_invent" not in parsed:
        raise VerifyException(f"{op_name}.payload missing 'select_vs_invent'")
    if parsed["select_vs_invent"] not in ("select", "invent"):
        raise VerifyException(
            f"{op_name}.payload.select_vs_invent must be 'select' or 'invent', "
            f"got {parsed['select_vs_invent']!r}"
        )


# ---------------------------------------------------------------------------
# Concrete propose ops
# ---------------------------------------------------------------------------


@irdl_op_definition
class ProposeLayoutPlanOp(IRDLOperation):
    """LLM proposes target-aligned physical layouts per region (P3 / Phase 3).

    Replaces ad-hoc encoding decisions with a typed plan the verification
    gate can accept/reject. See
    ``user_perspective/prototypes/invent_slots/propose_layout_plan/``.
    """

    name = "recipe.propose_layout_plan"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    payload = prop_def(StringAttr)              # JSON ProposePayload
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeFusionOp(IRDLOperation):
    """LLM proposes fusion boundaries (THE primary autocomp-cost knob).

    The payload contains the grouped regions + target kernel family +
    justification linking to a supported_kernel_families entry.
    """

    name = "recipe.propose_fusion"

    sym_name = opt_prop_def(StringAttr)
    grouped_regions = prop_def(ArrayAttr)        # ArrayAttr of SymbolRefAttr
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        if len(self.grouped_regions.data) < 1:
            raise VerifyException(
                f"{self.name} requires at least 1 region in grouped_regions"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeMultiOutputFusionOp(IRDLOperation):
    """Multi-output fusion variant (e.g. LN with residual side-output)."""

    name = "recipe.propose_multi_output_fusion"

    sym_name = opt_prop_def(StringAttr)
    grouped_regions = prop_def(ArrayAttr)
    producer_output_count = prop_def(IntegerAttr)
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        if self.producer_output_count.value.data < 2:
            raise VerifyException(
                f"{self.name} requires producer_output_count >= 2"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposePeepholePatternOp(IRDLOperation):
    """LLM proposes a novel peephole pattern not in the ported toolbox.

    Used for novel attention variants, new activation idioms, etc. Paid
    attention to IREE's observation that RaiseSpecialOps patterns are
    structure-sensitive.
    """

    name = "recipe.propose_peephole_pattern"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    pattern_class = prop_def(StringAttr)        # "attention_variant" / "activation_idiom" / ...
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeNumericsPlanOp(IRDLOperation):
    """LLM proposes a per-region numerics policy tied to target features."""

    name = "recipe.propose_numerics_plan"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeDequantFusionOp(IRDLOperation):
    """LLM proposes a novel dequant-matmul fusion pattern."""

    name = "recipe.propose_dequant_fusion"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeShardingPlanOp(IRDLOperation):
    """LLM proposes a multi-device sharding plan.

    Phase 3, multi-device. CP-SAT AutoSharding-style baseline seed.
    """

    name = "recipe.propose_sharding_plan"

    sym_name = opt_prop_def(StringAttr)
    module_ref = prop_def(SymbolRefAttr)        # whole-module reference
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeBufferLifetimePlanOp(IRDLOperation):
    """LLM proposes a buffer lifetime + aliasing plan (Phase 5)."""

    name = "recipe.propose_buffer_lifetime_plan"

    sym_name = opt_prop_def(StringAttr)
    plan_ref = prop_def(SymbolRefAttr)          # references an execution plan
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeRematerializationPlanOp(IRDLOperation):
    """LLM proposes a remat plan bounded by memory + recompute cost."""

    name = "recipe.propose_rematerialization_plan"

    sym_name = opt_prop_def(StringAttr)
    plan_ref = prop_def(SymbolRefAttr)
    memory_budget_bytes = prop_def(IntegerAttr)
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        if self.memory_budget_bytes.value.data <= 0:
            raise VerifyException(
                f"{self.name} memory_budget_bytes must be positive, "
                f"got {self.memory_budget_bytes.value.data}"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeCollectivePipelineOp(IRDLOperation):
    """LLM proposes a multi-device collective pipelining plan (Phase 5)."""

    name = "recipe.propose_collective_pipeline"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    direction = prop_def(StringAttr)            # "forward" / "backward" / "host_offload"
    payload = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        if self.direction.data not in ("forward", "backward", "host_offload"):
            raise VerifyException(
                f"{self.name} direction must be forward|backward|host_offload, "
                f"got {self.direction.data!r}"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeMegakernelSynthesisOp(IRDLOperation):
    """LLM proposes fusing a region cluster into a single persistent megakernel.

    Phase 4 invent-slot.  Models the Event Tensor Compiler abstraction
    (Jin et al., MLSys '26): rather than launching a sequence of kernels
    with implicit kernel-boundary sync, fuse them into one persistent
    kernel coordinated by Event Tensors (counter-based semaphores).

    The payload's ``chosen`` block records:
        - megakernel_name: stable identifier for the resulting megakernel
        - fused_region_refs: regions absorbed into the megakernel
        - event_tensor_decls: list of (name, shape, wait_count, scope)
        - task_partition: per-region task-grid shape
        - prefetch_annotations: optional per-tile weight prefetch hints

    The ``target_feature_justification`` must reference target capability
    flags ``persistent_kernels`` and ``semaphore_atomics``.
    """

    name = "recipe.propose_megakernel_synthesis"

    sym_name = opt_prop_def(StringAttr)
    fused_region_refs = prop_def(ArrayAttr)         # ArrayAttr of SymbolRefAttr
    target_device_ref = opt_prop_def(SymbolRefAttr)
    payload = prop_def(StringAttr)                  # JSON ProposePayload
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        if len(self.fused_region_refs.data) < 1:
            raise VerifyException(
                f"{self.name} requires at least 1 region in fused_region_refs"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


@irdl_op_definition
class ProposeSchedulingPolicyOp(IRDLOperation):
    """LLM proposes static vs dynamic scheduling for a megakernel.

    Phase 4 invent-slot, paired with ProposeMegakernelSynthesisOp.

    Static scheduling precomputes per-SM task queues at compile time
    (Algorithm 1 in the ETC paper) and yields minimal runtime overhead
    for predictable workloads.  Dynamic scheduling uses an on-GPU push/pop
    scheduler (Algorithm 2) and is required for data-dependent dynamism
    (e.g. MoE token routing).

    The payload's ``chosen`` block records:
        - policy: ``"static"`` or ``"dynamic"``
        - sm_count: target SM count (for static partitioning)
        - dynamic_features: list of data-dependent edges (only for dynamic)
        - early_push: bool, whether to enable Appendix-E early-push opt
    """

    name = "recipe.propose_scheduling_policy"

    sym_name = opt_prop_def(StringAttr)
    megakernel_ref = prop_def(SymbolRefAttr)
    payload = prop_def(StringAttr)                  # JSON ProposePayload
    guard_refs = opt_prop_def(ArrayAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    _VALID_POLICIES: ClassVar[tuple[str, ...]] = ("static", "dynamic")

    def verify_(self) -> None:
        _verify_payload_shape(self.payload.data, self.name)
        chosen = json.loads(self.payload.data).get("chosen", {})
        policy = chosen.get("policy")
        if policy is not None and policy not in self._VALID_POLICIES:
            raise VerifyException(
                f"{self.name} chosen.policy must be one of "
                f"{self._VALID_POLICIES}, got {policy!r}"
            )

    def get_payload(self) -> ProposePayload:
        return ProposePayload.from_json(self.payload.data)


# ---------------------------------------------------------------------------
# Group + exports
# ---------------------------------------------------------------------------

_PROPOSE_OPS = [
    ProposeLayoutPlanOp,
    ProposeFusionOp,
    ProposeMultiOutputFusionOp,
    ProposePeepholePatternOp,
    ProposeNumericsPlanOp,
    ProposeDequantFusionOp,
    ProposeShardingPlanOp,
    ProposeBufferLifetimePlanOp,
    ProposeRematerializationPlanOp,
    ProposeCollectivePipelineOp,
    ProposeMegakernelSynthesisOp,
    ProposeSchedulingPolicyOp,
]


__all__ = [
    "ProposePayload",
    "ProposeBufferLifetimePlanOp",
    "ProposeCollectivePipelineOp",
    "ProposeDequantFusionOp",
    "ProposeFusionOp",
    "ProposeLayoutPlanOp",
    "ProposeMegakernelSynthesisOp",
    "ProposeMultiOutputFusionOp",
    "ProposeNumericsPlanOp",
    "ProposePeepholePatternOp",
    "ProposeRematerializationPlanOp",
    "ProposeSchedulingPolicyOp",
    "ProposeShardingPlanOp",
    "SelectVsInvent",
    "_PROPOSE_OPS",
]
