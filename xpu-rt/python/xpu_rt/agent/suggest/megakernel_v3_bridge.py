"""Bridge legacy ``suggest_megakernel`` proposals → ``Granularity.MEGA``
``KernelContractV3``.

The pre-existing path (``suggest_megakernel.suggest_megakernel``) emits
``ProposalCandidate`` objects whose ``chosen["fused_region_refs"]``
lists the recipe-region symbols that would form a megakernel. That
record lives in the Recipe IR but never produces a v3 contract — so
the new oracles (``granularity_oracle``, ``fusion_oracle``) and the
runtime adapter never see the recipe-level decision.

This bridge closes the loop. Given:
  * the proposal's ``chosen["fused_region_refs"]`` (sub-region symbols)
  * a callable that returns a v3 contract per region symbol
  * the target ``HardwareEnvelope``

it returns ONE ``KernelContractV3(granularity=MEGA, body=[...sub-contracts...])``
that the v3 oracles + runtime adapter consume.

Two safety nets:
  * The v3 ``__post_init__`` enforces MEGA invariants (PERSISTENT
    dispatch, sub-buffers in REGISTER/SCRATCHPAD, no nested MEGA).
  * Sub-contracts whose memory tier isn't compatible are silently
    promoted to SCRATCHPAD residency (with a metadata note), so the
    legacy proposal doesn't get blocked by a downstream invariant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from compgen.kernels.contract_v3 import (
    DispatchModel,
    DispatchSpec,
    EventDecl,
    ExecutionEnvelope,
    Granularity,
    HardwareEnvelope,
    InternalEventEdge,
    IOContract,
    KernelContractV3,
    MemorySpec,
    MemoryTier,
    NumericsSpec,
    OrchestrationSpec,
    SyncSpec,
    TensorIO,
)


@dataclass(frozen=True)
class MegakernelBridgeResult:
    """What ``build_mega_contract_from_proposal`` returns."""

    contract: KernelContractV3
    sub_region_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _force_to_scratchpad_or_register(c: KernelContractV3) -> KernelContractV3:
    """Sub-kernels of a MEGA must keep buffers in REGISTER/SCRATCHPAD.

    If the legacy proposal's sub-contract has DEVICE_DRAM tiers, promote
    them to SCRATCHPAD so the MEGA invariant check passes. Records the
    promotion in metadata so the diagnosis can flag if it harms perf.
    """
    m = c.orchestration.memory
    needs_promo = any(t not in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD) for t in (*m.input_tiers, *m.output_tiers))
    if not needs_promo:
        return c

    new_input_tiers = tuple(
        MemoryTier.SCRATCHPAD if t not in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD) else t for t in m.input_tiers
    )
    new_output_tiers = tuple(
        MemoryTier.SCRATCHPAD if t not in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD) else t for t in m.output_tiers
    )
    new_memory = MemorySpec(
        input_tiers=new_input_tiers,
        output_tiers=new_output_tiers,
        lifetimes=m.lifetimes,
        in_place_safe=m.in_place_safe,
    )
    new_orch = OrchestrationSpec(
        execution=c.orchestration.execution,
        sync=c.orchestration.sync,
        memory=new_memory,
        fusion=c.orchestration.fusion,
        dispatch=c.orchestration.dispatch,
        observability=c.orchestration.observability,
    )
    new_metadata = dict(c.metadata)
    new_metadata.setdefault("notes", []).append(
        "tiers promoted to SCRATCHPAD by megakernel_v3_bridge to satisfy MEGA invariants"
    )
    return KernelContractV3(
        op_name=c.op_name,
        archetype=c.archetype,
        io=c.io,
        granularity=c.granularity,
        orchestration=new_orch,
        selection=c.selection,
        cost=c.cost,
        body=c.body,
        internal_events=c.internal_events,
        legacy=c.legacy,
        metadata=new_metadata,
    )


def build_mega_contract_from_proposal(
    proposal_chosen: dict[str, Any],
    *,
    contract_for_region: Callable[[str], KernelContractV3],
    envelope: HardwareEnvelope,
    op_name: str | None = None,
) -> MegakernelBridgeResult:
    """Materialise a v3 MEGA contract from a legacy megakernel proposal.

    Args:
        proposal_chosen: The ``chosen`` dict from
            ``ProposalCandidate.chosen`` produced by
            ``suggest_megakernel``. Must contain
            ``fused_region_refs: list[str]``.
        contract_for_region: Callable mapping a region symbol → its
            (NORMAL) v3 contract. Caller wires this from the recipe.
        envelope: The target ``HardwareEnvelope`` to attach to the
            outer MEGA contract's ``ExecutionEnvelope``.
        op_name: Optional override for the outer kernel's op_name.
            Defaults to ``proposal_chosen.get("megakernel_name")`` or
            ``"megakernel.<n>"``.
    """
    refs: list[str] = list(proposal_chosen.get("fused_region_refs", []))
    if not refs:
        raise ValueError("proposal_chosen has no 'fused_region_refs'; cannot build MEGA contract")

    # 1. Pull each sub-contract from the recipe + force scratchpad-residency.
    sub_contracts: list[KernelContractV3] = []
    notes: list[str] = []
    for rid in refs:
        sub = contract_for_region(rid)
        if sub.granularity is Granularity.MEGA:
            # Nested MEGA forbidden; either skip or fall back to NORMAL.
            notes.append(f"sub-region {rid!r} was already MEGA; skipping (MEGA cannot nest MEGA)")
            continue
        sub_contracts.append(_force_to_scratchpad_or_register(sub))

    if not sub_contracts:
        raise ValueError("no eligible sub-contracts after filtering nested MEGAs")

    # 2. Internal sync edges — naive linear chain (sub_i → sub_{i+1}).
    # Real codegen (Wave 5+) uses event_tensor analysis; here we just
    # wire a sequential chain so the v3 invariants pass.
    internal_events: list[InternalEventEdge] = []
    for i in range(len(sub_contracts) - 1):
        internal_events.append(
            InternalEventEdge(
                event_name=f"{sub_contracts[i].op_name}_done",
                producer_idx=i,
                consumer_idx=i + 1,
            )
        )

    # 3. Outer IO — first sub's inputs + last sub's outputs (linearised).
    first = sub_contracts[0]
    last = sub_contracts[-1]
    outer_io = IOContract(
        inputs=tuple(
            TensorIO(
                name=f"mega_in_{i}_{t.name}",
                shape=t.shape,
                dtype_class=t.dtype_class,
                layout=t.layout,
                alignment_bytes=t.alignment_bytes,
            )
            for i, t in enumerate(first.io.inputs)
        ),
        outputs=tuple(
            TensorIO(
                name=f"mega_out_{i}_{t.name}",
                shape=t.shape,
                dtype_class=t.dtype_class,
                layout=t.layout,
                alignment_bytes=t.alignment_bytes,
            )
            for i, t in enumerate(last.io.outputs)
        ),
        numerics=NumericsSpec(
            accumulator_dtype=first.io.numerics.accumulator_dtype,
            fast_math=False,
            max_relative_error=max(
                first.io.numerics.max_relative_error,
                last.io.numerics.max_relative_error,
            ),
        ),
    )

    # 4. Outer orchestration — PERSISTENT dispatch, fires one external event.
    outer_op_name = op_name or proposal_chosen.get("megakernel_name", f"megakernel.{first.op_name}_to_{last.op_name}")
    orch = OrchestrationSpec(
        execution=ExecutionEnvelope(hardware=envelope),
        sync=SyncSpec(
            event_decls=(EventDecl(name=f"{outer_op_name}_done", scope="device"),),
        ),
        memory=MemorySpec(
            input_tiers=tuple(MemoryTier.DEVICE_DRAM for _ in outer_io.inputs),
            output_tiers=tuple(MemoryTier.DEVICE_DRAM for _ in outer_io.outputs),
        ),
        dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
    )

    contract = KernelContractV3(
        op_name=outer_op_name,
        archetype=first.archetype,  # outer archetype = leading op's
        io=outer_io,
        granularity=Granularity.MEGA,
        orchestration=orch,
        body=tuple(sub_contracts),
        internal_events=tuple(internal_events),
        metadata={
            "source": "megakernel_v3_bridge",
            "sub_region_ids": list(refs),
            "promotion_notes": notes,
        },
    )
    return MegakernelBridgeResult(
        contract=contract,
        sub_region_ids=tuple(refs),
        notes=tuple(notes),
    )


__all__ = [
    "MegakernelBridgeResult",
    "build_mega_contract_from_proposal",
]
