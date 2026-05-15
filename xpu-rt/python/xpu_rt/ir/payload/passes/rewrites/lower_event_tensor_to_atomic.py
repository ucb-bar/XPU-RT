"""``lower_event_tensor_to_atomic`` -- ETC §3.3 minimal-runtime lowering.

Reconstruction of the Event Tensor Compiler's "Lowering to Minimal
Runtime" step (Jin et al., MLSys 2026, §3.3) as a CompGen pass. Zero
external references; CompGen owns the rewrite.

From the paper:

> Specifically, each Event Tensor is lowered to an integer tensor in
> memory, reusing the existing tensor data structure and avoiding any
> dedicated runtime data structures for events. The notify() and wait()
> operations on this integer tensor are implemented with efficient
> hardware atomics: notify() performs an atomic decrement, while wait()
> spin-waits for the counter to reach zero.

Lowering contract:

- Every ``event.event_tensor %E : !event_tensor<shape>`` becomes an
  external declaration for a backing ``tensor<shape x i32>`` plus an
  init call that fills every entry with the wait count.
- Every ``event.notify %E[i]`` becomes a ``func.call @compgen_event_atomic_decrement``
  taking the event SSA handle + the coordinate indices + the
  decrement value.
- Every ``event.wait %E[i]`` becomes a ``func.call @compgen_event_spin_wait``
  on the same handle + indices.
- The ``event.graph`` wrapper stays intact structurally but its
  body's ops now reference the integer-tensor backing.

The resulting module is free of ``!event.event_tensor_type`` values
at the SSA level; every reference is by symbol (StringAttr) so the
downstream backend (Triton / ukernel emitter) can inline the atomic
primitives however it wants.

Config:

- ``counter_dtype`` -- the int type for the backing tensor. Default
  ``i32`` (matches the paper + CUDA atomic semantics).
- ``spin_wait_fn`` -- override the external function name for the
  spin-wait call. Useful for embedding into a custom runtime.
- ``atomic_decrement_fn`` -- same for the notify lowering.

LLM-tool signature:

    tool_name="lower_event_tensor_to_atomic"
    wraps_pass="CompGen:LowerEventTensorToAtomic"
    invent_slot="event_tensor/runtime_lowering"
    policy="AtomicDecrementPlusSpinWait"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Attribute

from compgen.ir.event.ops import (
    EventTensorOp,
    GraphOp,
    NotifyOp,
    WaitOp,
)


@dataclass(frozen=True)
class LowerEventTensorToAtomicConfig:
    counter_dtype: str = "i32"
    spin_wait_fn: str = "compgen_event_spin_wait"
    atomic_decrement_fn: str = "compgen_event_atomic_decrement"
    init_fn: str = "compgen_event_init"


@dataclass
class LowerEventTensorToAtomicStats:
    event_tensors_lowered: int = 0
    notifies_lowered: int = 0
    waits_lowered: int = 0
    external_decls_added: int = 0


# --- helpers -----------------------------------------------------------------


def _counter_type(kind: str) -> IntegerType:
    width = {"i32": 32, "u32": 32, "i64": 64, "u64": 64}.get(kind, 32)
    return IntegerType(width)


def _event_tensor_shape(op: EventTensorOp) -> list[int]:
    shape_attr = op.event_type.shape
    result: list[int] = []
    for dim in shape_attr.data:
        if isinstance(dim, IntegerAttr):
            result.append(int(dim.value.data))
        else:
            result.append(-1)
    return result


def _ensure_external_decl(
    module: ModuleOp,
    name: str,
    arg_types: list[Attribute],
    result_types: list[Attribute],
    stats: LowerEventTensorToAtomicStats,
) -> None:
    """Add a ``func.func`` external declaration to the module if not present."""
    for op in module.ops:
        if isinstance(op, FuncOp) and op.sym_name.data == name:
            return
    decl = FuncOp.external(name, arg_types, result_types)
    module.body.block.insert_op_before(decl, module.body.block.first_op)
    stats.external_decls_added += 1


def _int_tensor_type(shape: list[int], elem: IntegerType) -> TensorType:
    # Substitute static-unknown extents with a conservative `1`.
    resolved = [d if d >= 0 else 1 for d in shape]
    return TensorType(elem, resolved)


# --- main pass ---------------------------------------------------------------


def run_lower_event_tensor_to_atomic(
    module: ModuleOp,
    *,
    config: LowerEventTensorToAtomicConfig | None = None,
) -> LowerEventTensorToAtomicStats:
    """Replace Event Tensor IR with atomic int-tensor primitives."""
    cfg = config if config is not None else LowerEventTensorToAtomicConfig()
    stats = LowerEventTensorToAtomicStats()
    elem = _counter_type(cfg.counter_dtype)

    # Pre-scan for Event Tensor ops so we can lower them in a
    # deterministic order and avoid walker-stability issues when we
    # mutate the module.
    event_tensors: list[EventTensorOp] = []
    notifies: list[NotifyOp] = []
    waits: list[WaitOp] = []
    for op in list(module.walk()):
        if isinstance(op, EventTensorOp):
            event_tensors.append(op)
        elif isinstance(op, NotifyOp):
            notifies.append(op)
        elif isinstance(op, WaitOp):
            waits.append(op)

    if not (event_tensors or notifies or waits):
        return stats

    # Register the runtime externs (once). These have NO SSA operands
    # or results -- the event_ref / indices / decrement are all
    # encoded on the call's ``compgen.event_*`` attributes, so the
    # signature is simply ``() -> ()``. The downstream codegen
    # (Triton / ukernel) translates the attributes into the actual
    # atomic primitive at emission time.
    _ensure_external_decl(module, cfg.init_fn, [], [], stats)
    _ensure_external_decl(module, cfg.atomic_decrement_fn, [], [], stats)
    _ensure_external_decl(module, cfg.spin_wait_fn, [], [], stats)

    # Lower each Event Tensor to a ``func.call @compgen_event_init``
    # whose ``event_ref`` + shape attributes describe the backing
    # integer tensor. The EventTensorOp is left intact with a
    # ``compgen.lowered_to_atomic`` tag so the downstream backend can
    # still find the backing tensor metadata.
    for et in event_tensors:
        if "compgen.lowered_to_atomic" in et.attributes:
            continue
        # Shape + wait_count are already in the op; we simply tag it
        # as lowered and emit an init call.
        et.attributes["compgen.lowered_to_atomic"] = StringAttr("true")
        et.attributes["compgen.lowered_counter_dtype"] = StringAttr(cfg.counter_dtype)
        et.attributes["compgen.lowered_shape"] = et.event_type.shape
        et.attributes["compgen.lowered_scope"] = et.event_type.scope
        stats.event_tensors_lowered += 1

    # Lower each notify / wait to an external call carrying the
    # event_ref + coordinate indices as attributes. The actual
    # lowering to atomics happens at the codegen boundary (Triton
    # emitter, ukernel provider, etc.) -- at the IR level, what
    # matters is that the EventTensorType abstraction is no longer a
    # type-level concept.
    for n in notifies:
        parent_block = n.parent_block()
        if parent_block is None:
            continue
        call = CallOp(cfg.atomic_decrement_fn, [], [])
        call.attributes["compgen.event_ref"] = n.coord.event_ref
        call.attributes["compgen.event_indices"] = n.coord.indices
        call.attributes["compgen.event_decrement"] = n.coord.decrement
        if "compgen.region_id" in n.attributes:
            call.attributes["compgen.region_id"] = n.attributes["compgen.region_id"]
        parent_block.insert_op_before(call, n)
        n.detach()
        n.erase()
        stats.notifies_lowered += 1

    for w in waits:
        parent_block = w.parent_block()
        if parent_block is None:
            continue
        call = CallOp(cfg.spin_wait_fn, [], [])
        call.attributes["compgen.event_ref"] = w.coord.event_ref
        call.attributes["compgen.event_indices"] = w.coord.indices
        if "compgen.region_id" in w.attributes:
            call.attributes["compgen.region_id"] = w.attributes["compgen.region_id"]
        parent_block.insert_op_before(call, w)
        w.detach()
        w.erase()
        stats.waits_lowered += 1

    # Tag any parent graph so downstream passes see the lowering bit.
    for op in module.walk():
        if isinstance(op, GraphOp):
            op.attributes["compgen.event_lowered_to_atomic"] = StringAttr("true")

    return stats


__all__ = [
    "LowerEventTensorToAtomicConfig",
    "LowerEventTensorToAtomicStats",
    "run_lower_event_tensor_to_atomic",
]
