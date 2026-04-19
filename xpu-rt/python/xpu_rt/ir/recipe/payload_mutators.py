"""Direct Recipe-IR → Payload-IR mutation pass.

xDSL doesn't ship a Transform Dialect interpreter that can apply the
``transform.structured.*`` scripts :func:`compgen.ir.recipe.lower.lower_recipe`
emits, so the scripts sit there as text. This module bypasses that
gap by walking the recipe ModuleOp and applying each candidate /
propose op DIRECTLY to the payload module via xDSL attribute mutation.

Concrete mutations applied today:

* :class:`FuseOp` and :class:`ProposeFusionOp` →
  every payload op whose ``compgen.region_id`` matches one of the
  fused regions gets a new ``compgen.fused_into`` attribute (a stable
  digest of the fused region set) and its ``compgen.region_id`` is
  rewritten to that digest. Downstream codegen reads ``region_id`` +
  ``fused_into`` and surfaces both, so a different fusion decision
  produces different ``kernels/*.c`` bytes.

* :class:`TileOp` →
  every payload op in the tiled region gets ``compgen.tile_sizes`` set
  to the tuple chosen by the agent.

* :class:`PlaceOnDeviceOp` →
  every payload op in the placed region gets its ``compgen.device``
  attribute updated to ``device_<index>``.

* :class:`ProposeMegakernelSynthesisOp` →
  every payload op in the fused set gets
  ``compgen.megakernel = "<chosen.megakernel_name>"``.

The pass is deliberately additive — it never removes ops or changes
operand SSA chains, so the resulting payload remains structurally
identical to xDSL's original (no surprise verification failures
downstream). The visibility is in the *attribute deltas*, which the
codegen surfaces as comment / dispatch metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Operation, SSAValue

from compgen.ir.recipe.ops_candidate import (
    FuseOp,
    PlaceOnDeviceOp,
    TileOp,
)
from compgen.ir.recipe.ops_propose import (
    ProposeFusionOp,
    ProposeMegakernelSynthesisOp,
    ProposePayload,
)

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class PayloadMutationReport:
    """Per-recipe-op summary of how the agent's decisions hit the payload."""

    fusions_applied: int = 0
    tiles_applied: int = 0
    placements_applied: int = 0
    megakernels_applied: int = 0
    payload_ops_touched: int = 0
    #: Subset of fusions where we actually collapsed N CallOps into 1 — the
    #: SSA chain shrank, intermediate buffers + ops were erased. The
    #: remainder were attribute-only stamps (chain didn't qualify).
    structural_fusions: int = 0
    structural_callees_added: int = 0
    diagnostics: list[str] = field(default_factory=list)

    def total(self) -> int:
        return (
            self.fusions_applied + self.tiles_applied
            + self.placements_applied + self.megakernels_applied
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusions_applied": self.fusions_applied,
            "tiles_applied": self.tiles_applied,
            "placements_applied": self.placements_applied,
            "megakernels_applied": self.megakernels_applied,
            "payload_ops_touched": self.payload_ops_touched,
            "structural_fusions": self.structural_fusions,
            "structural_callees_added": self.structural_callees_added,
            "diagnostics": list(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _digest(parts: list[str]) -> str:
    """Stable short digest of a list of region names."""
    canonical = "|".join(sorted(parts))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def _region_set_from(op: Operation, attr_name: str) -> list[str]:
    arr = op.properties.get(attr_name) or op.attributes.get(attr_name)
    if not isinstance(arr, ArrayAttr):
        return []
    out: list[str] = []
    for entry in arr.data:
        if isinstance(entry, SymbolRefAttr):
            out.append(entry.root_reference.data)
    return out


def _single_region(op: Operation, attr_name: str) -> str | None:
    val = op.properties.get(attr_name) or op.attributes.get(attr_name)
    if isinstance(val, SymbolRefAttr):
        return val.root_reference.data
    return None


def _walk_payload_ops(payload: ModuleOp):
    """Yield every op in the payload module body (deep)."""
    for op in payload.walk():
        # Skip ModuleOp itself (it has no compgen.region_id).
        if op is payload:
            continue
        yield op


def _build_payload_seed_index(payload: ModuleOp) -> dict[str, Operation]:
    """Mirror :func:`compgen.ir.recipe.seed._extract_significant_ops`'s walk
    so we can look up the op that ``payload_region_id`` refers to.

    The seed names ops ``<TypeName>_<N>`` by enumerating the payload
    walk skipping ``ModuleOp`` / ``FuncOp`` / ``ReturnOp``. Recipe
    propose-ops carry references back to those names through
    ``RecipeRegionOp.payload_region_id``. Building the same index
    here is the only stable bridge.
    """
    out: dict[str, Operation] = {}
    counter = 0
    for op in payload.walk():
        # Skip the wrappers seed.py also skips.
        cls_name = type(op).__name__
        if cls_name in {"ModuleOp", "FuncOp", "ReturnOp"}:
            continue
        out[f"{cls_name}_{counter}"] = op
        counter += 1
    return out


def _matches_region(op: Operation, region_set: set[str]) -> bool:
    rid = op.attributes.get("compgen.region_id")
    if not isinstance(rid, StringAttr):
        return False
    return rid.data in region_set


def _build_recipe_to_payload_map(recipe: ModuleOp) -> dict[str, str]:
    """Build ``recipe_sym_name -> payload_region_id`` from RecipeRegionOps.

    The seed generator names recipe regions ``r_0, r_1, ...`` but
    stores the original payload-op identifier in
    ``payload_region_id``. The mutator needs to translate the recipe-
    level reference (which is what propose ops carry) back to the
    payload-level ``compgen.region_id`` string the codegen uses.
    """
    out: dict[str, str] = {}
    for op in recipe.body.block.ops:
        if op.name != "recipe.region":
            continue
        sym = op.properties.get("sym_name") or op.attributes.get("sym_name")
        payload_id = (
            op.properties.get("payload_region_id")
            or op.attributes.get("payload_region_id")
        )
        if isinstance(sym, StringAttr) and isinstance(payload_id, StringAttr):
            out[sym.data] = payload_id.data
    return out


def _translate(regions: list[str], recipe_to_payload: dict[str, str]) -> list[str]:
    """Map a list of recipe-level region refs to payload-level region_ids.

    A name not found in the map passes through verbatim — this lets a
    caller pass payload-level names directly when they already know them.
    """
    return [recipe_to_payload.get(r, r) for r in regions]


# ---------------------------------------------------------------------------
# Per-recipe-op appliers
# ---------------------------------------------------------------------------


def _resolve_targets(
    region_refs: list[str],
    *,
    by_compgen_id: dict[str, list[Operation]],
    by_seed_index: dict[str, Operation],
) -> list[Operation]:
    """Translate a list of recipe-level region refs to concrete payload ops.

    Two lookup strategies, tried in order:
      1. Match against ``compgen.region_id`` strings on payload ops
         (e.g. ``matmul_0``, ``add_3``) — the form import_fx stamps.
      2. Match against the seed-generated synthetic ID
         (``CallOp_0``, ``EmptyOp_3``, ...) — the form
         :func:`compgen.ir.recipe.seed._extract_significant_ops` uses
         when constructing ``RecipeRegionOp.payload_region_id``.

    Either form may appear in propose-op payloads, so we accept both.
    """
    out: list[Operation] = []
    for ref in region_refs:
        if not ref:
            continue
        hits = by_compgen_id.get(ref)
        if hits:
            out.extend(hits)
            continue
        hit = by_seed_index.get(ref)
        if hit is not None:
            out.append(hit)
    # Stable ordering by walk-position via id(); dedupe.
    seen: set[int] = set()
    uniq: list[Operation] = []
    for op in out:
        if id(op) in seen:
            continue
        seen.add(id(op))
        uniq.append(op)
    return uniq


def _apply_fusion(
    targets: list[Operation], region_refs: list[str], *,
    fusion_kind: str = "producer_consumer",
    label: str = "fusion",
) -> tuple[int, str]:
    """Stamp ``targets`` with shared fused_into + region_id.

    Returns ``(touched_count, fused_id)``. The ``fused_id`` digest is
    derived from the AGENT-supplied region refs (not the resolved ops)
    so the same recipe → same id, regardless of payload walk order.
    """
    if not targets:
        return 0, ""
    fused_id = f"fused_{label}_{_digest(region_refs)}"
    fk_attr = StringAttr(fusion_kind)
    fid_attr = StringAttr(fused_id)
    for op in targets:
        op.attributes["compgen.fused_into"] = fid_attr
        op.attributes["compgen.region_id"] = fid_attr
        op.attributes["compgen.fusion_kind"] = fk_attr
    return len(targets), fused_id


def _apply_tile(
    targets: list[Operation], sizes: tuple[int, ...],
) -> int:
    if not targets or not sizes:
        return 0
    sizes_attr = ArrayAttr([
        IntegerAttr(int(s), IntegerType(64)) for s in sizes
    ])
    sizes_str = ",".join(str(s) for s in sizes)
    str_attr = StringAttr(sizes_str)
    for op in targets:
        op.attributes["compgen.tile_sizes"] = sizes_attr
        op.attributes["compgen.tile_sizes_str"] = str_attr
    return len(targets)


def _apply_place(
    targets: list[Operation], device_index: int,
) -> int:
    if not targets:
        return 0
    dev_attr = StringAttr(f"device_{device_index}")
    placed_attr = StringAttr("recipe")
    for op in targets:
        op.attributes["compgen.device"] = dev_attr
        op.attributes["compgen.placed_by"] = placed_attr
    return len(targets)


# ---------------------------------------------------------------------------
# Structural fusion: actually collapse SSA chains, not just stamp attributes.
# ---------------------------------------------------------------------------


def _sanitize_callee(name: str) -> str:
    """Make ``name`` safe for use as a C identifier — used when synthesising
    the fused callee symbol."""
    out: list[str] = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    if out and out[0].isdigit():
        out.insert(0, "_")
    return "".join(out) or "fn"


def _ensure_callee_decl(
    module: ModuleOp, name: str, in_types: list[Any], out_types: list[Any],
) -> None:
    """Add a private ``func.func`` declaration for ``name`` to the module
    body if one isn't already present. Mirrors how torch.export's import
    pass declares aten passthroughs (body-less ``FuncOp.external``)."""
    for op in module.body.block.ops:
        if isinstance(op, FuncOp) and op.sym_name.data == name:
            return
    # ``FuncOp.external(name, ins, outs)`` builds a body-less declaration
    # that passes xDSL's verifier. The plain ``FuncOp(...)`` ctor builds
    # an empty body which fails verify on most contexts.
    decl = FuncOp.external(name, in_types, out_types)
    # Insert near the top so the C codegen sees declarations before the
    # main body that calls them.
    first = module.body.block.first_op
    if first is not None:
        module.body.block.insert_op_before(decl, first)
    else:
        module.body.block.add_op(decl)


def _try_structural_fuse(
    module: ModuleOp,
    targets: list[Operation],
    *,
    fused_id: str,
    fusion_kind: str,
) -> tuple[bool, str]:
    """Try to merge a producer-consumer chain of ``func.call`` ops.

    Conditions for structural fusion:

    1. Every target is a :class:`CallOp` (the dominant op family in our
       captured Gemma payload).
    2. They share a parent block.
    3. They form a producer-consumer chain in block order: the result
       of target[i] is one of target[i+1]'s operands.
    4. Each non-last result has exactly one use (the next op) — required
       so we can erase the producer without breaking other consumers.

    On success: erases the N CallOps and inserts ONE new CallOp to
    a synthesised fused callee, registered with a private ``func.func``
    declaration on the module so the C codegen emits an extern proto
    for it. Returns ``(True, fused_callee_name)``.

    On failure: leaves the IR untouched and returns ``(False, "")`` —
    the caller falls back to attribute-only stamping.
    """
    if len(targets) < 2:
        return False, ""
    # Condition 1: all CallOp.
    if not all(isinstance(op, CallOp) for op in targets):
        return False, ""
    # Condition 2: same parent block.
    parent = targets[0].parent
    if parent is None or not all(op.parent is parent for op in targets):
        return False, ""

    # Order targets by their position in the block.
    block_op_list = list(parent.ops)
    pos = {id(op): i for i, op in enumerate(block_op_list)}
    ordered = sorted(targets, key=lambda o: pos[id(o)])

    # Condition 3 + 4: each non-last result must be the only consumer
    # of the next op's operand list.
    chain_results: list[SSAValue] = []
    for i in range(len(ordered) - 1):
        prod = ordered[i]
        cons = ordered[i + 1]
        if not prod.results:
            return False, ""
        prod_res = prod.results[0]
        if not any(arg is prod_res for arg in cons.operands):
            return False, ""
        if not prod_res.has_one_use():
            # Some op outside the chain still needs this value; safe
            # erasure isn't possible.
            return False, ""
        chain_results.append(prod_res)

    # Final op must produce a single result we can hand back to the
    # rest of the function body.
    last = ordered[-1]
    if not last.results:
        return False, ""
    last_res = last.results[0]

    # Build fused callee name from the constituent callees.
    callees = [_sanitize_callee(op.callee.root_reference.data) for op in ordered]
    fused_callee = "fused_" + "__".join(callees)

    # Inputs to the fused call: every operand to any op in the chain
    # that isn't itself produced inside the chain. Preserve order.
    in_chain_results = {id(v) for v in chain_results}
    new_operands: list[SSAValue] = []
    new_operand_types: list[Any] = []
    seen_ids: set[int] = set()
    for op in ordered:
        for arg in op.operands:
            if id(arg) in in_chain_results:
                continue
            if id(arg) in seen_ids:
                continue
            seen_ids.add(id(arg))
            new_operands.append(arg)
            new_operand_types.append(arg.type)

    # Output of the fused call mirrors the last op's result type.
    new_result_type = last_res.type

    # Declare the fused callee on the module (idempotent).
    _ensure_callee_decl(
        module, fused_callee, new_operand_types, [new_result_type],
    )

    # Build the new fused CallOp + stamp the agent-decision attributes
    # on it so the C codegen surfaces the fusion in the comment trail.
    new_call = CallOp(
        callee=fused_callee,
        arguments=new_operands,
        return_types=[new_result_type],
    )
    new_call.attributes["compgen.region_id"] = StringAttr(fused_id)
    new_call.attributes["compgen.fused_into"] = StringAttr(fused_id)
    new_call.attributes["compgen.fusion_kind"] = StringAttr(fusion_kind)
    new_call.attributes["compgen.fused_callees"] = StringAttr(",".join(callees))
    # Carry forward the device assignment from the first targeted op so
    # placement decisions aren't lost.
    dev = ordered[0].attributes.get("compgen.device")
    if dev is not None:
        new_call.attributes["compgen.device"] = dev

    # Insert before the FIRST targeted op so subsequent uses see it.
    first_target = ordered[0]
    parent.insert_op_before(new_call, first_target)

    # Rewire: the LAST op's result is what the rest of the function uses.
    last_res.replace_by(new_call.results[0])

    # Erase old ops in REVERSE order (consumer first, then producer)
    # so each erase sees has_one_use==False on its result.
    for op in reversed(ordered):
        # Detach + drop references defensively before erase.
        op.detach()
        op.drop_all_references()
    return True, fused_callee


def _apply_megakernel(
    targets: list[Operation], megakernel_name: str,
) -> int:
    if not targets or not megakernel_name:
        return 0
    name_attr = StringAttr(megakernel_name)
    member_attr = StringAttr("yes")
    for op in targets:
        op.attributes["compgen.megakernel"] = name_attr
        op.attributes["compgen.megakernel_member"] = member_attr
    return len(targets)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_recipe_to_payload(
    recipe: ModuleOp, payload: ModuleOp,
) -> PayloadMutationReport:
    """Walk ``recipe`` and apply every supported op to ``payload`` in place.

    Idempotent in the sense that applying the same recipe twice
    produces the same result (digests are deterministic, the writes
    are overwrites).
    """
    report = PayloadMutationReport()
    if recipe is None or payload is None:
        return report

    # Build TWO indices over the payload module so we can resolve
    # whichever name form the recipe carries:
    #   * by_compgen_id: ``compgen.region_id`` string → list of payload ops
    #     (e.g. "matmul_0", "add_3" — set by import_fx)
    #   * by_seed_index: ``<TypeName>_<N>`` → payload op
    #     (e.g. "CallOp_0" — set by seed._extract_significant_ops)
    by_compgen_id: dict[str, list[Operation]] = {}
    by_seed_index: dict[str, Operation] = {}
    seed_counter = 0
    for op in payload.walk():
        cls_name = type(op).__name__
        if cls_name not in {"ModuleOp", "FuncOp", "ReturnOp"}:
            by_seed_index[f"{cls_name}_{seed_counter}"] = op
            seed_counter += 1
        rid = op.attributes.get("compgen.region_id")
        if isinstance(rid, StringAttr):
            by_compgen_id.setdefault(rid.data, []).append(op)

    # The recipe ALSO carries an explicit recipe→payload region map via
    # RecipeRegionOp.payload_region_id. We thread it so any propose op
    # that uses the synthetic recipe sym (``r_0``, ``r_1``, ...) gets
    # resolved correctly.
    recipe_to_payload = _build_recipe_to_payload_map(recipe)

    def _resolve(refs: list[str]) -> list[Operation]:
        translated = _translate(refs, recipe_to_payload)
        return _resolve_targets(
            translated,
            by_compgen_id=by_compgen_id,
            by_seed_index=by_seed_index,
        )

    def _do_fusion(
        refs: list[str], *, fusion_kind: str, label: str,
    ) -> None:
        """Resolve targets, try a real SSA-collapsing fusion, fall back
        to attribute-stamping when the chain doesn't qualify."""
        targets = _resolve(refs)
        if not targets:
            return
        fused_id = f"fused_{label}_{_digest(refs)}"
        ok, callee = _try_structural_fuse(
            payload, targets,
            fused_id=fused_id, fusion_kind=fusion_kind,
        )
        if ok:
            report.fusions_applied += 1
            report.structural_fusions += 1
            report.structural_callees_added += 1
            report.payload_ops_touched += len(targets)
            report.diagnostics.append(
                f"structural_fuse {label}: {len(targets)} CallOps -> {callee}"
            )
            # Refresh indices (we erased the old ops + added a new one).
            by_compgen_id.setdefault(fused_id, []).append(
                # The newly-inserted CallOp is at the same position the
                # first target used to occupy; re-walk to be safe.
                next((o for o in payload.walk()
                      if o.attributes.get("compgen.region_id") is not None
                      and o.attributes["compgen.region_id"].data == fused_id),
                     None)  # type: ignore[arg-type]
            )
            # Re-build the seed-index lazily on the next op (cheap walk).
            return
        # Fallback: attribute-only stamping on the resolved targets.
        n, _ = _apply_fusion(
            targets, refs, fusion_kind=fusion_kind, label=label,
        )
        if n:
            report.fusions_applied += 1
            report.payload_ops_touched += n
            report.diagnostics.append(
                f"attribute_fuse {label}: {n} ops stamped"
            )

    for op in recipe.body.block.ops:
        if isinstance(op, ProposeFusionOp):
            refs = _region_set_from(op, "grouped_regions")
            try:
                payload_data = ProposePayload.from_json(op.payload.data)
                fk = str(payload_data.chosen.get(
                    "fusion_kind", "producer_consumer",
                ))
            except Exception:   # noqa: BLE001
                fk = "producer_consumer"
            label = (op.sym_name.data
                     if op.sym_name is not None else "propose_fusion")
            _do_fusion(refs, fusion_kind=fk, label=label)

        elif isinstance(op, FuseOp):
            refs = _region_set_from(op, "fuse_regions")
            label = (op.sym_name.data
                     if op.sym_name is not None else "fuse")
            _do_fusion(refs, fusion_kind="producer_consumer", label=label)

        elif isinstance(op, TileOp):
            ref = _single_region(op, "region_ref")
            targets = _resolve([ref] if ref else [])
            sizes = tuple(
                int(s.value.data) for s in op.tile_sizes.data
                if isinstance(s, IntegerAttr)
            )
            n = _apply_tile(targets, sizes)
            if n:
                report.tiles_applied += 1
                report.payload_ops_touched += n

        elif isinstance(op, PlaceOnDeviceOp):
            ref = _single_region(op, "region_ref")
            targets = _resolve([ref] if ref else [])
            device_index = 0
            try:
                device_index = int(op.device.index.value.data)
            except Exception:   # noqa: BLE001
                pass
            n = _apply_place(targets, device_index)
            if n:
                report.placements_applied += 1
                report.payload_ops_touched += n

        elif isinstance(op, ProposeMegakernelSynthesisOp):
            refs = _region_set_from(op, "fused_region_refs")
            targets = _resolve(refs)
            try:
                payload_data = ProposePayload.from_json(op.payload.data)
                megakernel_name = str(payload_data.chosen.get(
                    "megakernel_name", "agent_megakernel",
                ))
            except Exception:   # noqa: BLE001
                megakernel_name = "agent_megakernel"
            n = _apply_megakernel(targets, megakernel_name)
            if n:
                report.megakernels_applied += 1
                report.payload_ops_touched += n

    log.info(
        "payload.mutate",
        fusions=report.fusions_applied,
        tiles=report.tiles_applied,
        placements=report.placements_applied,
        megakernels=report.megakernels_applied,
        payload_ops_touched=report.payload_ops_touched,
    )
    return report


__all__ = [
    "PayloadMutationReport",
    "apply_recipe_to_payload",
]
