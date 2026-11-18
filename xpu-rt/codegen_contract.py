"""What ModelBlaster can generate code for, applied to the scheduler.

The core registry says what the hardware can run. ModelBlaster's
`cores/codegen_contract.json` says what the compiler can build, and the two
differ once dispatches may take several cores. A convolution's weights are
packed for its core width when the model is generated, so:

  * every periodic instance of one convolution dispatch must take the same
    width (`uniform_width_across_instances`), and
  * that width must divide the convolution's output channels
    (`width_divides_output_channels`).

A schedule that breaks either is valid for the runtime and cannot be built.
`apply_to_workload` keeps the solver inside the contract by pricing out the
options that would break it; `violations` checks a finished schedule.

Three more rules (k3-work TASK-016):

  * `shareable_ops` (hard): only ops with a pool path in ModelBlaster can use
    several cores; every other op gets no multi-core option at all. An op under
    `backend_prefixes` (TASK-017: the simple layers, pool paths on the K3 backends
    only) is shareable only where its module name names one of those backends.
  * `conv_split_axis`: on a K3 backend a conv2d_s8 with enough output rows is
    split by rows, so its width need not divide OC.
  * the width margin (soft, `margin`, default 10%): a wider width is offered
    only if it is at least `margin` faster than the next narrower width kept;
    a convolution's one width is the widest that passes. Exact ties already
    went to the smaller width; this extends that to near-ties, where profile
    noise would otherwise buy a wider group for no real gain.

The contract is read from `XPURT_CODEGEN_CONTRACT`, else from the ModelBlaster
checkout inside this repo (zephyr-chipyard-sw/modelblaster/cores/).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_REPO, "zephyr-chipyard-sw", "modelblaster", "cores",
                            "codegen_contract.json")

#: Cost (ms) given to an option the contract rules out. Every solver reads
#: per-combination processing times, so a cost no schedule would pick removes
#: the option from all of them.
PRICED_OUT_MS = 1.0e6

_OC = re.compile(r"(?:^|[x_])OC(\d+)(?:x|$)")
_OH = re.compile(r"(?:^|[x_])OH(\d+)(?:x|$)")
_OW = re.compile(r"(?:^|[x_])OW(\d+)(?:x|$)")

#: Default width margin: a wider width must be >= 10% faster than the next
#: narrower one kept (scheduler.width_margin overrides).
DEFAULT_MARGIN = 0.10


class ContractUnavailable(RuntimeError):
    """The contract file is missing or unreadable."""


def contract_path() -> str:
    return os.environ.get("XPURT_CODEGEN_CONTRACT") or DEFAULT_PATH


def load(path: str | None = None) -> dict:
    p = path or contract_path()
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ContractUnavailable(
            f"codegen contract not readable at {p} ({e}); set XPURT_CODEGEN_CONTRACT "
            f"to ModelBlaster's cores/codegen_contract.json") from e


def _rule_ops(contract: dict, rule: str) -> frozenset[str]:
    r = contract.get("rules", {}).get(rule, {})
    return frozenset(r.get("applies_to_ops") or r.get("ops") or ())


def uniform_width_ops(contract: dict) -> frozenset[str]:
    return _rule_ops(contract, "uniform_width_across_instances")


def divides_oc_ops(contract: dict) -> frozenset[str]:
    return _rule_ops(contract, "width_divides_output_channels")


def op_of(module_name: str, ops) -> str:
    """The op of `<net>$dispatch_<id>_<backend>_<op>_<SHAPE>` among `ops`, or "".

    Matched as a delimited substring because backend tags contain underscores
    too (rvv_k3_x100); the longest match wins, so conv2d_batchnorm2d_s8 is not
    mistaken for conv2d_s8.
    """
    mod = str(module_name or "")
    hits = [op for op in ops if f"_{op}_" in mod or mod.endswith(f"_{op}")]
    return max(hits, key=len) if hits else ""


def oc_of(module_name: str) -> int | None:
    m = _OC.search(str(module_name or "").split("$")[-1])
    return int(m.group(1)) if m else None


def shareable_ops(contract: dict) -> frozenset[str] | None:
    """Ops with a pool path (rules.shareable_ops), or None if the contract has no list."""
    r = contract.get("rules", {}).get("shareable_ops")
    return frozenset(r.get("ops") or ()) if r else None


def op_shareable(module_name: str, kind: str, contract: dict) -> bool:
    """Can `kind` (the op of `module_name`) be spread over several cores on the
    dispatch's backend? True when the contract has no shareable_ops list."""
    r = contract.get("rules", {}).get("shareable_ops")
    if not r:
        return True
    if kind not in frozenset(r.get("ops") or ()):
        return False
    prefixes = (r.get("backend_prefixes") or {}).get(kind)
    return prefixes is None or any(f"_{p}" in str(module_name or "") for p in prefixes)


def row_split(module_name: str, width: int, contract: dict) -> bool:
    """Does ModelBlaster split this dispatch by output rows at `width`?
    (rules.conv_split_axis: a K3 backend, a conv2d_s8, OH >= width and
    OH*OW >= min_pixels; ModelBlaster's pipeline/conv_split.py.)"""
    r = contract.get("rules", {}).get("conv_split_axis")
    if not r or width <= 1:
        return False
    mod = str(module_name or "")
    if op_of(mod, frozenset(r.get("applies_to_ops") or ())) == "":
        return False
    if not any(f"_{p}" in mod for p in r.get("backend_prefixes") or ()):
        return False
    tail = mod.split("$")[-1]
    oh, ow = _OH.search(tail), _OW.search(tail)
    if not (oh and ow):
        return False
    oh, ow = int(oh.group(1)), int(ow.group(1))
    return oh >= width and oh * ow >= int(r.get("min_pixels", 0))


def margin_widths(cost_by_width: dict[int, float], margin: float) -> list[int]:
    """The widths a dispatch keeps under the margin rule, narrowest first: the
    narrowest always, then each wider width that is at least `margin` faster
    than the widest width kept so far."""
    kept: list[int] = []
    for w in sorted(cost_by_width):
        if not kept or cost_by_width[w] <= (1.0 - margin) * cost_by_width[kept[-1]]:
            kept.append(w)
    return kept


def apply_to_workload(workload, module_name_of, contract: dict,
                      margin: float = 0.0) -> dict:
    """Restrict the workload's options to what the compiler can build.

    `module_name_of(op)` returns the op's module name and network as
    (module_name, network), or (None, None) when unknown. `margin` > 0 also
    applies the width margin (module docstring). Returns counts of what was
    restricted.
    """
    combos = workload.get_machine_combinations()
    machines = workload.machines
    widths = sorted({len(c) for c in combos})
    uniform = uniform_width_ops(contract)
    divides = divides_oc_ops(contract)
    share = shareable_ops(contract)
    known = uniform | divides | (share or frozenset())
    stats = {"oc_excluded_options": 0, "pinned_dispatches": 0, "unpinnable": [],
             "unshareable_options": 0, "margin_excluded_options": 0,
             "margin_narrowed_dispatches": 0}

    def price_out(op, ks):
        op.processing_times = list(op.processing_times)
        for k in ks:
            if k < len(op.processing_times) and op.processing_times[k] < PRICED_OUT_MS:
                op.processing_times[k] = PRICED_OUT_MS

    def usable(op, k):
        return k < len(op.processing_times) and op.processing_times[k] < PRICED_OUT_MS

    groups: dict[tuple[str, int], list] = defaultdict(list)
    free_groups: dict[tuple[str, int], list] = defaultdict(list)
    for op in workload.operations:
        mod, net = module_name_of(op)
        if not mod:
            continue
        kind = op_of(mod, known)
        if share is not None and not op_shareable(mod, kind, contract):
            # No pool path (on this backend): a multi-core option would hold idle cores.
            bad = [k for k, c in enumerate(combos) if len(c) > 1]
            stats["unshareable_options"] += sum(1 for k in bad if usable(op, k))
            price_out(op, bad)
            continue
        if kind in divides:
            oc = oc_of(mod)
            if oc:
                bad = [k for k, c in enumerate(combos)
                       if oc % len(c) and not row_split(mod, len(c), contract)]
                stats["oc_excluded_options"] += sum(1 for k in bad if usable(op, k))
                price_out(op, bad)
        if op.operation_id is None:
            continue
        if kind in uniform:
            groups[(net, int(op.operation_id))].append(op)
        elif margin > 0:
            free_groups[(net, int(op.operation_id))].append(op)

    def ks_any(width, op):
        return [k for k, c in enumerate(combos) if len(c) == width and usable(op, k)]

    def group_cost(width, ops):
        return sum(min(op.get_duration_for_combination(k, combos, machines)
                       for k in ks_any(width, op)) for op in ops)

    # Runtime-sliceable ops (linear): each instance keeps its own width choice,
    # but only among the widths that pass the margin.
    for key, ops in sorted(free_groups.items()):
        cand = [w for w in widths if all(ks_any(w, op) for op in ops)]
        if len(cand) < 2:
            continue
        kept = margin_widths({w: group_cost(w, ops) for w in cand}, margin)
        for op in ops:
            bad = [k for k, c in enumerate(combos) if len(c) not in kept]
            stats["margin_excluded_options"] += sum(1 for k in bad if usable(op, k))
            price_out(op, bad)

    for key, ops in sorted(groups.items()):
        if len(ops) < 2 and margin <= 0:
            continue
        candidates = [w for w in widths if all(ks_any(w, op) for op in ops)]
        if not candidates:
            stats["unpinnable"].append(key)
            continue
        if margin > 0:
            # The widest width whose every step from the narrowest is >= margin
            # faster: the cheapest of the kept widths by construction.
            best = margin_widths({w: group_cost(w, ops) for w in candidates}, margin)[-1]
            cheapest = min(candidates, key=lambda w: (group_cost(w, ops), w))
            if cheapest != best:
                stats["margin_narrowed_dispatches"] += 1
        else:
            best = min(candidates, key=lambda w: (group_cost(w, ops), w))
        for op in ops:
            price_out(op, [k for k, c in enumerate(combos) if len(c) != best])
        stats["pinned_dispatches"] += 1
    return stats


def violations(dispatches: dict, contract: dict) -> list[str]:
    """Contract violations in a schedule's `dispatches` dict, as messages."""
    uniform = uniform_width_ops(contract)
    divides = divides_oc_ops(contract)
    share = shareable_ops(contract)
    widths: dict[tuple[str, int], set] = defaultdict(set)
    out = []
    for name, d in dispatches.items():
        mod = d.get("module_name") or ""
        w = len([x for x in str(d.get("hardware_target", "")).split("+") if x])
        if share is not None and w > 1 and mod and not op_shareable(mod, op_of(mod, share), contract):
            out.append(f"{name}: no pool path for this op, but given {w} cores")
            continue
        kind = op_of(mod, uniform | divides)
        if not kind:
            continue
        net = mod.split("$", 1)[0]
        if kind in divides:
            oc = oc_of(mod)
            if oc and oc % w and not row_split(mod, w, contract):
                out.append(f"{name}: {kind} with OC={oc} on {w} cores "
                           f"(width must divide OC)")
        if kind in uniform:
            widths[(net, d.get("id"))].add(w)
    for (net, did), ws in sorted(widths.items(), key=lambda kv: str(kv[0])):
        if len(ws) > 1:
            out.append(f"{net} dispatch {did}: instances take widths {sorted(ws)} "
                       f"(one width per dispatch)")
    return out
