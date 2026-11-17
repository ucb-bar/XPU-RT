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


def apply_to_workload(workload, module_name_of, contract: dict) -> dict:
    """Restrict the workload's options to what the compiler can build.

    `module_name_of(op)` returns the op's module name and network as
    (module_name, network), or (None, None) when unknown. Returns counts of
    what was restricted.
    """
    combos = workload.get_machine_combinations()
    machines = workload.machines
    widths = sorted({len(c) for c in combos})
    uniform = uniform_width_ops(contract)
    divides = divides_oc_ops(contract)
    stats = {"oc_excluded_options": 0, "pinned_dispatches": 0, "unpinnable": []}

    def price_out(op, ks):
        op.processing_times = list(op.processing_times)
        for k in ks:
            if k < len(op.processing_times) and op.processing_times[k] < PRICED_OUT_MS:
                op.processing_times[k] = PRICED_OUT_MS

    def usable(op, k):
        return k < len(op.processing_times) and op.processing_times[k] < PRICED_OUT_MS

    groups: dict[tuple[str, int], list] = defaultdict(list)
    for op in workload.operations:
        mod, net = module_name_of(op)
        if not mod:
            continue
        kind = op_of(mod, uniform | divides)
        if kind in divides:
            oc = oc_of(mod)
            if oc:
                bad = [k for k, c in enumerate(combos) if oc % len(c)]
                stats["oc_excluded_options"] += sum(1 for k in bad if usable(op, k))
                price_out(op, bad)
        if kind in uniform and op.operation_id is not None:
            groups[(net, int(op.operation_id))].append(op)

    for key, ops in sorted(groups.items()):
        if len(ops) < 2:
            continue

        def ks(width, op):
            return [k for k, c in enumerate(combos) if len(c) == width and usable(op, k)]

        candidates = [w for w in widths if all(ks(w, op) for op in ops)]
        if not candidates:
            stats["unpinnable"].append(key)
            continue

        def cost(width):
            return sum(min(op.get_duration_for_combination(k, combos, machines)
                           for k in ks(width, op)) for op in ops)

        best = min(candidates, key=lambda w: (cost(w), w))
        for op in ops:
            price_out(op, [k for k, c in enumerate(combos) if len(c) != best])
        stats["pinned_dispatches"] += 1
    return stats


def violations(dispatches: dict, contract: dict) -> list[str]:
    """Contract violations in a schedule's `dispatches` dict, as messages."""
    uniform = uniform_width_ops(contract)
    divides = divides_oc_ops(contract)
    widths: dict[tuple[str, int], set] = defaultdict(set)
    out = []
    for name, d in dispatches.items():
        mod = d.get("module_name") or ""
        kind = op_of(mod, uniform | divides)
        if not kind:
            continue
        w = len([x for x in str(d.get("hardware_target", "")).split("+") if x])
        net = mod.split("$", 1)[0]
        if kind in divides:
            oc = oc_of(mod)
            if oc and oc % w:
                out.append(f"{name}: {kind} with OC={oc} on {w} cores "
                           f"(width must divide OC)")
        if kind in uniform:
            widths[(net, d.get("id"))].add(w)
    for (net, did), ws in sorted(widths.items(), key=lambda kv: str(kv[0])):
        if len(ws) > 1:
            out.append(f"{net} dispatch {did}: instances take widths {sorted(ws)} "
                       f"(one width per dispatch)")
    return out
