"""What the compiler can BUILD, as a constraint the scheduler can be held to.

`ModelBlaster/cores/spacemit_k1.json` says what the hardware can EXECUTE. This module
reads its sibling `cores/codegen_contract.json`, which says what ModelBlaster can
GENERATE CODE FOR, and checks a solved schedule against it.

WHY IT IS NEEDED. The scheduler's option space was wider than the compiler's and nothing
connected the two. `shard` machine-combination mode lets every periodic INSTANCE of a
dispatch choose its own aligned core block; for a convolution that is unbuildable,
because the packed weight array is materialised per shard while generating the skeleton,
so the width has to be one value per dispatch. The solver did not know, so it produced
schedules that were valid for the runtime and impossible for the compiler -- and the
board build discovered it at stage 1 of 5, after extracting and generating sources for
every model, with a ValueError raised from inside a shell script.

TWO WAYS TO USE IT, and the first is better:

  * `uniform_width_dispatch_ids(spec_or_graphs)` -- ask BEFORE solving which dispatches
    must take one width, and constrain the search. A schedule that cannot violate the
    contract needs no gate.
  * `violations(schedule)` -- ask AFTER solving. Milliseconds, and it names the network
    and dispatch instead of failing deep inside a build.

The contract is DATA, deliberately: adding a packed-weight op, or an op that turns out to
be runtime-sliceable after all, is a one-line edit to a json file that both sides read,
rather than two lists in two languages that drift.
"""
from __future__ import annotations

import functools
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

#: Overridable so a different target's contract can be used without editing code.
CONTRACT_PATH = os.environ.get(
    "XPURT_CODEGEN_CONTRACT",
    os.path.join(REPO, "ModelBlaster", "cores", "codegen_contract.json"))

#: `<net>$dispatch_<id>_<backend>_<op>_<SHAPE>`. The op is not its own field on a
#: schedule entry, and the shape tail is what separates it from the backend tag.
_SHAPE_TAIL = re.compile(r"_([NM]\d+x[A-Za-z0-9x]+)$")


class ContractUnavailable(RuntimeError):
    """The contract file is missing or unreadable.

    Raised rather than defaulted, because a silently empty contract is worse than no
    contract: every check passes and the build still fails.
    """


@functools.lru_cache(maxsize=4)
def load(path: str | None = None) -> dict:
    p = path or CONTRACT_PATH
    if not os.path.exists(p):
        raise ContractUnavailable(
            f"no codegen contract at {p}. It ships in the ModelBlaster submodule "
            f"(cores/codegen_contract.json); run `git submodule update --init "
            f"ModelBlaster`, or point XPURT_CODEGEN_CONTRACT at one.")
    try:
        c = json.load(open(p))
    except (OSError, ValueError) as e:
        raise ContractUnavailable(f"could not read the codegen contract at {p}: {e}")
    if not (c.get("rules") or {}):
        raise ContractUnavailable(f"the codegen contract at {p} declares no rules")
    return c


def _rule(contract, name) -> dict:
    return (contract.get("rules") or {}).get(name) or {}


def packed_weight_ops(contract=None) -> frozenset[str]:
    """Ops whose weights are packed per shard, so their width cannot vary."""
    c = contract or load()
    return frozenset(_rule(c, "uniform_width_across_instances").get(
        "applies_to_ops") or ())


def runtime_sliceable_ops(contract=None) -> frozenset[str]:
    c = contract or load()
    return frozenset(_rule(c, "runtime_sliceable_ops").get("ops") or ())


def op_of(module_name: str, ops) -> str:
    """The op in `<net>$dispatch_<id>_<backend>_<op>_<SHAPE>`, or "".

    Matched against the known op names rather than parsed positionally: the backend tag
    itself contains underscores (`rvv_x60`, `ime_x60`), so there is no split that works
    for every entry. The names do not prefix one another -- `conv2d_batchnorm2d_s8` does
    not contain `conv2d_s8` -- so a delimited match is exact.
    """
    mod = str(module_name or "")
    hits = [op for op in ops if f"_{op}_" in mod or mod.endswith(f"_{op}")]
    return max(hits, key=len) if hits else ""


def _oc_of(module_name: str):
    """OC from the shape tail, or None. `...OC32xOH56...` -> 32."""
    m = re.search(r"[x_]OC(\d+)", str(module_name or ""))
    return int(m.group(1)) if m else None


def width_of(entry) -> int:
    return len([x for x in str(entry.get("hardware_target", "")).split("+") if x.strip()])


def net_of(entry) -> str:
    """The NETWORK an entry belongs to, taken from `module_name` where possible.

    `job_name` is `<net><instance>`, and trimming trailing digits off it is wrong
    whenever the network's own name ends in a digit: `yolov8_nano_64x960` (instance 0 of
    `yolov8_nano_64x96`) becomes `yolov8_nano_64x`. That is not hypothetical -- it is the
    name this very checker reported violations under until it was fixed, and the same
    hazard bit the calibration emitter and the board runner's ingest.

    `module_name` carries the authoritative network name before the `$`, so use it and
    fall back to the digit trim only when the entry has no module name.
    """
    mod = str(entry.get("module_name") or "")
    if "$" in mod:
        return mod.split("$", 1)[0]
    job = str(entry.get("job_name", ""))
    return job.rstrip("0123456789") or job


def violations(schedule, contract=None) -> list[dict]:
    """Contract violations in a solved schedule, worst first.

    Each is `{rule, severity, network, dispatch_id, op, detail}`. An empty list means
    the schedule is buildable as far as the contract can tell -- which is not the same
    as "will build": the contract only covers rules whose `checkable_from` is
    "schedule".
    """
    c = contract or load()
    sched = json.load(open(schedule)) if isinstance(schedule, str) else schedule
    packed = packed_weight_ops(c)
    entries = list((sched.get("dispatches") or {}).values())

    widths: dict[tuple[str, int], set[int]] = {}
    ops: dict[tuple[str, int], str] = {}
    mods: dict[tuple[str, int], str] = {}
    for e in entries:
        key = (net_of(e), int(e["id"]))
        widths.setdefault(key, set()).add(width_of(e))
        mod = str(e.get("module_name") or "")
        if mod:
            mods[key] = mod
            ops[key] = op_of(mod, packed | runtime_sliceable_ops(c))

    out = []
    r_uni = _rule(c, "uniform_width_across_instances")
    r_div = _rule(c, "width_divides_output_channels")
    div_ops = frozenset(r_div.get("applies_to_ops") or ())
    for (net, did), ws in sorted(widths.items()):
        op = ops.get((net, did), "")
        if op in packed and len(ws) > 1:
            out.append(dict(
                rule="uniform_width_across_instances",
                severity=r_uni.get("severity", "refuse"), network=net,
                dispatch_id=did, op=op,
                detail=(f"instances take widths {sorted(ws)}; one generated model "
                        f"cannot encode two packed weight layouts for one dispatch")))
        if op in div_ops:
            oc = _oc_of(mods.get((net, did), ""))
            bad = [w for w in sorted(ws) if oc and w and oc % w]
            if bad:
                out.append(dict(
                    rule="width_divides_output_channels",
                    severity=r_div.get("severity", "refuse"), network=net,
                    dispatch_id=did, op=op,
                    detail=(f"OC={oc} is not divisible by width(s) {bad}; a shard's "
                            f"OC slice would have a remainder")))

    r_mc = _rule(c, "machine_combinations")
    if r_mc.get("same_kind_only"):
        for e in entries:
            kinds = {t.split("#")[0] for t in str(e.get("hardware_target", "")).split("+")
                     if t.strip()}
            if len(kinds) > 1:
                out.append(dict(
                    rule="machine_combinations", severity=r_mc.get("severity", "refuse"),
                    network=net_of(e), dispatch_id=int(e["id"]),
                    op=op_of(e.get("module_name"), packed | runtime_sliceable_ops(c)),
                    detail=(f"combination mixes core kinds {sorted(kinds)}; the runtime "
                            f"resolves a slot by taking the n-th core OF ITS KIND, so a "
                            f"mixed combination is attributed to one kind's backend")))

    r_ime = _rule(c, "ime")
    allowed = set(r_ime.get("clusters") or [])
    if allowed:
        # cluster is not on the entry; it is a property of the hart, and the only
        # cluster label the schedule carries is the machine KIND. `rvv_c1` is cluster 1
        # by construction (see the registry's core_kind note).
        for e in entries:
            if str(e.get("impl") or "").lower() != "ime":
                continue
            kinds = {t.split("#")[0] for t in str(e.get("hardware_target", "")).split("+")
                     if t.strip()}
            if any(k.upper().endswith("_E") or k.lower().endswith("_c1") for k in kinds):
                out.append(dict(
                    rule="ime", severity=r_ime.get("severity", "refuse"),
                    network=net_of(e), dispatch_id=int(e["id"]),
                    op=op_of(e.get("module_name"), packed | runtime_sliceable_ops(c)),
                    detail=(f"an IME dispatch is placed on {sorted(kinds)}; smt.vmadot "
                            f"TRAPS outside cluster {sorted(allowed)} -- it does not "
                            f"degrade, it dies")))

    order = {"refuse": 0, "warn": 1, "none": 2}
    out.sort(key=lambda v: (order.get(v["severity"], 9), v["network"], v["dispatch_id"]))
    return out


def uniform_width_dispatch_ids(dispatch_graphs, contract=None) -> dict[str, list[int]]:
    """`{network: [dispatch_id]}` that MUST take one width across all instances.

    Ask this before solving and constrain the search; a schedule that cannot violate the
    contract needs no gate afterwards. `dispatch_graphs` is `{net: path-or-dict}` of the
    dispatch-graph json the workload spec points at.
    """
    c = contract or load()
    packed = packed_weight_ops(c)
    out: dict[str, list[int]] = {}
    for net, g in (dispatch_graphs or {}).items():
        graph = json.load(open(g)) if isinstance(g, str) else g
        nodes = graph.get("dispatches") or graph.get("nodes") or graph.get("ops") or []
        items = nodes.values() if isinstance(nodes, dict) else nodes
        ids = []
        for n in items:
            if not isinstance(n, dict):
                continue
            op = str(n.get("op") or "")
            if not op:
                op = op_of(n.get("module_name") or n.get("name") or "", packed)
            did = n.get("dispatch_id", n.get("id"))
            if op in packed and did is not None:
                ids.append(int(did))
        if ids:
            out[net] = sorted(set(ids))
    return out


def describe(v: dict) -> str:
    return (f"{v['severity'].upper()} {v['rule']}: {v['network']} dispatch "
            f"{v['dispatch_id']}" + (f" ({v['op']})" if v.get("op") else "")
            + f" -- {v['detail']}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("schedule", nargs="+", help="solved schedule json(s) to check")
    ap.add_argument("--contract", default=None)
    a = ap.parse_args()
    c = load(a.contract)
    print(f"contract: {a.contract or CONTRACT_PATH}  ({c.get('schema')})")
    rc = 0
    for s in a.schedule:
        vs = violations(s, c)
        name = os.path.basename(s)
        if not vs:
            print(f"OK   {name}: no contract violation the schedule can show")
            continue
        refuse = [v for v in vs if v["severity"] == "refuse"]
        print(f"{'FAIL' if refuse else 'WARN'} {name}: {len(vs)} violation(s)")
        for v in vs:
            print("       " + describe(v))
        if refuse:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


#: Same sentinel `scheduler_cpsat` treats as an exclusion, so a pinned-out width is
#: excluded in CP-SAT and unaffordable in greedy by the same number.
_PINNED_OUT_COST_MS = 1e8


def packed_weight_groups(ops):
    """`{(network, dispatch_id): [op index, ...]}` for packed-weight dispatches only.

    Shared so the two schedulers group a dispatch's instances the SAME way. The op kind
    is not a field on an Operation; it lives in `operation_name`, and `op_kind` is
    attached from the profile database by `run_xpurt_schedule._annotate_op_kinds`, with
    the module-name parse as the fallback for callers that build a workload directly.
    """
    packed = packed_weight_ops()
    groups = {}
    for i, op in enumerate(ops):
        name = str(getattr(op, "operation_name", "") or "")
        kind = str(getattr(op, "op_kind", "") or "") or op_of(name, packed)
        if kind not in packed:
            continue
        did = getattr(op, "operation_id", None)
        if did is None:
            continue
        net = str(getattr(op, "op_network", "") or "")
        if not net:
            net = (name.split("$")[0] if "$" in name
                   else name.split("_dispatch_")[0])
        groups.setdefault((net, int(did)), []).append(i)
    return groups


def pin_uniform_widths(ops, combos, machines, log=None):
    """Restrict every packed-weight dispatch to ONE core width across its instances,
    by EXCLUDING the other widths. Returns the number of dispatches pinned.

    WHY A LIST SCHEDULER NEEDS THIS AND CP-SAT DOES NOT. CP-SAT has a
    combination-selection variable per op, so the contract can be expressed as a
    constraint coupling a dispatch's instances (`XPURT_UNIFORM_PACKED_WIDTH` in
    `scheduler_cpsat`) and the solver then chooses one width for all of them. A list
    scheduler picks each op's combination greedily as it walks the ready set, with
    nothing to couple, so the contract could only ever be checked AFTERWARDS and the
    whole candidate thrown away.

    That asymmetry was not cosmetic. On `w4_ffn_dronet_sensor` the greedy `shard`
    schedule is the best schedule anyone has produced for that workload -- 5 instance
    misses against the baseline's 10, worst lateness 3.09 ms against 17.95 -- and it was
    discarded for THREE dispatches (dronet 0, 8, 9) whose instances took different
    widths. The loop then reported w4 as a workload where no lever helps.

    Pinning restores the option by construction: excluded widths cannot be chosen, so
    the greedy schedule is uniform-width and passes the contract without a repair pass.

    THE WIDTH IS CHOSEN BY MEASURED COST, not by a rule of thumb. Among the widths every
    instance of the dispatch can actually take, we pick the one minimising the summed
    duration over those instances. Choosing "widest" would be wrong on real data --
    yolo's OC=2 detect-head convs measure SLOWER on four cores than on one -- and
    choosing "narrowest" would silently undo sharding, which is the whole point of the
    lever.
    """
    groups = packed_weight_groups(ops)
    widths = sorted({len(combos[k]) for k in range(len(combos))})
    n_pinned, skipped = 0, []
    for key, idxs in sorted(groups.items()):
        if len(idxs) < 2:
            continue

        def combos_of(width, i):
            return [k for k in range(len(combos))
                    if len(combos[k]) == width
                    and k not in ops[i].infeasible_combinations]

        usable = [w for w in widths if all(combos_of(w, i) for i in idxs)]
        if not usable:
            # No width every instance can take. Pinning cannot make this dispatch
            # buildable, and excluding everything would make the op unschedulable, so
            # leave it and let the contract check report it honestly.
            skipped.append(key)
            continue

        def cost(width):
            total = 0.0
            for i in idxs:
                try:
                    total += min(float(ops[i].get_duration_for_combination(
                        k, combos, machines)) for k in combos_of(width, i))
                except Exception:
                    return float("inf")
            return total

        best = min(usable, key=lambda w: (cost(w), w))
        for i in idxs:
            drop = {k for k in range(len(combos)) if len(combos[k]) != best}
            ops[i].infeasible_combinations = set(
                ops[i].infeasible_combinations) | drop
            # AND PRICE THEM OUT, because the list scheduler does not read
            # `infeasible_combinations` at all -- it picks the combination with the
            # earliest completion, full stop. Flagging alone pinned nothing: the greedy
            # w4 shard schedule still took widths [1, 4] on dronet 0/8/9 and still
            # failed the contract. `processing_times` is what both schedulers actually
            # read (`get_duration_for_combination` is a direct index into it), and
            # CP-SAT already folds a sentinel cost back into its exclusions, so writing
            # the sentinel is the one edit that binds both.
            try:
                for k in drop:
                    ops[i].processing_times[k] = _PINNED_OUT_COST_MS
            except Exception:
                pass
        n_pinned += 1
    if log:
        if skipped:
            log(f"[contract] {len(skipped)} packed-weight dispatch(es) have no width "
                f"every instance can take; left unpinned: {skipped[:4]}")
        if n_pinned:
            log(f"[contract] pinned {n_pinned} packed-weight dispatch(es) to one "
                f"measured-best width across their instances")
    return n_pinned


def restrict_shard_to_networks(ops, combos, machines, allowed, log=None):
    """Price out every multi-core combination for ops whose network is NOT in `allowed`.
    Returns the number of ops restricted.

    WHY A PER-NETWORK SHARD DECISION IS NEEDED. `machine_combination_mode: "shard"` is
    global: it opens multi-hart combinations for every network at once. That is fine
    while widening is free, and wrong as soon as the core budget binds. On
    `w5_ffn_dronet_yolo` the whole-workload shard lever is REJECTED -- worst deadline
    lateness goes from 24.67 ms to 34.87 -- because it also widens `yolov8_nano_64x96`,
    which is 191.6 core-ms of work and monopolises all eight harts while the 5 ms-period
    networks wait. The right decision on that rung is to widen `ffn_block` and `dronet`
    and leave yolo single-core, and with one global switch the loop cannot express it,
    so it takes the whole lever or none of it.

    That is exactly the choice `make_scaling_workloads.py` says the top rungs exist to
    force: "as networks accumulate it has to choose WHICH nets to widen ... a real
    scheduling decision with a right and a wrong answer".

    Restriction is by COST, not by `infeasible_combinations`, for the same reason
    `pin_uniform_widths` prices widths out: the list scheduler never reads the
    infeasibility set, and CP-SAT folds a sentinel cost back into its exclusions, so the
    sentinel is the one edit both schedulers honour.
    """
    allowed = {str(a) for a in (allowed or ())}
    n = 0
    for op in ops:
        net = str(getattr(op, "op_network", "") or "")
        if not net:
            name = str(getattr(op, "operation_name", "") or "")
            net = (name.split("$")[0] if "$" in name
                   else name.split("_dispatch_")[0])
        if net in allowed:
            continue
        touched = False
        for k in range(len(combos)):
            if len(combos[k]) > 1:
                try:
                    op.processing_times[k] = _PINNED_OUT_COST_MS
                    touched = True
                except Exception:
                    pass
                op.infeasible_combinations = set(op.infeasible_combinations) | {k}
        n += 1 if touched else 0
    if log and n:
        log(f"[contract] shard restricted to {sorted(allowed)}: {n} op(s) of other "
            f"networks held at one core")
    return n


def shard_only_networks_from_env():
    """`XPURT_SHARD_ONLY_NETS` as a set, or None when unset (shard everything).

    The literal `none` is the empty set, NOT "unset": it says no network may be
    widened. Without it "hold every network at one core" is inexpressible --
    an empty list reads as "no restriction", so the only way to say it was to
    name a network that happens to be unable to widen anyway, which states the
    intent nowhere and breaks the moment that network's pin changes.
    """
    raw = os.environ.get("XPURT_SHARD_ONLY_NETS", "").strip()
    if not raw:
        return None
    if raw.lower() == "none":
        return set()
    return {p for p in (x.strip() for x in raw.split(",")) if p} or None
