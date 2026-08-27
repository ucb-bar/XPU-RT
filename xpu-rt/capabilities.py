"""Which implementations are legal on which physical cores, and how to turn that
into scheduler combinations without letting two things occupy one core.

Why this module exists
----------------------
On the SpaceMiT K1 the IME matrix unit (`smt.vmadot`) is **not** an independent
engine. It is an instruction available on a subset of the CPU cores. Measured on
the board (`artifacts/k1_bringup/*/ime_capability_probe.txt`), with a per-core
SIGILL probe:

    cores 0-3  (L2 domain 0)   scalar, rvv, ime      <- executes
    cores 4-7  (L2 domain 1)   scalar, rvv           <- SIGILL

So an IME dispatch running on core 2 *occupies core 2*. Nothing else may run
there at the same time. The tempting way to give the scheduler an IME option --
declaring `{"ime": 4}` as four more machines alongside the CPU cores -- creates
exactly the bug that makes a schedule physically impossible: the IME "machine"
is busy while the core it actually lives on is still marked idle.

The fix leans on an invariant the scheduler already has.
`Workload.combinations_overlap` (workload.py) is set intersection over machine
*names*, and the MILP (scheduler.py) and the greedy scheduler both refuse to
overlap two combinations that intersect. So if the RVV option and the IME option
for a core are two combinations that **share the same machine set**, they
intersect by construction and are serialised for free -- no solver changes, and
double-booking is not representable rather than merely discouraged.

    combination i: ['CPU_P#0']  impl 'rvv'
    combination j: ['CPU_P#0']  impl 'ime'
    -> combinations_overlap(i, j) is True, always.

The second job here is legality. An IME kernel dispatched to cluster 1 does not
run slowly on real hardware -- it traps with SIGILL. That has to be rejected
when the workload is built, not discovered at runtime.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Measured on the board, not assumed. See the module docstring.
K1_CAPABILITIES: Dict[str, frozenset] = {
    "CPU_P": frozenset({"scalar", "rvv", "ime"}),   # cluster 0, cores 0-3
    "CPU_E": frozenset({"scalar", "rvv"}),          # cluster 1, cores 4-7
}

# Physical core ids per cluster, for affinity masks handed to the runtime.
K1_CLUSTER_CORE_IDS: Dict[str, Tuple[int, ...]] = {
    "CPU_P": (0, 1, 2, 3),
    "CPU_E": (4, 5, 6, 7),
}


class IllegalPlacement(ValueError):
    """An implementation was requested on a machine kind that cannot run it."""


def machine_type_prefix(machine_name: str) -> str:
    """'CPU_P#2' -> 'CPU_P'. Mirrors workload_factory.machine_type_prefix."""
    return machine_name.split("#")[0] if "#" in machine_name else machine_name


def check_implementation_legality(
    machine_impls: Dict[str, Sequence[str]],
    capabilities: Dict[str, frozenset] | None = None,
) -> None:
    """Reject illegal (machine kind, implementation) pairs up front.

    Every violation is reported in one exception rather than failing on the
    first, so a misconfigured sweep tells you everything that is wrong in one
    run instead of one item per edit-run cycle.
    """
    caps = K1_CAPABILITIES if capabilities is None else capabilities
    violations: List[str] = []
    for kind, impls in machine_impls.items():
        allowed = caps.get(kind)
        if allowed is None:
            violations.append(
                f"{kind}: unknown machine kind (known: {sorted(caps)})"
            )
            continue
        for impl in impls:
            if impl not in allowed:
                violations.append(
                    f"{kind}: implementation {impl!r} is not available there "
                    f"(that kind supports {sorted(allowed)})"
                )
    if violations:
        raise IllegalPlacement(
            "illegal implementation placement:\n  " + "\n  ".join(violations)
        )


def build_machine_combinations_with_impls(
    machine_core_counts: Dict[str, int],
    machine_impls: Dict[str, Sequence[str]],
    capabilities: Dict[str, frozenset] | None = None,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Cumulative core-group combinations, once per legal implementation.

    Returns ``(machines, combinations, combo_impls)`` where ``combo_impls[i]``
    is the implementation combination ``i`` runs. Combinations that differ only
    by implementation deliberately carry the **same** machine names, so
    ``combinations_overlap`` serialises them.

    For ``{'CPU_P': 2}`` with impls ``['rvv', 'ime']``::

        [['CPU_P#0'],              'rvv']
        [['CPU_P#0','CPU_P#1'],    'rvv']
        [['CPU_P#0'],              'ime']
        [['CPU_P#0','CPU_P#1'],    'ime']

    The per-kind ordering matches ``workload_factory.build_machine_combinations``
    so that a single-implementation config produces byte-identical combinations
    to the existing code path.
    """
    check_implementation_legality(machine_impls, capabilities)

    machines: List[str] = []
    for kind, count in machine_core_counts.items():
        machines.extend(f"{kind}#{i}" for i in range(count))

    combinations: List[List[str]] = []
    combo_impls: List[str] = []
    for kind, count in machine_core_counts.items():
        cores = [f"{kind}#{i}" for i in range(count)]
        impls = list(machine_impls.get(kind) or ())
        if not impls:
            raise IllegalPlacement(
                f"{kind}: no implementations declared; it would be unschedulable"
            )
        for impl in impls:
            for n in range(1, count + 1):
                combinations.append(cores[:n])
                combo_impls.append(impl)
    return machines, combinations, combo_impls


def legal_combination_indices(
    combinations: Sequence[Sequence[str]],
    combo_impls: Sequence[str],
    required_impl: str,
) -> List[int]:
    """Indices of combinations running ``required_impl``.

    Used to constrain a dispatch that only has a kernel for one implementation.
    """
    return [i for i, impl in enumerate(combo_impls) if impl == required_impl]


def core_ids_for_combination(
    combo: Sequence[str],
    cluster_core_ids: Dict[str, Tuple[int, ...]] | None = None,
) -> List[int]:
    """Physical core ids a combination occupies, for `sched_setaffinity`.

    'CPU_P#2' is the third core of cluster 0, i.e. physical core 2; 'CPU_E#1' is
    the second core of cluster 1, i.e. physical core 5. The runtime needs the
    physical id, not the per-kind index.
    """
    ids = K1_CLUSTER_CORE_IDS if cluster_core_ids is None else cluster_core_ids
    out: List[int] = []
    for name in combo:
        kind = machine_type_prefix(name)
        idx = int(name.split("#")[1]) if "#" in name else 0
        cluster = ids.get(kind)
        if cluster is None:
            raise IllegalPlacement(f"no physical core ids known for kind {kind!r}")
        if idx >= len(cluster):
            raise IllegalPlacement(
                f"{name}: cluster {kind} has only {len(cluster)} cores"
            )
        out.append(cluster[idx])
    return out
