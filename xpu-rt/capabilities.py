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
    granularity: str = "prefix",
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Core-group combinations, once per legal implementation.

    ``granularity`` selects what a combination means:

    ``"prefix"`` (default, and what `workload_factory.build_machine_combinations`
        does): cumulative prefixes ``['CPU_P#0']``, ``['CPU_P#0','CPU_P#1']``, …
        A combination is *one dispatch given N cores*. Because every cluster-0
        combination contains ``CPU_P#0``, they all intersect, so a cluster runs
        **one dispatch at a time**. Timings come from the N-hart profile
        (``topo_0_1_2_3`` for 4 cores).

    ``"per_core"``: every core is independently schedulable —
        ``['CPU_P#0']``, ``['CPU_P#1']``, ``['CPU_P#2']``, … Disjoint, so eight
        dispatches can genuinely run at once on the K1. Timings must come from
        the **single-core** profile (``topo_0``); using a 4-hart number here
        would credit each core with the throughput of the whole cluster.

    ``"shard"``: ``"per_core"`` **plus** aligned power-of-two core blocks, so the
        scheduler may either run a dispatch on one core or spread that single
        dispatch across several. This is what baseline B4 needs: a dispatch
        whose own latency exceeds its period cannot be rescheduled into
        compliance by any placement policy, only by being given more cores.

        Blocks are buddy-aligned (``{0,1}``, ``{2,3}``, ``{0,1,2,3}``) rather
        than every possible subset. That keeps the combination count linear
        instead of exponential, and every block is still a plain machine *set*,
        so ``combinations_overlap`` serialises a block against the singletons
        inside it for free — a 4-core shard on cluster 0 excludes anything else
        on cores 0-3, which is exactly the physical truth.

        Measured on the board: DroNet's 22.8 ms convolution takes 6.1 ms on four
        harts and 3.1 ms on eight, so these blocks are not a modelling
        convenience — IREE really does distribute the workgroups.

    The two answer different questions and need different profiles, so the
    choice belongs in the workload spec rather than being implied.

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

    if granularity not in ("prefix", "per_core", "shard"):
        raise ValueError(
            "granularity must be 'prefix', 'per_core' or 'shard', got "
            f"{granularity!r}"
        )

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
            if granularity == "prefix":
                for n in range(1, count + 1):
                    combinations.append(cores[:n])
                    combo_impls.append(impl)
            elif granularity == "per_core":
                for core in cores:
                    combinations.append([core])
                    combo_impls.append(impl)
            else:
                for block in aligned_core_blocks(cores):
                    combinations.append(block)
                    combo_impls.append(impl)
    return machines, combinations, combo_impls


def aligned_core_blocks(cores: Sequence[str]) -> List[List[str]]:
    """Singletons plus buddy-aligned power-of-two blocks of ``cores``.

    For four cores this is ``[0] [1] [2] [3] [0,1] [2,3] [0,1,2,3]``.

    Alignment is what keeps the set algebra honest *and* small. Every pair of
    blocks is either disjoint or nested, so two blocks that do not intersect can
    genuinely run at the same time, and two that do are serialised by
    ``combinations_overlap``. Allowing unaligned windows such as ``[1,2]`` would
    add partial overlaps that buy no real placement freedom -- the cores are
    interchangeable -- while multiplying the combination count.

    A trailing group smaller than the block size is emitted as-is rather than
    dropped, so a cluster whose core count is not a power of two still offers
    every core to the scheduler.
    """
    out: List[List[str]] = [[c] for c in cores]
    n = len(cores)
    size = 2
    while size <= n:
        for start in range(0, n, size):
            block = list(cores[start:start + size])
            if len(block) > 1:
                out.append(block)
        size *= 2
    return out


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

# Profile-hw labels in a networks JSON are build-variant names -- "RVV",
# "RVV_c1", "RVV_fused", "RVV_split", "IME", "IME_ukernel", "scalar_c1" -- while
# the capability table above is in terms of the ISA feature actually required:
# scalar, rvv, ime. Mapping is by prefix so a new variant of an existing ISA
# needs no change here; only a genuinely new ISA does.
_ISA_PREFIXES = (
    ("ime", "ime"),
    ("rvv", "rvv"),
    ("scalar", "scalar"),
)


def implementation_of_profile_hw(hw_label: str) -> str:
    """'RVV_c1' -> 'rvv', 'IME_ukernel' -> 'ime'. '' for an unknown label.

    Returning '' rather than guessing matters: an unrecognised label must not
    silently normalise to something legal everywhere, or the legality check it
    feeds becomes a no-op for exactly the case it was written to catch.
    """
    low = (hw_label or "").strip().lower()
    for prefix, impl in _ISA_PREFIXES:
        if low.startswith(prefix):
            return impl
    return ""


def check_profile_hw_map(profile_hw_map: Dict[str, str],
                         capabilities: Dict[str, frozenset] | None = None,
                         strict_unknown: bool = False) -> None:
    """Reject an illegal (machine kind, build variant) pairing at config time.

    This is the production entry point. The legality machinery below it was
    written, unit-tested, and then called by nothing -- so a config naming
    `profile_hw: {cpu_e: IME}` would schedule happily and SIGILL on the board's
    first cluster-1 dispatch, which is precisely the failure the capability
    table exists to prevent.

    Unknown labels are reported but not fatal by default, because the machine
    kinds in this repo extend well past the K1 (gemmini, rvv_opu, qrb5165 DSP
    and HTA, ...) and those have no entry in K1_CAPABILITIES. Pass
    ``strict_unknown=True`` on a K1 config to make them fatal.
    """
    caps = K1_CAPABILITIES if capabilities is None else capabilities
    machine_impls: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for kind_lower, hw in (profile_hw_map or {}).items():
        kind = kind_lower.upper()
        if kind not in caps:
            continue  # a machine kind this table says nothing about
        impl = implementation_of_profile_hw(hw)
        if not impl:
            unknown.append(f"{kind_lower}: {hw!r} is not a scalar/rvv/ime variant")
            continue
        machine_impls[kind] = [impl]
    if unknown and strict_unknown:
        raise IllegalPlacement("unrecognised profile_hw label(s):\n  "
                               + "\n  ".join(unknown))
    for u in unknown:
        print(f"WARN capability check: {u}")
    if machine_impls:
        check_implementation_legality(machine_impls, caps)
