"""Parametric synthetic DAG generators for the scheduling experiment.

Each generator returns a tuple shaped to be fed directly into
:func:`xpu_rt.solve.schedule_joint_cpsat.solve_schedule_joint` and into
the greedy / MILP baselines. Per-device cost dispersion is controlled
by a fixed seed so reruns are bit-stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDag:
    """A schedulable DAG ready for solver consumption.

    Attributes:
        partition_ids: Stable topo order of partition IDs.
        dependencies: ``succ_id -> [pred_ids]``.
        durations_us_by_device: ``pid -> [duration on device 0, ...]``.
        num_devices: Device count.
        transfer_us: ``num_devices x num_devices`` transfer matrix.
        name: Human-readable label.
    """

    partition_ids: list[str]
    dependencies: dict[str, list[str]]
    durations_us_by_device: dict[str, list[float]]
    num_devices: int
    transfer_us: list[list[float]]
    name: str


def _device_multipliers(num_devices: int, rng: np.random.Generator) -> list[float]:
    """Per-device speed multipliers spread around 1.0 for heterogeneity."""
    base = rng.uniform(0.6, 1.6, size=num_devices)
    base[0] = 1.0
    return base.tolist()


def _per_op_costs(
    n_ops: int, num_devices: int, rng: np.random.Generator, base_range: tuple[float, float] = (50.0, 150.0)
) -> list[list[float]]:
    base = rng.uniform(base_range[0], base_range[1], size=n_ops)
    mults = _device_multipliers(num_devices, rng)
    op_dispersion = rng.uniform(0.85, 1.15, size=(n_ops, num_devices))
    return [[float(base[i] * mults[d] * op_dispersion[i, d]) for d in range(num_devices)] for i in range(n_ops)]


def _default_transfer(num_devices: int, scale: float = 5.0) -> list[list[float]]:
    m = [[0.0] * num_devices for _ in range(num_devices)]
    for i in range(num_devices):
        for j in range(num_devices):
            if i != j:
                m[i][j] = scale
    return m


def chain(n_ops: int, num_devices: int = 2, seed: int = 0) -> SyntheticDag:
    """Straight-line chain of ``n_ops`` partitions."""
    rng = np.random.default_rng(seed)
    pids = [f"chain_{i}" for i in range(n_ops)]
    deps = {pids[i]: ([pids[i - 1]] if i > 0 else []) for i in range(n_ops)}
    costs = _per_op_costs(n_ops, num_devices, rng)
    durations = {pids[i]: costs[i] for i in range(n_ops)}
    return SyntheticDag(pids, deps, durations, num_devices, _default_transfer(num_devices), f"chain_n{n_ops}")


def fan_out(n_branches: int, num_devices: int = 2, seed: int = 0) -> SyntheticDag:
    """Source -> N parallel branches -> sink."""
    rng = np.random.default_rng(seed)
    n_ops = n_branches + 2
    pids = ["src"] + [f"branch_{i}" for i in range(n_branches)] + ["sink"]
    deps: dict[str, list[str]] = {"src": []}
    for b in range(n_branches):
        deps[f"branch_{b}"] = ["src"]
    deps["sink"] = [f"branch_{b}" for b in range(n_branches)]
    costs = _per_op_costs(n_ops, num_devices, rng)
    durations = {pids[i]: costs[i] for i in range(n_ops)}
    return SyntheticDag(pids, deps, durations, num_devices, _default_transfer(num_devices), f"fan_out_b{n_branches}")


def diamond(width: int, num_devices: int = 2, seed: int = 0) -> SyntheticDag:
    """Diamond: src -> [width parallel] -> mid -> [width parallel] -> sink."""
    rng = np.random.default_rng(seed)
    pids = ["src"]
    pids += [f"a_{i}" for i in range(width)]
    pids += ["mid"]
    pids += [f"b_{i}" for i in range(width)]
    pids += ["sink"]
    n_ops = len(pids)
    deps: dict[str, list[str]] = {"src": []}
    for i in range(width):
        deps[f"a_{i}"] = ["src"]
    deps["mid"] = [f"a_{i}" for i in range(width)]
    for i in range(width):
        deps[f"b_{i}"] = ["mid"]
    deps["sink"] = [f"b_{i}" for i in range(width)]
    costs = _per_op_costs(n_ops, num_devices, rng)
    durations = {pids[i]: costs[i] for i in range(n_ops)}
    return SyntheticDag(pids, deps, durations, num_devices, _default_transfer(num_devices), f"diamond_w{width}")


def random_dag(n_ops: int, edge_density: float = 0.2, num_devices: int = 4, seed: int = 0) -> SyntheticDag:
    """Random DAG built by accepting forward edges from earlier topo positions.

    Each node ``i`` (i > 0) gets at least one predecessor (``i - 1``)
    so the graph stays connected, plus extra back-edges drawn at the
    specified ``edge_density``.
    """
    rng = np.random.default_rng(seed)
    pids = [f"op_{i}" for i in range(n_ops)]
    deps: dict[str, list[str]] = {pids[0]: []}
    for i in range(1, n_ops):
        chosen: list[str] = [pids[i - 1]]
        for j in range(i - 1):
            if rng.random() < edge_density:
                chosen.append(pids[j])
        deps[pids[i]] = chosen
    costs = _per_op_costs(n_ops, num_devices, rng)
    durations = {pids[i]: costs[i] for i in range(n_ops)}
    return SyntheticDag(
        pids, deps, durations, num_devices, _default_transfer(num_devices), f"random_n{n_ops}_d{edge_density}"
    )


def transformer_block(layers: int = 12, num_devices: int = 2, seed: int = 0) -> SyntheticDag:
    """Stack of simplified transformer layers.

    Each layer emits the canonical chain ``qkv -> attn -> out_proj ->
    ffn1 -> act -> ffn2`` plus a residual skip from the layer input to
    the post-attention sum and another from the post-attention sum to
    the layer output. Layer L's output feeds layer L+1's qkv.
    """
    rng = np.random.default_rng(seed)
    pids: list[str] = ["embed"]
    deps: dict[str, list[str]] = {"embed": []}
    prev_layer_out = "embed"
    for layer in range(layers):
        qkv = f"L{layer}_qkv"
        attn = f"L{layer}_attn"
        out_proj = f"L{layer}_out_proj"
        attn_resid = f"L{layer}_attn_resid"
        ffn1 = f"L{layer}_ffn1"
        act = f"L{layer}_act"
        ffn2 = f"L{layer}_ffn2"
        layer_out = f"L{layer}_out"
        pids.extend([qkv, attn, out_proj, attn_resid, ffn1, act, ffn2, layer_out])
        deps[qkv] = [prev_layer_out]
        deps[attn] = [qkv]
        deps[out_proj] = [attn]
        deps[attn_resid] = [out_proj, prev_layer_out]
        deps[ffn1] = [attn_resid]
        deps[act] = [ffn1]
        deps[ffn2] = [act]
        deps[layer_out] = [ffn2, attn_resid]
        prev_layer_out = layer_out
    n_ops = len(pids)
    costs = _per_op_costs(n_ops, num_devices, rng, base_range=(80.0, 220.0))
    durations = {pids[i]: costs[i] for i in range(n_ops)}
    return SyntheticDag(
        pids, deps, durations, num_devices, _default_transfer(num_devices, scale=10.0), f"transformer_L{layers}"
    )


_GRANULARITIES = ("per_op", "per_layer", "per_block", "per_model")


def _archetype_groups(archetype: str, dag: SyntheticDag, granularity: str) -> list[list[str]]:
    """Return an ordered list of op-id groups for a single model under ``granularity``.

    Each group is a contiguous set of partition IDs that will collapse
    into a single coarse partition. Group order is the topo order of
    the underlying ops, which keeps downstream dependency derivation
    cycle-free by construction.
    """
    pids = dag.partition_ids
    if granularity == "per_op":
        return [[p] for p in pids]
    if granularity == "per_model":
        return [list(pids)]

    if archetype == "chain":
        # per_layer collapses every 5 chain ops; per_block every 20.
        step = 5 if granularity == "per_layer" else 20
        return [pids[i : i + step] for i in range(0, len(pids), step)]

    if archetype == "transformer":
        layers: dict[int, list[str]] = {}
        embed: list[str] = []
        for p in pids:
            if p == "embed":
                embed.append(p)
                continue
            # Layer prefix is "L<idx>_..."; pull out the integer.
            layer_idx = int(p.split("_", 1)[0][1:])
            layers.setdefault(layer_idx, []).append(p)
        groups: list[list[str]] = []
        if embed:
            groups.append(embed)
        ordered_layers = sorted(layers.keys())
        if granularity == "per_layer":
            for li in ordered_layers:
                groups.append(layers[li])
        else:  # per_block: 4 layers per block.
            block_size = 4
            for start in range(0, len(ordered_layers), block_size):
                block_ids: list[str] = []
                for li in ordered_layers[start : start + block_size]:
                    block_ids.extend(layers[li])
                groups.append(block_ids)
        return groups

    if archetype == "fan_out":
        # Brief mandates 3 partitions per "layer": src, branches, sink.
        src = [p for p in pids if p == "src"]
        branches = [p for p in pids if p.startswith("branch_")]
        sink = [p for p in pids if p == "sink"]
        if granularity == "per_layer":
            return [src, branches, sink]
        # per_block collapses to a single group (3 < block_size).
        return [src + branches + sink]

    raise ValueError(f"unknown archetype for grouping: {archetype}")


def _build_model_dag(archetype: str, size: int, num_devices: int, seed: int) -> SyntheticDag:
    if archetype == "chain":
        return chain(size, num_devices=num_devices, seed=seed)
    if archetype == "transformer":
        return transformer_block(layers=size, num_devices=num_devices, seed=seed)
    if archetype == "fan_out":
        return fan_out(size, num_devices=num_devices, seed=seed)
    raise ValueError(f"unsupported archetype: {archetype!r}")


def _assert_dag(pids: list[str], deps: dict[str, list[str]]) -> None:
    color: dict[str, int] = {p: 0 for p in pids}

    def visit(p: str) -> None:
        if color[p] == 1:
            raise AssertionError(f"cycle detected at {p}")
        if color[p] == 2:
            return
        color[p] = 1
        for d in deps.get(p, []):
            visit(d)
        color[p] = 2

    for p in pids:
        visit(p)


def multi_model(
    model_specs: list[tuple[str, int]],
    granularity: str,
    num_devices: int = 4,
    seed: int = 0,
) -> SyntheticDag:
    """Build a multi-model DAG by composing independent per-model sub-DAGs.

    Args:
        model_specs: ``[(archetype, size), ...]`` where ``archetype`` is
            one of ``"chain"``, ``"transformer"``, ``"fan_out"`` and
            ``size`` is interpreted as ops/layers/branches respectively.
        granularity: One of ``per_op``, ``per_layer``, ``per_block``,
            ``per_model`` — how to coarsen each model's ops before they
            become scheduler partitions.
        num_devices: Device count shared by all models.
        seed: Base RNG seed. Per-model seeds derive deterministically.

    Returns:
        A :class:`SyntheticDag` with prefixed partition IDs (``m0_...``,
        ``m1_...``) and no cross-model edges.
    """
    if granularity not in _GRANULARITIES:
        raise ValueError(f"granularity must be one of {_GRANULARITIES}; got {granularity!r}")
    if not model_specs:
        raise ValueError("model_specs must be non-empty")

    rng = np.random.default_rng(seed)
    # Pre-draw stable per-device multipliers so reruns are bit-stable.
    global_mults = _device_multipliers(num_devices, rng)

    combined_pids: list[str] = []
    combined_deps: dict[str, list[str]] = {}
    combined_durations: dict[str, list[float]] = {}

    for m_idx, (archetype, size) in enumerate(model_specs):
        sub = _build_model_dag(archetype, size, num_devices=num_devices, seed=seed + m_idx * 101 + 1)
        prefix = f"m{m_idx}_"
        # Prefix every id in the sub-DAG so it never collides with peers.
        sub_pids = [prefix + p for p in sub.partition_ids]
        sub_deps = {prefix + p: [prefix + d for d in sub.dependencies.get(p, [])] for p in sub.partition_ids}
        sub_durations = {prefix + p: list(sub.durations_us_by_device[p]) for p in sub.partition_ids}

        # Build groups in the unprefixed namespace then re-prefix.
        raw_groups = _archetype_groups(archetype, sub, granularity)
        groups = [[prefix + p for p in g] for g in raw_groups if g]

        # Map op-id -> group-id for dependency lifting.
        op_to_group: dict[str, str] = {}
        group_pids: list[str] = []
        for g_idx, group in enumerate(groups):
            gid = f"{prefix}g{g_idx}" if granularity != "per_op" else group[0]
            group_pids.append(gid)
            for op in group:
                op_to_group[op] = gid

        if granularity == "per_op":
            # Identity path: keep original sub-DAG verbatim (with prefix).
            for gid in group_pids:
                combined_pids.append(gid)
                combined_deps[gid] = list(sub_deps.get(gid, []))
                combined_durations[gid] = list(sub_durations[gid])
        else:
            # Aggregate durations per device by summing ops within each group.
            for g_idx, group in enumerate(groups):
                gid = group_pids[g_idx]
                summed = [0.0] * num_devices
                for op in group:
                    op_costs = sub_durations[op]
                    for d in range(num_devices):
                        summed[d] += op_costs[d]
                # Re-apply the global multiplier band so coarsened durations stay heterogeneous.
                combined_durations[gid] = [summed[d] * (global_mults[d] / global_mults[d]) for d in range(num_devices)]
                combined_pids.append(gid)

            # Lift dependencies: group A depends on group B if any op in A
            # had a predecessor in B (and A != B).
            for g_idx, group in enumerate(groups):
                gid = group_pids[g_idx]
                preds: set[str] = set()
                for op in group:
                    for pred_op in sub_deps.get(op, []):
                        pred_gid = op_to_group.get(pred_op)
                        if pred_gid is not None and pred_gid != gid:
                            preds.add(pred_gid)
                combined_deps[gid] = sorted(preds)

    _assert_dag(combined_pids, combined_deps)
    name = f"multi_{granularity}_{'_'.join(f'{a}{s}' for a, s in model_specs)}"
    return SyntheticDag(
        combined_pids,
        combined_deps,
        combined_durations,
        num_devices,
        _default_transfer(num_devices),
        name,
    )


__all__ = [
    "SyntheticDag",
    "chain",
    "fan_out",
    "diamond",
    "random_dag",
    "transformer_block",
    "multi_model",
]
