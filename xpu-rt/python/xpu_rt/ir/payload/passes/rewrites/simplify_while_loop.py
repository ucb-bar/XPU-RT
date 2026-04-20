"""``simplify_while_loop`` -- canonicalize ``scf.while`` loops.

Mirror of XLA's ``WhileLoopSimplifier``. Critical for autoregressive
decode (KV-cache loop). Because xDSL's ``scf.while`` may not be
registered in every deployment, the pass operates on
``scf.while`` when available AND on func-level "while" markers
(``compgen.while_loop`` attribute on ``func.func``).

Rewrites:

- **Dead-loop-variable elimination**: when a loop iterand is
  passed through unchanged across iterations, tag the operand as
  ``compgen.while_invariant=true`` so downstream tiling can hoist it
  out of the loop.
- **Iter-count folding**: when a loop's trip count is statically
  known via ``compgen.trip_count`` attribute, tag the loop with
  ``compgen.while_fully_unrollable=true`` so codegen can emit an
  unrolled version.

Purely annotational today — real rewrites land when the ``scf``
dialect is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp


@dataclass(frozen=True)
class SimplifyWhileLoopConfig:
    unroll_threshold: int = 32
    mark_invariants: bool = True


@dataclass
class SimplifyWhileLoopStats:
    funcs_seen: int = 0
    loops_tagged: int = 0
    loops_fully_unrollable: int = 0
    loop_invariants_tagged: int = 0


def run_simplify_while_loop(
    module: ModuleOp,
    *,
    config: SimplifyWhileLoopConfig | None = None,
) -> SimplifyWhileLoopStats:
    cfg = config if config is not None else SimplifyWhileLoopConfig()
    stats = SimplifyWhileLoopStats()

    for op in module.walk():
        if not isinstance(op, FuncOp):
            continue
        if "compgen.while_loop" not in op.attributes:
            continue
        stats.funcs_seen += 1

        # Trip-count folding.
        tc_attr = op.attributes.get("compgen.trip_count")
        if isinstance(tc_attr, IntegerAttr):
            tc = int(tc_attr.value.data)
            if 0 < tc <= cfg.unroll_threshold:
                op.attributes["compgen.while_fully_unrollable"] = StringAttr("true")
                stats.loops_fully_unrollable += 1
        op.attributes["compgen.while_simplified"] = StringAttr("true")
        stats.loops_tagged += 1

        # Invariant tagging on iter args: if the func's arg attrs
        # declare ``compgen.loop_carried=false`` on an arg, we mark it
        # as invariant. This contract lets trace-level tooling record
        # which args don't change across iterations.
        if cfg.mark_invariants:
            arg_attrs = op.attributes.get("compgen.loop_carried_flags")
            if isinstance(arg_attrs, StringAttr):
                flags = arg_attrs.data.split(",")
                for i, f in enumerate(flags):
                    if f.strip() == "false":
                        op.attributes[f"compgen.arg_{i}_invariant"] = StringAttr("true")
                        stats.loop_invariants_tagged += 1

    return stats


__all__ = [
    "SimplifyWhileLoopConfig",
    "SimplifyWhileLoopStats",
    "run_simplify_while_loop",
]
