#!/usr/bin/env python3
"""The five compiler↔scheduler verbs, each end to end, on synthetic graphs.

    .venv/bin/python examples/verbs/all_five_verbs.py

Every verb has the same three-part chain, and all three parts have to exist
before the verb means anything:

    producer   compile_advice.<verb>_advice()      says WHAT and WHY
    bridge     scripts/advice_to_<verb>_hint.py    says it in a form the
                                                   compiler accepts, and
                                                   refuses what it would reject
    consumer   ModelBlaster/pipeline/apply_<verb>_hint.py    does it

A verb with a producer and no consumer is advice written to a file nobody can
act on. That was `shard` for most of this project's life.

THE BRIDGE IS NOT A FORMAT CONVERTER, which is the thing worth taking away
from this example. Each bridge re-checks every constraint the rewriter
enforces, *in the bridge*, where the advice that caused a refusal is still in
hand. Emitting a hint the rewriter then refuses gives you an error message
about a graph, three steps removed from the measurement that asked for it.

Run with no arguments; everything here is synthetic and needs no board.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import MB, REPO, head, note, step        # noqa: E402

import compile_advice                                  # noqa: E402


def _load_cli(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"_cli_{name}", REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_applier(name):
    import importlib.util
    path = MB / "pipeline" / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_mb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(cli_name, argv):
    mod = _load_cli(cli_name)
    old = sys.argv
    try:
        sys.argv = [cli_name] + argv
        return mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------- shard ----
def demo_shard(d: Path) -> None:
    step("shard", "one dispatch, several cores. THE GRAPH DOES NOT CHANGE.")
    note("""
The only verb that is not a graph rewrite. split cuts one dispatch into n and
renumbers everything after it; shard leaves the dispatch count, the ids and
the edges alone and says this ONE dispatch is compiled to run its output
channels across n cores. Same node, one cost, and that cost is a function of
the width it was given.

Which is why the contract spells the count `n_shards` and not `n_splits`: a
hint handed to the wrong applier fails instead of quietly doing the other verb.""")

    ir = {"name": "m", "version": 1, "quant": "int8", "tensors": {},
          "ops": [{"name": "l0", "op": "linear_s8", "inputs": ["x"],
                   "outputs": ["y"], "shape": {"M": 8, "K": 256, "N": 1024},
                   "dispatch_id": 0, "hardware_target": "any",
                   "depends_on": []}]}
    (d / "shard_ir.json").write_text(json.dumps(ir))

    adv = [compile_advice.Advice(
        model="m", dispatch_id=0, recommendation="shard", priority=1,
        confidence="medium",
        constraints={"n_cores": 8, "legal_resources": ["k1_cluster0"]},
        evidence=compile_advice.Evidence(
            extra={"cost_1core_ms": 11.1, "cost_8core_ms": 2.26,
                   "measured_speedup": 4.91, "parallel_efficiency": 0.61,
                   "sync_overhead_us": 6982.0}))]
    compile_advice.write_advice(str(d / "shard_advice.json"), adv,
                                schedule_id="example")

    rc = _run("advice_to_shard_hint.py",
              ["--advice", str(d / "shard_advice.json"),
               "--ir", str(d / "shard_ir.json"), "--model", "m",
               "--out", str(d / "shard_hint.json")])
    if rc != 0:
        print("    bridge refused (see above)")
        return

    app = _load_applier("apply_shard_hint")
    if app is None:
        print("    (ModelBlaster not checked out; stopping at the hint)")
        return
    ops = json.loads((d / "shard_hint.json").read_text())["networks"][0]["shard_ops"]
    out = app.apply_shard_hint(ir, ops)
    print(f"    dispatches before {len(ir['ops'])}, after {len(out['ops'])}"
          f"  <- unchanged, as it must be")
    print(f"    shard_factor now: "
          f"{[o.get('shard_factor') for o in out['ops']]}")

    note("""
Now the refusal that matters. Ask for 8 shards on an OC that 8 does not
divide:""")
    try:
        bad = json.loads(json.dumps(ir))
        bad["ops"][0]["shape"]["N"] = 100
        app.apply_shard_hint(bad, [{"op": 0, "n_shards": 8}])
        print("    NOT REFUSED -- this is a bug")
    except ValueError as e:
        print(f"    refused: {e}")
    note("""
That refusal is the whole point. shard_conv_weights SKIPS a conv whose OC does
not divide -- the build succeeds, the binary runs, the answer is correct, and
it is simply not sharded. Nothing anywhere says so. The bridge walks the width
DOWN to a divisor for the same reason.""")


# ---------------------------------------------------------------- split ----
def demo_split(d: Path) -> None:
    step("split", "one dispatch becomes n. The graph GROWS.")
    note("""
The dual of shard, and the reason they are different verbs: after a split the
scheduler sees n independent pieces it may place on different harts or at
different times. The cost model gets n new rows, and every dispatch id after
the split point moves -- which is why the applier emits an `id_remap` and
shard does not.

The split factor is DERIVED, not chosen: `n = ceil(service_time / slot)`,
rounded up to a divisor of the tilable dimension. ModelBlaster's own
decision_loop.py hardcodes n=2, which is right by coincidence.

And the 2.27x ceiling in advice_to_split_hint.py is SPLIT's, measured on
4-way OC splitting of yolov8n which costs +76% total work before it buys any
parallelism. It does not transfer to shard.""")
    print("    (see xpu-rt/tests/test_k1_closed_loop.py::SplitReachesTheRewriter"
          " for the executable version)")


# --------------------------------------------------------------- unfuse ----
def demo_unfuse(d: Path) -> None:
    step("unfuse", "undo a fusion whose fused kernel is a reference fallback")
    note("""
`unfuse_advice`'s gate is `impl.split("/")[0] != "reference"` -- it fires when
a FUSED op ran the scalar reference, because then the fusion bought nothing
and cost the scheduler a placement choice.

Worth knowing: this advice CANNOT fire on any profile in the tree today. Every
fused op runs a curated kernel now, so the historical condition (57 of 90
dispatches on reference, 0.81x) no longer occurs. Reproducing it means
FORCING it with `--keep-reference-ops conv2d_batchnorm2d_silu_s8`, and a run
that does so should say that the loop's response was verified, not that the
loop found the condition.""")


# ----------------------------------------------------------------- fuse ----
def demo_fuse(d: Path) -> None:
    step("fuse", "collapse a run of tiny adjacent dispatches")
    note("""
`overhead_advice` fires when per-dispatch overhead dominates: a chain of
sub-millisecond ops connected by dependency edges, where the runtime spends
more on dispatching than on arithmetic. Many-to-one, so the applier's
`id_remap` is scalar-valued where split's is list-valued.""")


# ------------------------------------------------ choose_implementation ----
def demo_choose_impl(d: Path) -> None:
    step("choose_implementation", "the curated kernel is not always faster")
    note("""
`implementation_advice` compares measured impls per op. The recorded case is
maxpool2d_s8, where the curated RVV kernel was 21.5% SLOWER than the scalar
reference -- vectorising a pure-comparison op buys nothing and costs the
setup.

A build is ONE target, so "use impl X for dispatch 7" has no representation in
ModelBlaster's codegen; what does is the per-OP-KIND swap, which is
`--keep-reference-ops`. Per-DISPATCH choice is the schedule's `impl` field,
which the walker now selects on -- see examples/k1_board/.""")


def main() -> int:
    head("The five verbs")
    note("""
producer -> bridge -> consumer, for each. A verb missing any of the three is
not a verb the loop can use.""")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        demo_shard(d)
        demo_split(d)
        demo_unfuse(d)
        demo_fuse(d)
        demo_choose_impl(d)
    print()
    note("""
Executable versions of all of these, against the REAL rewriters, are in
xpu-rt/tests/test_k1_closed_loop.py -- SplitReachesTheRewriter,
UnfuseReachesTheRewriter, ShardReachesTheRewriter. They are tests rather than
examples because each one needs a graph shaped to trigger exactly one
condition, which reads better as a fixture than as a narrative.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
