"""The closed loop, end to end, with no board: profile -> schedule -> advice ->
hint -> rewritten graph -> re-schedule -> accept/reject.

WHY AN INTEGRATION TEST AT ALL. Each stage of this loop has (or now has) unit
tests, and every stage passed them while the loop as a whole did not run,
because the failures live in the *seams*:

* `emit_compile_advice.py` needs `<gen_root>/profile/...` while
  `profile_loader` needs `<repo>/<gen_root>/profile/...` -- the same flag name
  meaning two different things, so a value correct for one silently resolves
  nothing for the other, and "no profile" prints a WARN and continues.
* `advice_to_fusion_hint.py` keys advice by `model`, which must be the base
  network id (`mlp_control`), while the schedule's `job_name` is a periodic
  instance (`mlp_control3`). A mismatch there produces "no fusion advice for
  <model>; nothing to do" and exit 1 -- a clean failure that reads like a
  finding.
* `overhead_advice` addresses a whole model with `dispatch_id="*"`, and the
  fuse groups come from the IR, not from the advice. So the advice and the IR
  have to agree about which network they describe without ever exchanging a
  dispatch id.
* The rewrite renumbers every dispatch, so anything comparing before against
  after has to key on op identity rather than on the id (`dispatch_lineage`).
  On this particular workload the names alone are not enough either -- fusing
  three of `mlp_control`'s four `linear_s8` dispatches makes the survivor
  unidentifiable by signature -- so the loop has to carry the rewriter's
  `id_remap` and check it. Getting that wrong turns a fusion that changed
  nothing about the tail op into a reported 2x regression on it.

None of that is visible from inside one stage. What follows drives the real
production entry points -- `load_profiled_processing_times`,
`create_workload_from_network_hierarchy`, `greedy_periodic_schedule`,
`output_scheduled_json`, then the two CLIs as subprocess-free `main()` calls --
over a synthetic K1 profile in a temp directory. No board, no network, no
writes outside the tmpdir.

THE WORKLOAD is the documented worked example: `mlp_control` at 100 Hz as a
seven-dispatch linear chain whose dispatches all cost ~60 us regardless of the
work they do (so it is launch-overhead bound and the honest advice is "emit
fewer, larger dispatches"), co-running with a six-dispatch `dronet` at 30 Hz.
The expected outcome is `fuse_with_successor` on `mlp_control` and 7 ops
becoming 4 -- which is what the loop actually produced on the board.

The numbers are SYNTHETIC and deliberately so: they are chosen to put the
workload in the regime the advisor is being tested on. Real measured numbers
live in `fixtures/k1_profile/` and are used by `test_k1_profile_fixture.py` and
`test_compile_advice_schema.py`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
for _p in (_REPO, _XPURT, os.path.join(_REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candidate_objective  # noqa: E402
import dispatch_lineage  # noqa: E402
import schedule_trace  # noqa: E402
import trace_metrics  # noqa: E402
from greedy_scheduler import greedy_periodic_schedule  # noqa: E402
from postprocessing import output_scheduled_json  # noqa: E402
from profile_loader import load_profiled_processing_times  # noqa: E402
from workload_factory import create_workload_from_network_hierarchy  # noqa: E402

# The schema validator, reused rather than duplicated: the advice this loop
# emits has to satisfy the same contract as the advice produced from a real
# profile, and two copies of a contract are one contract too many.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from test_compile_advice_schema import _validate as validate_advice  # noqa: E402

_SCRIPTS = os.path.join(_REPO, "scripts")
_MB_REWRITER = os.path.join(_REPO, "ModelBlaster", "pipeline",
                            "apply_fusion_hint.py")


def _load_script(name):
    """Import a `scripts/*.py` CLI by path. They are not a package."""
    path = os.path.join(_SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


#: `pipeline/profile_writer.py`'s column set, `implementation` included.
CSV_COLUMNS = ["dispatch_id", "module_name", "vmfb_path", "mlir_path",
               "mean_time", "mean_unit", "mean_time_ns", "returncode",
               "log_path", "source", "op", "shape", "cycles", "implementation"]

IMPL = "rvv_x60"
TARGET = "spacemit_x60"

#: Seven dispatches within 6% of each other regardless of the work each does:
#: the signature of a launch-overhead-bound model. `overhead_advice`'s gate is
#: min/max >= 0.75; this is 0.94, so the finding is not marginal.
MLP_OPS = [("linear_s8", 0.0620), ("elu_s8", 0.0605), ("linear_s8", 0.0640),
           ("elu_s8", 0.0610), ("linear_s8", 0.0635), ("elu_s8", 0.0600),
           ("linear_s8", 0.0625)]

#: Costs spanning three orders of magnitude, so DroNet is NOT overhead-bound
#: and must not attract the same advice. Values are the real fixture's first
#: five dispatches plus its output head.
DRONET_OPS = [("conv2d_s8", 2.165125), ("maxpool2d_s8", 0.801750),
              ("batchnorm2d_s8", 0.060042), ("conv2d_s8", 1.216792),
              ("conv2d_s8", 0.777542), ("linear_s8", 0.002792)]

MACHINES = ["CPU_P#0", "CPU_E#0"]
COMBOS = [["CPU_P#0"], ["CPU_E#0"]]
COMBO_HW = [IMPL, IMPL]


def _module_name(model, did, op):
    return f"{model}$dispatch_{did}_{IMPL}_{op}_n256"


class Bench:
    """A self-contained K1-shaped tree in a temp directory."""

    def __init__(self, root):
        self.root = root
        self.gen = os.path.join(root, "gen")
        self.models = {"mlp_control": MLP_OPS, "dronet": DRONET_OPS}
        self.graphs = {}
        for model, ops in self.models.items():
            self._profile(model, ops)
            self.graphs[model] = self._dispatch_graph(model, len(ops))
        self.ir = self._modelblaster_ir("mlp_control", MLP_OPS)

    # ---------------------------------------------------------------- inputs

    def _profile(self, model, ops, subdir="topo_0"):
        """`results.csv` + `profile.jsonl` at the depth each reader expects.

        Both files, because the two consumers disagree: `profile_loader` (the
        scheduler's costs) reads `results.csv`, `compile_advice.load_profiles`
        (the advice's evidence) reads `profile.jsonl`. A test that wrote only
        one would exercise only half the loop and the other half would print a
        WARN and produce an empty document.
        """
        d = os.path.join(self.gen, "profile", IMPL, TARGET, model,
                         f"{model}.int8", subdir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for i, (op, ms) in enumerate(ops):
                w.writerow({
                    "dispatch_id": i, "module_name": _module_name(model, i, op),
                    "mean_time": f"{ms:.6f}", "mean_unit": "ms",
                    "mean_time_ns": f"{ms * 1e6:.3f}", "returncode": 0,
                    "source": "k1_synthetic", "op": op, "shape": "n=256",
                    "cycles": int(ms * 24_000),
                    "implementation": f"curated[rvv]/{op}",
                })
        with open(os.path.join(d, "profile.jsonl"), "w") as f:
            for i, (op, ms) in enumerate(ops):
                f.write(json.dumps({
                    "model": model, "basename": f"{model}.int8",
                    "dispatch_id": i, "module_name": _module_name(model, i, op),
                    "hw_label": IMPL, "target": TARGET, "n_cores": 1,
                    "median_ms": ms, "op": op,
                    "implementation": f"curated[rvv]/{op}"}) + "\n")

    def _dispatch_graph(self, model, n, ops=None):
        """The artifact XPU-RT schedules: `<model>.int8_dispatch_graph.json`."""
        d = os.path.join(self.gen, "vmfb", model, TARGET, IMPL, f"{model}.int8")
        os.makedirs(d, exist_ok=True)
        dispatches = {
            f"dispatch_{i}": {"id": i, "ordinal": 1, "total": 1,
                              "dependencies": [] if i == 0
                              else [f"dispatch_{i - 1}"]}
            for i in range(n)}
        path = os.path.join(d, f"{model}.int8_dispatch_graph.json")
        with open(path, "w") as f:
            json.dump({"dispatches": dispatches}, f)
        return os.path.relpath(path, self.root)

    def _modelblaster_ir(self, model, ops):
        """The IR `advice_to_fusion_hint` derives legal fuse groups from."""
        graph = {"name": model, "version": 1, "quant": "int8", "input": "x",
                 "output": f"t{len(ops) - 1}", "tensors": {}, "ops": []}
        for i, (op, _ms) in enumerate(ops):
            graph["ops"].append({
                "name": f"{model}.{i}", "op": op,
                "inputs": ["x" if i == 0 else f"t{i - 1}"],
                "outputs": [f"t{i}"], "shape": {"n": 256},
                "dispatch_id": i, "hardware_target": "any",
                "depends_on": [] if i == 0 else [i - 1]})
        path = os.path.join(self.root, "graph.json")
        with open(path, "w") as f:
            json.dump(graph, f)
        return path

    def networks(self, graphs=None):
        graphs = graphs or self.graphs
        return {
            "mlp_control": {"id": 0, "identifier": "mlp_control",
                            "dispatch_deps_path": graphs["mlp_control"],
                            "period": 10.0, "window_duration": 10.0,
                            "num_instances": 5},
            "dronet": {"id": 1, "identifier": "dronet",
                       "dispatch_deps_path": graphs["dronet"],
                       "period": 33.3, "window_duration": 33.3,
                       "num_instances": 2},
        }

    # ---------------------------------------------------------------- stages

    def schedule(self, out_name="schedule.json", graphs=None):
        """Stage 1-2: profiled costs -> workload -> solved schedule JSON."""
        networks = self.networks(graphs)
        proc, _p, _e, by_net = load_profiled_processing_times(
            networks, self.root, COMBOS, COMBO_HW, TARGET, IMPL, IMPL,
            np.random.default_rng(0), 1.0, gen_root="gen")
        workload = create_workload_from_network_hierarchy(
            {"networks": networks, "edges": []}, self.root, MACHINES,
            np.zeros((2, 2)), processing_times=proc, p_core_speedup=1.0,
            random_seed=0, machine_combinations=COMBOS)
        t, alpha = greedy_periodic_schedule(workload)
        path = os.path.join(self.root, out_name)
        output_scheduled_json(workload, t, alpha, path,
                              profiled_times_by_network=by_net)
        with open(path) as f:
            return path, json.load(f)

    def score(self, label, sched):
        """Stage 3: score a predicted schedule with the measured-run metrics."""
        rows = schedule_trace.trace_rows_from_schedule(sched)
        summary = trace_metrics.summarise_trace(
            rows, schedule_trace.periods_ms(sched))
        return candidate_objective.from_trace_summary(
            label, summary, critical_models=("mlp_control",),
            heavy_model="dronet",
            standalone_cycles=int(schedule_trace.standalone_service_us(sched)))

    def advise(self, schedule_path, out_name="compile_advice.json"):
        """Stage 4: measured profiles + schedule -> compile_advice.json."""
        cli = _load_script("emit_compile_advice.py")
        out = os.path.join(self.root, out_name)
        argv = ["--gen-root", self.gen, "--target", TARGET,
                "--schedule", schedule_path, "--out", out,
                "--models", "mlp_control:mlp_control.int8,dronet:dronet.int8",
                "--impls", IMPL, "--baseline-impl", IMPL]
        old = sys.argv
        try:
            sys.argv = ["emit_compile_advice.py"] + argv
            rc = cli.main()
        finally:
            sys.argv = old
        assert rc == 0, rc
        with open(out) as f:
            return out, json.load(f)

    def hint(self, advice_path, model="mlp_control", out_name="hint.json",
             pair_only=True):
        """Stage 5: advice -> `modelblaster.fusion_hints/v1`."""
        cli = _load_script("advice_to_fusion_hint.py")
        out = os.path.join(self.root, out_name)
        argv = ["--advice", advice_path, "--ir", self.ir, "--model", model,
                "--out", out] + (["--pair-only"] if pair_only else [])
        old = sys.argv
        try:
            sys.argv = ["advice_to_fusion_hint.py"] + argv
            rc = cli.main()
        finally:
            sys.argv = old
        return rc, (json.load(open(out)) if rc == 0 else None)


def _rewrite_dispatch_graph(dispatches, groups):
    """Apply fuse `groups` to a dispatch graph, renumbering contiguously.

    A local implementation of the contract, on the artifact XPU-RT owns (the
    dispatch graph), for the same reason `test_dispatch_lineage._fuse` is local:
    the renumbering is what XPU-RT has to survive, and it must be exercised
    whether or not ModelBlaster is checked out beside this repo.
    `AgainstTheRealRewriter` pins it to the real rewriter.

    Per the rewrite invariant ("a rewrite may not reduce modelled work unless a
    kernel exists that performs the merged work"), the fused dispatch keeps the
    SUM of its members' costs. Anything else makes the schedule count work that
    the hardware still does.
    """
    by_id = {d["id"]: (name, d) for name, d in dispatches.items()}
    member_of = {}
    for gi, g in enumerate(groups):
        for did in g:
            member_of[did] = gi
    keep = [did for did in sorted(by_id)
            if did not in member_of or min(groups[member_of[did]]) == did]

    new_id = {}
    for i, did in enumerate(keep):
        new_id[did] = i
    for did, gi in member_of.items():
        new_id[did] = new_id[min(groups[gi])]

    out = {}
    for did in keep:
        _name, d = by_id[did]
        nid = new_id[did]
        gi = member_of.get(did)
        members = groups[gi] if gi is not None else [did]
        deps = []
        for dep_name in d.get("dependencies", []):
            dep_id = dispatches[dep_name]["id"]
            if new_id[dep_id] != nid and new_id[dep_id] not in [
                    new_id[m] for m in members]:
                deps.append(f"dispatch_{new_id[dep_id]}")
        out[f"dispatch_{nid}"] = {"id": nid, "ordinal": 1, "total": 1,
                                  "dependencies": sorted(set(deps)),
                                  "fused_from": list(members)}
    return out, new_id


class TheLoopRuns(unittest.TestCase):
    """One pass of the loop, asserted at every seam."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bench = Bench(cls._tmp.name)
        cls.sched_path, cls.sched = cls.bench.schedule()
        cls.advice_path, cls.advice = cls.bench.advise(cls.sched_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ------------------------------------------------------------ stage 1-2

    def test_the_schedule_used_the_profiled_costs_not_synthetic_ones(self):
        """`load_profiled_processing_times` used to fall back to
        `rng.uniform(2, 10)` per missing dispatch, silently.

        A 62 us MLP dispatch coming back as a 2-10 ms one changes every
        conclusion downstream while looking like a normal schedule. So the first
        thing to check is that the durations in the schedule are the numbers
        that were written to disk.
        """
        durations = {}
        for key, d in self.sched["dispatches"].items():
            if key.startswith("mlp_control0_"):
                durations[d["id"]] = d["duration"]
        self.assertEqual(len(durations), len(MLP_OPS))
        for did, (_op, ms) in enumerate(MLP_OPS):
            self.assertAlmostEqual(durations[did], ms, places=9,
                                   msg=f"dispatch {did}")

    def test_the_schedule_carries_the_module_names_the_advice_needs(self):
        """Without `module_name`, nothing downstream can identify an op.

        `output_scheduled_json` routes it through a per-network bucket because
        the two networks share dispatch_ids 0..5; a regression there gave every
        `mlp_control` dispatch a `dronet` module name.
        """
        for key, d in self.sched["dispatches"].items():
            model = key.split("_dispatch_")[0].rstrip("0123456789")
            self.assertIn("module_name", d, key)
            self.assertTrue(d["module_name"].startswith(model + "$"),
                            f"{key} -> {d['module_name']}")

    def test_the_periods_survive_into_the_metadata(self):
        """`emit_compile_advice` and `trace_metrics` both read this map.

        If it is missing, every free slot comes out 0, which disables split and
        shard advice for the whole run without saying so.
        """
        self.assertEqual(self.sched["metadata"]["periodic_networks"],
                         {"mlp_control": 10.0, "dronet": 33.3})

    def test_the_predicted_schedule_scores_through_the_measured_metrics(self):
        """One definition of "deadline miss", not a second one for predictions.

        `schedule_trace` exists so a host-side sweep is scored by
        `trace_metrics` rather than by a private reimplementation. Rendering the
        schedule and summarising it must reproduce the instance counts the
        schedule was built with.
        """
        outcome = self.bench.score("baseline", self.sched)
        self.assertEqual(outcome.per_model["mlp_control"].instances, 5)
        self.assertEqual(outcome.per_model["dronet"].instances, 2)
        self.assertGreater(outcome.standalone_cycles, 0)

    # -------------------------------------------------------------- stage 4

    def test_the_advice_document_conforms_to_its_schema(self):
        self.assertEqual(self.advice["schedule_id"], "schedule.json")
        self.assertTrue(self.advice["advice"])
        for i, item in enumerate(self.advice["advice"]):
            validate_advice(item, self, f"loop/advice[{i}]")

    def test_the_advice_is_about_base_networks_not_periodic_instances(self):
        """The seam that breaks `advice_to_fusion_hint` if it is wrong.

        That script filters `x["model"] == args.model`, and its `--model` names
        a network (`mlp_control`). The schedule's `job_name` is an instance
        (`mlp_control3`). If advice leaked instance names, the filter would
        match nothing and the stage would exit 1 with "nothing to do" -- which
        reads like a finding rather than a plumbing error.
        """
        models = {a["model"] for a in self.advice["advice"]}
        self.assertEqual(models, {"mlp_control", "dronet"})

    def test_the_overhead_bound_model_is_told_to_coarsen(self):
        """The finding this workload was built to produce.

        All seven `mlp_control` dispatches cost within 6% of each other despite
        doing different amounts of work, so ~97% of its 434 us is launch
        overhead and the only lever is fewer, larger dispatches.
        """
        fuse = [a for a in self.advice["advice"]
                if a["recommendation"].startswith("fuse_")]
        self.assertEqual([a["model"] for a in fuse], ["mlp_control"])
        ev = fuse[0]["evidence"]
        self.assertEqual(ev["n_dispatches"], len(MLP_OPS))
        self.assertGreater(ev["estimated_overhead_fraction"], 0.9)
        self.assertEqual(fuse[0]["dispatch_id"], "*",
                         "the finding is about the model, not one dispatch")

    def test_the_model_with_real_cost_variation_is_not(self):
        """DroNet's dispatches span 2.8 us to 2.2 ms -- 775x.

        It is doing work, not paying launch cost, so recommending a granularity
        change for it would be advice with no evidence behind it. An advisor
        that flagged every model would be indistinguishable from one that
        flagged the right one.
        """
        for a in self.advice["advice"]:
            if a["model"] == "dronet":
                self.assertEqual(a["recommendation"], "unchanged", a["rationale"])

    def test_no_split_advice_when_nothing_overruns_its_slot(self):
        """`mlp_control` fits its 10 ms period 23x over.

        `blocking_advice` gated on the free slot rather than the raw period is
        what keeps this quiet; comparing against the period would still be
        quiet here, but comparing against 0 (the bug when the periodic map is
        missing) would flag every dispatch in the workload.
        """
        self.assertEqual(
            [a for a in self.advice["advice"] if a["recommendation"] == "split"],
            [])

    # -------------------------------------------------------------- stage 5

    def test_the_hint_names_the_contract_and_only_legal_chains(self):
        rc, hint = self.bench.hint(self.advice_path)
        self.assertEqual(rc, 0)
        self.assertEqual(hint["contract"], "modelblaster.fusion_hints/v1")
        net = hint["networks"][0]
        self.assertEqual(net["network"], "mlp_control")
        self.assertEqual(net["fuse_groups"], [[0, 1], [2, 3], [4, 5]],
                         "--pair-only bounds each fused unit to two ops, so the "
                         "non-preemptive blocker stays small")
        flat = [d for g in net["fuse_groups"] for d in g]
        self.assertEqual(len(flat), len(set(flat)),
                         "groups must be disjoint; the validator on the other "
                         "side rejects overlapping ones")

    def test_the_hint_carries_the_evidence_that_produced_it(self):
        """A hint whose provenance is a file name is not auditable.

        `apply_fusion_hint` will fuse whatever it is told to. The only record of
        *why* is what the hint carries, so it has to travel with the numbers.
        """
        _rc, hint = self.bench.hint(self.advice_path, out_name="hint_prov.json")
        prov = hint["_provenance"]
        self.assertEqual(prov["from_advice"], "schedule.json")
        self.assertEqual(prov["recommendations"], ["fuse_with_successor"])
        self.assertGreater(prov["evidence"][0]["estimated_overhead_fraction"],
                           0.9)

    def test_asking_for_a_hint_about_an_unadvised_model_fails_loudly(self):
        """DroNet got no fusion advice, so there is no hint to emit.

        Returning 0 with an empty hint would let the next stage fuse a model
        nothing recommended fusing.
        """
        rc, _ = self.bench.hint(self.advice_path, model="dronet",
                                out_name="hint_dronet.json")
        self.assertEqual(rc, 1)

    # ------------------------------------------------------------ stage 6-7

    def test_the_rewritten_graph_reschedules_and_the_loop_closes(self):
        """The full round trip, ending in an accept/reject decision.

        7 dispatches become 4 and the modelled work is unchanged (the fused
        dispatch keeps the sum of its members), so what can improve is the
        schedule, not the cycle count -- which is exactly the case the
        lexicographic objective exists for and a summed-cycles criterion would
        call a tie.
        """
        _rc, hint = self.bench.hint(self.advice_path, out_name="hint_rt.json")
        groups = hint["networks"][0]["fuse_groups"]

        graph_path = os.path.join(self.bench.root,
                                  self.bench.graphs["mlp_control"])
        with open(graph_path) as f:
            before = json.load(f)["dispatches"]
        after, remap = _rewrite_dispatch_graph(before, groups)

        self.assertEqual(len(after), 4, sorted(after))
        self.assertEqual(sorted(d["id"] for d in after.values()), [0, 1, 2, 3],
                         "ids must be contiguous after the rewrite")
        self.assertEqual([remap[i] for i in range(7)], [0, 0, 1, 1, 2, 2, 3])

        # Re-profile: the fused dispatch costs what its members cost. Refusing
        # to model a saving no kernel provides is rewrite invariant #1.
        fused_ops = []
        for name in sorted(after, key=lambda k: after[k]["id"]):
            members = after[name]["fused_from"]
            fused_ops.append(("_".join(MLP_OPS[m][0] for m in members),
                              sum(MLP_OPS[m][1] for m in members)))
        self.assertAlmostEqual(sum(ms for _o, ms in fused_ops),
                               sum(ms for _o, ms in MLP_OPS), places=9,
                               msg="the rewrite must not make work vanish")

        # Stage 6: re-profile and re-schedule through the same entry points.
        bench2 = Bench(tempfile.mkdtemp(dir=self.bench.root))
        bench2._profile("mlp_control", fused_ops)
        graphs = dict(bench2.graphs)
        graphs["mlp_control"] = bench2._dispatch_graph("mlp_control",
                                                       len(fused_ops))
        path2, sched2 = bench2.schedule("schedule.fused.json", graphs=graphs)
        self.assertEqual(
            sum(1 for k in sched2["dispatches"] if k.startswith("mlp_control0_")),
            4)

        # Stage 7: the accept/reject decision, on the objective that owns it.
        base = self.bench.score("baseline", self.sched)
        cand = bench2.score("fused", sched2)
        ok, why = candidate_objective.accept(cand, base)
        self.assertIsInstance(ok, bool)
        self.assertTrue(why.startswith(("accepted", "rejected")), why)
        # Whichever way it goes, it must not have been decided on cycles: the
        # summed service time is identical to within rounding.
        self.assertAlmostEqual(cand.standalone_cycles / base.standalone_cycles,
                               1.0, places=2)

    def _before_after_profiles(self, out_name):
        """The pre-rewrite profile and the post-rewrite one it will be compared
        against: 7 dispatches becoming 4, costs summed per fused group."""
        _rc, hint = self.bench.hint(self.advice_path, out_name=out_name)
        groups = hint["networks"][0]["fuse_groups"] + [[6]]
        before = {i: {"module_name": _module_name("mlp_control", i, op),
                      "median_ms": ms}
                  for i, (op, ms) in enumerate(MLP_OPS)}
        after = {}
        for gi, members in enumerate(groups):
            op = "_".join(MLP_OPS[m][0] for m in members)
            after[gi] = {"module_name": _module_name("mlp_control", gi, op),
                         "median_ms": sum(MLP_OPS[m][1] for m in members)}
        remap = {m: gi for gi, members in enumerate(groups) for m in members}
        return before, after, remap

    def test_a_dispatch_id_join_would_misreport_this_rewrite(self):
        """The concrete damage, on this loop's own artifacts.

        `mlp_control`'s tail linear moves from dispatch 6 to dispatch 3, and a
        fused PAIR lands in slot 6's place -- except slot 6 no longer exists, so
        an id join drops the tail entirely and compares slots 0-3 (single ops
        before, fused pairs after) as if they were the same work. Every one of
        those comparisons shows the fusion roughly doubling the cost of an op it
        did not touch.
        """
        before, after, _remap = self._before_after_profiles("hint_id.json")
        shared = sorted(set(before) & set(after))
        self.assertEqual(shared, [0, 1, 2, 3])
        self.assertNotIn(6, after, "the tail's old slot is gone after the "
                                   "renumbering, so an id join silently loses it")
        inflated = [d for d in shared
                    if after[d]["median_ms"] / before[d]["median_ms"] > 1.5]
        self.assertEqual(len(inflated), 3,
                         "three of the four surviving slots would report a ~2x "
                         "regression that never happened")

    def test_a_signature_join_refuses_rather_than_guessing_here(self):
        """This rewrite fuses members of a REPEATED signature family.

        `mlp_control` runs `linear_s8` four times (dispatches 0, 2, 4, 6) and
        three of them are fused away, so after the rewrite one `linear_s8`
        remains. Nothing in the names says which one it is: pairing by order of
        appearance would match the surviving tail to dispatch 0, and report the
        fusion's saving against an op that was fused, not kept.

        So the honest answer for THIS shape of rewrite is a refusal, and the
        loop has to use the `id_remap` the rewriter emits (next test). The
        signature join carries a rewrite on its own only when the fused ops have
        signatures that occur once -- see `test_dispatch_lineage`, where DroNet's
        does.
        """
        before, after, _remap = self._before_after_profiles("hint_sig.json")
        j = dispatch_lineage.join(before, after)
        self.assertFalse(j.is_unambiguous)
        sig = dispatch_lineage.op_signature(before[6]["module_name"])
        self.assertIn(sig, j.ambiguous)
        self.assertEqual(j.ambiguous[sig], ([0, 2, 4, 6], [3]))
        self.assertEqual(j.matched, {},
                         "nothing may be matched by ordinal inside a family "
                         "whose multiplicity changed")

    def test_the_remap_resolves_it_and_agrees_with_the_artifact(self):
        """`id_remap` is what makes this rewrite joinable, and it is checkable.

        The rewriter states 0,1 -> 0; 2,3 -> 1; 4,5 -> 2; 6 -> 3. The
        many-to-one entries are fusions, so no signature can be expected to
        survive them; the one-to-one entry (the tail) must, and
        `check_id_remap` says whether the remap and the module names tell the
        same story. That check is the thing that would have caught the remap
        being stale, which is the one failure mode the field itself cannot
        report.
        """
        before, after, remap = self._before_after_profiles("hint_remap.json")
        self.assertEqual([remap[i] for i in range(7)], [0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(
            dispatch_lineage.check_id_remap(before, after, remap), [])
        # And the tail's cost really did travel unchanged, which is what the
        # loop needs to conclude the fusion left it alone.
        self.assertAlmostEqual(after[remap[6]]["median_ms"],
                               before[6]["median_ms"], places=9)


class ASaturatedModelStillGetsAdvice(unittest.TestCase):
    """THE BUG THIS CLASS FOUND, and what made it invisible.

    `emit_compile_advice.py` carried a function, `dispatch_budget`, whose
    docstring names this exact case: DroNet needs 113.7 ms against a 33.3 ms
    window while its largest single dispatch is 22.9 ms, so comparing each
    dispatch to the whole period "reports 'no dispatch is too long' about a
    model that misses every deadline".

    That function was never called. `blocking_advice` was handed `budget_for`,
    which returns the free slot -- and the free slot of a saturated model is
    zero, so it falls back to the whole period. The result: a model overrunning
    its deadline by 3.4x produced ten `unchanged` items and nothing actionable,
    which reads as "the advisor looked and found nothing wrong". The fix that
    was written for it sat one scope away, dead, with the numbers in its
    docstring.

    It was invisible because nothing failed: the run succeeded, the document
    validated, and an empty result from an advisor is indistinguishable from a
    healthy workload unless you already know the workload is not healthy.

    Saturation is also the ONLY regime where this matters, which is why it
    survived: for a model that fits its period the free-slot test is right, and
    every test workload in the tree fits.
    """

    #: 113.7 ms of work against a 33.3 ms window -- the shape from the
    #: docstring, largest dispatch 22.87 ms, i.e. comfortably UNDER the period.
    SATURATED = [("conv2d_s8", 22.87), ("conv2d_s8", 18.0), ("conv2d_s8", 16.0),
                 ("conv2d_s8", 14.0), ("conv2d_s8", 12.0), ("conv2d_s8", 10.0),
                 ("conv2d_s8", 8.0), ("conv2d_s8", 6.0), ("conv2d_s8", 4.0),
                 ("linear_s8", 2.83)]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.bench = Bench(self._tmp.name)
        # Replace dronet with the saturated variant, keeping every other part
        # of the pipeline identical.
        self.bench._profile("dronet", self.SATURATED)
        graphs = dict(self.bench.graphs)
        graphs["dronet"] = self.bench._dispatch_graph("dronet",
                                                      len(self.SATURATED))
        self.sched_path, self.sched = self.bench.schedule(graphs=graphs)
        self.advice_path, self.advice = self.bench.advise(self.sched_path)

    def _for(self, model):
        return [a for a in self.advice["advice"] if a["model"] == model]

    def test_the_premise_holds_no_single_dispatch_exceeds_the_period(self):
        """Otherwise the whole-period test would have caught it by accident.

        This is what makes the next test a real regression rather than a
        restatement: the budget has to come from the proportional share,
        because the period alone finds nothing here.
        """
        period = self.sched["metadata"]["periodic_networks"]["dronet"]
        biggest = max(ms for _op, ms in self.SATURATED)
        total = sum(ms for _op, ms in self.SATURATED)
        self.assertLess(biggest, period)
        self.assertGreater(total, period * 2)

    def test_the_saturated_model_produces_actionable_split_advice(self):
        """Before the fix this list was empty for this exact workload."""
        splits = [a for a in self._for("dronet")
                  if a["recommendation"] == "split"]
        self.assertTrue(
            splits,
            "a model overrunning its window by >2x with no dispatch longer "
            "than the window must still be told to split; an empty result "
            "here is the advisor silently declining to look")
        for a in splits:
            validate_advice(a, self, f"saturated/{a['dispatch_id']}")
        # The heaviest dispatch is the one that has to shrink first, so it must
        # be among them.
        heaviest = max(range(len(self.SATURATED)),
                       key=lambda i: self.SATURATED[i][1])
        self.assertIn(heaviest, [a["dispatch_id"] for a in splits])

    def test_the_budget_is_the_proportional_share_not_the_period(self):
        """The number in the evidence has to be the one the decision used.

        A `split` whose `max_target_piece_us` is the whole period tells the
        compiler to cut to a size that still does not fit, so the next round
        measures no improvement and the loop concludes fusion/splitting does not
        help.
        """
        period_ms = self.sched["metadata"]["periodic_networks"]["dronet"]
        total = sum(ms for _op, ms in self.SATURATED)
        biggest = max(ms for _op, ms in self.SATURATED)
        expect_us = period_ms * (biggest / total) * 1000.0

        splits = [a for a in self._for("dronet")
                  if a["recommendation"] == "split"]
        self.assertTrue(splits)
        for a in splits:
            self.assertAlmostEqual(a["evidence"]["periodic_free_slot_us"],
                                   expect_us, places=1)
            self.assertAlmostEqual(a["constraints"]["max_target_piece_us"],
                                   expect_us, places=1)
            self.assertLess(expect_us, period_ms * 1000.0,
                            "the share must be strictly tighter than the "
                            "period, or nothing has changed")

    def test_the_co_running_model_that_fits_is_not_dragged_in(self):
        """`mlp_control` fits its 10 ms period 23x over.

        An earlier version of the budget took the minimum across models, so one
        saturated model zeroed the budget for every other one. Splitting a 62 us
        dispatch is not a thing anyone can do, and advice that says to is noise
        that hides the real finding.
        """
        self.assertEqual(
            [a for a in self._for("mlp_control")
             if a["recommendation"] == "split"], [])


@unittest.skipUnless(os.path.exists(_MB_REWRITER),
                     "ModelBlaster checkout absent; the IR rewriter lives at "
                     "ModelBlaster/pipeline/apply_fusion_hint.py")
class AgainstTheRealRewriter(unittest.TestCase):
    """Feed the emitted hint to the rewriter that will actually consume it.

    `_rewrite_dispatch_graph` above is XPU-RT's model of what the rewrite does.
    This checks the hint XPU-RT emits is one ModelBlaster accepts, and that the
    two agree on the resulting shape -- the seam a contract-version bump or a
    validator change would break, one repo away.
    """

    def test_the_emitted_hint_is_accepted_and_produces_the_same_shape(self):
        spec = importlib.util.spec_from_file_location("_mb_fuse", _MB_REWRITER)
        mb = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mb)

        with tempfile.TemporaryDirectory() as d:
            bench = Bench(d)
            sched_path, _ = bench.schedule()
            advice_path, _ = bench.advise(sched_path)
            rc, hint = bench.hint(advice_path)
            self.assertEqual(rc, 0)
            hint_path = os.path.join(d, "hint.json")
            out = os.path.join(d, "graph.fused.json")
            rc = mb.main(["--hint", hint_path, "--model", "mlp_control",
                          "--ir", bench.ir, "--out", out])
            self.assertEqual(rc, 0, "the rewriter rejected XPU-RT's own hint")
            with open(out) as f:
                g = json.load(f)

        self.assertEqual(len(g["ops"]), 4,
                         "the documented result: mlp_control 7 ops -> 4")
        remap = {int(k): v for k, v in g["id_remap"].items()}
        self.assertEqual([remap[i] for i in range(7)], [0, 0, 1, 1, 2, 2, 3],
                         "XPU-RT's model of the renumbering must match the "
                         "rewriter's, or every cross-rung join is off")
        # Every fused op records what it was built from, which is what makes
        # the cost bookkeeping auditable after the fact.
        fused = [o for o in g["ops"] if o.get("fused_from")]
        self.assertEqual(len(fused), 3)
        self.assertEqual(sorted(m for o in fused for m in o["fused_from"]),
                         [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
