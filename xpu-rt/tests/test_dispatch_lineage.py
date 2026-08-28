"""A fuse/split rewrite renumbers dispatches, so `dispatch_id` is not identity.

THE DEFECT THIS FILE GUARDS, and why it was plausible enough to write a file
about: comparing a profile taken before a granularity rewrite against one taken
after it is the whole point of the closed loop, and `dispatch_id` is the obvious
key. It is present on both sides, it is an int, it lines up in length, and the
join produces a full table with no warning.

It is also wrong. `realize-hint` reassigns dispatch_ids contiguously over the
rewritten graph -- stated in
`ModelBlaster/.claude/skills/realize-hint/SKILL.md` and in
`apply_fusion_hint.py`'s own docstring, which spells out that "consumers keyed
on dispatch_id must translate through this before joining a pre-rewrite profile
/ cost DB against a post-rewrite graph". Fusing DroNet's dispatches 1 and 2
shifts 3..20 down to 2..19. A `dispatch_id` join then compares a 1.6 ms
convolution against an 0.11 ms one, and the round reports a 14x speedup it never
got, from numbers that all look reasonable.

`test_a_dispatch_id_join_pairs_different_ops` establishes that the hazard is
real on this exact data, so the tests that follow are not asserting a property
nobody could have got wrong.

The profile these tests join is the real 21-dispatch K1 measurement in
`fixtures/k1_profile/`. Real names matter here: DroNet genuinely contains two
dispatches with the identical signature (`linear_s8_M1xK2048xN1`, its two heads)
and three more (`conv2d_batchnorm2d_s8_noshape`), and the IREE profiler's names
embed the dispatch index twice. Synthetic names would have missed both
subtleties that make this more than `key=module_name`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

import dispatch_lineage  # noqa: E402
from dispatch_lineage import join, op_signature  # noqa: E402

FIXTURE = os.path.join(_HERE, "fixtures", "k1_profile")
RVV_CSV = os.path.join(
    FIXTURE, "gen", "profile", "rvv_x60", "spacemit_x60", "dronet",
    "dronet.int8", "dronet_spacemit_x60_rvv_x60_dronet.int8", "topo_0",
    "results.csv")
SCALAR_CSV = os.path.join(
    FIXTURE, "gen", "profile", "scalar", "spacemit_x60", "dronet",
    "dronet.int8", "dronet_spacemit_x60_scalar_dronet.int8", "topo_0",
    "results.csv")

_MB_REWRITER = os.path.join(_REPO, "ModelBlaster", "pipeline",
                            "apply_fusion_hint.py")

#: maxpool2d + batchnorm2d(27x27). Chosen because both signatures occur
#: exactly once in DroNet: fusing a member of a REPEATED family is a
#: different case, and `test_a_family_whose_multiplicity_changed...` owns it.
FUSE_GROUP = [1, 2]


def _profile(path):
    """`{dispatch_id: {"module_name", "median_ms"}}` from a results.csv."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["dispatch_id"])] = {
                "module_name": row["module_name"],
                "median_ms": float(row["mean_time"]),
            }
    return out


def _fuse(profile, group):
    """Mirror `realize-hint`: collapse `group` into one op, renumber contiguously.

    Deliberately a local reimplementation of the *contract* rather than a call
    into ModelBlaster: the contract is what XPU-RT has to be robust against, and
    it must hold whether ModelBlaster is checked out beside this repo or not.
    `AgainstTheRealRewriter` below pins the reimplementation to the real one.

    `group` is a topologically-ordered run of pre-rewrite dispatch_ids. The
    fused op takes the group's first slot; its module name is the concatenation
    the rewriter builds (`__fused__<op0>__<op1>`), which is by construction not
    equal to any member's.
    """
    survivors = [d for d in sorted(profile) if d not in group]
    fused_at = min(group)
    order = sorted(survivors + [fused_at])
    out, remap = {}, {}
    for new_id, old_id in enumerate(order):
        if old_id == fused_at:
            names = [profile[d]["module_name"].split("$", 1)[-1] for d in group]
            out[new_id] = {
                "module_name": "dronet$dispatch_{}___fused__{}".format(
                    new_id, "__".join(names)),
                "median_ms": sum(profile[d]["median_ms"] for d in group),
            }
            for d in group:
                remap[d] = new_id
        else:
            rec = dict(profile[old_id])
            # The renumbering rewrites the index inside the module name too --
            # which is exactly why the raw string is not a stable key either.
            rec["module_name"] = rec["module_name"].replace(
                f"$dispatch_{old_id}_", f"$dispatch_{new_id}_", 1)
            out[new_id] = rec
            remap[old_id] = new_id
    return out, remap


def _split(profile, victim, n_tiles=2):
    """Mirror `apply_split_hint`: one op becomes `n_tiles`, ids shift up."""
    order = sorted(profile)
    out, remap = {}, {}
    new_id = 0
    for old_id in order:
        if old_id == victim:
            share = profile[old_id]["median_ms"] / n_tiles
            tiles = []
            for t in range(n_tiles):
                base = profile[old_id]["module_name"].split("_", 2)[-1]
                out[new_id] = {
                    "module_name": f"dronet$dispatch_{new_id}_{base}_tile_{t}",
                    "median_ms": share,
                }
                tiles.append(new_id)
                new_id += 1
            remap[old_id] = tiles
        else:
            rec = dict(profile[old_id])
            rec["module_name"] = rec["module_name"].replace(
                f"$dispatch_{old_id}_", f"$dispatch_{new_id}_", 1)
            out[new_id] = rec
            remap[old_id] = new_id
            new_id += 1
    return out, remap


class TheHazardIsReal(unittest.TestCase):
    """Before asserting the fix works, show the mistake it prevents."""

    def test_a_dispatch_id_join_pairs_different_ops(self):
        """A `dispatch_id` join across one fuse mispairs almost every dispatch.

        This is the assertion that makes the rest of the file non-vacuous. If a
        future rewriter stopped renumbering, this test would fail and every
        other test here would become an assertion about nothing -- which is the
        right way round for that news to arrive.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)

        mispaired = [
            did for did in sorted(set(before) & set(after))
            if op_signature(before[did]["module_name"])
            != op_signature(after[did]["module_name"])
        ]
        self.assertGreater(
            len(mispaired), 10,
            "fusing two ops must renumber everything downstream; if it does "
            "not, the lineage machinery below is guarding nothing")
        # And the mispairing is not harmless: the costs it would compare differ
        # by orders of magnitude, so it produces a large fake delta rather than
        # noise.
        ratios = [before[d]["median_ms"] / after[d]["median_ms"]
                  for d in mispaired if after[d]["median_ms"] > 0]
        self.assertGreater(max(ratios), 5.0,
                           "a mispaired join should show up as an implausible "
                           "speedup, which is how this was found")

    def test_the_raw_module_name_is_not_a_stable_key_either(self):
        """`key=module_name` is not the fix: the index is inside the name.

        ModelBlaster writes `dronet$dispatch_4_rvv_x60_conv2d_s8_...`, so
        renumbering rewrites the string. Anyone reaching for `module_name`
        without normalising it gets a key that changes for the same op.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)
        raw_before = {r["module_name"] for r in before.values()}
        raw_after = {r["module_name"] for r in after.values()}
        self.assertLess(
            len(raw_before & raw_after), 4,
            "raw module names must mostly differ after a renumbering -- if "
            "they matched, normalisation would be unnecessary")


class SignatureIsStable(unittest.TestCase):

    def test_the_index_is_stripped_everywhere_it_appears(self):
        """The IREE profiler embeds the index TWICE in one module name.

        `dronet$async_dispatch_1_embedded_elf_riscv_64_dronet$async_dispatch_1_conv_...`
        -- stripping only the first occurrence leaves a key that still moves
        when the graph is renumbered, which is the original bug in disguise.
        This is a real name from
        `gen/profile/RVV/spacemit_x60/dronet/dronet.q.int8/topo_0/profile.jsonl`.
        """
        at_1 = ("dronet$async_dispatch_1_embedded_elf_riscv_64_"
                "dronet$async_dispatch_1_conv_32x56x56x3x3x3_i8xi8xi32")
        at_7 = at_1.replace("dispatch_1_", "dispatch_7_")
        self.assertEqual(op_signature(at_1), op_signature(at_7))
        self.assertNotIn("_1_", op_signature(at_1).replace("i8xi8xi32", ""))

    def test_both_naming_conventions_survive(self):
        """ModelBlaster's `$dispatch_N` and IREE's `$async_dispatch_N`."""
        mb = "dronet$dispatch_4_rvv_x60_conv2d_s8_N1xIC32xIH14xIW14"
        self.assertEqual(op_signature(mb),
                         op_signature(mb.replace("dispatch_4", "dispatch_11")))

    def test_the_backend_tag_stays_in_the_signature(self):
        """rvv_x60 and scalar builds of the same op must NOT join.

        The two fixture profiles are the same DroNet graph compiled two ways.
        Their dispatch_ids are identical and their shapes are identical, so a
        signature that dropped the backend tag would happily pair them -- and
        report a 15x kernel improvement as a granularity result. Keeping the
        tag makes that a non-join, which is the honest answer; comparing
        implementations is `implementation_advice`'s job and needs no lineage.
        """
        rvv, scalar = _profile(RVV_CSV), _profile(SCALAR_CSV)
        j = join(rvv, scalar)
        self.assertEqual(j.matched, {},
                         "a scalar profile must not join to a vector one")
        self.assertEqual(len(j.only_before), len(rvv))
        self.assertEqual(len(j.only_after), len(scalar))


class JoinAcrossAFuse(unittest.TestCase):

    def test_every_surviving_op_is_paired_with_itself(self):
        """The load-bearing test: it FAILS if the join keys on dispatch_id.

        19 of DroNet's 21 dispatches survive fusing 2 and 3, and each must be
        matched to the row measuring the same kernel on the same shape. Under a
        dispatch_id join, 17 of those pairs carry different signatures (see
        `TheHazardIsReal`), so the signature assertion below fails for them.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)

        j = join(before, after)
        self.assertTrue(j.is_unambiguous, j.ambiguous)
        self.assertEqual(len(j.matched), len(before) - 2)
        for key, (b, a) in j.matched.items():
            self.assertEqual(op_signature(before[b]["module_name"]),
                             op_signature(after[a]["module_name"]), key)
            # And the cost travels with the op, not with the slot.
            self.assertAlmostEqual(before[b]["median_ms"],
                                   after[a]["median_ms"], places=9, msg=key)

    def test_the_fused_members_are_reported_gone_and_the_fused_op_new(self):
        """A fused group is not "unchanged" and not "missing data".

        The two members disappear and one op appears whose signature is neither
        of theirs. Collapsing that into a plain absence is how a rewrite's cost
        stops being accounted for at all.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)
        j = join(before, after)

        gone = {op_signature(before[d]["module_name"]) for d in FUSE_GROUP}
        self.assertEqual({k.rsplit("#", 1)[0] for k in j.only_before}, gone)
        self.assertEqual(len(j.only_after), 1)
        self.assertIn("__fused__", next(iter(j.only_after)))

    def test_repeated_signatures_keep_their_own_slots(self):
        """DroNet 18 and 19 are both `linear_s8_M1xK2048xN1`.

        Two real dispatches, identical kernel, identical shape -- the network
        has two heads. A dict keyed on the bare signature drops one of them, and
        which one it drops depends on iteration order. Both must survive the
        join as distinct entries.
        """
        before = _profile(RVV_CSV)
        sigs = [op_signature(before[d]["module_name"]) for d in (18, 19)]
        self.assertEqual(sigs[0], sigs[1],
                         "fixture must still contain the duplicate-signature "
                         "pair this test exists for")

        after, _ = _fuse(before, FUSE_GROUP)
        j = join(before, after)
        pairs = {k: v for k, v in j.matched.items() if sigs[0] in k}
        self.assertEqual(len(pairs), 2, "both linear_s8 heads must be matched")
        self.assertEqual({b for b, _ in pairs.values()}, {18, 19})
        self.assertEqual({a for _, a in pairs.values()}, {17, 18})

    def test_a_family_whose_multiplicity_changed_is_ambiguous_not_matched(self):
        """Fusing one of the two identical `linear_s8` heads.

        Now the survivor's ordinal has moved from #1 to #0 and nothing in the
        names says which head lived. Pairing by ordinal would match dispatch 18
        (fused away) to dispatch 19 (untouched) and report the fusion's saving
        against the wrong op. Refusing to match is the only honest outcome.
        """
        before = _profile(RVV_CSV)
        sig = op_signature(before[18]["module_name"])
        after, _ = _fuse(before, [17, 18])   # relu + the first linear head

        j = join(before, after)
        self.assertIn(sig, j.ambiguous,
                      "a repeated signature that lost a member must not be "
                      "silently paired by ordinal")
        self.assertEqual(j.ambiguous[sig], ([18, 19], [18]))
        self.assertFalse(j.is_unambiguous)
        self.assertNotIn(f"{sig}#0", j.matched)


class JoinAcrossASplit(unittest.TestCase):

    def test_tiles_are_new_ops_not_the_ops_that_moved_into_their_slots(self):
        """Splitting dispatch 1 shifts every later id up by one.

        The hazard here is the mirror image of the fuse case: under an id join,
        old dispatch 2 is compared against the *second tile* of dispatch 1 --
        two rows that have nothing to do with each other. The tiles must come
        back as `only_after` and the shifted ops must still match themselves.
        """
        before = _profile(RVV_CSV)
        after, remap = _split(before, victim=1, n_tiles=2)
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(remap[1], [1, 2])

        j = join(before, after)
        self.assertTrue(j.is_unambiguous, j.ambiguous)
        self.assertEqual(len(j.matched), len(before) - 1)
        for key, (b, a) in j.matched.items():
            self.assertEqual(op_signature(before[b]["module_name"]),
                             op_signature(after[a]["module_name"]), key)
        self.assertEqual(len(j.only_after), 2, "the two tiles are new ops")
        self.assertTrue(all("tile_" in k for k in j.only_after), j.only_after)

    def test_the_split_op_is_reported_gone_so_its_cost_is_not_double_counted(self):
        before = _profile(RVV_CSV)
        after, _ = _split(before, victim=1)
        j = join(before, after)
        self.assertEqual(
            {v for v in j.only_before.values()}, {1},
            "the split parent no longer exists and must be reported as such")


class RemapCrossCheck(unittest.TestCase):
    """`id_remap` is a claim by the rewriter; the names are the artifact."""

    def test_a_correct_remap_is_accepted(self):
        before = _profile(RVV_CSV)
        after, remap = _fuse(before, FUSE_GROUP)
        self.assertEqual(dispatch_lineage.check_id_remap(before, after, remap),
                         [])

    def test_an_identity_remap_is_caught(self):
        """An identity remap IS the dispatch_id join, written as a remap.

        This is the shape a stale remap takes -- one carried over from before
        the rewrite, or synthesised by a consumer that assumed ids were stable.
        It is indistinguishable from a correct remap by shape: same keys, same
        value type, same range, every entry a plain int. Only the signatures can
        tell, which is the argument for keeping this check rather than believing
        the field.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)
        identity = {i: i for i in sorted(before) if i in after}

        problems = dispatch_lineage.check_id_remap(before, after, identity)
        self.assertGreater(len(problems), 10, problems)
        self.assertTrue(all("signatures differ" in p for p in problems))

    def test_string_keys_from_json_are_handled(self):
        """`id_remap` round-trips through JSON, so its keys arrive as strings.

        A checker that silently matched nothing for a JSON-loaded remap would
        report "no problems" for every input, which is worse than not having it
        -- it would look like corroboration.
        """
        before = _profile(RVV_CSV)
        after, _ = _fuse(before, FUSE_GROUP)
        ints = {i: i for i in sorted(before) if i in after}
        strs = {str(k): v for k, v in ints.items()}
        self.assertEqual(
            dispatch_lineage.check_id_remap(before, after, strs),
            dispatch_lineage.check_id_remap(before, after, ints))
        self.assertTrue(dispatch_lineage.check_id_remap(before, after, strs))


@unittest.skipUnless(os.path.exists(_MB_REWRITER),
                     "ModelBlaster checkout absent; the rewriter under test "
                     "lives at ModelBlaster/pipeline/apply_fusion_hint.py")
class AgainstTheRealRewriter(unittest.TestCase):
    """Pin `_fuse` above to what `realize-hint` actually does.

    Without this, the local reimplementation could drift from the contract and
    every test in this file would keep passing while guarding a rewrite nobody
    performs.
    """

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_mb_fuse", _MB_REWRITER)
        cls.mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.mod)

    def _ir(self, n=7):
        """A linear int8 MLP-shaped IR, the case the fusion hint targets."""
        ops = []
        for i in range(n):
            ops.append({
                "name": f"mlp.{i}",
                "op": "linear_s8" if i % 2 == 0 else "elu_s8",
                "inputs": ["x" if i == 0 else f"t{i - 1}"],
                "outputs": [f"t{i}"],
                "shape": {"n": 16},
                "dispatch_id": i,
                "hardware_target": "any",
                "depends_on": [] if i == 0 else [i - 1],
            })
        return {"name": "mlp_control", "version": 1, "quant": "int8",
                "input": "x", "output": f"t{n - 1}", "tensors": {}, "ops": ops}

    def test_ids_really_are_reassigned_contiguously(self):
        with tempfile.TemporaryDirectory() as d:
            ir = os.path.join(d, "graph.json")
            out = os.path.join(d, "graph.fused.json")
            hint = os.path.join(d, "hint.json")
            with open(ir, "w") as f:
                json.dump(self._ir(), f)
            with open(hint, "w") as f:
                json.dump({"contract": "modelblaster.fusion_hints/v1",
                           "networks": [{"network": "mlp_control",
                                         "fuse_groups": [[0, 1]],
                                         "n_tiny": 2}]}, f)
            rc = self.mod.main(["--hint", hint, "--model", "mlp_control",
                                "--ir", ir, "--out", out])
            self.assertEqual(rc, 0)
            g = json.load(open(out))

        ids = sorted(o["dispatch_id"] for o in g["ops"])
        self.assertEqual(ids, list(range(len(ids))),
                         "the contract is contiguous reassignment")
        remap = {int(k): v for k, v in g["id_remap"].items()}
        self.assertNotEqual(
            remap, {i: i for i in remap},
            "if the real rewriter stopped renumbering, the hazard this module "
            "exists for would be gone and `_fuse` should be revisited")
        # Fusing ops 0 and 1 must shift 2..6 down by one -- the same rule
        # `_fuse` implements.
        self.assertEqual([remap[i] for i in range(7)], [0, 0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SingletonListRemapEntries(unittest.TestCase):
    """`apply_split_hint` writes EVERY entry as a list, singletons included.

    The B3 split graphs on disk are `{"0": [0, 1], "1": [2], "2": [3], ...}` --
    twenty-one entries of which twenty are plain one-to-one mappings that
    happen to be wrapped in a list. `check_id_remap` skipped every list value,
    so on the only remap format that exists in this tree it checked ZERO
    entries and returned no problems.

    Passing vacuously is the worst outcome for a check whose entire job is to
    catch a wrong claim, so these pin both halves: singletons are checked, and
    genuine one-to-many entries are still skipped.
    """

    def _pair(self):
        before = {0: {"module_name": "m$dispatch_0_rvv_conv2d_s8_A"},
                  1: {"module_name": "m$dispatch_1_rvv_linear_s8_B"}}
        after = {0: {"module_name": "m$dispatch_0_rvv_conv2d_s8_A"},
                 1: {"module_name": "m$dispatch_1_rvv_linear_s8_B"}}
        return before, after

    def test_a_singleton_list_is_checked_not_skipped(self):
        before, after = self._pair()
        # 0 -> [1] is a lie: a conv did not become a linear.
        problems = dispatch_lineage.check_id_remap(before, after, {0: [1]})
        self.assertTrue(problems, "a singleton list must be checked")
        self.assertIn("0 -> 1", problems[0])

    def test_a_correct_singleton_list_passes(self):
        before, after = self._pair()
        self.assertEqual(
            dispatch_lineage.check_id_remap(before, after, {0: [0], 1: [1]}),
            [])

    def test_a_real_one_to_many_split_is_still_skipped(self):
        """A split's pieces do not carry the parent's signature by
        construction, so checking them would report every split as broken."""
        before, after = self._pair()
        self.assertEqual(
            dispatch_lineage.check_id_remap(before, after, {0: [0, 1]}), [])

    def test_a_custom_signature_extractor_is_used(self):
        """An IR graph.json has no `module_name` -- it is assigned later by
        `profile_writer._module_name` -- so a caller holding IR supplies its
        own reader over the op's `op`/`shape` fields."""
        before = {0: {"op": "conv2d_s8", "shape": "A"}}
        after = {1: {"op": "linear_s8", "shape": "B"}}
        problems = dispatch_lineage.check_id_remap(
            before, after, {0: [1]},
            signature_of=lambda r: f"{r['op']}_{r['shape']}")
        self.assertTrue(problems)
        self.assertIn("conv2d_s8_A", problems[0])
        self.assertIn("linear_s8_B", problems[0])
