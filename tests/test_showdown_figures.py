"""The rules the showdown figures count flights and board traces by, pinned on synthetic rows.

A figure panel is only as honest as the rule that turns campaign rows into a k/n, and none of these
rules are visible once they go wrong:

1. fair counts — per (cruise, seed) every drawn arm keeps the same number of flights (`equalise`),
   and one flight per seed elsewhere;
2. a scene is a scene — the people's height is part of the cell key, so the 1.7 m breaking-point
   campaign never pools into a 2.4 m panel (`flight_cells`, `dprime_counts`, `cells_by`);
3. the forest's summary counts each flight once and does not pretend the conditions are independent;
4. a waterfall's total is the chain's own median, not the sum of component medians;
5. a flight still airborne at the simulator's horizon is censored, and a seed one arm left airborne is
   dropped for both (`censor_and_pair`);
6. the crash definition in the simulator config (contact above 1 N) is the one the docs quote;
7. every latency the figures use as a CSV selector is a registry value that `measured_timing` re-derives.

Everything here runs on the host interpreter: no Isaac, no rendering (the figure modules import
matplotlib with the Agg backend and are used for their pure functions only).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "scripts"), os.path.join(_REPO, "sims", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("ENERGY_CSV", os.path.join(_REPO, "results", "codesign_feedback", "flight_energy_v2.csv"))

import showdown_v3_figure as V  # noqa: E402
import showdown_atlas as AT  # noqa: E402
import story_figures as SF  # noqa: E402
import make_measured_gantt_pair as MG  # noqa: E402

try:
    import showdown_final_figure as F  # noqa: E402
except Exception:   # pragma: no cover — the final figure imports the envelope panel, which needs the ablation CSV
    F = None
try:
    import figure_constants as FC  # noqa: E402
except ImportError:
    FC = None


def _load_verifier():
    path = os.path.join(_REPO, "scripts", "verify_showdown_figure.py")
    spec = importlib.util.spec_from_file_location("verify_showdown_figure", path)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)   # the module chdirs to the repo on import
    finally:
        os.chdir(cwd)
    return mod


def row(camp="campaign_percep", trace="xpu_a_cpsat_hard.csv", course="a", dens=0.30, walk=0.0, cross=0, gain=0.0055, lat=56.8, hold=0.0,
        cruise=1.0, gates=4, outcome="success", seed=1000, ph=2.4, crash_type=""):
    """one campaign row in the shape `load_campaigns` produces."""
    return dict(camp=camp, trace=trace, course=course, dens=dens, walk=walk, cross=cross, gain=gain, lat=lat, hold=hold, cruise=cruise,
                gates=gates, outcome=outcome, crash_type=crash_type, seed=seed, ph=ph)


X, R, G, P = "xpu_a_cpsat_hard.csv", "ros_vanilla445.csv", "xpu_a_greedy.csv", "ros_p345.csv"


def cells_of(rows):
    out = {}
    for r in rows:
        out.setdefault((r["trace"], r["cruise"]), []).append(r)
    return out


class Equalise(unittest.TestCase):
    """the fair-count rule: per (cruise, seed) keep the first k flights of each drawn arm, k = the fewest any arm flew."""

    def test_min_replicates_per_seed_and_the_set_aside_count(self):
        rows = [row(trace=X, seed=1, outcome="success"), row(trace=X, seed=1, outcome="crash", gates=1), row(trace=X, seed=1, outcome="crash", gates=2),
                row(trace=X, seed=2, outcome="success"),
                row(trace=R, seed=1, outcome="crash", gates=1), row(trace=R, seed=2, outcome="crash", gates=1), row(trace=R, seed=2, outcome="crash", gates=2)]
        cells, aside = V.equalise(cells_of(rows), [X, R])
        self.assertEqual(len(cells[(X, 1.0)]), 2)              # seed 1: 3 vs 1 -> 1 each; seed 2: 1 vs 2 -> 1 each
        self.assertEqual(len(cells[(R, 1.0)]), 2)
        self.assertEqual(aside, 3)
        self.assertEqual([r["outcome"] for r in cells[(X, 1.0)]], ["success", "success"])   # the FIRST replicate is the one kept

    def test_arms_not_drawn_are_left_untouched(self):
        rows = [row(trace=X, seed=1), row(trace=R, seed=1), row(trace=G, seed=1), row(trace=G, seed=1), row(trace=G, seed=7)]
        cells, aside = V.equalise(cells_of(rows), [X, R])
        self.assertEqual(len(cells[(G, 1.0)]), 3)
        self.assertEqual(aside, 0)

    def test_a_seed_only_one_arm_flew_is_kept_for_that_arm(self):
        # documents the current behaviour: `equalise` matches replicate counts, `censor_and_pair` is what intersects seeds
        rows = [row(trace=X, seed=1), row(trace=X, seed=9), row(trace=R, seed=1)]
        cells, aside = V.equalise(cells_of(rows), [X, R])
        self.assertEqual({r["seed"] for r in cells[(X, 1.0)]}, {1, 9})
        self.assertEqual(aside, 0)

    def test_cruises_are_independent(self):
        rows = [row(trace=X, seed=1, cruise=1.0), row(trace=X, seed=1, cruise=1.0), row(trace=R, seed=1, cruise=1.0),
                row(trace=X, seed=1, cruise=1.2), row(trace=R, seed=1, cruise=1.2), row(trace=R, seed=1, cruise=1.2)]
        cells, aside = V.equalise(cells_of(rows), [X, R])
        self.assertEqual((len(cells[(X, 1.0)]), len(cells[(R, 1.0)]), len(cells[(X, 1.2)]), len(cells[(R, 1.2)])), (1, 1, 1, 1))
        self.assertEqual(aside, 2)


class FlightCells(unittest.TestCase):
    """the cell key: course, density, gain, walk, crossing, people height; latency-replayed rows only."""

    def test_the_breaking_point_scene_never_pools_into_the_tall_scene(self):
        rows = [row(camp="campaign_percep", ph=2.4, seed=s) for s in range(12)] + [row(camp="campaign_break", ph=1.7, seed=s) for s in range(12)]
        tall = V.flight_cells(rows, "a", 0.30, 0.0055)
        short = V.flight_cells(rows, "a", 0.30, 0.0055, person_h=1.7)
        self.assertEqual(len(tall[(X, 1.0)]), 12)
        self.assertTrue(all(r["ph"] == 2.4 for r in tall[(X, 1.0)]))
        self.assertEqual(len(short[(X, 1.0)]), 12)
        self.assertTrue(all(r["ph"] == 1.7 for r in short[(X, 1.0)]))

    def test_latency_replay_filter_and_camps(self):
        rows = [row(lat=56.8, camp="campaign_percep"), row(lat=0.0, camp="campaign_env"), row(lat=56.8, camp="campaign_seeds24")]
        with_lat = V.flight_cells(rows, "a", 0.30, 0.0055)
        self.assertEqual(len(with_lat[(X, 1.0)]), 2)
        cadence_only = V.flight_cells(rows, "a", 0.30, 0.0055, latency=False)
        self.assertEqual(len(cadence_only[(X, 1.0)]), 3)
        one_camp = V.flight_cells(rows, "a", 0.30, 0.0055, camps={"campaign_seeds24"})
        self.assertEqual([r["camp"] for r in one_camp[(X, 1.0)]], ["campaign_seeds24"])

    def test_gain_per_arm(self):
        rows = [row(trace=X, gain=0.0052), row(trace=R, gain=0.01277), row(trace=R, gain=0.0055)]
        cells = V.flight_cells(rows, "a", 0.30, latency=True, gain_by_trace={X: 0.0052, R: 0.01277})
        self.assertEqual(len(cells[(X, 1.0)]), 1)
        self.assertEqual([r["gain"] for r in cells[(R, 1.0)]], [0.01277])

    def test_course_density_and_crossing_split_cells(self):
        rows = [row(course="a"), row(course="b"), row(dens=0.40), row(cross=1), row(walk=1.5)]
        self.assertEqual(sum(len(v) for v in V.flight_cells(rows, "a", 0.30, 0.0055).values()), 1)
        self.assertEqual(sum(len(v) for v in V.flight_cells(rows, "b", 0.30, 0.0055).values()), 1)
        self.assertEqual(sum(len(v) for v in V.flight_cells(rows, "a", 0.40, 0.0055).values()), 1)
        self.assertEqual(sum(len(v) for v in V.flight_cells(rows, "a", 0.30, 0.0055, cross=1).values()), 1)


@unittest.skipUnless(hasattr(V, "dprime_counts"), "showdown_v3_figure.dprime_counts not landed yet")
class Dprime(unittest.TestCase):
    """the environment map pools one people height; a column needs every family's 12 flights."""

    def _scene(self, ph, camp, outcome_x="success", n=12):
        return ([row(camp=camp, trace=X, ph=ph, seed=s, outcome=outcome_x) for s in range(n)]
                + [row(camp=camp, trace=R, lat=242.0, ph=ph, seed=s, outcome="crash", gates=1) for s in range(n)]
                + [row(camp=camp, trace=P, lat=55.8, ph=ph, seed=s, outcome="crash", gates=2) for s in range(n)])

    def test_breaking_point_rows_do_not_displace_the_tall_scene(self):
        # the 1.7 m rows come from a campaign whose name sorts first, exactly as on disk
        rows = self._scene(1.7, "campaign_break", outcome_x="crash") + self._scene(2.4, "campaign_percep", outcome_x="success")
        envs, K, N = V.dprime_counts(rows, ("xpu", "ros", "p3"), (1.0, 1.2, 1.4), 0.0055, person_h=2.4)
        self.assertEqual(envs, [("a", 0.30, 0)])
        self.assertEqual(int(N[0, 0]), 12)
        self.assertEqual(int(K[0, 0]), 12)   # the 2.4 m XPU-RT flights all completed; the 1.7 m ones (all crashes) were not pooled
        envs17, K17, N17 = V.dprime_counts(rows, ("xpu", "ros", "p3"), (1.0, 1.2, 1.4), 0.0055, person_h=1.7)
        self.assertEqual(int(K17[0, 0]), 0)

    def test_a_column_needs_twelve_flights_of_every_family(self):
        rows = self._scene(2.4, "campaign_percep") + [row(course="b", camp="campaign_percep", trace=X, seed=s) for s in range(12)]   # course B: XPU-RT only
        envs, K, N = V.dprime_counts(rows, ("xpu", "ros", "p3"), (1.0, 1.2, 1.4), 0.0055)
        self.assertEqual(envs, [("a", 0.30, 0)])

    def test_one_flight_per_seed(self):
        rows = self._scene(2.4, "campaign_percep") + [row(camp="campaign_percep", trace=X, seed=0, outcome="crash", gates=0) for _ in range(5)]
        envs, K, N = V.dprime_counts(rows, ("xpu", "ros", "p3"), (1.0, 1.2, 1.4), 0.0055)
        self.assertEqual(int(N[0, 0]), 12)


@unittest.skipUnless(hasattr(AT, "forest_rows") and hasattr(AT, "forest_summary"), "showdown_atlas.forest_rows / forest_summary not landed yet")
class Forest(unittest.TestCase):
    """the ablation forest: per-condition deltas and a summary that counts each flight once."""

    def _conds(self):
        def Fl(lat=56.8, hold=0.0, gain=0.0055, course="a", dens=0.30, walk=0.0, cross=0, ph=2.4):
            return dict(lat=lat, hold=hold, gain=gain, course=course, dens=dens, walk=walk, cross=cross, ph=ph)
        return [("cond one", X, Fl(), R, Fl(242.0), ("campaign_percep",), "vanilla"),
                ("cond two (same XPU-RT flights)", X, Fl(), P, Fl(55.8), ("campaign_percep",), "hand-pinned")]

    def _rows(self):
        rows = [row(trace=X, seed=s, cruise=c, outcome="success" if s % 2 else "crash", gates=1) for s in range(12) for c in (1.0, 1.2, 1.4)]
        rows += [row(trace=R, lat=242.0, seed=s, cruise=c, outcome="crash", gates=1) for s in range(12) for c in (1.0, 1.2, 1.4)]
        rows += [row(trace=P, lat=55.8, seed=s, cruise=c, outcome="success" if s % 3 == 0 else "crash", gates=1) for s in range(12) for c in (1.0, 1.2, 1.4)]
        return rows

    def test_shared_flights_are_counted_once_in_the_summary(self):
        res, fbc = AT.forest_rows(self._rows(), self._conds())
        self.assertEqual(len(res), 2)
        kx, nx = res[0][2], res[0][3]
        self.assertEqual((kx, nx), (18, 36))
        summ = AT.forest_summary(res, fbc, seed=0, n_boot=500)
        self.assertEqual(summ["summed_counts"][1], 72)             # the sum over rows counts the XPU-RT flights twice
        self.assertEqual(summ["distinct_flights"]["xpu"], [18, 36])   # the summary does not
        self.assertEqual(summ["distinct_flights"]["ros"][1], 72)      # the two baselines are different flights
        self.assertEqual(summ["n_conditions"], 2)

    def test_mean_delta_is_the_mean_of_the_rows_and_the_ci_is_seeded(self):
        res, fbc = AT.forest_rows(self._rows(), self._conds())
        s1 = AT.forest_summary(res, fbc, seed=0, n_boot=500); s2 = AT.forest_summary(res, fbc, seed=0, n_boot=500)
        self.assertAlmostEqual(s1["mean_delta"], sum(r[6] for r in res) / len(res), places=3)   # the summary rounds for the sidecar
        self.assertEqual(s1["ci"], s2["ci"])
        self.assertLessEqual(s1["ci"][0], s1["mean_delta"] + 1e-9)
        self.assertGreaterEqual(s1["ci"][1], s1["mean_delta"] - 1e-9)
        self.assertIn("bootstrap", s1["method"])

    def test_newcombe_is_antisymmetric(self):
        d, lo, hi = AT.newcombe(7, 48, 2, 48); d2, lo2, hi2 = AT.newcombe(2, 48, 7, 48)
        self.assertAlmostEqual(d, -d2); self.assertAlmostEqual(lo, -hi2); self.assertAlmostEqual(hi, -lo2)
        self.assertLess(lo, d); self.assertGreater(hi, d)


@unittest.skipUnless(hasattr(SF, "waterfall_summary"), "story_figures.waterfall_summary not landed yet")
class Waterfall(unittest.TestCase):
    """the label is the chain's median; the bars are component medians; the two totals differ."""

    def test_sum_of_component_medians_is_not_the_chain_median(self):
        w = [(0.0, 40.0, 5.0, 1.0), (0.0, 60.0, 5.0, 1.0), (30.0, 50.0, 5.0, 1.0)]   # chain sums: 46, 66, 86 -> median 66; component medians: 0+50+5+1 = 56
        s = SF.waterfall_summary(w)
        self.assertEqual(s["frames"], 3)
        self.assertEqual(s["parts_ms"], [0.0, 50.0, 5.0, 1.0])
        self.assertEqual(s["sum_of_part_medians_ms"], 56.0)
        self.assertEqual(s["chain_median_ms"], 66.0)
        self.assertNotEqual(s["sum_of_part_medians_ms"], s["chain_median_ms"])


@unittest.skipUnless(F is not None and hasattr(F, "censor_and_pair"), "showdown_final_figure.censor_and_pair not landed yet")
class TimeoutCensoring(unittest.TestCase):
    """a flight airborne at the horizon is neither a success nor a collision; a seed one arm left airborne is dropped for both."""

    def test_timeouts_removed_and_seeds_paired(self):
        rows = [row(camp="campaign_break", ph=1.7, trace=X, seed=1), row(camp="campaign_break", ph=1.7, trace=X, seed=2, outcome="timeout", gates=2),
                row(camp="campaign_break", ph=1.7, trace=X, seed=3), row(camp="campaign_break", ph=1.7, trace=X, seed=4),
                row(camp="campaign_break", ph=1.7, trace=R, lat=242.0, seed=1, outcome="crash", gates=1), row(camp="campaign_break", ph=1.7, trace=R, lat=242.0, seed=2, outcome="crash", gates=1),
                row(camp="campaign_break", ph=1.7, trace=R, lat=242.0, seed=3, outcome="timeout", gates=1),
                row(camp="campaign_break", ph=1.7, trace=G, lat=748.0, seed=1, outcome="timeout", gates=0)]
        cells = V.flight_cells(rows, "a", 0.30, 0.0055, person_h=1.7, camps={"campaign_break"})
        out, n_air = F.censor_and_pair(cells, X, R)
        self.assertEqual(n_air, 3)
        self.assertEqual({r["seed"] for r in out[(X, 1.0)]}, {1})     # seed 2 XPU-RT airborne, seed 3 ROS 2 airborne, seed 4 ROS 2 never flew
        self.assertEqual({r["seed"] for r in out[(R, 1.0)]}, {1})
        self.assertTrue(all(r["outcome"] != "timeout" for v in out.values() for r in v))
        self.assertEqual(out.get((G, 1.0), []), [])                     # greedy is censored, not paired


@unittest.skipUnless(hasattr(MG, "pick_busy_source"), "make_measured_gantt_pair.pick_busy_source not landed yet")
class BusySource(unittest.TestCase):
    """which instrument the Gantt lanes' busy % comes from, and that the sidecar names it."""

    def test_precedence(self):
        self.assertEqual(MG.pick_busy_source(True, True, True, True), "sampler_rdtime")
        self.assertEqual(MG.pick_busy_source(False, True, True, True), "hart_acc")
        self.assertEqual(MG.pick_busy_source(True, False, False, True), "sampler_epoch")
        self.assertIsNone(MG.pick_busy_source(False, False, False, False))


class CrashThreshold(unittest.TestCase):
    """the simulator's crash definition, read from the config's AST, is the one the docs quote."""

    CFG = os.path.join(_REPO, "sims", "isaaclab_tasks", "warehouse_nav", "config", "crazyflie", "warehouse_nav_env_cfg.py")

    def _terms(self):
        tree = ast.parse(open(self.CFG).read()); out = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DoneTerm":
                func = next((k.value.id for k in node.keywords if k.arg == "func" and isinstance(k.value, ast.Name)), None)
                params = next((k.value for k in node.keywords if k.arg == "params"), None)
                vals = {}
                if isinstance(params, ast.Dict):
                    for kk, vv in zip(params.keys, params.values):
                        if isinstance(kk, ast.Constant) and isinstance(vv, ast.Constant):
                            vals[kk.value] = vv.value
                out[func] = vals
        return out

    def test_contact_threshold_is_one_newton_and_the_ground_floor_ten_centimetres(self):
        terms = self._terms()
        self.assertIn("illegal_contact", terms)
        self.assertEqual(terms["illegal_contact"].get("threshold"), 1.0)
        self.assertEqual(terms.get("root_height_below_minimum", {}).get("minimum_height"), 0.1)

    def test_the_docs_quote_the_same_threshold(self):
        docs = "".join(open(os.path.join(_REPO, "docs", d)).read() for d in ("Artifact/reproduction_full.md", "Evaluation/measurements_and_ablations.md"))
        self.assertTrue("above 1 N" in docs or "above 1.0 N" in docs, "the reproduction docs must state the 1 N contact threshold the config enforces")


@unittest.skipUnless(FC is not None, "figure_constants (the registry) not landed yet")
class Registry(unittest.TestCase):
    """every latency the figures select CSV cells by is a registry value, and the registry re-derives from measured_timing."""

    EXEMPT = ("ros_spin45_r1.csv", "xpurt_best45alt2long.csv", "xpurt_greedy_long25.csv")

    def test_every_replayed_trace_has_an_arm(self):
        tr_dir = os.path.join(_REPO, "results", "codesign_feedback", "ctrl_traces")
        traces = [f for f in sorted(os.listdir(tr_dir)) if f.endswith(".csv") and "_smoke" not in f and f not in self.EXEMPT]
        missing = [t for t in traces if t not in FC.ARM_BY_TRACE]
        self.assertEqual(missing, [], f"ctrl traces without a registry arm: {missing}")

    def test_every_csv_selector_in_the_figure_scripts_resolves(self):
        keys = {(a[1], a[2].get("lat", 56.8), a[2].get("hold", 0.0)) for a in AT.CONDS} | {(a[3], a[4].get("lat", 56.8), a[4].get("hold", 0.0)) for a in AT.CONDS} if hasattr(AT, "CONDS") else set()
        keys |= {(tr, lat, hold) for _, _, tr, lat, hold in SF.RATE_ARMS}
        bad = []
        for tr, lat, hold in keys:
            if lat <= 0:
                continue   # cadence-only cells carry no latency
            try:
                arm = FC.arm_for(tr, lat, hold)
            except Exception as e:   # noqa: BLE001
                bad.append((tr, lat, hold, str(e))); continue
            if abs(arm.csv_lat - lat) > 1e-6:
                bad.append((tr, lat, hold, arm.csv_lat))
        self.assertEqual(bad, [])

    def test_fallback_raises_when_strict_and_records_when_allowed(self):
        with mock.patch.dict(os.environ, {"XPURT_FIG_ALLOW_FALLBACK": "0"}):
            with self.assertRaises(FC.MissingMeasurement):
                FC.fallback("eff_hz", 100.0, "dump has no eff_cmd_hz")
        n0 = len(FC.FALLBACKS)
        with mock.patch.dict(os.environ, {"XPURT_FIG_ALLOW_FALLBACK": "1"}):
            self.assertEqual(FC.fallback("eff_hz", 100.0, "dump has no eff_cmd_hz"), 100.0)
        self.assertEqual(len(FC.FALLBACKS), n0 + 1)
        self.assertEqual(FC.FALLBACKS[-1]["name"], "eff_hz")
        del FC.FALLBACKS[-1]

    def test_labels_are_derived_from_the_registry(self):
        self.assertRegex(FC.lat_label("xpu_a_cpsat_hard.csv"), r"^\d+ ms$")
        self.assertRegex(FC.ctrl_hz_label("ros_vanilla445.csv"), r"^\d+ Hz$")
        self.assertIn(FC.lat_label("xpu_a_cpsat_hard.csv"), FC.allowed_display_literals())

    def test_module_docstring_examples_are_still_true(self):
        """figure_constants' own usage examples are the first thing a reader trusts.

        An example such as ctrl_hz_label("ros_vanilla445.csv") -> "38 Hz" drifts when the recorded
        gap changes, and nothing else checks the docstring; this evaluates every
        `FC.<call>(...)  # "<literal>"` line in it against the live function.
        """
        import re as _re
        pat = _re.compile(r'^\s*FC\.(\w+)\((\"[^\"]*\")\)\s*#\s*(?:"([^"]*)"|([\d.]+))')
        checked = 0
        for line in (FC.__doc__ or "").splitlines():
            m = pat.match(line)
            if not m:
                continue
            fn, arg, want_s, want_n = m.group(1), m.group(2).strip('"'), m.group(3), m.group(4)
            got = getattr(FC, fn)(arg)
            want = want_s if want_s is not None else float(want_n)
            self.assertEqual(got, want, f"figure_constants docstring: FC.{fn}({arg!r}) is {got!r}, "
                                        f"the docstring says {want!r}")
            checked += 1
        self.assertGreaterEqual(checked, 3, "the docstring's examples stopped being parseable")

    def test_no_board_timing_literal_outside_the_registry(self):
        ver = _load_verifier(); allowed = FC.allowed_display_literals(); found = {}
        for sp in ver.FIGURE_SCRIPTS:
            hits = ver._check_literals(os.path.join(_REPO, sp), allowed=allowed)
            if hits:
                found[sp] = hits
        self.assertEqual(found, {}, f"ms/Hz literals outside the registry: {found}")


class VerifierLiteralScan(unittest.TestCase):
    """the literal scan reads string literals only: docstrings and comments are not drawn text."""

    def test_scan_skips_docstrings_and_finds_fstring_parts(self):
        import tempfile
        src = '"""module docstring says 999 ms"""\n\ndef f(x):\n    """function docstring 888 Hz"""\n    # a comment with 777 ms\n    return f"chain {x} ms at 45 Hz" + "242 ms"\n'
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src); path = fh.name
        try:
            ver = _load_verifier()
            hits = ver._check_literals(path, allowed={"45 Hz"})
            self.assertEqual(sorted(h for _, h, _ in hits), ["242 ms"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
