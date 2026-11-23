#!/usr/bin/env python3
"""Host tests for timing_loop.py (no board, no solver).

  pytest scripts/k3_timing_loop                      # fast tests
  K3TL_REPLAY_...=... pytest scripts/k3_timing_loop -k replay      # replay of recorded rounds (minutes)

fast:   the stop rule; the table and sender's-delay drift checks; the state machine with every step replaced
        by a stub; resume after an interrupted step; fail closed when a round fails its checks; adopting
        rounds that were run before; what the table rule writes into a round's config.
replay: recorded board rounds through the REAL fit / apply / audit / plan summary / analysis, with the board
        and the solver replaced by the recorded raw folders and plan: the loop must rebuild the recorded
        corrected tree byte for byte. The recorded data is not in the repository; the test is skipped unless
        these point at it:
          K3TL_REPLAY_CONFIG      the workload config of the recorded rounds
          K3TL_REPLAY_RAW_TREE    the raw profile tree
          K3TL_REPLAY_ROUND0      raw folder of round 0 (20 runs)
          K3TL_REPLAY_ROUND1      raw folder of round 1 (20 runs), and
          K3TL_REPLAY_SCHEDULE1   the plan it ran
          K3TL_REPLAY_TREE1       the corrected tree round 1 was planned with (to be rebuilt)
        optional: K3TL_REPLAY_SUMMARY0 (recorded summary.json of round 0; its numbers must come out the same),
        K3TL_MODELBLASTER (checkout with cores/spacemit_k3.json), K3TL_PROFILE_AUDIT (see tree_audit.py).
"""
from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_loop as TL  # noqa: E402


# ------------------------------------------------------------------ stop rule
def model(planned, board, late=0, planned_late=0):
    return dict(planned_ms=planned, board_ms=board, late_frames=late, planned_late_frames=planned_late)


def test_stop_rule_converges_when_every_model_is_within_delta_and_on_time():
    v = TL.stop_rule({"a": model(4.0, 4.1), "b": model(1.0, 0.98)}, 0.03)
    assert v["converged"] and v["reasons"] == []


def test_stop_rule_rejects_a_model_outside_delta():
    v = TL.stop_rule({"a": model(4.0, 4.2), "b": model(1.0, 1.0)}, 0.03)
    assert not v["converged"] and "a:" in v["reasons"][0] and v["per_model"]["b"]["agrees"]


def test_stop_rule_is_two_sided():
    assert not TL.stop_rule({"a": model(5.0, 4.5)}, 0.03)["converged"]


def test_stop_rule_rejects_late_frames_of_a_model_the_plan_puts_on_time():
    v = TL.stop_rule({"a": model(4.0, 4.0, late=2)}, 0.03)
    assert not v["converged"] and "late frame" in v["reasons"][0]


def test_stop_rule_does_not_hold_planned_misses_against_the_loop():
    assert TL.stop_rule({"a": model(6.0, 6.05, late=40, planned_late=3)}, 0.03)["converged"]


def test_stop_rule_boundary_and_absolute_floor():
    assert TL.stop_rule({"a": model(1.0, 1.03)}, 0.03)["converged"]
    assert not TL.stop_rule({"a": model(0.05, 0.08)}, 0.03)["converged"]
    assert TL.stop_rule({"a": model(0.05, 0.08)}, 0.03, abs_tol_ms=0.05)["converged"]


# ------------------------------------------------------------------ the kept table and its drift check
TABLE = {"cpu_p": {"cpu_p": 0.0110, "cpu_e": 0.0169}, "cpu_e": {"cpu_p": 0.0137, "cpu_e": 0.0170}}
FLAT = 0.00056


def measured(**over):
    m = {"X100 -> X100": dict(n=5000, mean_gap_us=11.6, median_gap_us=11.5), "X100 -> A100": dict(n=9000, mean_gap_us=17.4, median_gap_us=15.3),
         "A100 -> X100": dict(n=9000, mean_gap_us=14.3, median_gap_us=13.7), "A100 -> A100": dict(n=400, mean_gap_us=17.5, median_gap_us=16.5)}
    m.update(over)
    return m


def test_table_drift_keeps_a_table_that_matches():
    v = TL.table_drift(TABLE, FLAT, measured())
    assert {x["verdict"] for x in v.values()} == {"kept"} and abs(v["X100 -> A100"]["off_by_us"]) < 0.1


def test_table_drift_marks_only_the_pair_that_moved():
    v = TL.table_drift(TABLE, FLAT, measured(**{"A100 -> A100": dict(n=400, mean_gap_us=22.4, median_gap_us=22.0)}))
    assert v["A100 -> A100"]["verdict"] == "updated" and v["A100 -> A100"]["off_by_us"] > 2
    assert [n for n, x in v.items() if x["verdict"] == "updated"] == ["A100 -> A100"]


def test_table_drift_does_not_update_on_too_few_samples():
    v = TL.table_drift(TABLE, FLAT, measured(**{"A100 -> A100": dict(n=27, mean_gap_us=30.0, median_gap_us=30.0)}))
    assert v["A100 -> A100"]["verdict"] == "too few samples"
    v = TL.table_drift(TABLE, FLAT, measured(**{"A100 -> A100": dict(n=0, mean_gap_us=float("nan"), median_gap_us=float("nan"))}))
    assert v["A100 -> A100"]["verdict"] == "too few samples" and v["A100 -> A100"]["measured_mean_us"] is None


def test_table_drift_missing_pair_is_a_clear_error():
    with pytest.raises(ValueError, match="no entry for the hand-off pair"):
        TL.table_drift(TABLE, FLAT, {k: v for k, v in measured().items() if k != "X100 -> X100"})
    no_aa = {"cpu_p": dict(TABLE["cpu_p"]), "cpu_e": {"cpu_p": 0.0137}}
    with pytest.raises(ValueError, match="the table has no value for A100 -> A100"):
        TL.table_drift(no_aa, FLAT, measured())
    assert TL.table_drift(no_aa, FLAT, measured(**{"A100 -> A100": dict(n=0, mean_gap_us=0, median_gap_us=0)}))["A100 -> A100"]["verdict"] == "too few samples"


SENDER = {"cpu_p": 0.0054, "cpu_e": 0.0097}


def sent(**over):
    m = {"X100": dict(n=5000, mean_gap_us=6.0, median_gap_us=5.8), "A100": dict(n=4000, mean_gap_us=10.3, median_gap_us=10.1)}
    m.update(over)
    return m


def test_sender_drift_kept_updated_too_few_and_missing():
    assert {v["verdict"] for v in TL.sender_drift(SENDER, FLAT, sent()).values()} == {"kept"}
    v = TL.sender_drift(SENDER, FLAT, sent(A100=dict(n=4000, mean_gap_us=14.0, median_gap_us=13.9)))
    assert v["A100"]["verdict"] == "updated" and v["X100"]["verdict"] == "kept"
    assert TL.sender_drift(SENDER, FLAT, sent(A100=dict(n=50, mean_gap_us=30.0, median_gap_us=30.0)))["A100"]["verdict"] == "too few samples"
    with pytest.raises(ValueError, match="no entry for the sender kind"):
        TL.sender_drift(SENDER, FLAT, {"X100": sent()["X100"]})
    with pytest.raises(ValueError, match="no sender's delay for A100"):
        TL.sender_drift({"cpu_p": 0.0054}, FLAT, sent())


def test_table_rule_writes_the_senders_delay_unless_turned_off(tmp_path):
    table = dict(handoff_ms=TABLE, flat_start_cost_ms=FLAT, sender_ms=SENDER)
    for extra, want in (((), SENDER), (("--no-sender",), None)):
        a = args(tmp_path, "--gap-rule", "table", *extra)
        lay = TL.Layout(a)
        os.makedirs(lay.corr_dir(1), exist_ok=True)
        with open(os.path.join(lay.corr_dir(1), "handoff_table.json"), "w") as f:
            json.dump(table, f)
        sched = json.load(open(TL.Steps(lay).write_config(1)))["scheduler"]
        assert sched["handoff_ms"] == TABLE and sched.get("sender_ms") == want


def test_a_higher_round_limit_on_a_later_call_continues_the_loop(tmp_path):
    state, _ = run(tmp_path, dict(ratios=[1.5]), "--max-rounds", "2")
    assert state["message"] == "stop: round limit reached" and len(state["rounds"]) == 2
    a = args(tmp_path, "--max-rounds", "3")
    again = StubSteps(TL.Layout(a), ratios=[1.5, 1.5, 1.0])
    state = TL.run_loop(a, again, say=lambda *_: None)
    assert len(state["rounds"]) == 3 and state["message"] == "stop: plan and board agree" and ("board", 1) not in again.calls


# ------------------------------------------------------------------ state machine with stub steps
class StubSteps(TL.Steps):
    """Every step is a stub; board ratios come from a script, and steps can be told to fail once."""

    def __init__(self, lay, ratios, fail_once=(), bad_round=None):
        super().__init__(lay)
        self.ratios, self.fail_once, self.bad_round = ratios, set(fail_once), bad_round
        self.calls = []

    def _maybe_fail(self, name, k):
        self.calls.append((name, k))
        if (name, k) in self.fail_once:
            self.fail_once.discard((name, k))
            raise RuntimeError(f"interrupted in {name} of round {k}")

    def make_tree(self, k, rounds):
        self._maybe_fail("tree", k)
        return dict(tree=f"tree_{k}", corrections=f"corr_{k}.json", rows_measured=10 * k, fitted_from=[r["tag"] for r in rounds], audit="PASS")

    def plan(self, k, tree, prev_schedule):
        self._maybe_fail("plan", k)
        return self.describe_plan(k, f"sched_{k}.json", f"cfg_{k}.json", tree, prev_schedule, None)

    def describe_plan(self, k, schedule, cfg, tree, prev_schedule, solve_log):
        return dict(schedule=schedule, config=cfg, solver_says="misses=0", models={"a": dict(planned_frame_ms=4.0, planned_late_frames=0, layers=10,
                    layers_moved=(3 if prev_schedule else None))}, rows_used=10, rows_used_still_guessed=max(0, 4 - k))

    def traces_complete(self, raw):
        return True

    def board(self, k, schedule, cfg):
        self._maybe_fail("board", k)
        return dict(raw=f"raw_{k}", runs=self.a.runs)

    def analyze(self, k, raw, tree):
        self._maybe_fail("analyze", k)
        if k == self.bad_round:
            raise TL.LoopStop(f"round {k}: 1 of {self.a.runs} runs failed a check")
        ratio = self.ratios[min(k, len(self.ratios) - 1)]
        return dict(summary=f"summary_{k}.json", runs=self.a.runs, runs_ok=self.a.runs, runs_with_any_late_frame=0, worst_lateness_ms=-1.0,
                    models={"a": dict(planned_ms=4.0, board_ms=4.0 * ratio, board_p75_ms=4.0 * ratio, board_max_ms=4.5, board_over_planned=ratio,
                                      late_frames=0, runs_with_a_late_frame=0, planned_late_frames=0, handoffs_per_frame=20.0, discrepancy_ms=0.0)})


def args(tmp_path, *extra):
    cfg = tmp_path / "networks_demo.json"
    cfg.write_text(json.dumps({"networks": {}}))
    return TL.parser().parse_args(["--out", str(tmp_path / "loop"), "--config", str(cfg), "--raw-tree", str(tmp_path), "--runs", "3",
                                   "--results-root", str(tmp_path / "results"), "--raw-root", str(tmp_path / "raw"),
                                   "--tree-root", str(tmp_path / "trees"), "--solve-dir", str(tmp_path / "solve"), *extra])


def run(tmp_path, steps_kw, *extra):
    a = args(tmp_path, *extra)
    steps = StubSteps(TL.Layout(a), **steps_kw)
    return TL.run_loop(a, steps, say=lambda *_: None), steps


def test_loop_stops_at_the_first_round_that_agrees(tmp_path):
    state, steps = run(tmp_path, dict(ratios=[1.47, 0.92, 1.01]))
    assert state["status"] == "stopped" and state["message"] == "stop: plan and board agree"
    assert [r["steps"]["verdict"]["decision"] for r in state["rounds"]] == ["next round", "next round", "stop: plan and board agree"]
    assert state["rounds"][2]["steps"]["tree"]["fitted_from"] == ["r0", "r1"]      # the fit sees EVERY finished round
    assert ("tree", 0) not in steps.calls                                         # round 0 plans with the raw tree
    report = (tmp_path / "loop" / "loop_report.md").read_text()
    assert "| r2 |" in report and "stop: plan and board agree" in report


def test_loop_stops_at_the_round_limit(tmp_path):
    state, _ = run(tmp_path, dict(ratios=[1.5]), "--max-rounds", "3")
    assert len(state["rounds"]) == 3 and state["message"] == "stop: round limit reached"


def test_loop_fails_closed_on_a_bad_round_and_fits_nothing_on_it(tmp_path):
    state, steps = run(tmp_path, dict(ratios=[1.5, 1.2, 1.0], bad_round=1))
    assert state["status"] == "failed" and "failed a check" in state["message"]
    assert ("tree", 2) not in steps.calls and len(state["rounds"]) == 2
    assert not state["rounds"][1]["steps"].get("analyze", {}).get("done")


def test_resume_after_an_interrupted_step_repeats_nothing_that_finished(tmp_path):
    a = args(tmp_path)
    first = StubSteps(TL.Layout(a), ratios=[1.5, 1.0], fail_once={("board", 1)})
    with pytest.raises(RuntimeError):
        TL.run_loop(a, first, say=lambda *_: None)
    saved = json.loads((tmp_path / "loop" / "loop_state.json").read_text())
    assert saved["rounds"][1]["steps"]["plan"]["done"] and not saved["rounds"][1]["steps"].get("board", {}).get("done")
    second = StubSteps(TL.Layout(a), ratios=[1.5, 1.0])
    state = TL.run_loop(a, second, say=lambda *_: None)
    assert state["status"] == "stopped" and state["message"] == "stop: plan and board agree"
    assert second.calls == [("board", 1), ("analyze", 1)]                          # round 0, tree 1 and plan 1 were NOT redone
    third = StubSteps(TL.Layout(a), ratios=[1.5, 1.0])
    assert TL.run_loop(a, third, say=lambda *_: None)["status"] == "stopped" and third.calls == []   # a finished loop does nothing


def test_start_from_adopts_hand_run_rounds_without_running_them(tmp_path):
    state, steps = run(tmp_path, dict(ratios=[1.5, 1.2, 1.0]), "--start-from", "raw_a", "raw_b,tree_b")
    assert [r.get("adopted", False) for r in state["rounds"]] == [True, True, False]
    assert ("board", 0) not in steps.calls and ("board", 1) not in steps.calls and ("plan", 1) not in steps.calls
    assert ("analyze", 0) in steps.calls and ("analyze", 1) in steps.calls         # adopted rounds are judged by the same checks
    assert state["rounds"][1]["steps"]["tree"]["tree"] == "tree_b" and ("board", 2) in steps.calls


def test_gap_rule_table_writes_the_measured_table_into_the_round_config(tmp_path):
    a = args(tmp_path, "--gap-rule", "table")
    lay = TL.Layout(a)
    steps = TL.Steps(lay)
    os.makedirs(lay.corr_dir(1))
    table = {"cpu_p": {"cpu_p": 0.011, "cpu_e": 0.017}, "cpu_e": {"cpu_p": 0.013, "cpu_e": 0.021}}
    with open(os.path.join(lay.corr_dir(1), "handoff_table.json"), "w") as f:
        json.dump({"handoff_ms": table}, f)
    assert json.load(open(steps.write_config(1)))["scheduler"]["handoff_ms"] == table
    assert "handoff_ms" not in json.load(open(steps.write_config(0))).get("scheduler", {})     # round 0: raw tree, no table
    a2 = args(tmp_path)                                                                       # the layer rule never writes one
    assert "handoff_ms" not in json.load(open(TL.Steps(TL.Layout(a2)).write_config(1))).get("scheduler", {})


# ------------------------------------------------------------------ the names the loop gives its jobs and folders
def test_everything_the_loop_writes_goes_under_out_by_default(tmp_path):
    cfg = tmp_path / "networks_demo.json"
    cfg.write_text(json.dumps({"networks": {}}))
    a = TL.parser().parse_args(["--out", str(tmp_path / "my loop"), "--config", str(cfg), "--raw-tree", str(tmp_path)])
    lay = TL.Layout(a)
    out = str(tmp_path / "my loop")
    assert all(p.startswith(out + os.sep) for p in (lay.results(1), lay.raw(1), lay.tree(1), lay.config(1), lay.schedule(1), lay.corr_dir(1)))
    assert lay.tree(0) == str(tmp_path)                                    # round 0 plans with the raw tree itself
    assert lay.job(2) == "my_loop_r2" and lay.build(2).endswith(os.path.join("build", "k3_timing_loop_my_loop_r2"))
    assert lay.modelblaster == os.path.join(TL.REPO, "zephyr-chipyard-sw", "modelblaster")
    assert TL.Layout(TL.parser().parse_args(["--out", "x", "--config", str(cfg), "--raw-tree", ".", "--modelblaster", str(tmp_path)])).modelblaster == str(tmp_path)


# ------------------------------------------------------------------ replay of recorded rounds (needs data outside the repo)
REPLAY = {k: os.environ.get(f"K3TL_REPLAY_{k}", "") for k in ("CONFIG", "RAW_TREE", "ROUND0", "ROUND1", "SCHEDULE1", "TREE1", "SUMMARY0")}
REPLAY_READY = all(REPLAY[k] and os.path.exists(REPLAY[k]) for k in ("CONFIG", "RAW_TREE", "ROUND0", "ROUND1", "SCHEDULE1", "TREE1"))


class ReplaySteps(TL.Steps):
    """Real fit / apply / audit / plan reading / analysis; the solver and the board are the recorded files."""

    def solve(self, k, tree):
        os.makedirs(os.path.dirname(self.lay.schedule(k)), exist_ok=True)
        shutil.copyfile(REPLAY["SCHEDULE1"], self.lay.schedule(k))

    def board(self, k, schedule, cfg):
        os.makedirs(self.lay.raw_root, exist_ok=True)
        if not os.path.exists(self.lay.raw(k)):
            os.symlink(os.path.abspath(REPLAY[f"ROUND{k}"]), self.lay.raw(k))
        return dict(raw=TL.rel(self.lay.raw(k)), runs=self.a.runs)

    def analyze(self, k, raw, tree):                 # the recorded folder's goldens, not the symlink's (empty) neighbourhood
        self.env["K3TL_GOLDEN_C_DIR"] = os.path.join(os.path.dirname(os.path.abspath(REPLAY[f"ROUND{k}"])), "_golden_check", "c_float")
        return super().analyze(k, raw, tree)


@pytest.mark.skipif(not REPLAY_READY, reason="needs recorded board rounds: set the K3TL_REPLAY_* variables (see the top of this file)")
def test_replay_rebuilds_the_recorded_tree_byte_for_byte(tmp_path):
    extra = ["--profile-audit", os.environ["K3TL_PROFILE_AUDIT"]] if os.environ.get("K3TL_PROFILE_AUDIT") else []
    a = TL.parser().parse_args(["--out", str(tmp_path / "loop"), "--config", REPLAY["CONFIG"], "--raw-tree", REPLAY["RAW_TREE"],
                                "--runs", "20", "--max-rounds", "2", "--prefix", "m", "--start-from", REPLAY["ROUND0"],
                                "--modelblaster", os.environ.get("K3TL_MODELBLASTER", "")] + extra)
    state = TL.run_loop(a, ReplaySteps(TL.Layout(a)), say=lambda *_: None)
    assert state["status"] == "stopped", state["message"]
    if REPLAY["SUMMARY0"]:
        hand, mine = json.load(open(REPLAY["SUMMARY0"])), json.load(open(tmp_path / "loop" / "results" / "m0" / "summary.json"))
        assert mine["models"] == hand["models"] and mine["late_frames_per_run"] == hand["late_frames_per_run"]
    new, old = str(tmp_path / "loop" / "trees" / "corr_m1"), REPLAY["TREE1"]

    def same(d):
        return not (d.left_only or d.right_only or d.diff_files or d.funny_files) and all(same(x) for x in d.subdirs.values())
    assert same(filecmp.dircmp(os.path.join(new, "profile"), os.path.join(old, "profile")))
    assert filecmp.cmp(os.path.join(new, TL.TREE_ENTRIES), os.path.join(old, TL.TREE_ENTRIES), shallow=False)
