"""scripts/ros_baseline.py and data/ros_arms.json: the ROS 2 baseline as a selectable effort ladder.

No board, no GPU: everything here reads the repository (the matrix script, the run manifests, the
summary table) or runs ros_traced_matrix.sh with ssh stubbed out.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import gen_ros_arms as GEN  # noqa: E402
import ros_baseline as RB  # noqa: E402
import ros_effort_ladder as LAD  # noqa: E402

PY = sys.executable


def run(*args):
    return subprocess.run([PY, os.path.join(REPO, "scripts", "ros_baseline.py"), *args],
                          capture_output=True, text=True, cwd=REPO, timeout=300)


def test_arm_table_is_fresh():
    r = subprocess.run([PY, os.path.join(REPO, "scripts", "gen_ros_arms.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_ladder_arm_resolves_to_a_launch():
    arms = RB.arms()
    for key, _label, _trace in LAD.LADDER:
        e = arms[key.split("@")[0]]
        assert e["processes"], key
        assert e["matrix_command"].endswith(f"scripts/ros_traced_matrix.sh {e['base_arm']}")
        # the ladder arms' launches agree with what their latest runs recorded
        assert not e["manifest_mismatch"], (key, e["manifest_mismatch"])


def test_ladder_order():
    order = [r["arm"] for r in RB.ladder_rungs()]
    lad = list(dict.fromkeys(k.split("@")[0] for k, _, _ in LAD.LADDER))
    assert [a for a in order if a in lad] == lad          # the figure's order is kept
    assert order[0] == "vanilla" and order[1] == "vanilla_c50"
    assert order.index("cp3") == order.index("p3") - 1    # chained pinning, then its timer
    assert order.index("vanilla4_q1") < order.index("vanilla4tm") < order.index("p3") < order.index("p3_q1")
    assert order.index("vanilla4x2") < order.index("vanilla4x2tm") < order.index("vanilla4x2ns4c")
    assert order[-2:] == ["cp3n4", "cp3n4_d"]


def test_rung_changes_are_single_steps():
    by = {r["arm"]: r for r in RB.ladder_rungs()}
    assert by["vanilla4_q1"]["parent"] == "vanilla4" and "qos_depth 10->1" in by["vanilla4_q1"]["changes"]
    assert by["vanilla4_q1"]["takes"].startswith("one QoS setting")
    assert by["vanilla4tm"]["parent"] == "vanilla4" and "ctrl_mode chained->timer" in by["vanilla4tm"]["changes"]
    assert by["p3"]["parent"] == "cp3" and by["p3"]["takes"].startswith("one flag on the control node")
    assert "expert per-network core choice" in by["cp3"]["takes"]
    assert "second model instance" in by["vanilla4x2"]["takes"]
    assert "inert" in by["vanilla_c50"]["changes"]         # a chained control node runs no timer


def test_out_of_the_box_arm_is_every_ros2_default():
    e = RB.resolve("vanilla")
    d = GEN.load()["binary_defaults"]
    assert d["executor"] == "single" and d["qos_depth"] == 10 and d["yolo_pool"] == 0
    nodes = [p["effective"]["nodes"] for p in e["processes"]]
    assert sorted(nodes) == ["camera", "control", "nav", "perception"]   # one node per process
    for p in e["processes"]:
        eff = p["effective"]
        assert p["taskset"] is None and eff["pin_main"] == -1 and eff["expected_affinity_mask"] == "0xff"
        assert eff["executor"] == "single"                  # single-threaded executor
        assert eff["qos_depth"] == 10                       # keep-last-10
        assert eff["yolo_pool"] == 0                        # the model on the callback thread
        assert "--executor" not in p["flags"] and "--yolo-pool" not in p["flags"]
    ctl = next(p for p in e["processes"] if p["effective"]["nodes"] == "control")
    assert ctl["effective"]["ctrl_mode"] == "chained"       # control chained to perception
    assert RB.resolve("vanilla_c50")["base_arm"] == "vanilla"


def test_ladder_marks_unmeasured_rates():
    r = run("ladder", "--camera-hz", "45")
    assert r.returncode == 0, r.stderr
    block = r.stdout.split("vanilla4x2ns4c", 1)[1].split("\n12.", 1)[0]
    assert "not measured at 45 Hz" in block and "measured at 36 Hz" in block
    assert "control 100.0 Hz, camera->goal   31.1 ms" in r.stdout      # p3_q1 @45, docs/Baselines/ros_arm_ranking.md §1
    assert "ROS 5/192 vs XPU-RT 28/192" in r.stdout                    # vanilla4 @45, ros_ladder_paired.py


@pytest.mark.parametrize("arm,expect", [
    ("vanilla", ["ROS 2 defaults", "ros_vanilla45.csv", "Tier C: 18 board run(s)"]),
    ("p3_q1", ["QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh p3", "Tier A: supported", "ros_p3_q145.csv"]),
    ("vanilla4x2ns4", ["not recoverable"]),
    ("vanilla4x2", ["--alternate", "two perception instances", "ros_vanilla4x236.csv"]),
])
def test_show(arm, expect):
    r = run("show", arm)
    assert r.returncode == 0, r.stderr
    for s in expect:
        assert s in r.stdout, (arm, s)


def test_deploy_prints_the_arms_flags():
    r = run("deploy", "p3_q1", "--camera-hz", "45", "--reps", "2")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert 'RATES="45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh p3 2' in out   # r1 exists: not overwritten
    for flag in ("--qos-depth 1", "taskset -c 0-3", "--yolo-pool 4 --pool-harts 0,1,2,3", "--pin-main 0"):
        assert flag in out
    assert "scripts/ctrl_trace_from_board.py" in out and "scripts/pull_ros_traced.py" in out
    r = run("deploy", "vanilla4x2ns4c", "--camera-hz", "36", "--reps", "1")
    assert "BINSUF=_nav4 NAVPOOL=4 SUFFIX=ns4c scripts/ros_traced_matrix.sh vanilla4x2" in r.stdout
    assert "--nav-pool 4" in r.stdout
    assert run("deploy", "vanilla4x2ns4", "--camera-hz", "36").returncode == 2


def test_model_tier_a_and_refusals():
    r = run("model", "p3", "--tier", "A", "--camera-hz", "45")
    assert r.returncode == 0 and "camera->control" in r.stdout and "measured (Tier C" in r.stdout
    assert run("model", "vanilla4x2", "--tier", "A", "--camera-hz", "36").returncode == 2
    assert run("model", "spin", "--tier", "B", "--camera-hz", "45").returncode == 2
    assert run("model", "vanilla", "--tier", "A", "--camera-hz", "45").returncode == 2
    assert run("model", "vanilla", "--tier", "A", "--camera-hz", "45", "--force").returncode == 0
