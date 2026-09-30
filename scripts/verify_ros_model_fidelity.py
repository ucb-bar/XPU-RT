#!/usr/bin/env python3
"""The ROS 2 model-fidelity figures, re-derived by re-running their producer from the traces.

`ros_model_fidelity_figure.py` draws the ROS 2 baseline at three tiers -- the analytical (Tier A) model,
the same recurrence on board-profiled costs, the board -- and records in each stem's `_metrics.json`
every number it drew: per (arm, rate) row the three tiers' camera->goal latency and control rate, the
residual statistics and which rows they exclude, the Tier A metric on each input set, the control-rate
literal and the file it was read from, the census and every flight record behind the story's panels,
the tier-B timeline, and the sha256 of every input.

A recorded number is only as good as the check that can fail it. This verifier re-runs the producer into
a tempdir -- with the parse cache switched off, so every per-node cost is re-read from the board traces --
and compares every leaf of every sidecar with the committed one, numbers by tolerance, everything else
exactly. It then checks that the headline statistics each sidecar states follow from the rows the same
sidecar records, so a hand-edited summary cannot sit on top of rows that say something else.

    scripts/verify_ros_model_fidelity.py [--stem NAME ...] [-v]

Needs the repository (traces, schedules, campaign CSV) and, for the story stems, the untracked flight
records under results/codesign_feedback/campaign_percep/records/.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from verify_hil_feedback import close   # noqa: E402  (numbers by tolerance, containers walked)

# `written` is the render time and `figure` names the output path, which is the tempdir on the re-run
VOLATILE = {"written", "figure"}
FAILED = PASSED = 0


def check(ok, msg):
    global FAILED, PASSED
    print(("PASS  " if ok else "FAIL  ") + msg)
    FAILED += not ok; PASSED += bool(ok)
    return ok


def leaves(o, path=""):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from leaves(o[k], f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def compare(have, got, verbose):
    """Every leaf of the committed sidecar against the re-derived one, grouped by top-level section."""
    for sec in sorted(set(have) | set(got)):
        if sec in VOLATILE:
            continue
        if not check(sec in have and sec in got, f"section `{sec}` present in both"):
            continue
        h, g = dict(leaves(have[sec])), dict(leaves(got[sec]))
        bad = [k for k in sorted(set(h) | set(g)) if k not in h or k not in g or not close(h[k], g[k], 1e-6)]
        # a recorded file is identified by its sha256; where it sits on disk is machine state
        bad = [k for k in bad if not (k.endswith(".path") and k[:-5] + ".sha256" in h
                                      and h.get(k[:-5] + ".sha256") == g.get(k[:-5] + ".sha256"))]
        if verbose:
            for k in sorted(h):
                print(f"      {sec}.{k} = {h[k]!r}" + ("   <-- differs" if k in bad else ""))
        check(not bad, f"`{sec}`: {len(h)} recorded values re-derive" + (f" -- {len(bad)} differ, first "
              f"{bad[0]}: recorded {h.get(bad[0])!r}, re-derived {g.get(bad[0])!r}" if bad else ""))


def consistency(stem, s):
    """The sidecar's own summaries, recomputed from the rows it records."""
    if s.get("option") == "three_tier":
        rows = s["rows"]; un = [r for r in rows if not r["saturated"]]
        for tier, k in (("A_submitted", "chain_assumed_submitted_ms"), ("A_recost", "chain_assumed_ms"),
                        ("B_profiled", "chain_model_queue_ms")):
            m = statistics.mean(r[k] - r["chain_measured_ms"] for r in un)
            check(close(m, s["residuals"]["latency_unsaturated"][tier]["mean_ms"], 1e-9),
                  f"{stem}: unsaturated camera->goal residual {tier} {m:+.2f} ms follows from its {len(un)} rows")
        for tier, k in (("A_submitted", "ctrl_assumed_submitted_hz"), ("B_profiled", "ctrl_pred_hz")):
            e = [abs(r[k] - r["ctrl_measured_hz"]) for r in rows]
            check(close(statistics.mean(e), s["residuals"]["ctrl_all_rows"][tier]["mean_abs_hz"], 1e-9)
                  and close(max(e), s["residuals"]["ctrl_all_rows"][tier]["max_abs_hz"], 1e-9),
                  f"{stem}: control-rate error {tier} mean {statistics.mean(e):.2f} max {max(e):.2f} Hz follows from its rows")
        check(sorted(f"{r['arm']}@{r['camera_hz']}" for r in rows if r["saturated"]) == sorted(s["residuals"]["saturated_rows"]),
              f"{stem}: the excluded rows are exactly the saturated ones ({len(s['residuals']['saturated_rows'])})")
    if "numbers" in s:
        n = s["numbers"]; k = n["instance"]
        check(close(n["response_ms"]["B_profiled"], n["series_ms"]["B_profiled"][k], 1e-12)
              and close(n["ratio_to_xpu"]["B_profiled"], n["response_ms"]["B_profiled"] / n["response_ms"]["xpu_as_reported"], 1e-12),
              f"{stem}: instance-{k} profiled response {n['response_ms']['B_profiled']:.2f} ms and its "
              f"{n['ratio_to_xpu']['B_profiled']:.2f}x ratio follow from the recorded series")
        c = n["control"]
        check(close(c["literal_rate_hz"], 1000.0 / (10.0 * -(-c["literal_response_ms"] // 10.0)), 1e-9),
              f"{stem}: {c['literal_response_ms']:.2f} ms under the 10 ms hold is {c['literal_rate_hz']:.0f} Hz")
        src = c["literal_source"]
        if src.get("present"):
            check(src.get("drawn_ros_literal_is_dronet") is True,
                  f"{stem}: the 12.40 ms literal is the dronet line of {os.path.basename(src['path'])} ({src['per_net_worst_response_ms'].get('dronet')} ms)")
    if s.get("option") == "story":
        for t, f in s["flights"].items():
            check(f["k"] == sum(x["outcome"] == "success" for x in f["flights"]) and f["n"] == len(f["flights"]),
                  f"{stem}: panel A {t} {f['k']}/{f['n']} follows from the records it lists")
        pc = s["census"]
        for t, p in pc["pooled"].items():
            check(p["k"] == sum(v["k"] for v in pc["per_cruise"][t].values()) and p["n"] == sum(v["n"] for v in pc["per_cruise"][t].values()),
                  f"{stem}: census {t} {p['k']}/{p['n']} is the sum of its cruise speeds")


def main():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import ros_model_fidelity_figure as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    stems = a.stem or sorted(P.STEMS)
    opts = sorted({P.STEMS[s][0] for s in stems})
    env = {k: v for k, v in os.environ.items() if k != "RMF_CACHE"}
    with tempfile.TemporaryDirectory() as td:
        # one run parses the traces once; a subset of stems re-runs only the options it needs
        for opt in (["all"] if set(opts) == {v[0] for v in P.STEMS.values()} else opts):
            r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "ros_model_fidelity_figure.py"),
                                "--option", opt, "--out-dir", td, "--dpi", "40"], capture_output=True, text=True, cwd=REPO, env=env)
            check(r.returncode == 0, f"the producer re-runs option {opt}" + ("" if r.returncode == 0 else f" (exit {r.returncode}: {r.stderr.strip()[-200:]})"))
        for stem in stems:
            print(f"\n=== {stem}")
            side = os.path.join(REF, stem + "_metrics.json"); got_p = os.path.join(td, stem + "_metrics.json")
            if not check(os.path.exists(side), "the committed sidecar exists") or not check(os.path.exists(got_p), "the re-run wrote it"):
                continue
            have, got = json.load(open(side)), json.load(open(got_p))
            check(all(os.path.exists(os.path.join(REF, stem + e)) for e in (".png", ".pdf")), "the .png and .pdf exist")
            compare(have, got, a.verbose)
            consistency(stem, have)
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
