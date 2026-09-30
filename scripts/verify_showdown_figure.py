#!/usr/bin/env python3
"""Every number on the showdown figures, checked against the artifact it must come from.

    scripts/verify_showdown_figure.py [--xpu-arm best45alt2] [--ros-tag 45_spin_r1]     # the v2 panel-I path
    scripts/verify_showdown_figure.py --metrics results/codesign_feedback/refined/<figure>_metrics.json
    scripts/verify_showdown_figure.py --all                                                 # every sidecar in refined/

Exit code = number of failed checks. Nothing here carries an expected value of its own: each check
re-derives a quantity from raw files and compares it with what a figure script recorded in its
sidecar. Sidecars written before the `script` / `inputs` / `fallbacks_used` contract are verified
where their keys allow; what cannot be re-derived is printed as INFO, never silently skipped.

  constants    scripts/measured_timing.py --verify reports zero drift
  fallbacks    the render recorded no silent numeric fallback
  inputs       every campaign CSV / roll-up the render read still has the hash it had at render time
  provenance   every Gantt row names a trace, its manifest, and (XPU-RT) the executed table whose
               sha256 the manifest recorded; ROS rows name the kernels they ran
  literals     no board-timing number (ms / Hz) is typed into a figure script's text outside the registry
  flights      every k/n on a flight panel equals the campaign CSVs under the panel's own rule
               (one flight per seed, equal replicates per seed across the arms drawn, timeouts censored
               and seeds paired in the breaking-point panel)
  board        every latency / frames-late / chain figure equals a re-read of the traces and the ROS roll-up
  refined/     every figure file carries a sidecar this script can verify, or is on the allowlist
"""
from __future__ import annotations
import argparse, ast, collections, csv, fnmatch, glob, hashlib, json, os, re, statistics, subprocess, sys, tempfile
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
RES = "results/codesign_feedback"
fails = 0
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
os.environ.setdefault("ENERGY_CSV", os.path.join(REPO, RES, "flight_energy_v2.csv"))

FIGURE_SCRIPTS = ["scripts/showdown_v3_figure.py", "scripts/showdown_atlas.py", "scripts/showdown_final_figure.py",
                  "scripts/showdown_paper10_figure.py", "scripts/story_figures.py"]
STORY_NAMES = ("crash_position", "course_progress", "rate_speed_map", "ros_ladder", "seed_pairs", "latency_waterfall")
# refined/ files that are figures without a sidecar this script verifies (named producers in the docs)
REFINED_ALLOWLIST = ["warehouse_showdown_story",   # the Tier A render (plots/fig_hil_showdown.pdf), from the composer sims/scripts/compose_warehouse_showdown.py
                     "env_crash_map_*", "env_sweep", "hil_envelope_story_v3*", "deployment_layers", "throughput_latency",
                     # a Gantt-only render draws measured schedule rows and nothing else, so it has no
                     # showdown sidecar to re-derive: its provenance is the measured_gantt_*_metrics.json
                     # of each row it drew, which names the board trace and carries its sha256
                     "gantt_*",
                     # cores_yolo_service and the paper's fig_cores_yolo come from fig_fair_v6.py, which
                     # lives in a second checkout (XPU-RT/results/codesign_feedback/refined_src/) and is
                     # tracked by neither repo. Its 30 points are literals: the two greedy rows are the
                     # median per-frame YOLO response of scheduled_m_greedy_shard_K{4..8}_{predicted,board}
                     # and re-derive exactly; the two CP-SAT rows are the tightest deadline the
                     # scheduled_{fine,cbp}_K*_D* searches accepted, not an achieved service; the two ROS
                     # rows have a source only at K=8 (scheduled_ros_pin_{predicted,board}). Allowlisted
                     # because there is no sidecar to re-derive, not because a producer is recorded.
                     # scripts/cores_yolo_service.py derives those points from the schedules and
                     # reproduces both published forms; verify_cores_yolo.py checks them.
                     "cores_yolo_service", "warehouse_gatecourse", "warehouse_showdown_envelope",
                     "warehouse_showdown_v2", "hil_feedback_*", "schedule_evolution_*"]
# Sidecars this script does not verify because another one does. Named here so that asking this
# verifier about them says where the check lives instead of "no verifier", and so the set of figures
# with no check at all is the complement of this table rather than something to be counted by hand.
VERIFIED_ELSEWHERE = {
    "schedule_evolution_short": "verify_schedule_evolution.py",
    "schedule_evolution_tall": "verify_schedule_evolution.py",
    "control_rate_response": "verify_control_rate_response.py",
    "control_rate_response_cp3": "verify_control_rate_response.py",
    "control_rate_response_v2": "verify_control_rate_response.py",
    "ros_effort_ladder": "verify_ros_effort_ladder.py --stem ros_effort_ladder",
    "ros_effort_ladder_v2": "verify_ros_effort_ladder.py",
    "hil_feedback_a90": "verify_hil_feedback.py",
    "hil_feedback_a90h": "verify_hil_feedback.py",
    "hil_feedback_a120h": "verify_hil_feedback.py",
    "hil_feedback_a120e": "verify_hil_feedback.py",
    "hil_feedback_b5": "verify_hil_feedback.py",
    "hil_feedback_close_120": "verify_hil_feedback.py",
    "audit_showdown_claims": "audit_showdown_claims.py (it recomputes its own output and exits non-zero)",
    "loop_overview": "verify_loop_overview.py",
    "cores_yolo_service_derived_published": "verify_cores_yolo.py",
    "cores_yolo_service_derived_published_a24p5": "verify_cores_yolo.py",
    "cores_yolo_service_derived_flat": "verify_cores_yolo.py",
    "cores_yolo_service_derived_flat_a22": "verify_cores_yolo.py",
}


# the six trace arms of the latency waterfall, by the keyword that names them in any sidecar label
WATERFALL_TRACES = [("cpsat", "xpurt_long/trace_acpsat_hardr{k}_other_run1.csv"), ("greedy", "xpurt_long/trace_agreedyr{k}_other_run1.csv"),
                    ("p3", "ros_traced/45_p3_r{k}/trace.csv"), ("vanilla4", "ros_traced/45_vanilla4_r{k}/trace.csv"),
                    ("vanilla4x2", "ros_traced/45_vanilla4x2_r{k}/trace.csv"), ("vanilla", "ros_traced/45_vanilla_r{k}/trace.csv")]


EMITTED: list = []          # (kind, msg) for every check/info, so --coverage can bucket them


def check(ok, msg):
    global fails
    EMITTED.append(("PASS" if ok else "FAIL", msg))
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails += 1


def info(msg):
    EMITTED.append(("INFO", msg))
    print("INFO " + msg)


# which panel a check's own message is about. The map is derived from what the checks say rather
# than maintained beside them, so a new check joins the coverage table by being written.
PANELS = [("A", r"\bpanel A\b|backdrop|clearance"),
          ("a-d", r"\bstrip|panels? a–d|panels? a-d|^moment "),
          ("B",  r"\bpanel B\b"),
          ("C",  r"\bpanel C\b"),
          ("D",  r"\bpanel D\b"),
          ("E-H", r"panels E|telemetry "),
          ("I",  r"\bpanel I\b|^gantt row "),
          ("whole figure", r"^the render used|^the \d+ input file|figure on disk|every ms/Hz")]


def coverage_rows():
    """One row per panel: how many checks spoke about it, and whether any did."""
    import re as _re
    rows = []
    for name, pat in PANELS:
        rx = _re.compile(pat)
        hits = [(k, m) for k, m in EMITTED if rx.search(m)]
        n_fail = sum(1 for k, _ in hits if k == "FAIL")
        n_info = sum(1 for k, _ in hits if k == "INFO")
        rows.append((name, len(hits) - n_info - n_fail, n_info, n_fail))
    seen = set()
    for name, pat in PANELS:
        import re as _re2
        seen |= {m for _, m in EMITTED if _re2.search(pat, m)}
    rows.append(("(unattributed)", sum(1 for k, m in EMITTED if m not in seen and k == "PASS"),
                 sum(1 for k, m in EMITTED if m not in seen and k == "INFO"),
                 sum(1 for k, m in EMITTED if m not in seen and k == "FAIL")))
    return rows


def print_coverage():
    print("\n=== panel coverage (checks that named each panel)")
    print(f"  {'panel':<16} {'checked':>7} {'asserted only':>14} {'failed':>7}")
    for name, npass, ninfo, nfail in coverage_rows():
        flag = "" if npass else "   <- nothing checks this panel"
        print(f"  {name:<16} {npass:>7} {ninfo:>14} {nfail:>7}{flag}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path, anchor):
    """a recorded path, absolute on the render host: as given only when it lies inside THIS repo, else re-rooted at
    REPO from `anchor` on. Another checkout of the same repo can sit on the same filesystem, so an existing absolute
    path is not evidence that it belongs to the tree being verified."""
    if not path:
        return path
    if os.path.exists(path) and not os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).startswith(os.pardir):
        return path
    i = path.find(anchor)
    return os.path.join(REPO, path[i:]) if i >= 0 else path


def _fc():
    try:
        import figure_constants as FC   # noqa: E402
        return FC
    except ImportError:
        return None


def _arm_key(label):
    """which waterfall trace arm a sidecar label names (labels differ between the atlas, the final figure and the story figure)."""
    l = label.lower()
    if "cp-sat" in l or "cpsat" in l:
        return "cpsat"
    if "greedy" in l:
        return "greedy"
    if "pinned" in l or "hand-tuned" in l or "p3" in l:
        return "p3"
    if "8 cores" in l or "two yolo" in l or "x2" in l or "all 8" in l:
        return "vanilla4x2"
    if "serial" in l:
        return "vanilla"
    if "4-hart" in l or "vanilla4" in l:
        return "vanilla4"
    if "vanilla" in l:
        return "vanilla4"
    return None


# ------------------------------------------------------------------------------------------ common checks
def _check_fallbacks(m):
    if "fallbacks_used" not in m:
        info("fallbacks_used not recorded (sidecar predates the contract)"); return
    fb = m["fallbacks_used"]
    check(not fb, f"the render used no silent numeric fallback ({len(fb)} recorded" + (": " + "; ".join(str(x.get("name", x)) for x in fb[:5]) if fb else "") + ")")


def _check_inputs(m):
    inputs = m.get("inputs")
    if not inputs:
        info("inputs (path -> sha256) not recorded (sidecar predates the contract)"); return
    bad = []
    for rel, sha in inputs.items():
        p = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: missing")
        elif _sha256(p) != sha:
            bad.append(f"{rel}: changed since the render")
    check(not bad, f"the {len(inputs)} input file(s) the render read are unchanged" + (" — " + "; ".join(bad[:6]) if bad else ""))


def _check_gantt_busy(arm, s, src, cpu, man):
    """Panel I's per-hart busy percentages, re-derived from the sampler the sidecar names.

    The panel labels a lane that looks empty with the percentage the sampler saw it busy at -- that is
    the whole claim a reader takes from an idle-looking hart, so the numbers themselves are checked,
    not only their recorded *source*. They come back from the tracked sampler CSV,
    over the window the renderer used, which is recoverable: the manifest carries the run's origin and
    the trace its span. Both files are already sha256-checked above, so re-deriving here makes this
    panel checked rather than asserted.

    The arithmetic is imported from the renderer rather than restated, so the two cannot drift.
    """
    rec = s.get("busy_pct") or {}
    if not s.get("busy_source"):
        info(f"gantt row {arm}: busy_source not recorded ({s.get('busy_note')})")
        return
    extra = ", harness kernel fraction recorded alongside" if s.get("kernel_frac_pct") else ""
    if not rec:
        info(f"gantt row {arm}: busy % from {s['busy_source']}{extra}; no per-hart values recorded")
        return
    if not (cpu and os.path.exists(cpu) and src and os.path.exists(src)):
        info(f"gantt row {arm}: busy % from {s['busy_source']}{extra}; "
             f"{'sampler' if not (cpu and os.path.exists(cpu)) else 'trace'} not on disk, "
             f"{len(rec)} value(s) not re-derived")
        return
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import make_measured_gantt_pair as G
        rows = G.read_trace(src)
        if s["busy_source"] == "sampler_rdtime":
            t0 = int(man["run_t0_rdtime"])
            ticks = (max(r["e"] for r in rows) - min(r["s"] for r in rows)) * G.HZ / 1000.0
            got = G.busy_from_cpu(cpu, t0, t0 + ticks, key="rdtime_ticks")
        elif s["busy_source"] == "sampler_epoch":
            t0 = man["wall_start_epoch_ms"]
            if "wall_end_epoch_ms" in man:
                got = G.busy_from_cpu(cpu, t0 + G.WARMUP_MS, man["wall_end_epoch_ms"])
            else:
                run_ms = max(r["e"] for r in rows) - min(r["s"] for r in rows)
                got = G.busy_from_cpu(cpu, t0, t0 + run_ms + 4000)
        else:
            info(f"gantt row {arm}: busy % from {s['busy_source']}{extra}; "
                 f"not a sampler source, {len(rec)} value(s) not re-derived")
            return
    except Exception as e:                      # a missing manifest field, not a wrong number
        info(f"gantt row {arm}: busy % from {s['busy_source']}{extra}; "
             f"could not re-derive ({type(e).__name__})")
        return
    off = {k: (rec[k], got.get(k)) for k in rec
           if got.get(k) is None or abs(float(rec[k]) - float(got[k])) > 0.15}
    check(not off, f"gantt row {arm}: all {len(rec)} per-hart busy % re-derive from "
                   f"{os.path.basename(cpu)} over the {s['busy_source']} window"
                   + ("" if not off else " — " + "; ".join(f"{k} drawn {a}, sampler {b}"
                                                           for k, (a, b) in list(off.items())[:4])))


def _check_gantt_sidecar(side):
    """one Gantt row's sidecar: its sources exist, recorded hashes match, the executed table is the one the manifest names."""
    side = _resolve(side, "schedules/")
    if not os.path.exists(side):
        check(False, f"gantt sidecar {side} exists"); return
    s = json.load(open(side)); arm = s.get("arm", os.path.basename(side))
    src = _resolve(s.get("source", ""), "results/"); cpu = _resolve(s.get("cpu_source", ""), "results/"); manp = _resolve(s.get("manifest", ""), "results/")
    check(bool(src) and os.path.exists(src), f"gantt row {arm}: trace {os.path.basename(src or '?')} exists (chain {s.get('chain_ms_median')} ms, {s.get('frames_late')}/{s.get('frames_checked')} late)")
    for key, path, what in (("source_sha256", src, "trace"), ("cpu_source_sha256", cpu, "sampler"), ("manifest_sha256", manp, "manifest")):
        if s.get(key):
            check(path and os.path.exists(path) and _sha256(path) == s[key], f"gantt row {arm}: {what} has the recorded sha256")
        else:
            info(f"gantt row {arm}: {what} sha256 not recorded (sidecar predates the contract)")
    man = json.load(open(manp)) if manp and os.path.exists(manp) else {}
    _check_gantt_busy(arm, s, src, cpu, man)
    if not man:
        check(False, f"gantt row {arm}: manifest {manp} readable"); return
    if s.get("kind") == "xpu":
        sched = _resolve(man.get("schedule", ""), "schedules/")
        check(bool(sched) and os.path.exists(sched), f"gantt row {arm}: executed table {os.path.basename(sched or '?')} on disk")
        rec = man.get("schedule_sha256") or man.get("schedule_sha256_16")
        if sched and os.path.exists(sched) and rec:
            got = _sha256(sched)
            if got[:len(rec)] == rec:
                check(True, f"gantt row {arm}: executed table's sha256 matches the manifest ({len(rec)} hex recorded)")
            else:   # the file's metadata may have been annotated after the run: the ledger holds the executed dispatch table's hash
                import executed_tables as ET
                ok, msg = ET.check(sched)
                check(bool(ok), f"gantt row {arm}: {msg}")
        elif not rec:
            info(f"gantt row {arm}: manifest records no schedule sha256")
        meta = json.load(open(sched)).get("metadata", {}) if sched and os.path.exists(sched) else {}
        solver = meta.get("solver") or meta.get("scheduler") or ""
        check(bool(solver), f"gantt row {arm}: executed table names its solver ({solver[:50]})")
        # both systems must run the same network, or the comparison is between two models
        iy = man.get("ir_sha256_yolo") or ""
        staged = os.path.join(REPO, "ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/.staged_from")
        if iy and os.path.exists(staged):
            sha = next((l.split("=", 1)[1].strip() for l in open(staged) if l.startswith("sha256=")), "")
            check(iy == sha, f"gantt row {arm}: runs the staged YOLO IR the baseline runs (ir={iy[:12]})")
        elif iy:
            info(f"gantt row {arm}: ir_sha256_yolo {iy[:12]} not re-derived (staged build tree absent)")
        else:
            info(f"gantt row {arm}: manifest records no YOLO IR hash")
    else:
        ks = man.get("kernels_sha") or ""
        staged = os.path.join(REPO, "ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/.staged_from")
        ir = re.search(r"ir=([0-9a-f]+)", ks)
        if ir and os.path.exists(staged):
            sha = next((l.split("=", 1)[1].strip() for l in open(staged) if l.startswith("sha256=")), "")
            check(sha.startswith(ir.group(1)), f"gantt row {arm}: kernels_sha ir={ir.group(1)} is the staged YOLO IR")
        else:
            info(f"gantt row {arm}: kernels_sha {ks!r} not re-derived (staged build tree absent)" if ks else f"gantt row {arm}: manifest records no kernels_sha")


def _check_provenance(m):
    gs = (m.get("sources") or {}).get("gantt_sidecars") or []
    if not gs and isinstance(m.get("D"), list):   # Figure 10 before the contract: D lists the schedule JSONs
        gs = [p.replace(".json", "_metrics.json") for p in m["D"]]
    if not gs:
        info("no gantt sidecars named"); return
    for side in gs:
        _check_gantt_sidecar(side)


def _string_literals(path):
    """(lineno, text) of every string literal in a script, docstrings excluded, f-string literal parts included."""
    tree = ast.parse(open(path).read(), path)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out.append((node.lineno, node.value))
    return out


LIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|Hz)\b")


def _check_literals(script_path, allowed=None):
    """every `<number> ms|Hz` inside the script's string literals must be a registry value; returns the findings."""
    hits = []
    for ln, s in _string_literals(script_path):
        for h in LIT_RE.findall(s):
            hits.append((ln, h, s.strip()[:70]))
    if allowed is None:
        FC = _fc()
        allowed = FC.allowed_display_literals() if FC else None
    if allowed is None:
        return hits
    return [(ln, h, s) for ln, h, s in hits if h not in allowed and h.replace(" ", "") not in {a.replace(" ", "") for a in allowed}]


def _literals_check(scripts=FIGURE_SCRIPTS):
    FC = _fc()
    for sp in scripts:
        p = os.path.join(REPO, sp)
        if not os.path.exists(p):
            info(f"{sp}: not present"); continue
        found = _check_literals(p, allowed=FC.allowed_display_literals() if FC else None)
        if FC is None:
            info(f"{sp}: registry (figure_constants) not present; {len(found)} ms/Hz literal(s) found, not judged"); continue
        check(not found, f"{sp}: every ms/Hz in drawn text is a registry value" + (f" — {len(found)} outside: " + "; ".join(f"L{ln} {h!r}" for ln, h, _ in found[:6]) if found else ""))


def _replicate_report():
    """duplicate cell keys per campaign CSV: what the fairness rule has to absorb (informational)."""
    rep = {}
    for f in sorted(glob.glob(os.path.join(REPO, RES, "campaign_*", "campaign.csv"))):
        camp = os.path.basename(os.path.dirname(f)); cnt = collections.Counter()
        for r in flight_rows(f):
            try:
                k = (os.path.basename(r.get("ctrl_trace") or ""), round(float(r["cruise_speed"]), 2), int(float(r["seed"])), round(float(r.get("percep_latency_ms") or 0), 1),
                     round(float(r.get("percep_hold_ms") or 0), 1), round(float(r["moment_scale"]), 5), round(float(r.get("prop_density") or 0.3), 2), r.get("course") or "a",
                     float(r.get("walk_speed") or 0), int(float(r.get("walk_cross") or 0)), round(float(r.get("person_h") or 2.4), 2))
            except (KeyError, ValueError):
                continue
            cnt[k] += 1
        extra = sum(v - 1 for v in cnt.values() if v > 1); rep[camp] = (len(cnt), extra)
        info(f"{camp}: {sum(cnt.values())} rows, {len(cnt)} distinct cells, {extra} replicate row(s) — absorbed by one-flight-per-seed / equal replicates")
    return rep


def _refined_allowlist():
    ref = os.path.join(REPO, RES, "refined"); missing = []
    for f in sorted(os.listdir(ref)):
        stem, ext = os.path.splitext(f)
        if ext not in (".png", ".pdf"):
            continue
        if os.path.exists(os.path.join(ref, stem + "_metrics.json")) or any(fnmatch.fnmatch(stem, pat) for pat in REFINED_ALLOWLIST):
            continue
        missing.append(f)
    check(not missing, f"refined/: every figure carries a sidecar or a named producer ({len(missing)} unattributed" + (": " + ", ".join(sorted({os.path.splitext(x)[0] for x in missing})[:12]) if missing else "") + ")")


# ------------------------------------------------------------------------------------------ recomputations
def _mods():
    import showdown_v3_figure as V  # noqa: E402
    import showdown_atlas as AT  # noqa: E402
    import story_figures as SF  # noqa: E402
    return V, AT, SF


def _recompute_break(rows, V, dense=False):
    """the breaking-point panel's rule: the 1.7 m scene, timeouts censored, seeds paired, equal replicates."""
    try:
        import showdown_final_figure as F  # noqa: E402
    except Exception:
        F = None
    X_, R_ = "xpu_a_cpsat_hard.csv", "ros_vanilla445.csv"
    cells = V.flight_cells(rows, "a", 0.30, 0.0055, person_h=1.7, camps={"campaign_break"})
    if F is not None and hasattr(F, "censor_and_pair"):
        cells, n_air = F.censor_and_pair(cells, X_, R_)
    else:
        n_air = sum(r["outcome"] == "timeout" for v in cells.values() for r in v)
        cells = {k: [r for r in v if r["outcome"] != "timeout"] for k, v in cells.items()}
        for c in {k[1] for k in cells}:
            if (X_, c) in cells and (R_, c) in cells:
                common = {r["seed"] for r in cells[(X_, c)]} & {r["seed"] for r in cells[(R_, c)]}
                for t in (X_, R_):
                    cells[(t, c)] = [r for r in cells[(t, c)] if r["seed"] in common]
    arms = [t for t in (X_, R_, "xpu_a_greedy.csv", "ros_p345.csv") if any(k[0] == t for k in cells)]
    cells, _ = V.equalise(cells, arms) if arms else (cells, 0)
    out = {}
    for tr in arms:
        lab = V.FLIGHT_ARMS[tr][0]
        out[lab] = {f"c{c}": [sum(x["outcome"] == "success" for x in v), len(v)] for (t, c), v in sorted(cells.items()) if t == tr and len(v) >= 6}
    if dense:
        dn = V.flight_cells(rows, "a", 0.40, 0.0055, person_h=1.7, camps={"campaign_break"}); dn = {k: [r for r in v if r["outcome"] != "timeout"] for k, v in dn.items()}
        if all(any(k[0] == t for k in dn) for t in (X_, R_)):
            dn, _ = V.equalise(dn, [X_, R_])
            for tr in (X_, R_):
                out[V.FLIGHT_ARMS[tr][0] + ", density 0.40"] = {f"c{c}": [sum(x["outcome"] == "success" for x in v), len(v)] for (t, c), v in sorted(dn.items()) if t == tr and len(v) >= 6}
    return out, n_air


def _compare_break(rec, got, n_air, panel):
    ok = True; n = 0
    for lab, cells in rec.items():
        if not isinstance(cells, dict) or lab in ("zones",):
            continue
        for ck, kn in cells.items():
            if not re.fullmatch(r"c[\d.]+", ck):
                continue
            n += 1
            if got.get(lab, {}).get(ck) != list(kn):
                ok = False; print(f"     {panel} {lab} {ck}: sidecar {kn}, re-derived {got.get(lab, {}).get(ck)}")
    check(ok and n > 0, f"{panel}: every k/n of the breaking-point curves equals the campaign CSVs under the panel's rule ({n} cells; timeouts censored, seeds paired, equal replicates)")
    if "airborne_at_horizon_left_out" in rec:
        check(rec["airborne_at_horizon_left_out"] == n_air, f"{panel}: flights left out as airborne at the horizon ({n_air}) equal the render's ({rec['airborne_at_horizon_left_out']})")


def _recompute_pairs2(rows, SF):
    def pairs(sub, ph):
        A = SF.cells_by(sub, "xpu_a_cpsat_hard.csv", 56.8, 0.0, person_h=ph); B = SF.cells_by(sub, "ros_vanilla445.csv", 242.0, 0.0, person_h=ph)
        tot = collections.Counter()
        for c in sorted(set(A) & set(B)):
            a = {r["seed"]: r for r in A[c]}; b = {r["seed"]: r for r in B[c]}
            for sd in set(a) & set(b):
                ga = 4 if a[sd]["outcome"] == "success" else a[sd]["gates"]; gb = 4 if b[sd]["outcome"] == "success" else b[sd]["gates"]
                tot["xpu" if ga > gb else ("ros" if gb > ga else "tie")] += 1
        return dict(tot)
    t17 = pairs([r for r in rows if r["camp"] == "campaign_break" and r["dens"] == 0.30], 1.7)
    t24 = pairs([r for r in rows if r.get("ph", 2.4) == 2.4 and r["dens"] == 0.30 and r["course"] == "a"], 2.4)
    return t17, t24


def _compare_pairs2(rec, rows, SF, panel):
    t17, t24 = _recompute_pairs2(rows, SF)
    for key, got in (("scene_1.7", t17), ("scene_2.4", t24)):
        r = {k: v for k, v in (rec.get(key) or {}).items() if v}
        check(r == {k: v for k, v in got.items() if v}, f"{panel} {key}: paired seeds {got} equal the render's {rec.get(key)}")


def _waterfall_chain_medians(SF):
    from make_measured_gantt_pair import read_trace  # noqa: E402
    out = {}
    for key, pat in WATERFALL_TRACES:
        w = []
        for k in (1, 2, 3):
            p = os.path.join(REPO, RES, pat.format(k=k))
            if os.path.exists(p):
                w += SF.waterfall(read_trace(p))
        if not w:
            continue
        if hasattr(SF, "waterfall_summary"):
            s = SF.waterfall_summary(w); out[key] = (s["chain_median_ms"], s["sum_of_part_medians_ms"], s["frames"])
        else:
            med = [statistics.median([x[j] for x in w]) for j in range(4)]
            out[key] = (statistics.median([sum(x) for x in w]), sum(med), len(w))
    return out


def _compare_waterfall(rec, SF, panel):
    got = _waterfall_chain_medians(SF); ok = True; n = 0
    for lab, v in rec.items():
        key = _arm_key(lab)
        if key not in got:
            info(f"{panel} {lab}: traces not on disk, not re-derived"); continue
        n += 1
        if "chain_median_ms" in v:
            if abs(v["chain_median_ms"] - got[key][0]) > 0.05:
                ok = False; print(f"     {panel} {lab}: chain median {v['chain_median_ms']} vs re-derived {got[key][0]:.1f}")
        elif "total_ms" in v:
            if abs(v["total_ms"] - got[key][1]) > 0.05:
                ok = False; print(f"     {panel} {lab}: total_ms {v['total_ms']} is neither the sum of component medians ({got[key][1]:.1f}) nor the chain median ({got[key][0]:.1f})")
            info(f"{panel} {lab}: sidecar predates the contract — total_ms is the sum of component medians ({got[key][1]:.1f}); the chain's median is {got[key][0]:.1f} ms")
        if v.get("frames") is not None and v["frames"] != got[key][2]:
            ok = False; print(f"     {panel} {lab}: frames {v['frames']} vs {got[key][2]}")
    check(ok and n > 0, f"{panel}: every arm's latency total equals a re-read of its traces ({n} arms)")


SCENE_ARMS = (("XPU-RT · CP-SAT", "xpu_cpsat", "xpu_a_cpsat_hard.csv"), ("ROS 2 vanilla", "ros_vanilla", "ros_vanilla445.csv"),
              ("XPU-RT · greedy", "xpu_greedy", "xpu_a_greedy.csv"))


def _scene_counts(recdir):
    """completed / flown per arm in one recorded scene, from its per-episode records when they are on disk and from the
    scene's own campaign table otherwise: the records are regenerable and live in archive_v3, the table is tracked."""
    import numpy as np  # noqa: E402
    recdir = _resolve(recdir, "results/")
    if not recdir or not os.path.isdir(recdir):
        return None
    out = {}
    for lab, sub, _ in SCENE_ARMS:
        fl = sorted(glob.glob(os.path.join(recdir, sub, "ep*.npz")))
        if fl:
            out[lab] = {"completed": sum(str(np.load(f, allow_pickle=True)["outcome"]) == "success" for f in fl), "n": len(fl)}
    if out:
        return out
    camp = os.path.join(recdir, "campaign.csv")
    if not os.path.exists(camp):
        return None
    tally = {}
    for r in flight_rows(camp):
        t = os.path.basename(r.get("ctrl_trace", "") or ""); k, n = tally.get(t, (0, 0))
        tally[t] = (k + (r.get("outcome") == "success"), n + 1)
    for lab, _, trace in SCENE_ARMS:
        if trace in tally:
            out[lab] = {"completed": tally[trace][0], "n": tally[trace][1]}
    return out or None


def _compare_scene(rec, recdir, panel):
    got = _scene_counts(recdir)
    if got is None:
        info(f"{panel}: scene records {recdir} absent (regenerable), not re-derived"); return
    check({k: v for k, v in rec.items() if k in got} == got and set(rec) == set(got), f"{panel}: every recorded flight of the scene counted ({got})")


def _compare_forest(rec, rows_all, AT, panel):
    if not hasattr(AT, "forest_rows"):
        info(f"{panel}: forest recomputation needs showdown_atlas.forest_rows (not present) — per-row k/n not re-derived"); return
    res, fbc = AT.forest_rows(rows_all, AT.CONDS); ok = True; n = 0
    for lab, rlab, kx, nx, kr, nr, d, lo, hi, speeds in res:
        r = rec.get(lab)
        if r is None:
            ok = False; print(f"     {panel}: condition {lab!r} drawn now but not in the sidecar"); continue
        n += 1
        if r.get("xpu") != [kx, nx] or r.get("ros") != [kr, nr]:
            ok = False; print(f"     {panel} {lab}: sidecar {r.get('xpu')} vs {r.get('ros')}, re-derived {[kx, nx]} vs {[kr, nr]}")
    check(ok and n > 0, f"{panel}: every condition's k/n equals the campaign CSVs ({n} rows, one flight per seed)")
    summ = rec.get("all conditions (summary)")
    if summ is None:
        if "all conditions pooled" in rec:
            info(f"{panel}: sidecar predates the summary contract — its pooled row sums flights across conditions ({rec['all conditions pooled'].get('xpu')} vs {rec['all conditions pooled'].get('ros')}); distinct flights are fewer")
        return
    got = AT.forest_summary(res, fbc)
    check(abs(got["mean_delta"] - summ["mean_delta"]) < 1e-6 and all(abs(a - b) < 1e-6 for a, b in zip(got["ci"], summ["ci"])),
          f"{panel}: summary mean Δ {got['mean_delta']:+.3f} [{got['ci'][0]:+.3f}, {got['ci'][1]:+.3f}] re-derives (bootstrap over conditions, seeded)")
    check(got["distinct_flights"] == summ.get("distinct_flights"), f"{panel}: distinct flights {got['distinct_flights']} equal the render's")


def _ros_rollup():
    roll = {}
    for r in csv.DictReader(open(f"{RES}/ros_traced/summary.csv")):
        mm = re.match(r"(\d+)_(.+)_r(\d+)$", r["tag"])
        if mm and r.get("e2e_goal_med_ms"):
            roll.setdefault((mm.group(2), int(mm.group(1))), []).append(float(r["e2e_goal_med_ms"]))
    return roll


def _compare_camrate(F, panel):
    """the camera-rate points: CP-SAT on time everywhere, greedy late above 30 Hz, ROS latencies equal the roll-up."""
    cp = F.get("XPURTCPSAT", {}); gr = F.get("XPURTgreedy", {})
    check(cp and all(v["late"] == 0 for v in cp.values()), f"{panel}: CP-SAT every frame on time at {sorted(int(float(v['cam'])) for v in cp.values())} Hz")
    check(gr and all(v["late"] > 0 for h, v in gr.items() if v["cam"] > 31), f"{panel}: greedy loses frames at every rate above 30 Hz")
    check(cp and all(v["lat"] < 65 for v in cp.values()), f"{panel}: CP-SAT camera→control under 65 ms at every rate" + (f" (max {max(v['lat'] for v in cp.values()):.1f})" if cp else ""))
    roll = _ros_rollup(); ok = True; n = 0
    for lab, key in (("ROSvanillahartYOLO", "vanilla4"), ("ROShandtunedpinned", "p3"), ("ROSonallcores", "vanilla4x2"), ("ROSvanillaserialYOLO", "vanilla")):
        for hk, v in F.get(lab, {}).items():
            have = roll.get((key, int(float(v["cam"])))); n += 1
            ok &= bool(have) and abs(statistics.median(have) - v["lat"]) < 0.5
    check(ok and n > 0, f"{panel}: ROS 2 latencies equal the per-run roll-up (e2e_goal_med_ms, replicates pooled; {n} points)")


def _seed_note(es):
    """how the panel words the pair's episode seeds: one shared draw, or one per arm"""
    x, r = es.get("xpu"), es.get("ros")
    return f"both {x}" if x == r else f"XPU-RT {x}, ROS 2 {r}"


def _baseline_fails_before_the_course(A):
    """Whether this figure's baseline is drawn failing short of the first gate rather than after one.

    Every form but one draws a baseline that reaches the course and crashes inside it, and that is what
    this returns False for, including forms whose two arms fly different gains. The exception declares
    itself: a render whose baseline never reaches the first gate writes ros_failure = "before_first_gate"
    into its sidecar, and only then is a gateless flight the outcome to require rather than a failure.
    The 30 Hz per-rate-gain form is that case -- it hands the 30 Hz arm 1.7x the gain its controller was
    tuned at, and across 96 census flights the baseline never reached the first gate.
    """
    return A.get("ros_failure") == "before_first_gate"


def _compare_display_pair(m, panel):
    import numpy as np  # noqa: E402
    src = m.get("sources") or {}; A = m.get("A", {})
    per_rate = _baseline_fails_before_the_course(A)
    xd, rd = _resolve(src.get("xpu_dir", ""), "results/"), _resolve(src.get("ros_dir", ""), "results/")
    if xd and rd and os.path.exists(os.path.join(xd, "figure_data.npz")) and os.path.exists(os.path.join(rd, "figure_data.npz")):
        zx = np.load(os.path.join(xd, "figure_data.npz"), allow_pickle=True); zr = np.load(os.path.join(rd, "figure_data.npz"), allow_pickle=True)
        # Two different things were being asserted together here. That the panel's numbers match the
        # flight behind it is correctness, and always checked. That the pair reads the way a display
        # figure should -- XPU-RT completing, the baseline crashing mid-course rather than before the
        # first gate -- is a convention about which flights make a good figure, and several renders
        # break it deliberately (an `_xpu_3of4` variant exists to show a non-completing XPU-RT
        # flight). Asserting the convention marked those renders as disagreeing with their own data
        # when they agree with it exactly, so it is reported and not failed.
        check(int(zx["gates_passed"]) == A.get("xpu_gates") and str(zx["outcome"]) == str(A.get("xpu_outcome", zx["outcome"])),
              f"{panel}: the XPU-RT flight drawn is the one the panel states "
              f"({int(zx['gates_passed'])} gates, {zx['outcome']})")
        check(int(zr["gates_passed"]) == A.get("ros_gates"),
              f"{panel}: the ROS 2 flight drawn is the one the panel states "
              f"({int(zr['gates_passed'])} gates, {zr['outcome']}, hits {A.get('hit')})")
        if not (str(zx["outcome"]) == "success" and int(zx["gates_passed"]) >= 4):
            info(f"{panel}: the XPU-RT flight drawn does not complete the course "
                 f"({int(zx['gates_passed'])} gates) -- a deliberate variant, not the display convention")
        if per_rate:
            check(str(zr["outcome"]) in ("crash", "timeout") and int(zr["gates_passed"]) == 0,
                  f"{panel}: displayed ROS 2 flight never reaches the first gate ({zr['outcome']} at {int(zr['steps']) if 'steps' in zr else '?'} steps)")
        elif not (str(zr["outcome"]) == "crash" and 1 <= int(zr["gates_passed"]) <= 2):
            info(f"{panel}: the ROS 2 flight drawn is {zr['outcome']} at {int(zr['gates_passed'])} gate(s), "
                 f"outside the mid-course crash the display convention prefers")
        # "same scene" is a claim about the layout seed; the episode seed is a separate draw and the
        # panel states it per arm, so both are re-derived from the dumps rather than trusted
        ls, es = A.get("layout_seed") or {}, A.get("episode_seed") or {}
        if ls:
            check(int(zx["layout_seed"]) == ls.get("xpu") and int(zr["layout_seed"]) == ls.get("ros")
                  and int(zx["layout_seed"]) == int(zr["layout_seed"]),
                  f"{panel}: both flights were drawn from the same scene (layout seed {int(zx['layout_seed'])})")
        if es:
            check(int(zx["seed"]) == es.get("xpu") and int(zr["seed"]) == es.get("ros"),
                  f"{panel}: each flight's episode seed is the one the panel states "
                  f"({_seed_note(es)})")
        # the wording the panel uses for the baseline's end -- "hits a crate", "runs out of time" --
        # is chosen from this field at render time, so it has to be the outcome the dump recorded
        if A.get("ros_outcome"):
            check(str(zr["outcome"]) == A["ros_outcome"],
                  f"{panel}: the baseline's drawn ending is its recorded one ({A['ros_outcome']})")
        gx, gr = float(zx["moment_scale"]), float(zr["moment_scale"])
        dx, dr = A.get("gain_xpu"), A.get("gain_ros")
        if dx is not None and dr is not None:   # the figure declares the gain policy it flew; the dumps must be that policy
            check(abs(gx - dx) < 1e-9 and abs(gr - dr) < 1e-9,
                  f"{panel}: both flights carry the gain the figure declares ({'same gain ' + format(gx, '.5f') if abs(gx - gr) < 1e-9 else f'0.5 / control rate per arm, {gx:.5f} vs {gr:.5f}'})")
        else:
            check(abs(gx - gr) < 1e-9, f"{panel}: both displayed flights at the same gain ({gx})")
    else:
        info(f"{panel}: display dumps not on disk (archive_v3), A re-derivation skipped; sidecar says XPU-RT {A.get('xpu_gates')} gates, ROS 2 {A.get('ros_gates')} gate(s) then {A.get('hit')}")
        if per_rate:
            check(A.get("xpu_gates", 0) >= 4 and A.get("ros_gates", -1) == 0, f"{panel}: sidecar's pair is the story's (XPU-RT completes, ROS 2 never reaches the first gate)")
        else:
            check(A.get("xpu_gates", 0) >= 4 and 1 <= A.get("ros_gates", 0) <= 2, f"{panel}: sidecar's pair is the story's (XPU-RT completes, ROS 2 clears 1–2 gates then crashes)")


# ------------------------------------------------------------------------------------------ verifiers
def verify_v3(path):
    """The third-form composite: every number in its sidecar re-derived from the artifact it names.
    The sidecar is the render's own record; the checks below recompute B/C/H from the campaign CSVs
    with the same cell key, F from the ROS roll-up and the XPU-RT traces, G/J for internal consistency,
    and the display pair from its dumps."""
    import numpy as np
    V, AT, SF = _mods()
    m = json.load(open(path)); fig = m["figure"]
    check(os.path.exists(fig) and os.path.exists(fig.replace(".png", ".pdf")), f"v3 figure on disk: {fig}")
    _check_fallbacks(m); _check_inputs(m)
    # A: the display pair and its backdrop
    A = m["A"]; cell = m["variant"]["cell"]
    check(A["plate_std"] > 40, f"panel A backdrop rendered ({A['backdrop']}, contrast {A['plate_std']})")
    check(A["xpu_gates"] >= 4, f"displayed XPU-RT flight completes the course ({A['xpu_gates']} gates)")
    want = (2,) if cell == "tall1008" else (1, 2)          # the tall 1.2 m/s pair is searched for exactly two gates; the others accept one or two
    check(A["ros_outcome"] == "crash" and A["ros_gates"] in want, f"displayed ROS 2 flight: {A['ros_outcome']} after {A['ros_gates']} gate(s), hits {A['ros_hit']}")
    for k in ("xpu_dir", "ros_dir"):
        f = os.path.join(_resolve(m["sources"][k], "results/"), "figure_data.npz")
        if not os.path.exists(f):
            info(f"{k}: display dump not on disk (archive_v3); gain check skipped"); continue
        z = np.load(f, allow_pickle=True)
        want_g = {"xpu_dir": 0.0052, "ros_dir": 0.01277}[k] if cell == "cal17" else m["variant"]["gain"]
        check(abs(float(z["moment_scale"]) - want_g) < 1e-6, f"{k}: gain {float(z['moment_scale'])} is the cell's ({want_g})")
    # B / C: counts equal the campaign CSVs, same key
    rows = V.load_campaigns()
    if cell == "cal17":
        rows += V.load_extra_csv(os.path.join(RES, "campaign_v2", "campaign_v2.csv"), "campaign_v2")
        cells = V.flight_cells(rows, "a", 0.30, latency=False, gain_by_trace={"xpu_a_cpsat_hard.csv": 0.0052, "ros_vanilla445.csv": 0.01277}, camps={"campaign_v2"})
    else:
        cells = V.flight_cells(rows, "a", 0.30, m["variant"]["gain"])
    arms_drawn = [next((t for t, (lab, _) in V.FLIGHT_ARMS.items() if re.sub(r"[^A-Za-z]", "", lab) == arm), None) for arm in m["B"]]
    cells, set_aside = V.equalise(cells, [t for t in arms_drawn if t])          # the panel's rule: same seeds, same replicates per arm
    check(set_aside == m["variant"].get("flights_set_aside"), f"panel B: the equal-replicates rule sets aside the same flights as the render ({set_aside})")
    ok = True
    for arm, vals in m["B"].items():
        tr = next((t for t, (lab, _) in V.FLIGHT_ARMS.items() if re.sub(r"[^A-Za-z]", "", lab) == arm), None)
        for ck, (k, n) in vals.items():
            c = next(c for (t, c) in cells if t == tr and f"Cruise{V.word(c)}" == ck)
            v = cells[(tr, c)]; ok &= (sum(x["outcome"] == "success" for x in v) == k and len(v) == n)
    check(ok, f"panel B: every k/n equals the campaign CSVs under that rule ({sum(len(v) for v in m['B'].values())} cells)")
    pair = [m["B"].get("XPURTCPSAT", {}), m["B"].get("ROSvanillahartYOLO", {})]
    same = all(pair[0][c][1] == pair[1][c][1] for c in pair[0] if c in pair[1])
    check(same and pair[0] and pair[1], "panel B: the display pair carries the same n at every cruise (" + ", ".join(f"{c[6:]}: {pair[0][c][1]}" for c in pair[0] if c in pair[1]) + ")")
    check(all(abs(sum(h) - 1.0) < 1e-6 for h in m["C"].values()), "panel C: each arm's gate histogram sums to one")
    # D′: the environment map pools 2.4 m people only
    Dp = m.get("Dprime") or {}
    if Dp:
        if hasattr(V, "dprime_counts"):
            envs, K, N = V.dprime_counts(rows, tuple(Dp["families"]), (1.0, 1.2, 1.4), m["variant"]["gain"], person_h=Dp.get("person_h", 2.4))
            got_envs = [f"{c}_d{d:.2f}{'_cross' if x else ''}" for c, d, x in envs]
            check(got_envs == Dp["envs"] and [list(map(int, r)) for r in K] == Dp["k"] and [list(map(int, r)) for r in N] == Dp["n"], f"panel D′: environments and k/n equal the campaign CSVs at people {Dp.get('person_h', 2.4)} m ({len(got_envs)} columns)")
        else:
            info("panel D′: not re-derived (showdown_v3_figure.dprime_counts not present); sidecar predates the person_h filter")
    # F: XPU-RT on time / late verdicts from the traces; ROS latencies from the roll-up
    _compare_camrate(m["F"], "panel F")
    # G: shares
    G = m["G"]
    check(all(abs(sum(v.values()) - 100.0) < 0.1 for v in G.values()), "panel G: each arm's hart shares sum to 100 %")
    cps = next(v for k, v in G.items() if k.startswith("XPURTCPSAT")); van = next(v for k, v in G.items() if "vanilla" in k and "hart" in k)
    check(sum(x >= 3.0 for x in cps.values()) == 8, "panel G: CP-SAT places work on all eight harts")
    check(sum(x >= 3.0 for x in van.values()) <= 7, f"panel G: the 4-hart-pool baseline leaves harts idle ({sum(x >= 3.0 for x in van.values())} of 8 busy)")
    # H: the schedule beats every ROS 2 layout under the added load, flights re-counted
    H = m["H"]; xc = H.get("XPURTCPSAT", {})
    check(xc and all(xc["lat"] < v["lat"] for k, v in H.items() if k.startswith("ROS")), f"panel H: CP-SAT camera→control {xc.get('lat')} ms under every ROS 2 layout's")
    rich = {k: v for k, v in V.flight_cells(rows, "a", 0.30, 0.0055).items() if k[0].startswith(("xpu_b5", "ros_rvanilla"))}
    rich, _ = V.equalise(rich, [t for t in ("xpu_b5_cpsat.csv", "xpu_b5_greedy.csv", "ros_rvanilla445.csv", "ros_rvanilla490.csv") if any(k[0] == t for k in rich)])
    n_cp = sum(len(v) for (t, c), v in rich.items() if t == "xpu_b5_cpsat.csv"); k_cp = sum(sum(x["outcome"] == "success" for x in v) for (t, c), v in rich.items() if t == "xpu_b5_cpsat.csv")
    check(xc.get("flightsK") == k_cp and xc.get("flightsN") == n_cp, f"panel H: CP-SAT heavier-stack flights {k_cp}/{n_cp} equal the campaign CSV")
    # J: the loop closes
    J = m["J"]
    if J:
        iso = J["isolatedprofile"]; cal = J["boardcalibratedperdispatchcost"]
        check(iso["final_drift_ms"] > cal["final_drift_ms"] + 50, f"panel J: isolated table drifts {iso['final_drift_ms']} ms, board-calibrated {cal['final_drift_ms']} ms")
        ref = f"{RES}/refined/hil_feedback_close_120_metrics.json"
        if os.path.exists(ref):
            rr = json.load(open(ref))
            check(abs(rr["isolated"]["span_ms"] - iso["span_ms"]) < 1 and abs(rr["calibrated"]["span_ms"] - cal["span_ms"]) < 1, "panel J: spans equal the close-up figure's metrics")
    # I: the Gantt rows and their sources; the 45 Hz solver rows agree with the committed pair
    for side in m["sources"]["gantt_sidecars"]:
        side = _resolve(side, "schedules/")
        s = json.load(open(side)); src = _resolve(s["source"], "results/")
        check(os.path.exists(src), f"panel I row {s['arm']}: trace {os.path.basename(src)} exists, chain {s['chain_ms_median']} ms")
        if s["kind"] == "xpu":
            check(bool(s.get("solver_name")), f"panel I row {s['arm']}: executed table names its solver ({s.get('solver_name')})")
            ref = f"schedules/measured_gantt_{'xpu' if s['arm'] == 'xpu' else 'xpu2'}_metrics.json"
            if os.path.exists(ref):
                rr = json.load(open(ref)); check(abs(rr["chain_ms_median"] - s["chain_ms_median"]) < 0.05 and os.path.basename(rr["source"]) == os.path.basename(s["source"]), f"panel I row {s['arm']}: equals the committed 45 Hz pair")
    _check_provenance(m)
    # K: same-scene records
    for lab, v in m.get("K", {}).items():
        if v["n"]:
            recs = glob.glob(os.path.join(_resolve(m["sources"]["scene_records"], "results/"), "*", "ep*.npz"))
            if recs:
                seeds = {int(np.load(f, allow_pickle=True)["layout_seed"]) for f in recs}
                check(len(seeds) == 1, f"panel K: every recorded run shares one layout seed ({seeds})")
            else:
                info("panel K: scene records absent (regenerable), layout seed not re-derived")
            break
    _literals_check(["scripts/showdown_v3_figure.py"])
    return fails


def verify_final(m):
    """The final composite (showdown_final_figure.py): the flight panels re-counted under their rules, the board panels
    re-read from the traces, the Gantt rows' provenance."""
    V, AT, SF = _mods()
    fig = m["figure"]; check(os.path.exists(fig), f"final figure on disk: {fig}")
    _check_fallbacks(m); _check_inputs(m)
    rows_all = AT.all_flights(); rows = [r for r in rows_all if r["camp"].startswith("campaign_")]
    cells = V.flight_cells(rows, "a", 0.30, 0.0055)
    b_arms = [t for t in ("xpu_a_cpsat_hard.csv", "xpu_a_greedy.csv", "ros_vanilla445.csv", "ros_rvanilla445.csv", "ros_vanilla4x245.csv") if any(k[0] == t for k in cells)]
    _, n_aside = V.equalise(cells, b_arms)
    check(m.get("flights_set_aside") == n_aside, f"panel I′: the equal-replicates rule sets aside the same flights as the render ({n_aside} vs {m.get('flights_set_aside')})")
    _compare_display_pair(m, "panel A")
    got, n_air = _recompute_break(rows, V, dense=True); _compare_break(m.get("I_break", {}), got, n_air, "panel I")
    _compare_pairs2(m.get("J", {}), rows, SF, "panel J")
    _compare_forest(m.get("M", {}), rows_all, AT, "panel M")
    _compare_camrate(m.get("N", {}), "panel N")
    _compare_waterfall(m.get("O", {}), SF, "panel O")
    src = m.get("sources") or {}
    _compare_scene(m.get("H", {}), src.get("scene_records", ""), "panel H"); _compare_scene(m.get("H2", {}), src.get("scene2_records", ""), "panel H′")
    Q = m.get("Q", {})
    if Q:
        check(all(abs(sum(v.values()) - 100.0) < 0.1 for v in Q.values()), "panel Q: each arm's hart shares sum to 100 %")
    _check_provenance(m)
    _literals_check(["scripts/showdown_final_figure.py"])
    return fails


def verify_paper10(m):
    """Figure 10 (showdown_paper10_figure.py): the same rules as the final figure on the panels it keeps, plus its layout audit."""
    V, AT, SF = _mods()
    fig = m["figure"]; check(os.path.exists(fig) and os.path.exists(fig.replace(".pdf", ".png")), f"Figure 10 on disk: {fig}")
    _check_fallbacks(m); _check_inputs(m)
    rows_all = AT.all_flights(); rows = [r for r in rows_all if r["camp"].startswith("campaign_")]
    if not (m.get("sources") or {}).get("xpu_dir"):
        info("panel A: sources not recorded (sidecar predates the contract); the default display pair (tall s1005) is assumed")
        m = dict(m); m["sources"] = dict(m.get("sources") or {}, xpu_dir=os.path.join(RES, "campaign_v2/display_same/xpu_s1005_figdata"), ros_dir=os.path.join(RES, "campaign_v2/display_same/ros_s1005_figdata"))
    _compare_display_pair(m, "panel A")
    got, n_air = _recompute_break(rows, V, dense=False); _compare_break(m.get("B", {}), got, n_air, "panel B")
    _compare_camrate(m.get("C", {}), "panel C")
    S = m.get("S", {})
    if S:
        _compare_scene(S.get("counts", {}), S.get("records", ""), "panel S")
    _compare_pairs2(m.get("F", {}), rows, SF, "panel F")
    _compare_waterfall(m.get("E", {}), SF, "panel E")
    aud = m.get("audit")
    check(aud == [] or aud is None, f"layout audit recorded no text off the page, under 5 pt or colliding ({len(aud or [])} findings)")
    _check_provenance(m)
    _literals_check(["scripts/showdown_paper10_figure.py"])
    return fails


def verify_paper(m):
    """The paper's figure in its paper layout (showdown_paper_figure.py): the display pair, the rate-injected envelope
    and the unseen-course envelope re-counted from their CSVs, the two measured Gantt rows, the scene's own counts."""
    import csv as _csv
    fig = m["figure"]; check(os.path.exists(fig) and os.path.exists(fig.replace(".png", ".pdf")), f"paper figure on disk: {fig}")
    _check_fallbacks(m); _check_inputs(m)
    _compare_display_pair(m, "panel A")
    src = m.get("sources") or {}
    def counts(path):
        cell = {}
        for r in flight_rows(path):
            hz = str(round(float(r["eff_cmd_hz"]))); k, n = cell.get(hz, (0, 0)); cell[hz] = (k + (r["outcome"] == "success"), n + 1)
        return {h: list(v) for h, v in cell.items()}
    for key, panel, csvk in (("B", "panel B", "envelope_csv"), ("C", "panel C", "courseB_csv")):
        rec = (m.get(key) or {}).get("k_n_per_hz" if key == "B" else "courseB_k_n_per_hz") or {}
        path = _resolve(src.get(csvk, ""), "results/")
        if rec and path and os.path.exists(path):
            got = counts(path); check(got == rec, f"{panel}: k/n per control rate equal {os.path.basename(path)} ({', '.join(f'{h} Hz {v[0]}/{v[1]}' for h, v in sorted(got.items(), key=lambda t: int(t[0])))})")
            # The axis the k/n are plotted against, and the population they are drawn from, are
            # compared too, not only recorded (docs/Artifact/mutation_audit.md). The rates the panel draws
            # have to be the rates the census actually holds, or the counts sit at the wrong x.
            blk = m.get(key) or {}
            if blk.get("rates"):
                have = sorted(round(float(r), 2) for r in blk["rates"])
                want_r = sorted({round(float(r["eff_cmd_hz"]), 2) for r in flight_rows(path)})
                # the drawn axis is the census's distinct rates, each rounded to the label it carries
                near = len(have) == len(want_r) and all(
                    min(abs(h - w) for w in want_r) < 0.6 for h in have)
                check(near, f"{panel}: the rate axis is the census's own rates "
                            f"({', '.join(f'{h:g}' for h in have)} Hz)")
            if blk.get("flights") is not None:
                n_rows = sum(v[1] for v in got.values())
                check(int(blk["flights"]) == n_rows,
                      f"{panel}: the flight count is the census's ({blk['flights']})")
        else:
            info(f"{panel}: envelope counts not recorded or CSV absent")
    A = m.get("A", {}); sc = A.get("scene_completed") or {}
    rec_dir = _resolve(src.get("scene_records", ""), "results/"); scp = os.path.join(rec_dir, "campaign.csv") if rec_dir else ""
    if sc and scp and os.path.exists(scp):
        got = {}
        for r in flight_rows(scp):
            arm = os.path.basename(r.get("ctrl_trace", "") or ""); k, n = got.get(arm, (0, 0)); got[arm] = (k + (r.get("outcome") == "success"), n + 1)
        xt, rt = A.get("xpu_trace", "xpu_a_cpsat_hard.csv"), A.get("ros_trace", "ros_vanilla445.csv")   # older sidecars predate the fields
        ok = list(got.get(xt, (None, 0))) == sc.get("xpu") and list(got.get(rt, (None, 0))) == sc.get("ros")
        check(ok, f"panel A legend: the scene's completed/flown per arm equal its campaign table (XPU-RT {sc.get('xpu')}, ROS 2 {sc.get('ros')})")
        # on a scene neither arm completes, the gates each reaches is what the legend separates them by.
        # Each arm is checked on its own: the ladder renders name a ROS trace the scene campaign never
        # flew, so its mean is null, and pairing the two checks let a real XPU-RT number ride along
        # unverified on those figures.
        sg = A.get("scene_mean_gates") or {}
        if sg:
            tot = {}
            for r in flight_rows(scp):
                arm = os.path.basename(r.get("ctrl_trace", "") or "")
                g, n = tot.get(arm, (0.0, 0)); tot[arm] = (g + float(r.get("gates_passed", 0) or 0), n + 1)
            mean = {k: round(g / n, 2) for k, (g, n) in tot.items() if n}
            for label, key, trace in (("XPU-RT", "xpu", xt), ("ROS 2", "ros", rt)):
                if sg.get(key) is None:
                    info(f"panel A legend: {label} flew no row of this scene campaign, so it states no mean gates")
                    continue
                check(mean.get(trace) == sg[key],
                      f"panel A legend: {label}'s mean gates equal its campaign table ({sg[key]})")
    # The drawn pair must carry the same camera-to-control latency the scene census counts, or the
    # hero flight is a different experiment from the numbers printed beside it.
    for arm, key in (("xpu_dir", "xpu"), ("ros_dir", "ros")):
        d = src.get(arm)
        f = os.path.join(d, "figure_data.npz") if d else None
        if not (f and os.path.exists(f)):
            continue
        import numpy as _np
        z = _np.load(f, allow_pickle=True)
        tr = os.path.basename(str(z["ctrl_trace"])) if "ctrl_trace" in z.files else ""
        want = None
        _FC = _fc()
        if _FC is not None and tr:
            try:
                want = float(_FC.lat_ms(tr))
            except Exception:
                want = None
        if "percep_latency_ms" not in z.files:
            check(False, f"panel A {key}: the drawn flight records no camera-to-control latency "
                         f"(the dump predates the field; re-fly to state what it carried)")
        elif want is not None:
            got = float(z["percep_latency_ms"])
            check(abs(got - want) < 1.0,
                  f"panel A {key}: the drawn flight carries its arm's measured latency ({got:.1f} ms, {tr} is {want:.1f} ms)")

    # panel D: the bars re-derived from the energy CSV the render read
    D = m.get("D") or {}
    if D.get("energy_csv") and os.path.exists(_resolve(D["energy_csv"], "results/") or ""):
        import showdown_paper_figure as _PF
        got = _PF.energy_ratios(_resolve(D["energy_csv"], "results/"))
        check(got["conditions"] == D.get("conditions"),
              f"panel D: flights and moment/power ratios equal {os.path.basename(D['energy_csv'])} ("
              + ", ".join(f"{c} n={v['n']} {v['moment_x']}x/{v['power_x']}x" for c, v in got["conditions"].items()) + ")")
    elif "D" in m:
        check(False, "panel D: the energy CSV the render read is not on disk")
    # the XPU-RT arm is named by the solver that produced the schedule it flew: the label panels A and I print
    # has to agree with the manifest of the board run the replayed trace was cut from
    xl = (m.get("A") or {}).get("xpu_label")
    _FC = _fc()
    xt_ = (m.get("A") or {}).get("xpu_trace")
    if xl and _FC is not None and xt_ in getattr(_FC, "ARM_BY_TRACE", {}):
        dv = _FC.ARM_BY_TRACE[xt_].derive
        man = sorted(glob.glob(os.path.join(REPO, RES, "xpurt_long", f"manifest_{dv[1]}1_other_run1.json"))) if dv and dv[0] == "SOLVER_ARMS" else []
        if man:
            solver = str(json.load(open(man[0])).get("solver") or "")
            says = "CP-SAT" if "cpsat" in solver.lower() else ("greedy" if "greedy" in solver.lower() else solver)
            check(says.lower() in xl.lower(), f"panel A/I label '{xl}' names the solver of the board run behind {xt_} ({solver})")
    # the body rate behind "the baseline thrashes" and the speed behind "the same cruise"
    T = m.get("telemetry", {})
    if T:
        import numpy as _np
        for arm, key in (("xpu_dir", "xpu"), ("ros_dir", "ros")):
            d = src.get(arm)
            f = os.path.join(d, "figure_data.npz") if d else None
            if not (f and os.path.exists(f)):
                info(f"telemetry {key}: dump absent, not re-derived"); continue
            z = _np.load(f, allow_pickle=True)
            w = float(_np.linalg.norm(_np.asarray(z["imu_w"]), axis=1).mean())
            pz = _np.asarray(z["poses"])[:, :2]; ts = _np.asarray(z["t_s"])
            v = _np.linalg.norm(_np.diff(pz, axis=0), axis=1) / _np.maximum(_np.diff(ts), 1e-9)
            v = float(v[_np.isfinite(v)].mean())
            # the control rate the title draws for this arm
            hz_want = m.get("A", {}).get(f"{key}_eff_hz")
            if hz_want is not None and "eff_cmd_hz" in z.files:
                check(abs(float(z["eff_cmd_hz"]) - float(hz_want)) < 0.1,
                      f"panel A: {key} control rate {float(z['eff_cmd_hz']):.1f} Hz equals the flight it is drawn from ({hz_want})")
            for got, name in ((w, f"{key}_mean_w"), (v, f"{key}_mean_speed")):
                want = T.get(name)
                if want is None:
                    continue
                check(abs(got - float(want)) < 0.005, f"telemetry {name}: {got:.3f} equals the flight it is drawn from ({want})")

    # panels E-H: the drawn curves themselves, not only their means. A mean survives a curve coming
    # from a different flight; the per-series digest does not. Recomputed through the same
    # showdown_final_figure.telemetry_series the render draws from, so this tests the drawing rather
    # than restating it.
    want_ser = (m.get("telemetry") or {}).get("series")
    if want_ser:
        xd, rd = src.get("xpu_dir"), src.get("ros_dir")
        fx = os.path.join(xd, "figure_data.npz") if xd else None
        fr = os.path.join(rd, "figure_data.npz") if rd else None
        if fx and fr and os.path.exists(fx) and os.path.exists(fr):
            import numpy as _np
            import showdown_final_figure as _SFF
            got = _SFF.series_fingerprint(_SFF.telemetry_series(_np.load(fx, allow_pickle=True),
                                                            _np.load(fr, allow_pickle=True)))
            bad = [k for k in want_ser if got.get(k) != want_ser[k]]
            panels = sorted({k.split("_")[0] for k in want_ser})
            check(not bad, f"panels {'/'.join(panels)}: every drawn series is the one in the dumps "
                           f"({len(want_ser)} series" + (f"; differs: {', '.join(bad[:4])}" if bad else "") + ")")
        else:
            info("panels E-H: dumps absent, drawn series not re-derived")
    else:
        info("panels E-H: this render recorded no series fingerprint, only the four means")

    A = m.get("A", {})
    for mo in A.get("moments", []):
        d = src.get("ros_dir" if mo["arm"] == "ROS" else "xpu_dir")
        f = os.path.join(d, "figure_data.npz") if d else None
        if not (f and os.path.exists(f)):
            continue
        import numpy as _np
        ts = _np.load(f, allow_pickle=True)["t_s"]
        got = float(ts[min(int(mo["step"]), len(ts) - 1)])
        # tolerance smaller than one step: the flight is sampled every 0.01 s, so a 0.02 s window
        # accepted a callout drawn one step off its recorded timestamp, and a mutation audit found
        # this check unable to fail on exactly that
        check(abs(got - float(mo["t_s"])) < 0.005,
              f"moment {mo['label'][:38]}: t={got:.2f}s equals the flight it is drawn from ({mo['t_s']})")
    # panels a-d: each strip is an image, and only its timestamp was checked. The frame the strip
    # draws is picked out of the dump by the callout's step, so the step has to land on a captured
    # frame -- a strip from a neighbouring capture would keep the timestamp and change the picture.
    if A.get("moments"):
        import numpy as _np
        off = []
        for mo in A["moments"]:
            d = src.get("ros_dir" if mo["arm"] == "ROS" else "xpu_dir")
            f = os.path.join(d, "figure_data.npz") if d else None
            if not (f and os.path.exists(f)):
                continue
            z = _np.load(f, allow_pickle=True)
            if "frame_steps" not in z.files:
                continue
            fs = _np.asarray(z["frame_steps"]).ravel()
            if fs.size < 2:
                continue
            stride = float(_np.median(_np.diff(fs)))
            off.append((mo["label"][:24], float(_np.min(_np.abs(fs - int(mo["step"])))), stride))
        if off:
            # sims/scripts/*.frame_at picks argmin|frame_steps - step|, so the strip is the nearest
            # capture by construction. What is worth checking is that a capture exists near the
            # moment at all: one stride is the bound, because a callout at the very end of a crashed
            # flight has no frame after it and falls back on the last one, up to a stride earlier.
            bad = [f"{l} is {d:.0f} steps from any captured frame (stride {st:.0f})"
                   for l, d, st in off if d > st + 1e-9]
            check(not bad, f"panels a-d: every strip is a capture within one stride of its callout "
                           f"({len(off)} callouts, max {max(d for _, d, _ in off):.0f} steps"
                           + ("; " + "; ".join(bad[:3]) if bad else "") + ")")

    # panel A's backdrop is a composite, and the sidecar states the recipe. Re-run it.
    if A.get("backdrop") and src.get("xpu_dir"):
        import numpy as _np
        fx = os.path.join(src["xpu_dir"], "figure_data.npz")
        if os.path.exists(fx):
            V_ = _mods()[0]
            _img, note, std = V_.backdrop(_np.load(fx, allow_pickle=True))
            ok = note == A["backdrop"] and abs(float(std) - float(A.get("plate_std", -1))) < 0.05
            check(ok, f"panel A: the backdrop re-composes from the dump ({note}, plate std {std:.1f})")

    # The cruise the panel says it flew, and the two control rates panel B marks, are all recorded
    # in the dumps the render read. None was compared until a mutation audit looked.
    import numpy as _np
    for field, dumpkey, where in (("display_cruise", "cruise_speed", "xpu_dir"),):
        if A.get(field) is not None and src.get(where):
            f = os.path.join(_resolve(src[where], "results/"), "figure_data.npz")
            if os.path.exists(f):
                z = _np.load(f, allow_pickle=True)
                if dumpkey in z.files:
                    check(abs(float(z[dumpkey]) - float(A[field])) < 1e-6,
                          f"panel A: the cruise it states is the flight's ({A[field]} m/s)")
    B = m.get("B") or {}
    for field, where, who in (("xpu_hz", "xpu_dir", "XPU-RT"), ("ros_hz", "ros_dir", "ROS 2")):
        if B.get(field) is not None and src.get(where):
            f = os.path.join(_resolve(src[where], "results/"), "figure_data.npz")
            if os.path.exists(f):
                z = _np.load(f, allow_pickle=True)
                if "eff_cmd_hz" in z.files:
                    check(abs(float(z["eff_cmd_hz"]) - float(B[field])) < 0.05,
                          f"panel B: the {who} rate it marks is the displayed flight's "
                          f"({float(B[field]):.1f} Hz)")

    if "clearance_m" in A:
        # the panel draws a clearance; it is only a measurement when an obstacle was found
        check(bool(A.get("clearance_measured")),
              f"panel A: the drawn clearance {A['clearance_m']} m was measured against an obstacle")

    # The clearance and the four callout steps are what showdown_v3_figure.moments_for() returns,
    # so they are re-derived by calling it on the same two dumps rather than by restating the
    # geometry -- the same way backdrop() is used above, so each recorded value has a reader
    # that can falsify it. The clearance in particular is not a plain minimum distance (it is
    # measured to the nearest TALL obstacle from the path start), so a restated formula here
    # would check a different quantity from the one the renderer draws.
    xd_, rd_ = _resolve(src.get("xpu_dir", ""), "results/"), _resolve(src.get("ros_dir", ""), "results/")
    fx_, fr_ = (os.path.join(d or "", "figure_data.npz") for d in (xd_, rd_))
    if A.get("moments") and xd_ and rd_ and os.path.exists(fx_) and os.path.exists(fr_):
        V_ = _mods()[0]
        try:
            import showdown_gatecourse as S_          # the loader the renderer itself uses
            X_, R_ = S_.load(xd_), S_.load(rd_)
            hx_ = V_.eff_hz(X_) or 100.0
            hr_ = V_.eff_hz(R_) or 39.0
            r_out_ = str(R_["outcome"]) if "outcome" in R_.files else ""
            mm_, nm_, _hit_ = V_.moments_for(X_, R_, 85, hx_, hr_, outcome=r_out_)
        except Exception as e:
            info(f"panel A: moments_for could not be re-run ({type(e).__name__}: {e})")
            mm_ = nm_ = None
        if nm_ is not None and A.get("clearance_m") is not None:
            check(abs(float(nm_[2]) - float(A["clearance_m"])) < 0.0015,
                  f"panel A: the clearance re-derives from the flight ({A['clearance_m']} m to the "
                  f"nearest tall obstacle)")
        if mm_ is not None:
            # moments_for yields (arm, step, label) tuples; the sidecar records them as dicts
            want_steps = [int(x[1]) for x in mm_ if not isinstance(x, dict) and len(x) > 1]
            got_steps = [int(x["step"]) for x in A["moments"] if "step" in x]
            check(want_steps == got_steps,
                  f"panel A: every callout is drawn at the step moments_for picks "
                  f"({', '.join(str(x) for x in got_steps)})")

    # the camera rate the panel states is a property of the arm being replayed, and the registry
    # already carries it per cadence trace
    if A.get("camera_hz") is not None and A.get("ros_trace"):
        import figure_constants as FC_
        arm_ = FC_.ARM_BY_TRACE.get(A["ros_trace"])
        if arm_ is not None and getattr(arm_, "cam_hz", None):
            check(abs(float(arm_.cam_hz) - float(A["camera_hz"])) < 0.05,
                  f"panel A: the camera rate it states is the replayed arm's "
                  f"({A['camera_hz']} Hz, {A['ros_trace']})")
    I = m.get("I", {})
    for _gi, side in enumerate(src.get("gantt_sidecars", [])):
        sd = _resolve(side, "schedules/")
        if os.path.exists(sd):
            s_ = json.load(open(sd))
            # key by the arm the sidecar names, so a figure may carry more than one ROS row
            row = I.get(s_.get("arm") or ("xpu" if s_.get("kind") == "xpu" else "ros"), {})
            check(abs(float(s_["chain_ms_median"]) - float(row.get("chain_ms_median", -1))) < 0.05 and s_.get("frames_late") == row.get("frames_late"), f"panel I row {s_.get('arm')}: title numbers equal the Gantt sidecar ({s_['chain_ms_median']} ms, {s_.get('frames_late')}/{s_.get('frames_checked')} late)")
            # The row carries three more numbers the Gantt sidecar also records, and none of them was
            # compared: the control cadence the arm delivered, how many frames the lateness count is
            # out of, and the dispatch counts the bar chart draws. A mutation audit found them
            # recorded but unchecked, which is indistinguishable from verified until something moves.
            for key, tol, what in (("ctrl_gap_mean_ms", 0.05, "the control cadence drawn"),
                                   ("frames_checked", 0, "the frames the lateness is out of")):
                if key in s_ and key in row:
                    a_, b_ = float(s_[key]), float(row[key])
                    check(abs(a_ - b_) <= tol,
                          f"panel I row {s_.get('arm')}: {what} equals the Gantt sidecar "
                          f"({row[key]})")
            # bars are keyed by the label the panel prints ("ROS 2 chained"), the rows by the arm
            # ("p3c"), and nothing in the sidecar ties the two. Both lists are written by the same
            # render loop, in the same order, so position is the mapping -- matching on the label
            # text instead would pick the wrong bar as soon as a figure draws two ROS rows (a
            # three-row figure).
            bar_items = list((I.get("bars") or {}).items())
            bars = bar_items[_gi][1] if _gi < len(bar_items) else {}
            if bars.get("dispatches") is not None:
                # counted from the schedule the row was built from, not taken from the sidecar:
                # the bar chart's height is a claim about how much work the arm dispatched
                big_ = sd.replace("_metrics.json", ".json")
                nd = None
                if os.path.exists(big_):
                    dsp = json.load(open(big_)).get("dispatches")
                    nd = len(dsp) if isinstance(dsp, (dict, list)) else None
                if nd is not None:
                    check(int(bars["dispatches"]) == int(nd),
                          f"panel I row {s_.get('arm')}: the bar's dispatch count is the schedule's "
                          f"({bars['dispatches']})")
                # "drawn" is how many bars the row ends up with: the raw dispatches when the panel
                # draws them one by one, and the joined runs when it merges. An inequality against
                # the raw count passed for any value below it, so the count is re-derived by calling
                # the producer's own merge -- never a second implementation of the joining rule.
                if bars.get("drawn") is not None and os.path.exists(big_):
                    doc_ = json.load(open(big_))
                    if I.get("merged_per_hart_and_frame"):
                        import showdown_paper_figure as P_       # the render's own joining rule
                        doc_ = P_.merge_dispatches(doc_)
                    check(int(bars["drawn"]) == len(doc_["dispatches"]),
                          f"panel I row {s_.get('arm')}: the bars drawn are the "
                          + ("joined runs" if I.get("merged_per_hart_and_frame") else "dispatches")
                          + f" of the schedule ({bars['drawn']})")
                check(int(bars.get("drawn", -1)) <= int(bars["dispatches"]),
                      f"panel I row {s_.get('arm')}: the bar draws no more dispatches than exist "
                      f"({bars.get('drawn')}/{bars['dispatches']})")
            # the drawn hart spread is a claim about placement, so re-derive it from the schedule
            want = row.get("harts_per_net")
            if want:
                big = sd.replace("_metrics.json", ".json")
                got = {}
                if os.path.exists(big):
                    for v in json.load(open(big))["dispatches"].values():
                        jn = v["job_name"]
                        kk = "control" if jn.startswith("mlp") else ("nav" if jn.startswith("fused") else "YOLO")
                        for h in str(v.get("traced_target") or v["hardware_target"]).split("+"):
                            got.setdefault(kk, set()).add(h)
                got = {k2: sorted(v2) for k2, v2 in got.items()}
                # the bars span every lane of a sharded dispatch, and panel I's note states that span,
                # so the span is re-derived too rather than only the hart the trace named
                drawn = row.get("drawn_harts_per_net")
                if drawn and os.path.exists(big):
                    dg = {}
                    for v in json.load(open(big))["dispatches"].values():
                        jn = v["job_name"]
                        kk = "control" if jn.startswith("mlp") else ("nav" if jn.startswith("fused") else "YOLO")
                        for h in str(v["hardware_target"]).split("+"):
                            dg.setdefault(kk, set()).add(h)
                    dg = {k2: sorted(v2) for k2, v2 in dg.items()}
                    check(dg == {k2: list(v2) for k2, v2 in drawn.items()},
                          f"panel I row {s_.get('arm')}: the lanes each network is drawn across equal the "
                          f"schedule (" + ", ".join(f"{k2} on {len(v2)}" for k2, v2 in sorted(drawn.items())) + ")")
                check(got == {k2: list(v2) for k2, v2 in want.items()},
                      f"panel I row {s_.get('arm')}: each network's traced harts equal the schedule ("
                      + ", ".join(f"{k2} on {len(v2)}" for k2, v2 in sorted(want.items())) + ")")
    _check_provenance(m)
    _literals_check(["scripts/showdown_paper_figure.py"])
    return fails


def verify_atlas(m):
    """The atlas (showdown_atlas.py): census, forest, waterfall, pairs, solver points and added-load flights re-derived."""
    V, AT, SF = _mods()
    fig = m["figure"]; check(os.path.exists(fig), f"atlas on disk: {fig}")
    _check_fallbacks(m); _check_inputs(m)
    rows_all = AT.all_flights(); rows_c = [r for r in rows_all if r["camp"].startswith("campaign_")]
    A = m.get("A", {})
    check(A.get("flights") == len(rows_all), f"panel A: census counts every recorded flight ({len(rows_all)} on disk, {A.get('flights')} drawn)")
    _compare_forest(m.get("F", {}), rows_all, AT, "panel F")
    _compare_waterfall(m.get("E", {}), SF, "panel E")
    # G: draw_pairs — speeds 0.8–1.8, the 2.4 m scene, one flight per seed
    A_ = SF.cells_by(rows_c, "xpu_a_cpsat_hard.csv", 56.8, 0.0); B_ = SF.cells_by(rows_c, "ros_vanilla445.csv", 242.0, 0.0); tot = collections.Counter()
    for c in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        a = {r["seed"]: r for r in A_.get(c, [])}; b = {r["seed"]: r for r in B_.get(c, [])}
        for s in set(a) & set(b):
            ga = 4 if a[s]["outcome"] == "success" else a[s]["gates"]; gb = 4 if b[s]["outcome"] == "success" else b[s]["gates"]
            tot["xpu" if ga > gb else "ros" if gb > ga else "tie"] += 1
    check({k: v for k, v in m.get("G", {}).items() if v} == {k: v for k, v in dict(tot).items() if v}, f"panel G: paired seeds {dict(tot)} equal the render's {m.get('G')}")
    # I: solver points from the traces
    cp = V.xpu_rate_points("cpsat"); gr = V.xpu_rate_points("greedy"); ok = True; n = 0
    for key, v in m.get("I", {}).items():
        lab, hz = key.split("@"); pts = cp if lab == "CP-SAT" else gr; p = pts.get(int(hz))
        if not p:
            ok = False; continue
        n += 1; ok &= abs(100.0 * p["late"] / max(1, p["checked"]) - v["late_pct"]) < 0.1 and abs(p["lat"] - v["lat_ms"]) < 0.1
    check(ok and n > 0, f"panel I: frames-late and latency per solver and rate equal the traces ({n} points)")
    # J: added-load flights
    cells = V.flight_cells(rows_c, "a", 0.30, 0.0055); rich = {k: v for k, v in cells.items() if k[0].startswith(("xpu_b5", "ros_rvanilla"))}
    rich, _ = V.equalise(rich, [t for t in ("xpu_b5_cpsat.csv", "xpu_b5_greedy.csv", "ros_rvanilla445.csv", "ros_rvanilla490.csv") if any(k[0] == t for k in rich)])
    ok = True; n = 0
    for key, tr in (("XPURTCPSAT", "xpu_b5_cpsat.csv"), ("XPURTgreedy", "xpu_b5_greedy.csv")):
        v = m.get("J", {}).get(key, {})
        if "flightsK" in v:
            n += 1; k_ = sum(sum(x["outcome"] == "success" for x in rs) for (t, c), rs in rich.items() if t == tr); n_ = sum(len(rs) for (t, c), rs in rich.items() if t == tr)
            ok &= v["flightsK"] == k_ and v["flightsN"] == n_
    check(ok and n > 0, f"panel J: heavier-stack flights per solver equal the campaign CSV ({n} arms)")
    _literals_check(["scripts/showdown_atlas.py"])
    return fails


def verify_story(m, name):
    """One story figure's sidecar re-derived from the campaign CSVs, the ROS roll-up or the traces; the flight records are
    regenerable and their absence is reported, not failed."""
    V, AT, SF = _mods()
    _check_fallbacks(m); _check_inputs(m)
    body = {k: v for k, v in m.items() if k not in ("script", "inputs", "fallbacks_used", "figure", "written")}
    if name == "rate_speed_map":
        rows = V.load_campaigns(); ok = True; n = 0
        for key, kn in body.items():
            mm = re.match(r"(.+)@(\d+)Hz_c([\d.]+)$", key)
            if not mm:
                continue
            arm = next((a for a in SF.RATE_ARMS if a[0] == mm.group(1) and a[1] == int(mm.group(2))), None)
            if not arm:
                ok = False; print(f"     {key}: no RATE_ARMS entry"); continue
            v = SF.cells_by(rows, arm[2], arm[3], arm[4]).get(float(mm.group(3)), []); n += 1
            if [sum(x["outcome"] == "success" for x in v), len(v)] != list(kn):
                ok = False; print(f"     {key}: sidecar {kn}, re-derived {[sum(x['outcome'] == 'success' for x in v), len(v)]}")
        check(ok and n > 0, f"rate_speed_map: every cell equals the campaign CSVs, one flight per seed ({n} cells)")
    elif name == "seed_pairs":
        rows = V.load_campaigns(); A_ = SF.cells_by(rows, "xpu_a_cpsat_hard.csv", 56.8, 0.0); B_ = SF.cells_by(rows, "ros_vanilla445.csv", 242.0, 0.0); tot = collections.Counter()
        for c in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
            a = {r["seed"]: r for r in A_.get(c, [])}; b = {r["seed"]: r for r in B_.get(c, [])}
            for s in set(a) & set(b):
                ga = 4 if a[s]["outcome"] == "success" else a[s]["gates"]; gb = 4 if b[s]["outcome"] == "success" else b[s]["gates"]
                tot["xpu_further" if ga > gb else "ros_further" if gb > ga else "tie"] += 1
        got = {k: tot.get(k, 0) for k in ("xpu_further", "tie", "ros_further")}
        check(all(body.get(k, 0) == got[k] for k in got), f"seed_pairs: {got} equal the render's {dict((k, body.get(k)) for k in got)}")
    elif name == "latency_waterfall":
        _compare_waterfall(body, SF, "latency_waterfall")
    elif name == "ros_ladder":
        roll = _ros_rollup(); xp = V.xpu_rate_points("cpsat"); lay = {"vanilla serial YOLO": "vanilla", "vanilla 4-hart YOLO": "vanilla4", "+ control on a timer": "vanilla4tm", "+ QoS depth 1": "vanilla4_q1",
                                                                     "two YOLO nodes, 8 cores": "vanilla4x2", "hand-pinned 3 processes": "p3", "hand-pinned + QoS 1": "p3_q1", "XPU-RT CP-SAT": None}
        ok = True; n = 0
        for key, v in body.items():
            mm = re.match(r"(.+)@(\d+)$", key)
            if not mm or not isinstance(v, dict) or v.get("lat") is None:
                continue
            l = lay.get(mm.group(1)); hz = int(mm.group(2))
            got = xp.get(hz, {}).get("lat") if l is None else (statistics.median(roll[(l, hz)]) if (l, hz) in roll else None)
            if got is None:
                info(f"ros_ladder {key}: no board run to re-derive from"); continue
            n += 1; ok &= abs(got - v["lat"]) < 0.5
        check(ok and n > 0, f"ros_ladder: every layout's camera→control equals the ROS roll-up / XPU-RT traces ({n} bars)")
        fl = {k: v for k, v in body.items() if isinstance(v, dict) and "flights_k" in v}
        if fl:
            info(f"ros_ladder: {len(fl)} flight bars recorded (1.0–1.4 m/s pooled); recount needs the ladder's per-bar trace/latency table, not re-derived here")
    elif name in ("crash_position", "course_progress"):
        recs = SF.load_records()
        if not recs:
            info(f"{name}: flight records absent (regenerable), not re-derived"); return fails
        ok = True; n = 0
        for key, v in body.items():
            mm = re.match(r"(.+)_c([\d.]+)$", key)
            if not mm:
                continue
            got = recs.get((mm.group(1), float(mm.group(2))), [])
            if not got:
                info(f"{name} {key}: records absent, not re-derived"); continue
            n += 1; ok &= len(got) == v["n"] and ("completed" not in v or sum(r["outcome"] == "success" for r in got) == v["completed"])
        check(ok, f"{name}: recorded flights per arm and speed equal the records on disk ({n} cells re-derived)")
    _literals_check(["scripts/story_figures.py"])
    return fails


def dispatch(path):
    """run the verifier that matches a sidecar; returns the number of fails it added, or None when there is no verifier."""
    global fails
    m = json.load(open(path)); before = fails; base = os.path.basename(path).replace("_metrics.json", "")
    if not isinstance(m, dict):
        return None
    script = m.get("script") or ""
    kind = None
    if script.startswith("story_figures"):
        kind = ("story", script.split(":", 1)[1] if ":" in script else base)
    elif script:
        kind = ({"showdown_v3_figure": "v3", "showdown_atlas": "atlas", "showdown_final_figure": "final", "showdown_paper10_figure": "paper10", "showdown_paper_figure": "paper"}.get(script), None)
    if not kind or kind[0] is None:   # sidecars written before the `script` field
        if base.startswith("warehouse_showdown_v3"):
            kind = ("v3", None)
        elif "paper10" in base:
            kind = ("paper10", None)
        elif "atlas" in base:
            kind = ("atlas", None)
        elif base.startswith("warehouse_showdown_final"):
            kind = ("final", None)
        elif base in STORY_NAMES:
            kind = ("story", base)
        else:
            return None
    print(f"\n=== {base} ({kind[0]}{'' if not script else ', script ' + script})")
    {"v3": lambda: verify_v3(path), "final": lambda: verify_final(m), "paper10": lambda: verify_paper10(m), "paper": lambda: verify_paper(m), "atlas": lambda: verify_atlas(m), "story": lambda: verify_story(m, kind[1])}[kind[0]]()
    return fails - before


def verify_all():
    global fails
    print("=== constants"); r = subprocess.run([sys.executable, "scripts/measured_timing.py", "--verify"], capture_output=True, text=True)
    check(r.returncode == 0 and "DRIFT" not in r.stdout, "measured_timing --verify: every recorded constant re-derives from its artifact")
    FC = _fc()
    if FC is not None and hasattr(FC, "check_registry"):
        drift = FC.check_registry(); check(not drift, f"figure_constants registry re-derives from measured_timing ({len(drift)} drift)" + (": " + "; ".join(drift[:4]) if drift else ""))
    else:
        info("figure_constants registry not present; CSV latency keys not checked against measured_timing")
    print("\n=== campaign replicates"); _replicate_report()
    print("\n=== refined/"); _refined_allowlist()
    print("\n=== literals"); _literals_check()
    table = []
    for p in sorted(glob.glob(os.path.join(REPO, RES, "refined", "*_metrics.json"))):
        try:
            d = dispatch(p)
        except Exception as e:   # a verifier that cannot run is a failure of the sidecar contract, not a crash of the run
            check(False, f"{os.path.basename(p)}: verifier raised {type(e).__name__}: {e}"); d = 1
        if d is None:
            stem = os.path.basename(p).replace("_metrics.json", "")
            who = VERIFIED_ELSEWHERE.get(stem)
            info(f"{os.path.basename(p)}: " + (f"verified by scripts/{who}" if who
                 else "no verifier for this sidecar (not a showdown figure)")); continue
        table.append((os.path.basename(p).replace("_metrics.json", ""), d))
    print("\n=== summary")
    for name, d in table:
        print(f"  {name:<44} {'ok' if d == 0 else f'{d} FAIL'}")
    print(f"\n{fails} FAIL")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-arm", default="best45alt2"); ap.add_argument("--ros-tag", default="45_spin_r1")
    ap.add_argument("--campaign", default=f"{RES}/campaign/campaign.csv")
    ap.add_argument("--xpu-dir", default=os.environ.get("XPU_DIR", ""), help="figure_data dir of the displayed XPU-RT flight")
    ap.add_argument("--ros-dir", default=os.environ.get("ROS_DIR", ""), help="figure_data dir of the displayed ROS 2 flight")
    ap.add_argument("--v3-metrics", default="", help="verify the third-form composite from its sidecar (alias of --metrics)")
    ap.add_argument("--metrics", default="", help="verify any showdown figure from its _metrics.json sidecar")
    ap.add_argument("--all", action="store_true", help="every sidecar under results/codesign_feedback/refined/, plus the repo-wide checks")
    ap.add_argument("--coverage", action="store_true", help="after the checks, print which panel each one spoke about")
    a = ap.parse_args()
    if a.all:
        return verify_all()
    if a.v3_metrics or a.metrics:
        p = a.metrics or a.v3_metrics
        if dispatch(p) is None:
            stem = os.path.basename(p).replace("_metrics.json", "")
            who = VERIFIED_ELSEWHERE.get(stem)
            if who:
                info(f"{stem}: this figure is verified by scripts/{who}, not by this script")
            else:
                check(False, f"{p}: no verifier for this sidecar")
        if a.coverage:
            print_coverage()
        print(f"\n{fails} FAIL")
        return fails

    # 1. constants re-derive from the raw files
    r = subprocess.run([sys.executable, "scripts/measured_timing.py", "--verify"], capture_output=True, text=True)
    check(r.returncode == 0 and "DRIFT" not in r.stdout, "measured_timing --verify: every recorded constant re-derives from its artifact")

    # 2. the panel-I pair rebuilds identically from the named traces
    xt = f"{RES}/xpurt_long/trace_{a.xpu_arm}_other_run1.csv"; rt = f"{RES}/ros_traced/{a.ros_tag}/trace.csv"
    check(os.path.exists(xt), f"XPU-RT trace on disk: {xt}")
    check(os.path.exists(rt), f"ROS trace on disk: {rt}")
    if os.path.exists(xt) and os.path.exists(rt):
        sched = {"best45alt2": "schedules/best45_alt2.json", "best45alt4s": "schedules/best45_alt4s.json",
                 "best25p4": "schedules/best25_p4.json", "tiled25": "schedules/tiled_coupled_25hz.json"}.get(a.xpu_arm, "")
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "scripts/make_measured_gantt_pair.py", "--xpu-trace", xt,
                   "--xpu-cpu", xt.replace("trace_", "cpu_"), "--xpu-manifest", xt.replace("trace_", "manifest_").replace(".csv", ".json"),
                   "--ros-trace", rt, "--ros-cpu", f"{RES}/ros_traced/{a.ros_tag}/cpu.csv",
                   "--ros-manifest", f"{RES}/ros_traced/{a.ros_tag}/manifest.json", "--window-ms", "120", "--skip-ms", "400",
                   "--out-prefix", os.path.join(td, "g")] + (["--xpu-schedule", sched] if sched else [])
            subprocess.run(cmd, capture_output=True, text=True)
            for arm in ("xpu", "ros"):
                fresh = json.load(open(os.path.join(td, f"g_{arm}_metrics.json")))
                have = json.load(open(f"schedules/measured_gantt_{arm}_metrics.json"))
                same = all(abs(float(fresh.get(k) or 0) - float(have.get(k) or 0)) < 0.05
                           for k in ("chain_ms_median", "ctrl_gap_mean_ms", "frame_start_lag_median_ms"))
                check(same and fresh.get("source") == have.get("source"),
                      f"panel I {arm}: committed sidecar equals a fresh rebuild from {os.path.basename(have.get('source', '?'))} "
                      f"(chain {have.get('chain_ms_median')} ms, control every {have.get('ctrl_gap_mean_ms')} ms)")

    # 3. labels are derived, never typed
    src = open("sims/scripts/showdown_gatecourse.py").read()
    check("_hz(a.lat_xpu)" in src and "_hz(a.lat_ros)" in src, "panel A rates come from ceil(latency / control_dt)")
    _literals_check(["sims/scripts/showdown_gatecourse.py"])
    check("idle — core unused" not in src, "empty Gantt lanes are labelled with the measured busy %")
    check("static pin " not in src.replace("static pinning simply", ""), "the baseline is named ROS 2, not a stand-in")

    # 4. flight counts: replicates per campaign, what the figures' rules absorb
    _replicate_report()

    # 4b. the displayed flights are what the panel says they are: the XPU-RT flight completes the
    #     course, the baseline flight crashes after entering it (not before the first gate, not at
    #     the last), and both were flown at the same gain
    import numpy as np
    dumps = {}
    for arm, dd in (("xpu", a.xpu_dir), ("ros", a.ros_dir)):
        f = os.path.join(dd, "figure_data.npz") if dd else ""
        if f and os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            dumps[arm] = {"outcome": str(z["outcome"]), "gates": int(z["gates_passed"]),
                          "gain": float(z["moment_scale"]) if "moment_scale" in z else None,
                          "lat": float(z["sched_latency_ms"]) if "sched_latency_ms" in z else None}
    if dumps:
        x, r = dumps.get("xpu"), dumps.get("ros")
        if x:
            check(x["outcome"] == "success" and x["gates"] >= 4, f"displayed XPU-RT flight: {x['outcome']}, {x['gates']} gates")
        if r:
            check(r["outcome"] == "crash" and 1 <= r["gates"] <= 3,
                  f"displayed ROS 2 flight: {r['outcome']} after {r['gates']} gate(s) (must enter the course and lose it before the last gate)")
        if x and r and x["gain"] is not None and r["gain"] is not None:
            check(abs(x["gain"] - r["gain"]) < 1e-9, f"both displayed flights at the same gain ({x['gain']} vs {r['gain']})")
    else:
        info("displayed-flight dumps not given (--xpu-dir/--ros-dir); outcome check skipped")

    # 4c. the Gantt rows: every row's sidecar names its source trace and, for XPU-RT rows, the
    #     solver that produced the executed table (from the run manifest, never typed); the ROS row is
    #     a multi-process, unpinned deployment when the caption calls it vanilla
    for arm in ("xpu", "xpu2", "ros"):
        side = f"schedules/measured_gantt_{arm}_metrics.json"
        if not os.path.exists(side):
            if arm != "xpu2":
                check(False, f"panel I row {arm}: sidecar missing")
            continue
        m = json.load(open(side))
        check(bool(m.get("source")) and os.path.exists(m["source"]), f"panel I row {arm}: trace {m.get('source')} exists")
        if arm.startswith("xpu"):
            manp = m.get("manifest"); man = json.load(open(manp)) if manp and os.path.exists(manp) else {}
            sched = man.get("schedule", ""); meta = json.load(open(sched)).get("metadata", {}) if sched and os.path.exists(sched) else {}
            solver = meta.get("solver") or meta.get("scheduler") or ""
            check(bool(solver), f"panel I row {arm}: executed table {os.path.basename(sched)} names its solver ({solver[:60]})")
            check(m.get("frames_late") is not None, f"panel I row {arm}: lateness counted by window ({m.get('frames_late')}/{m.get('frames_checked')} late)")
        else:
            check("unpinned" in str(m.get("arm_label", "")) or "pinned" in str(m.get("arm_label", "")),
                  f"panel I ROS row: placement stated from its manifest ({m.get('arm_label')})")
        _check_gantt_sidecar(side)
    # 4d. replayed flights: the displayed dumps name the board trace whose cadence they flew
    for arm, dd in (("xpu", a.xpu_dir), ("ros", a.ros_dir)):
        f = os.path.join(dd, "figure_data.npz") if dd else ""
        if f and os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            ct = str(z["ctrl_trace"]) if "ctrl_trace" in z else ""
            if ct:
                check(os.path.exists(ct) or os.path.exists(os.path.join(RES, "ctrl_traces", os.path.basename(ct))),
                      f"displayed {arm} flight replayed {os.path.basename(ct)} ({float(z['eff_cmd_hz']):.1f} Hz effective)")
            else:
                info(f"displayed {arm} flight used a fixed hold ({float(z['sched_latency_ms']):.2f} ms), not a replayed trace")

    # 5. the ROS arm's manifest says what the caption claims
    man = f"{RES}/ros_traced/{a.ros_tag}/manifest.json"
    if os.path.exists(man):
        m = json.load(open(man))
        check(m.get("executor") == "single" and m.get("ctrl_mode", "timer") in ("timer", "chained"),
              f"ROS arm {a.ros_tag}: executor={m.get('executor')} ctrl_mode={m.get('ctrl_mode', 'timer')} pool={m.get('yolo_pool')} affinity={m.get('affinity_mask')} rmw={m.get('rmw')}")
    mt = glob.glob(f"{RES}/ros_traced/*_multi_r1/summary.json")
    check(bool(mt), f"multi-threaded executor result present to disclose ({len(mt)} runs)")
    p3 = glob.glob(f"{RES}/ros_traced/*_p3_r1/summary.json")
    check(bool(p3), f"control-on-its-own-process result present to disclose ({len(p3)} runs)")

    print(f"\n{fails} FAIL")
    return fails


if __name__ == "__main__":
    sys.exit(main())
