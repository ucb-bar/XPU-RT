#!/usr/bin/env python3
"""Drive the ModelBlaster half of the co-design loop, and gate it.

THE GAP THIS CLOSES. Every stage of the graph-rewrite arm exists as a CLI, and until now
nothing called any of them: `emit_compile_advice.py`, all five `advice_to_*_hint.py`
bridges and `ModelBlaster/scripts/verify_ir_rewrite_host.py` had ZERO programmatic
callers outside the test suite. The arm was a human typing eleven commands in the right
order, several of whose failure modes are silent -- and one of them, the bit-exact host
verify, is documented as mandatory and was invoked by nobody. A rewrite whose correctness
is unknown is not a measurement.

WHAT THIS DOES, per verb (fuse, unfuse, split, shard, choose_implementation):

    advice  ->  bridge  ->  applier  ->  graph gate  ->  host verify  ->  eligible?

and writes `arm_report.json` recording, for each verb, exactly how far it got and why it
stopped. Every stop is a fact worth keeping: "the advisor did not fire" and "the applier
refused" and "the rewrite is not bit-exact" are three different findings, and the old
"driver-mediated" arm conflated them into whatever the human remembered.

WHAT IT DOES NOT DO. It does not schedule anything, and it does not claim a board
measurement. A rewritten graph has no profile: the honest order is rewrite -> rebuild ->
reprofile on the board -> schedule on the measured finer profile, and steps 2-3 need the
cross toolchain and the physical K1. So this arm's output is a set of *eligible* rewrites
-- host-verified, graph-gate-passed -- which is the input the scheduling loop may then
consume. `--require-verify` (default) refuses to mark anything eligible that did not pass
the bit-exact gate, so an unverified rewrite cannot reach a figure by accident.

Usage:
  scripts/run_modelblaster_arm.py \\
      --schedule schedules/scheduled_<stem>_greedy_profiled.json \\
      --ir mlp_control=/path/to/ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json \\
      --ir yolov8_nano_64x96=/path/to/.../yolov8_nano_64x96/int8/graph.json \\
      --out-dir results/modelblaster_arm/<stem>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)
MB = os.environ.get("MB_ROOT") or os.path.join(REPO, "ModelBlaster")

# The appliers do not agree on what to call the model: apply_shard_hint.py takes
# --network, the other three take --model. Passing the wrong one is a usage error, which
# must not be reported as "applier refused" (that reads like a rejected rewrite).
APPLIER_MODEL_FLAG = {"shard": "--network"}

# verb -> (bridge script, applier relative to ModelBlaster, hint contract)
VERBS = {
    "fuse": ("scripts/advice_to_fusion_hint.py", "pipeline/apply_fusion_hint.py",
             "modelblaster.fusion_hints/v1"),
    "unfuse": ("scripts/advice_to_unfuse_hint.py", "pipeline/apply_unfuse_hint.py",
               "modelblaster.unfuse_hints/v1"),
    "split": ("scripts/advice_to_split_hint.py", "pipeline/apply_split_hint.py",
              "modelblaster.split_hints/v1"),
    "shard": ("scripts/advice_to_shard_hint.py", "pipeline/apply_shard_hint.py",
              "modelblaster.shard_hints/v1"),
    # choose_implementation has a bridge and NO applier: its consumer is
    # `generate_kernels --keep-reference-ops <op,...>`, which is per OP KIND rather
    # than per dispatch. Recorded as a structural limit, not as a failure.
    "choose_implementation": ("scripts/advice_to_kernel_choice.py", None,
                              "modelblaster.kernel_choice/v1"),
}


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True)


def tail(r, n=400):
    return ((r.stderr or "") + (r.stdout or "")).strip()[-n:]


def emit_advice(schedule, irs, out, gen_root, target, models, impls, log):
    """Stage 1: the advisor. Its output is the only input the bridges accept."""
    cmd = [PY, "scripts/emit_compile_advice.py", "--schedule", schedule,
           "--out", out, "--gen-root", gen_root, "--target", target]
    for m, p in irs.items():
        # emit_compile_advice wants <network>:<graph.json>; the bridges want a bare path.
        cmd += ["--ir", f"{m}:{p}"]
    if not models:
        # DERIVE IT. emit_compile_advice defaults to `mlp:mlp.q.int8,dronet:dronet.q.int8`,
        # which names two models this workload does not contain -- so the advisor looked
        # for profiles that do not exist and returned 0 records, and the whole arm read as
        # "the advisor has nothing to say" when in fact it was pointed at the wrong nets.
        # <network>:<profile basename>, taken from the IR's own name/quant.
        pairs = []
        for m, pth in irs.items():
            quant = "int8"
            try:
                quant = (json.load(open(pth)).get("quant") or "int8")
            except Exception:
                pass
            pairs.append(f"{m}:{m}.{quant}")
        models = ",".join(pairs)
        log(f"advice: --models derived from the IRs -> {models}")
    if models:
        cmd += ["--models", models]
    if impls:
        cmd += ["--impls", impls]
    r = run(cmd)
    if r.returncode != 0 or not os.path.exists(out):
        log(f"advice: FAILED — {tail(r)}")
        return None
    adv = json.load(open(out))
    recs = adv.get("advice") or adv.get("records") or []
    kinds = {}
    for rec in recs if isinstance(recs, list) else []:
        k = rec.get("recommendation") or rec.get("verb") or rec.get("kind") or "?"
        kinds[k] = kinds.get(k, 0) + 1
    log(f"advice: {len(recs) if isinstance(recs, list) else '?'} records "
        f"-> {kinds or '(none)'}")
    return {"path": out, "kinds": kinds, "n": len(recs) if isinstance(recs, list) else None}


# diff_dispatch_graph.py exit codes: 0 = GRANULARITY CHANGED, 3 = UNCHANGED, 4 = error.
# Zero means "changed" here, which is the opposite of the usual convention, and reading it
# the usual way inverts every verdict -- it reported split (a real 5->6 op rewrite) as a
# no-op and shard (annotation-only, correctly unchanged) as a rewrite.
GRAPH_CHANGED, GRAPH_UNCHANGED, GRAPH_ERROR = 0, 3, 4

# Which verbs the bit-exact host verify applies to. apply_shard_hint is annotation-only --
# same dispatch count, same ids, same edges, one extra `shard_factor` field -- so it cannot
# change numerics and there is nothing for a numeric gate to check. What a shard changes is
# the COST of the dispatch, and the gate for that is a board reprofile, not host verify.
VERIFY_APPLIES = {"fuse", "unfuse", "split"}


def ir_tensor_name(ir_path, dispatch_id):
    """The pre-rewrite tensor name for a dispatch, which is what --tensor wants."""
    try:
        for o in json.load(open(ir_path)).get("ops") or []:
            if str(o.get("dispatch_id", o.get("id"))) == str(dispatch_id):
                return o.get("name")
    except Exception:
        pass
    return None


def hint_dispatch_ids(hint_path):
    """The dispatch ids a hint targets, so the gate knows which tensor to compare."""
    ids = []
    try:
        h = json.load(open(hint_path))
    except Exception:
        return ids
    for net in h.get("networks") or []:
        for key in ("split_ops", "shard_ops", "fuse_ops", "unfuse_ops", "ops"):
            for e in net.get(key) or []:
                if isinstance(e, dict):
                    v = e.get("op", e.get("dispatch_id"))
                    if isinstance(v, int):
                        ids.append(v)
    return ids


def shard_annotated(ir_path):
    """`[(dispatch_id, shard_factor)]` recorded by apply_shard_hint."""
    out = []
    try:
        for o in json.load(open(ir_path)).get("ops") or []:
            if "shard_factor" in o:
                out.append((o.get("dispatch_id", o.get("id")), o["shard_factor"]))
    except Exception:
        pass
    return out


def hint_is_empty(path):
    """A bridge that writes a hint with no entries has not proposed anything."""
    try:
        h = json.load(open(path))
    except Exception:
        return True
    for key in ("hints", "entries", "dispatches", "pairs", "splits", "shards", "ops"):
        v = h.get(key)
        if isinstance(v, list):
            return len(v) == 0
        if isinstance(v, dict):
            return len(v) == 0
    # No recognised container: treat a bare contract-only document as empty.
    return len([k for k in h if k not in ("contract", "schema", "model", "version")]) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                    help="a solved schedule; the advisor reads it to find what binds")
    ap.add_argument("--ir", action="append", default=[], metavar="MODEL=PATH",
                    help="ModelBlaster graph.json per model (repeatable)")
    ap.add_argument("--out-dir", default="results/modelblaster_arm/run")
    ap.add_argument("--gen-root", default="gen/mb")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--models", default=None, help="passed through to emit_compile_advice")
    ap.add_argument("--impls", default=None, help="passed through to emit_compile_advice")
    ap.add_argument("--verbs", default=",".join(VERBS),
                    help="comma-separated subset of " + ",".join(VERBS))
    ap.add_argument("--require-verify", dest="require_verify", action="store_true",
                    default=True,
                    help="(default) a rewrite is eligible only if the bit-exact host "
                         "verify passed")
    ap.add_argument("--no-require-verify", dest="require_verify", action="store_false",
                    help="mark rewrites eligible without the bit-exact gate; the report "
                         "then says so on every row")
    ap.add_argument("--weights", default=None,
                    help="weights dir for verify_ir_rewrite_host (without it the gate "
                         "cannot run and nothing becomes eligible under the default)")
    ap.add_argument("--io", default=None, help="io dir for verify_ir_rewrite_host")
    a = ap.parse_args()

    irs = {}
    for spec in a.ir:
        if "=" not in spec:
            print(f"--ir expects MODEL=PATH, got {spec!r}", file=sys.stderr)
            return 2
        m, p = spec.split("=", 1)
        p_abs = p if os.path.isabs(p) else os.path.join(REPO, p)
        if not os.path.exists(p_abs):
            print(f"--ir {m}: no IR at {p_abs}", file=sys.stderr)
            return 2
        irs[m] = p_abs

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    log(f"== ModelBlaster arm ==  {len(irs)} IR(s): {', '.join(irs) or '(none)'}")
    if not os.path.isdir(MB) or not os.listdir(MB):
        log(f"ModelBlaster not populated at {MB} — run "
            f"`git submodule update --init ModelBlaster`. Nothing can be applied.")
        json.dump({"error": "ModelBlaster submodule not populated", "mb_root": MB},
                  open(os.path.join(out_dir, "arm_report.json"), "w"), indent=1)
        return 1

    advice = emit_advice(a.schedule, irs, os.path.join(out_dir, "compile_advice.json"),
                         a.gen_root, a.target, a.models, a.impls, log)

    rows = []
    for verb in [v.strip() for v in a.verbs.split(",") if v.strip()]:
        if verb not in VERBS:
            log(f"{verb}: unknown verb, skipped")
            continue
        bridge, applier, contract = VERBS[verb]
        for model, ir in irs.items():
            row = {"verb": verb, "model": model, "contract": contract,
                   "advice": bool(advice), "hint": None, "applied": None,
                   "graph_changed": None, "host_verified": None,
                   "eligible": False, "stopped_at": None, "why": None}
            if not advice:
                row["stopped_at"] = "advice"
                row["why"] = "the advisor did not produce a usable advice file"
                rows.append(row)
                continue
            hint = os.path.join(out_dir, f"{model}.{verb}_hint.json")
            r = run([PY, bridge, "--advice", advice["path"], "--ir", ir,
                     "--model", model, "--out", hint])
            if r.returncode != 0 or not os.path.exists(hint):
                row["stopped_at"] = "bridge"
                row["why"] = f"bridge declined or failed: {tail(r, 240)}"
                rows.append(row)
                log(f"{verb}/{model}: bridge -> no hint ({row['why'][:100]})")
                continue
            if hint_is_empty(hint):
                row["stopped_at"] = "bridge"
                row["why"] = ("the advisor fired nothing this verb could act on "
                              "(an empty hint is a finding, not a failure)")
                rows.append(row)
                log(f"{verb}/{model}: hint is empty — advice did not fire")
                continue
            row["hint"] = os.path.relpath(hint, REPO)
            if applier is None:
                row["stopped_at"] = "applier"
                row["why"] = ("this verb has no applier by construction: its consumer is "
                              "generate_kernels --keep-reference-ops, which is per OP "
                              "KIND, not per dispatch")
                rows.append(row)
                log(f"{verb}/{model}: hint written, no applier by construction")
                continue

            rewritten = os.path.join(out_dir, f"{model}.{verb}.graph.json")
            model_flag = APPLIER_MODEL_FLAG.get(verb, "--model")
            r = run([PY, os.path.join(MB, applier), "--hint", hint, model_flag, model,
                     "--ir", ir, "--out", rewritten])
            if r.returncode != 0 or not os.path.exists(rewritten):
                row["applied"] = False
                row["stopped_at"] = "applier"
                row["why"] = f"applier refused or failed: {tail(r, 240)}"
                rows.append(row)
                log(f"{verb}/{model}: applier refused ({row['why'][:100]})")
                continue
            row["applied"] = True

            # Graph gate: did the rewrite actually change the dispatch graph?
            g = run([PY, "scripts/diff_dispatch_graph.py", "--before", ir,
                     "--after", rewritten,
                     "--json", os.path.join(out_dir, f"{model}.{verb}.diff.json")])
            if g.returncode == GRAPH_ERROR:
                row["stopped_at"] = "graph gate"
                row["why"] = f"graph gate could not compare the two IRs: {tail(g, 240)}"
                rows.append(row)
                log(f"{verb}/{model}: graph gate ERROR")
                continue
            row["graph_changed"] = (g.returncode == GRAPH_CHANGED)

            if verb == "shard":
                # Annotation-only by construction: an unchanged granularity is the
                # CORRECT outcome, and the thing to check is that the annotation landed.
                ann = shard_annotated(rewritten)
                row["annotation"] = [{"dispatch_id": d, "shard_factor": f}
                                     for d, f in ann]
                row["host_verified"] = None
                if not ann:
                    row["stopped_at"] = "applier"
                    row["why"] = ("apply_shard_hint wrote no shard_factor annotation; "
                                  "nothing was actually recorded")
                else:
                    row["eligible"] = True
                    row["why"] = (f"annotation-only rewrite recorded "
                                  f"{row['annotation']}; numerics cannot change, so the "
                                  f"remaining gate is a BOARD REPROFILE at that width, "
                                  f"not a numeric check")
                    row["gate"] = "reprofile-required"
                rows.append(row)
                log(f"{verb}/{model}: annotation={row['annotation']} "
                    f"-> eligible={row['eligible']} (gate: reprofile)")
                continue

            if not row["graph_changed"]:
                row["stopped_at"] = "graph gate"
                row["why"] = ("the rewrite left the dispatch graph unchanged -- a no-op, "
                              "and the gate's own words: do not profile this as a rewrite")
                rows.append(row)
                log(f"{verb}/{model}: applied but graph UNCHANGED — no-op")
                continue

            # The gate that was never called. Its inputs sit next to the IR in a
            # ModelBlaster build, so discover them rather than making the caller pass them.
            ir_dir = os.path.dirname(ir)
            weights = a.weights or os.path.join(ir_dir, "weights.npz")
            io = a.io or os.path.join(ir_dir, "io.npz")
            ids = hint_dispatch_ids(hint)
            tensor = ir_tensor_name(ir, ids[0]) if ids else None
            missing = [n for n, p in (("weights", weights), ("io", io))
                       if not os.path.exists(p)]
            if missing or not tensor:
                row["host_verified"] = None
                row["stopped_at"] = "host verify"
                row["why"] = ("bit-exact host verify could not run ("
                              + (f"no {', '.join(missing)} beside the IR"
                                 if missing else
                                 "could not determine the pre-rewrite tensor name")
                              + "). Under --require-verify this rewrite is NOT eligible: "
                                "a rewrite of unknown correctness is not a result")
                row["eligible"] = not a.require_verify
                rows.append(row)
                log(f"{verb}/{model}: graph changed; host verify SKIPPED "
                    f"-> eligible={row['eligible']}")
                continue
            vjson = os.path.join(out_dir, f"{model}.{verb}.verify.json")
            v = run([PY, os.path.join(MB, "scripts/verify_ir_rewrite_host.py"),
                     "--baseline-ir", ir, "--rewritten-ir", rewritten,
                     "--weights", weights, "--io", io, "--tensor", tensor,
                     "--work", os.path.join(out_dir, f"host_verify_{model}_{verb}"),
                     "--json", vjson])
            row["verify_tensor"] = tensor
            row["host_verified"] = (v.returncode == 0)
            row["eligible"] = bool(row["host_verified"]) or not a.require_verify
            row["stopped_at"] = None if row["host_verified"] else "host verify"
            row["why"] = ("bit-exact" if row["host_verified"]
                          else f"host verify failed: {tail(v, 300)}")
            rows.append(row)
            log(f"{verb}/{model}: graph changed, tensor={tensor}, "
                f"host_verified={row['host_verified']} -> eligible={row['eligible']}")

    report = {
        "schema": "modelblaster_arm/v1",
        "schedule": a.schedule,
        "mb_root": MB,
        "require_verify": a.require_verify,
        "advice": advice,
        "rows": rows,
        "eligible_rewrites": [r for r in rows if r["eligible"]],
        "note": ("A rewritten graph has no profile. Eligible means host-verified and "
                 "graph-changed, NOT measured: rebuild + board reprofile come next, and "
                 "scheduling a rewrite on costs derived from its parent measures the "
                 "derivation rather than the rewrite."),
    }
    json.dump(report, open(os.path.join(out_dir, "arm_report.json"), "w"), indent=1)
    open(os.path.join(out_dir, "arm_log.txt"), "w").write("\n".join(log_lines) + "\n")
    n_el = len(report["eligible_rewrites"])
    log(f"\n{len(rows)} verb×model attempt(s); {n_el} eligible rewrite(s). "
        f"report: {os.path.relpath(os.path.join(out_dir, 'arm_report.json'), REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
