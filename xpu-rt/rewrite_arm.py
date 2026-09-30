"""The ModelBlaster half of the loop, as a library rather than three CLIs.

WHY THIS MODULE EXISTS. `run_modelblaster_arm.py` proposes and gates graph rewrites;
`run_board_round.py` rebuilds, reprofiles and adjudicates one; `run_codesign_loop.py`
searches scheduling levers. All three were separate commands, so a graph rewrite could
never *compete* with a scheduling lever inside one round -- a human had to run the arm,
pick a rewrite, run a board round, read the verdict, and decide. That is the last hand
step in a loop that is otherwise automatic, and it is the reason `fuse` was "reported,
not applied": nothing could cost it.

Everything here is backend-parameterised through `Backend`, because the same chain
should work for any target ModelBlaster can generate for -- the K1 pieces are the
`board` runner, not the arm.

The honest boundary this module keeps: a rewritten graph has NO profile, and costing it
by dividing its parent's cost is measuring the derivation rather than the rewrite. So a
rewrite becomes a schedulable candidate only after a runner has measured it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# diff_dispatch_graph.py exit codes: 0 = GRANULARITY CHANGED, 3 = UNCHANGED, 4 = error.
# Zero means "changed", the opposite of the usual convention; reading it the usual way
# inverts every verdict.
GRAPH_CHANGED, GRAPH_UNCHANGED, GRAPH_ERROR = 0, 3, 4

# The bit-exact host verify applies to rewrites that change what is computed. It does
# not apply to `shard`: apply_shard_hint is annotation-only -- same dispatch count, same
# ids, one extra `shard_factor` field -- so numerics cannot change and there is nothing
# for a numeric gate to check. Its gate is a reprofile at that width.
VERIFY_APPLIES = {"fuse", "unfuse", "split"}

# The appliers disagree on what to call the model: apply_shard_hint takes --network.
APPLIER_MODEL_FLAG = {"shard": "--network"}

#: verb -> (bridge, applier relative to ModelBlaster, hint contract)
VERBS: Dict[str, tuple] = {
    "fuse": ("scripts/advice_to_fusion_hint.py", "pipeline/apply_fusion_hint.py",
             "modelblaster.fusion_hints/v1"),
    "unfuse": ("scripts/advice_to_unfuse_hint.py", "pipeline/apply_unfuse_hint.py",
               "modelblaster.unfuse_hints/v1"),
    "split": ("scripts/advice_to_split_hint.py", "pipeline/apply_split_hint.py",
              "modelblaster.split_hints/v1"),
    "shard": ("scripts/advice_to_shard_hint.py", "pipeline/apply_shard_hint.py",
              "modelblaster.shard_hints/v1"),
    # choose_implementation has a bridge and no applier by construction: its consumer is
    # `generate_kernels --keep-reference-ops`, which is per OP KIND, not per dispatch.
    "choose_implementation": ("scripts/advice_to_kernel_choice.py", None,
                              "modelblaster.kernel_choice/v1"),
}

REWRITE_VERBS = ("fuse", "unfuse", "split", "shard")


@dataclass
class Backend:
    """Where a backend's graphs, profiles and build live.

    The loop hardcoded `spacemit_x60` / `rvv_x60` / `ime_x60` / `gen/mb`, which is one
    of the ten backends ModelBlaster knows. Everything that varies per backend is here
    so the same chain runs for another one by passing a different Backend.
    """
    target: str = "spacemit_x60"
    hw: str = "rvv_x60"
    accel_hw: Optional[str] = "ime_x60"      # the per-dispatch alternative, if any
    gen_root: str = "gen/mb"                  # where dispatch graphs live
    profile_root: str = "gen/profile_mb"      # where measured profiles live
    mb_root: str = field(default_factory=lambda: os.environ.get(
        "MB_ROOT") or os.path.join(REPO, "ModelBlaster"))

    def profile_dir(self, model: str, quant: str = "int8",
                    hw: Optional[str] = None, topo: str = "topo_0") -> str:
        hw = hw or self.hw
        return os.path.join(REPO, self.profile_root, hw, self.target, model,
                            f"{model}.{quant}",
                            f"{model}_{self.target}_{hw}_{model}.{quant}", topo)

    def dispatch_graph(self, model: str, quant: str = "int8") -> str:
        return (f"{self.gen_root}/vmfb/{model}/{self.target}/{self.hw}/"
                f"{model}.{quant}/{model}.{quant}_dispatch_graph.json")


def _py() -> str:
    venv = os.path.join(REPO, ".venv/bin/python")
    return os.environ.get("XPURT_PY") or (venv if os.path.exists(venv) else sys.executable)


def sh(cmd, env=None, cwd=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, cwd=cwd or REPO, env=e, capture_output=True, text=True)


def _tail(r, n=300):
    return ((r.stderr or "") + (r.stdout or "")).strip()[-n:]


def arm_model(net: str, tag: str) -> str:
    """A per-arm model identity, `<net>_<tag>`.

    The tag goes in the NAME, not the quant: generate_kernels accepts only
    fp32/fp16/int8 and refuses an IR whose quant disagrees with its flag, while the name
    separates build dir, dispatch graph and profile basename -- all three derive from it.
    """
    return f"{net}_{tag}"


def stage_ir(ir_path: str, net: str, tag: str, work_dir: str, log) -> str:
    """A copy of the IR renamed to this arm's model, so no derived path is shared."""
    g = json.load(open(ir_path))
    m = arm_model(net, tag)
    g["name"] = m
    g["quant"] = "int8"
    out = os.path.join(work_dir, f"{m}.int8.graph.json")
    os.makedirs(work_dir, exist_ok=True)
    json.dump(g, open(out, "w"), indent=1)
    log(f"  staged IR as name={m} ({len(g.get('ops') or [])} ops)")
    return out


def emit_graph(ir: str, backend: Backend, log) -> bool:
    r = sh([_py(), os.path.join(backend.mb_root, "pipeline/emit_dispatch_graph.py"),
            "--ir", ir, "--out-root", f"{backend.gen_root}/vmfb",
            "--target", backend.target, "--hw", backend.hw])
    if r.returncode != 0:
        log(f"  emit_dispatch_graph failed: {_tail(r)}")
        return False
    last = (r.stdout or "").strip().splitlines()
    log("  " + (last[-1] if last else "emitted"))
    return True


# --------------------------------------------------------------- advice + proposal

def _hint_is_empty(path: str) -> bool:
    try:
        h = json.load(open(path))
    except Exception:
        return True
    for key in ("hints", "entries", "dispatches", "pairs", "splits", "shards", "ops",
                "networks"):
        v = h.get(key)
        if isinstance(v, list):
            if not v:
                return True
            if key == "networks":
                return not any(
                    any(n.get(k) for k in ("split_ops", "shard_ops", "fuse_ops",
                                           "unfuse_ops", "ops"))
                    for n in v if isinstance(n, dict))
            return False
        if isinstance(v, dict):
            return len(v) == 0
    return len([k for k in h if k not in ("contract", "schema", "model", "version",
                                          "reason", "_provenance")]) == 0


def _ir_tensor_name(ir_path: str, dispatch_id) -> Optional[str]:
    try:
        for o in json.load(open(ir_path)).get("ops") or []:
            if str(o.get("dispatch_id", o.get("id"))) == str(dispatch_id):
                return o.get("name")
    except Exception:
        pass
    return None


def _hint_dispatch_ids(hint_path: str) -> List[int]:
    ids: List[int] = []
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


def _shard_annotated(ir_path: str):
    out = []
    try:
        for o in json.load(open(ir_path)).get("ops") or []:
            if "shard_factor" in o:
                out.append((o.get("dispatch_id", o.get("id")), o["shard_factor"]))
    except Exception:
        pass
    return out


def emit_advice(schedule: str, irs: Dict[str, str], out: str, backend: Backend,
                log, models: Optional[str] = None,
                impls: Optional[str] = None) -> Optional[dict]:
    """Stage 1: the advisor. Its output is the only input the bridges accept."""
    if not models:
        # emit_compile_advice defaults to `mlp:mlp.q.int8,dronet:dronet.q.int8`, two
        # models most workloads do not contain -- it then finds no profiles and returns
        # zero records, which reads as "the advisor has nothing to say".
        pairs = []
        for m, pth in irs.items():
            quant = "int8"
            try:
                quant = json.load(open(pth)).get("quant") or "int8"
            except Exception:
                pass
            pairs.append(f"{m}:{m}.{quant}")
        models = ",".join(pairs)
    cmd = [_py(), "scripts/emit_compile_advice.py", "--schedule", schedule,
           "--out", out, "--gen-root", backend.gen_root, "--target", backend.target,
           "--models", models]
    for m, p in irs.items():
        cmd += ["--ir", f"{m}:{p}"]
    if impls:
        cmd += ["--impls", impls]
    r = sh(cmd)
    if r.returncode != 0 or not os.path.exists(out):
        log(f"  advice FAILED — {_tail(r)}")
        return None
    adv = json.load(open(out))
    recs = adv.get("advice") or adv.get("records") or []
    kinds: Dict[str, int] = {}
    for rec in recs if isinstance(recs, list) else []:
        k = rec.get("recommendation") or rec.get("verb") or "?"
        kinds[k] = kinds.get(k, 0) + 1
    log(f"  advice: {len(recs)} records -> {kinds or '(none)'}")
    return {"path": out, "kinds": kinds, "n": len(recs)}


def propose(schedule: str, irs: Dict[str, str], backend: Backend, out_dir: str, log,
            verbs=REWRITE_VERBS, require_verify: bool = True,
            advice_path: Optional[str] = None) -> List[dict]:
    """advice -> bridge -> applier -> graph gate -> bit-exact verify, per (verb, model).

    Returns one row per attempt, with `eligible` set only when the rewrite passed the
    gates that apply to it. Rows that stopped early carry `stopped_at` and `why`,
    because "the advisor did not fire", "the applier refused" and "the rewrite is not
    bit-exact" are three different findings.
    """
    os.makedirs(out_dir, exist_ok=True)
    adv = None
    if advice_path and os.path.exists(advice_path):
        adv = {"path": advice_path}
    else:
        adv = emit_advice(schedule, irs, os.path.join(out_dir, "compile_advice.json"),
                          backend, log)
    rows: List[dict] = []
    if not adv:
        return rows
    for verb in verbs:
        bridge, applier, contract = VERBS[verb]
        for model, ir in irs.items():
            row = {"verb": verb, "model": model, "contract": contract, "ir": ir,
                   "hint": None, "rewritten_ir": None, "graph_changed": None,
                   "host_verified": None, "eligible": False, "stopped_at": None,
                   "why": None}
            hint = os.path.join(out_dir, f"{model}.{verb}_hint.json")
            r = sh([_py(), bridge, "--advice", adv["path"], "--ir", ir,
                    "--model", model, "--out", hint])
            if r.returncode != 0 or not os.path.exists(hint):
                row.update(stopped_at="bridge",
                           why=f"bridge declined: {_tail(r, 200)}")
                rows.append(row)
                continue
            if _hint_is_empty(hint):
                row.update(stopped_at="bridge",
                           why="the advisor fired nothing this verb could act on")
                rows.append(row)
                continue
            row["hint"] = hint
            if applier is None:
                row.update(stopped_at="applier",
                           why="no applier by construction (per op kind, not dispatch)")
                rows.append(row)
                continue
            rewritten = os.path.join(out_dir, f"{model}.{verb}.graph.json")
            flag = APPLIER_MODEL_FLAG.get(verb, "--model")
            r = sh([_py(), os.path.join(backend.mb_root, applier), "--hint", hint,
                    flag, model, "--ir", ir, "--out", rewritten])
            if r.returncode != 0 or not os.path.exists(rewritten):
                row.update(stopped_at="applier",
                           why=f"applier refused: {_tail(r, 220)}")
                rows.append(row)
                continue
            row["rewritten_ir"] = rewritten
            g = sh([_py(), "scripts/diff_dispatch_graph.py", "--before", ir,
                    "--after", rewritten,
                    "--json", os.path.join(out_dir, f"{model}.{verb}.diff.json")])
            if g.returncode == GRAPH_ERROR:
                row.update(stopped_at="graph gate",
                           why=f"gate could not compare: {_tail(g, 200)}")
                rows.append(row)
                continue
            row["graph_changed"] = (g.returncode == GRAPH_CHANGED)

            if verb == "shard":
                ann = _shard_annotated(rewritten)
                row["annotation"] = [{"dispatch_id": d, "shard_factor": f}
                                     for d, f in ann]
                if not ann:
                    row.update(stopped_at="applier",
                               why="no shard_factor annotation was written")
                else:
                    row.update(eligible=True, gate="reprofile-required",
                               why=f"annotation-only rewrite {row['annotation']}; "
                                   f"numerics cannot change, so the gate is a reprofile")
                rows.append(row)
                continue

            if not row["graph_changed"]:
                row.update(stopped_at="graph gate",
                           why="the rewrite left the dispatch graph unchanged")
                rows.append(row)
                continue

            ir_dir = os.path.dirname(ir)
            weights, io = (os.path.join(ir_dir, "weights.npz"),
                           os.path.join(ir_dir, "io.npz"))
            ids = _hint_dispatch_ids(hint)
            tensor = _ir_tensor_name(ir, ids[0]) if ids else None
            missing = [n for n, p in (("weights", weights), ("io", io))
                       if not os.path.exists(p)]
            if missing or not tensor:
                row.update(host_verified=None, stopped_at="host verify",
                           eligible=(not require_verify),
                           why=("bit-exact verify could not run ("
                                + (f"no {', '.join(missing)} beside the IR" if missing
                                   else "no pre-rewrite tensor name")
                                + "); a rewrite of unknown correctness is not a result"))
                rows.append(row)
                continue
            v = sh([_py(), os.path.join(backend.mb_root,
                                        "scripts/verify_ir_rewrite_host.py"),
                    "--baseline-ir", ir, "--rewritten-ir", rewritten,
                    "--weights", weights, "--io", io, "--tensor", tensor,
                    "--work", os.path.join(out_dir, f"hv_{model}_{verb}"),
                    "--json", os.path.join(out_dir, f"{model}.{verb}.verify.json")])
            row["host_verified"] = (v.returncode == 0)
            row["verify_tensor"] = tensor
            row["eligible"] = bool(row["host_verified"]) or not require_verify
            row["stopped_at"] = None if row["host_verified"] else "host verify"
            row["why"] = ("bit-exact" if row["host_verified"]
                          else f"host verify failed: {_tail(v, 260)}")
            rows.append(row)
    return rows


# ------------------------------------------------------------------- measurement

def measure_on_board(ir: str, net: str, tag: str, seed_dir: str, backend: Backend,
                     log, cores: Optional[str] = None) -> Optional[dict]:
    """Rebuild for the target and profile on the physical board.

    This is the K1 runner. Another backend supplies its own; the rest of the chain does
    not change. `MB_IR` is what makes a REWRITE profilable at all -- no torch model
    corresponds to it -- and it needs weights/goldens beside it, which `seed_dir` gives.
    """
    m = arm_model(net, tag)
    work = os.path.join("/tmp", f"mbk1_{m}")
    shutil.rmtree(work, ignore_errors=True)
    dst = os.path.join(work, m, "int8")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(seed_dir):
        s = os.path.join(seed_dir, f)
        if os.path.isfile(s):
            shutil.copy2(s, dst)
    staged = stage_ir(ir, net, tag, dst, log)
    out_root = os.path.join(REPO, "results", "board_runs", m)
    shutil.rmtree(out_root, ignore_errors=True)
    env = {"MB_IR": staged, "PROFILE_OUT_ROOT": out_root, "OUT_ROOT": work,
           "PATH": os.path.dirname(_py()) + os.pathsep + os.environ.get("PATH", "")}
    if cores:
        env["MB_CORES"] = cores
    log(f"  board: run_model_k1.sh {m} int8 {backend.hw} 0")
    r = sh([os.path.join(backend.mb_root, "scripts/run_model_k1.sh"), m, "int8",
            backend.hw, "0"], env=env)
    if r.returncode != 0:
        log(f"  BOARD RUN FAILED: {(r.stderr or r.stdout)[-600:]}")
        return None
    if "max_abs_err=0" not in r.stdout:
        log("  board verify did not report max_abs_err=0 — refusing these costs")
        return None
    n_ref = r.stdout.count("falling back to reference_impl")
    if n_ref:
        log(f"  WARNING: {n_ref} curated kernel(s) fell back to the reference; these "
            f"costs measure the reference, not the curated kernel")
    src = os.path.join(out_root, backend.hw, backend.target, m, f"{m}.int8",
                       f"{m}_{backend.target}_{backend.hw}_{m}.int8", "topo_0",
                       "results.csv")
    if not os.path.exists(src):
        log(f"  board run produced no results.csv at {src}")
        return None
    dest = backend.profile_dir(m)
    os.makedirs(dest, exist_ok=True)
    shutil.copy2(src, os.path.join(dest, "results.csv"))
    n = len(list(open(src))) - 1
    log(f"  measured {n} dispatch(es) -> {os.path.relpath(dest, REPO)}/results.csv")
    return {"tag": tag, "model": m, "staged_ir": staged, "n_dispatches": n,
            "profile": os.path.relpath(dest, REPO), "reference_fallbacks": n_ref,
            "runner": "board"}


RUNNERS: Dict[str, Callable] = {"board": measure_on_board}


def spec_with(workload: str, net: str, model: str, backend: Backend,
              out_path: str, quant: str = "int8") -> str:
    """The workload with one net repointed at a rewritten graph.

    The network KEY stays `net`, so its period, window and criticality are untouched;
    only which graph -- and therefore which measured profile -- it points at changes.
    """
    spec = json.loads(json.dumps(json.load(open(workload))))
    spec["networks"][net]["dispatch_deps_path"] = backend.dispatch_graph(model, quant)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(spec, open(out_path, "w"), indent=1)
    return out_path
