#!/usr/bin/env python3
"""Close every actionable gap across the canonical 6-model suite.

For each unique ``(gap_kind, fx_target)`` pair in the suite's
``materialization_plan.json`` that is **not** already in the registry:

1. Pick a representative gap from the highest-ranked entry.
2. Materialize the workspace.
3. Write a real ``extension.py`` from a hand-curated dispatch table that
   maps each fx_target to the stdlib torch implementation.
4. Verify + register.

This is the agent-side action that ``plan-extensions`` was preparing.

Run:

    .venv/bin/python scripts/dev/close_suite_gaps.py

After this script, re-run gap-discovery on the suite with
``--extension-registry`` and the queue should shrink dramatically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "bin" / "python"


def _gd(model_run_dir: Path) -> Path:
    """Return the gap_discovery dir, accepting either the new
    ``03_gap_discovery`` layout or the legacy ``02_gap_discovery``."""
    new = model_run_dir / "03_gap_discovery"
    return new if new.exists() else model_run_dir / "02_gap_discovery"


SUITE_PLAN = REPO / "results" / "graph_compilation" / "extension_planning" / "materialization_plan.json"
SUITE_RUN_DIR = REPO / "results" / "graph_compilation" / "severity_audit_suite"
# Round-2: re-plan against the after-registry runs to catch gaps that
# were rebucketed (e.g. tanh going from noncritical → critical_path
# once the bigger gaps were closed).
ROUND2_PLAN = REPO / "results" / "graph_compilation" / "extension_planning_round2" / "materialization_plan.json"
ROUND2_RUN_DIR = REPO / "results" / "graph_compilation" / "severity_audit_suite_closed"
EXT_ROOT = REPO / ".crg-artifacts" / "extensions"
REGISTRY = REPO / "user_extensions" / "registry.yaml"

# --------------------------------------------------------------------------- #
# Hand-curated extension implementations.
#
# Each entry maps an fx_target (canonicalised — same string the gap record
# carries) to the body of ``def extension(*args)``. We deliberately keep
# these to thin wrappers around stdlib torch — the point of the proof is
# *closure*, not novel kernels. Real kernels would slot into this table.
# --------------------------------------------------------------------------- #


_HEADER = "from __future__ import annotations\n\nimport torch\nimport torch.nn.functional as F\n\n\n"


_EXTENSIONS: dict[str, str] = {
    # ---- Dynamo built-ins ------------------------------------------------- #
    "<built-in function linear>": (
        "def extension(input, weight, bias=None):\n"
        "    return F.linear(input, weight, bias)\n"
    ),
    "<built-in function gelu>": (
        "def extension(input):\n"
        "    return F.gelu(input)\n"
    ),
    "<built-in function matmul>": (
        "def extension(a, b):\n"
        "    return torch.matmul(a, b)\n"
    ),
    "<built-in method conv2d of type object>": (
        "def extension(input, weight, bias=None, stride=1, padding=0,\n"
        "              dilation=1, groups=1):\n"
        "    return torch.conv2d(input, weight, bias, stride, padding,\n"
        "                        dilation, groups)\n"
    ),
    "<function embedding>": (
        "def extension(input, weight, padding_idx=None,\n"
        "              max_norm=None, norm_type=2.0,\n"
        "              scale_grad_by_freq=False, sparse=False):\n"
        "    return F.embedding(input, weight, padding_idx, max_norm,\n"
        "                       norm_type, scale_grad_by_freq, sparse)\n"
    ),
    "<function batch_norm>": (
        "def extension(input, running_mean, running_var, weight=None,\n"
        "              bias=None, training=False, momentum=0.1, eps=1e-5):\n"
        "    return F.batch_norm(input, running_mean, running_var,\n"
        "                        weight, bias, training, momentum, eps)\n"
    ),
    "<function relu>": (
        "def extension(input, inplace=False):\n"
        "    return F.relu(input, inplace=inplace)\n"
    ),
    "<built-in method tanh of type object>": (
        "def extension(input):\n"
        "    return torch.tanh(input)\n"
    ),
    # ---- aten.* targets --------------------------------------------------- #
    "aten.relu.default": (
        "def extension(input):\n"
        "    return torch.ops.aten.relu(input)\n"
    ),
    "aten.tanh.default": (
        "def extension(input):\n"
        "    return torch.ops.aten.tanh(input)\n"
    ),
    "aten._native_batch_norm_legit_no_training.default": (
        # Gap record drops the (momentum, eps) scalar args — Dynamo only
        # captured tensor shapes. Defaults match torch.nn.BatchNorm2d.
        "def extension(input, weight, bias, running_mean, running_var,\n"
        "              momentum=0.1, eps=1e-5):\n"
        "    return torch.ops.aten._native_batch_norm_legit_no_training(\n"
        "        input, weight, bias, running_mean, running_var, momentum, eps\n"
        "    )\n"
    ),
    "aten.native_batch_norm.default": (
        "def extension(input, weight, bias, running_mean, running_var,\n"
        "              training=False, momentum=0.1, eps=1e-5):\n"
        "    return torch.ops.aten.native_batch_norm(\n"
        "        input, weight, bias, running_mean, running_var,\n"
        "        training, momentum, eps\n"
        "    )\n"
    ),
    "aten.empty.memory_format": (
        # Allocator op — verifier passes a shape list as the only arg.
        "def extension(*args, **kwargs):\n"
        "    return torch.ops.aten.empty.memory_format(*args, **kwargs)\n"
    ),
    "aten.select.int": (
        "def extension(input, dim, index):\n"
        "    return torch.ops.aten.select.int(input, dim, index)\n"
    ),
}


def _registered_pairs() -> set[tuple[str, str]]:
    if not REGISTRY.exists():
        return set()
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return {(e["gap_kind"], e["fx_target"]) for e in raw.get("entries", [])
            if e.get("verification_status") == "pass"}


def _materialize(model_run_dir: Path, gap_id: str) -> Path:
    """Materialize ``gap_id`` from this model's queue and return workspace path."""
    queue = _gd(model_run_dir) / "gap_action_queue.json"
    out = subprocess.run(
        [str(PY), "-m", "compgen.graph_compilation", "materialize-extension",
         "--queue", str(queue), "--gap-id", gap_id,
         "--extensions-root", str(EXT_ROOT)],
        cwd=str(REPO), check=True, capture_output=True, text=True,
    )
    # Output: "materialized: <abs path>"
    line = out.stdout.strip().splitlines()[-1]
    return Path(line.split(":", 1)[1].strip())


def _write_extension(workspace: Path, body: str) -> None:
    (workspace / "extension.py").write_text(_HEADER + body, encoding="utf-8")


def _verify(workspace: Path) -> bool:
    out = subprocess.run(
        [str(PY), "-m", "compgen.graph_compilation", "verify-extension",
         "--extension", str(workspace)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"  verify FAIL ({workspace.name}): rc={out.returncode}")
        print(f"    stdout: {out.stdout[-400:]}")
        print(f"    stderr: {out.stderr[-400:]}")
        return False
    return True


def _register(workspace: Path) -> bool:
    out = subprocess.run(
        [str(PY), "-m", "compgen.graph_compilation", "register-extension",
         "--extension", str(workspace), "--registry", str(REGISTRY)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"  register FAIL: rc={out.returncode}")
        print(f"    stderr: {out.stderr[-400:]}")
        return False
    return True


def _gap_has_clean_shapes(model_run_dir: Path, gap_id: str) -> bool:
    """Reject gaps whose Dynamo shape_signature lost an input shape.

    A shape signature like ``inputs: [[], [32,32], [32]]`` means torch
    Dynamo couldn't infer a shape for the first arg — the verifier
    would build a 0-D random tensor and most ops would reject it. We
    pick a different model's gap for the same fx_target instead.
    """
    queue_path = _gd(model_run_dir) / "gap_action_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    g = next((g for g in queue["gaps"] if g["gap_id"] == gap_id), None)
    if g is None:
        return False
    for shape in g["shape_signature"].get("inputs", []):
        if not shape:
            return False
        for d in shape:
            if not isinstance(d, int) or d <= 0:
                return False
    return True


def main() -> int:
    global SUITE_RUN_DIR  # noqa: PLW0603 — single-shot CLI driver
    # CLI arg ``--round=1|2`` selects which planning result to read from.
    round_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--round=")), "1")
    if round_arg == "2":
        plan_path = ROUND2_PLAN
        SUITE_RUN_DIR = ROUND2_RUN_DIR
    else:
        plan_path = SUITE_PLAN
    print(f"plan = {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registered = _registered_pairs()
    print(f"already registered: {len(registered)} extensions")

    # Pick the highest-ranked gap for each fx_target whose shape signature
    # is clean. If every variant of that target has malformed shapes we
    # report it as failed-to-pick.
    by_target: dict[str, list[dict]] = {}
    for r in plan["backlog"]:
        by_target.setdefault(r["fx_target"], []).append(r)

    todo: list[dict] = []
    no_clean: list[str] = []
    for target, candidates in by_target.items():
        key = ("unsupported_op", target)
        if key in registered:
            continue
        chosen = None
        for r in candidates:  # already sorted by rank desc by plan-extensions
            model_run = SUITE_RUN_DIR / r["model_run_dir"]
            if _gap_has_clean_shapes(model_run, r["gap_id"]):
                chosen = r
                break
        if chosen is None:
            no_clean.append(target)
            continue
        todo.append(chosen)

    if not todo:
        print("nothing to close.")
        return 0
    print(f"to close: {len(todo)} unique fx_targets\n")

    succeeded = 0
    failed: list[str] = []
    for r in todo:
        target = r["fx_target"]
        body = _EXTENSIONS.get(target)
        if body is None:
            print(f"  SKIP {target!r}: no implementation in dispatch table")
            failed.append(target)
            continue

        model_run = SUITE_RUN_DIR / r["model_run_dir"]
        print(f"closing {target!r} via {r['model']}/{r['gap_id']} …")
        ws = _materialize(model_run, r["gap_id"])
        _write_extension(ws, body)
        if not _verify(ws):
            failed.append(target)
            continue
        if not _register(ws):
            failed.append(target)
            continue
        succeeded += 1
        print(f"  ok → {ws.name}")

    print("\n=== closure summary ===")
    print(f"  succeeded:   {succeeded}")
    print(f"  failed:      {len(failed)}")
    for f in failed:
        print(f"    - {f}")
    print(f"  no_clean_shapes ({len(no_clean)}): malformed shape signatures across all variants")
    for n in no_clean:
        print(f"    - {n}")
    return 0 if (not failed and not no_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())
