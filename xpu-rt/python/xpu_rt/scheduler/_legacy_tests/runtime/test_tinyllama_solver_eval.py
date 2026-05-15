"""Regression test: TinyLlama solver evaluation runs end-to-end.

This test exercises the full Phase E stack against the real
TinyLlama-1.1B-Chat-v1.0 architecture (topology read from the
HuggingFace ``config.json`` in the local cache). Skips honestly
when the config is not present.

It runs the script with 2 layers to keep CI time bounded; full
22-layer runs are validated separately by the operator (the MOSEK
MILP grows quadratically in the alias-pair count and can take
minutes to hours on the full model).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _has_tinyllama_config() -> bool:
    hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    base = Path(hf_cache) / "hub"
    if not base.is_dir():
        return False
    for cfg in base.rglob("config.json"):
        if "TinyLlama-1.1B-Chat-v1.0" in str(cfg):
            return True
    return False


@pytest.mark.skipif(
    not _has_tinyllama_config(),
    reason="TinyLlama config absent from HF cache; eval is operator-driven",
)
def test_eval_runs_end_to_end(tmp_path: Path):
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/dev/eval_tinyllama_solvers.py",
            "--out", str(tmp_path),
            "--kv-len", "32",
            "--num-layers", "2",
            "--num-devices", "2",
            "--z3-proof-required",
        ],
        check=True,
    )
    assert rc.returncode == 0

    report = json.loads((tmp_path / "tinyllama_solver_eval_report.json").read_text())
    # The three solver stages must have run.
    for stage in ("placement", "overlap", "memory"):
        info = report["solvers"][stage]
        assert info["status"] in {"optimal", "feasible"}, (
            f"{stage} did not solve: {info}"
        )
        assert info["time_ms"] > 0
        assert isinstance(info["formulation_hash"], str) and len(info["formulation_hash"]) == 16

    # Z3 obligations: every distinct K in the topology must prove K%16==0.
    z3 = report.get("z3")
    assert z3 is not None and z3["obligations"], "Z3 obligations not produced"
    for o in z3["obligations"]:
        assert o["status"] == "proved", f"unexpected Z3 status: {o}"
        assert o["selected_backend"] == "z3"

    # gates pass on the produced run-dir.
    from compgen.audit.solver_gates import all_solver_gates

    gates = all_solver_gates(run_dir=tmp_path)
    failed = [g for g in gates if g.status == "fail"]
    assert not failed, f"M-69 gates failed on TinyLlama run-dir: {failed}"


@pytest.mark.skipif(
    not _has_tinyllama_config(),
    reason="TinyLlama config absent from HF cache",
)
def test_eval_honest_when_z3_off(tmp_path: Path):
    """When ``--z3-proof-required`` is omitted, the eval still runs
    the three other solvers, but the Z3 block in the report is None
    — never faked."""

    subprocess.run(
        [
            sys.executable,
            "scripts/dev/eval_tinyllama_solvers.py",
            "--out", str(tmp_path),
            "--kv-len", "32",
            "--num-layers", "2",
            "--num-devices", "1",
        ],
        check=True,
    )
    report = json.loads((tmp_path / "tinyllama_solver_eval_report.json").read_text())
    assert report["z3"] is None
    for stage in ("placement", "overlap", "memory"):
        assert report["solvers"][stage]["status"] in {"optimal", "feasible"}
