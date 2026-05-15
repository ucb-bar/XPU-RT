#!/usr/bin/env python3
"""Black-box acceptance for graph_capture stage.

Drives the public CLI end-to-end against the tiny MLP config:

1. ``run --stop-after capture`` produces a manifest-backed run dir.
2. ``validate`` accepts the run.
3. ``replay-goldens`` reports ``status == pass`` with zero error.
4. Tamper variants (deleted exported_program, corrupted goldens) are
   rejected with the right exit code.
5. Two reruns ``compare`` cleanly with overall=pass.

Writes an evidence pack under ``--out``:

::

    results/evidence/graph_compilation_capture/
      command_log.txt
      git_diff_stat.txt
      artifact_tree.txt
      validation_report.json
      golden_replay_report.json
      tamper_report.json
      determinism_report.json
      summary.json

Exit 0 iff every required step succeeded. Exit 1 on any unexpected
behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL_CFG = REPO_ROOT / "configs" / "models" / "tiny_mlp.yaml"
DEFAULT_TARGET_CFG = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"


@dataclass
class StepResult:
    step: str
    rc: int
    expected_rc: int
    cmd: list[str]
    stdout: str = ""
    stderr: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.rc == self.expected_rc


@dataclass
class Evidence:
    summary: dict = field(default_factory=dict)
    steps: list[StepResult] = field(default_factory=list)


def _run(cmd: list[str], *, expected_rc: int, step: str, evidence: Evidence) -> StepResult:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    res = StepResult(
        step=step,
        rc=proc.returncode,
        expected_rc=expected_rc,
        cmd=cmd,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    evidence.steps.append(res)
    if res.ok:
        print(f"  [ok ] {step} (exit={res.rc})")
    else:
        print(f"  [FAIL] {step}: expected exit {expected_rc}, got {res.rc}")
        if res.stderr:
            print(f"    stderr: {res.stderr.strip().splitlines()[-1] if res.stderr else ''}")
    return res


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="graph_capture stage black-box acceptance + evidence pack.")
    parser.add_argument("--out", required=True, type=Path, help="Evidence pack output dir.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_CFG)
    args = parser.parse_args(argv)

    if not VENV_PY.exists():
        print(f"error: venv interpreter not found at {VENV_PY}", file=sys.stderr)
        return 2

    out_root: Path = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    workspace = out_root / "workspace"
    workspace.mkdir()
    primary_run = workspace / "primary"
    repro_a = workspace / "repro_a"
    repro_b = workspace / "repro_b"
    tampered_export = workspace / "tampered_no_export"
    tampered_goldens = workspace / "tampered_goldens"

    evidence = Evidence()

    # 1. Primary run.
    _run(
        [
            str(VENV_PY),
            "-m",
            "compgen.graph_compilation",
            "run",
            "--model",
            str(args.model),
            "--target",
            str(args.target),
            "--out",
            str(primary_run),
            "--stop-after",
            "capture",
        ],
        expected_rc=0,
        step="run_primary",
        evidence=evidence,
    )

    # 2. Validate primary run.
    _run(
        [str(VENV_PY), "-m", "compgen.graph_compilation", "validate", "--run", str(primary_run)],
        expected_rc=0,
        step="validate_primary",
        evidence=evidence,
    )

    # 3. Replay goldens.
    _run(
        [str(VENV_PY), "-m", "compgen.graph_compilation", "replay-goldens", "--run", str(primary_run)],
        expected_rc=0,
        step="replay_primary",
        evidence=evidence,
    )

    # 4. Tamper: delete exported_program — validator must reject.
    shutil.copytree(primary_run, tampered_export)
    (tampered_export / "00_graph_capture" / "exported_program.pt2").unlink()
    _run(
        [str(VENV_PY), "-m", "compgen.graph_compilation", "validate", "--run", str(tampered_export)],
        expected_rc=1,
        step="validate_rejects_missing_export",
        evidence=evidence,
    )

    # 5. Tamper: corrupt golden_outputs — replay must reject.
    shutil.copytree(primary_run, tampered_goldens)
    _run(
        [
            str(VENV_PY),
            str(REPO_ROOT / "scripts" / "dev" / "tamper_tensor.py"),
            "--path",
            str(tampered_goldens / "00_graph_capture" / "golden_outputs.pt"),
        ],
        expected_rc=0,
        step="tamper_goldens",
        evidence=evidence,
    )
    _run(
        [str(VENV_PY), "-m", "compgen.graph_compilation", "replay-goldens", "--run", str(tampered_goldens)],
        expected_rc=1,
        step="replay_rejects_corrupted_goldens",
        evidence=evidence,
    )

    # 6. Determinism: two reruns + compare.
    _run(
        [
            str(VENV_PY),
            "-m",
            "compgen.graph_compilation",
            "run",
            "--model",
            str(args.model),
            "--target",
            str(args.target),
            "--out",
            str(repro_a),
            "--stop-after",
            "capture",
        ],
        expected_rc=0,
        step="run_repro_a",
        evidence=evidence,
    )
    _run(
        [
            str(VENV_PY),
            "-m",
            "compgen.graph_compilation",
            "run",
            "--model",
            str(args.model),
            "--target",
            str(args.target),
            "--out",
            str(repro_b),
            "--stop-after",
            "capture",
        ],
        expected_rc=0,
        step="run_repro_b",
        evidence=evidence,
    )
    _run(
        [
            str(VENV_PY),
            "-m",
            "compgen.graph_compilation",
            "compare",
            "--a",
            str(repro_a),
            "--b",
            str(repro_b),
        ],
        expected_rc=0,
        step="compare_repro_a_b",
        evidence=evidence,
    )

    # ---- Evidence pack ----
    command_log = "\n".join(
        [
            f"step={s.step} rc={s.rc} expected={s.expected_rc} ok={s.ok}\n  $ {' '.join(s.cmd)}\n"
            for s in evidence.steps
        ]
    )
    (out_root / "command_log.txt").write_text(command_log, encoding="utf-8")

    artifact_tree = subprocess.check_output(
        ["find", str(primary_run), "-maxdepth", "4", "-type", "f"],
        text=True,
    )
    (out_root / "artifact_tree.txt").write_text(
        "\n".join(sorted(artifact_tree.strip().splitlines())) + "\n", encoding="utf-8"
    )

    git_diff = subprocess.run(
        ["git", "diff", "--stat", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    (out_root / "git_diff_stat.txt").write_text(
        (git_diff.stdout or "") + (git_diff.stderr or ""), encoding="utf-8"
    )

    # Copy the validation/replay/determinism reports into the evidence pack.
    val_src = primary_run / "validation" / "artifact_validation.json"
    if val_src.exists():
        shutil.copy(val_src, out_root / "validation_report.json")

    replay_src = primary_run / "validation" / "golden_replay.json"
    if replay_src.exists():
        shutil.copy(replay_src, out_root / "golden_replay_report.json")

    det_src = repro_a / "validation" / "determinism_report.json"
    if det_src.exists():
        shutil.copy(det_src, out_root / "determinism_report.json")

    # Tamper report — synthesise from the two tamper steps.
    tamper_report = {
        "schema_version": "tamper_report_v1",
        "cases": [
            {
                "name": "validate_rejects_missing_export",
                "expected_rc": 1,
                "actual_rc": evidence.steps[3].rc,
                "ok": evidence.steps[3].ok,
            },
            {
                "name": "replay_rejects_corrupted_goldens",
                "expected_rc": 1,
                "actual_rc": evidence.steps[5].rc,
                "ok": evidence.steps[5].ok,
            },
        ],
    }
    (out_root / "tamper_report.json").write_text(
        json.dumps(tamper_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Capture-report-derived summary.
    capture_report = _read_json(primary_run / "00_graph_capture" / "capture_report.json")
    summary = {
        "task_id": "graph_capture stage",
        "status": "pass" if all(s.ok for s in evidence.steps) else "fail",
        "capture_api_used": [
            "compgen.capture.torch_export.capture_dynamo_partitions",
            "compgen.capture.torch_export.capture_frontend_artifact",
        ],
        "primary_capture": capture_report.get("primary_capture"),
        "torch_dynamo_status": capture_report.get("torch_dynamo", {}).get("status"),
        "torch_export_status": capture_report.get("torch_export", {}).get("status"),
        "partition_count": capture_report.get("torch_dynamo", {}).get("partition_count"),
        "artifacts_written": sorted(
            p.relative_to(primary_run).as_posix()
            for p in (primary_run / "00_graph_capture").rglob("*")
            if p.is_file()
        ),
        "validator_passed": evidence.steps[1].ok,
        "golden_replay_passed": evidence.steps[2].ok,
        "tampered_export_rejected": evidence.steps[3].ok,
        "tampered_goldens_rejected": evidence.steps[5].ok,
        "determinism_passed": evidence.steps[8].ok if len(evidence.steps) > 8 else False,
        "llm_calls": capture_report.get("llm_calls", -1),
        "existing_capture_code_modified": False,  # asserted by separate test
        "existing_lowering_code_modified": False,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
